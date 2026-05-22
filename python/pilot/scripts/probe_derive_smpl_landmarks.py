"""
probe_derive_smpl_landmarks.py — load the SMPL_NEUTRAL template
mesh in T-pose and pick anatomical-landmark vertex indices by
geometric position. Writes a JSON table to
docs/PR-7a4_PROBE/smpl_landmark_indices.json.

T-pose SMPL coordinate convention (verified via render_smpl.py
reference + LBS template):
  +x = subject right side (image left when facing camera)
  +y = up (head direction)
  +z = forward (out of chest)
Origin near pelvis center.

This script is the SELF-CONTAINED alternative to fetching external
SMPL vertex-segmentation references (web blocked). It picks indices
by maximum/minimum of body-part regions in T-pose. Visual verification
happens later in probe_smpl_vertex_landmarks.py — if any index lands
on the wrong body region, that's caught at render time.

Run:
    .venv-benchmark/Scripts/python.exe \\
        python/pilot/scripts/probe_derive_smpl_landmarks.py
"""
from __future__ import annotations

import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np


# chumpy 0.70 is incompatible with Python 3.11 + NumPy 1.26.
# SMPL_NEUTRAL.pkl was pickled with chumpy.Ch objects wrapping numpy
# arrays. We only need v_template (raw numpy array of shape (6890, 3)),
# so we stub chumpy.Ch with a plain Python class that captures the
# underlying array via __setstate__ and exposes it via __array__.
class _ChumpyShim:
    """Pickle-protocol-compliant chumpy.Ch replacement."""
    def __init__(self, *args, **kwargs):
        self._x = None
    def __setstate__(self, state):
        # chumpy.Ch.__reduce__ returns (cls, (), state_dict). State dict
        # may contain 'x' (the underlying ndarray) plus chain-rule
        # bookkeeping we ignore.
        if isinstance(state, dict):
            x = state.get("x", None)
            self._x = np.asarray(x) if x is not None else None
        else:
            self._x = np.asarray(state) if state is not None else None
    def __array__(self, dtype=None):
        if self._x is None:
            raise ValueError("ChumpyShim has no value")
        return self._x.astype(dtype) if dtype is not None else self._x
    def __repr__(self):
        return f"ChumpyShim({self._x.shape if self._x is not None else None})"

_stub_module = types.ModuleType("chumpy")
_stub_module.Ch = _ChumpyShim
_stub_module.ch = _ChumpyShim
sys.modules["chumpy"] = _stub_module
sys.modules["chumpy.ch"] = _stub_module

REPO_ROOT = Path(__file__).resolve().parents[3]
SMPL_PKL = (
    REPO_ROOT / "local_models" / "smpl" / "_extracted"
    / "SMPL_python_v.1.1.0" / "smpl" / "models"
    / "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
)
OUT_JSON = REPO_ROOT / "docs" / "PR-7a4_PROBE" / "smpl_landmark_indices.json"


def main() -> None:
    if not SMPL_PKL.exists():
        raise SystemExit(f"missing: {SMPL_PKL}")
    with SMPL_PKL.open("rb") as f:
        data = pickle.load(f, encoding="latin1")

    # Template vertices: 'v_template' in the SMPL pkl is T-pose mesh,
    # betas = 0. Shape: (6890, 3). v_template may be a ChumpyShim
    # (per the stub above) or a plain ndarray depending on pkl version.
    raw_v = data["v_template"]
    V = np.asarray(raw_v, dtype=np.float32)
    assert V.shape == (6890, 3), f"unexpected v_template shape {V.shape}"
    print(f"[probe] T-pose template: shape={V.shape}")
    print(f"[probe] x range: [{V[:, 0].min():+.3f}, {V[:, 0].max():+.3f}] (subject L<->R)")
    print(f"[probe] y range: [{V[:, 1].min():+.3f}, {V[:, 1].max():+.3f}] (down<->up)")
    print(f"[probe] z range: [{V[:, 2].min():+.3f}, {V[:, 2].max():+.3f}] (back<->front)")

    # Helper: pick vertex by argmax/argmin of an axis within a region
    # (boolean mask on V).
    def pick(mask: np.ndarray, axis: int, kind: str) -> int:
        idxs = np.nonzero(mask)[0]
        if idxs.size == 0:
            raise SystemExit(f"empty mask for axis={axis} kind={kind}")
        if kind == "max":
            return int(idxs[np.argmax(V[idxs, axis])])
        return int(idxs[np.argmin(V[idxs, axis])])

    landmarks: dict[str, dict] = {}

    # ── Head crown: highest y across whole mesh.
    idx = int(np.argmax(V[:, 1]))
    landmarks["head_crown"] = {
        "index": idx, "xyz": V[idx].tolist(),
        "rule": "argmax y over full mesh",
    }

    # ── Throat midpoint: highest y on the midline neck slice. Pick
    # vertices with |x| < 0.02 (midline), y in [0.30, 0.55], z > 0
    # (anterior surface, throat — not C7 at back).
    neck_throat = (
        (np.abs(V[:, 0]) < 0.02)
        & (V[:, 1] > 0.30) & (V[:, 1] < 0.55)
        & (V[:, 2] > 0.02)
    )
    idx = pick(neck_throat, axis=1, kind="max")   # highest throat point
    landmarks["throat_midpoint"] = {
        "index": idx, "xyz": V[idx].tolist(),
        "rule": "midline, y in [0.30, 0.55], z>0.02 anterior; argmax y",
    }

    # ── 7th cervical (C7): highest y on midline POSTERIOR neck slice
    # (z < 0).
    c7_region = (
        (np.abs(V[:, 0]) < 0.02)
        & (V[:, 1] > 0.20) & (V[:, 1] < 0.55)
        & (V[:, 2] < -0.02)
    )
    idx = pick(c7_region, axis=1, kind="max")
    landmarks["c7"] = {
        "index": idx, "xyz": V[idx].tolist(),
        "rule": "midline, y in [0.20, 0.55], z<-0.02 posterior; argmax y",
    }

    # ── Acromion (top-of-shoulder bony peak — skin surface).
    # T-pose y-slice geometry: below y=0.25 the body width balloons
    # (arms extend horizontally → |x| up to 0.87 at wrist). The acromion
    # is the TOP of the shoulder pad, sitting just BEFORE the arm
    # extension — narrow torso-edge band at y in [0.23, 0.32],
    # |x| in [0.13, 0.22], near depth midline.
    for side, x_sign in (("left", -1), ("right", +1)):
        side_mask = (
            (V[:, 0] * x_sign > 0.13) & (V[:, 0] * x_sign < 0.22)
            & (V[:, 1] > 0.23) & (V[:, 1] < 0.32)
            & (np.abs(V[:, 2]) < 0.10)
        )
        idx = pick(side_mask, axis=1, kind="max")   # topmost in y
        landmarks[f"acromion_{side}"] = {
            "index": idx, "xyz": V[idx].tolist(),
            "rule": (
                f"side={side}, |x| in [0.13, 0.22] (torso width — not arm), "
                f"y in [0.23, 0.32], |z|<0.10; argmax y (top of shoulder pad)"
            ),
        }

    # ── Greater trochanter (outer hip): for each side, most lateral
    # vertex around y near the hip joint (y in [-0.10, +0.10]).
    for side, x_sign in (("left", -1), ("right", +1)):
        side_mask = (
            (V[:, 0] * x_sign > 0.05)
            & (V[:, 1] > -0.15) & (V[:, 1] < 0.05)
        )
        idx = pick(side_mask, axis=0, kind="max" if x_sign > 0 else "min")
        landmarks[f"greater_trochanter_{side}"] = {
            "index": idx, "xyz": V[idx].tolist(),
            "rule": f"side={side}, y in [-0.15, 0.05]; argmax|min x (most lateral)",
        }

    # ── Lateral epicondyle (outer knee). Knee in T-pose is around
    # y ~ -0.55 to -0.70.
    for side, x_sign in (("left", -1), ("right", +1)):
        side_mask = (
            (V[:, 0] * x_sign > 0.03)
            & (V[:, 1] > -0.80) & (V[:, 1] < -0.50)
        )
        idx = pick(side_mask, axis=0, kind="max" if x_sign > 0 else "min")
        landmarks[f"lateral_epicondyle_{side}"] = {
            "index": idx, "xyz": V[idx].tolist(),
            "rule": f"side={side}, y in [-0.80, -0.50]; argmax|min x (most lateral)",
        }

    # ── Lateral malleolus (outer ankle bone). Ankle T-pose y ~ -1.10.
    for side, x_sign in (("left", -1), ("right", +1)):
        side_mask = (
            (V[:, 0] * x_sign > 0.0)
            & (V[:, 1] > -1.15) & (V[:, 1] < -0.95)
        )
        idx = pick(side_mask, axis=0, kind="max" if x_sign > 0 else "min")
        landmarks[f"lateral_malleolus_{side}"] = {
            "index": idx, "xyz": V[idx].tolist(),
            "rule": f"side={side}, y in [-1.15, -0.95]; argmax|min x (most lateral)",
        }

    print()
    print(f"[probe] derived {len(landmarks)} landmark vertex indices:")
    for name, info in landmarks.items():
        x, y, z = info["xyz"]
        print(f"  {name:<30s} idx={info['index']:>5d}  "
              f"T-pose xyz=({x:+.3f}, {y:+.3f}, {z:+.3f})")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(landmarks, indent=2))
    print()
    print(f"[probe] wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
