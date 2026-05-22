"""
fit_offsets_from_gt.py — derive per-joint **body-local** 3D
anatomical offset vectors from Jason's 15 ground-truth labels.

PR-7a Path B (post-Issue-2): camera-frame offset vectors worked at
setup but mis-aligned through the swing as the golfer's body
rotates (Issue 2: at finish/impact, neck offset's camera-frame dy
sent the coaching anchor flying off-screen). The fix is to express
offsets in body-local coordinates — they rotate with the body and
remain anatomically correct across all phases.

Math (per (joint, view) pair; aggregated over all GT samples):

    For each GT sample (joint J, raw_3d, gt_2d) at depth Z = raw_3d[2]:

        # Project raw 3D to 2D and compute pixel residual.
        raw_2d = project(raw_3d)
        du, dv = gt_2d.x - raw_2d.x, gt_2d.y - raw_2d.y

        # Inverse-project pixel residual to 3D camera-frame delta at
        # THIS sample's depth (pinhole linearization at Z).
        cam_offset = (du * Z / fx, dv * Z / fy, 0.0)

        # Express in body-local basis at THIS sample's pose:
        #   horizontal     = unit(cross(spine_up, cam_z))
        #   spine_up       = unit(neck - pelvis)
        #   body_forward   = unit(cross(horizontal, spine_up))
        body_local_offset = (cam_offset · horizontal,
                             cam_offset · spine_up,
                             cam_offset · body_forward)

    Final fitted body-local vector for (joint, view) = trimmed-mean
    per axis across samples. 10%-trimmed mean for n>=5, median for
    smaller samples. Outliers (e.g., WHAM tracking failures at
    impact/finish) are excluded.

    For hip-class joints (per anatomical_offset.HORIZONTAL_ONLY_OFFSET_KEYS),
    the d_v (body vertical) component is zeroed in storage AND at
    apply time — SMPL hip bone-center error is purely lateral in body
    coords too (Finding G constraint preserved in body-local frame).

Output structure (per view): mixes per-joint body-local 3D vectors
(for joints with GT labels) and group-keyed scalars (for joints
without: knee/ankle/wrist). The engine dispatches by value type —
list of 3 floats → mode A body-local vector, scalar → mode B legacy
inward-pull.

CLI:
    .venv-benchmark/Scripts/python.exe \\
        python/motion_correction/scripts/fit_offsets_from_gt.py [--write]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))

from motion_correction.engine.anatomical_offset import (
    HORIZONTAL_ONLY_OFFSET_KEYS,
    KEYPOINT_TO_OFFSET_KEY,
    body_local_basis,
    camera_to_body_local,
)
from motion_correction.engine.projection import (
    default_intrinsics,
    project_xyz_to_uv,
)

GT_DIR  = REPO_ROOT / "docs" / "PR-7_GROUND_TRUTH" / "golf"
WHAM_DIR = REPO_ROOT / "python" / "pilot" / "output" / "wham"
CONFIG_PATH = (
    REPO_ROOT / "python" / "motion_correction" / "domains" / "golf" / "config.py"
)

# Mapping GT-label keys → raw WHAM joint name.
# GT corpus uses "neck_center" for what WHAM calls "neck".
GT_KEY_TO_WHAM_NAME: dict[str, str] = {
    "left_shoulder":  "left_shoulder",
    "right_shoulder": "right_shoulder",
    "left_hip":       "left_hip",
    "right_hip":      "right_hip",
    "neck_center":    "neck",
}

# Trimmed-mean trim fraction (per tail). 0.10 = drop bottom 10% + top 10%.
# For 10 face_on samples that's 1 from each tail (keep middle 8); for 5
# DTL samples it falls back to median (statistics.median).
TRIM_FRACTION = 0.10

# Carry-over scalars for joints without ground-truth labels.
FALLBACK_NON_FITTED_FACE_ON = {
    "knee_inward":  0.05,
    "ankle_inward": 0.03,
    "wrist_inward": 0.00,
}
FALLBACK_NON_FITTED_DTL = {
    "knee_inward":  0.05,
    "ankle_inward": 0.03,
    "wrist_inward": 0.00,
}
# Pre-Option-2 hardcoded scalar fallbacks, kept for documentation
# (ANATOMICAL_OFFSETS_FALLBACK in the generated config).
FALLBACK_FACE_ON_SCALAR_LEGACY = {
    "shoulder_inward": 0.14,
    "hip_inward":      0.16,
    "head_inward":     0.08,
    "knee_inward":     0.05,
    "ankle_inward":    0.03,
    "wrist_inward":    0.00,
}
FALLBACK_DTL_SCALAR_LEGACY = {
    "shoulder_inward": 0.18,
    "hip_inward":      0.20,
    "head_inward":     0.08,
    "knee_inward":     0.05,
    "ankle_inward":    0.03,
    "wrist_inward":    0.00,
}


def _trimmed_mean(values: list[float]) -> float:
    """80%-trimmed mean for n>=5; median for smaller samples."""
    if not values:
        return 0.0
    if len(values) < 5:
        return statistics.median(values)
    s = sorted(values)
    k = max(1, int(round(len(s) * TRIM_FRACTION)))
    trimmed = s[k:-k] if k * 2 < len(s) else s
    return sum(trimmed) / len(trimmed)


def _fit_one_sample(
    raw_frame: dict,
    gt_label: dict,
    wham_name: str,
    intr: dict[str, float],
) -> Optional[tuple[float, float, float]]:
    """
    Compute the per-sample body-local 3D offset that would move raw_2d
    to gt_2d at this sample's depth Z. Returns None when unusable
    (body basis indeterminate, depth invalid, etc.).
    """
    kp = raw_frame.get("joint_centers_3d", {})
    raw_3d = kp.get(wham_name)
    if raw_3d is None or len(raw_3d) != 3 or raw_3d[2] is None or raw_3d[2] <= 0:
        return None

    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    raw_2d = project_xyz_to_uv(raw_3d, fx, fy, cx, cy)
    if raw_2d is None:
        return None

    gx, gy = float(gt_label["x"]), float(gt_label["y"])
    du = gx - raw_2d[0]
    dv = gy - raw_2d[1]

    Z = raw_3d[2]
    # Inverse-project pixel residual to 3D camera-frame delta at depth Z.
    cam_offset = [du * Z / fx, dv * Z / fy, 0.0]

    # Express in body-local basis at THIS sample's pose. If the spine
    # basis is indeterminate (degenerate pose), skip the sample.
    basis = body_local_basis(kp.get("pelvis"), kp.get("neck"))
    if basis is None:
        return None
    d_h, d_v, d_f = camera_to_body_local(cam_offset, basis)
    return (d_h, d_v, d_f)


LABELED_PHASES: tuple[str, ...] = ("setup", "top", "transition", "impact", "finish")
ALL_PHASES: tuple[str, ...] = (
    "setup", "backswing", "top", "transition", "downswing", "impact", "finish",
)
# Inter-phase lerp recipe for unlabeled phases.
LERP_RECIPE: dict[str, tuple[str, str, float]] = {
    "backswing": ("setup",      "top",    0.5),
    "downswing": ("transition", "impact", 0.5),
}
# Conservative clamp: if interp magnitude exceeds setup-vec magnitude
# × this factor, fall back to setup-vec (avoids extrapolation blowups).
INTERP_MAGNITUDE_CLAMP_FACTOR: float = 1.5


def _vec_mag(v: list[float]) -> float:
    return math.sqrt(sum(c * c for c in v))


def _lerp_vec(a: list[float], b: list[float], t: float) -> list[float]:
    return [a[i] + (b[i] - a[i]) * t for i in range(3)]


def fit_all() -> dict[str, dict[str, dict[str, list[float]]]]:
    """
    Walk every GT file, accumulate per-sample body-local offset
    vectors keyed by (view, joint, phase), aggregate via trimmed-mean
    per channel per phase. Lerp unlabeled phases (backswing,
    downswing) from labeled neighbors with magnitude clamp.

    Returns: {view: {joint_name: {phase: [d_h, d_v, d_f]}}}
    """
    intr_by_dims: dict[tuple[int, int], dict[str, float]] = {}
    # Samples keyed by (view, joint, phase).
    samples: dict[
        tuple[str, str, str],
        list[tuple[float, float, float, str, int]],
    ] = {}
    skipped: list[str] = []

    for gt_path in sorted(GT_DIR.glob("*.json")):
        gt = json.loads(gt_path.read_text())
        view  = gt["view"]
        vid   = gt["video_id"]
        fidx  = int(gt["frame_idx"])
        phase = gt.get("phase", "setup")
        labels = gt.get("labels", {})

        raw_json = WHAM_DIR / vid / "joint_centers_3d.json"
        if not raw_json.exists():
            skipped.append(f"{gt_path.name} → no WHAM at {raw_json}")
            continue
        raw = json.loads(raw_json.read_text())
        try:
            raw_frame = next(f for f in raw["frames"] if int(f["frame_idx"]) == fidx)
        except StopIteration:
            skipped.append(f"{gt_path.name} → frame {fidx} not in WHAM output")
            continue

        W = int(raw.get("video_width", gt.get("video_width", 0)))
        H = int(raw.get("video_height", gt.get("video_height", 0)))
        intr = intr_by_dims.setdefault((W, H), default_intrinsics(W, H))

        for gt_key, wham_name in GT_KEY_TO_WHAM_NAME.items():
            gt_label = labels.get(gt_key)
            if gt_label is None or "x" not in gt_label or "y" not in gt_label:
                continue
            triple = _fit_one_sample(raw_frame, gt_label, wham_name, intr)
            if triple is None:
                continue
            dx, dy, dz = triple
            samples.setdefault((view, wham_name, phase), []).append(
                (dx, dy, dz, gt["video_id"][:8], fidx),
            )

    if skipped:
        print(f"[fit] skipped {len(skipped)} GT files:")
        for s in skipped:
            print(f"  - {s}")
        print()

    # Step 1: aggregate per (view, joint, labeled_phase).
    per_phase_fit: dict[str, dict[str, dict[str, list[float]]]] = {}
    print(f"{'view':<14s} {'joint':<18s} {'phase':<11s} {'n':>3s}  "
          f"{'fit_d_h':>9s} {'fit_d_v':>9s} {'fit_d_f':>9s}")
    print("-" * 80)
    for (view, joint, phase), entries in sorted(samples.items()):
        dhs = [e[0] for e in entries]
        dvs = [e[1] for e in entries]
        dfs = [e[2] for e in entries]
        fdh = _trimmed_mean(dhs)
        fdv = _trimmed_mean(dvs)
        fdf = _trimmed_mean(dfs)
        offset_key = KEYPOINT_TO_OFFSET_KEY.get(joint, "")
        if offset_key in HORIZONTAL_ONLY_OFFSET_KEYS:
            fdv = 0.0
        print(f"{view:<14s} {joint:<18s} {phase:<11s} {len(entries):>3d}  "
              f"{fdh:>+9.4f} {fdv:>+9.4f} {fdf:>+9.4f}")
        per_phase_fit.setdefault(view, {}).setdefault(joint, {})[phase] = [
            round(fdh, 5), round(fdv, 5), round(fdf, 5),
        ]

    # Step 2: lerp unlabeled phases (backswing, downswing).
    print()
    print("[fit] lerp unlabeled phases (backswing, downswing):")
    for view in per_phase_fit:
        for joint in per_phase_fit[view]:
            phases_present = per_phase_fit[view][joint]
            setup_vec = phases_present.get("setup")
            for target_phase, (a, b, t) in LERP_RECIPE.items():
                va = phases_present.get(a)
                vb = phases_present.get(b)
                if va is None or vb is None:
                    continue
                interp = _lerp_vec(va, vb, t)
                # Conservative magnitude clamp.
                if setup_vec is not None:
                    if _vec_mag(interp) > _vec_mag(setup_vec) * INTERP_MAGNITUDE_CLAMP_FACTOR:
                        print(f"  CLAMP  {view}/{joint}/{target_phase}: "
                              f"|interp|={_vec_mag(interp):.3f} > "
                              f"|setup|×{INTERP_MAGNITUDE_CLAMP_FACTOR}="
                              f"{_vec_mag(setup_vec) * INTERP_MAGNITUDE_CLAMP_FACTOR:.3f}, "
                              f"using setup vector")
                        interp = list(setup_vec)
                # Hip d_v zero enforcement (defense in depth).
                offset_key = KEYPOINT_TO_OFFSET_KEY.get(joint, "")
                if offset_key in HORIZONTAL_ONLY_OFFSET_KEYS:
                    interp[1] = 0.0
                interp = [round(x, 5) for x in interp]
                per_phase_fit[view][joint][target_phase] = interp
                print(f"  lerp   {view}/{joint}/{target_phase}: {interp}")

    return per_phase_fit


def render_config(
    fitted: dict[str, dict[str, dict[str, list[float]]]],
) -> dict[str, dict]:
    """
    Merge fitted per-phase vector dicts with the carry-over scalars
    (knee/ankle/wrist) into the final per-view dict.

    Output shape per view: {joint_name: {phase: [d_h, d_v, d_f]}, ...,
                            "knee_inward": float, ...}
    """
    out: dict[str, dict] = {}
    for view, non_fitted in (("face_on", FALLBACK_NON_FITTED_FACE_ON),
                              ("down_the_line", FALLBACK_NON_FITTED_DTL)):
        merged: dict = {}
        for joint, per_phase in fitted.get(view, {}).items():
            merged[joint] = {p: list(v) for p, v in per_phase.items()}
        for k, v in non_fitted.items():
            merged[k] = v
        out[view] = merged
    return out


def write_config(final_offsets: dict[str, dict]) -> None:
    """
    Replace the ANATOMICAL_OFFSETS dict in golf/config.py with the
    fitted vectors + carry-over scalars.

    Uses a sentinel-comment-bracketed region for safe in-place edit.
    Preserves the legacy scalar dict as ANATOMICAL_OFFSETS_FALLBACK.
    """
    src = CONFIG_PATH.read_text(encoding="utf-8")

    BEGIN = "# === BEGIN FITTED ANATOMICAL_OFFSETS (auto-generated by fit_offsets_from_gt.py) ==="
    END   = "# === END FITTED ANATOMICAL_OFFSETS ==="

    body_lines = ["", BEGIN, ""]
    body_lines.append(
        "# Per-joint 3D offset vectors for fitted joints (shoulder/hip/neck)"
    )
    body_lines.append(
        "# + legacy group-keyed scalars for non-fitted (knee/ankle/wrist)."
    )
    body_lines.append(
        "# Engine dispatches by value type: list[3] → mode A vector; scalar → mode B."
    )
    body_lines.append("ANATOMICAL_OFFSETS: dict[str, dict] = {")
    for view, kvs in final_offsets.items():
        body_lines.append(f"    {view!r}: {{")
        for k, v in kvs.items():
            if isinstance(v, dict):
                body_lines.append(f"        {k!r:24s}: {{")
                for phase, vec in v.items():
                    if isinstance(vec, (list, tuple)) and len(vec) == 3:
                        body_lines.append(
                            f"            {phase!r:14s}: "
                            f"[{float(vec[0]):>+8.5f}, "
                            f"{float(vec[1]):>+8.5f}, "
                            f"{float(vec[2]):>+8.5f}],"
                        )
                    else:
                        body_lines.append(f"            {phase!r:14s}: {vec!r},")
                body_lines.append("        },")
            elif isinstance(v, (list, tuple)) and len(v) == 3:
                body_lines.append(
                    f"        {k!r:24s}: [{float(v[0]):>+8.5f}, "
                    f"{float(v[1]):>+8.5f}, {float(v[2]):>+8.5f}],"
                )
            else:
                body_lines.append(f"        {k!r:24s}: {v:>+7.4f},")
        body_lines.append("    },")
    body_lines.append("}")
    body_lines += [
        "",
        "# Pre-Option-2 hardcoded scalar fallback. Preserved for reproducibility",
        "# and as a guard against config-corruption — engine still supports the",
        "# scalar inward-pull math (mode B) if a future hot-fix needs to revert.",
        "ANATOMICAL_OFFSETS_FALLBACK: dict[str, dict[str, float]] = {",
        f"    'face_on':       {FALLBACK_FACE_ON_SCALAR_LEGACY!r},",
        f"    'down_the_line': {FALLBACK_DTL_SCALAR_LEGACY!r},",
        "}",
        "",
        END,
        "",
    ]
    new_block = "\n".join(body_lines)

    if BEGIN in src and END in src:
        pre = src.split(BEGIN, 1)[0].rstrip() + "\n"
        post_idx = src.find(END) + len(END)
        post = src[post_idx:].lstrip("\n")
        new_src = pre + new_block.lstrip("\n") + "\n" + post
    else:
        # First-time install: replace the original ANATOMICAL_OFFSETS
        # assignment via simple bracket-matching.
        anchor = "ANATOMICAL_OFFSETS: dict[str, dict[str, float]] = {"
        start = src.find(anchor)
        if start == -1:
            # Maybe already-installed sentinel-less variant.
            anchor = "ANATOMICAL_OFFSETS: dict[str, dict] = {"
            start = src.find(anchor)
        if start == -1:
            raise SystemExit(
                f"[fit] could not locate ANATOMICAL_OFFSETS anchor in "
                f"{CONFIG_PATH} — write skipped"
            )
        depth = 0
        i = src.find("{", start)
        end = None
        for j in range(i, len(src)):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end is None:
            raise SystemExit(f"[fit] could not match braces in {CONFIG_PATH}")
        pre = src[:start].rstrip() + "\n"
        post = src[end:].lstrip("\n")
        new_src = pre + new_block.lstrip("\n") + "\n" + post

    CONFIG_PATH.write_text(new_src, encoding="utf-8")
    print(f"[fit] wrote fitted offsets to {CONFIG_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="Replace ANATOMICAL_OFFSETS in golf/config.py with fitted vectors")
    args = ap.parse_args()

    print("[fit] fitting per-joint 3D offset vectors from GT labels")
    print(f"[fit] gt dir : {GT_DIR}")
    print(f"[fit] wham   : {WHAM_DIR}")
    print()
    fitted = fit_all()
    print()
    print("[fit] fitted per-(view, joint, phase) body-local vectors:")
    for view, kvs in fitted.items():
        print(f"  {view}:")
        for k, per_phase in kvs.items():
            print(f"    {k}:")
            for ph in ALL_PHASES:
                if ph in per_phase:
                    v = per_phase[ph]
                    print(f"      {ph:<12s} d_h={v[0]:>+8.4f}  d_v={v[1]:>+8.4f}  d_f={v[2]:>+8.4f}")
    print()
    final = render_config(fitted)
    print("[fit] final ANATOMICAL_OFFSETS structure (joint → per-phase dict or scalar):")
    for view, kvs in final.items():
        print(f"  {view}:")
        for k, v in kvs.items():
            if isinstance(v, dict):
                print(f"    {k:<20s} (per-phase dict, {len(v)} phases)")
            elif isinstance(v, (list, tuple)) and len(v) == 3:
                print(f"    {k:<20s} [{v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f}]  (constant)")
            else:
                print(f"    {k:<20s} {v:+.4f}  (scalar fallback)")

    if args.write:
        write_config(final)


if __name__ == "__main__":
    main()
