"""
render_mesh_cloud.py — PR-8b.2 vertex-cloud projection sanity test.

CHEAP ALTERNATIVE to running WHAM's pytorch3d-based canonical mesh
renderer. If our CLIFF pinhole projection matches WHAM's internal
projection math, then projecting a SAMPLE of the SMPL mesh vertices
should produce a dot cloud whose silhouette aligns with the body in
the video frame.

  Aligned dot cloud → our projection chain is correct; any residual
                      visual drift is WHAM internal (PR-8b.3 territory:
                      trans stabilization).
  Misaligned       → projection bug remains; install pytorch3d for
                      definitive canonical render comparison.

Requires:
  - Modal-side function that returns sampled mesh vertices for 3 key
    frames (we don't want to ship 120*6890*3*4 = 10 MB per video back).
  - We sample 600 vertices spread across the mesh (every ~11th vertex).

Output: docs/PR-8b_VISUAL_VERIFY/v4_canonical/f{frame}_mesh_cloud.png
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
VIDEO = ROOT / "python/benchmark/test_videos/b32e0f21-2656-473c-aa87-e1eaf6e1221f.mp4"
OUT_DIR = ROOT / "docs/PR-8b_VISUAL_VERIFY/v4_canonical"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CLIFF projection (matches PR-8b.2 _extract_camera fallback).
W, H = 720, 1280
FOCAL = float((W ** 2 + H ** 2) ** 0.5)
CX, CY = W / 2, H / 2


def project(verts_3d: np.ndarray) -> np.ndarray:
    """(N, 3) camera-frame meters → (N, 2) pixel coords."""
    X = verts_3d[:, 0]
    Y = verts_3d[:, 1]
    Z = verts_3d[:, 2]
    eps = 1e-6
    Z_safe = np.where(np.abs(Z) < eps, eps, Z)
    u = FOCAL * X / Z_safe + CX
    v = FOCAL * Y / Z_safe + CY
    return np.stack([u, v], axis=-1)


def main() -> int:
    import modal
    import httpx

    sys.path.insert(0, str(ROOT))
    # Re-use the env-loader from run_inspect_pkl
    spec_dir = ROOT / "tmp"
    if str(spec_dir) not in sys.path:
        sys.path.insert(0, str(spec_dir))

    # Load .env.local and get signed URL the same way.
    env_path = ROOT / ".env.local"
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    supa_url = os.environ["NEXT_PUBLIC_SUPABASE_URL"]
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["NEXT_PUBLIC_SUPABASE_ANON_KEY"]
    video_id = "b32e0f21-2656-473c-aa87-e1eaf6e1221f"
    r = httpx.get(
        f"{supa_url}/rest/v1/swing_videos",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
        params={"id": f"eq.{video_id}", "select": "storage_path"},
        timeout=30,
    )
    r.raise_for_status()
    storage_path = r.json()[0]["storage_path"]
    sign = httpx.post(
        f"{supa_url}/storage/v1/object/sign/swing-videos/{storage_path}",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
        json={"expiresIn": 1200},
        timeout=30,
    )
    sign.raise_for_status()
    signed_url = f"{supa_url}/storage/v1{sign.json()['signedURL']}"

    print(f"[mesh_cloud] invoking sample_mesh_verts on Modal ...")
    fn = modal.Function.from_name("swingcue-pilot", "sample_mesh_verts")
    out = fn.remote(
        video_id=video_id,
        video_url=signed_url,
        sample_frame_indices=[8, 44, 60, 90],
        stride=12,   # ~574 verts per frame (6890 / 12)
    )

    cap = cv2.VideoCapture(str(VIDEO))
    for entry in out["frames"]:
        fi = entry["frame_idx"]
        verts = np.asarray(entry["verts_sampled"], dtype=np.float32)
        pts = project(verts)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            print(f"[mesh_cloud] could not read frame {fi}")
            continue
        canvas = frame.copy()
        for (u, v) in pts:
            if 0 <= int(u) < W and 0 <= int(v) < H:
                cv2.circle(canvas, (int(u), int(v)), 2, (0, 200, 255), -1)  # orange dots
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(canvas, f"f={fi:03d}  WHAM mesh ({len(pts)} verts) projected via CLIFF pinhole",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        out_path = OUT_DIR / f"f{fi:03d}_mesh_cloud.png"
        cv2.imwrite(str(out_path), canvas)
        print(f"[mesh_cloud] wrote {out_path.name}  {canvas.shape[1]}x{canvas.shape[0]}")
    cap.release()
    print(f"[mesh_cloud] DONE — {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
