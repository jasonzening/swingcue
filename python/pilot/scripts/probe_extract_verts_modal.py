"""
probe_extract_verts_modal.py — sibling Modal entrypoint that runs WHAM
on one video and returns SMPL verts at a small set of frame indices.

Probe-only. Does NOT replace or modify wham_runner.py. Used once per
investigation to test SMPL vertex sampling vs the current PR-7a
offset-vector correction (PR-7a.4 probe per Jason's spec).

Output: writes verts_at_frames.json (sidecar) next to the existing
joint_centers_3d.json under python/pilot/output/wham/<video_id>/.

CLI (from repo root):
    .venv-pilot/Scripts/python.exe \\
        python/pilot/scripts/probe_extract_verts_modal.py \\
        --video-id <uuid> \\
        --frames 7,56,70,90,125

The signed Supabase URL is fetched by reusing run_wham_one.py's helper.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# REPO_ROOT is only used by the LOCAL CLI driver (cli_entry, sidecar
# write). Inside the Modal container __file__ is /root/probe_extract...
# and parents[3] would IndexError. Use a safe local-only resolution.
try:
    REPO_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(REPO_ROOT / "python" / "pilot"))
except (IndexError, AttributeError):
    REPO_ROOT = None  # type: ignore[assignment]

# Reuse the same Image + volumes the production runner uses —
# guarantees parity with joint_centers_3d.json we already have.
from modal_app import app, model_volume, wham_image, _MODAL_AVAILABLE  # type: ignore

# Inlined from runners/wham_runner.py — Modal's package mount
# auto-detection doesn't include sibling `runners/` from the scripts/
# entrypoint, so importing from it crashes the container. Inlining is
# the cheap fix for this probe-only sibling script. Keep in sync with
# wham_runner.py if those helpers change.
WHAM_REPO_ROOT         = "/opt/wham"
WHAM_CHECKPOINTS_DIR   = f"{WHAM_REPO_ROOT}/checkpoints"
WHAM_BODY_MODELS_DIR   = f"{WHAM_REPO_ROOT}/dataset/body_models"
VOLUME_WHAM_DIR        = "/models/wham"
VOLUME_BODY_MODELS_DIR = "/models/body_models"


def _setup_workspace() -> None:
    os.makedirs(WHAM_CHECKPOINTS_DIR, exist_ok=True)
    if os.path.isdir(VOLUME_WHAM_DIR):
        for fname in os.listdir(VOLUME_WHAM_DIR):
            src = os.path.join(VOLUME_WHAM_DIR, fname)
            dst = os.path.join(WHAM_CHECKPOINTS_DIR, fname)
            if os.path.exists(dst) or os.path.islink(dst):
                continue
            os.symlink(src, dst)
    else:
        raise RuntimeError(f"{VOLUME_WHAM_DIR} not present")
    if os.path.isdir(VOLUME_BODY_MODELS_DIR):
        parent = os.path.dirname(WHAM_BODY_MODELS_DIR)
        os.makedirs(parent, exist_ok=True)
        if not (
            os.path.exists(WHAM_BODY_MODELS_DIR)
            or os.path.islink(WHAM_BODY_MODELS_DIR)
        ):
            os.symlink(VOLUME_BODY_MODELS_DIR, WHAM_BODY_MODELS_DIR)
    else:
        raise RuntimeError(f"{VOLUME_BODY_MODELS_DIR} not present")


def _download_video(video_url: str, dst_path: str) -> None:
    import urllib.parse
    import urllib.request
    parts = urllib.parse.urlsplit(video_url)
    safe_path = urllib.parse.quote(parts.path, safe="/")
    encoded_url = urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, safe_path, parts.query, parts.fragment)
    )
    print(f"[probe] downloading video -> {dst_path}")
    urllib.request.urlretrieve(encoded_url, dst_path)
    sz_mb = os.path.getsize(dst_path) / 1024 / 1024
    print(f"[probe]   ({sz_mb:.2f} MB)")

BUCKET = "swing-videos"
SIGNED_URL_EXPIRY_SEC = 600


def _load_env_local(path: Path) -> None:
    if not path.exists():
        sys.exit(f"[probe] missing {path}")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _fetch_storage_path(supa_url: str, key: str, video_id: str) -> str:
    import httpx
    r = httpx.get(
        f"{supa_url}/rest/v1/swing_videos",
        headers={"apikey": key, "Authorization": f"Bearer {key}",
                 "Accept": "application/json"},
        params={"id": f"eq.{video_id}", "select": "id,storage_path"},
        timeout=30,
    )
    r.raise_for_status()
    rows = r.json()
    if not rows:
        sys.exit(f"[probe] no DB row for {video_id}")
    return rows[0]["storage_path"]


def _sign_url(supa_url: str, key: str, path: str) -> str:
    import httpx
    r = httpx.post(
        f"{supa_url}/storage/v1/object/sign/{BUCKET}/{path}",
        headers={"apikey": key, "Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        json={"expiresIn": SIGNED_URL_EXPIRY_SEC},
        timeout=30,
    )
    r.raise_for_status()
    signed = r.json().get("signedURL") or r.json().get("signedUrl")
    if not signed:
        sys.exit("[probe] sign failed")
    if signed.startswith("/"):
        signed = f"{supa_url}/storage/v1{signed}"
    return signed


# ── Modal function: WHAM + verts extraction ─────────────────────────
if _MODAL_AVAILABLE:
    import modal

    @app.function(
        image=wham_image,
        volumes={"/models": model_volume},
        gpu="A10G",
        timeout=600,
    )
    def run_wham_extract_verts(
        video_id: str,
        video_url: str,
        frame_indices: list[int],
        primary_checkpoint: str = "wham_vit_w_3dpw.pth.tar",
    ) -> dict:
        """
        Run WHAM on a video, then load the resulting .pkl and return
        SMPL verts at the requested frame indices.

        Returns: {
            "video_id": str,
            "video_width": int, "video_height": int, "fps_native": float,
            "frame_indices": list[int],
            "verts_at_frames": {frame_idx: list[6890][3]},
            "wham_frame_ids": list[int],   # actual frame_ids WHAM emitted
        }
        """
        import sys
        _setup_workspace()
        local_video = f"/tmp/{video_id}.mp4"
        _download_video(video_url, local_video)

        out_dir = f"{WHAM_REPO_ROOT}/output/demo/{video_id}"
        os.makedirs(out_dir, exist_ok=True)
        demo_cmd = [
            sys.executable, f"{WHAM_REPO_ROOT}/demo.py",
            "--video", local_video,
            "--output_pth", f"{out_dir}/wham_output.pth",
            "--save_pkl",
        ]
        print(f"[probe] running WHAM demo: {' '.join(demo_cmd)}")
        # WHAM demo.py uses relative paths for configs/yamls/demo.yaml —
        # must run from WHAM_REPO_ROOT (matches production wham_runner.py).
        proc = subprocess.run(demo_cmd, capture_output=True, text=True,
                              cwd=WHAM_REPO_ROOT)
        print(f"[probe] demo stdout ({len(proc.stdout)} chars):\n{proc.stdout[:400]}")
        if proc.returncode != 0:
            raise RuntimeError(
                f"WHAM demo failed exit={proc.returncode}\n"
                f"stderr={proc.stderr[:600]}"
            )

        # Locate the .pkl (path mirrors wham_runner.py's logic).
        import glob
        pkl_paths = glob.glob(f"{out_dir}/**/wham_output.pkl", recursive=True)
        if not pkl_paths:
            raise RuntimeError(f"no wham_output.pkl under {out_dir}")
        pkl_path = pkl_paths[0]
        print(f"[probe] loading pkl: {pkl_path}")

        import joblib
        import numpy as np
        # WHAM writes via joblib (not vanilla pickle).
        wham_out = joblib.load(pkl_path)
        track_ids = list(wham_out.keys())
        track = wham_out[track_ids[0]]
        verts = np.asarray(track["verts"])          # (T, 6890, 3)
        frame_ids = [int(x) for x in track.get("frame_ids", list(range(len(verts))))]
        print(f"[probe] verts shape={verts.shape} dtype={verts.dtype} "
              f"frame_ids[0..3]={frame_ids[:3]} ... [-3..]={frame_ids[-3:]}")

        # Read video metadata.
        import cv2
        cap = cv2.VideoCapture(local_video)
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_native = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        # Map requested frame_idx → row in WHAM output. WHAM's frame_ids
        # may not be 0..T-1 (some frames are skipped if YOLO didn't
        # detect). Build a lookup.
        idx_to_row = {fi: i for i, fi in enumerate(frame_ids)}
        verts_at_frames: dict[str, list] = {}
        for fi in frame_indices:
            if fi not in idx_to_row:
                # Pick nearest WHAM frame_id.
                nearest = min(frame_ids, key=lambda f: abs(f - fi))
                row = idx_to_row[nearest]
                print(f"[probe] frame {fi} not in WHAM output; using nearest {nearest}")
                verts_at_frames[str(fi)] = {
                    "wham_frame_id": nearest,
                    "verts": verts[row].tolist(),
                }
            else:
                row = idx_to_row[fi]
                verts_at_frames[str(fi)] = {
                    "wham_frame_id": fi,
                    "verts": verts[row].tolist(),
                }

        return {
            "video_id":        video_id,
            "video_width":     video_w,
            "video_height":    video_h,
            "fps_native":      fps_native,
            "frame_indices":   frame_indices,
            "verts_at_frames": verts_at_frames,
            "n_wham_frames":   int(verts.shape[0]),
            "verts_shape":     list(verts.shape),
        }


if _MODAL_AVAILABLE:
    @app.local_entrypoint()
    def main(
        video_id: str,
        video_url: str,
        frames: str,   # comma-separated frame indices, e.g. "7,56,70,90,125"
    ) -> None:
        """Local driver — fetches signed URL and invokes the Modal function."""
        frame_indices = [int(s.strip()) for s in frames.split(",")]
        print(f"[probe] requesting verts at frames {frame_indices} for {video_id}")
        result = run_wham_extract_verts.remote(
            video_id=video_id,
            video_url=video_url,
            frame_indices=frame_indices,
        )
        out_dir = REPO_ROOT / "python" / "pilot" / "output" / "wham" / video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "verts_at_frames.json"
        out_path.write_text(json.dumps(result, indent=2))
        sz_kb = out_path.stat().st_size / 1024
        print(f"[probe] wrote {out_path} ({sz_kb:.1f} KB)")
        print(f"[probe] n_wham_frames={result['n_wham_frames']}  "
              f"verts_shape={result['verts_shape']}")


# Optional: convenience wrapper that signs URLs from .env.local + invokes Modal.
def cli_entry() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--frames", required=True,
                    help="comma-separated frame indices, e.g. 7,56,70,90,125")
    args = ap.parse_args()

    _load_env_local(REPO_ROOT / ".env.local")
    supa_url = os.environ.get("NEXT_PUBLIC_SUPABASE_URL", "").rstrip("/")
    key      = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not supa_url or not key:
        sys.exit("[probe] env not set")
    storage_path = _fetch_storage_path(supa_url, key, args.video_id)
    signed_url = _sign_url(supa_url, key, storage_path)
    print(f"[probe] signed (expires {SIGNED_URL_EXPIRY_SEC}s)")

    cmd = [
        str(REPO_ROOT / ".venv-pilot" / "Scripts" / "modal.exe"),
        "run",
        str(REPO_ROOT / "python" / "pilot" / "scripts" / "probe_extract_verts_modal.py")
        + "::main",
        "--video-id", args.video_id,
        "--video-url", signed_url,
        "--frames", args.frames,
    ]
    print(f"[probe] exec: modal run probe_extract_verts_modal.py::main "
          f"--video-id {args.video_id} --video-url <signed:redacted> "
          f"--frames {args.frames}")
    sub_env = os.environ.copy()
    sub_env["PYTHONIOENCODING"] = "utf-8"
    sub_env["PYTHONUTF8"] = "1"
    rc = subprocess.call(cmd, cwd=str(REPO_ROOT), env=sub_env)
    sys.exit(rc)


if __name__ == "__main__":
    cli_entry()
