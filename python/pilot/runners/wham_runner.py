"""
wham_runner.py — phase2b Modal entrypoint for WHAM bone-center inference.

Runs WHAM (yohanshin/WHAM @ pinned commit) on a single golf swing video
on Modal A10G GPU. Returns the 3D joint-center timeline + 2D back-
projection (for overlay rendering) in the PilotRunResult schema
(see runners/_base.py).

This is the FIRST inference function in the Phase 2 pilot. Real GPU
spend: ~$0.01 per 7-second clip on A10G ($1.10/hr × ~30s). Modal Image
first-build is ~10-15 min (DPVO CUDA-extension compile dominates);
cached afterward.

Invocation (CC drives, after setup_models has populated the Volume):

    ./.venv-pilot/Scripts/python.exe -m modal run \\
        python/pilot/runners/wham_runner.py::run_wham \\
        --video-id b3fea3f0-e248-44d7-a923-0bb43172b5bf \\
        --video-url 'https://<signed-supabase-url>'

The function downloads the video into /tmp inside the Modal container,
symlinks /models/wham/* + /models/body_models/* into WHAM's expected
repo layout, invokes WHAM's official demo entrypoint, then parses the
SMPL/joint output into PilotRunResult JSON.

Output: written to the local filesystem AFTER the Modal function
returns. Location:
    python/pilot/output/wham/<video_id>/
      ├── joint_centers_3d.json       (PilotRunResult shape)
      └── overlay_2d.mp4              (rendered locally, see _render_overlay)

Inference failure modes (expected first run):
  - WHAM Image build fails at DPVO compile → re-check CUDA toolkit /
    nvcc availability in modal_app.py wham_image apt_install.
  - Body models missing → setup_models.py warning becomes a hard error
    here. Re-run `modal volume put` per modal_app.py docstring.
  - WHAM checkpoint format mismatch → spec §9 Q5: pinned commit + pinned
    weight versions should keep this in lockstep, but verify SHA-256 if
    nondeterministic.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

# Make `from modal_app import ...` work when invoked as a script
# (`python wham_runner.py` or `modal run wham_runner.py`) — modal_app
# lives one directory up. Relative `from ..modal_app` is tried first
# and works when the runner is invoked as a package
# (`python -m pilot.runners.wham_runner`).
_PILOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PILOT_DIR not in sys.path:
    sys.path.insert(0, _PILOT_DIR)

try:
    from ..modal_app import app, model_volume, wham_image, _MODAL_AVAILABLE
except ImportError:  # pragma: no cover
    from modal_app import app, model_volume, wham_image, _MODAL_AVAILABLE


# ---------------------------------------------------------------------------
# WHAM repo conventions inside the Image.
# ---------------------------------------------------------------------------

WHAM_REPO_ROOT       = "/opt/wham"
WHAM_CHECKPOINTS_DIR = f"{WHAM_REPO_ROOT}/checkpoints"
WHAM_BODY_MODELS_DIR = f"{WHAM_REPO_ROOT}/dataset/body_models"

# Mount points inside the function:
VOLUME_WHAM_DIR        = "/models/wham"
VOLUME_BODY_MODELS_DIR = "/models/body_models"


def _setup_workspace() -> None:
    """
    Stitch the Volume's weight files into WHAM's expected repo layout
    via symlinks. WHAM's demo expects:
      - /opt/wham/checkpoints/{wham_vit_w_3dpw.pth.tar, ...}
      - /opt/wham/dataset/body_models/{smpl, smplh, smplx}/...
    Volume mounts under /models/{wham, body_models}.
    """
    # checkpoints/ symlink farm
    os.makedirs(WHAM_CHECKPOINTS_DIR, exist_ok=True)
    if os.path.isdir(VOLUME_WHAM_DIR):
        for fname in os.listdir(VOLUME_WHAM_DIR):
            src = os.path.join(VOLUME_WHAM_DIR, fname)
            dst = os.path.join(WHAM_CHECKPOINTS_DIR, fname)
            if os.path.exists(dst) or os.path.islink(dst):
                continue
            os.symlink(src, dst)
            print(f"[wham_runner] symlink {dst} → {src}")
    else:
        raise RuntimeError(
            f"{VOLUME_WHAM_DIR} not present — run setup_models.py first"
        )

    # body_models/ symlink (single tree symlink — WHAM expects the
    # smpl/smplh/smplx subdirs underneath).
    if os.path.isdir(VOLUME_BODY_MODELS_DIR):
        parent = os.path.dirname(WHAM_BODY_MODELS_DIR)
        os.makedirs(parent, exist_ok=True)
        if not (
            os.path.exists(WHAM_BODY_MODELS_DIR)
            or os.path.islink(WHAM_BODY_MODELS_DIR)
        ):
            os.symlink(VOLUME_BODY_MODELS_DIR, WHAM_BODY_MODELS_DIR)
            print(
                f"[wham_runner] symlink {WHAM_BODY_MODELS_DIR} → "
                f"{VOLUME_BODY_MODELS_DIR}"
            )
    else:
        raise RuntimeError(
            f"{VOLUME_BODY_MODELS_DIR} not present — "
            f"upload via `modal volume put swingcue-pilot-models "
            f"./local-body-models /models/body_models`"
        )


def _download_video(video_url: str, dst_path: str) -> None:
    """Pull the source video onto local disk inside the Modal function."""
    import urllib.request
    print(f"[wham_runner] downloading video → {dst_path}")
    urllib.request.urlretrieve(video_url, dst_path)
    sz_mb = os.path.getsize(dst_path) / 1024 / 1024
    print(f"[wham_runner]   ({sz_mb:.2f} MB)")


# ---------------------------------------------------------------------------
# Modal function entrypoint.
# ---------------------------------------------------------------------------

if _MODAL_AVAILABLE:
    import modal

    @app.function(
        image=wham_image,
        volumes={"/models": model_volume},
        gpu="A10G",
        timeout=600,
    )
    def run_wham(
        video_id: str,
        video_url: str,
        primary_checkpoint: str = "wham_vit_w_3dpw.pth.tar",
    ) -> dict:
        """
        Run WHAM on one video and return the joint-center timeline.

        Args:
            video_id:           string for output naming.
            video_url:          HTTP(S) URL the function can fetch from.
                                Supabase signed URLs work; local path
                                via 'file://' also works inside the
                                Modal sandbox.
            primary_checkpoint: which WHAM weight to use. Default is
                                the 3dpw-trained primary. Alternate:
                                "wham_vit_bedlam_w_3dpw.pth.tar".

        Returns: PilotRunResult-shaped dict (see runners/_base.py).
        """
        import json
        import sys

        _setup_workspace()

        # Download the swing video into /tmp.
        local_video = f"/tmp/{video_id}.mp4"
        _download_video(video_url, local_video)

        # WHAM exposes a demo.py at repo root. It writes outputs under
        # output/demo/<video_basename>/.
        # TODO(phase2b): re-check demo.py CLI signature against WHAM's
        # README — flags may be `--video <path> --output_pth output.pth`
        # vs `--input <path>`. The pinned commit's CLI is what we
        # benchmark against; if it changes, update here.
        out_dir = f"{WHAM_REPO_ROOT}/output/demo/{video_id}"
        os.makedirs(out_dir, exist_ok=True)
        demo_cmd = [
            sys.executable,
            f"{WHAM_REPO_ROOT}/demo.py",
            "--video", local_video,
            "--output_pth", f"{out_dir}/wham_output.pth",
            "--save_pkl",
            "--visualize",  # writes the WHAM-native overlay too; useful
                             # to compare against our PilotRunResult-based
                             # render.
        ]
        print(f"[wham_runner] running WHAM demo: {' '.join(demo_cmd)}")
        # Stream stdout/stderr so build/inference logs are visible in
        # Modal's run output.
        completed = subprocess.run(
            demo_cmd,
            cwd=WHAM_REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        print(completed.stdout)
        if completed.returncode != 0:
            print(f"[wham_runner] STDERR:\n{completed.stderr}")
            raise RuntimeError(
                f"WHAM demo failed with exit {completed.returncode}"
            )

        # WHAM's demo writes wham_output.pkl by default with the
        # following keys (per the pinned commit's demo.py source):
        #   pose:            (T, 24, 3, 3)  rotation matrices  (SMPL pose)
        #   trans:            (T, 3)         root translation
        #   betas:            (T, 10)        SMPL shape
        #   verts:            (T, 6890, 3)   posed SMPL vertices
        #   joints:           (T, 24, 3)     world-frame joint centers (METERS)
        #   contact:          (T, 4)
        #   frame_ids:        (T,)
        # The pkl path is implicitly at out_dir/wham_output.pkl.
        # We translate that into our PilotRunResult shape.
        # TODO(phase2b): verify the actual pkl filename WHAM uses
        # (could be 'demo.pkl' or '{video_id}.pkl') and parse accordingly.
        pkl_path = f"{out_dir}/wham_output.pkl"
        if not os.path.exists(pkl_path):
            # Fall back: scan out_dir for any .pkl
            pkls = [p for p in os.listdir(out_dir) if p.endswith(".pkl")]
            if pkls:
                pkl_path = os.path.join(out_dir, pkls[0])
                print(f"[wham_runner] using pkl fallback: {pkl_path}")
            else:
                raise RuntimeError(f"WHAM produced no .pkl under {out_dir}")

        import joblib
        wham_out = joblib.load(pkl_path)
        print(f"[wham_runner] wham_output keys: {list(wham_out.keys())}")

        # Translate WHAM joint output → PilotRunResult.frames.
        # WHAM emits 24 SMPL joints; we keep 20 (drop hands + toes).
        # SMPL_JOINT_INDEX_TO_NAME for the 20 we keep:
        SMPL_TO_PILOT_NAME = {
            0:  "pelvis",
            3:  "spine1",
            6:  "spine2",
            9:  "spine3",
            12: "neck",
            15: "head",
            16: "left_shoulder",   17: "right_shoulder",
            18: "left_elbow",      19: "right_elbow",
            20: "left_wrist",      21: "right_wrist",
            1:  "left_hip",        2:  "right_hip",
            4:  "left_knee",       5:  "right_knee",
            7:  "left_ankle",      8:  "right_ankle",
            10: "left_foot",       11: "right_foot",
        }

        joints_3d = wham_out.get("joints")
        if joints_3d is None:
            raise RuntimeError(
                f"WHAM pkl missing 'joints' field. keys={list(wham_out.keys())}"
            )

        frame_ids = wham_out.get("frame_ids")
        n_frames = len(joints_3d)

        # Read video metadata locally (ffprobe via opencv was apt-installed
        # in wham_image).
        import cv2
        cap = cv2.VideoCapture(local_video)
        fps_native = cap.get(cv2.CAP_PROP_FPS) or 30.0
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_native = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        frames_out = []
        for i in range(n_frames):
            joints_world = joints_3d[i]  # (24, 3) numpy
            joint_centers_3d = {}
            for smpl_idx, pilot_name in SMPL_TO_PILOT_NAME.items():
                xyz = joints_world[smpl_idx].tolist()
                joint_centers_3d[pilot_name] = xyz
            fi = int(frame_ids[i]) if frame_ids is not None else i
            frames_out.append({
                "ts":                       round(fi / fps_native, 3),
                "frame_idx":                fi,
                "joint_centers_3d":         joint_centers_3d,
                # TODO(phase2b): 2D back-projection requires the camera
                # extrinsics. WHAM's SLAM stage emits per-frame
                # cam_R / cam_t — extract from wham_out['cam_*'] keys
                # and project here. For first smoke, leave as None;
                # local render_overlay can fall back to verts-mean
                # projection.
                "joint_centers_2d_projected": None,
                "smpl_betas":                wham_out.get("betas")[i].tolist()
                                              if "betas" in wham_out else None,
                "smpl_pose":                 None,  # 24x3x3 too verbose; skip for smoke
            })

        result = {
            "video_id":     video_id,
            "runner":       "wham",
            "video_width":  video_w,
            "video_height": video_h,
            "fps_native":   round(fps_native, 2),
            "fps_sampled":  round(fps_native, 2),  # WHAM runs at native fps
            "duration_sec": round(n_native / fps_native, 3),
            "frames":       frames_out,
            "camera": {
                # TODO(phase2b): pull from wham_out keys
                "rotation":    None,
                "translation": None,
                "focal_px":    None,
            },
            "notes": [
                f"wham_commit=2b54f77",
                f"primary_checkpoint={primary_checkpoint}",
                f"n_frames_wham={n_frames}",
                f"n_native_frames={n_native}",
                f"video_url={video_url[:80]}{'...' if len(video_url) > 80 else ''}",
                f"smpl_to_pilot_dropped_keys=[L/R_hand_joints, L/R_toe_joints]",
            ],
        }
        return result


# ---------------------------------------------------------------------------
# Local entry (Modal-side run + local result-fetch + overlay render).
# Invoke with: modal run wham_runner.py::run_wham_local --video-id ... --video-url ...
# ---------------------------------------------------------------------------

if _MODAL_AVAILABLE:
    @app.local_entrypoint()
    def run_wham_local(
        video_id: str,
        video_url: str,
        primary_checkpoint: str = "wham_vit_w_3dpw.pth.tar",
    ) -> None:
        """
        Local driver: invokes run_wham on Modal, then writes the result
        JSON to python/pilot/output/wham/<video_id>/joint_centers_3d.json.
        """
        import json
        from pathlib import Path

        print(f"[wham_runner] invoking run_wham on Modal for {video_id}")
        result = run_wham.remote(
            video_id=video_id,
            video_url=video_url,
            primary_checkpoint=primary_checkpoint,
        )
        out_dir = Path(f"python/pilot/output/wham/{video_id}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "joint_centers_3d.json"
        out_path.write_text(json.dumps(result, indent=2))
        sz_kb = out_path.stat().st_size / 1024
        print(f"[wham_runner] wrote {out_path} ({sz_kb:.1f} KB)")
        print(
            f"[wham_runner] next: render 2D overlay via "
            f"python -m pilot.runners._overlay {video_id}"
        )


# ---------------------------------------------------------------------------
# Local self-test (no Modal cost).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _MODAL_AVAILABLE:
        print("[wham_runner] modal not installed")
    else:
        print(f"[wham_runner] app          = {app!r}")
        print(f"[wham_runner] image        = wham_image")
        print(f"[wham_runner] gpu          = A10G")
        print(f"[wham_runner] timeout_sec  = 600")
        print(
            "[wham_runner] invoke remotely with: "
            "modal run python/pilot/runners/wham_runner.py::run_wham_local "
            "--video-id <uuid> --video-url '<https URL>'"
        )
