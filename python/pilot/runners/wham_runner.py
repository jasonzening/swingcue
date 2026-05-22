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
    """
    Pull the source video onto local disk inside the Modal function.

    Stdlib urllib is strict about URL syntax: spaces in the path
    (Supabase preserves original filenames like
    "Video Project 6-miao.mp4") trigger `InvalidURL: URL can't contain
    control characters`. Percent-encode the path component before the
    request so urllib accepts it. Token-bearing query string is left
    untouched (its `+`/`=`/`.` chars are already legal).
    """
    import urllib.parse
    import urllib.request

    # Split URL into scheme://netloc/path?query so we can quote() only
    # the path (and leave the JWT query parameters alone).
    parts = urllib.parse.urlsplit(video_url)
    safe_path = urllib.parse.quote(parts.path, safe="/")
    encoded_url = urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, safe_path, parts.query, parts.fragment)
    )

    print(f"[wham_runner] downloading video → {dst_path}")
    urllib.request.urlretrieve(encoded_url, dst_path)
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
        save_smpl_params: bool = False,   # PR-7a.5 probe opt-in (default OFF)
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
            # --visualize intentionally OFF: that flag pulls in
            # pytorch3d which has no py310+cu113+pyt1110 prebuilt wheel
            # (would force a slow source build). We render our own
            # 2D back-projection overlay locally from the joint output
            # in PilotRunResult.frames[*].joint_centers_2d_projected.
        ]
        print(f"[wham_runner] running WHAM demo: {' '.join(demo_cmd)}")
        # Stream stdout/stderr so build/inference logs are visible in
        # Modal's run output. Always print both, regardless of exit
        # code — a 0 exit doesn't guarantee the pkl was written.
        completed = subprocess.run(
            demo_cmd,
            cwd=WHAM_REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        print(f"[wham_runner] demo stdout ({len(completed.stdout)} chars):")
        print(completed.stdout)
        print(f"[wham_runner] demo stderr ({len(completed.stderr)} chars):")
        print(completed.stderr)
        print(f"[wham_runner] demo exit code: {completed.returncode}")
        if completed.returncode != 0:
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
            # Fall back: scan the entire /opt/wham/output tree for any
            # .pkl. WHAM may write outputs at a different path than the
            # --output_pth arg (e.g. derived from video basename).
            print(f"[wham_runner] {pkl_path} not present; scanning WHAM output tree:")
            found_pkls: list[str] = []
            for root, _dirs, files in os.walk(f"{WHAM_REPO_ROOT}/output"):
                for f in files:
                    fp = os.path.join(root, f)
                    sz_kb = os.path.getsize(fp) / 1024
                    print(f"[wham_runner]   {sz_kb:>8.1f} KB  {fp}")
                    if f.endswith(".pkl"):
                        found_pkls.append(fp)
            if found_pkls:
                pkl_path = found_pkls[0]
                print(f"[wham_runner] using pkl fallback: {pkl_path}")
            else:
                raise RuntimeError(
                    f"WHAM produced no .pkl anywhere under "
                    f"{WHAM_REPO_ROOT}/output (demo exit was 0 — output dir "
                    f"layout differs from expected; check stdout/stderr above)"
                )

        import joblib
        wham_out = joblib.load(pkl_path)
        print(f"[wham_runner] wham_output top-level keys: {list(wham_out.keys())}")

        # WHAM's pkl is keyed by track-id (one entry per detected person
        # across the clip). For golf swings we expect exactly one
        # person → take the first track. Each track value is a dict with
        # the SMPL keys (pose, trans, betas, joints, frame_ids).
        track_ids = sorted(wham_out.keys())
        if not track_ids:
            raise RuntimeError(f"WHAM pkl had no tracks: {pkl_path}")
        primary_track_id = track_ids[0]
        track = wham_out[primary_track_id]
        print(
            f"[wham_runner] using track_id={primary_track_id} of "
            f"{len(track_ids)} total; track keys: {list(track.keys())}"
        )

        # Joint name → H36M-regressor index mapping (17 joints).
        # H36M's joint order is well-documented and stable; WHAM's
        # 31-joint J_regressor_wham has an undocumented ordering and
        # produced anatomically-impossible mappings on the first try
        # (e.g. "ankles" higher than "pelvis"). We trade off spine2/
        # spine3 + feet (H36M doesn't have them) for cleaner anatomy.
        # Foot positions can be inferred from ankle for overlay.
        H36M_TO_PILOT_NAME = {
            0:  "pelvis",
            7:  "spine1",          # H36M has 1 spine joint; spine2/3 not available
            9:  "neck",
            10: "head",
            11: "left_shoulder",   14: "right_shoulder",
            12: "left_elbow",      15: "right_elbow",
            13: "left_wrist",      16: "right_wrist",
            4:  "left_hip",        1:  "right_hip",
            5:  "left_knee",       2:  "right_knee",
            6:  "left_ankle",      3:  "right_ankle",
        }

        # WHAM doesn't emit pre-computed joint positions; it gives the
        # full posed mesh vertices + SMPL params. Joints are derived by
        # multiplying the WHAM joint regressor (24×6890) against the
        # vertex array. The J_regressor_wham.npy file we downloaded
        # into body_models is purpose-built for this.
        import numpy as np
        verts = track.get("verts")
        if verts is None:
            raise RuntimeError(
                f"WHAM track[{primary_track_id}] missing 'verts' field. "
                f"keys={list(track.keys())}"
            )
        # verts shape: (T, 6890, 3) — posed mesh vertices in WHAM's
        # output frame (camera-frame; pose_world + trans_world give the
        # SLAM-grounded world frame variant).
        verts = np.asarray(verts)
        if verts.ndim != 3 or verts.shape[-1] != 3:
            raise RuntimeError(
                f"WHAM verts shape unexpected: {verts.shape} "
                f"(expected (T, V, 3))"
            )

        # Switched from J_regressor_wham (31 joints, undocumented order)
        # to J_regressor_h36m (17 joints, well-known order) for cleaner
        # anatomy. Both .npy files live in body_models from the WHAM
        # extras tarball.
        j_regressor_path = "/models/body_models/J_regressor_h36m.npy"
        if not os.path.exists(j_regressor_path):
            raise RuntimeError(
                f"J_regressor_h36m.npy missing at {j_regressor_path} — "
                f"setup_models extras step needs to run first"
            )
        J_regressor = np.load(j_regressor_path)
        # Expected shape (24, 6890). Some variants store it as (6890, 24)
        # or (J, V) where J may be != 24; handle both orientations.
        if J_regressor.shape[1] == verts.shape[1]:
            # (J, V) — canonical
            pass
        elif J_regressor.shape[0] == verts.shape[1]:
            J_regressor = J_regressor.T
        else:
            raise RuntimeError(
                f"J_regressor shape {J_regressor.shape} incompatible with "
                f"verts shape {verts.shape} (need shared V dim)"
            )
        print(
            f"[wham_runner] computing joints: "
            f"J_regressor {J_regressor.shape} @ verts {verts.shape}"
        )
        # joints[t, j, d] = sum_v J_regressor[j, v] * verts[t, v, d]
        joints_3d = np.einsum("jv,tvd->tjd", J_regressor, verts)
        print(f"[wham_runner] derived joints shape: {joints_3d.shape}")

        # DEBUG: dump all 31 joint xyz for frame[0] so the WHAM joint
        # name ordering can be reverse-engineered. Print sorted by y
        # (typically vertical) — lowest y = pelvis/ankles, highest = head.
        # Standard SMPL convention: y is vertical (up positive in world
        # frame; here in WHAM's camera frame, y direction depends on
        # camera orientation). Either way, sorted-by-y gives a clear
        # spatial profile.
        debug_frame_idx = 0
        f0_joints = joints_3d[debug_frame_idx]
        idx_sorted = sorted(range(len(f0_joints)), key=lambda i: f0_joints[i][1])
        print(f"[wham_runner] DEBUG frame[{debug_frame_idx}] all joints (sorted by y):")
        for i in idx_sorted:
            x, y, z = f0_joints[i]
            print(f"[wham_runner]   idx={i:2d}  ({x:+7.3f}, {y:+7.3f}, {z:+7.3f})")

        frame_ids = track.get("frame_ids")
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
            joints_world = joints_3d[i]  # (17, 3) numpy (H36M order)
            joint_centers_3d = {}
            for h36m_idx, pilot_name in H36M_TO_PILOT_NAME.items():
                xyz = joints_world[h36m_idx].tolist()
                joint_centers_3d[pilot_name] = xyz

            # ── PR-7a.2 chirality normalization (upper-body arm chain) ──
            # WHAM emits H36M-ordered joints. Per PR-7a.2 cross-pair
            # diagnostic, WHAM's H36M upper-body (shoulder/elbow/wrist)
            # uses anatomy convention (golfer-anat-left = image-RIGHT for
            # a face-on camera), but lower-body (hip/knee/ankle) AND our
            # ground-truth labels both use image-orientation convention
            # (left = image-LEFT). Without this normalization, downstream
            # fitting tries to encode the convention mismatch as huge
            # body-local offset vectors. Swap the arm chain so all WHAM
            # joints match the GT image-orientation convention.
            for left, right in (
                ("left_shoulder", "right_shoulder"),
                ("left_elbow",    "right_elbow"),
                ("left_wrist",    "right_wrist"),
            ):
                joint_centers_3d[left], joint_centers_3d[right] = (
                    joint_centers_3d[right], joint_centers_3d[left],
                )
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
                "smpl_betas":                track["betas"][i].tolist()
                                              if "betas" in track else None,
                # PR-7a.5: pose + trans saved only when save_smpl_params=True.
                # pose adds ~1.7 KB/frame (24*3*3 floats), trans adds 24 bytes/frame.
                "smpl_pose":  (
                    track["pose"][i].tolist()
                    if save_smpl_params and "pose" in track else None
                ),
                "smpl_trans": (
                    track["trans"][i].tolist()
                    if save_smpl_params and "trans" in track else None
                ),
            })

        result = {
            "video_id":     video_id,
            "runner":       "wham",
            "_wham_runner_version": "7a5",   # PR-7a.5 schema marker
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
                f"j_regressor=J_regressor_h36m.npy (17 joints, H36M order)",
                f"h36m_to_pilot_dropped_keys=[spine2,spine3,left_foot,right_foot]",
            ],
        }

        # ── PR-7a.5: opt-in SMPL params packing (verts to .npz sidecar) ──
        # When save_smpl_params=True, pack (verts, pose, trans, betas) into
        # a compressed .npz blob and ship as base64 string in the result
        # dict. Modal returns it across the wire; local_entrypoint pops +
        # writes to python/pilot/output/wham/<id>/smpl_params.npz.
        #
        # Default OFF — production runs unchanged.
        if save_smpl_params and "verts" in track:
            import base64
            import io
            verts_arr = np.asarray(track["verts"], dtype=np.float32)
            pose_arr  = np.asarray(track["pose"],  dtype=np.float32) if "pose"  in track else None
            trans_arr = np.asarray(track["trans"], dtype=np.float32) if "trans" in track else None
            betas_arr = np.asarray(track["betas"], dtype=np.float32) if "betas" in track else None
            buf = io.BytesIO()
            np.savez_compressed(
                buf, verts=verts_arr,
                **({"pose":  pose_arr}  if pose_arr  is not None else {}),
                **({"trans": trans_arr} if trans_arr is not None else {}),
                **({"betas": betas_arr} if betas_arr is not None else {}),
            )
            raw_npz_bytes = buf.getvalue()
            # Size log BEFORE encoding — if Modal hangs we can tell
            # size-related from inference-related.
            print(f"[wham_runner] smpl_params npz raw size = "
                  f"{len(raw_npz_bytes) / 1024 / 1024:.1f} MB")
            encoded = base64.b64encode(raw_npz_bytes).decode("ascii")
            print(f"[wham_runner] smpl_params b64-encoded size = "
                  f"{len(encoded) / 1024 / 1024:.1f} MB")
            result["smpl_params_npz_b64"] = encoded
            result["smpl_params_shapes"] = {
                "verts": list(verts_arr.shape),
                **({"pose":  list(pose_arr.shape)}  if pose_arr  is not None else {}),
                **({"trans": list(trans_arr.shape)} if trans_arr is not None else {}),
                **({"betas": list(betas_arr.shape)} if betas_arr is not None else {}),
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
        save_smpl_params: bool = False,
    ) -> None:
        """
        Local driver: invokes run_wham on Modal, then writes the result
        JSON to python/pilot/output/wham/<video_id>/joint_centers_3d.json.

        PR-7a.5: when save_smpl_params=True, also pops the SMPL params
        blob from the result dict and writes it as a .npz sidecar at
        python/pilot/output/wham/<video_id>/smpl_params.npz.
        """
        import base64
        import json
        from pathlib import Path

        print(f"[wham_runner] invoking run_wham on Modal for {video_id} "
              f"(save_smpl_params={save_smpl_params})")
        result = run_wham.remote(
            video_id=video_id,
            video_url=video_url,
            primary_checkpoint=primary_checkpoint,
            save_smpl_params=save_smpl_params,
        )
        out_dir = Path(f"python/pilot/output/wham/{video_id}")
        out_dir.mkdir(parents=True, exist_ok=True)

        # PR-7a.5: pop the SMPL params blob BEFORE JSON dump (too large
        # to inline). Sidecar .npz lands next to joint_centers_3d.json.
        npz_b64 = result.pop("smpl_params_npz_b64", None)
        shapes  = result.pop("smpl_params_shapes", None)

        out_path = out_dir / "joint_centers_3d.json"
        out_path.write_text(json.dumps(result, indent=2))
        sz_kb = out_path.stat().st_size / 1024
        print(f"[wham_runner] wrote {out_path} ({sz_kb:.1f} KB)")

        if npz_b64 is not None:
            npz_bytes = base64.b64decode(npz_b64)
            npz_path = out_dir / "smpl_params.npz"
            npz_path.write_bytes(npz_bytes)
            sz_mb = npz_path.stat().st_size / 1024 / 1024
            print(f"[wham_runner] wrote {npz_path} "
                  f"({sz_mb:.1f} MB) shapes={shapes}")
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
