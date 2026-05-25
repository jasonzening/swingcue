"""
wham_runner.py — WHAM bone-center inference on Modal A10G GPU.

Two deployed Modal functions + one local driver share a common pipeline:

  • _run_wham_pipeline()    pipeline-shared helper (PR-8b refactor).
                            Sets up workspace, downloads video, runs WHAM
                            demo.py, parses pkl, regresses H36M joints,
                            applies PR-7a.2 chirality swap at the array
                            level. Returns a rich dict that formatters
                            translate into their preferred schemas.

  • run_wham()              [PR-7a.5 schema, dev CLI]
                            Returns PilotRunResult-shaped dict (joint_centers_3d
                            per frame, _wham_runner_version="7a5").
                            run_wham_local local_entrypoint calls this
                            for Jason's existing dev workflow.

  • infer_video()           [PR-8b schema, deployed for Railway]
                            Returns wham_video_meta + wham_pose_timeline
                            shaped dict matching PR-8a' Supabase schema.
                            Implements 2D back-projection from WHAM SLAM
                            camera. Status/error envelope for Railway
                            error handling (PR-8c will consume).

WHAM commit pin: 2b54f7797391c94876848b905ed875b154c4a295 (2026-05-20).

Cost: ~$0.01 per 7-second clip on A10G. Modal Image first-build ~10-15 min
(DPVO CUDA compile dominates); cached afterward.

Invocations:

  Dev (PR-7a.5 schema):
    ./.venv-pilot/Scripts/python.exe -m modal run \\
        python/pilot/runners/wham_runner.py::run_wham_local \\
        --video-id <uuid> --video-url '<signed-supabase-url>'

  Deployed inference (PR-8b schema, used by Railway in PR-8c):
    modal deploy python/pilot/modal_app.py
    # Then invoke via Modal client:
    #   infer_video.spawn(video_url=..., video_id=...)
"""

from __future__ import annotations

import os
import subprocess
import sys

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

WHAM_COMMIT_SHORT = "2b54f77"


# ---------------------------------------------------------------------------
# Joint mapping — H36M 17-joint index → coaching name.
#
# H36M's joint order is well-documented and stable; WHAM's 31-joint
# J_regressor_wham has an undocumented ordering and produced
# anatomically-impossible mappings on the first try (e.g. "ankles"
# higher than "pelvis"). We trade off spine2/spine3 + feet (H36M
# doesn't have them) for cleaner anatomy. Foot positions can be
# inferred from ankle for overlay.
#
# PR-7a.2 chirality swap (upper-body arm chain) is applied at the
# ARRAY level inside _run_wham_pipeline (was previously applied during
# formatting in run_wham). After swap, idx 11/14, 12/15, 13/16 hold
# image-orientation coordinates — so "left_shoulder" in our output
# refers to image-left, matching ground-truth label convention.
# ---------------------------------------------------------------------------

H36M_TO_PILOT_NAME = {
    0:  "pelvis",
    7:  "spine1",
    9:  "neck",
    10: "head",
    11: "left_shoulder",   14: "right_shoulder",
    12: "left_elbow",      15: "right_elbow",
    13: "left_wrist",      16: "right_wrist",
    4:  "left_hip",        1:  "right_hip",
    5:  "left_knee",       2:  "right_knee",
    6:  "left_ankle",      3:  "right_ankle",
}

# Inverse: pilot name → H36M index. Emitted in PR-8b meta.joint_index_mapping
# so consumers can look up the array position by joint name.
_PILOT_NAME_TO_H36M = {v: k for k, v in H36M_TO_PILOT_NAME.items()}


# ---------------------------------------------------------------------------
# PR-8b.1: SMPL mesh vertex landmarks.
#
# H36M's "head" joint lies in the face/nose region — fine for body
# pose, wrong for golf head-sway metrics that need the cranial top.
# The SMPL mesh has 6890 vertices; vertex 411 was identified as the
# head crown via argmax-y over the mesh in PR-7a4 PROBE
# (docs/PR-7a4_PROBE/smpl_landmark_indices.json).
#
# Vertex 411 is anatomical truth — no scale tuning needed (which would
# be the alternative: extrapolate H36M neck→head by a magic factor).
#
# Scope (PR-8b.1): head_crown only. Other vertex landmarks
# (throat_midpoint=444, c7=414, acromion_left=4721, acromion_right=...)
# stay in the probe file pending a follow-up PR.
# ---------------------------------------------------------------------------

HEAD_CROWN_VERTEX_INDEX = 411


# ---------------------------------------------------------------------------
# Workspace + video setup helpers (unchanged from PR-7a.5).
# ---------------------------------------------------------------------------

def _setup_workspace() -> None:
    """
    Stitch the Volume's weight files into WHAM's expected repo layout
    via symlinks. WHAM's demo expects:
      - /opt/wham/checkpoints/{wham_vit_w_3dpw.pth.tar, ...}
      - /opt/wham/dataset/body_models/{smpl, smplh, smplx}/...
    Volume mounts under /models/{wham, body_models}.
    """
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
    trigger `InvalidURL`. Percent-encode the path component; leave the
    query string untouched (its `+`/`=`/`.` chars are already legal).
    """
    import urllib.parse
    import urllib.request

    parts = urllib.parse.urlsplit(video_url)
    safe_path = urllib.parse.quote(parts.path, safe="/")
    encoded_url = urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, safe_path, parts.query, parts.fragment)
    )

    print(f"[wham_runner] downloading video → {dst_path}")
    urllib.request.urlretrieve(encoded_url, dst_path)
    sz_mb = os.path.getsize(dst_path) / 1024 / 1024
    print(f"[wham_runner]   ({sz_mb:.2f} MB)")


# ===========================================================================
# Modal-side code below — only registered when modal is importable.
# ===========================================================================

if _MODAL_AVAILABLE:
    import modal  # noqa: F401  (referenced in decorators below)

    # -----------------------------------------------------------------------
    # Shared pipeline helper (PR-8b refactor)
    #
    # Runs the WHAM inference end-to-end and returns a rich raw-data dict.
    # Two callers (run_wham, infer_video) format this into their preferred
    # output schemas. Imports done lazily (cv2/numpy/joblib live in
    # wham_image and aren't available outside the Modal container).
    # -----------------------------------------------------------------------

    def _run_wham_pipeline(
        video_id: str,
        video_url: str,
        primary_checkpoint: str = "wham_vit_w_3dpw.pth.tar",
    ) -> dict:
        """
        Run WHAM on one video. Setup + download + demo + pkl parse +
        H36M joint regression + PR-7a.2 chirality swap (array-level).

        Returns:
            {
                "track":              dict,            # raw WHAM track dict
                "wham_out":           dict,            # raw top-level WHAM output
                "joints_3d":          np.ndarray,      # (T, 17, 3) chirality-swapped
                "frame_ids":          np.ndarray | None,
                "video_w":            int,
                "video_h":            int,
                "fps_native":         float,
                "n_native":           int,
                "primary_track_id":   any,
                "primary_checkpoint": str,
                "n_frames":           int,
            }

        Raises RuntimeError on WHAM-side failure (missing models, demo.py
        crash, no .pkl produced, missing 'verts', etc.).
        """
        import cv2
        import joblib
        import numpy as np

        _setup_workspace()

        # Download the swing video into /tmp.
        local_video = f"/tmp/{video_id}.mp4"
        _download_video(video_url, local_video)

        # WHAM exposes a demo.py at repo root. It writes outputs under
        # output/demo/<video_basename>/.
        out_dir = f"{WHAM_REPO_ROOT}/output/demo/{video_id}"
        os.makedirs(out_dir, exist_ok=True)
        demo_cmd = [
            sys.executable,
            f"{WHAM_REPO_ROOT}/demo.py",
            "--video", local_video,
            "--output_pth", f"{out_dir}/wham_output.pth",
            "--save_pkl",
            # --visualize OFF — pytorch3d source build is slow.
        ]
        print(f"[wham_runner] running WHAM demo: {' '.join(demo_cmd)}")
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

        # Locate the pkl — preferred path or fallback tree scan.
        pkl_path = f"{out_dir}/wham_output.pkl"
        if not os.path.exists(pkl_path):
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
                    f"layout differs from expected)"
                )

        wham_out = joblib.load(pkl_path)
        print(f"[wham_runner] wham_output top-level keys: {list(wham_out.keys())}")

        # WHAM's pkl is keyed by track-id. For golf swings, expect 1 person.
        track_ids = sorted(wham_out.keys())
        if not track_ids:
            raise RuntimeError(f"WHAM pkl had no tracks: {pkl_path}")
        primary_track_id = track_ids[0]
        track = wham_out[primary_track_id]
        print(
            f"[wham_runner] using track_id={primary_track_id} of "
            f"{len(track_ids)} total; track keys: {list(track.keys())}"
        )

        # Regress joints from posed vertices via J_regressor_h36m.
        verts = track.get("verts")
        if verts is None:
            raise RuntimeError(
                f"WHAM track[{primary_track_id}] missing 'verts'. "
                f"keys={list(track.keys())}"
            )
        verts = np.asarray(verts)
        if verts.ndim != 3 or verts.shape[-1] != 3:
            raise RuntimeError(
                f"WHAM verts shape unexpected: {verts.shape} "
                f"(expected (T, V, 3))"
            )

        j_regressor_path = "/models/body_models/J_regressor_h36m.npy"
        if not os.path.exists(j_regressor_path):
            raise RuntimeError(
                f"J_regressor_h36m.npy missing at {j_regressor_path} — "
                f"setup_models extras step needs to run first"
            )
        J_regressor = np.load(j_regressor_path)
        # (J, V) canonical; (V, J) needs transpose.
        if J_regressor.shape[1] == verts.shape[1]:
            pass
        elif J_regressor.shape[0] == verts.shape[1]:
            J_regressor = J_regressor.T
        else:
            raise RuntimeError(
                f"J_regressor shape {J_regressor.shape} incompatible with "
                f"verts shape {verts.shape}"
            )
        print(
            f"[wham_runner] computing joints: "
            f"J_regressor {J_regressor.shape} @ verts {verts.shape}"
        )
        joints_3d = np.einsum("jv,tvd->tjd", J_regressor, verts)
        # joints_3d shape: (T, 17, 3) in WHAM's output frame.
        print(f"[wham_runner] derived joints shape: {joints_3d.shape}")

        # ── PR-7a.2 chirality swap at the ARRAY level ────────────────────
        # Per PR-7a.2 cross-pair diagnostic, WHAM's H36M upper-body
        # (shoulder/elbow/wrist) uses anatomy convention (anat-left =
        # image-right for face-on camera), but lower-body AND our GT
        # labels both use image-orientation convention. Swap the upper
        # arm chain indices so all joints follow GT image-orientation.
        # Doing the swap at the array level (vs at the formatter level
        # as PR-7a.5 did) means downstream code reading
        # joints_3d[H36M_TO_PILOT_NAME^-1["left_shoulder"]] gets image-
        # left coords directly.
        arm_swap_pairs = [(11, 14), (12, 15), (13, 16)]
        for left_idx, right_idx in arm_swap_pairs:
            joints_3d[:, [left_idx, right_idx], :] = (
                joints_3d[:, [right_idx, left_idx], :]
            )

        # Debug dump frame[0] all joint xyz so the joint name ordering
        # can be reverse-engineered if anatomy looks off. Print sorted
        # by y (typically vertical) — lowest y = pelvis/ankles, highest
        # = head.
        debug_frame_idx = 0
        f0_joints = joints_3d[debug_frame_idx]
        idx_sorted = sorted(range(len(f0_joints)), key=lambda i: f0_joints[i][1])
        print(f"[wham_runner] DEBUG frame[{debug_frame_idx}] all joints (sorted by y, post-chirality-swap):")
        for i in idx_sorted:
            x, y, z = f0_joints[i]
            name = H36M_TO_PILOT_NAME.get(i, f"(idx {i})")
            print(f"[wham_runner]   idx={i:2d} {name:>15}  ({x:+7.3f}, {y:+7.3f}, {z:+7.3f})")

        frame_ids = track.get("frame_ids")
        n_frames = len(joints_3d)

        # Read video metadata locally.
        cap = cv2.VideoCapture(local_video)
        fps_native = cap.get(cv2.CAP_PROP_FPS) or 30.0
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_native = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # PR-8b.1: extract head_crown 3D directly from SMPL mesh
        # vertex 411 (per PR-7a4 PROBE). verts are in the same camera-
        # frame meters as joints_3d, so the same projection intrinsics
        # apply. Per-frame Z<0.1 / NaN guards live in the formatter so
        # the rest of the body still emits coordinates even if the
        # crown vertex is degenerate for an isolated frame.
        head_crown_3d = verts[:, HEAD_CROWN_VERTEX_INDEX, :].astype(np.float32)
        print(
            f"[wham_runner] head_crown_3d shape={head_crown_3d.shape} "
            f"vertex_idx={HEAD_CROWN_VERTEX_INDEX} "
            f"(frame[0] xyz = {head_crown_3d[0].tolist()})"
        )

        return {
            "track":              track,
            "wham_out":           wham_out,
            "joints_3d":          joints_3d,
            "head_crown_3d":      head_crown_3d,
            "frame_ids":          frame_ids,
            "video_w":            video_w,
            "video_h":            video_h,
            "fps_native":         float(fps_native),
            "n_native":           n_native,
            "primary_track_id":   primary_track_id,
            "primary_checkpoint": primary_checkpoint,
            "n_frames":           n_frames,
        }


    # -----------------------------------------------------------------------
    # run_wham — PR-7a.5 schema, kept for dev CLI workflow.
    # -----------------------------------------------------------------------

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
        save_smpl_params: bool = False,
    ) -> dict:
        """
        Run WHAM on one video and return the PR-7a.5 PilotRunResult dict.

        run_wham_local local_entrypoint calls this remotely. Behavior
        unchanged from PR-7a.5 (modulo the chirality swap moving from
        the formatter to _run_wham_pipeline — net frame-level output
        is identical).
        """
        import base64
        import io
        import numpy as np

        pipe = _run_wham_pipeline(
            video_id=video_id,
            video_url=video_url,
            primary_checkpoint=primary_checkpoint,
        )

        track            = pipe["track"]
        joints_3d        = pipe["joints_3d"]
        frame_ids        = pipe["frame_ids"]
        fps_native       = pipe["fps_native"]
        video_w          = pipe["video_w"]
        video_h          = pipe["video_h"]
        n_native         = pipe["n_native"]
        n_frames         = pipe["n_frames"]

        frames_out: list[dict] = []
        for i in range(n_frames):
            joints_world = joints_3d[i]
            joint_centers_3d: dict[str, list[float]] = {}
            for h36m_idx, pilot_name in H36M_TO_PILOT_NAME.items():
                joint_centers_3d[pilot_name] = joints_world[h36m_idx].tolist()

            fi = int(frame_ids[i]) if frame_ids is not None else i
            frames_out.append({
                "ts":                       round(fi / fps_native, 3),
                "frame_idx":                fi,
                "joint_centers_3d":         joint_centers_3d,
                # 2D projection deferred to infer_video (PR-8b).
                "joint_centers_2d_projected": None,
                "smpl_betas":                track["betas"][i].tolist()
                                              if "betas" in track else None,
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
            "_wham_runner_version": "7a5",
            "video_width":  video_w,
            "video_height": video_h,
            "fps_native":   round(fps_native, 2),
            "fps_sampled":  round(fps_native, 2),
            "duration_sec": round(n_native / fps_native, 3),
            "frames":       frames_out,
            "camera": {
                # Camera extraction is PR-8b's job (infer_video). Old
                # PR-7a.5 schema kept None for back-compat.
                "rotation":    None,
                "translation": None,
                "focal_px":    None,
            },
            "notes": [
                f"wham_commit={WHAM_COMMIT_SHORT}",
                f"primary_checkpoint={pipe['primary_checkpoint']}",
                f"n_frames_wham={n_frames}",
                f"n_native_frames={n_native}",
                f"video_url={video_url[:80]}{'...' if len(video_url) > 80 else ''}",
                f"j_regressor=J_regressor_h36m.npy (17 joints, H36M order)",
                f"h36m_to_pilot_dropped_keys=[spine2,spine3,left_foot,right_foot]",
                f"chirality_swap=array_level (PR-8b refactor)",
            ],
        }

        # PR-7a.5: opt-in SMPL params packing to .npz sidecar.
        if save_smpl_params and "verts" in track:
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


    # -----------------------------------------------------------------------
    # infer_video — PR-8b schema, deployed for Railway consumption.
    # -----------------------------------------------------------------------

    @app.function(
        image=wham_image,
        gpu="A10G",
        volumes={"/models": model_volume},
        timeout=600,
    )
    def infer_video(
        video_url: str,
        video_id: str,
        save_smpl_params: bool = False,
    ) -> dict:
        """
        PR-8b: deployed WHAM inference returning the PR-8a' schema
        (wham_video_meta + wham_pose_timeline shapes).

        Implements 2D back-projection from WHAM SLAM camera output
        (was None/TODO in PR-7a.5). Camera extraction is defensive —
        runtime-inspects WHAM track + top-level dict keys, falls back
        to `focal = max(w, h)` and identity extrinsics if WHAM didn't
        expose the expected fields. The actual key names discovered
        are recorded in `meta.camera.notes` so Spec 2 can document
        what WHAM produces in this build.

        NO Supabase writes — PR-8c (Railway) wires DB ingestion.
        """
        import time
        import numpy as np

        start_ms = int(time.time() * 1000)

        try:
            pipe = _run_wham_pipeline(
                video_id=video_id,
                video_url=video_url,
            )
        except Exception as exc:
            # Catastrophic failure — return failed envelope. Caller
            # (Railway in PR-8c) parses status + error_message for
            # retry/give-up logic.
            elapsed_ms = int(time.time() * 1000) - start_ms
            print(f"[wham_runner.infer_video] FAILED after {elapsed_ms}ms: {exc}")
            return {
                "status":              "failed",
                "video_id":            video_id,
                "modal_call_id":       None,
                "inference_ms_total":  elapsed_ms,
                "error_message":       str(exc),
                "meta":                _empty_pr8b_meta(),
                "frames":              [],
            }

        # Pipeline succeeded — extract camera + format frames.
        cam = _extract_camera(
            wham_out=pipe["wham_out"],
            track=pipe["track"],
            image_w=pipe["video_w"],
            image_h=pipe["video_h"],
            n_frames=pipe["n_frames"],
        )

        meta = _build_pr8b_meta(pipe, cam)
        frames_out, n_partial = _build_pr8b_frames(pipe, cam, save_smpl_params)

        status = "completed" if n_partial == 0 else "partial"
        elapsed_ms = int(time.time() * 1000) - start_ms
        print(
            f"[wham_runner.infer_video] {status} — "
            f"{pipe['n_frames']} frames ({n_partial} partial), "
            f"{elapsed_ms}ms total"
        )

        return {
            "status":              status,
            "video_id":            video_id,
            "modal_call_id":       None,  # populated by Railway (PR-8c)
            "inference_ms_total":  elapsed_ms,
            "error_message":       None,
            "meta":                meta,
            "frames":              frames_out,
        }


    # -----------------------------------------------------------------------
    # PR-8b helpers — camera extraction, 2D projection, schema builders.
    # -----------------------------------------------------------------------

    def _extract_camera(
        wham_out: dict,
        track: dict,
        image_w: int,
        image_h: int,
        n_frames: int,
    ) -> dict:
        """
        Defensively pull camera intrinsics + extrinsics from the WHAM
        output. Returns a dict with R_per_frame / t_per_frame / focal /
        cx / cy / notes. R_per_frame and t_per_frame may be None if
        WHAM didn't expose them; projection falls back to identity in
        that case (joints assumed already in camera frame).

        WHAM's actual output key names are undocumented for this commit
        pin; we try a range of common patterns and log the result so
        future PRs can lock to the verified key names.
        """
        import numpy as np

        notes: list[str] = []
        notes.append(f"wham_out_top_keys={list(wham_out.keys())[:12]}")
        notes.append(f"track_keys={list(track.keys())[:24]}")

        R_pf: "np.ndarray | None" = None
        t_pf: "np.ndarray | None" = None
        focal: "float | None" = None

        # Try separate R/t key pairs at track level.
        for r_key, t_key in (
            ("cam_R", "cam_t"),
            ("cam_rotation", "cam_translation"),
            ("R", "trans_cam"),
            ("world_cam_R", "world_cam_t"),
        ):
            if r_key in track and t_key in track:
                R_pf = np.asarray(track[r_key], dtype=np.float32)
                t_pf = np.asarray(track[t_key], dtype=np.float32)
                notes.append(f"extrinsics_source=track[{r_key},{t_key}] R={R_pf.shape} t={t_pf.shape}")
                break

        # Try combined 4x4 / 3x4 cam matrix at track level.
        if R_pf is None and "cam" in track:
            cam = np.asarray(track["cam"], dtype=np.float32)
            if cam.ndim == 3 and cam.shape[1:] == (4, 4):
                R_pf = cam[:, :3, :3]
                t_pf = cam[:, :3, 3]
                notes.append(f"extrinsics_source=track[cam] (T,4,4) {cam.shape}")
            elif cam.ndim == 3 and cam.shape[1:] == (3, 4):
                R_pf = cam[:, :3, :3]
                t_pf = cam[:, :3, 3]
                notes.append(f"extrinsics_source=track[cam] (T,3,4) {cam.shape}")
            elif cam.ndim == 2 and cam.shape == (4, 4):
                # Single global pose; broadcast to per-frame.
                R_pf = np.broadcast_to(cam[None, :3, :3], (n_frames, 3, 3)).copy()
                t_pf = np.broadcast_to(cam[None, :3, 3], (n_frames, 3)).copy()
                notes.append(f"extrinsics_source=track[cam] global (4,4)")

        # Try top-level wham_out for camera (some WHAM forks emit there).
        if R_pf is None:
            for r_key, t_key in (("cam_R", "cam_t"), ("global_R", "global_t")):
                if r_key in wham_out and t_key in wham_out:
                    R_pf = np.asarray(wham_out[r_key], dtype=np.float32)
                    t_pf = np.asarray(wham_out[t_key], dtype=np.float32)
                    notes.append(f"extrinsics_source=wham_out[{r_key},{t_key}]")
                    break

        # Focal length — try a few common name patterns.
        for key in ("focal", "focal_length", "focal_px", "fx"):
            if key in track:
                val = track[key]
                arr = np.asarray(val).flatten()
                if arr.size > 0:
                    focal = float(arr[0])
                    notes.append(f"focal_source=track[{key}]={focal:.2f}")
                    break
        if focal is None and "K" in track:
            K = np.asarray(track["K"], dtype=np.float32)
            if K.ndim == 2 and K.shape == (3, 3):
                focal = float(K[0, 0])
                notes.append(f"focal_source=track[K][0,0]={focal:.2f}")
            elif K.ndim == 3 and K.shape[1:] == (3, 3):
                focal = float(K[0, 0, 0])
                notes.append(f"focal_source=track[K][0,0,0]={focal:.2f}")
        if focal is None:
            # PR-8b.2 calibration fix: WHAM's canonical projection
            # (lib/vis/run_vis.py line 22) uses the CLIFF formula
            # focal = sqrt(width² + height²). Previous max(w,h) fallback
            # underestimated by ~13% for portrait videos — caused
            # ~10-20px drift through the swing visible only in 120-frame
            # MP4 review (static frames at setup hid the issue because
            # the person was centered at fixed depth).
            #
            # Reference: tmp/wham_src/run_vis.py + renderer.py — WHAM's
            # own visualizer uses this focal + cx=w/2 + cy=h/2 + identity
            # extrinsics, matching what we already use modulo this focal.
            focal = float((image_w ** 2 + image_h ** 2) ** 0.5)
            notes.append(
                f"focal_fallback=CLIFF=sqrt(w^2+h^2)={focal:.1f} "
                f"(WHAM upstream canonical: run_vis.py L22; "
                f"renderer.py K[0,0]=K[1,1]=focal_length)"
            )

        if R_pf is None:
            notes.append(
                "extrinsics_unavailable=identity_fallback "
                "(joints projected as if already in camera frame)"
            )

        cx = image_w / 2.0
        cy = image_h / 2.0
        notes.append(f"principal_point=image_center=({cx:.0f},{cy:.0f})")

        return {
            "R_per_frame": R_pf,
            "t_per_frame": t_pf,
            "focal":       focal,
            "cx":          cx,
            "cy":          cy,
            "notes":       "; ".join(notes),
        }

    def _project_joints_2d(
        joints_world,  # (J, 3) np.ndarray, J=17 H36M
        R,             # (3, 3) np.ndarray or None
        t,             # (3,)   np.ndarray or None
        focal: float,
        cx: float,
        cy: float,
    ):
        """
        Pinhole projection. Returns (J, 2) np.ndarray.

        If R/t is None, treat joints_world as already in camera frame
        (identity fallback per spec). Z near zero is guarded against
        with a small epsilon — joints behind the camera produce huge
        out-of-bounds u/v which the fit_ok check later flags.
        """
        import numpy as np
        joints = np.asarray(joints_world, dtype=np.float32)
        if R is not None and t is not None:
            R = np.asarray(R, dtype=np.float32)
            t = np.asarray(t, dtype=np.float32)
            joints_cam = joints @ R.T + t  # (J, 3)
        else:
            joints_cam = joints
        X = joints_cam[:, 0]
        Y = joints_cam[:, 1]
        Z = joints_cam[:, 2]
        eps = 1e-6
        Z_safe = np.where(np.abs(Z) < eps, eps * np.sign(Z + eps), Z)
        u = focal * X / Z_safe + cx
        v = focal * Y / Z_safe + cy
        return np.stack([u, v], axis=-1)  # (J, 2)

    def _empty_pr8b_meta() -> dict:
        """Meta block for `failed` envelope — fields populated to satisfy
        Zod schema in PR-8d but values are minimal/null."""
        return {
            "source":                "wham_smplh_v1",
            "joint_type":            "bone_center",
            "coordinate_space_2d":   "video_px",
            "coordinate_space_3d":   "smpl_world_m",
            "wham_model":            "wham_vit_w_3dpw",
            "wham_commit":           WHAM_COMMIT_SHORT,
            "image_width":           0,
            "image_height":          0,
            "processed_fps":         0.0,
            "frame_count":           0,
            "camera": {
                "rotation":            None,
                "translation":         None,
                "focal_px":            None,
                "principal_point_px":  None,
                "notes":               "pipeline failed before camera extraction",
            },
            "joint_index_mapping":   _build_joint_index_mapping(),
        }

    def _build_joint_index_mapping() -> dict:
        """Emitted into meta.joint_index_mapping — single source of
        truth for downstream consumers. Includes the chirality-swap
        flag so consumers understand the names follow image-orientation
        convention (PR-7a.2). PR-8b.1: adds the head_crown vertex
        landmark so frontends know head_crown is mesh-derived (not
        H36M-derived)."""
        m: dict = dict(_PILOT_NAME_TO_H36M)
        m["_source"] = "h36m_17"
        m["_chirality_normalized"] = (
            "upper_body_arm_chain_swapped (PR-7a.2): "
            "indices 11/14, 12/15, 13/16 swapped at array level so "
            "name 'left_*' refers to image-left after a face-on camera, "
            "matching ground-truth label convention."
        )
        # PR-8b.1: head_crown is NOT an H36M joint — it's a direct SMPL
        # mesh vertex. The existing 'head' field (H36M idx 10) sits in
        # the face/nose region; head_crown sits at the cranial top.
        # Use head_crown for golf head-sway metrics; head is retained
        # for evidence + backward compat only.
        m["head_crown_vertex_index"] = HEAD_CROWN_VERTEX_INDEX
        m["_landmark_source_for_head_crown"] = "smpl_mesh_vertex"
        m["_notes"] = (
            f"head_crown derived from SMPL mesh vertex "
            f"{HEAD_CROWN_VERTEX_INDEX} (per PR-7a4 PROBE: "
            f"docs/PR-7a4_PROBE/smpl_landmark_indices.json). head_crown "
            f"is distinct from the H36M `head` joint which lies in the "
            f"face/nose region. Use head_crown for golf head-sway "
            f"metrics; H36M head is retained for evidence and backward "
            f"compat only."
        )
        return m

    def _build_pr8b_meta(pipe: dict, cam: dict) -> dict:
        """Build the wham_video_meta-shaped meta dict from pipeline +
        camera output. Camera rotation/translation report the frame-0
        snapshot; per-frame variation is consumed by projection inside
        the formatter."""
        R_pf = cam["R_per_frame"]
        t_pf = cam["t_per_frame"]
        cam_rotation = None
        cam_translation = None
        if R_pf is not None and len(R_pf) > 0:
            cam_rotation = R_pf[0].tolist()
        if t_pf is not None and len(t_pf) > 0:
            cam_translation = t_pf[0].tolist()

        notes_full = cam["notes"]
        if R_pf is not None and len(R_pf) > 1:
            notes_full += (
                f"; meta.camera.rotation/translation = frame-0 snapshot, "
                f"per-frame variation applied during 2D projection "
                f"(R={R_pf.shape}, t={t_pf.shape if t_pf is not None else None})"
            )

        return {
            "source":                "wham_smplh_v1",
            "joint_type":            "bone_center",
            "coordinate_space_2d":   "video_px",
            "coordinate_space_3d":   "smpl_world_m",
            "wham_model":            "wham_vit_w_3dpw",
            "wham_commit":           WHAM_COMMIT_SHORT,
            "image_width":           pipe["video_w"],
            "image_height":          pipe["video_h"],
            "processed_fps":         round(pipe["fps_native"], 2),
            "frame_count":           pipe["n_frames"],
            "camera": {
                "rotation":            cam_rotation,
                "translation":         cam_translation,
                "focal_px":            cam["focal"],
                "principal_point_px":  [cam["cx"], cam["cy"]],
                "notes":               notes_full,
            },
            "joint_index_mapping":   _build_joint_index_mapping(),
        }

    def _build_pr8b_frames(pipe: dict, cam: dict, save_smpl_params: bool):
        """
        Build the per-frame list for wham_pose_timeline shape. Applies
        per-frame 2D projection from cam.R_per_frame[i] / t_per_frame[i]
        (or identity fallback). Computes fit_ok per spec:
          - rule 1: any 3D joint non-finite → fit_ok=False, sub-dicts null
          - rule 2: >50% of joints' projected (u,v) out-of-image-bounds
                    → fit_ok=False, sub-dicts still populated

        fit_quality: per spec, computed as
            1.0 - clip(mean_reprojection_error_px / 50, 0, 1)
        Requires WHAM's 2D detections (kp2d). If not exposed in the
        track dict, set to None and explained in code comment.

        Returns (frames_out_list, n_partial).
        """
        import math
        import numpy as np

        joints_3d_all = pipe["joints_3d"]   # (T, 17, 3) chirality-swapped
        head_crown_3d_all = pipe.get("head_crown_3d")  # (T, 3) or None
        frame_ids     = pipe["frame_ids"]
        fps_native    = pipe["fps_native"]
        track         = pipe["track"]
        image_w       = pipe["video_w"]
        image_h       = pipe["video_h"]
        n_frames      = pipe["n_frames"]

        R_pf  = cam["R_per_frame"]
        t_pf  = cam["t_per_frame"]
        focal = cam["focal"]
        cx    = cam["cx"]
        cy    = cam["cy"]

        # kp2d availability check — if WHAM didn't expose the 2D ground-
        # truth (YOLO/ViTPose) detections, we can't compute reprojection
        # error and fit_quality stays None for all frames. Inspect once
        # up front.
        kp2d_per_frame = None
        for key in ("kp2d", "keypoints_2d", "vitpose_kp2d", "input_kp2d"):
            if key in track:
                kp2d_per_frame = np.asarray(track[key])
                print(f"[wham_runner.infer_video] kp2d source=track[{key}] shape={kp2d_per_frame.shape}")
                break
        if kp2d_per_frame is None:
            print(
                "[wham_runner.infer_video] no 2D ground-truth keypoints "
                "(kp2d/keypoints_2d/vitpose_kp2d) exposed by WHAM → "
                "fit_quality=null for all frames"
            )

        frames_out: list[dict] = []
        n_partial = 0
        # Iterate the 16 named joints; map to H36M array index.
        named_indices = [
            (name, idx) for name, idx in _PILOT_NAME_TO_H36M.items()
        ]
        n_named = len(named_indices)

        for i in range(n_frames):
            joints_world = joints_3d_all[i]   # (17, 3)
            fi = int(frame_ids[i]) if frame_ids is not None else i
            ts_ms = int(round((fi / fps_native) * 1000))

            # Rule 1: any non-finite 3D joint → hard skip.
            if not np.isfinite(joints_world).all():
                n_partial += 1
                frames_out.append({
                    "frame_idx":               fi,
                    "frame_timestamp_ms":      ts_ms,
                    "fit_ok":                  False,
                    "fit_quality":             None,
                    "smpl_pose":               None,
                    "smpl_shape":              None,
                    "smpl_trans":              None,
                    "keypoints_2d_projected":  None,
                    "keypoints_3d_smpl":       None,
                })
                continue

            # Project ALL 17 joints (we only emit the 16 named in the
            # sub-dict but compute on the full array for vector ops).
            R_i = R_pf[i] if R_pf is not None and i < len(R_pf) else None
            t_i = t_pf[i] if t_pf is not None and i < len(t_pf) else None
            kp2d_proj = _project_joints_2d(
                joints_world, R_i, t_i, focal, cx, cy,
            )  # (17, 2)

            keypoints_2d: dict = {}
            keypoints_3d: dict = {}
            n_oob = 0
            for name, idx in named_indices:
                u = float(kp2d_proj[idx, 0])
                v = float(kp2d_proj[idx, 1])
                x = float(joints_world[idx, 0])
                y = float(joints_world[idx, 1])
                z = float(joints_world[idx, 2])
                keypoints_2d[name] = {"x": u, "y": v}
                keypoints_3d[name] = {"x": x, "y": y, "z": z}
                # Out-of-bounds with ±10% slack per spec.
                slack_x = 0.10 * image_w
                slack_y = 0.10 * image_h
                if not (
                    -slack_x <= u <= image_w + slack_x
                    and -slack_y <= v <= image_h + slack_y
                ):
                    n_oob += 1

            # PR-8b.1: head_crown — SMPL vertex 411 (cranial top), in
            # the same camera-frame meters as the H36M joints. Project
            # with same intrinsics. Per-frame guard: Z<0.1m or any NaN
            # → emit head_crown=None for THIS frame only (does NOT set
            # fit_ok=False — body joints may still be valid).
            if head_crown_3d_all is not None:
                hc3d = head_crown_3d_all[i]   # (3,)
                hc_x, hc_y, hc_z = float(hc3d[0]), float(hc3d[1]), float(hc3d[2])
                hc_valid = (
                    np.isfinite(hc3d).all()
                    and hc_z >= 0.1
                )
                if hc_valid:
                    # Project — same R/t/focal/cx/cy as torso joints.
                    hc_proj = _project_joints_2d(
                        hc3d[None, :],  # shape (1, 3) for vectorized helper
                        R_i, t_i, focal, cx, cy,
                    )
                    hc_u = float(hc_proj[0, 0])
                    hc_v = float(hc_proj[0, 1])
                    keypoints_2d["head_crown"] = {"x": hc_u, "y": hc_v}
                    keypoints_3d["head_crown"] = {"x": hc_x, "y": hc_y, "z": hc_z}
                else:
                    keypoints_2d["head_crown"] = None
                    keypoints_3d["head_crown"] = None
            else:
                # Pipeline didn't return head_crown_3d (older path?) —
                # emit None so the schema field exists.
                keypoints_2d["head_crown"] = None
                keypoints_3d["head_crown"] = None

            # Rule 2: >50% of joints OOB → flag low-quality but still emit.
            fit_ok = (n_oob <= n_named // 2)
            if not fit_ok:
                n_partial += 1

            # fit_quality from reprojection error — only computable if
            # WHAM exposed its 2D detection input. Else None.
            fit_quality = None
            if kp2d_per_frame is not None and i < len(kp2d_per_frame):
                gt2d = kp2d_per_frame[i]  # expected (K, 2) or (K, 3) with conf
                if gt2d.ndim == 2 and gt2d.shape[0] >= n_named:
                    # Match by H36M index if possible; otherwise fall back
                    # to first 17 entries. WHAM's kp2d convention isn't
                    # documented at this commit pin — defensive.
                    try:
                        diffs = kp2d_proj[:n_named, :2] - gt2d[:n_named, :2]
                        per_joint_err = np.linalg.norm(diffs, axis=1)
                        mean_err = float(np.mean(per_joint_err))
                        # Saturate at 50px (per spec).
                        normalized = max(0.0, min(1.0, mean_err / 50.0))
                        fit_quality = 1.0 - normalized
                    except Exception:
                        fit_quality = None

            # SMPL params: shape always emit (small), pose/trans opt-in.
            smpl_shape_i = (
                track["betas"][i].tolist()
                if "betas" in track and i < len(track["betas"])
                else None
            )
            smpl_pose_i = (
                track["pose"][i].tolist()
                if save_smpl_params and "pose" in track and i < len(track["pose"])
                else None
            )
            smpl_trans_i = (
                track["trans"][i].tolist()
                if save_smpl_params and "trans" in track and i < len(track["trans"])
                else None
            )

            frames_out.append({
                "frame_idx":               fi,
                "frame_timestamp_ms":      ts_ms,
                "fit_ok":                  fit_ok,
                "fit_quality":             fit_quality,
                "smpl_pose":               smpl_pose_i,
                "smpl_shape":              smpl_shape_i,
                "smpl_trans":              smpl_trans_i,
                "keypoints_2d_projected":  keypoints_2d,
                "keypoints_3d_smpl":       keypoints_3d,
            })

        # Avoid math being flagged unused — import only above for clarity.
        _ = math
        return frames_out, n_partial


# ---------------------------------------------------------------------------
# Local entry — Modal-side run + local result-fetch + overlay render.
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
        Local driver: invokes run_wham on Modal, then writes the PR-7a.5
        result JSON to python/pilot/output/wham/<video_id>/joint_centers_3d.json.

        PR-8b note: this entrypoint still calls `run_wham` (PR-7a.5
        schema). The new PR-8b function is `infer_video` — invoke via
        the deployed Modal function once `modal deploy` has landed.
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


# ---------------------------------------------------------------------------
# Local self-test — describes the deployable surface.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _MODAL_AVAILABLE:
        print("[wham_runner] modal not installed")
    else:
        print(f"[wham_runner] app          = {app!r}")
        print(f"[wham_runner] image        = wham_image")
        print(f"[wham_runner] gpu          = A10G")
        print(f"[wham_runner] timeout_sec  = 600")
        print(f"[wham_runner] functions    = run_wham (PR-7a.5), infer_video (PR-8b)")
        print(
            "[wham_runner] dev CLI (PR-7a.5 schema):\n"
            "    modal run python/pilot/runners/wham_runner.py::run_wham_local "
            "--video-id <uuid> --video-url '<https URL>'"
        )
        print(
            "[wham_runner] deploy (PR-8b infer_video):\n"
            "    modal deploy python/pilot/modal_app.py\n"
            "Then invoke via Modal client (PR-8c Railway driver)."
        )
