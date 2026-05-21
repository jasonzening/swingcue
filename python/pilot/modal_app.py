"""
modal_app.py — Modal infrastructure for the Phase 2 bone-center pilot.

phase2a + phase2b deliverable. Defines:
  - Modal App stub (name: "swingcue-pilot")
  - Persistent Volume "swingcue-pilot-models" (model weight cache)
  - WHAM Image (PyTorch 1.11 + CUDA 11.3 + Python 3.9 — WHAM's tested stack)

phase2c additions later: human3r_image, smplest_x_image, etc. Each
library gets its own Image to keep PyTorch/CUDA/pytorch3d trees
isolated (CC review §2, no shared image).

Bootstrap order (Jason runs locally, one-time):

    # 1. Fresh venv + Modal client
    python3.11 -m venv .venv-pilot
    .\.venv-pilot\Scripts\Activate.ps1
    pip install -r python/pilot/requirements_pilot.txt

    # 2. Auth
    modal token new

    # 3. SMPL family weights — Jason downloads locally via the SMPL
    #    research-license site, then `modal volume put` directly into
    #    the persisted Volume (license-clean; setup_models.py does NOT
    #    scrape smpl.is.tue.mpg.de).
    #    Files expected at /models/body_models/ inside the Volume:
    #      - smpl/SMPL_NEUTRAL.pkl (from smpl.is.tue.mpg.de)
    #      - smpl/SMPL_MALE.pkl
    #      - smpl/SMPL_FEMALE.pkl
    #      - smplx/SMPLX_NEUTRAL.npz (from smpl-x.is.tue.mpg.de)
    #      - smplh/SMPLH_NEUTRAL.npz (from mano.is.tue.mpg.de)
    #    Once downloaded locally + unzipped:
    #      modal volume put swingcue-pilot-models \
    #        ./local-body-models /models/body_models

    # 4. CC drives the rest (volume create, setup_models, wham_runner).

WHAM commit pin: 2b54f7797391c94876848b905ed875b154c4a295
Captured via `git ls-remote https://github.com/yohanshin/WHAM HEAD` on
2026-05-20.
"""

from __future__ import annotations

try:
    import modal
    _MODAL_AVAILABLE = True
except ImportError:  # pragma: no cover — Modal not installed locally
    modal = None  # type: ignore
    _MODAL_AVAILABLE = False


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

APP_NAME = "swingcue-pilot"

if _MODAL_AVAILABLE:
    app = modal.App(APP_NAME)
else:
    app = None  # type: ignore


# ---------------------------------------------------------------------------
# Volume — single persistent volume holds all model weights.
# Layout (populated by setup_models.py + Jason's `modal volume put`):
#   /models/wham/                  WHAM checkpoint family (6 files,
#                                  setup_models downloads via gdown)
#       wham_vit_w_3dpw.pth.tar    primary WHAM checkpoint
#       wham_vit_bedlam_w_3dpw.pth.tar  alt (bedlam-trained)
#       hmr2a.ckpt                 HMR2A subnet
#       dpvo.pth                   DPVO SLAM subnet
#       yolov8x.pt                 YOLOv8 person detector
#       vitpose-h-multi-coco.pth   ViTPose-H 2D keypoint subnet
#   /models/body_models/           SMPL family — Jason uploads locally:
#       smpl/SMPL_*.pkl            SMPL Neutral/Male/Female
#       smplx/SMPLX_NEUTRAL.npz    SMPL-X
#       smplh/SMPLH_NEUTRAL.npz    SMPL-H (WHAM specifically needs SMPL-H)
# ---------------------------------------------------------------------------

VOLUME_NAME = "swingcue-pilot-models"

if _MODAL_AVAILABLE:
    model_volume = modal.Volume.from_name(
        VOLUME_NAME, create_if_missing=True,
    )
else:
    model_volume = None  # type: ignore


# ---------------------------------------------------------------------------
# WHAM commit pin (captured 2026-05-20).
# Phase2b smoke results are reproducible from this SHA. Re-pin in
# phase2c if upstream changes affect benchmark numbers.
# ---------------------------------------------------------------------------

WHAM_REPO = "https://github.com/yohanshin/WHAM.git"
WHAM_COMMIT = "2b54f7797391c94876848b905ed875b154c4a295"


# ---------------------------------------------------------------------------
# WHAM Image — phase2b deliverable.
#
# Base: PyTorch 1.11.0 + CUDA 11.3 + cuDNN 8 (devel variant — needed
# for DPVO's CUDA-extension compile step at install time). Python 3.9
# is what comes in pytorch/pytorch:1.11.0 official images.
#
# Why this stack: WHAM's official INSTALL.md specifies exactly
# pytorch==1.11.0 + cudatoolkit=11.3 + python=3.9. DPVO's CUDA kernels
# are written against this toolkit. Newer torch/cu118 likely fails
# at DPVO compile time.
#
# CUDA 11.3 user-space + CUDA 12+ driver on Modal A10G: forward-
# compatible by NVIDIA's driver model.
#
# Build time on Modal first run: ~10-15 min (DPVO compile dominates).
# Cached afterward; rebuild only when this file changes.
# ---------------------------------------------------------------------------

if _MODAL_AVAILABLE:
    wham_image = (
        modal.Image.from_registry(
            "pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel",
            add_python="3.9",
        )
        .apt_install(
            "git",
            "ffmpeg",
            "libgl1",
            "libglib2.0-0",
            "libsm6",
            "libxext6",
            "libxrender1",
            "build-essential",   # DPVO CUDA-ext compile
            "ninja-build",       # PyTorch CUDA-ext build accel
        )
        # WHAM runtime deps. smplx loads SMPL/SMPL-H/SMPL-X files from
        # /models/body_models at inference time. gdown is needed inside
        # the Image so setup_models can download from Google Drive.
        .pip_install(
            "smplx==0.1.28",
            "joblib==1.3.2",
            "yacs==0.1.8",
            "pyyaml>=6.0",
            "scipy==1.10.1",
            "opencv-python-headless==4.8.1.78",
            "ffmpeg-python==0.2.0",
            "numpy<2.0",
            "tqdm",
            "loguru",
            "gdown==5.2.0",      # phase2b weight download
            # Vision / pose deps WHAM imports transitively:
            "chumpy==0.70",      # legacy SMPL dep, often required
            "torchgeometry==0.1.2",
            "einops==0.7.0",
            "matplotlib==3.7.4",
            "Pillow==10.2.0",
            # ViT pose head submodule + HMR2A:
            "timm==0.9.10",
        )
        # Clone WHAM at the pinned commit + init DPVO submodule.
        # DPVO CUDA extensions compile at first import; the build deps
        # above (build-essential + ninja + CUDA 11.3 devel headers) are
        # what they need.
        .run_commands(
            "git clone https://github.com/yohanshin/WHAM.git /opt/wham",
            f"cd /opt/wham && git checkout {WHAM_COMMIT}",
            "cd /opt/wham && git submodule update --init --recursive",
        )
        .env({
            "PYTHONPATH": "/opt/wham:$PYTHONPATH",
            # WHAM expects checkpoints/ in CWD; we symlink to Volume
            # at function entry (see runners/wham_runner.py setup_workspace).
            "WHAM_REPO": "/opt/wham",
        })
        # Modal 1.0 future-compat: explicit local source declarations
        # replace implicit auto-mounting. Both modal_app and setup_models
        # are imported inside the Image at runtime (e.g. setup_all_models
        # references constants from modal_app), so list them here.
        .add_local_python_source("modal_app")
        .add_local_python_source("setup_models")
    )
else:
    wham_image = None  # type: ignore


# ---------------------------------------------------------------------------
# Image registry — phase2c expands this with one entry per library.
# ---------------------------------------------------------------------------

IMAGES = {
    "wham": wham_image,
}


# ---------------------------------------------------------------------------
# Self-test — runs when this file is executed directly. NOT a Modal
# deploy; just verifies the scaffold definitions are syntactically + API
# valid against whatever Modal version is installed locally.
# ---------------------------------------------------------------------------

def _self_test() -> None:
    if not _MODAL_AVAILABLE:
        print(
            "[modal_app] modal package not installed — "
            "pip install -r python/pilot/requirements_pilot.txt"
        )
        return
    print(f"[modal_app] APP_NAME      = {APP_NAME}")
    print(f"[modal_app] VOLUME_NAME   = {VOLUME_NAME}")
    print(f"[modal_app] WHAM_COMMIT   = {WHAM_COMMIT}")
    print(f"[modal_app] images        = {list(IMAGES.keys())}")
    print(f"[modal_app] app           = {app!r}")
    print(f"[modal_app] model_volume  = {model_volume!r}")
    print("[modal_app] scaffold OK")


if __name__ == "__main__":
    _self_test()
