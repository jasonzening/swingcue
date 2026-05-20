"""
modal_app.py — Modal infrastructure scaffold for the Phase 2 pilot.

phase2a deliverable. Defines:
  - Modal App stub (name: "swingcue-pilot")
  - Persistent Volume "swingcue-pilot-models" (model weight cache)
  - Per-library Images (one per candidate 3D fitter — no shared image
    to avoid PyTorch/CUDA/pytorch3d cross-pollination per CC review §2)
  - Modal Secret refs (Modal token, SMPL research-license credentials)

This module does NOT do inference. phase2a is pure plumbing. phase2b
adds the WHAM runner that actually executes on GPU and writes joint
center timelines.

Bootstrap once-per-environment (commands Jason runs locally, NOT
invoked by this module):

    # 1. Install Modal client + auth
    pip install -r python/pilot/requirements_pilot.txt
    modal token new

    # 2. Create Volume + Secrets (read setup steps in README.md):
    modal volume create swingcue-pilot-models
    modal secret create smpl-research-creds USERNAME=... PASSWORD=...

    # 3. Verify scaffold imports clean (no Modal cost):
    python -c "from python.pilot import modal_app; print(modal_app.app.name)"

After bootstrap, phase2b adds `@app.function(...)` decorated entries
and `modal deploy modal_app.py` actually creates the Modal endpoints.

Cost model: defining Images here is free; building them happens at
deploy time. Volume + Secret creation is free. First real $$ comes
when phase2b runs WHAM inference on GPU.

Library priority per spec §2 (WHAM-first):
  1. WHAM        — A10G, CVPR 2024, 2D-keypoint input compatible
  2. Human3R     — A10G (tight on 24 GB), feed-forward fastest
  3. SMPLest-X   — H100 (needs >24 GB), highest reported accuracy
  4. EasyMocap   — A10G, mature Apache code
  5. SMPLify-X   — A10G, classical iterative baseline (slowest)
  6. 4D-Humans   — deprioritized (older than 2024+ alternatives)

phase2a defines (1) WHAM image only — others added in phase2c when
WHAM-first proves the pipeline.
"""

from __future__ import annotations

# Modal import is deferred to module evaluation, but we want to keep
# the Modal-not-installed case usable for static analysis / py_compile.
# Local dev installs `modal` via python/pilot/requirements_pilot.txt;
# CI / IDE without the dep gets the placeholder branch.
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
# Volume — single persistent volume holds ALL library model weights.
# Layout (populated by setup_models.py, run-once):
#   /models/smpl/        SMPL/SMPL-X/SMPL-H research-license weights
#   /models/wham/        WHAM checkpoint + auxiliary configs
#   /models/human3r/     Human3R weights (phase2c)
#   /models/smplest_x/   SMPLest-X 8.2 GB weights (phase2c)
#   /models/easymocap/   EasyMocap configs (phase2c)
# ---------------------------------------------------------------------------

VOLUME_NAME = "swingcue-pilot-models"

if _MODAL_AVAILABLE:
    # create_if_missing=True lets the first call from `modal deploy`
    # auto-provision the Volume; subsequent calls reuse it. Read-only
    # at inference time (mounted via @app.function(volumes=...)).
    model_volume = modal.Volume.from_name(
        VOLUME_NAME, create_if_missing=True,
    )
else:
    model_volume = None  # type: ignore


# ---------------------------------------------------------------------------
# Secrets — never embedded in code; provisioned by Jason via:
#   modal secret create smpl-research-creds USERNAME=... PASSWORD=...
# SMPL/SMPL-X research-license credentials, fetched at Image build time
# OR at setup_models() run time when downloading from smpl.is.tue.mpg.de.
# ---------------------------------------------------------------------------

SMPL_SECRET_NAME = "smpl-research-creds"

if _MODAL_AVAILABLE:
    smpl_credentials = modal.Secret.from_name(SMPL_SECRET_NAME)
else:
    smpl_credentials = None  # type: ignore


# ---------------------------------------------------------------------------
# WHAM Image — phase2a scaffold.
#
# WHAM repo: https://github.com/yohanshin/WHAM
# Trained on body7 (SMPL-H 16-beta), takes 2D keypoints + RGB video as
# input, outputs SMPL parameters + 3D joint world coords.
#
# Dependencies (from WHAM's environment.yml + pip requirements):
#   torch 2.0.1 + torchvision 0.15.2 with CUDA 11.8
#   smplx (SMPL/SMPL-X/SMPL-H model loader)
#   joblib, yacs, pyyaml, scipy, opencv-python, ffmpeg-python
#
# A10G has 24 GB; WHAM peak ~6 GB so plenty of headroom. We use the
# debian_slim base + apt_install git/ffmpeg for repo clone + video I/O.
#
# Build time on Modal: ~5-10 min first time, cached afterward. Each
# Image is ~5-10 GB stored.
# ---------------------------------------------------------------------------

if _MODAL_AVAILABLE:
    wham_image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install(
            "git",
            "ffmpeg",
            "libgl1",        # opencv runtime
            "libglib2.0-0",  # opencv runtime
            "libsm6",        # opencv runtime
            "libxext6",      # opencv runtime
            "libxrender1",   # opencv runtime
        )
        # CUDA 11.8 torch wheels — match WHAM's training env.
        .pip_install(
            "torch==2.0.1",
            "torchvision==0.15.2",
            index_url="https://download.pytorch.org/whl/cu118",
        )
        # WHAM's runtime deps. smplx pulls SMPL family model loader;
        # SMPL weight files come from the Volume at inference time.
        .pip_install(
            "smplx==0.1.28",
            "joblib==1.3.2",
            "yacs==0.1.8",
            "pyyaml>=6.0",
            "scipy==1.11.4",
            "opencv-python-headless==4.10.0.84",
            "ffmpeg-python==0.2.0",
            "numpy<2.0",     # WHAM uses np 1.x APIs
            "tqdm",
            "loguru",
        )
        # Clone WHAM into the image. Pinning the commit means subsequent
        # rebuilds are reproducible. TODO(phase2b): pin a specific
        # release/commit before serious benchmarking.
        .run_commands(
            "git clone https://github.com/yohanshin/WHAM.git /opt/wham",
            # Provide /opt/wham in PYTHONPATH so `from lib...` imports
            # used by WHAM's demo work without cwd gymnastics.
        )
        .env({"PYTHONPATH": "/opt/wham:$PYTHONPATH"})
    )
else:
    wham_image = None  # type: ignore


# ---------------------------------------------------------------------------
# Image registry — single dict so phase2c additions surface clearly and
# the pilot CLI can iterate over "which images are currently defined".
# ---------------------------------------------------------------------------

IMAGES = {
    "wham": wham_image,
    # phase2c additions:
    # "human3r":   human3r_image,
    # "smplest_x": smplest_x_image,
    # "easymocap": easymocap_image,
    # "smplify_x": smplify_x_image,
}


# ---------------------------------------------------------------------------
# Self-test — runs when this file is executed directly. NOT a Modal
# deploy; just verifies the scaffold definitions are syntactically + API
# valid against whatever Modal version is installed locally.
#
#   python -m python.pilot.modal_app
# ---------------------------------------------------------------------------

def _self_test() -> None:
    if not _MODAL_AVAILABLE:
        print(
            "[modal_app] modal package not installed — "
            "pip install -r python/pilot/requirements_pilot.txt"
        )
        return
    print(f"[modal_app] APP_NAME           = {APP_NAME}")
    print(f"[modal_app] VOLUME_NAME        = {VOLUME_NAME}")
    print(f"[modal_app] SMPL_SECRET_NAME   = {SMPL_SECRET_NAME}")
    print(f"[modal_app] images defined     = {list(IMAGES.keys())}")
    print(f"[modal_app] app                = {app!r}")
    print(f"[modal_app] model_volume       = {model_volume!r}")
    print("[modal_app] scaffold OK — phase2b adds @app.function entries")


if __name__ == "__main__":
    _self_test()
