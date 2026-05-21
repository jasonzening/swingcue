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
            # WHAM's INSTALL.md pins python=3.9 but Modal's add_python
            # supports only 3.10+. 3.10 is the smallest valid bump;
            # WHAM is unlikely to use 3.9-specific syntax. The base
            # image's own Python 3.7 + conda env stays unused under
            # this layer (we install everything under add_python's
            # standalone interpreter).
            add_python="3.10",
        )
        # Two pre-flight cleanups for the pytorch/pytorch:1.11 base:
        #
        #   1. NVIDIA CUDA apt repo: Ubuntu 18.04 (bionic) base ships
        #      with /etc/apt/sources.list.d/cuda.list referencing a GPG
        #      key NVIDIA rotated in April 2022 (NO_PUBKEY
        #      A4B469963BF863CC). We don't need it (nvcc + headers come
        #      from /usr/local/cuda, torch wheels bundle CUDA runtime),
        #      so remove BEFORE apt_install's internal apt-get update.
        #
        #   2. /opt/conda bundled miniconda env: ships Python 3.8 +
        #      pre-built torch tied to that conda interpreter. Modal's
        #      runtime container tries to invoke python via PATH and
        #      finds /opt/conda/bin/python (3.8) first, but modal>=1.0
        #      requires >=3.10 → crash-loop. We install everything
        #      under add_python's standalone 3.10 at /usr/local, so
        #      conda is dead weight. Remove the entire tree.
        .run_commands(
            "rm -f /etc/apt/sources.list.d/cuda.list "
            "      /etc/apt/sources.list.d/nvidia-ml.list",
            "rm -rf /opt/conda",
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
        # Re-install torch 1.11.0+cu113 under the standalone Python
        # 3.10 — the base image's torch is bound to its conda env (py
        # 3.7) and not visible to add_python's interpreter.
        .pip_install(
            "torch==1.11.0+cu113",
            "torchvision==0.12.0+cu113",
            extra_index_url="https://download.pytorch.org/whl/cu113",
        )
        # WHAM runtime deps — matches WHAM repo requirements.txt at
        # pinned commit 2b54f77. Reproduced verbatim from upstream so
        # tested compatibility is preserved. Key version pins:
        #   numpy==1.22.3 (per WHAM; relaxed to numpy<2 to dodge pin
        #                  conflicts with other deps; numpy 1.26.4
        #                  resolves and is API-compatible)
        #   mmcv==1.3.9   (lite — no CUDA ops needed by WHAM demo.py.
        #                  If a runtime ImportError says it needs ops,
        #                  switch to mmcv-full==1.3.9 with find_links
        #                  for cu113/torch1.11 wheels)
        #   timm==0.4.9   (downgrade from my earlier 0.9.10 — WHAM was
        #                  developed against the older timm API)
        #   setuptools==59.5.0 (WHAM's own pin; some legacy build
        #                  scripts in WHAM expect pre-60 behavior)
        # smplx loads SMPL/SMPL-H/SMPL-X files from /models/body_models
        # at inference time. gdown is needed inside the Image so
        # setup_models can download from Drive. chumpy is split out
        # below — its setup.py needs special --no-build-isolation flags.
        .pip_install(
            # Core (mirrors WHAM requirements.txt order):
            "yacs==0.1.8",
            "joblib==1.3.2",
            "scikit-image",
            "opencv-python-headless==4.8.1.78",
            "imageio[ffmpeg]",
            "matplotlib==3.7.4",
            "tensorboard",
            "smplx==0.1.28",
            "progress",
            "einops==0.7.0",
            # mmcv==1.3.9 split out below — needs --no-build-isolation
            "timm==0.4.9",
            "munkres",
            "xtcocotools>=1.8",
            "loguru",
            "setuptools==59.5.0",
            "tqdm",
            "ultralytics",
            "gdown==5.2.0",
            # Other transitive deps I had before:
            "numpy<2.0",
            "ffmpeg-python==0.2.0",
            "scipy==1.10.1",
            "torchgeometry==0.1.2",
            "Pillow==10.2.0",
            "pyyaml>=6.0",
        )
        # chumpy 0.70 is unmaintained legacy SMPL math; its setup.py
        # uses `from pip import ...` patterns removed in pip ≥20. Modern
        # pip's build isolation hides the running pip module from the
        # build subprocess, so chumpy's setup.py crashes with
        # ModuleNotFoundError: No module named 'pip'. Workaround:
        # --no-build-isolation exposes the parent env's pip to the
        # build. That in turn requires `wheel` to be installed parent-
        # side (without it, chumpy fails with
        # `error: invalid command 'bdist_wheel'`). setuptools is already
        # bundled with add_python's Python 3.10. numpy<2 came from the
        # bulk pip_install above.
        # WHAM needs chumpy to unpickle the legacy SMPL .pkl files
        # (chumpy.ch object format from the original SMPL release).
        # mmcv 1.3.9 has the same setup.py pattern — its setup.py does
        # `from pkg_resources import ...` which modern setuptools no
        # longer exposes by default. --no-build-isolation exposes the
        # parent env's setuptools 59.5.0 (pinned in the bulk install
        # above) which still ships pkg_resources.
        .run_commands(
            "pip install wheel",
            "pip install --no-build-isolation chumpy==0.70",
            "pip install --no-build-isolation mmcv==1.3.9",
        )
        # Clone WHAM at the pinned commit + init DPVO + ViTPose submodules.
        # DPVO CUDA extensions compile at first import (or the explicit
        # install layer below); the build deps above
        # (build-essential + ninja + CUDA 11.3 devel headers) provide
        # nvcc + headers DPVO's CUDA kernels need.
        .run_commands(
            "git clone https://github.com/yohanshin/WHAM.git /opt/wham",
            f"cd /opt/wham && git checkout {WHAM_COMMIT}",
            "cd /opt/wham && git submodule update --init --recursive",
        )
        # ── Extra deps discovered iteratively after first WHAM run ──
        #
        # WHAM's requirements.txt is incomplete. INSTALL.md's conda flow
        # also installs:
        #   - fvcore + iopath        (Facebook AI Research utils, transitively
        #                             needed by some WHAM modules)
        #   - mmcv-full              (instead of mmcv lite — mmpose 0.x
        #                             requires the CUDA-compiled variant)
        #   - DPVO from submodule    (SLAM, CUDA-compiled extension)
        #   - ViTPose from submodule (pose head; transitively pulls
        #                             mmpose into the env)
        #
        # Each goes in its own layer so a single-step failure invalidates
        # only the smallest cache slice, and the previous heavy layers
        # (torch+cu113, bulk WHAM deps, chumpy+mmcv lite) stay cached.
        .pip_install("fvcore==0.1.5.post20221221", "iopath==0.1.10")
        # mmcv-full version choice:
        # - WHAM's INSTALL.md pins 1.3.9 (no py3.10 wheel; would force
        #   source build with patches)
        # - I tried 1.7.2 (last 1.x, has py3.10 wheel) but mmpose 0.24.0
        #   bundled inside WHAM's ViTPose fork enforces a strict
        #   `mmcv>=1.3.8, <=1.5.0` check via AssertionError at import
        # - 1.5.0 = highest version mmpose accepts. May or may not have
        #   a py3.10 wheel in openmmlab's index; if not, source build
        #   should now succeed (we have clang + Eigen 3.4 + nvcc +
        #   TORCH_CUDA_ARCH_LIST set up for DPVO already).
        # First uninstall the mmcv lite we installed in the chumpy step
        # above to avoid pip's "two packages named mmcv" confusion.
        .run_commands("pip uninstall -y mmcv")
        .pip_install(
            "mmcv-full==1.5.0",
            # Modal's pip_install takes find_links as a single string,
            # not a list — `expected string or bytes-like object, got 'list'`
            # if you pass [url].
            find_links="https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html",
        )
        # apt deps for DPVO + ViTPose CUDA-extension builds:
        #   clang        — torch 1.11's cpp_extension._check_abi calls
        #                  `which clang++` unconditionally on Linux even
        #                  when CXX defaults to g++. Without clang on
        #                  PATH the check raises CalledProcessError and
        #                  aborts ALL CUDA-extension builds. Actual
        #                  compilation still uses g++ from build-essential.
        #   libeigen3-dev — DPVO's lietorch CUDA kernel does
        #                  `#include <Eigen/Dense>` for Lie-algebra ops.
        #                  Installs Eigen 3.x headers to /usr/include/eigen3
        #                  (3.3.7 on Ubuntu 20.04 / 18.04 — close enough
        #                  to WHAM's pinned 3.4.0).
        .apt_install("clang", "libeigen3-dev")
        # Modal Image builds run on CPU-only workers (no GPU during
        # build). torch.utils.cpp_extension._get_cuda_arch_flags tries
        # to auto-detect the GPU arch and crashes with `IndexError:
        # list index out of range` when nothing is detected. Force the
        # target arch explicitly. 8.6 = Ampere = A10G. Add 7.5 (Turing)
        # / 8.0 (A100) / 8.9 (Ada/L4) only if we want broader cross-GPU
        # compat — each extra arch adds ~30s to the DPVO CUDA build.
        .env({"TORCH_CUDA_ARCH_LIST": "8.6"})
        # Install Eigen 3.4.0 manually instead of using libeigen3-dev's
        # 3.3.4 (Ubuntu 18.04 base). 3.3.4's Eigen/Core does
        # `#include <math_functions.hpp>` inside #ifdef EIGEN_CUDACC —
        # that CUDA header was removed in CUDA 9+ (we're on 11.3), so
        # DPVO's CUDA-extension compile fails with
        # `fatal error: math_functions.hpp: No such file or directory`.
        # Eigen 3.4.0 dropped that include. WHAM's INSTALL.md pins 3.4.0
        # too. Header-only library — just download tarball + drop headers
        # into /usr/include/{Eigen,unsupported}.
        # Removes the libeigen3-dev symlinks established earlier.
        .run_commands(
            # wget isn't in the pytorch base image; install inline.
            "apt-get update && apt-get install -y --no-install-recommends wget && "
            "rm -f /usr/include/Eigen /usr/include/unsupported && "
            "cd /tmp && "
            "wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz && "
            "tar -xzf eigen-3.4.0.tar.gz && "
            "mv eigen-3.4.0/Eigen /usr/include/Eigen && "
            "mv eigen-3.4.0/unsupported /usr/include/unsupported && "
            "rm -rf eigen-3.4.0 eigen-3.4.0.tar.gz",
        )
        # DPVO compiles CUDA extensions at install time. -v flag surfaces
        # nvcc output so failures are diagnosable from the build log.
        # The submodule's setup.py needs PYTHONPATH to find local imports
        # at build time; --no-build-isolation exposes it.
        .run_commands(
            "cd /opt/wham/third-party/DPVO && "
            "pip install -v --no-build-isolation .",
        )
        # ViTPose (WHAM's bundled fork) installs editable + brings its
        # vendored mmpose into the env. --no-build-isolation here too —
        # its setup.py is in the same legacy era as mmcv/chumpy.
        .run_commands(
            "cd /opt/wham/third-party/ViTPose && "
            "pip install -v --no-build-isolation -e .",
        )
        # chumpy 0.70 uses `np.bool` which numpy 1.24+ removed. My
        # bulk pip_install ended up at numpy 1.26.4 ("numpy<2.0").
        # WHAM's actual requirements.txt pins numpy==1.22.3 which still
        # has np.bool. Downgrade as a final layer so the heavy build
        # layers above (torch+cu113, bulk pip, mmcv-full, DPVO, ViTPose
        # CUDA compiles) stay cached. Pure-python packages don't care
        # about a numpy minor version change; C-extension wrappers
        # against numpy 1.26 are forward-compatible with 1.22 ABI.
        .run_commands(
            "pip install --force-reinstall --no-deps numpy==1.22.3",
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
