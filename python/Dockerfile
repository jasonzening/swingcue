FROM python:3.11-slim

# ── Layer 1: apt + system libs ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Layer 2: CPU-only torch wheel ──────────────────────────────────────────
# Critical for image size: the default PyPI `torch` pulls the CUDA wheel and
# nvidia-* packages (~3-4 GB). Railway is CPU-only so we install the CPU
# wheel from pytorch.org/whl/cpu first; when ultralytics is resolved in
# layer 3 it sees torch is already satisfied and does not upgrade it.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

# ── Layer 2.5: pin numpy<2.0 — torch's default numpy 2.x breaks the
# mediapipe/opencv binaries which were compiled against numpy 1.x ABI.
# Must come AFTER torch install (so torch's numpy 2.x gets downgraded)
# and BEFORE the rest of requirements (so ultralytics resolves against it).
RUN pip install --no-cache-dir --upgrade "numpy<2.0"

# ── Layer 3: requirements (ultralytics + everything else) ──────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Layer 3.4: disable Ultralytics auto-install ─────────────────────────────
# ultralytics.checks.check_requirements() will silently `pip install -U <pkg>`
# at first YOLO() call for any dep it considers out-of-date. That auto-upgrade
# was the actual cause of numpy being reverted to 2.x at build time (during
# preload below) and at runtime. Killing it here is the load-bearing fix.
ENV YOLO_AUTOINSTALL=False
ENV ULTRALYTICS_OFFLINE=True

# ── Layer 3.5: pre-download yolo11m-pose.pt (~44 MB) ───────────────────────
# Bakes the weights into the image so the first request after a redeploy is
# not penalised by a 30s download from GitHub.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11m-pose.pt')"

# ── Layer 3.6: belt-and-suspenders re-pin AFTER preload ────────────────────
# Even with AUTOINSTALL off, any pip resolver hiccup could shift numpy.
# --force-reinstall --no-deps restores the version we want without touching
# the rest of the resolved graph.
RUN pip install --no-cache-dir --force-reinstall --no-deps "numpy<2.0"

# ── Layer 3.7: FINAL ABI verify — image's true runtime state ───────────────
# Runs AFTER preload + numpy re-pin so the build fails if anything in the
# preload chain shifted numpy past 2.0. Also prints versions of every
# relevant package to the build log for visibility.
RUN pip freeze | grep -iE "^(numpy|opencv|mediapipe|ultralytics|torch)"
RUN python -c "import numpy; v=numpy.__version__; print(f'[final] numpy {v}'); assert v.startswith('1.'), f'numpy must be 1.x, got {v}'"
RUN python -c "import cv2; print(f'[final] cv2 {cv2.__version__} OK')"
RUN python -c "import mediapipe; print(f'[final] mediapipe {mediapipe.__version__} OK')"
RUN python -c "from ultralytics import YOLO; YOLO('yolo11m-pose.pt'); print('[final] YOLO loaded OK')"

# Hotfix v5: assert opencv-python (the full version, separate from
# opencv-python-headless) is either absent or pinned to 4.10.0.84. Both
# packages install to the same site-packages/cv2/ dir; if ultralytics
# pulled in opencv-python at a later 4.11.x version it overwrote our
# headless binaries with numpy-2.x-compiled ones.
RUN python -c "import importlib.metadata as m; names = [d.metadata['Name'] for d in m.distributions()]; ver = m.version('opencv-python') if 'opencv-python' in names else None; print(f'[final] opencv-python={ver}'); assert ver is None or ver == '4.10.0.84', f'opencv-python must be 4.10.0.84 or absent, got {ver}'"

# Hotfix v6: assert opencv-contrib-python is either absent or 4.10.0.84.
# This is the THIRD opencv variant — ultralytics-thop transitively pulls
# it, and like its siblings it writes to the same cv2/ dir. Last write
# wins, so an unpinned 4.11.x here will overwrite both opencv-python and
# opencv-python-headless 4.10.0.84 with numpy-2.x-compiled binaries.
RUN python -c "import importlib.metadata as m; names = [d.metadata['Name'] for d in m.distributions()]; ver = m.version('opencv-contrib-python') if 'opencv-contrib-python' in names else None; print(f'[final] opencv-contrib-python={ver}'); assert ver is None or ver == '4.10.0.84', f'opencv-contrib-python must be 4.10.0.84 or absent, got {ver}'"

# ── Layer 3.8: fresh-subprocess cv2 cold import ────────────────────────────
# Hotfix v4: same-process imports above can pass due to module caching when
# cv2 was loaded earlier in the build (e.g. by Ultralytics during preload)
# while numpy was transiently 2.x. The runtime container always starts with
# a fresh interpreter — replicate that here so an ABI mismatch crashes the
# BUILD, not the first /analyze request.
RUN python -c "import subprocess, sys; out = subprocess.check_output([sys.executable, '-c', 'import cv2; print(f\"cv2 cold import OK in fresh process: {cv2.__version__}\")'], stderr=subprocess.STDOUT).decode(); print(out)"

# ── Layer 4: app code ──────────────────────────────────────────────────────
COPY *.py .
# Subpackages — the `COPY *.py` glob above only matches root-level files.
COPY sam3d/ ./sam3d/
COPY yolo/  ./yolo/

# Python handles PORT env var directly — no shell interpolation issues
CMD ["python", "main.py"]
