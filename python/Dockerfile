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

# ── Layer 2.9: dry-run dependency dump (diagnostic) ────────────────────────
# Hotfix v8: show which deps pull numpy/opencv/mediapipe and what versions
# pip would pick before the real install. Useful if a future dep constrains
# numpy back to <2 — the dry-run will name it explicitly in the build log.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --dry-run -r /tmp/requirements.txt 2>&1 \
    | grep -iE "numpy|opencv|mediapipe" || true

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

# ── Layer 3.7: FINAL ABI verify — image's true runtime state ───────────────
# Runs AFTER preload so the build fails if anything in the preload chain
# pulled an unexpected numpy major. Also prints versions of every relevant
# package to the build log for visibility.
RUN pip freeze | grep -iE "^(numpy|opencv|mediapipe|ultralytics|torch)"
RUN python -c "import numpy; v=numpy.__version__; print(f'[final] numpy {v}'); assert v.startswith('2.'), f'expected numpy 2.x, got {v}'"
RUN python -c "import cv2; print(f'[final] cv2 {cv2.__version__} OK')"
RUN python -c "import mediapipe; print(f'[final] mediapipe {mediapipe.__version__} OK')"
RUN python -c "from ultralytics import YOLO; YOLO('yolo11m-pose.pt'); print('[final] YOLO loaded OK')"

# Hotfix v8: assert opencv-python (full version) is absent or 4.11.0.86.
# All three opencv variants (python, headless, contrib-python) install to
# the SAME site-packages/cv2/ dir, so if any transitive dep pulls a
# different version it overwrites the others' .so files. We pin them all
# at the same numpy-2.x-compatible release and verify post-install.
RUN python -c "import importlib.metadata as m; names = [d.metadata['Name'] for d in m.distributions()]; ver = m.version('opencv-python') if 'opencv-python' in names else None; print(f'[final] opencv-python={ver}'); assert ver is None or ver == '4.11.0.86', f'opencv-python must be 4.11.0.86 or absent, got {ver}'"

RUN python -c "import importlib.metadata as m; names = [d.metadata['Name'] for d in m.distributions()]; ver = m.version('opencv-contrib-python') if 'opencv-contrib-python' in names else None; print(f'[final] opencv-contrib-python={ver}'); assert ver is None or ver == '4.11.0.86', f'opencv-contrib-python must be 4.11.0.86 or absent, got {ver}'"

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
