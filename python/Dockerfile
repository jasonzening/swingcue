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

# ── Layer 3: requirements (ultralytics + everything else) + preload weights ─
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Pre-download yolo11m-pose.pt (~44 MB) into the image so the first request
# after a redeploy is not penalised by a 30s download from GitHub.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11m-pose.pt')"

# ── Layer 4: app code ──────────────────────────────────────────────────────
COPY *.py .
# Subpackages — the `COPY *.py` glob above only matches root-level files.
COPY sam3d/ ./sam3d/
COPY yolo/  ./yolo/

# Python handles PORT env var directly — no shell interpolation issues
CMD ["python", "main.py"]
