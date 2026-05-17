# =============================================================================
# PR-3 Option C: multi-stage build.
#
#   Stage 1 (yolo-builder): install ultralytics + torch, export
#   yolo11m-pose.pt → yolo11m-pose.onnx, verify the decoder agrees with
#   ultralytics native inference within 5px. This stage's filesystem
#   (PyTorch + 3-cv2-variant chaos + numpy 2.x) is DISCARDED.
#
#   Stage 2 (runtime): slim image with pre-PR-3 production deps +
#   onnxruntime. The ONLY artifact crossing the stage boundary is the
#   .onnx file from /build.
#
# This design eliminates the entire class of opencv-python / numpy ABI
# conflicts hotfixes v1-v8 fought against. See
# docs/PR-3_AUDIT.md and docs/decisions/PR-3_C_ONNX_DESIGN.md.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: yolo-builder — ONNX export + decoder smoke test
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS yolo-builder

# Same X11 + multimedia libs as Stage 2 — Stage 1 installs ultralytics
# which pulls opencv-python (full, not headless), and cv2 .so dlopens
# libxcb / libsm / libxext / libxrender / libgl at import time.
# Stage 1 image is discarded after ONNX export, so size is not a concern.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# CPU torch wheel — ultralytics needs torch to load the .pt and run export.
# This stage's image is discarded, so we don't fight over deps here.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

# ultralytics + onnxruntime (both needed for export + decoder verify)
RUN pip install --no-cache-dir ultralytics onnxruntime

# Bring in the decoder helpers so the verify script can import them.
COPY yolo/__init__.py /build/yolo/__init__.py
COPY yolo/decoder.py  /build/yolo/decoder.py

# Pre-download reference image for build-time decoder verification.
# ultralytics 8.x removed bundled assets — zidane.jpg is fetched on-demand
# from ultralytics.com. Download it once here to a fixed path so
# verify_onnx_export.py has a deterministic, offline-safe input.
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://ultralytics.com/images/zidane.jpg', '/build/zidane.jpg')"

# Export yolo11m-pose.pt → yolo11m-pose.onnx with fixed parameters.
# - imgsz=640         deterministic input shape (anchors = 8400)
# - dynamic=False     static input, decoder may rely on 8400 anchors
# - nms=False         we argmax-by-confidence in postprocess()
# - simplify=True     graph constant-folding
# - opset=12          conservative, broad onnxruntime compatibility
RUN python -c "from ultralytics import YOLO; \
    m = YOLO('yolo11m-pose.pt'); \
    m.export(format='onnx', imgsz=640, simplify=True, opset=12, \
             dynamic=False, nms=False)"

# Build-time smoke test: manual decoder must agree with ultralytics native
# inference within 5px on a reference image. Failure aborts the build.
COPY scripts/verify_onnx_export.py /build/verify_onnx_export.py
RUN python /build/verify_onnx_export.py

# -----------------------------------------------------------------------------
# Stage 2: runtime — slim image, NO ultralytics, NO torch
# -----------------------------------------------------------------------------
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The ONLY artifact pulled from the builder stage.
COPY --from=yolo-builder /build/yolo11m-pose.onnx /app/yolo11m-pose.onnx

# ── Stage-2 ABI verify ─────────────────────────────────────────────────────
# Back to the pre-PR-3 production state (numpy 1.x + opencv-python-headless
# 4.10.0.84 + mediapipe 0.10.14), plus onnxruntime.
RUN python -c "import numpy; v=numpy.__version__; \
    assert v.startswith('1.'), f'expected numpy 1.x in runtime, got {v}'; \
    print(f'[final] numpy {v}')"
RUN python -c "import cv2; print(f'[final] cv2 {cv2.__version__} OK')"
RUN python -c "import mediapipe; print(f'[final] mediapipe {mediapipe.__version__} OK')"
RUN python -c "import onnxruntime as ort; print(f'[final] onnxruntime {ort.__version__} OK')"
RUN python -c "import os; \
    p='/app/yolo11m-pose.onnx'; \
    assert os.path.exists(p), 'ONNX model missing at /app/yolo11m-pose.onnx'; \
    sz = os.path.getsize(p) / 1024 / 1024; \
    print(f'[final] yolo11m-pose.onnx present ({sz:.1f} MB)')"

# Fresh-subprocess cv2 cold import (kept from hotfix v4 as belt-and-suspenders).
RUN python -c "import subprocess, sys; out = subprocess.check_output([sys.executable, '-c', 'import cv2; print(f\"cv2 cold import OK in fresh process: {cv2.__version__}\")'], stderr=subprocess.STDOUT).decode(); print(out)"

# ── App code ───────────────────────────────────────────────────────────────
# `COPY *.py .` is non-recursive — picks up only main.py, analyzer.py,
# phase_detector.py at the root of the python/ build context. The
# python/scripts/ directory (verify script) is NOT included; it stays in
# Stage 1 where it belongs.
COPY *.py .
COPY sam3d/ ./sam3d/
COPY yolo/  ./yolo/

CMD ["python", "main.py"]
