"""
YOLO11-pose inference wrapper.

Loads the model once (module-level singleton) and exposes an async
`infer_pose(png_bytes)` that runs the synchronous Ultralytics call inside
`asyncio.to_thread` so the 5 phase tasks can overlap their CPU work.

The model file (~44 MB for yolo11m-pose.pt) is pre-downloaded at Docker
build time so the first request after a redeploy is not penalised by a
~30s download from GitHub.
"""

import asyncio
import io
import logging
import time
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

MODEL_NAME = "yolo11m-pose.pt"

_model: Optional[YOLO] = None


def _get_model() -> YOLO:
    """Lazy singleton — first call pays the ~1–3s torch load, rest are hot."""
    global _model
    if _model is None:
        logger.info(f"[yolo] loading model {MODEL_NAME}...")
        t0 = time.time()
        _model = YOLO(MODEL_NAME)
        logger.info(f"[yolo] model loaded in {(time.time() - t0) * 1000:.0f}ms")
    return _model


def _run_sync(model: YOLO, image: np.ndarray) -> Optional[dict]:
    """Synchronous inference body — called via asyncio.to_thread."""
    t0 = time.time()
    results = model(image, verbose=False)
    inference_ms = int((time.time() - t0) * 1000)

    if not results:
        return None
    r = results[0]
    if r.keypoints is None or r.keypoints.data is None or len(r.keypoints.data) == 0:
        return None

    # Single-person swing video — take person 0 (highest-confidence).
    kpts_tensor = r.keypoints.data[0]  # shape: [17, 3] = (x, y, conf)
    kpts_list: list[list[float]] = kpts_tensor.cpu().tolist()

    h, w = r.orig_shape  # tensor returns (H, W)

    bbox: Optional[list[float]] = None
    if r.boxes is not None and r.boxes.xyxy is not None and len(r.boxes.xyxy) > 0:
        bbox = [float(v) for v in r.boxes.xyxy[0].cpu().tolist()]

    return {
        "keypoints_2d": kpts_list,       # 17 × [x, y, conf] in source pixels
        "bbox": bbox,                     # [x1, y1, x2, y2] or None
        "image_width": int(w),
        "image_height": int(h),
        "inference_ms": inference_ms,
        "model": MODEL_NAME.replace(".pt", ""),  # e.g. "yolo11m-pose"
    }


async def infer_pose(png_bytes: bytes) -> Optional[dict]:
    """
    Decode a single PNG frame and run YOLO11-pose. Returns a dict (see
    `_run_sync`) or None if no person was detected.

    Raises:
      ValueError — if PNG bytes cannot be decoded (caller treats as failure).
    """
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            f"[yolo] cv2.imdecode returned None ({len(png_bytes)} bytes input)"
        )

    model = _get_model()
    return await asyncio.to_thread(_run_sync, model, image)
