"""
YOLO11-pose runtime inference via onnxruntime.

PR-3 Option C: ultralytics + torch are no longer in the runtime image
(they live in Dockerfile's `yolo-builder` stage which exports the .onnx
file). At runtime we load the exported model with `onnxruntime` and use
the manual decoder in `yolo/decoder.py`.

Public surface is intentionally identical to the previous ultralytics-
based version:

    MODEL_NAME: str
    async def infer_pose(png_bytes: bytes) -> Optional[dict]

`yolo/orchestrator.py` and `python/main.py` do not change.
"""

from __future__ import annotations
import asyncio
import logging
import os
import time
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from yolo.decoder import postprocess, preprocess

logger = logging.getLogger(__name__)

# Public model identifier written to pose_3d_phases.yolo_model column.
# Keeping the "-pose" suffix matches the ultralytics naming convention
# so the frontend label stays stable; "-onnx" suffix flags the runtime.
MODEL_NAME = "yolo11m-pose-onnx"

# Filesystem path where the Docker stage 2 COPYs the model.
MODEL_PATH = "/app/yolo11m-pose.onnx"

# Module-level singleton — InferenceSession construction is ~200ms on CPU,
# so we cache it after first use. Subsequent infer_pose() calls are hot.
_session: Optional[ort.InferenceSession] = None
_input_name: Optional[str] = None


def _get_session() -> tuple[ort.InferenceSession, str]:
    """Lazy singleton — first call pays the load; rest are hot."""
    global _session, _input_name
    if _session is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"[yolo] ONNX model missing at {MODEL_PATH}; "
                "expected Dockerfile yolo-builder stage to have produced it"
            )
        logger.info(f"[yolo] loading {MODEL_PATH}...")
        t0 = time.time()
        _session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"],
        )
        _input_name = _session.get_inputs()[0].name
        out0 = _session.get_outputs()[0]
        logger.info(
            f"[yolo] session loaded in {(time.time() - t0) * 1000:.0f}ms "
            f"(input={_input_name} {_session.get_inputs()[0].shape}, "
            f"output={out0.name} {out0.shape})"
        )
    assert _input_name is not None
    return _session, _input_name


def _run_sync(png_bytes: bytes) -> Optional[dict]:
    """Synchronous inference body — called via asyncio.to_thread."""
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            f"[yolo] cv2.imdecode returned None ({len(png_bytes)} bytes input)"
        )

    orig_h, orig_w = image.shape[:2]
    input_arr, scale, pad_xy = preprocess(image)

    session, input_name = _get_session()
    t0 = time.time()
    outputs = session.run(None, {input_name: input_arr})
    inference_ms = int((time.time() - t0) * 1000)

    keypoints, meta = postprocess(
        outputs[0], scale, pad_xy, orig_h=orig_h, orig_w=orig_w,
    )
    if keypoints is None or meta is None:
        return None

    return {
        "keypoints_2d": keypoints.tolist(),  # 17 × [x, y, conf] in source px
        "bbox": meta["bbox"],                 # [x1, y1, x2, y2]
        "image_width": int(orig_w),
        "image_height": int(orig_h),
        "inference_ms": inference_ms,
        "model": MODEL_NAME,
    }


async def infer_pose(png_bytes: bytes) -> Optional[dict]:
    """
    Decode a single PNG frame and run YOLO11-pose ONNX inference.

    Returns:
        dict with keypoints_2d / bbox / image_width / image_height /
        inference_ms / model, or None if no person was detected above
        the confidence threshold.

    Raises:
        ValueError — if PNG bytes cannot be decoded.
        FileNotFoundError — if /app/yolo11m-pose.onnx is missing.
    """
    return await asyncio.to_thread(_run_sync, png_bytes)
