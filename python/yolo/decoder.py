"""
YOLO11-pose ONNX preprocess + postprocess (no model loading).

Split out from `inference.py` so the build-time verify script
(scripts/verify_onnx_export.py) can import the exact same decoder used
at runtime, without pulling in `onnxruntime` or the session singleton.

Tensor convention (after `model.export(format='onnx', imgsz=640,
nms=False, dynamic=False, opset=12, simplify=True)`):

    output[0] shape: [1, 56, 8400]    (batch, channels, anchors)
                                       (8400 = 80² + 40² + 20² grid cells)

    channels[0:4]  = bbox cx, cy, w, h  (640×640 input pixel space)
    channels[4]    = person confidence (single-class detector)
    channels[5:56] = 17 × (kx, ky, kvis)  (640×640 input pixel space;
                                          kvis is sigmoid-activated)

After transpose to [8400, 56] each row is one candidate detection. We
argmax over column 4 (skip full NMS — single-person golf swing) to pick
the top detection, then reverse-letterbox keypoint coords to the source
image's pixel space.
"""

from __future__ import annotations
from typing import Optional

import cv2
import numpy as np

INPUT_SIZE: int = 640
LETTERBOX_PAD_VALUE: int = 114
DETECTION_CONF_THRESHOLD: float = 0.25
NUM_KEYPOINTS: int = 17
CHANNELS_PER_DET: int = 4 + 1 + NUM_KEYPOINTS * 3  # = 56


def preprocess(image_bgr: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Letterbox-resize a BGR image to (640, 640), returning the model input
    and the inverse-letterbox parameters needed to map keypoints back.

    Returns:
        input_arr: float32 [1, 3, 640, 640], BGR → RGB → CHW, /255.0
        scale:     uniform scale factor applied during resize
        (pad_x, pad_y): padding added on the left and top of the canvas
    """
    h, w = image_bgr.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), LETTERBOX_PAD_VALUE, dtype=np.uint8)
    pad_y = (INPUT_SIZE - new_h) // 2
    pad_x = (INPUT_SIZE - new_w) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    arr = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    return arr, scale, (pad_x, pad_y)


def postprocess(
    output: np.ndarray,
    scale: float,
    pad_xy: tuple[int, int],
    orig_h: int,
    orig_w: int,
    conf_threshold: float = DETECTION_CONF_THRESHOLD,
) -> tuple[Optional[np.ndarray], Optional[dict]]:
    """
    Decode YOLO11-pose raw output into 17 keypoints in source pixel coords.

    Args:
        output:         ONNX session output[0], shape [1, 56, 8400]
        scale, pad_xy:  from preprocess()
        orig_h, orig_w: source image dims (for bbox clipping)
        conf_threshold: minimum person confidence; below → return (None, None)

    Returns:
        keypoints_2d:  np.ndarray [17, 3] = (x, y, visibility) in source px,
                       or None if no detection passes conf_threshold
        meta:          dict with 'conf' and 'bbox' (xyxy in source px),
                       or None
    """
    if output.ndim != 3 or output.shape[1] != CHANNELS_PER_DET:
        raise ValueError(
            f"[yolo] unexpected ONNX output shape {output.shape!r}, "
            f"expected [1, {CHANNELS_PER_DET}, N]"
        )

    # [1, 56, 8400] → [56, 8400] → [8400, 56]
    candidates = output[0].T  # shape [N, 56]

    conf_col = candidates[:, 4]
    best_idx = int(conf_col.argmax())
    best = candidates[best_idx]
    conf = float(best[4])
    if conf < conf_threshold:
        return None, None

    pad_x, pad_y = pad_xy

    kps_raw = best[5:].reshape(NUM_KEYPOINTS, 3)
    kps_out = np.empty_like(kps_raw)
    kps_out[:, 0] = (kps_raw[:, 0] - pad_x) / scale
    kps_out[:, 1] = (kps_raw[:, 1] - pad_y) / scale
    kps_out[:, 2] = kps_raw[:, 2]
    kps_out[:, 0] = np.clip(kps_out[:, 0], 0, orig_w - 1)
    kps_out[:, 1] = np.clip(kps_out[:, 1], 0, orig_h - 1)

    cx, cy, w, h = best[:4]
    bbox_xyxy = [
        float((cx - w / 2 - pad_x) / scale),
        float((cy - h / 2 - pad_y) / scale),
        float((cx + w / 2 - pad_x) / scale),
        float((cy + h / 2 - pad_y) / scale),
    ]

    return kps_out, {"conf": conf, "bbox": bbox_xyxy}
