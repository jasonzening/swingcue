"""
Build-time decoder sanity check (Stage 1 only).

Runs at the END of the yolo-builder Docker stage, while both
`ultralytics` and `onnxruntime` are still installed. Fails the build if
our manual ONNX decoder produces keypoints more than 5 pixels away from
ultralytics' own native YOLO() inference on a reference image.

The Stage 2 runtime image never sees this script — it lives under
python/scripts/, which the runtime's `COPY *.py .` glob does not match.
"""

from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

# Ensure python/ is on sys.path so we can import the production decoder.
PYTHON_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PYTHON_DIR))
from yolo.decoder import postprocess, preprocess  # noqa: E402

REF_IMG = "ultralytics/assets/zidane.jpg"  # ships with the ultralytics package
MAX_PIX_DIVERGENCE = 5.0


def main() -> int:
    # ── Reference: ultralytics native path ──────────────────────────────
    m = YOLO("yolo11m-pose.pt")
    ref = m(REF_IMG, verbose=False, conf=0.25)[0]
    if ref.keypoints is None or ref.keypoints.data is None or len(ref.keypoints.data) == 0:
        print("[verify] FAIL: ultralytics found no person in zidane.jpg")
        return 1
    ref_kps = ref.keypoints.data[0].cpu().numpy()  # (17, 3) = x, y, conf

    # ── Test: our onnxruntime decoder ───────────────────────────────────
    session = ort.InferenceSession(
        "yolo11m-pose.onnx",
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    image = cv2.imread(REF_IMG)
    if image is None:
        print(f"[verify] FAIL: cv2.imread returned None for {REF_IMG}")
        return 1

    input_arr, scale, pad = preprocess(image)
    output = session.run(None, {input_name: input_arr})[0]
    test_kps, meta = postprocess(
        output, scale, pad,
        orig_h=image.shape[0], orig_w=image.shape[1],
    )
    if test_kps is None:
        print("[verify] FAIL: ONNX decoder returned no detection")
        return 1

    # ── Compare keypoint coordinates ────────────────────────────────────
    diff = float(np.abs(ref_kps[:, :2] - test_kps[:, :2]).max())
    print(f"[verify] reference (ultralytics) and onnxruntime keypoints:")
    for i, (rkp, tkp) in enumerate(zip(ref_kps, test_kps)):
        print(
            f"  kp{i:2d}: ref=({rkp[0]:7.2f}, {rkp[1]:7.2f}, "
            f"{rkp[2]:.2f})  test=({tkp[0]:7.2f}, {tkp[1]:7.2f}, "
            f"{tkp[2]:.2f})"
        )
    print(f"[verify] max coordinate divergence: {diff:.2f}px (threshold {MAX_PIX_DIVERGENCE}px)")

    if diff >= MAX_PIX_DIVERGENCE:
        print(
            f"[verify] FAIL: ONNX decoder vs ultralytics divergence too large. "
            f"Decoder math is wrong — review yolo/decoder.py."
        )
        return 1

    print("[verify] OK: ONNX decoder agrees with ultralytics within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
