#!/usr/bin/env python3
"""
render_sam2_silhouette.py
Segment the golfer using SAM2 with RTMPose keypoints as prompt points.
Outputs: semi-transparent green outline+fill overlaid on impact frame.
"""

import cv2
import numpy as np
import torch
from pathlib import Path

OUT = Path("/home/jason/projects/swingcue-postest/keyframes/preview")
OUT.mkdir(parents=True, exist_ok=True)

FRAME_IDX   = 47
DTL_VIDEO   = "/home/jason/projects/swingcue-postest/input/test-dwontheline.mp4"
MODEL_CFG   = "configs/sam2.1/sam2.1_hiera_t.yaml"
CHECKPOINT  = "/home/jason/projects/swingcue-postest/models/sam2/sam2.1_hiera_tiny.pt"

# RTMPose keypoints at impact frame 47 — body points as positive prompts
PROMPT_POINTS = np.array([
    [463, 455],   # nose
    [375, 447],   # left_shoulder
    [398, 479],   # right_shoulder
    [413, 543],   # left_elbow
    [398, 582],   # right_elbow
    [435, 628],   # left_wrist
    [435, 657],   # right_wrist
    [267, 607],   # left_hip
    [310, 612],   # right_hip
    [316, 759],   # left_knee
    [367, 766],   # right_knee
    [291, 897],   # left_ankle
    [333, 920],   # right_ankle
], dtype=np.float32)

PROMPT_LABELS = np.ones(len(PROMPT_POINTS), dtype=np.int32)  # all positive


def load_frame(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    assert ret
    return frame   # BGR


def main():
    # Load frame
    frame_bgr = load_frame(DTL_VIDEO, FRAME_IDX)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_bgr.shape[:2]

    print(f"Frame size: {w}x{h}")
    print(f"Loading SAM2 (tiny)...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(MODEL_CFG, CHECKPOINT, device=device)
    predictor = SAM2ImagePredictor(model)

    print("Running inference...")
    predictor.set_image(frame_rgb)

    masks, scores, logits = predictor.predict(
        point_coords=PROMPT_POINTS,
        point_labels=PROMPT_LABELS,
        multimask_output=True,
    )

    # Pick best mask
    best_idx = np.argmax(scores)
    mask = masks[best_idx]   # bool (H, W)
    print(f"Best mask score: {scores[best_idx]:.3f}")
    print(f"Mask coverage: {mask.sum()} px / {h*w} px = {100*mask.sum()/(h*w):.1f}%")

    # ── Compose output ────────────────────────────────────────────────────────
    # 1. Semi-transparent green fill (40% opacity)
    fill_color  = np.array([40, 210, 55], dtype=np.uint8)   # BGR green
    result = frame_bgr.copy()

    mask_bool = mask.astype(bool)
    fill_layer = frame_bgr.copy()
    fill_layer[mask_bool] = fill_color
    cv2.addWeighted(fill_layer, 0.30, result, 0.70, 0, result)

    # 2. Crisp green outline on top (contour of the mask)
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(result, contours, -1, (40, 220, 60), 3, cv2.LINE_AA)

    out = OUT / "SAM2_body_silhouette_DTL_impact_fr47.png"
    cv2.imwrite(str(out), result)
    print(f"\nSaved: {out}")

    # Also copy to desktop
    import shutil
    desktop = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview")
    desktop.mkdir(parents=True, exist_ok=True)
    shutil.copy(out, desktop / out.name)
    print(f"Copied to: {desktop / out.name}")


if __name__ == "__main__":
    main()
