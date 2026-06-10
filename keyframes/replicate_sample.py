#!/usr/bin/env python3
"""
replicate_sample.py - strict sample copy, one indicator at a time.

SAMPLE STYLE RULES (from analysis of reference images):
  - ZERO neon glow: completely flat, matte lines (broadcast annotation style)
  - NO joint dots: lines just meet at endpoints, no circular markers
  - Flat solid-fill badge circles: bold O (correct) or X (wrong), no border ring, no glow
  - Full-brightness background: never dimmed
  - Line thickness: ~5px solid
  - Badge: ~120px diameter, upper area of frame
  - GREEN = correct, RED = wrong
"""

import cv2
import numpy as np
from pathlib import Path

OUT_DIR = Path("/home/jason/projects/swingcue-postest/keyframes/preview")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DTL_VIDEO = "/home/jason/projects/swingcue-postest/input/test-dwontheline.mp4"
FO_VIDEO  = "/home/jason/projects/swingcue-postest/input/test-faceon.mp4"

# Flat colors (BGR) - exactly as in samples
C_GREEN = (0,   230,   0)
C_RED   = (30,   30, 220)
C_WHITE = (255, 255, 255)


def load_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    assert ret, f"Cannot read frame {frame_idx}"
    return frame


def flat_line(img, p1, p2, color, thickness=5):
    """Flat solid line, zero glow."""
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def badge(img, center, verdict, diameter=120):
    """
    Flat solid circle + bold O or X.
    No glow, no border ring - matches broadcast annotation style.
    """
    r = diameter // 2
    color = C_GREEN if verdict == 'ok' else C_RED
    cx, cy = center
    thick = max(13, r // 5)

    cv2.circle(img, center, r, color, -1, cv2.LINE_AA)

    if verdict == 'ok':
        cv2.circle(img, (cx, cy), int(r * 0.58), C_WHITE, thick, cv2.LINE_AA)
    else:
        off = int(r * 0.50)
        cv2.line(img, (cx-off, cy-off), (cx+off, cy+off), C_WHITE, thick, cv2.LINE_AA)
        cv2.line(img, (cx+off, cy-off), (cx-off, cy+off), C_WHITE, thick, cv2.LINE_AA)


# Keypoint data (RTMPose smoothed coords)
DTL_IMP = {
    "right_shoulder": (398, 479),
    "right_elbow":    (398, 582),
    "right_wrist":    (435, 657),
    "left_shoulder":  (375, 447),
    "left_elbow":     (413, 543),
    "left_wrist":     (435, 628),
}

DTL_ADDR = {
    "right_shoulder": (401, 470),
    "right_elbow":    (404, 580),
    "right_wrist":    (405, 684),
    "left_shoulder":  (387, 473),
    "left_elbow":     (386, 583),
    "left_wrist":     (388, 671),
}

FO_ADDR = {
    "left_shoulder":  (528, 474),
    "right_shoulder": (391, 484),
    "left_elbow":     (491, 575),
    "right_elbow":    (416, 584),
    "left_wrist":     (464, 666),
    "right_wrist":    (432, 683),
    "left_hip":       (494, 634),
    "right_hip":      (413, 632),
}

FO_IMP = {
    "left_shoulder":  (529, 456),
    "right_shoulder": (406, 486),
    "left_elbow":     (491, 558),
    "right_elbow":    (427, 579),
    "left_wrist":     (456, 641),
    "right_wrist":    (430, 654),
    "left_hip":       (542, 620),
    "right_hip":      (468, 626),
}


# =============================================================================
# RENDER 1: DTL IMPACT - Wrist V (ref: 125_1, 128_1)
# Green V-shape: right shoulder -> elbow -> wrist
# Green O badge upper-center
# =============================================================================
def render_dtl_wrist_v():
    img = load_frame(DTL_VIDEO, 47)
    h, w = img.shape[:2]

    sh  = DTL_IMP["right_shoulder"]
    elb = DTL_IMP["right_elbow"]
    wr  = DTL_IMP["right_wrist"]

    flat_line(img, sh, elb, C_GREEN, 5)
    flat_line(img, elb, wr,  C_GREEN, 5)

    badge(img, (w // 2, 80), 'ok', 120)

    path = OUT_DIR / "DTL_impact_wrist_V__ref_125_128.png"
    cv2.imwrite(str(path), img)
    print(f"Saved: {path}")


if __name__ == "__main__":
    render_dtl_wrist_v()
    print("Done.")
