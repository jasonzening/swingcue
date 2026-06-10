#!/usr/bin/env python3
"""
render_skeleton.py
Raw skeleton overlay — no indicators, no arrows, no judgment.
Just draw all detected keypoints and connections on the impact frame.
"""

import cv2
import numpy as np
from pathlib import Path

OUT = Path("/home/jason/projects/swingcue-postest/keyframes/preview")
OUT.mkdir(parents=True, exist_ok=True)
DTL_VIDEO = "/home/jason/projects/swingcue-postest/input/test-dwontheline.mp4"

# RTMPose keypoints — impact frame 47, all 17 COCO joints
KPS = {
    "nose":           (463, 455),
    "left_eye":       (472, 439),
    "right_eye":      (463, 437),
    "left_ear":       (435, 422),
    "right_ear":      (434, 416),
    "left_shoulder":  (375, 447),
    "right_shoulder": (398, 479),
    "left_elbow":     (413, 543),
    "right_elbow":    (398, 582),
    "left_wrist":     (435, 628),
    "right_wrist":    (435, 657),
    "left_hip":       (267, 607),
    "right_hip":      (310, 612),
    "left_knee":      (316, 759),
    "right_knee":     (367, 766),
    "left_ankle":     (291, 897),
    "right_ankle":    (333, 920),
}

# Skeleton connections
BONES = [
    # Head
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    # Spine
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    # Torso (shoulder mid → hip mid — computed below)
    # Arms
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    # Legs
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    # Shoulder to hip
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
]

C = (0, 230, 0)    # bright green
DOT_R  = 5
LINE_W = 2


def render():
    cap = cv2.VideoCapture(DTL_VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 47)
    ret, frame = cap.read()
    cap.release()
    assert ret

    img = frame.copy()

    # Draw bones
    for a, b in BONES:
        cv2.line(img, KPS[a], KPS[b], C, LINE_W, cv2.LINE_AA)

    # Spine: shoulder mid → hip mid
    sh_mid  = ((KPS["left_shoulder"][0] + KPS["right_shoulder"][0]) // 2,
               (KPS["left_shoulder"][1] + KPS["right_shoulder"][1]) // 2)
    hip_mid = ((KPS["left_hip"][0] + KPS["right_hip"][0]) // 2,
               (KPS["left_hip"][1] + KPS["right_hip"][1]) // 2)
    cv2.line(img, sh_mid, hip_mid, C, LINE_W, cv2.LINE_AA)
    # Neck to nose
    cv2.line(img, sh_mid, KPS["nose"], C, LINE_W, cv2.LINE_AA)

    # Draw joint dots on top
    for name, p in KPS.items():
        cv2.circle(img, p, DOT_R, C, -1, cv2.LINE_AA)
        cv2.circle(img, p, DOT_R, (255, 255, 255), 1, cv2.LINE_AA)  # white outline

    out = OUT / "skeleton_DTL_impact_fr47.png"
    cv2.imwrite(str(out), img)
    print(f"Saved: {out}")


if __name__ == "__main__":
    render()
