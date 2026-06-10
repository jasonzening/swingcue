#!/usr/bin/env python3
"""
render_body_capsules.py
Body silhouette from joint capsules — no segmentation model needed.
Each limb segment = fat rounded capsule drawn from joint to joint.
Single 60% blend pass for uniform transparency.
"""

import cv2
import numpy as np
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

OUT = Path("/home/jason/projects/swingcue-postest/keyframes/preview")
OUT.mkdir(parents=True, exist_ok=True)
DTL_VIDEO = "/home/jason/projects/swingcue-postest/input/test-dwontheline.mp4"

C_GREEN = (40, 210, 55)   # BGR
OPACITY = 0.60


# ── Data structure ─────────────────────────────────────────────────────────────
@dataclass
class Capsule:
    """One limb segment: two endpoints + half-width."""
    p1: Tuple[int, int]
    p2: Tuple[int, int]
    hw: int       # half-width in pixels


def draw_capsule(canvas, cap: Capsule, color):
    """
    Draw a filled rounded capsule between p1 and p2 with given half-width.
    Method: filled rectangle aligned to segment + two end circles.
    """
    x1, y1 = cap.p1
    x2, y2 = cap.p2
    hw = cap.hw

    dx = x2 - x1; dy = y2 - y1
    length = math.hypot(dx, dy)
    if length < 1:
        cv2.circle(canvas, cap.p1, hw, color, -1, cv2.LINE_AA)
        return

    # Unit perpendicular
    px = -dy / length; py = dx / length

    # Four corners of the rectangle
    corners = np.array([
        [x1 + px*hw, y1 + py*hw],
        [x2 + px*hw, y2 + py*hw],
        [x2 - px*hw, y2 - py*hw],
        [x1 - px*hw, y1 - py*hw],
    ], dtype=np.int32)

    cv2.fillPoly(canvas, [corners], color, cv2.LINE_AA)

    # Round end caps
    cv2.circle(canvas, cap.p1, hw, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, cap.p2, hw, color, -1, cv2.LINE_AA)


# ── Keypoints — impact frame 47 ────────────────────────────────────────────────
K = {
    "nose":           (463, 455),
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

# Derived midpoints
SH_MID  = ((K["left_shoulder"][0] + K["right_shoulder"][0]) // 2,
            (K["left_shoulder"][1] + K["right_shoulder"][1]) // 2)
HIP_MID = ((K["left_hip"][0]  + K["right_hip"][0])  // 2,
            (K["left_hip"][1]  + K["right_hip"][1])  // 2)


def build_capsules():
    """
    Each segment is independent (joint-pair + width).
    Widths in pixels approximate real limb proportions at this camera distance.
    Frame is 720px wide — rough calibration: shoulder-to-shoulder ~25px.
    """
    return [
        # ── Head / neck ──────────────────────────────────────
        Capsule(SH_MID,               K["nose"],             hw=14),

        # ── Torso ────────────────────────────────────────────
        Capsule(K["left_shoulder"],   K["right_shoulder"],   hw=18),  # shoulder girdle
        Capsule(K["left_hip"],        K["right_hip"],        hw=18),  # hip girdle
        Capsule(SH_MID,               HIP_MID,               hw=22),  # spine/trunk

        # ── Left arm ─────────────────────────────────────────
        Capsule(K["left_shoulder"],   K["left_elbow"],       hw=11),  # upper arm
        Capsule(K["left_elbow"],      K["left_wrist"],       hw=9),   # forearm

        # ── Right arm ────────────────────────────────────────
        Capsule(K["right_shoulder"],  K["right_elbow"],      hw=11),
        Capsule(K["right_elbow"],     K["right_wrist"],      hw=9),

        # ── Left leg ─────────────────────────────────────────
        Capsule(K["left_hip"],        K["left_knee"],        hw=15),  # thigh
        Capsule(K["left_knee"],       K["left_ankle"],       hw=11),  # shin

        # ── Right leg ────────────────────────────────────────
        Capsule(K["right_hip"],       K["right_knee"],       hw=15),
        Capsule(K["right_knee"],      K["right_ankle"],      hw=11),
    ]


def render():
    cap = cv2.VideoCapture(DTL_VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 47)
    ret, frame = cap.read()
    cap.release()
    assert ret

    # Draw ALL capsules onto one overlay layer
    overlay = frame.copy()
    for seg in build_capsules():
        draw_capsule(overlay, seg, C_GREEN)

    # Single 60% blend — every capsule transparent equally, body shows through
    result = frame.copy()
    cv2.addWeighted(overlay, OPACITY, frame, 1.0 - OPACITY, 0, result)

    out = OUT / "body_capsules_DTL_impact_fr47.png"
    cv2.imwrite(str(out), result)
    print(f"Saved: {out}")


if __name__ == "__main__":
    render()
