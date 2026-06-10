#!/usr/bin/env python3
"""
render_two_previews.py
Renders exactly two indicator previews, one indicator per image.

Rules:
  - No O/X badge
  - Yellow = user current actual position
  - Green  = correct target/reference position
  - Flat lines, no glow
  - White dot at key joint bend point
  - Full-brightness background, no dimming
"""

import cv2
import numpy as np
from pathlib import Path

OUT = Path("/home/jason/projects/swingcue-postest/keyframes/preview")
OUT.mkdir(parents=True, exist_ok=True)

DTL_VIDEO = "/home/jason/projects/swingcue-postest/input/test-dwontheline.mp4"

# Colors (BGR)
C_YELLOW = (0,   215, 255)   # current user posture
C_GREEN  = (0,   220,   0)   # target/correct position
C_WHITE  = (255, 255, 255)   # joint dot

LINE_W   = 5    # line thickness px
DOT_R    = 8    # white joint dot radius px


def load_frame(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, f = cap.read()
    cap.release()
    assert ret
    return f


def line(img, p1, p2, color, w=LINE_W):
    cv2.line(img, p1, p2, color, w, cv2.LINE_AA)


def dot(img, p, r=DOT_R):
    """White filled circle - marks the key bend joint."""
    cv2.circle(img, p, r, C_WHITE, -1, cv2.LINE_AA)
    # thin colored outline so dot stands out on bright backgrounds
    cv2.circle(img, p, r, (80, 80, 80), 1, cv2.LINE_AA)


def extend_line(p1, p2, extra=600):
    """Extend a line segment beyond both endpoints by `extra` pixels."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    L  = (dx**2 + dy**2) ** 0.5
    if L == 0:
        return p1, p2
    ux, uy = dx / L, dy / L
    far1 = (int(p1[0] - ux * extra), int(p1[1] - uy * extra))
    far2 = (int(p2[0] + ux * extra), int(p2[1] + uy * extra))
    return far1, far2


# ── Keypoints ─────────────────────────────────────────────────────────────────
# Frame 10 = address
A = {
    "l_sh":  (387, 473), "r_sh":  (401, 470),
    "l_el":  (386, 583), "r_el":  (404, 580),
    "l_wr":  (388, 671), "r_wr":  (405, 684),
    "l_hip": (263, 608), "r_hip": (270, 610),
}
# Frame 47 = impact
I = {
    "l_sh":  (375, 447), "r_sh":  (398, 479),
    "l_el":  (413, 543), "r_el":  (398, 582),
    "l_wr":  (435, 628), "r_wr":  (435, 657),
    "l_hip": (267, 607), "r_hip": (310, 612),
}


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE 1: Downtheline IMPACT — Wrist V
# Yellow = current (impact frame right arm)
# Green  = target  (address frame right arm — where hands should return)
# White dot at elbow bend vertex (both lines)
# ═══════════════════════════════════════════════════════════════════════════════
def render_wrist_v():
    img = load_frame(DTL_VIDEO, 47)

    # Yellow line (current at impact): r_shoulder → r_elbow → r_wrist
    line(img, I["r_sh"],  I["r_el"], C_YELLOW)
    line(img, I["r_el"],  I["r_wr"], C_YELLOW)
    dot(img, I["r_el"])   # white dot at elbow bend

    # Green line (target): same structure but at address coords
    line(img, A["r_sh"],  A["r_el"], C_GREEN)
    line(img, A["r_el"],  A["r_wr"], C_GREEN)
    dot(img, A["r_el"])   # white dot at address elbow

    out = OUT / "1_DTL_impact_wrist_V.png"
    cv2.imwrite(str(out), img)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE 2: Downtheline ADDRESS — Posture / Spine Line
# Yellow = current spine tilt (impact frame)
# Green  = address reference spine tilt (address frame)
# Extended full-length lines through head and ground
# ═══════════════════════════════════════════════════════════════════════════════
def render_posture_line():
    img = load_frame(DTL_VIDEO, 47)   # current = impact frame

    # Midpoints
    sh_cur  = ((I["l_sh"][0]+I["r_sh"][0])//2, (I["l_sh"][1]+I["r_sh"][1])//2)
    hip_cur = ((I["l_hip"][0]+I["r_hip"][0])//2, (I["l_hip"][1]+I["r_hip"][1])//2)

    sh_ref  = ((A["l_sh"][0]+A["r_sh"][0])//2, (A["l_sh"][1]+A["r_sh"][1])//2)
    hip_ref = ((A["l_hip"][0]+A["r_hip"][0])//2, (A["l_hip"][1]+A["r_hip"][1])//2)

    # Extend each line through the full body height
    t1_y, b1_y = extend_line(sh_cur, hip_cur, extra=500)
    t2_y, b2_y = extend_line(sh_ref, hip_ref, extra=500)

    # Green reference line first (drawn under yellow)
    line(img, t2_y, b2_y, C_GREEN, LINE_W)

    # Yellow current line on top
    line(img, t1_y, b1_y, C_YELLOW, LINE_W)

    out = OUT / "2_DTL_impact_posture_line.png"
    cv2.imwrite(str(out), img)
    print(f"Saved: {out}")


if __name__ == "__main__":
    render_wrist_v()
    render_posture_line()
    print("Done.")
