#!/usr/bin/env python3
"""
render_arm_line.py
arm_line indicator per INDICATOR_ENGINEERING_SPEC §3:
  - Yellow = user current arm (shoulder->elbow->wrist, on real body)
  - Green  = target arm (same shoulder anchor, different angle)
  - Shared shoulder origin, lines fan out at different angles (NOT parallel)
  - Line width: 4px
  - Opacity: 60% (body shows through — single blend pass)
  - No joint dots, no angle labels
  - Full-brightness background
"""

import cv2
import numpy as np
from pathlib import Path
import math

OUT = Path("/home/jason/projects/swingcue-postest/keyframes/preview")
OUT.mkdir(parents=True, exist_ok=True)

DTL_VIDEO = "/home/jason/projects/swingcue-postest/input/test-dwontheline.mp4"

C_YELLOW = (0,   215, 255)
C_GREEN  = (0,   210,   0)
LINE_W   = 4
OPACITY  = 0.60


def load_frame(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, f = cap.read()
    cap.release()
    assert ret
    return f


def angle_point(origin, angle_deg_from_down, length):
    """
    Point at `length` px from origin.
    angle_deg_from_down: degrees clockwise from straight-down (+y).
    0 = straight down, +ve = toward right, -ve = toward left.
    """
    rad = math.radians(angle_deg_from_down)
    dx = length * math.sin(rad)
    dy = length * math.cos(rad)
    return (int(origin[0] + dx), int(origin[1] + dy))


# RTMPose impact frame 47 keypoints
DTL_IMP = {
    "r_sh": (398, 479),
    "r_el": (398, 582),
    "r_wr": (435, 657),
}


def render_arm_line():
    img = load_frame(DTL_VIDEO, 47)

    sh = DTL_IMP["r_sh"]
    el = DTL_IMP["r_el"]
    wr = DTL_IMP["r_wr"]

    # Compute current arm angles (from straight-down, clockwise)
    dx_upper = el[0] - sh[0]
    dy_upper = el[1] - sh[1]
    len_upper = math.hypot(dx_upper, dy_upper)
    cur_upper_ang = math.degrees(math.atan2(dx_upper, dy_upper))

    dx_lower = wr[0] - el[0]
    dy_lower = wr[1] - el[1]
    len_lower = math.hypot(dx_lower, dy_lower)
    cur_lower_ang = math.degrees(math.atan2(dx_lower, dy_lower))

    # Target angles: 12° more toward body (tighter, straighter arm path)
    tgt_upper_ang = cur_upper_ang - 12
    tgt_lower_ang = cur_lower_ang - 10

    el_tgt = angle_point(sh,     tgt_upper_ang, len_upper)
    wr_tgt = angle_point(el_tgt, tgt_lower_ang, len_lower)

    # ── Draw ALL lines onto ONE overlay layer, then blend ONCE ──────────────
    # This gives true 60% opacity — body shows through both lines equally.
    overlay = img.copy()

    # Green target lines (draw first — underneath yellow)
    cv2.line(overlay, sh,     el_tgt, C_GREEN,  LINE_W, cv2.LINE_AA)
    cv2.line(overlay, el_tgt, wr_tgt, C_GREEN,  LINE_W, cv2.LINE_AA)

    # Yellow current lines (draw on top — shared shoulder start point)
    cv2.line(overlay, sh, el, C_YELLOW, LINE_W, cv2.LINE_AA)
    cv2.line(overlay, el, wr, C_YELLOW, LINE_W, cv2.LINE_AA)

    # Single blend: 60% overlay on 100% original — body visible through lines
    result = img.copy()
    cv2.addWeighted(overlay, OPACITY, img, 1.0 - OPACITY, 0, result)

    out = OUT / "3_DTL_impact_arm_line.png"
    cv2.imwrite(str(out), result)
    print(f"Saved: {out}")


if __name__ == "__main__":
    render_arm_line()
    print("Done.")
