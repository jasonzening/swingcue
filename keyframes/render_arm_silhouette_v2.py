#!/usr/bin/env python3
"""
render_arm_silhouette_v2.py

arm_line indicator — lead arm / chicken wing.
Golf knowledge basis: straight lead arm = shoulder-elbow-wrist collinear.
Chicken wing = elbow bends/flares outward from that line.

Render spec:
  - No yellow line (real arm IS current state)
  - Green semi-transparent arm silhouette at TARGET (straight arm position)
  - Arc arrow: from actual elbow → target elbow on straight line
  - Thickness: 14px (thinner than v1, "arm-like" not "line-like")
  - Opacity: 60% — single blend pass, body MUST show through
  - Rounded caps, no dots, no numbers
  - ArmSegment object with {current, target}
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

C_GREEN = (40, 210, 50)   # BGR
ARM_THICK = 14             # "arm-like" not too thin, not blocking
OPACITY   = 0.60           # single blend — body shows through


# ── Data object ───────────────────────────────────────────────────────────────
@dataclass
class ArmSegment:
    current: dict   # {"shoulder": (x,y), "elbow": (x,y), "wrist": (x,y)}
    target:  dict   # {"shoulder": (x,y), "elbow": (x,y), "wrist": (x,y)}


def make_arm(sh, el, wr) -> dict:
    return {"shoulder": sh, "elbow": el, "wrist": wr}


def straight_elbow(sh, wr, el_actual):
    """
    Return the elbow position on the straight line from sh to wr,
    at the same distance from sh as the actual elbow.
    This is the TARGET — fully straight arm, no chicken wing.
    """
    dx = wr[0] - sh[0]
    dy = wr[1] - sh[1]
    len_sw = math.hypot(dx, dy)
    len_upper = math.hypot(el_actual[0] - sh[0], el_actual[1] - sh[1])
    t = len_upper / len_sw if len_sw > 0 else 0.5
    return (int(sh[0] + t * dx), int(sh[1] + t * dy))


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_arm_silhouette(canvas, arm: dict, color, thickness):
    """
    Thick rounded arm shape: shoulder → elbow → wrist.
    cv2 LINE_AA + round caps via filled circles at endpoints.
    """
    sh = arm["shoulder"]
    el = arm["elbow"]
    wr = arm["wrist"]
    r  = thickness // 2

    cv2.line(canvas, sh, el, color, thickness, cv2.LINE_AA)
    cv2.line(canvas, el, wr, color, thickness, cv2.LINE_AA)
    # Round caps at shoulder and wrist (elbow is a joint — smooth it)
    cv2.circle(canvas, sh, r, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, el, r, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, wr, r, color, -1, cv2.LINE_AA)


def bezier(p0, ctrl, p2, n=50):
    pts = []
    for i in range(n + 1):
        t = i / n
        x = (1-t)**2 * p0[0] + 2*(1-t)*t * ctrl[0] + t**2 * p2[0]
        y = (1-t)**2 * p0[1] + 2*(1-t)*t * ctrl[1] + t**2 * p2[1]
        pts.append((int(x), int(y)))
    return pts


def draw_arc_arrow(canvas, p_start, p_end, color, line_thick=3, head_len=20):
    """
    Curved arrow from p_start to p_end.
    Control point offset perpendicular to the midpoint, outward from body.
    """
    mx = (p_start[0] + p_end[0]) // 2
    my = (p_start[1] + p_end[1]) // 2

    dx = p_end[0] - p_start[0]
    dy = p_end[1] - p_start[1]
    norm = math.hypot(dx, dy) or 1

    # Perpendicular — offset 80px outward (to the right in DTL view)
    ctrl = (int(mx + (dy / norm) * 80), int(my - (dx / norm) * 80))

    pts = bezier(p_start, ctrl, p_end, n=50)
    for i in range(len(pts) - 1):
        cv2.line(canvas, pts[i], pts[i+1], color, line_thick, cv2.LINE_AA)

    # Arrowhead at p_end
    if len(pts) >= 3:
        tip = pts[-1]
        d   = (pts[-1][0] - pts[-4][0], pts[-1][1] - pts[-4][1])
        norm2 = math.hypot(*d) or 1
        ux, uy = d[0]/norm2, d[1]/norm2
        ang = math.radians(25)
        ca, sa = math.cos(ang), math.sin(ang)
        w1 = (int(tip[0] - head_len*(ux*ca - uy*sa)),
              int(tip[1] - head_len*(uy*ca + ux*sa)))
        w2 = (int(tip[0] - head_len*(ux*ca + uy*sa)),
              int(tip[1] - head_len*(uy*ca - ux*sa)))
        pts_arr = np.array([tip, w1, w2], dtype=np.int32)
        cv2.fillPoly(canvas, [pts_arr], color, cv2.LINE_AA)


# ── Main render ───────────────────────────────────────────────────────────────
def render():
    cap = cv2.VideoCapture(DTL_VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 47)
    ret, frame = cap.read()
    cap.release()
    assert ret

    # Lead arm (left arm) at impact — RTMPose keypoints
    l_sh = (375, 447)
    l_el = (413, 543)   # actual elbow (slight chicken wing)
    l_wr = (435, 628)

    # Target: fully straight arm — elbow on sh→wr line
    l_el_straight = straight_elbow(l_sh, l_wr, l_el)
    # l_el_straight = (407, 545) — 6px inward from actual

    arm = ArmSegment(
        current=make_arm(l_sh, l_el,          l_wr),
        target= make_arm(l_sh, l_el_straight,  l_wr),
    )

    # Render everything onto ONE overlay, then single 60% blend
    overlay = frame.copy()

    # 1. Green silhouette at TARGET (straight arm)
    draw_arm_silhouette(overlay, arm.target, C_GREEN, ARM_THICK)

    # 2. Arc arrow: actual elbow → target elbow
    draw_arc_arrow(overlay,
                   arm.current["elbow"],   # (413, 543)
                   arm.target["elbow"],    # (407, 545)
                   C_GREEN,
                   line_thick=3,
                   head_len=20)

    # 3. Single blend — 60% opacity so body shows through
    result = frame.copy()
    cv2.addWeighted(overlay, OPACITY, frame, 1.0 - OPACITY, 0, result)

    out = OUT / "3_DTL_impact_arm_silhouette_v2.png"
    cv2.imwrite(str(out), result)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    render()
