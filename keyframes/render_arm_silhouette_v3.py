#!/usr/bin/env python3
"""
render_arm_silhouette_v3.py

arm_line / lead arm straight indicator.

KEY FIX from v2:
  - Real arm is nearly straight (7° deviation = 6px) → target silhouette
    was invisible because it overlapped the actual arm.
  - Per spec "示意画即可": we place the target silhouette at a clearly
    VISIBLE illustrative position. The instruction is directional, not exact.
  - Target = arm positioned so elbow sits ON the sh→wr straight line
    AND we shift it enough inward to be clearly visible (50px offset).
  - Arrow now spans a real visible distance.

Render rules:
  - No yellow line (real arm is current state, visible in video)
  - Green semi-transparent silhouette at target (straight arm = sh-el-wr collinear)
  - Arc arrow: actual elbow → target elbow  (large enough to see)
  - Thickness 14px, opacity 45% (more transparent than 60% for clearer see-through)
  - Single blend pass, rounded caps, no dots, no numbers
"""

import cv2, numpy as np, math
from pathlib import Path
from dataclasses import dataclass

OUT = Path("/home/jason/projects/swingcue-postest/keyframes/preview")
OUT.mkdir(parents=True, exist_ok=True)
DTL_VIDEO = "/home/jason/projects/swingcue-postest/input/test-dwontheline.mp4"

C_GREEN = (40, 210, 50)   # BGR
ARM_THICK = 14
OPACITY   = 0.45           # lowered for obvious see-through


@dataclass
class ArmSegment:
    current: dict   # shoulder / elbow / wrist
    target:  dict


def pt(x, y): return (int(x), int(y))


def straight_elbow_target(sh, wr, el_actual, visual_offset=50):
    """
    Returns an elbow position that is ON the sh→wr line AND
    shifted inward by visual_offset px to be clearly visible
    against the actual arm.
    """
    dx = wr[0] - sh[0]
    dy = wr[1] - sh[1]
    dist = math.hypot(dx, dy) or 1
    t = math.hypot(el_actual[0]-sh[0], el_actual[1]-sh[1]) / dist

    # Ideal point on straight line
    on_line = pt(sh[0] + t*dx, sh[1] + t*dy)

    # Push it further inward (toward body center, leftward in DTL view)
    # Perpendicular to sh→wr: (-dy, dx) normalised
    px, py = -dy/dist, dx/dist
    return pt(on_line[0] + px*visual_offset, on_line[1] + py*visual_offset)


def draw_arm(canvas, arm: dict, color, thickness):
    sh, el, wr = arm["shoulder"], arm["elbow"], arm["wrist"]
    r = thickness // 2
    cv2.line(canvas, sh, el, color, thickness, cv2.LINE_AA)
    cv2.line(canvas, el, wr, color, thickness, cv2.LINE_AA)
    for p in (sh, el, wr):
        cv2.circle(canvas, p, r, color, -1, cv2.LINE_AA)


def bezier(p0, ctrl, p2, n=50):
    pts = []
    for i in range(n+1):
        t = i/n
        x = (1-t)**2*p0[0] + 2*(1-t)*t*ctrl[0] + t**2*p2[0]
        y = (1-t)**2*p0[1] + 2*(1-t)*t*ctrl[1] + t**2*p2[1]
        pts.append((int(x), int(y)))
    return pts


def draw_arc_arrow(canvas, src, dst, color, lw=3, head=22):
    mx = (src[0]+dst[0])//2
    my = (src[1]+dst[1])//2
    dx = dst[0]-src[0]; dy = dst[1]-src[1]
    n = math.hypot(dx, dy) or 1
    # Control point: perpendicular outward 70px
    ctrl = (int(mx + (dy/n)*70), int(my - (dx/n)*70))

    pts = bezier(src, ctrl, dst)
    for i in range(len(pts)-1):
        cv2.line(canvas, pts[i], pts[i+1], color, lw, cv2.LINE_AA)

    # Arrowhead
    tip = pts[-1]
    d   = (pts[-1][0]-pts[-4][0], pts[-1][1]-pts[-4][1])
    nm  = math.hypot(*d) or 1
    ux, uy = d[0]/nm, d[1]/nm
    ang = math.radians(26)
    ca, sa = math.cos(ang), math.sin(ang)
    w1 = (int(tip[0]-head*(ux*ca - uy*sa)), int(tip[1]-head*(uy*ca + ux*sa)))
    w2 = (int(tip[0]-head*(ux*ca + uy*sa)), int(tip[1]-head*(uy*ca - ux*sa)))
    cv2.fillPoly(canvas, [np.array([tip, w1, w2], dtype=np.int32)], color, cv2.LINE_AA)


def render():
    cap = cv2.VideoCapture(DTL_VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 47)
    ret, frame = cap.read()
    cap.release()
    assert ret

    # Lead arm (left arm) keypoints — impact frame 47
    l_sh = pt(375, 447)
    l_el = pt(413, 543)   # actual — slight chicken wing
    l_wr = pt(435, 628)

    # Target elbow: on the sh→wr straight line, shifted 50px inward for visibility
    l_el_tgt = straight_elbow_target(l_sh, l_wr, l_el, visual_offset=50)

    arm = ArmSegment(
        current={"shoulder": l_sh, "elbow": l_el,     "wrist": l_wr},
        target= {"shoulder": l_sh, "elbow": l_el_tgt, "wrist": l_wr},
    )

    overlay = frame.copy()

    # 1. Green target silhouette (straight arm)
    draw_arm(overlay, arm.target, C_GREEN, ARM_THICK)

    # 2. Arc arrow: actual elbow → target elbow
    draw_arc_arrow(overlay, arm.current["elbow"], arm.target["elbow"], C_GREEN)

    # 3. Single blend — 45% opacity
    result = frame.copy()
    cv2.addWeighted(overlay, OPACITY, frame, 1.0 - OPACITY, 0, result)

    out = OUT / "3_DTL_impact_arm_silhouette_v3.png"
    cv2.imwrite(str(out), result)
    print(f"Saved: {out}")
    print(f"Elbow actual:  {l_el}")
    print(f"Elbow target:  {l_el_tgt}  (offset {50}px inward)")


if __name__ == "__main__":
    render()
