#!/usr/bin/env python3
"""
render_arm_silhouette_v4.py

arm_line / lead arm straight indicator.

ROOT CAUSE of v2/v3 failure:
  This golfer's lead arm IS already ~173 degrees at impact (nearly straight).
  Geometrically the "target" elbow is only 6px from actual → they overlap.

FIX for "示意画即可":
  DTL view: a straight arm hangs more VERTICALLY from the shoulder.
  A chicken-wing arm flares OUT (more horizontal).
  So the target = rotate the UPPER ARM vector 15° toward vertical,
  keeping lower arm length the same.
  This creates a visually clear, correctly-directed separation:
    actual elbow → outward (right)
    target elbow → more inward/downward (correct straight arm)
  Arrow and silhouette are now clearly separate from real arm.
"""

import cv2, numpy as np, math
from pathlib import Path
from dataclasses import dataclass

OUT = Path("/home/jason/projects/swingcue-postest/keyframes/preview")
OUT.mkdir(parents=True, exist_ok=True)
DTL_VIDEO = "/home/jason/projects/swingcue-postest/input/test-dwontheline.mp4"

C_GREEN = (40, 210, 50)
ARM_THICK = 14
OPACITY   = 0.55


@dataclass
class ArmSegment:
    current: dict   # {"shoulder": (x,y), "elbow": (x,y), "wrist": (x,y)}
    target:  dict


def pt(x, y): return (int(round(x)), int(round(y)))


def rotate_vec(dx, dy, deg):
    """Rotate vector (dx,dy) by deg degrees clockwise."""
    r = math.radians(deg)
    return (dx*math.cos(r) + dy*math.sin(r),
           -dx*math.sin(r) + dy*math.cos(r))


def target_arm(sh, el, wr, rotate_upper_deg=-15):
    """
    Compute target arm for straight-arm visual.
    Rotate upper arm (sh→el) by rotate_upper_deg toward vertical.
    Then place lower arm from new elbow toward original wrist direction.
    rotate_upper_deg < 0 = rotate counter-clockwise = more vertical in DTL.
    """
    # Upper arm vector and length
    ux = el[0] - sh[0]
    uy = el[1] - sh[1]
    ul = math.hypot(ux, uy)

    # Rotate upper arm toward vertical (more downward, less outward)
    ux_t, uy_t = rotate_vec(ux, uy, rotate_upper_deg)
    norm = math.hypot(ux_t, uy_t) or 1
    el_t = pt(sh[0] + ux_t/norm * ul, sh[1] + uy_t/norm * ul)

    # Lower arm: keep same direction as sh→wr (overall arm direction),
    # length = actual lower arm length
    lx = wr[0] - el[0]
    ly = wr[1] - el[1]
    ll = math.hypot(lx, ly)
    # Align lower arm with the same overall sh→wr direction but from new elbow
    sw_x = wr[0] - sh[0]
    sw_y = wr[1] - sh[1]
    sw_l = math.hypot(sw_x, sw_y) or 1
    wr_t = pt(el_t[0] + sw_x/sw_l * ll, el_t[1] + sw_y/sw_l * ll)

    return el_t, wr_t


def draw_arm(canvas, arm: dict, color, thick):
    sh = arm["shoulder"]; el = arm["elbow"]; wr = arm["wrist"]
    r  = thick // 2
    cv2.line(canvas, sh, el, color, thick, cv2.LINE_AA)
    cv2.line(canvas, el, wr, color, thick, cv2.LINE_AA)
    for p in (sh, el, wr):
        cv2.circle(canvas, p, r, color, -1, cv2.LINE_AA)


def draw_arc_arrow(canvas, src, dst, color, lw=3, head=22):
    mx = (src[0]+dst[0])//2; my = (src[1]+dst[1])//2
    dx = dst[0]-src[0];      dy = dst[1]-src[1]
    n  = math.hypot(dx, dy) or 1
    ctrl = (int(mx + (dy/n)*65), int(my - (dx/n)*65))

    def bezier(p0, c, p2, steps=50):
        return [(int((1-t)**2*p0[0]+2*(1-t)*t*c[0]+t**2*p2[0]),
                 int((1-t)**2*p0[1]+2*(1-t)*t*c[1]+t**2*p2[1]))
                for t in [i/steps for i in range(steps+1)]]

    pts = bezier(src, ctrl, dst)
    for i in range(len(pts)-1):
        cv2.line(canvas, pts[i], pts[i+1], color, lw, cv2.LINE_AA)

    tip = pts[-1]
    d   = (pts[-1][0]-pts[-4][0], pts[-1][1]-pts[-4][1])
    nm  = math.hypot(*d) or 1
    ux, uy = d[0]/nm, d[1]/nm
    a  = math.radians(26)
    ca, sa = math.cos(a), math.sin(a)
    w1 = (int(tip[0]-head*(ux*ca-uy*sa)), int(tip[1]-head*(uy*ca+ux*sa)))
    w2 = (int(tip[0]-head*(ux*ca+uy*sa)), int(tip[1]-head*(uy*ca-ux*sa)))
    cv2.fillPoly(canvas, [np.array([tip,w1,w2], np.int32)], color, cv2.LINE_AA)


def render():
    cap = cv2.VideoCapture(DTL_VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 47)
    ret, frame = cap.read()
    cap.release()
    assert ret

    # Lead arm (left arm), impact frame 47
    l_sh = pt(375, 447)
    l_el = pt(413, 543)
    l_wr = pt(435, 628)

    # Target: rotate upper arm 25° counter-clockwise (clearly visible separation)
    l_el_t, l_wr_t = target_arm(l_sh, l_el, l_wr, rotate_upper_deg=-25)

    arm = ArmSegment(
        current={"shoulder": l_sh, "elbow": l_el,   "wrist": l_wr},
        target= {"shoulder": l_sh, "elbow": l_el_t, "wrist": l_wr_t},
    )

    print(f"Actual  elbow: {l_el}")
    print(f"Target  elbow: {l_el_t}  (15deg rotation toward vertical)")
    print(f"Separation: {int(math.hypot(l_el[0]-l_el_t[0], l_el[1]-l_el_t[1]))}px")

    overlay = frame.copy()

    # Green target silhouette (straight arm)
    draw_arm(overlay, arm.target, C_GREEN, ARM_THICK)

    # Arc arrow: actual elbow → target elbow
    draw_arc_arrow(overlay, arm.current["elbow"], arm.target["elbow"], C_GREEN)

    # Single 55% blend
    result = frame.copy()
    cv2.addWeighted(overlay, OPACITY, frame, 1.0 - OPACITY, 0, result)

    out = OUT / "3_DTL_impact_arm_silhouette_v4.png"
    cv2.imwrite(str(out), result)
    print(f"Saved: {out}")


if __name__ == "__main__":
    render()
