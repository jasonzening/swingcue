#!/usr/bin/env python3
"""
render_arm_silhouette.py
arm_line indicator — new scheme per spec:
  - No yellow line (real arm is current state)
  - Green semi-transparent arm SILHOUETTE at target position
    (thick rounded shape like a real arm, not a thin line)
  - One arc arrow: current arm → target silhouette
  - Silhouette opacity 60%, rounded caps, no dots, no numbers
  - Engineering: ArmSegment object with {current, target} coords
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

# Colors (BGR)
C_GREEN = (30, 200, 40)
C_ARROW = (30, 200, 40)   # arrow same green
C_WHITE = (255, 255, 255)

ARM_THICKNESS  = 22    # px — thick enough to look like an arm, not a line
OPACITY        = 0.60  # silhouette opacity
ARROW_THICK    = 3
ARROW_HEAD_LEN = 22


# ── Data structure ────────────────────────────────────────────────────────────
@dataclass
class ArmSegment:
    """
    Arm represented as shoulder/elbow/wrist for current and target positions.
    Shoulder is the shared anchor — same point for both.
    """
    current: dict   # {"shoulder": (x,y), "elbow": (x,y), "wrist": (x,y)}
    target:  dict   # {"shoulder": (x,y), "elbow": (x,y), "wrist": (x,y)}


# ── Geometry helpers ──────────────────────────────────────────────────────────
def angle_point(origin: Tuple, angle_from_down_deg: float, length: float) -> Tuple:
    """Point at `length` from origin, angle measured clockwise from +y."""
    rad = math.radians(angle_from_down_deg)
    return (int(origin[0] + length * math.sin(rad)),
            int(origin[1] + length * math.cos(rad)))


def bezier_points(p0, p1, p2, n=60):
    """Quadratic Bezier from p0→p2 through control point p1, n samples."""
    pts = []
    for i in range(n + 1):
        t = i / n
        x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
        y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
        pts.append((int(x), int(y)))
    return pts


def draw_arrowhead(img, tip, direction_vec, length=ARROW_HEAD_LEN,
                   color=C_ARROW, thickness=ARROW_THICK):
    """Draw filled arrowhead at `tip` pointing in `direction_vec`."""
    dx, dy = direction_vec
    norm = math.hypot(dx, dy)
    if norm == 0:
        return
    ux, uy = dx / norm, dy / norm

    # Two wing points
    angle = math.radians(28)
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    # Right wing
    wx1 = int(tip[0] - length * (ux * cos_a - uy * sin_a))
    wy1 = int(tip[1] - length * (uy * cos_a + ux * sin_a))
    # Left wing
    wx2 = int(tip[0] - length * (ux * cos_a + uy * sin_a))
    wy2 = int(tip[1] - length * (uy * cos_a - ux * sin_a))

    pts = np.array([[tip[0], tip[1]], [wx1, wy1], [wx2, wy2]], dtype=np.int32)
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)


# ── Silhouette renderer ───────────────────────────────────────────────────────
def draw_arm_silhouette(canvas: np.ndarray, arm_pts: list, color: tuple,
                        thickness: int = ARM_THICKNESS):
    """
    Draw a thick rounded arm shape through [shoulder, elbow, wrist].
    Rounded caps at each end, round joints at elbow.
    Done on `canvas` in-place (caller handles opacity blend).
    """
    sh, el, wr = arm_pts

    # Upper arm segment
    cv2.line(canvas, sh, el, color, thickness, cv2.LINE_AA)
    # Lower arm segment
    cv2.line(canvas, el, wr, color, thickness, cv2.LINE_AA)

    # Round joints: filled circles at each key point
    r = thickness // 2
    cv2.circle(canvas, sh, r, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, el, r, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, wr, r, color, -1, cv2.LINE_AA)


# ── Arc arrow renderer ────────────────────────────────────────────────────────
def draw_arc_arrow(canvas: np.ndarray, arm_current: list, arm_target: list,
                   color: tuple = C_ARROW):
    """
    Curved arrow from current arm elbow/wrist to target arm.
    Uses a quadratic Bezier — control point offset perpendicular outward.
    """
    # Use elbow as the "representative" point for the arm position
    cur_el = arm_current[1]   # current elbow
    tgt_el = arm_target[1]    # target elbow

    # Midpoint between the two elbows
    mx = (cur_el[0] + tgt_el[0]) // 2
    my = (cur_el[1] + tgt_el[1]) // 2

    # Perpendicular offset outward (away from body — to the right in DTL view)
    dx = tgt_el[0] - cur_el[0]
    dy = tgt_el[1] - cur_el[1]
    # Perpendicular: (-dy, dx) normalized, scale by 60px to bow the arrow
    norm = math.hypot(dx, dy) or 1
    perp_x = int(mx + (-dy / norm) * 70)
    perp_y = int(my + ( dx / norm) * 70)
    ctrl = (perp_x, perp_y)

    # Bezier points from current elbow to target elbow
    pts = bezier_points(cur_el, ctrl, tgt_el, n=50)

    # Draw the curve (thin line)
    for i in range(len(pts) - 1):
        cv2.line(canvas, pts[i], pts[i+1], color, ARROW_THICK, cv2.LINE_AA)

    # Arrowhead at the tip (tgt_el), direction = last segment of bezier
    if len(pts) >= 2:
        tip_dir = (pts[-1][0] - pts[-3][0], pts[-1][1] - pts[-3][1])
        draw_arrowhead(canvas, tgt_el, tip_dir, color=color)


# ── Main render ───────────────────────────────────────────────────────────────
def render():
    cap = cv2.VideoCapture(DTL_VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 47)
    ret, frame = cap.read()
    cap.release()
    assert ret

    # Impact frame keypoints (right arm, from RTMPose)
    sh = (398, 479)
    el = (398, 582)
    wr = (435, 657)

    # Target position: same shoulder, rotate arm ~14° more toward body
    # (illustrative — not calculated from biomechanics)
    dx_u = el[0] - sh[0]; dy_u = el[1] - sh[1]
    len_u = math.hypot(dx_u, dy_u)
    ang_u = math.degrees(math.atan2(dx_u, dy_u))

    dx_l = wr[0] - el[0]; dy_l = wr[1] - el[1]
    len_l = math.hypot(dx_l, dy_l)
    ang_l = math.degrees(math.atan2(dx_l, dy_l))

    el_t = angle_point(sh,    ang_u - 14, len_u)
    wr_t = angle_point(el_t,  ang_l - 11, len_l)

    # ArmSegment object
    arm = ArmSegment(
        current={"shoulder": sh, "elbow": el,   "wrist": wr},
        target= {"shoulder": sh, "elbow": el_t, "wrist": wr_t},
    )

    # ── Compose ───────────────────────────────────────────────────────────────
    # Layer 1: silhouette + arrow on transparent canvas, then blend
    overlay = frame.copy()

    # Green silhouette at TARGET position (thick rounded arm shape)
    tgt_pts = [arm.target["shoulder"], arm.target["elbow"], arm.target["wrist"]]
    draw_arm_silhouette(overlay, tgt_pts, C_GREEN, thickness=ARM_THICKNESS)

    # Arc arrow: from current arm → target arm
    cur_pts = [arm.current["shoulder"], arm.current["elbow"], arm.current["wrist"]]
    draw_arc_arrow(overlay, cur_pts, tgt_pts, color=C_ARROW)

    # Single 60% blend: silhouette + arrow both at 60% opacity
    result = frame.copy()
    cv2.addWeighted(overlay, OPACITY, frame, 1.0 - OPACITY, 0, result)

    out = OUT / "3_DTL_impact_arm_silhouette.png"
    cv2.imwrite(str(out), result)
    print(f"Saved: {out}")


if __name__ == "__main__":
    render()
