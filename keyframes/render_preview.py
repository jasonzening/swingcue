#!/usr/bin/env python3
"""
render_preview.py
Generate 4 indicator preview PNGs onto real video frames.

Outputs (all to keyframes/preview/):
  DTL_address_posture_line.png   — 侧面 ADDRESS: posture line + ghost
  DTL_impact_wrist_v.png         — 侧面 IMPACT:  elbow-wrist V + angle arc
  FO_address_center_axis.png     — 正面 ADDRESS: center axis + shoulder/hip discs
  FO_impact_rotation_disc.png    — 正面 IMPACT:  rotation disc delta vs address
"""

import cv2
import json
import math
import numpy as np
from pathlib import Path

OUT_DIR = Path("/home/jason/projects/swingcue-postest/keyframes/preview")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── color palette (BGR) ──────────────────────────────────────────────────────
# Three-color system: green=goal/ghost, yellow=current, red=deviation
C_GHOST   = (80,  210, 80)    # green — address ghost / target
C_CURRENT = (30,  210, 230)   # yellow (rendered as warm gold-cyan) — current state
C_DEV     = (50,   70, 230)   # red — deviation marker
C_WHITE   = (255, 255, 255)
C_BLACK   = (0,   0,   0)
C_GRAY    = (160, 160, 160)
C_LABEL_BG = (20,  20,  20)

DIMMER_ALPHA = 0.45           # frame darkening amount


# ── helpers ──────────────────────────────────────────────────────────────────

def load_frame(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    assert ret, f"Cannot read frame {frame_idx} from {video_path}"
    return frame


def dim_frame(frame: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Darken the frame so overlays pop."""
    dark = frame.copy()
    dark[:] = (dark * (1 - alpha)).astype(np.uint8)
    return dark


def pt(kps: dict, name: str):
    """Return (x, y) int tuple from keypoint dict."""
    v = kps[name]
    return (int(round(v[0])), int(round(v[1])))


def midpt(a, b):
    return (int((a[0]+b[0])//2), int((a[1]+b[1])//2))


def draw_dashed_line(img, p1, p2, color, thickness=2, dash=14, gap=8):
    """Draw a dashed line between p1 and p2."""
    x1,y1 = p1; x2,y2 = p2
    dist = math.hypot(x2-x1, y2-y1)
    if dist == 0:
        return
    steps = int(dist / (dash+gap))
    dx = (x2-x1)/dist; dy = (y2-y1)/dist
    for i in range(steps+1):
        s = i*(dash+gap)
        e = min(s+dash, dist)
        ps = (int(x1+dx*s), int(y1+dy*s))
        pe = (int(x1+dx*e), int(y1+dy*e))
        cv2.line(img, ps, pe, color, thickness, cv2.LINE_AA)


def draw_angle_arc(img, center, p1, p2, color, radius=40, thickness=2):
    """Draw an arc showing the angle at 'center' between p1 and p2."""
    def ang(pt):
        return math.degrees(math.atan2(pt[1]-center[1], pt[0]-center[0]))
    a1 = ang(p1); a2 = ang(p2)
    if abs(a2-a1) > 180:
        if a2 > a1: a1 += 360
        else:       a2 += 360
    cv2.ellipse(img, center, (radius, radius), 0,
                min(a1,a2), max(a1,a2), color, thickness, cv2.LINE_AA)


def draw_label(img, text, pos, color=C_WHITE, bg=C_LABEL_BG, font_scale=0.65):
    """Draw a label with dark background."""
    font = cv2.FONT_HERSHEY_DUPLEX
    thick = 1
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, thick)
    x, y = pos
    pad = 5
    cv2.rectangle(img, (x-pad, y-th-pad), (x+tw+pad, y+bl+pad), bg, -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thick, cv2.LINE_AA)


def add_ui_frame(img, title: str, indicator_name: str, frame_label: str):
    """Add a clean gray UI border/header to the image."""
    h, w = img.shape[:2]
    border = 44
    canvas = np.full((h + border, w, 3), 28, dtype=np.uint8)
    canvas[border:, :] = img

    # header bar gradient-like
    cv2.rectangle(canvas, (0,0), (w, border), (40,40,40), -1)
    cv2.line(canvas, (0, border-1), (w, border-1), (70,70,70), 1)

    font = cv2.FONT_HERSHEY_DUPLEX

    # LEFT: short title (ASCII safe)
    cv2.putText(canvas, title, (12, 29), font, 0.58, C_WHITE, 1, cv2.LINE_AA)

    # CENTER: indicator name - keep it short so it fits
    short_ind = indicator_name[:52]   # truncate if too long
    (tw,_),_ = cv2.getTextSize(short_ind, font, 0.52, 1)
    cv2.putText(canvas, short_ind, (w//2 - tw//2, 29),
                font, 0.52, C_GHOST, 1, cv2.LINE_AA)

    # RIGHT: frame label
    (fw,_),_ = cv2.getTextSize(frame_label, font, 0.52, 1)
    cv2.putText(canvas, frame_label, (w - fw - 12, 29),
                font, 0.52, C_GRAY, 1, cv2.LINE_AA)

    return canvas


def save(img, name: str):
    path = OUT_DIR / name
    cv2.imwrite(str(path), img)
    print(f"  Saved: {path}")


# ── keypoint data (pre-computed from JSON) ───────────────────────────────────

DTL_ADDR = {"nose": [473.9, 471.7, 0.9], "left_eye": [481.0, 456.6, 0.8], "right_eye": [476.9, 453.6, 1.0], "left_ear": [429.6, 445.5, 0.4], "right_ear": [450.8, 432.4, 0.9], "left_shoulder": [387.4, 472.7, 0.7], "right_shoulder": [400.5, 469.6, 0.8], "left_elbow": [386.4, 583.4, 0.7], "right_elbow": [403.5, 580.3, 0.8], "left_wrist": [388.4, 670.9, 0.8], "right_wrist": [404.5, 684.0, 0.8], "left_hip": [262.6, 607.5, 0.7], "right_hip": [269.7, 609.5, 0.7], "left_knee": [313.9, 769.5, 0.7], "right_knee": [328.0, 769.5, 0.8], "left_ankle": [285.8, 916.4, 0.7], "right_ankle": [298.8, 952.6, 0.9]}
DTL_IMP  = {"nose": [463.3, 455.4, 0.9], "left_eye": [471.5, 439.0, 0.8], "right_eye": [463.3, 436.9, 0.9], "left_ear": [434.5, 421.5, 0.4], "right_ear": [433.5, 416.4, 0.8], "left_shoulder": [374.9, 447.2, 0.7], "right_shoulder": [397.5, 479.1, 0.8], "left_elbow": [412.9, 542.8, 0.6], "right_elbow": [397.5, 581.9, 0.7], "left_wrist": [434.5, 628.2, 0.6], "right_wrist": [434.5, 657.0, 0.8], "left_hip": [266.9, 606.6, 0.7], "right_hip": [310.1, 611.7, 0.7], "left_knee": [316.2, 758.8, 0.8], "right_knee": [366.6, 766.0, 0.8], "left_ankle": [290.5, 896.6, 0.8], "right_ankle": [332.7, 920.2, 0.8]}
FO_ADDR  = {"nose": [444.4, 439.0, 1.0], "left_eye": [457.4, 426.0, 1.0], "right_eye": [432.6, 427.1, 1.0], "left_ear": [482.2, 414.1, 0.8], "right_ear": [415.3, 420.6, 0.7], "left_shoulder": [527.5, 473.5, 0.9], "right_shoulder": [390.5, 484.3, 0.9], "left_elbow": [490.9, 574.9, 0.9], "right_elbow": [416.4, 583.6, 0.9], "left_wrist": [463.9, 665.6, 0.9], "right_wrist": [431.5, 682.8, 0.9], "left_hip": [494.1, 634.3, 0.8], "right_hip": [413.1, 632.1, 0.8], "left_knee": [524.3, 754.0, 0.8], "right_knee": [372.1, 750.8, 0.8], "left_ankle": [538.3, 909.4, 0.9], "right_ankle": [350.6, 897.6, 0.9]}
FO_IMP   = {"nose": [427.8, 443.0, 0.8], "left_eye": [436.7, 429.1, 0.9], "right_eye": [416.5, 431.6, 0.9], "left_ear": [467.0, 413.9, 0.6], "right_ear": [405.1, 434.1, 0.5], "left_shoulder": [528.9, 455.6, 0.9], "right_shoulder": [406.4, 485.9, 0.9], "left_elbow": [491.0, 557.9, 0.8], "right_elbow": [426.6, 579.4, 0.9], "left_wrist": [455.6, 641.2, 0.8], "right_wrist": [430.4, 653.9, 0.8], "left_hip": [541.5, 619.8, 0.8], "right_hip": [468.2, 626.1, 0.7], "left_knee": [568.0, 767.5, 0.8], "right_knee": [416.5, 763.7, 0.8], "left_ankle": [549.0, 908.9, 0.9], "right_ankle": [350.8, 888.7, 0.9]}

DTL_VIDEO = "/home/jason/projects/swingcue-postest/input/test-dwontheline.mp4"
FO_VIDEO  = "/home/jason/projects/swingcue-postest/input/test-faceon.mp4"


# ── RENDER 1: 侧面 ADDRESS — Posture Line + Ghost ────────────────────────────
def render_dtl_posture_address():
    kps = DTL_ADDR
    frame = dim_frame(load_frame(DTL_VIDEO, 10))
    overlay = frame.copy()

    # Posture line = shoulder mid → hip mid (approximate spinal tilt)
    sh_mid  = midpt(pt(kps, "left_shoulder"),  pt(kps, "right_shoulder"))
    hip_mid = midpt(pt(kps, "left_hip"),        pt(kps, "right_hip"))

    # Extend line upward through head and downward through feet (posture guideline)
    dx = hip_mid[0] - sh_mid[0]
    dy = hip_mid[1] - sh_mid[1]
    length = math.hypot(dx, dy)
    ndx, ndy = dx/length, dy/length

    # Extend ±2× torso height
    top_pt    = (int(sh_mid[0]  - ndx*400), int(sh_mid[1]  - ndy*400))
    bottom_pt = (int(hip_mid[0] + ndx*300), int(hip_mid[1] + ndy*300))

    # Ghost line (address reference) — green dashed, semi-transparent
    ghost_layer = frame.copy()
    draw_dashed_line(ghost_layer, top_pt, bottom_pt, C_GHOST, thickness=3)
    cv2.addWeighted(ghost_layer, 0.7, overlay, 0.3, 0, overlay)

    # Current posture line — yellow solid (at address this overlaps ghost)
    # Slightly offset to show both
    cv2.line(overlay, sh_mid, hip_mid, C_CURRENT, 3, cv2.LINE_AA)

    # Anchor dots
    cv2.circle(overlay, sh_mid,  8, C_CURRENT, -1)
    cv2.circle(overlay, hip_mid, 8, C_CURRENT, -1)
    cv2.circle(overlay, sh_mid,  8, C_WHITE,    2)
    cv2.circle(overlay, hip_mid, 8, C_WHITE,    2)

    # Spine tilt from vertical (forward lean angle, typically 30–50° for golf)
    angle_deg = math.degrees(math.atan2(abs(dx), dy))
    draw_label(overlay, f"Spine tilt: {angle_deg:.0f}deg", (sh_mid[0]+15, sh_mid[1]-20), C_CURRENT)
    draw_label(overlay, "ADDRESS baseline", (top_pt[0]+10, top_pt[1]+40), C_GHOST)

    # Skeleton keypoints for context (dim dots)
    for name in ("left_shoulder","right_shoulder","left_hip","right_hip","nose"):
        p = pt(kps, name)
        cv2.circle(overlay, p, 5, C_GRAY, -1)

    result = add_ui_frame(overlay,
        title="Downtheline — ADDRESS",
        indicator_name="POSTURE LINE  (yellow=current  green-dashed=ghost)",
        frame_label="Frame 10")
    save(result, "DTL_address_posture_line.png")


# ── RENDER 2: 侧面 IMPACT — Wrist V + angle arc ──────────────────────────────
def render_dtl_wrist_v_impact():
    kps = DTL_IMP
    addr = DTL_ADDR
    frame = dim_frame(load_frame(DTL_VIDEO, 47))
    overlay = frame.copy()

    # Ghost wrist position from address (green dashed)
    addr_lw = pt(addr, "left_wrist"); addr_rw = pt(addr, "right_wrist")
    addr_le = pt(addr, "left_elbow"); addr_re = pt(addr, "right_elbow")
    addr_wm = midpt(addr_lw, addr_rw)
    addr_em = midpt(addr_le, addr_re)

    ghost_layer = frame.copy()
    draw_dashed_line(ghost_layer, addr_em, addr_wm, C_GHOST, thickness=2)
    cv2.circle(ghost_layer, addr_wm, 8, C_GHOST, -1)
    cv2.addWeighted(ghost_layer, 0.55, overlay, 0.45, 0, overlay)

    # Current: V shape = elbow-mid → wrist-mid (both arms)
    lw = pt(kps, "left_wrist");  rw = pt(kps, "right_wrist")
    le = pt(kps, "left_elbow");  re = pt(kps, "right_elbow")
    wm = midpt(lw, rw)
    em = midpt(le, re)

    # Left arm line
    cv2.line(overlay, le, lw, C_CURRENT, 3, cv2.LINE_AA)
    # Right arm line
    cv2.line(overlay, re, rw, C_CURRENT, 3, cv2.LINE_AA)
    # Wrist-to-wrist connector
    cv2.line(overlay, lw, rw, C_CURRENT, 2, cv2.LINE_AA)

    # Angle arc at wrist midpoint (elbow-wrist-wrist angle ≈ wrist hinge)
    draw_angle_arc(overlay, wm, em, le, (80, 180, 255), radius=38, thickness=2)

    # Deviation marker: how far wrist returned to address height
    addr_y = addr_wm[1]; curr_y = wm[1]
    delta = addr_y - curr_y   # positive = curr still above addr
    dev_color = C_GHOST if abs(delta) < 30 else C_DEV
    cv2.line(overlay, (wm[0]-20, addr_y), (wm[0]+20, addr_y), C_GHOST, 1)
    cv2.line(overlay, (wm[0]-20, curr_y), (wm[0]+20, curr_y), dev_color, 2)
    cv2.line(overlay, (wm[0], addr_y), (wm[0], curr_y), dev_color, 2)
    draw_label(overlay, f"dy={delta:.0f}px", (wm[0]+22, (addr_y+curr_y)//2), dev_color)

    # Dots
    for p in (lw, rw, le, re, wm, em):
        cv2.circle(overlay, p, 6, C_CURRENT, -1)
        cv2.circle(overlay, p, 6, C_WHITE,    1)

    draw_label(overlay, "IMPACT wrist V", (lw[0]-60, lw[1]-25), C_CURRENT)

    result = add_ui_frame(overlay,
        title="Downtheline — IMPACT",
        indicator_name="WRIST V  (yellow=current  green=address ghost  arc=wrist angle)",
        frame_label="Frame 47")
    save(result, "DTL_impact_wrist_v.png")


# ── RENDER 3: 正面 ADDRESS — Center Axis + shoulder/hip discs ───────────────
def render_fo_center_axis_address():
    kps = FO_ADDR
    frame = dim_frame(load_frame(FO_VIDEO, 60))
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    # Shoulder midpoint and hip midpoint
    sh_l  = pt(kps, "left_shoulder");  sh_r = pt(kps, "right_shoulder")
    hip_l = pt(kps, "left_hip");       hip_r = pt(kps, "right_hip")
    sh_mid  = midpt(sh_l, sh_r)
    hip_mid = midpt(hip_l, hip_r)

    # Center axis = vertical through hip_mid (anti-sway baseline)
    cx = hip_mid[0]
    cv2.line(overlay, (cx, 0), (cx, h), C_GHOST, 2, cv2.LINE_AA)
    draw_label(overlay, "CENTER AXIS", (cx+8, h//4), C_GHOST)

    # Expanded shoulders (coaching anchor: +40% outward from mid)
    sh_mid_x = sh_mid[0]
    exp = 0.40
    sh_l_exp  = (int(sh_mid_x + (sh_l[0]  - sh_mid_x) * (1+exp)), sh_l[1])
    sh_r_exp  = (int(sh_mid_x + (sh_r[0]  - sh_mid_x) * (1+exp)), sh_r[1])

    # Shoulder disc (rotation reference arc) — semi-transparent ellipse
    sh_half_w = abs(sh_l_exp[0] - sh_r_exp[0]) // 2
    sh_disc_layer = overlay.copy()
    cv2.ellipse(sh_disc_layer, sh_mid, (sh_half_w, max(sh_half_w//5, 12)),
                0, 0, 360, C_GHOST, 2, cv2.LINE_AA)
    # fill with very low alpha
    cv2.ellipse(sh_disc_layer, sh_mid, (sh_half_w, max(sh_half_w//5, 12)),
                0, 0, 360, C_GHOST, -1)
    cv2.addWeighted(sh_disc_layer, 0.12, overlay, 0.88, 0, overlay)
    # solid border
    cv2.ellipse(overlay, sh_mid, (sh_half_w, max(sh_half_w//5, 12)),
                0, 0, 360, C_GHOST, 2, cv2.LINE_AA)
    draw_label(overlay, "SH", (sh_mid[0]-10, sh_mid[1]+5), C_GHOST)

    # Hip disc
    hip_half_w = abs(hip_l[0] - hip_r[0]) // 2
    hip_disc_layer = overlay.copy()
    cv2.ellipse(hip_disc_layer, hip_mid, (hip_half_w, max(hip_half_w//5, 10)),
                0, 0, 360, C_CURRENT, -1)
    cv2.addWeighted(hip_disc_layer, 0.12, overlay, 0.88, 0, overlay)
    cv2.ellipse(overlay, hip_mid, (hip_half_w, max(hip_half_w//5, 10)),
                0, 0, 360, C_CURRENT, 2, cv2.LINE_AA)
    draw_label(overlay, "HIP", (hip_mid[0]-12, hip_mid[1]+5), C_CURRENT)

    # Shoulder line and hip line
    cv2.line(overlay, sh_l_exp, sh_r_exp, C_GHOST,   2, cv2.LINE_AA)
    cv2.line(overlay, hip_l,    hip_r,    C_CURRENT, 2, cv2.LINE_AA)

    # Shoulder rotation angle label (at address, ~0° rotation — illustrative)
    draw_label(overlay, "Shoulder 0deg", (sh_l_exp[0]-10, sh_mid[1]-22), C_GHOST)
    draw_label(overlay, "Hip 0deg",      (hip_l[0]-10,    hip_mid[1]-22), C_CURRENT)

    # Keypoint dots
    for p in (sh_l, sh_r, sh_l_exp, sh_r_exp, hip_l, hip_r, sh_mid, hip_mid):
        cv2.circle(overlay, p, 5, C_WHITE, -1)

    result = add_ui_frame(overlay,
        title="Face-on — ADDRESS",
        indicator_name="CENTER AXIS + ROTATION DISCS  (green=shoulder  yellow=hip)",
        frame_label="Frame 60")
    save(result, "FO_address_center_axis.png")


# ── RENDER 4: 正面 IMPACT — Rotation disc delta ──────────────────────────────
def render_fo_rotation_disc_impact():
    kps  = FO_IMP
    addr = FO_ADDR
    frame = dim_frame(load_frame(FO_VIDEO, 111))
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    def sh_angle(kps_dict):
        """Shoulder rotation angle vs horizontal (°)."""
        l = pt(kps_dict, "left_shoulder"); r = pt(kps_dict, "right_shoulder")
        return math.degrees(math.atan2(l[1]-r[1], l[0]-r[0]))

    def hip_angle(kps_dict):
        l = pt(kps_dict, "left_hip"); r = pt(kps_dict, "right_hip")
        return math.degrees(math.atan2(l[1]-r[1], l[0]-r[0]))

    addr_sh_ang  = sh_angle(addr);  curr_sh_ang  = sh_angle(kps)
    addr_hip_ang = hip_angle(addr); curr_hip_ang = hip_angle(kps)
    delta_sh  = curr_sh_ang  - addr_sh_ang
    delta_hip = curr_hip_ang - addr_hip_ang

    # Center axis (hip at address frame for stable reference)
    hip_addr_mid = midpt(pt(addr,"left_hip"), pt(addr,"right_hip"))
    cx = hip_addr_mid[0]
    cv2.line(overlay, (cx, 0), (cx, h), C_GHOST, 1, cv2.LINE_AA)

    # Ghost discs (address position)
    sh_addr_mid = midpt(pt(addr,"left_shoulder"), pt(addr,"right_shoulder"))
    sh_half_w_a = abs(pt(addr,"left_shoulder")[0] - pt(addr,"right_shoulder")[0]) // 2

    ghost_layer = overlay.copy()
    cv2.ellipse(ghost_layer, sh_addr_mid,
                (sh_half_w_a, max(sh_half_w_a//5, 12)), 0, 0, 360, C_GHOST, -1)
    cv2.addWeighted(ghost_layer, 0.10, overlay, 0.90, 0, overlay)
    cv2.ellipse(overlay, sh_addr_mid,
                (sh_half_w_a, max(sh_half_w_a//5, 12)),
                0, 0, 360, C_GHOST, 1, cv2.LINE_AA)

    # Current shoulder disc (rotated ellipse)
    sh_l = pt(kps,"left_shoulder"); sh_r = pt(kps,"right_shoulder")
    sh_mid = midpt(sh_l, sh_r)
    sh_half_w = abs(sh_l[0]-sh_r[0])//2
    cv2.ellipse(overlay, sh_mid, (max(sh_half_w, 30), max(sh_half_w//5, 12)),
                int(curr_sh_ang), 0, 360, C_CURRENT, 2, cv2.LINE_AA)
    cv2.line(overlay, sh_l, sh_r, C_CURRENT, 3, cv2.LINE_AA)

    # Current hip disc
    hip_l = pt(kps,"left_hip"); hip_r = pt(kps,"right_hip")
    hip_mid = midpt(hip_l, hip_r)
    hip_half_w = abs(hip_l[0]-hip_r[0])//2
    cv2.ellipse(overlay, hip_mid, (max(hip_half_w, 20), max(hip_half_w//5, 10)),
                int(curr_hip_ang), 0, 360, C_CURRENT, 2, cv2.LINE_AA)
    cv2.line(overlay, hip_l, hip_r, C_CURRENT, 3, cv2.LINE_AA)

    # Delta labels
    sh_color  = C_GHOST if abs(delta_sh)  < 5 else C_DEV
    hip_color = C_GHOST if abs(delta_hip) < 5 else C_DEV
    draw_label(overlay, f"SH  d={delta_sh:+.1f}deg",  (sh_mid[0]-50,  sh_mid[1]-28),  sh_color)
    draw_label(overlay, f"HIP d={delta_hip:+.1f}deg", (hip_mid[0]-50, hip_mid[1]-28), hip_color)
    draw_label(overlay, "ghost=address", (sh_addr_mid[0]+sh_half_w_a+8, sh_addr_mid[1]), C_GHOST)

    # Keypoint dots
    for p in (sh_l, sh_r, sh_mid, hip_l, hip_r, hip_mid):
        cv2.circle(overlay, p, 5, C_CURRENT, -1)
        cv2.circle(overlay, p, 5, C_WHITE,    1)

    result = add_ui_frame(overlay,
        title="Face-on — IMPACT",
        indicator_name="ROTATION DISC DELTA  (green=address ghost  yellow=impact  red=deviation)",
        frame_label="Frame 111")
    save(result, "FO_impact_rotation_disc.png")


# ── run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Rendering preview images...")
    render_dtl_posture_address()
    render_dtl_wrist_v_impact()
    render_fo_center_axis_address()
    render_fo_rotation_disc_impact()
    print(f"\nAll saved to {OUT_DIR}")
