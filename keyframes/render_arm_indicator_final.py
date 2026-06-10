#!/usr/bin/env python3
"""
render_arm_indicator_final.py

Three layers on DTL impact frame (fr47):
  1. SAM2 body contour — green, continuous, thin, no fill
  2. Arm indicator (single, point-select mode):
       - Current arm: orange chain l_shoulder→l_elbow→l_wrist (real position)
       - Target arm:  green semi-transparent "straight arm" silhouette
                      (shoulder→elbow on sh-wr line→wrist, illustrative)
       - Arc arrow:   from actual elbow → target elbow
  3. All other joints dimmed (50% opacity gray) — point-select discipline

Knowledge basis (GOLF_SWING_KNOWLEDGE §3 impact×arm confirmed):
  Correct = lead arm straight, shoulder-elbow-wrist collinear.
  Chicken wing = elbow flares out from that line.

Lead arm for right-handed golfer = LEFT arm.
Target position: rotate upper arm vector toward vertical to create
visible illustrative separation (spec: 示意画即可).
"""

import cv2
import numpy as np
import torch
import math
from pathlib import Path

OUT = Path("/home/jason/projects/swingcue-postest/keyframes/preview")
OUT.mkdir(parents=True, exist_ok=True)
DTL_VIDEO  = "/home/jason/projects/swingcue-postest/input/test-dwontheline.mp4"
MODEL_CFG  = "configs/sam2.1/sam2.1_hiera_t.yaml"
CHECKPOINT = "/home/jason/projects/swingcue-postest/models/sam2/sam2.1_hiera_tiny.pt"

# Colors (BGR)
C_BODY    = (40,  220,  55)   # green  — body contour
C_CURRENT = (0,   160, 255)   # orange — current arm chain
C_TARGET  = (40,  220,  55)   # green  — target straight arm
C_ARROW   = (40,  220,  55)   # green  — arc arrow
C_DIM     = (90,  90,   90)   # dark gray — dimmed other joints
C_WHITE   = (255, 255, 255)

BODY_W    = 2
CHAIN_W   = 2
C_ARROW   = (220, 220, 255)   # near-white — arc arrow, stands out from green
ARROW_W    = 3
ARROW_HEAD = 22
DOT_R      = 5
ARM_THICK  = 16   # silhouette half-stroke for target arm shape
OPACITY    = 0.60  # single blend for semi-transparency

# Keypoints — frame 47 (from RTMPose JSON)
KPS = {
    "nose":       (463, 455),
    "l_shoulder": (375, 447),
    "r_shoulder": (398, 479),
    "l_elbow":    (413, 543),
    "r_elbow":    (398, 582),
    "l_wrist":    (435, 628),
    "r_wrist":    (435, 657),
    "l_hip":      (267, 607),
    "r_hip":      (310, 612),
    "l_knee":     (316, 759),
    "r_knee":     (367, 766),
    "l_ankle":    (291, 897),
    "r_ankle":    (333, 920),
}


# ── Geometry ──────────────────────────────────────────────────────────────────
def pt(name): return tuple(int(v) for v in KPS[name])
def arr(name): return np.array(KPS[name], dtype=float)

def angle_from_down(dx, dy):
    """Angle in degrees clockwise from straight down (+y axis)."""
    return math.degrees(math.atan2(dx, dy))

def rotate_vec(dx, dy, deg):
    r = math.radians(deg)
    return (dx*math.cos(r) + dy*math.sin(r),
           -dx*math.sin(r) + dy*math.cos(r))

def make_target_arm(rotate_upper_deg=-18):
    """
    Compute target arm position.
    Rotate upper arm (sh→el) rotate_upper_deg toward vertical (more straight-down).
    Lower arm follows sh→wr direction from new elbow.
    Returns (el_target, wr_target) as int tuples.
    """
    sh = arr("l_shoulder"); el = arr("l_elbow"); wr = arr("l_wrist")

    ux = el[0]-sh[0]; uy = el[1]-sh[1]
    ul = math.hypot(ux, uy)

    ux_t, uy_t = rotate_vec(ux, uy, rotate_upper_deg)
    n = math.hypot(ux_t, uy_t) or 1
    el_t = sh + np.array([ux_t, uy_t]) / n * ul

    # Lower arm: keep sh→wr overall direction, same length as actual lower arm
    ll = math.hypot(wr[0]-el[0], wr[1]-el[1])
    sw = wr - sh
    sw_n = math.hypot(*sw) or 1
    wr_t = el_t + sw / sw_n * ll

    return (int(el_t[0]), int(el_t[1])), (int(wr_t[0]), int(wr_t[1]))


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_arm_silhouette(canvas, sh_pt, el_pt, wr_pt, color, thickness):
    """Thick rounded capsule arm shape."""
    r = thickness // 2
    for a, b in [(sh_pt, el_pt), (el_pt, wr_pt)]:
        cv2.line(canvas, a, b, color, thickness, cv2.LINE_AA)
    for p in (sh_pt, el_pt, wr_pt):
        cv2.circle(canvas, p, r, color, -1, cv2.LINE_AA)


def draw_joint_dot(canvas, p, color, r=DOT_R):
    cv2.circle(canvas, p, r, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, p, r, C_WHITE, 1, cv2.LINE_AA)


def bezier_pts(p0, ctrl, p2, n=50):
    return [(int((1-t)**2*p0[0]+2*(1-t)*t*ctrl[0]+t**2*p2[0]),
             int((1-t)**2*p0[1]+2*(1-t)*t*ctrl[1]+t**2*p2[1]))
            for t in [i/n for i in range(n+1)]]


def draw_arc_arrow(canvas, src, dst, color, lw=ARROW_W, head=ARROW_HEAD):
    mx = (src[0]+dst[0])//2; my = (src[1]+dst[1])//2
    dx = dst[0]-src[0]; dy = dst[1]-src[1]
    n  = math.hypot(dx, dy) or 1
    # Control point: perpendicular outward (right side in DTL view)
    ctrl = (int(mx + (dy/n)*70), int(my - (dx/n)*70))

    pts = bezier_pts(src, ctrl, dst)
    for i in range(len(pts)-1):
        cv2.line(canvas, pts[i], pts[i+1], color, lw, cv2.LINE_AA)

    # Arrowhead
    tip = pts[-1]
    d = (pts[-1][0]-pts[-4][0], pts[-1][1]-pts[-4][1])
    nm = math.hypot(*d) or 1
    ux, uy = d[0]/nm, d[1]/nm
    a = math.radians(26); ca, sa = math.cos(a), math.sin(a)
    w1 = (int(tip[0]-head*(ux*ca-uy*sa)), int(tip[1]-head*(uy*ca+ux*sa)))
    w2 = (int(tip[0]-head*(ux*ca+uy*sa)), int(tip[1]-head*(uy*ca-ux*sa)))
    cv2.fillPoly(canvas, [np.array([tip,w1,w2], np.int32)], color, cv2.LINE_AA)


# ── SAM2 ──────────────────────────────────────────────────────────────────────
def get_body_contour(frame_rgb):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(MODEL_CFG, CHECKPOINT, device=device)
    pred  = SAM2ImagePredictor(model)
    pred.set_image(frame_rgb)
    pts  = np.array([list(v) for v in KPS.values()], dtype=np.float32)
    lbls = np.ones(len(pts), dtype=np.int32)
    masks, scores, _ = pred.predict(
        point_coords=pts, point_labels=lbls, multimask_output=True)
    best = np.argmax(scores)
    mask_u8 = masks[best].astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(f"SAM2 score: {scores[best]:.3f}")
    return cnts


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(DTL_VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 47)
    ret, frame = cap.read(); cap.release(); assert ret
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    print("SAM2...")
    body_cnts = get_body_contour(frame_rgb)

    # Compute target arm
    el_tgt, wr_tgt = make_target_arm(rotate_upper_deg=-18)
    print(f"Actual elbow:  {pt('l_elbow')}")
    print(f"Target elbow:  {el_tgt}  sep={int(math.hypot(pt('l_elbow')[0]-el_tgt[0], pt('l_elbow')[1]-el_tgt[1]))}px")

    # ── All drawing onto overlay (single 60% blend at the end) ────────────────
    overlay = frame.copy()

    # ── Dimmed other joints (point-select: arm is active, rest gray) ──────────
    dim_pairs = [
        ("r_shoulder","r_elbow"), ("r_elbow","r_wrist"),  # right arm (trail)
        ("l_hip","l_knee"), ("l_knee","l_ankle"),          # left leg
        ("r_hip","r_knee"), ("r_knee","r_ankle"),          # right leg
    ]
    for a, b in dim_pairs:
        cv2.line(overlay, pt(a), pt(b), C_DIM, 1, cv2.LINE_AA)
    for name in ("r_shoulder","r_elbow","r_wrist","l_hip","r_hip","l_knee","r_knee","l_ankle","r_ankle"):
        cv2.circle(overlay, pt(name), 3, C_DIM, -1, cv2.LINE_AA)

    # ── Target arm silhouette — green, thick rounded shape ────────────────────
    draw_arm_silhouette(overlay, pt("l_shoulder"), el_tgt, wr_tgt,
                        C_TARGET, ARM_THICK)

    # ── Arc arrow: actual elbow → target elbow ────────────────────────────────
    draw_arc_arrow(overlay, pt("l_elbow"), el_tgt, C_ARROW)

    # ── Current arm chain — orange, on top ────────────────────────────────────
    cv2.line(overlay, pt("l_shoulder"), pt("l_elbow"), C_CURRENT, CHAIN_W, cv2.LINE_AA)
    cv2.line(overlay, pt("l_elbow"),    pt("l_wrist"),  C_CURRENT, CHAIN_W, cv2.LINE_AA)
    for name in ("l_shoulder", "l_elbow", "l_wrist"):
        draw_joint_dot(overlay, pt(name), C_CURRENT)

    # ── Single 60% blend: indicator layers semi-transparent ───────────────────
    result = frame.copy()
    cv2.addWeighted(overlay, OPACITY, frame, 1.0 - OPACITY, 0, result)

    # ── Body contour on TOP of blend — full opacity, crisp edge ───────────────
    cv2.drawContours(result, body_cnts, -1, C_BODY, BODY_W, cv2.LINE_AA)

    out = OUT / "indicator_arm_lead_DTL_impact_fr47.png"
    cv2.imwrite(str(out), result)
    print(f"Saved: {out}")

    import shutil
    desk = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview")
    desk.mkdir(parents=True, exist_ok=True)
    shutil.copy(out, desk / out.name)
    print(f"Desktop: {desk / out.name}")


if __name__ == "__main__":
    main()
