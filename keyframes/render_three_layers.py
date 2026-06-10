#!/usr/bin/env python3
"""
render_three_layers.py

Three-layer overlay:
  Layer 1 — Full body outer contour (green, continuous, thin)
  Layer 2 — Per-limb contours (each arm/leg traced along its real edge,
             root connected to body, no fill, slightly thicker than skeleton)
  Layer 3 — Skeleton chain lines (thin, joint positions)

Per-limb contour method:
  For each limb (l_arm, r_arm, l_leg, r_leg):
    1. Build a "fat strip" mask around the joint chain (shoulder→elbow→wrist etc.)
       with generous half-width so it covers the real limb width
    2. Intersect with SAM2 full-body mask → real limb pixels
    3. findContours on that intersection → actual limb edge
    4. Draw only the external contour (no fill)
  The root region naturally overlaps with the body contour — no cut.

Engineering: each limb is a LimbContour object (per-frame, root-anchored).
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

OUT = Path("/home/jason/projects/swingcue-postest/keyframes/preview")
OUT.mkdir(parents=True, exist_ok=True)

FRAME_IDX  = 47
DTL_VIDEO  = "/home/jason/projects/swingcue-postest/input/test-dwontheline.mp4"
MODEL_CFG  = "configs/sam2.1/sam2.1_hiera_t.yaml"
CHECKPOINT = "/home/jason/projects/swingcue-postest/models/sam2/sam2.1_hiera_tiny.pt"

# Colors (BGR)
C_BODY_CONTOUR = (40,  220,  55)   # green  — full body outline
C_LIMB_CONTOUR = (60,  200, 255)   # yellow — per-limb outline
C_ARM_SKEL     = (0,   180, 255)   # orange — arm skeleton
C_LEG_SKEL     = (80,  100, 255)   # red    — leg skeleton
C_SPINE_SKEL   = (200, 200,  80)   # cyan   — spine

BODY_W  = 2    # full body contour width
LIMB_W  = 2    # per-limb contour width
SKEL_W  = 1    # skeleton line width (thinner, just reference)
DOT_R   = 4    # joint dot radius

KPS = {
    "nose":       np.array([463, 455]),
    "l_shoulder": np.array([375, 447]),
    "r_shoulder": np.array([398, 479]),
    "l_elbow":    np.array([413, 543]),
    "r_elbow":    np.array([398, 582]),
    "l_wrist":    np.array([435, 628]),
    "r_wrist":    np.array([435, 657]),
    "l_hip":      np.array([267, 607]),
    "r_hip":      np.array([310, 612]),
    "l_knee":     np.array([316, 759]),
    "r_knee":     np.array([367, 766]),
    "l_ankle":    np.array([291, 897]),
    "r_ankle":    np.array([333, 920]),
}


# ── Engineering object ────────────────────────────────────────────────────────
@dataclass
class LimbContour:
    """
    Per-frame limb contour object.
    contours: OpenCV contour list from this limb's SAM2-intersected region.
    root_joint: anchor (never moves on drag).
    tip_joint:  draggable end-effector.
    Future drag: tip moved → recompute joint angles → regenerate strip mask
                 → re-intersect with SAM2 → update contours. Root stays fixed.
    """
    name:       str
    root_joint: str
    tip_joint:  str
    joints:     List[str]          # ordered root→tip
    contours:   list               # OpenCV contour list
    color:      Tuple[int,int,int]

    def draw_contour(self, canvas, lw=LIMB_W):
        cv2.drawContours(canvas, self.contours, -1, self.color, lw, cv2.LINE_AA)

    def draw_skeleton(self, canvas, lw=SKEL_W, dot_r=DOT_R):
        pts = [KPS[k] for k in self.joints]
        for i in range(len(pts)-1):
            cv2.line(canvas, tuple(pts[i].astype(int)),
                     tuple(pts[i+1].astype(int)), self.color, lw, cv2.LINE_AA)
        for i, p in enumerate(pts):
            r = dot_r+2 if i == 0 else dot_r
            cv2.circle(canvas, tuple(p.astype(int)), r, self.color, -1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(p.astype(int)), r, (255,255,255), 1, cv2.LINE_AA)


# ── Strip mask builder ────────────────────────────────────────────────────────
def limb_strip_mask(h, w, joint_pts, half_width):
    """
    Fat strip polygon mask along a sequence of joints.
    Generous half_width so the strip fully contains the real limb.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(len(joint_pts)-1):
        a = joint_pts[i].astype(float)
        b = joint_pts[i+1].astype(float)
        d = b - a
        length = np.linalg.norm(d)
        if length < 1:
            continue
        perp = np.array([-d[1], d[0]]) / length * half_width
        quad = np.array([a+perp, b+perp, b-perp, a-perp], dtype=np.int32)
        cv2.fillPoly(mask, [quad], 255)
        cv2.circle(mask, tuple(a.astype(int)), int(half_width), 255, -1)
        cv2.circle(mask, tuple(b.astype(int)), int(half_width), 255, -1)
    return mask


def build_limb_contours(h, w, sam2_mask: np.ndarray) -> List[LimbContour]:
    """
    For each limb: create strip mask → intersect with SAM2 → extract contour.
    half_width is generous (covers real limb + some margin) so SAM2 intersection
    carves the true body edge.
    """
    # (name, root, tip, joints, skel_color, half_width_px)
    specs = [
        ("left_arm",  "l_shoulder", "l_wrist",
         ["l_shoulder","l_elbow","l_wrist"], C_ARM_SKEL, 48),

        ("right_arm", "r_shoulder", "r_wrist",
         ["r_shoulder","r_elbow","r_wrist"], C_ARM_SKEL, 48),

        ("left_leg",  "l_hip", "l_ankle",
         ["l_hip","l_knee","l_ankle"], C_LEG_SKEL, 58),

        ("right_leg", "r_hip", "r_ankle",
         ["r_hip","r_knee","r_ankle"], C_LEG_SKEL, 58),
    ]

    limbs = []
    for name, root, tip, jnames, color, hw in specs:
        jpts = [KPS[k] for k in jnames]

        # Strip mask → intersect with SAM2 body mask
        strip = limb_strip_mask(h, w, jpts, half_width=hw)
        sam2_u8 = sam2_mask.astype(np.uint8) * 255
        intersection = cv2.bitwise_and(strip, sam2_u8)

        # Find external contour of this limb region
        cnts, _ = cv2.findContours(intersection, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)

        limbs.append(LimbContour(
            name=name, root_joint=root, tip_joint=tip,
            joints=jnames, contours=cnts, color=C_LIMB_CONTOUR,
        ))
        # Store skel color separately for skeleton drawing
        limbs[-1]._skel_color = color

    return limbs


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_frame(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, f = cap.read()
    cap.release()
    assert ret; return f


def get_sam2_mask(frame_rgb, kps_pts):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(MODEL_CFG, CHECKPOINT, device=device)
    pred  = SAM2ImagePredictor(model)
    pred.set_image(frame_rgb)
    pts  = np.array(kps_pts, dtype=np.float32)
    lbls = np.ones(len(pts), dtype=np.int32)
    masks, scores, _ = pred.predict(
        point_coords=pts, point_labels=lbls, multimask_output=True)
    best = np.argmax(scores)
    print(f"SAM2 score: {scores[best]:.3f}")
    return masks[best]   # bool H×W


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    frame_bgr = load_frame(DTL_VIDEO, FRAME_IDX)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_bgr.shape[:2]

    print("Running SAM2...")
    kps_pts = [v.tolist() for v in KPS.values()]
    sam2_mask = get_sam2_mask(frame_rgb, kps_pts)   # bool

    # Build per-limb contour objects
    limbs = build_limb_contours(h, w, sam2_mask)

    # Full-body contour from SAM2
    mask_u8 = sam2_mask.astype(np.uint8) * 255
    body_cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_NONE)

    result = frame_bgr.copy()

    # ── Layer 3: skeleton lines (thinnest, drawn first = bottom) ──────────────
    sh_mid  = ((KPS["l_shoulder"] + KPS["r_shoulder"]) / 2).astype(int)
    hip_mid = ((KPS["l_hip"]      + KPS["r_hip"])      / 2).astype(int)
    cv2.line(result, tuple(sh_mid), tuple(hip_mid), C_SPINE_SKEL, SKEL_W, cv2.LINE_AA)

    for lmb in limbs:
        skel_color = lmb._skel_color
        pts = [KPS[k] for k in lmb.joints]
        for i in range(len(pts)-1):
            cv2.line(result,
                     tuple(pts[i].astype(int)), tuple(pts[i+1].astype(int)),
                     skel_color, SKEL_W, cv2.LINE_AA)
        for i, p in enumerate(pts):
            r = DOT_R+2 if i == 0 else DOT_R
            cv2.circle(result, tuple(p.astype(int)), r, skel_color, -1, cv2.LINE_AA)
            cv2.circle(result, tuple(p.astype(int)), r, (255,255,255), 1, cv2.LINE_AA)

    # ── Layer 2: per-limb contours (middle layer) ─────────────────────────────
    for lmb in limbs:
        lmb.draw_contour(result, lw=LIMB_W)

    # ── Layer 1: full body outer contour (top layer, most visible) ────────────
    cv2.drawContours(result, body_cnts, -1, C_BODY_CONTOUR, BODY_W, cv2.LINE_AA)

    out = OUT / "three_layers_DTL_impact_fr47.png"
    cv2.imwrite(str(out), result)
    print(f"Saved: {out}")

    import shutil
    desk = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview")
    desk.mkdir(parents=True, exist_ok=True)
    shutil.copy(out, desk / out.name)
    print(f"Desktop: {desk / out.name}")

    # Summary
    print("\nLimb contour summary:")
    for lmb in limbs:
        total_pts = sum(len(c) for c in lmb.contours)
        print(f"  {lmb.name:12s}  root={lmb.root_joint:12s} tip={lmb.tip_joint}  contour_pts={total_pts}")


if __name__ == "__main__":
    main()
