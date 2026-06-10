#!/usr/bin/env python3
"""
render_contour_with_chains.py

Visual: continuous SAM2 body contour + RTMPose joint chains inside.
  - Outer contour: single continuous green line (no fill, thin)
  - Inner chains: colored lines for each limb + spine

Engineering scaffold for future drag interaction:
  Each chain is a KinematicChain object:
    - root (anchor joint, never moves: shoulder or hip)
    - links: list of joint names in order
    - end_effector: the draggable tip (wrist or ankle)
  On drag: root fixed, angles recalculated from end_effector position,
           contour deforms only in that limb's region.
  (Interaction not implemented here — structure only.)
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

OUT = Path("/home/jason/projects/swingcue-postest/keyframes/preview")
OUT.mkdir(parents=True, exist_ok=True)

FRAME_IDX  = 47
DTL_VIDEO  = "/home/jason/projects/swingcue-postest/input/test-dwontheline.mp4"
MODEL_CFG  = "configs/sam2.1/sam2.1_hiera_t.yaml"
CHECKPOINT = "/home/jason/projects/swingcue-postest/models/sam2/sam2.1_hiera_tiny.pt"

# Colors (BGR)
C_CONTOUR = (40, 220, 55)    # green — outer body contour
C_ARM     = (0,  180, 255)   # yellow-orange — arm chains
C_LEG     = (80, 100, 255)   # red-orange — leg chains
C_SPINE   = (200, 200, 80)   # cyan-ish — spine/torso

CONTOUR_W = 2    # outer contour line width (half of previous 3px)
CHAIN_W   = 2    # joint chain line width
DOT_R     = 5    # joint dot radius

# RTMPose keypoints, impact frame 47
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


# ── Engineering scaffold ───────────────────────────────────────────────────────
@dataclass
class KinematicChain:
    """
    A draggable limb chain.
    root: anchor joint (fixed). end_effector: draggable tip.
    links: ordered joint names from root to end_effector.
    color: BGR draw color.

    Future drag behavior:
      - root stays fixed (shoulder / hip)
      - drag end_effector → recompute angles via inverse kinematics
      - only this chain's segment of the contour deforms
    """
    name:          str
    root:          str                # anchor joint key in KPS
    links:         List[str]          # [root, intermediate..., end_effector]
    end_effector:  str                # draggable tip
    color:         Tuple[int,int,int]

    def joints(self) -> List[np.ndarray]:
        """Ordered (x,y) positions for all links."""
        return [KPS[k] for k in self.links]

    def draw(self, canvas: np.ndarray, lw: int, dot_r: int):
        pts = self.joints()
        for i in range(len(pts) - 1):
            p1 = tuple(pts[i].astype(int))
            p2 = tuple(pts[i+1].astype(int))
            cv2.line(canvas, p1, p2, self.color, lw, cv2.LINE_AA)
        # Joint dots — root slightly larger (anchor), tip hollow-ish
        for i, p in enumerate(pts):
            cx, cy = int(p[0]), int(p[1])
            r = dot_r + 2 if i == 0 else dot_r   # root dot larger
            cv2.circle(canvas, (cx, cy), r, self.color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), r, (255,255,255), 1, cv2.LINE_AA)


def build_chains() -> List[KinematicChain]:
    """
    Define all draggable kinematic chains.
    Spine is a special chain: root = shoulder_mid, tip = hip_mid (both derived).
    """
    sh_mid  = ((KPS["l_shoulder"] + KPS["r_shoulder"]) / 2).astype(int)
    hip_mid = ((KPS["l_hip"]      + KPS["r_hip"])      / 2).astype(int)

    # Temporarily add midpoints to KPS for spine chain drawing
    KPS["sh_mid"]  = sh_mid
    KPS["hip_mid"] = hip_mid

    return [
        KinematicChain("left_arm",   "l_shoulder", ["l_shoulder","l_elbow","l_wrist"], "l_wrist",  C_ARM),
        KinematicChain("right_arm",  "r_shoulder", ["r_shoulder","r_elbow","r_wrist"], "r_wrist",  C_ARM),
        KinematicChain("left_leg",   "l_hip",      ["l_hip","l_knee","l_ankle"],       "l_ankle",  C_LEG),
        KinematicChain("right_leg",  "r_hip",      ["r_hip","r_knee","r_ankle"],       "r_ankle",  C_LEG),
        KinematicChain("spine",      "sh_mid",     ["sh_mid","hip_mid"],               "hip_mid",  C_SPINE),
    ]


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_frame(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, f = cap.read()
    cap.release()
    assert ret
    return f


def get_sam2_contour(frame_rgb):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(MODEL_CFG, CHECKPOINT, device=device)
    pred  = SAM2ImagePredictor(model)
    pred.set_image(frame_rgb)

    pts  = np.array([v for k,v in KPS.items()
                     if k not in ("sh_mid","hip_mid")], dtype=np.float32)
    lbls = np.ones(len(pts), dtype=np.int32)

    masks, scores, _ = pred.predict(
        point_coords=pts, point_labels=lbls, multimask_output=True)
    best = np.argmax(scores)
    print(f"SAM2 score: {scores[best]:.3f}")

    mask_u8 = masks[best].astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    return contours


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    frame_bgr = load_frame(DTL_VIDEO, FRAME_IDX)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # 1. SAM2 — full continuous body contour
    print("Running SAM2...")
    contours = get_sam2_contour(frame_rgb)

    result = frame_bgr.copy()

    # Outer contour: continuous green line, no fill, thin (CONTOUR_W px)
    cv2.drawContours(result, contours, -1, C_CONTOUR, CONTOUR_W, cv2.LINE_AA)

    # 2. Build kinematic chains and draw
    chains = build_chains()
    for chain in chains:
        chain.draw(result, CHAIN_W, DOT_R)

    out = OUT / "SAM2_contour_with_chains_DTL_impact_fr47.png"
    cv2.imwrite(str(out), result)
    print(f"Saved: {out}")

    import shutil
    desk = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview")
    desk.mkdir(parents=True, exist_ok=True)
    shutil.copy(out, desk / out.name)
    print(f"Desktop: {desk / out.name}")

    # Print chain summary for engineering reference
    print("\nKinematic chains (engineering scaffold):")
    for c in chains:
        print(f"  {c.name:12s} root={c.root:12s} tip={c.end_effector}")


if __name__ == "__main__":
    main()
