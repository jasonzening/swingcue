#!/usr/bin/env python3
"""
render_sam2_parts.py
Split SAM2 full-body mask into per-body-part contours using RTMPose keypoints.

Strategy: for each body part, create a convex-hull mask from its joint points,
intersect with the SAM2 full-body mask, then draw only the contour (no fill).

Body parts (each independent object):
  head        - nose + shoulder midpoint region
  torso       - shoulder quad + hip quad
  left_arm    - l_shoulder + l_elbow + l_wrist strip
  right_arm   - r_shoulder + r_elbow + r_wrist strip
  left_leg    - l_hip + l_knee + l_ankle strip
  right_leg   - r_hip + r_knee + r_ankle strip
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import List

OUT = Path("/home/jason/projects/swingcue-postest/keyframes/preview")
OUT.mkdir(parents=True, exist_ok=True)

FRAME_IDX  = 47
DTL_VIDEO  = "/home/jason/projects/swingcue-postest/input/test-dwontheline.mp4"
MODEL_CFG  = "configs/sam2.1/sam2.1_hiera_t.yaml"
CHECKPOINT = "/home/jason/projects/swingcue-postest/models/sam2/sam2.1_hiera_tiny.pt"

C_GREEN    = (40, 220, 55)   # BGR
CONTOUR_W  = 2               # half of previous 3px → 2px (close enough)

# RTMPose keypoints, impact frame 47
KPS = {
    "nose":           np.array([463, 455]),
    "l_shoulder":     np.array([375, 447]),
    "r_shoulder":     np.array([398, 479]),
    "l_elbow":        np.array([413, 543]),
    "r_elbow":        np.array([398, 582]),
    "l_wrist":        np.array([435, 628]),
    "r_wrist":        np.array([435, 657]),
    "l_hip":          np.array([267, 607]),
    "r_hip":          np.array([310, 612]),
    "l_knee":         np.array([316, 759]),
    "r_knee":         np.array([367, 766]),
    "l_ankle":        np.array([291, 897]),
    "r_ankle":        np.array([333, 920]),
}


# ── Part mask builder ─────────────────────────────────────────────────────────
@dataclass
class BodyPart:
    name: str
    contour_mask: np.ndarray   # bool (H, W) — this part's region from SAM2


def limb_strip_mask(h, w, p1, p2, p3, half_width):
    """
    Create a filled polygon mask for a 3-joint limb (upper + lower segment).
    Each segment is fattened by half_width pixels perpendicular to the bone.
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    def segment_quad(a, b, hw):
        """Fattened rectangle around segment a→b."""
        d = b - a
        length = np.linalg.norm(d)
        if length < 1:
            return None
        perp = np.array([-d[1], d[0]]) / length * hw
        return np.array([
            a + perp, b + perp,
            b - perp, a - perp,
        ], dtype=np.int32)

    for a, b in [(p1, p2), (p2, p3)]:
        quad = segment_quad(a, b, half_width)
        if quad is not None:
            cv2.fillPoly(mask, [quad], 255)
        # round joint
        for pt in (a, b):
            cv2.circle(mask, tuple(pt.astype(int)), half_width, 255, -1)

    return mask.astype(bool)


def head_mask(h, w, nose, l_sh, r_sh, radius_frac=0.12):
    """Circular region around the head (nose + area above shoulders)."""
    sh_mid = ((l_sh + r_sh) / 2).astype(int)
    # Head center = midway between nose and shoulder midpoint
    center = ((nose + sh_mid) / 2).astype(int)
    radius = int(np.linalg.norm(nose - sh_mid) * 0.9)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, tuple(center), radius, 255, -1)
    return mask.astype(bool)


def torso_mask(h, w, l_sh, r_sh, l_hip, r_hip):
    """Quadrilateral: left_shoulder, right_shoulder, right_hip, left_hip."""
    pts = np.array([l_sh, r_sh, r_hip, l_hip], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask.astype(bool)


def build_parts(h, w) -> List[dict]:
    """
    Returns list of {name, region_mask (bool H×W)}.
    Each region_mask will be intersected with SAM2 mask.
    """
    k = KPS
    hw_arm  = 30   # half-width pixels for arm strips
    hw_leg  = 40   # half-width pixels for leg strips

    parts = [
        {"name": "head",
         "region": head_mask(h, w, k["nose"], k["l_shoulder"], k["r_shoulder"])},

        {"name": "torso",
         "region": torso_mask(h, w, k["l_shoulder"], k["r_shoulder"],
                               k["l_hip"], k["r_hip"])},

        {"name": "left_arm",
         "region": limb_strip_mask(h, w,
                                   k["l_shoulder"], k["l_elbow"], k["l_wrist"],
                                   hw_arm)},

        {"name": "right_arm",
         "region": limb_strip_mask(h, w,
                                   k["r_shoulder"], k["r_elbow"], k["r_wrist"],
                                   hw_arm)},

        {"name": "left_leg",
         "region": limb_strip_mask(h, w,
                                   k["l_hip"], k["l_knee"], k["l_ankle"],
                                   hw_leg)},

        {"name": "right_leg",
         "region": limb_strip_mask(h, w,
                                   k["r_hip"], k["r_knee"], k["r_ankle"],
                                   hw_leg)},
    ]
    return parts


# ── Main ──────────────────────────────────────────────────────────────────────
def load_frame(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, f = cap.read()
    cap.release()
    assert ret
    return f


def get_sam2_mask(frame_rgb, h, w):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SAM2 device: {device}")

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(MODEL_CFG, CHECKPOINT, device=device)
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(frame_rgb)

    pts = np.array([v.tolist() for v in KPS.values()], dtype=np.float32)
    lbls = np.ones(len(pts), dtype=np.int32)

    masks, scores, _ = predictor.predict(
        point_coords=pts, point_labels=lbls, multimask_output=True
    )
    best = np.argmax(scores)
    print(f"SAM2 mask score: {scores[best]:.3f}")
    return masks[best].astype(bool)


def main():
    frame_bgr = load_frame(DTL_VIDEO, FRAME_IDX)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_bgr.shape[:2]

    # 1. SAM2 full-body mask
    sam2_mask = get_sam2_mask(frame_rgb, h, w)
    print(f"Full mask coverage: {sam2_mask.sum()/(h*w)*100:.1f}%")

    # 2. Build per-part region masks, intersect with SAM2
    parts_data = build_parts(h, w)
    body_parts: List[BodyPart] = []

    for p in parts_data:
        part_mask = p["region"] & sam2_mask
        body_parts.append(BodyPart(name=p["name"], contour_mask=part_mask))
        coverage = part_mask.sum() / (h * w) * 100
        print(f"  {p['name']:12s}: {coverage:.2f}%")

    # 3. Draw contours only (no fill) — each part independent
    result = frame_bgr.copy()
    for bp in body_parts:
        mask_u8 = bp.contour_mask.astype(np.uint8) * 255
        # Find external contours of this part's region
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(result, contours, -1, C_GREEN, CONTOUR_W, cv2.LINE_AA)

    out = OUT / "SAM2_parts_contour_DTL_impact_fr47.png"
    cv2.imwrite(str(out), result)
    print(f"\nSaved: {out}")

    import shutil
    desk = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview")
    desk.mkdir(parents=True, exist_ok=True)
    shutil.copy(out, desk / out.name)
    print(f"Desktop: {desk / out.name}")


if __name__ == "__main__":
    main()
