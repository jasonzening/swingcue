"""
engine/c_features/hip_rear_extractor.py
C Layer — R2' Hip Rear Edge Displacement  (v1.1)

v1.1 changes vs v1.0
--------------------
1. Window hard cap: P5 → impact (inclusive). No post-impact frames.
   hip_mid window also unified to the same range for comparability.

2. 4-point prompt: SAM2 receives [shoulder_mid, hip_mid, knee_mid, ankle_mid]
   (4 positive points), all must pass kp_guard. Single-point prompt forbidden.
   After prediction, the mask must contain ≥ 6 valid guarded joints; otherwise
   the frame is NaN (counted in nan_count).

3. Rear-edge geometry validity:
   rear_x must satisfy  0.05×torso_h ≤ |rear_x − hip_mid_x| ≤ 0.60×torso_h
   Values outside this range → NaN.  This catches cases like dtl-wrong-2 fr85
   where the rear edge jumped to a far-field artifact.

All three changes are isolated from the existing R2 (hip_mid) code path.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from engine.a_measurement.pose_pipeline import FrameMeasurement
from engine.b_phase.swing_phase import AnchorFrames
from engine.a_measurement.kp_guard import kp_guard

PROJ = Path(__file__).resolve().parents[2]
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_CKPT = str(PROJ / "models/sam2/sam2.1_hiera_tiny.pt")

BAND_FRAC       = 0.12    # hip band = hip_mid_y ± band_frac × torso_h
MIN_BAND_PX     = 50      # min pixels in hip band to accept mask
MIN_VALID_JOINTS = 6      # joints that must land inside mask after SAM2
REAR_DIST_MIN   = 0.05    # |rear_x - hip_mid_x| must be ≥ this × torso_h
REAR_DIST_MAX   = 0.60    # |rear_x - hip_mid_x| must be ≤ this × torso_h

# All COCO-17 joint names for the ≥6 containment check
ALL_JOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


@dataclass
class HipRearResult:
    hip_rear_disp:  np.ndarray   # per-frame fraction of torso_h (+toward ball)
    mask_quality:   np.ndarray   # per-frame SAM2 score (NaN outside window)
    addr_rear_x:    float        # rear-edge x at address (pixels)
    torso_h:        float
    band_y_lo:      float
    band_y_hi:      float
    toward_ball_sign: int
    window_frames:  list[int]    # frames actually in window (P5→impact)
    nan_count:      int          # frames that became NaN (any reason)
    nan_reasons:    dict         # fr → reason string
    meta:           dict = field(default_factory=dict)


def build_predictor(device: str = "cuda"):
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = build_sam2(SAM2_CFG, SAM2_CKPT, device=device)
    return SAM2ImagePredictor(model), device


def _mid(a, b):
    return ((a[0]+b[0])/2, (a[1]+b[1])/2)


def _build_4pt_prompt(kps: dict, thr: float = 0.3):
    """
    Build 4-positive-point prompt from shoulder/hip/knee/ankle mids.
    Returns (coords_array, labels_array) or (None, None) if any mid unavailable.

    All 8 constituent joints must pass kp_guard.
    """
    ls = kp_guard(kps, "left_shoulder",  thr)
    rs = kp_guard(kps, "right_shoulder", thr)
    lh = kp_guard(kps, "left_hip",       thr)
    rh = kp_guard(kps, "right_hip",      thr)
    lk = kp_guard(kps, "left_knee",      thr)
    rk = kp_guard(kps, "right_knee",     thr)
    la = kp_guard(kps, "left_ankle",     thr)
    ra = kp_guard(kps, "right_ankle",    thr)

    if not all([ls, rs, lh, rh, lk, rk, la, ra]):
        return None, None

    sh_mid  = _mid(ls, rs)
    hip_mid = _mid(lh, rh)
    kn_mid  = _mid(lk, rk)
    an_mid  = _mid(la, ra)

    coords = np.array([
        [sh_mid[0],  sh_mid[1]],
        [hip_mid[0], hip_mid[1]],
        [kn_mid[0],  kn_mid[1]],
        [an_mid[0],  an_mid[1]],
    ], dtype=np.float32)
    labels = np.ones(4, dtype=np.int32)
    return coords, labels


def _count_joints_in_mask(mask: np.ndarray, kps: dict, thr: float = 0.3) -> int:
    """Count how many kp_guard-valid joints land inside the mask."""
    h, w = mask.shape
    count = 0
    for nm in ALL_JOINTS:
        pt = kp_guard(kps, nm, thr)
        if pt is None:
            continue
        xi, yi = int(round(pt[0])), int(round(pt[1]))
        if 0 <= xi < w and 0 <= yi < h and mask[yi, xi]:
            count += 1
    return count


class HipRearExtractor:
    """
    R2' extractor v1.1 — SAM2 4-point mask + geometry validity.

    Parameters
    ----------
    device      : 'cuda' | 'cpu'
    kp_thr      : keypoint confidence threshold for kp_guard (default 0.3)
    """

    def __init__(self, device: str = "cuda", kp_thr: float = 0.3):
        self.device      = device
        self.kp_thr      = kp_thr
        self._predictor  = None

    def _get_predictor(self):
        if self._predictor is None:
            self._predictor, self.device = build_predictor(self.device)
        return self._predictor

    def extract(
        self,
        video_path:   str,
        measurements: List[FrameMeasurement],
        anchors:      AnchorFrames,
        ball_side:    str,
        phase_labels: list[str],
        kp_json:      dict,          # raw kp_json for multi-joint prompt + containment
    ) -> HipRearResult:
        """
        Extract hip_rear_disp for window [P5, impact] (inclusive, no post-impact).

        Parameters
        ----------
        video_path   : source .mp4
        measurements : A-layer FrameMeasurements
        anchors      : B-layer AnchorFrames
        ball_side    : "right" | "left"
        phase_labels : per-frame phase names
        kp_json      : raw kp cache dict (needed for 4-pt prompt + containment check)
        """
        n      = len(measurements)
        addr   = anchors.address
        impact = anchors.impact  # hard cap: window ends HERE

        # P5 = first transition frame; fall back to addr
        p5_fr = next(
            (i for i, p in enumerate(phase_labels) if p == "transition"),
            addr
        )

        # Window: P5 → impact (inclusive) — HARD CAP, no post-impact frames
        window_frames = list(range(p5_fr, min(impact + 1, n)))
        # Always include address (may be before P5)
        if addr not in window_frames:
            window_frames = [addr] + window_frames

        # Address biomechanics
        addr_m  = measurements[addr]
        torso_h = addr_m.torso_height()
        if torso_h <= 0:
            ths = [m.torso_height() for m in measurements if m.torso_height() > 0]
            torso_h = float(np.median(ths)) if ths else 200.0

        addr_hip = addr_m.hip_mid()
        if addr_hip is None:
            sh = addr_m.shoulder_mid()
            addr_hip = (sh[0], sh[1] + torso_h * 0.5) if sh else (360.0, 400.0)

        band_y_lo = addr_hip[1] - BAND_FRAC * torso_h
        band_y_hi = addr_hip[1] + BAND_FRAC * torso_h

        toward_ball_sign = 1 if ball_side == "right" else -1
        rear_is_left     = (ball_side == "right")

        # Output arrays
        hip_rear_disp = np.full(n, np.nan)
        mask_quality  = np.full(n, np.nan)
        nan_reasons: dict[int, str] = {}

        predictor = self._get_predictor()
        cap = cv2.VideoCapture(str(video_path))
        addr_rear_x = None

        frames_raw = kp_json.get("frames", [])

        try:
            for fr in window_frames:
                if fr < 0 or fr >= n or fr >= len(frames_raw):
                    nan_reasons[fr] = "out_of_range"
                    continue

                fd  = frames_raw[fr]
                kps = fd["persons"][0]["keypoints"] if fd.get("persons") else {}

                # --- Build 4-point prompt ---
                coords, labels = _build_4pt_prompt(kps, self.kp_thr)
                if coords is None:
                    nan_reasons[fr] = "prompt_insufficient_joints"
                    continue

                # Per-frame hip_mid for geometry check
                lh = kp_guard(kps, "left_hip",  self.kp_thr)
                rh = kp_guard(kps, "right_hip", self.kp_thr)
                if lh and rh:
                    frame_hip_mid = _mid(lh, rh)
                elif lh:
                    frame_hip_mid = lh
                elif rh:
                    frame_hip_mid = rh
                else:
                    nan_reasons[fr] = "hip_mid_unavailable"
                    continue

                # Read video frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
                ret, frame_bgr = cap.read()
                if not ret:
                    nan_reasons[fr] = "frame_read_fail"
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img_h, img_w = frame_rgb.shape[:2]

                # SAM2 prediction
                predictor.set_image(frame_rgb)
                masks, scores, _ = predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    multimask_output=False,
                )
                mask  = masks[0].astype(bool)
                score = float(scores[0])
                mask_quality[fr] = score

                # --- Containment check: ≥ MIN_VALID_JOINTS joints inside mask ---
                joints_in = _count_joints_in_mask(mask, kps, self.kp_thr)
                if joints_in < MIN_VALID_JOINTS:
                    nan_reasons[fr] = f"containment_fail({joints_in}<{MIN_VALID_JOINTS})"
                    continue

                # Hip band crop
                row_lo = max(0, int(band_y_lo))
                row_hi = min(img_h - 1, int(band_y_hi) + 1)
                band_mask = mask[row_lo:row_hi, :]
                if int(np.sum(band_mask)) < MIN_BAND_PX:
                    nan_reasons[fr] = f"band_px_fail(<{MIN_BAND_PX})"
                    continue

                # Find rear edge
                col_occ   = np.any(band_mask, axis=0)
                occ_cols  = np.where(col_occ)[0]
                if len(occ_cols) == 0:
                    nan_reasons[fr] = "no_occupied_cols"
                    continue

                raw_rear_x = float(occ_cols.min() if rear_is_left else occ_cols.max())

                # --- Geometry validity: |rear_x - hip_mid_x| ∈ [0.05, 0.60] × torso_h ---
                dist = abs(raw_rear_x - frame_hip_mid[0])
                if dist < REAR_DIST_MIN * torso_h:
                    nan_reasons[fr] = (f"geo_too_close(dist={dist:.1f}<"
                                       f"{REAR_DIST_MIN*torso_h:.1f})")
                    continue
                if dist > REAR_DIST_MAX * torso_h:
                    nan_reasons[fr] = (f"geo_too_far(dist={dist:.1f}>"
                                       f"{REAR_DIST_MAX*torso_h:.1f})")
                    continue

                if fr == addr:
                    addr_rear_x = raw_rear_x

                hip_rear_disp[fr] = raw_rear_x   # normalise after addr is set

        finally:
            cap.release()

        # Normalise relative to address
        if addr_rear_x is None:
            valid_frs = [f for f in window_frames if not np.isnan(hip_rear_disp[f])]
            addr_rear_x = float(hip_rear_disp[valid_frs[0]]) if valid_frs else 0.0

        for fr in window_frames:
            if not np.isnan(hip_rear_disp[fr]):
                raw_x = hip_rear_disp[fr]
                hip_rear_disp[fr] = ((raw_x - addr_rear_x)
                                     * toward_ball_sign
                                     / max(torso_h, 1.0))

        nan_count = len(nan_reasons)

        meta = {
            "version":          "v1.1",
            "addr_fr":          addr,
            "impact_fr":        impact,
            "p5_fr":            p5_fr,
            "window_end":       impact,   # hard cap
            "ball_side":        ball_side,
            "toward_ball_sign": toward_ball_sign,
            "band_y_lo":        round(band_y_lo, 1),
            "band_y_hi":        round(band_y_hi, 1),
            "torso_h":          round(torso_h, 1),
            "addr_rear_x":      round(addr_rear_x, 1),
            "nan_count":        nan_count,
            "nan_reasons":      nan_reasons,
        }

        return HipRearResult(
            hip_rear_disp=hip_rear_disp,
            mask_quality=mask_quality,
            addr_rear_x=addr_rear_x,
            torso_h=torso_h,
            band_y_lo=band_y_lo,
            band_y_hi=band_y_hi,
            toward_ball_sign=toward_ball_sign,
            window_frames=window_frames,
            nan_count=nan_count,
            nan_reasons=nan_reasons,
            meta=meta,
        )
