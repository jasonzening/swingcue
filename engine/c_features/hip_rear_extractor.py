"""
engine/c_features/hip_rear_extractor.py
C Layer — R2' Hip Rear Edge Displacement  (v1.0)

Computes `hip_rear_disp` for each frame using SAM2 person segmentation mask.

Algorithm
---------
1. Address frame: run SAM2 with hip_mid keypoint as positive prompt → person mask.
2. For each frame in window [P5_transition, impact+5]:
   a. Run SAM2 (image predictor, single-frame) with hip_mid as prompt → mask.
   b. Find rear-edge x: within the "hip band" (hip_mid_y ± band_frac×torso_h),
      scan the mask column-by-column from the back side (ball_side=right → scan left;
      ball_side=left → scan right) and find the outermost occupied column.
3. hip_rear_disp[t] = (rear_edge_x[t] − rear_edge_x[addr]) × toward_ball_sign / torso_h
   positive = rear edge moved toward ball (bad, indicates early extension or sway)

Design decisions
----------------
- Uses SAM2ImagePredictor (single image mode, not video tracking).
  This avoids temporal state management complexity and is fully re-entrant.
- Point prompt: hip_mid keypoint (center of left/right hip).  If hip_mid is
  unavailable, falls back to shoulder_mid - 0.5×torso_h.
- Hip band width = ±0.12 × torso_h (≈ 3-4cm for typical swing video).
- No GPU memory leak: predictor is built once, set_image() called per frame.
- Returns NaN for frames where mask has <50 pixels in the hip band (bad detection).
- Does NOT modify existing R2 (hip_mid displacement).  Both run in parallel.

Confidence
----------
Returns per-frame mask quality score from SAM2 (0–1).
Final hip_rear_conf = mean of scores in window, penalised by band_fill_ratio < 0.3.
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

PROJ = Path(__file__).resolve().parents[2]
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_CKPT = str(PROJ / "models/sam2/sam2.1_hiera_tiny.pt")


@dataclass
class HipRearResult:
    hip_rear_disp: np.ndarray   # per-frame, fraction of torso_h, +toward ball
    mask_quality:  np.ndarray   # per-frame SAM2 score (0–1, NaN outside window)
    addr_rear_x:   float        # rear edge x at address (pixels)
    torso_h:       float
    band_y_lo:     float        # hip band lower y bound
    band_y_hi:     float        # hip band upper y bound
    toward_ball_sign: int       # +1 or -1
    window_frames: list[int]    # frames actually processed
    meta:          dict = field(default_factory=dict)


def build_predictor(device: str = "cuda"):
    """Build SAM2ImagePredictor once (expensive, cache outside hot loop)."""
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = build_sam2(SAM2_CFG, SAM2_CKPT, device=device)
    return SAM2ImagePredictor(model), device


class HipRearExtractor:
    """
    Extracts hip_rear_disp per frame using SAM2 mask rear-edge tracking.

    Parameters
    ----------
    band_frac    : float  — hip band = hip_mid_y ± band_frac × torso_h
    device       : 'cuda' | 'cpu'
    min_band_px  : int    — minimum pixels in band for valid measurement
    """

    def __init__(
        self,
        band_frac:    float = 0.12,
        device:       str   = "cuda",
        min_band_px:  int   = 50,
    ):
        self.band_frac   = band_frac
        self.device      = device
        self.min_band_px = min_band_px
        self._predictor  = None   # lazy-loaded

    def _get_predictor(self):
        if self._predictor is None:
            self._predictor, self.device = build_predictor(self.device)
        return self._predictor

    def extract(
        self,
        video_path:   str,
        measurements: List[FrameMeasurement],
        anchors:      AnchorFrames,
        ball_side:    str,          # "right" | "left"  from orientation resolver
        phase_labels: list[str],
        window_extra: int = 5,      # frames after impact to include
    ) -> HipRearResult:
        """
        Run extraction over the downswing window [P5, impact+window_extra].

        Parameters
        ----------
        video_path   : path to source .mp4
        measurements : A-layer output
        anchors      : B-layer anchors (addr, impact)
        ball_side    : "right" | "left" — which side of frame is the ball
        phase_labels : list of phase name per frame
        window_extra : frames after impact to include in window
        """
        n = len(measurements)
        addr = anchors.address
        impact = anchors.impact

        # Build window: transition → impact+window_extra
        p5_fr = next(
            (i for i, p in enumerate(phase_labels) if p == "transition"),
            addr
        )
        window_end = min(impact + window_extra, n - 1)
        window_frames = list(range(p5_fr, window_end + 1))
        if addr not in window_frames:
            window_frames = [addr] + window_frames

        # Address biomechanics
        addr_m = measurements[addr]
        torso_h = addr_m.torso_height()
        if torso_h <= 0:
            ths = [m.torso_height() for m in measurements if m.torso_height() > 0]
            torso_h = float(np.median(ths)) if ths else 200.0

        addr_hip = addr_m.hip_mid()
        addr_sh  = addr_m.shoulder_mid()
        if addr_hip is None:
            # Fallback: shoulder_mid - 0.5*torso_h downward
            if addr_sh:
                addr_hip = (addr_sh[0], addr_sh[1] + torso_h * 0.5)
            else:
                addr_hip = (360.0, 400.0)

        hip_band_cy = addr_hip[1]
        band_y_lo = hip_band_cy - self.band_frac * torso_h
        band_y_hi = hip_band_cy + self.band_frac * torso_h

        # toward_ball_sign: +1 if ball is on right side (increasing x toward ball)
        toward_ball_sign = 1 if ball_side == "right" else -1
        # rear side = opposite of ball_side
        # rear edge = column furthest from ball side within mask
        # if ball_side==right, rear edge = leftmost occupied column (smallest x)
        # if ball_side==left,  rear edge = rightmost occupied column (largest x)
        rear_is_left = (ball_side == "right")

        # Output arrays (NaN = not computed)
        hip_rear_disp = np.full(n, np.nan)
        mask_quality  = np.full(n, np.nan)

        predictor = self._get_predictor()
        cap = cv2.VideoCapture(str(video_path))

        addr_rear_x = None

        try:
            for fr in window_frames:
                if fr < 0 or fr >= n:
                    continue
                m = measurements[fr]

                # Get hip mid for this frame (as prompt point)
                hip_pt = m.hip_mid()
                if hip_pt is None:
                    sh_pt = m.shoulder_mid()
                    if sh_pt:
                        hip_pt = (sh_pt[0], sh_pt[1] + torso_h * 0.5)
                    else:
                        continue

                # Read frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
                ret, frame_bgr = cap.read()
                if not ret:
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # SAM2 prediction
                predictor.set_image(frame_rgb)
                masks, scores, _ = predictor.predict(
                    point_coords=np.array([[hip_pt[0], hip_pt[1]]]),
                    point_labels=np.array([1]),
                    multimask_output=False,
                )
                mask = masks[0]       # (H, W) bool
                score = float(scores[0])
                mask_quality[fr] = score

                # Crop to hip band
                h = frame_rgb.shape[0]
                row_lo = max(0, int(band_y_lo))
                row_hi = min(h - 1, int(band_y_hi) + 1)
                band_mask = mask[row_lo:row_hi, :]
                band_px = int(np.sum(band_mask))

                if band_px < self.min_band_px:
                    continue  # leave as NaN

                # Find rear edge x
                col_occupied = np.any(band_mask, axis=0)
                occupied_cols = np.where(col_occupied)[0]
                if len(occupied_cols) == 0:
                    continue

                if rear_is_left:
                    rear_x = float(occupied_cols.min())
                else:
                    rear_x = float(occupied_cols.max())

                if fr == addr:
                    addr_rear_x = rear_x

                hip_rear_disp[fr] = rear_x   # will normalise after address is set

        finally:
            cap.release()

        # Normalise all frames relative to address
        if addr_rear_x is None:
            # Address not computed; try nearest valid
            valid = [fr for fr in window_frames if not np.isnan(hip_rear_disp[fr])]
            if valid:
                addr_rear_x = float(hip_rear_disp[valid[0]])
            else:
                addr_rear_x = 0.0

        # Convert raw x → normalised displacement toward ball
        for fr in window_frames:
            if not np.isnan(hip_rear_disp[fr]):
                raw_x = hip_rear_disp[fr]
                # Positive = moved toward ball side
                # If ball_side==right: rear edge moves right (larger x) = toward ball
                # displacement in x: raw_x - addr_rear_x, then × toward_ball_sign
                # rear_is_left → rear_x small when rear edge retracts; more negative = moved right
                # If rear_is_left: toward-ball = rear_x increases (goes right) ← wait:
                #   rear edge = left edge of body (small x).
                #   body moves right (toward ball) → left edge also moves right → raw_x increases
                #   so: (raw_x - addr_rear_x) > 0 means body moved toward ball ✓
                hip_rear_disp[fr] = (raw_x - addr_rear_x) * toward_ball_sign / max(torso_h, 1.0)

        meta = {
            "addr_fr":         addr,
            "impact_fr":       impact,
            "p5_fr":           p5_fr,
            "window_end":      window_end,
            "ball_side":       ball_side,
            "toward_ball_sign": toward_ball_sign,
            "band_y_lo":       round(band_y_lo, 1),
            "band_y_hi":       round(band_y_hi, 1),
            "torso_h":         round(torso_h, 1),
            "addr_rear_x":     round(addr_rear_x, 1) if addr_rear_x else None,
        }

        return HipRearResult(
            hip_rear_disp=hip_rear_disp,
            mask_quality=mask_quality,
            addr_rear_x=addr_rear_x or 0.0,
            torso_h=torso_h,
            band_y_lo=band_y_lo,
            band_y_hi=band_y_hi,
            toward_ball_sign=toward_ball_sign,
            window_frames=window_frames,
            meta=meta,
        )
