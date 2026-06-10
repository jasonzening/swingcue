"""
engine/c_features/feature_extractor.py
C Layer — Feature Extraction (first closed loop, DTL only)

Extracts two biomechanical features per frame from FrameMeasurement data:
  1. spine_angle_delta  : change in spine forward-tilt from address baseline (degrees)
  2. hip_forward_disp   : hip displacement toward ball, normalised by torso height

Both features use the ADDRESS frame as the individual baseline (not a population norm).
Only enabled for down-the-line (DTL) angle.  Returns zero-arrays for face-on.

Outputs
-------
FeatureResult:
  spine_delta   : np.ndarray[float], shape (n,) — degrees, positive = rising/straightening
  hip_disp      : np.ndarray[float], shape (n,) — fraction of torso_h, positive = toward ball
  joint_conf    : np.ndarray[float], shape (n,) — per-frame confidence (mean shoulder+hip score)
  unreliable    : np.ndarray[bool],  shape (n,) — bone-length sentinel flags
  address_frame : int
  torso_h       : float  — px, at address frame
  meta          : dict   — debug info (baseline angles, hip_ball_dir, etc.)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.signal import savgol_filter

from engine.a_measurement.pose_pipeline import FrameMeasurement


@dataclass
class FeatureResult:
    spine_delta:   np.ndarray   # degrees per frame
    hip_disp:      np.ndarray   # normalised fraction per frame
    joint_conf:    np.ndarray   # 0–1 per frame
    unreliable:    np.ndarray   # bool per frame (bone-length sentinel)
    address_frame: int
    torso_h:       float
    meta:          dict = field(default_factory=dict)


def _sg(arr: np.ndarray, fps: float, ms: int = 200) -> np.ndarray:
    w = max(7, int(fps * ms / 1000)) | 1
    return savgol_filter(arr, w, 3)


class FeatureExtractor:
    """
    C-layer feature extractor for the first closed-loop diagnostic.

    Parameters
    ----------
    sg_window_ms : smoothing window for spine/hip signals (ms)
    bone_sentinel_thr : bone-length deviation fraction that marks a frame unreliable
    """

    def __init__(self, sg_window_ms: int = 200, bone_sentinel_thr: float = 0.20):
        self.sg_window_ms    = sg_window_ms
        self.bone_sentinel_thr = bone_sentinel_thr

    def extract(
        self,
        measurements: List[FrameMeasurement],
        fps: float,
        angle: str,
        address_frame: int,
    ) -> FeatureResult:
        """
        Run C-layer extraction.

        Parameters
        ----------
        measurements  : A-layer output (one per frame, sorted by frame_idx)
        fps           : source video fps
        angle         : "down-the-line" | "face-on"
        address_frame : frame index of address anchor (from B layer)

        Returns
        -------
        FeatureResult
        """
        n = len(measurements)

        # Zero-fill arrays
        spine_raw  = np.zeros(n, dtype=float)
        hip_raw    = np.zeros(n, dtype=float)
        conf_arr   = np.zeros(n, dtype=float)
        unreliable = np.zeros(n, dtype=bool)

        # ── Address baseline ──────────────────────────────────────────────────
        addr_m = measurements[min(address_frame, n-1)]
        torso_h = addr_m.torso_height()
        if torso_h <= 0:
            # Fallback: median torso height across all frames
            ths = [m.torso_height() for m in measurements if m.torso_height() > 0]
            torso_h = float(np.median(ths)) if ths else 200.0

        # Spine angle at address
        addr_sh = addr_m.shoulder_mid()
        addr_hp = addr_m.hip_mid()
        if addr_sh is not None and addr_hp is not None:
            addr_spine_angle = _spine_angle_deg(addr_sh, addr_hp)
        else:
            addr_spine_angle = 0.0

        # Hip mid-x at address → defines the "toward-ball" axis (DTL: +x = toward ball)
        addr_hip_x = addr_hp[0] if addr_hp is not None else 0.0
        addr_hip_y = addr_hp[1] if addr_hp is not None else 0.0

        # Ball direction: in DTL view, "toward ball" = increasing x
        # (golfer stands to the right of ball in standard DTL footage, ball is left/forward)
        # We use SIGNED displacement from address hip_x:
        #   positive = hip moved in +x direction (toward ball side)
        # The sign convention is fixed per-video by whichever side the ball is on.
        # For the first closed loop we simply use signed x-displacement and let
        # the rule threshold catch any camera-flip issues.

        # Bone-length sentinel: rolling per-video median for each tracked segment
        # Hip/knee bones only — arm bones excluded because perspective foreshortening
        # during downswing causes systematic 25-40% apparent length change (DTL views).
        # That is normal biomechanics, not keypoint error.  Using arm bones causes
        # 50%+ of frames to be flagged as unreliable on every normal swing.
        bone_keys = ["left_hip_left_knee", "right_hip_right_knee"]
        bone_medians: dict[str, float] = {}
        for bk in bone_keys:
            lengths = [m.bone_lengths.get(bk, 0.0) for m in measurements
                       if m.bone_lengths.get(bk, 0.0) > 0]
            if lengths:
                bone_medians[bk] = float(np.median(lengths))

        # ── Per-frame extraction ──────────────────────────────────────────────
        for fi, m in enumerate(measurements):
            # Quality factor
            q = (1.0 if m.measurement_quality == "ok"
                 else 0.5 if m.measurement_quality == "degraded" else 0.1)

            # Bone sentinel
            for bk, med in bone_medians.items():
                bl = m.bone_lengths.get(bk, 0.0)
                if bl > 0 and med > 0 and abs(bl / med - 1.0) > self.bone_sentinel_thr:
                    unreliable[fi] = True
                    break

            # Joint confidence: mean of shoulder+hip keypoint scores
            sh_conf  = np.mean([m.confidences.get("left_shoulder",  0.0),
                                m.confidences.get("right_shoulder", 0.0)])
            hip_conf = np.mean([m.confidences.get("left_hip",  0.0),
                                m.confidences.get("right_hip", 0.0)])
            conf_arr[fi] = float((sh_conf + hip_conf) / 2.0) * q

            if angle != "down-the-line":
                # C-layer features are DTL-only in first closed loop
                continue

            sh  = m.shoulder_mid()
            hp  = m.hip_mid()

            # Spine angle delta
            if sh is not None and hp is not None:
                curr_angle = _spine_angle_deg(sh, hp)
                # Positive delta = spine becoming more upright (loss of forward tilt)
                spine_raw[fi] = curr_angle - addr_spine_angle
            else:
                spine_raw[fi] = 0.0

            # Hip forward displacement (normalised by torso_h)
            if hp is not None and torso_h > 0:
                dx = hp[0] - addr_hip_x
                # Positive = moved in +x direction from address
                hip_raw[fi] = dx / torso_h
            else:
                hip_raw[fi] = 0.0

        # ── Smoothing (only for DTL) ──────────────────────────────────────────
        if angle == "down-the-line":
            spine_delta = _sg(spine_raw, fps, self.sg_window_ms)
            hip_disp    = _sg(hip_raw,   fps, self.sg_window_ms)
        else:
            spine_delta = spine_raw
            hip_disp    = hip_raw

        meta = {
            "angle":             angle,
            "address_frame":     address_frame,
            "torso_h":           round(torso_h, 1),
            "addr_spine_angle":  round(addr_spine_angle, 2),
            "addr_hip_x":        round(addr_hip_x, 1),
        }

        return FeatureResult(
            spine_delta=spine_delta,
            hip_disp=hip_disp,
            joint_conf=conf_arr,
            unreliable=unreliable,
            address_frame=address_frame,
            torso_h=torso_h,
            meta=meta,
        )


def _spine_angle_deg(shoulder_mid: tuple, hip_mid: tuple) -> float:
    """
    Forward spine tilt from vertical (degrees).
    0° = fully upright.  Positive = leaning forward (address posture).

    Formula: atan2(|Δx|, Δy) where Δy = hip_y - shoulder_y (> 0 since hip is below shoulder)
    The absolute value of Δx prevents sign confusion for slight lateral lean.
    """
    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]   # positive in image coords (y increases downward)
    if dy <= 0:
        return 0.0
    return math.degrees(math.atan2(abs(dx), dy))
