"""
engine/profiler/swing_type_detector.py
Layer 1 Video Profiler — swing_type detection from keypoint motion proxy.

Classifies: full_swing / static_demo / mixed / unknown
Uses frame-to-frame wrist-Y motion proxy (no video file needed — works from kp_json).
If wrists KPs are sparse, falls back to shoulder-Y motion.

Rules (from split_motion_diag.py calibration):
  peak_motion >= SWING_THR  → full_swing
  peak_motion in [BORDERLINE_LO, SWING_THR) → mixed (borderline)
  peak_motion < BORDERLINE_LO → static_demo

Confidence:
  clear full_swing (peak >= 2×SWING_THR):      0.90
  borderline full_swing:                         0.65
  clear static_demo (peak < 0.5×SWING_THR):   0.90
  borderline static_demo:                        0.65
  mixed (multiple swings detected):             0.80
"""

from __future__ import annotations
import math
from typing import Optional

KP_THR: float = 0.25         # lower threshold for wrists (often occluded mid-swing)
SWING_THR: float = 25.0      # wrist-Y px/frame peak to classify as full swing
BORDERLINE_LO: float = 10.0  # below this = definitely static

# Sampling: scan full clip, 3-frame steps for efficiency
STEP: int = 3


def _get_kps(fd: dict) -> dict:
    persons = fd.get("persons", [])
    if not persons:
        return {}
    return persons[0].get("keypoints", {})


def _safe_y(kps: dict, name: str, thr: float = KP_THR) -> Optional[float]:
    k = kps.get(name, {})
    if k.get("score", 0.0) >= thr:
        return k["y"]
    return None


def _wrist_mid_y(kps: dict) -> Optional[float]:
    lw = _safe_y(kps, "left_wrist")
    rw = _safe_y(kps, "right_wrist")
    if lw is not None and rw is not None:
        return (lw + rw) / 2.0
    return lw or rw


def _shoulder_mid_y(kps: dict) -> Optional[float]:
    ls = _safe_y(kps, "left_shoulder", thr=0.30)
    rs = _safe_y(kps, "right_shoulder", thr=0.30)
    if ls is not None and rs is not None:
        return (ls + rs) / 2.0
    return ls or rs


def detect_swing_type(kp_json: dict) -> dict:
    """
    Detect swing_type from kp_json.
    Returns dict with: swing_type, confidence, peak_motion_px, method, valid_frames
    """
    frames = kp_json.get("frames", [])
    n = len(frames)
    if n < 4:
        return {
            "swing_type": "unknown",
            "confidence": 0.1,
            "peak_motion_px": None,
            "method": "too_few_frames",
            "valid_frames": 0,
        }

    # Build wrist-Y series
    ys = []
    for i in range(0, n, STEP):
        kps = _get_kps(frames[i])
        y = _wrist_mid_y(kps)
        ys.append((i, y))

    wrist_valid = sum(1 for _, y in ys if y is not None)
    method = "wrist_y"

    # Fallback to shoulder if wrists too sparse
    if wrist_valid < len(ys) * 0.4:
        ys = []
        for i in range(0, n, STEP):
            kps = _get_kps(frames[i])
            y = _shoulder_mid_y(kps)
            ys.append((i, y))
        method = "shoulder_y_fallback"
        wrist_valid = sum(1 for _, y in ys if y is not None)

    if wrist_valid < 4:
        return {
            "swing_type": "unknown",
            "confidence": 0.2,
            "peak_motion_px": None,
            "method": "insufficient_kp",
            "valid_frames": wrist_valid,
        }

    # Compute frame-to-frame deltas
    ys_vals = [(i, y) for i, y in ys if y is not None]
    deltas = []
    for k in range(len(ys_vals) - 1):
        _, y0 = ys_vals[k]
        _, y1 = ys_vals[k + 1]
        deltas.append(abs(y1 - y0))

    if not deltas:
        return {
            "swing_type": "unknown",
            "confidence": 0.2,
            "peak_motion_px": None,
            "method": "no_deltas",
            "valid_frames": wrist_valid,
        }

    # Smooth with simple moving average (window=3)
    smoothed = []
    for k in range(len(deltas)):
        lo = max(0, k - 1); hi = min(len(deltas), k + 2)
        smoothed.append(sum(deltas[lo:hi]) / (hi - lo))

    peak = max(smoothed)

    # Classify
    if peak >= SWING_THR * 2.0:
        swing_type = "full_swing"; conf = 0.90
    elif peak >= SWING_THR:
        swing_type = "full_swing"; conf = 0.75
    elif peak >= BORDERLINE_LO:
        swing_type = "mixed"; conf = 0.60
    else:
        swing_type = "static_demo"; conf = 0.88

    return {
        "swing_type": swing_type,
        "confidence": round(conf, 3),
        "peak_motion_px": round(peak, 2),
        "method": method,
        "valid_frames": wrist_valid,
    }
