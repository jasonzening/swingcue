"""
D-layer Rule Judges — first closed loop (two rules + bone sentinel).

All rules operate on feature sequences (numpy arrays), not video frames.
Thresholds are configurable parameters, not hardcoded magic numbers.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Shared output dataclass
# ---------------------------------------------------------------------------

@dataclass
class FaultDetection:
    """Unified fault output format from D-layer rule judges."""
    fault_type: str
    phase_window: str
    evidence: dict
    severity: str          # "mild" | "significant" | "none"
    confidence: float      # 0.0 – 1.0
    # onset_frame is inside evidence dict; kept separate for E-layer timing
    onset_frame: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "fault_type": self.fault_type,
            "phase_window": self.phase_window,
            "evidence": self.evidence,
            "severity": self.severity,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# Bone-length sentinel (physical judge, minimum version)
# ---------------------------------------------------------------------------

def bone_length_sentinel(
    bone_length_ratios: dict[str, np.ndarray],
    change_threshold: float = 0.20,
) -> np.ndarray:
    """
    Flag frames where any tracked bone length deviates >change_threshold
    (default 20%) from the per-bone median (address baseline).

    Parameters
    ----------
    bone_length_ratios : dict
        bone_name -> 1-D array of per-frame ratios vs. that bone's median.
        A ratio of 1.0 means no change; 1.25 means +25% change.
    change_threshold : float
        Fraction deviation that marks a frame as unreliable (default 0.20).

    Returns
    -------
    unreliable_mask : np.ndarray (bool)
        True at every frame that should be excluded from rule evaluation.
    """
    if not bone_length_ratios:
        # No bone data provided — can't flag anything
        return np.array([], dtype=bool)

    # Determine total number of frames from first bone array
    first = next(iter(bone_length_ratios.values()))
    n_frames = len(first)
    unreliable = np.zeros(n_frames, dtype=bool)

    for bone_name, ratios in bone_length_ratios.items():
        arr = np.asarray(ratios, dtype=float)
        # Deviation = |ratio - 1.0|
        deviation = np.abs(arr - 1.0)
        unreliable |= deviation > change_threshold

    return unreliable


# ---------------------------------------------------------------------------
# R1: loss_of_posture (spine angle change)
# ---------------------------------------------------------------------------

def r1_loss_of_posture(
    spine_angle_deltas: np.ndarray,
    phase_labels: list[str],
    joint_confidences: Optional[np.ndarray] = None,
    unreliable_mask: Optional[np.ndarray] = None,
    # --- configurable thresholds ---
    spine_threshold_deg: float = 8.0,
    mild_max_deg: float = 12.0,
    min_consecutive_frames: int = 3,
    min_joint_confidence: float = 0.4,
    active_phases: tuple[str, ...] = ("downswing", "impact"),
) -> Optional[FaultDetection]:
    """
    R1: Detect loss of posture (spine straightening beyond threshold).

    Parameters
    ----------
    spine_angle_deltas : np.ndarray
        Per-frame spine angle delta from address baseline (degrees).
        Positive = spine becoming more upright (loss of forward tilt).
    phase_labels : list[str]
        Phase name for each frame (length must match spine_angle_deltas).
    joint_confidences : np.ndarray, optional
        Per-frame minimum confidence across shoulder/hip joints (0–1).
        If None, all frames are assumed fully confident.
    unreliable_mask : np.ndarray (bool), optional
        Frames flagged by bone_length_sentinel. These are excluded from
        the consecutive-frame count and reduce overall confidence.
    spine_threshold_deg : float
        Delta in degrees above which "straightening" is considered a fault.
    mild_max_deg : float
        Delta above which severity upgrades from "mild" to "significant".
    min_consecutive_frames : int
        Minimum consecutive valid frames above threshold to trigger.
    min_joint_confidence : float
        If mean joint confidence in window is below this, rule is skipped.
    active_phases : tuple[str]
        Phase names in which this rule is evaluated.

    Returns
    -------
    FaultDetection or None
    """
    spine_deltas = np.asarray(spine_angle_deltas, dtype=float)
    n = len(spine_deltas)

    if joint_confidences is None:
        joint_confidences = np.ones(n, dtype=float)
    else:
        joint_confidences = np.asarray(joint_confidences, dtype=float)

    if unreliable_mask is None:
        unreliable_mask = np.zeros(n, dtype=bool)
    else:
        unreliable_mask = np.asarray(unreliable_mask, dtype=bool)

    # Build active-phase mask
    phase_arr = np.array(phase_labels)
    in_window = np.isin(phase_arr, list(active_phases))

    # Check joint confidence in window
    window_conf = joint_confidences[in_window]
    if len(window_conf) == 0 or float(np.mean(window_conf)) < min_joint_confidence:
        return None

    # Find frames: in window, reliable, above threshold
    above_thresh = spine_deltas >= spine_threshold_deg
    valid_frame = in_window & ~unreliable_mask & above_thresh

    # Find longest run of consecutive valid frames
    onset, best_run = _longest_run(valid_frame)
    if best_run < min_consecutive_frames:
        return None

    # Gather metrics over triggered window
    triggered_indices = np.where(valid_frame)[0]
    peak_value = float(np.max(spine_deltas[triggered_indices]))
    frames_sustained = int(np.sum(valid_frame))
    conf = float(np.mean(window_conf))

    # Adjust confidence downward if many unreliable frames were present
    unreliable_ratio = float(np.mean(unreliable_mask[in_window])) if np.any(in_window) else 0.0
    conf = conf * (1.0 - 0.5 * unreliable_ratio)

    severity = "mild" if peak_value <= mild_max_deg else "significant"

    return FaultDetection(
        fault_type="loss_of_posture",
        phase_window="downswing-impact",
        evidence={
            "feature": "spine_delta",
            "peak_value": round(peak_value, 2),
            "onset_frame": int(onset),
            "frames_sustained": frames_sustained,
        },
        severity=severity,
        confidence=round(conf, 4),
        onset_frame=int(onset),
    )


# ---------------------------------------------------------------------------
# R2: hip_toward_ball (hip forward displacement)
# ---------------------------------------------------------------------------

def r2_hip_toward_ball(
    hip_forward_displacements: np.ndarray,
    phase_labels: list[str],
    joint_confidences: Optional[np.ndarray] = None,
    unreliable_mask: Optional[np.ndarray] = None,
    # --- configurable thresholds ---
    hip_threshold: float = 0.05,
    mild_max: float = 0.09,
    min_consecutive_frames: int = 3,
    min_joint_confidence: float = 0.4,
    active_phases: tuple[str, ...] = ("transition", "downswing", "impact"),
) -> Optional[FaultDetection]:
    """
    R2: Detect hip forward displacement toward ball.

    Parameters
    ----------
    hip_forward_displacements : np.ndarray
        Per-frame hip displacement normalized by torso height.
        Positive = toward ball (ball direction as defined by address wrist midpoint).
    phase_labels : list[str]
        Phase name per frame.
    joint_confidences : np.ndarray, optional
        Per-frame hip/wrist joint confidence.
    unreliable_mask : np.ndarray (bool), optional
        Frames flagged by bone_length_sentinel.
    hip_threshold : float
        Displacement fraction above which hip-toward-ball is a fault.
    mild_max : float
        Displacement above which severity upgrades to "significant".
    min_consecutive_frames : int
        Minimum consecutive frames to trigger.
    min_joint_confidence : float
        Confidence threshold; rule skipped if below.
    active_phases : tuple[str]
        Phases in which this rule is evaluated (starts at transition).

    Returns
    -------
    FaultDetection or None
    """
    hip_disp = np.asarray(hip_forward_displacements, dtype=float)
    n = len(hip_disp)

    if joint_confidences is None:
        joint_confidences = np.ones(n, dtype=float)
    else:
        joint_confidences = np.asarray(joint_confidences, dtype=float)

    if unreliable_mask is None:
        unreliable_mask = np.zeros(n, dtype=bool)
    else:
        unreliable_mask = np.asarray(unreliable_mask, dtype=bool)

    phase_arr = np.array(phase_labels)
    in_window = np.isin(phase_arr, list(active_phases))

    window_conf = joint_confidences[in_window]
    if len(window_conf) == 0 or float(np.mean(window_conf)) < min_joint_confidence:
        return None

    above_thresh = hip_disp >= hip_threshold
    valid_frame = in_window & ~unreliable_mask & above_thresh

    onset, best_run = _longest_run(valid_frame)
    if best_run < min_consecutive_frames:
        return None

    triggered_indices = np.where(valid_frame)[0]
    peak_value = float(np.max(hip_disp[triggered_indices]))
    frames_sustained = int(np.sum(valid_frame))
    conf = float(np.mean(window_conf))

    unreliable_ratio = float(np.mean(unreliable_mask[in_window])) if np.any(in_window) else 0.0
    conf = conf * (1.0 - 0.5 * unreliable_ratio)

    severity = "mild" if peak_value <= mild_max else "significant"

    return FaultDetection(
        fault_type="hip_toward_ball",
        phase_window="transition-impact",
        evidence={
            "feature": "hip_shift",
            "peak_value": round(peak_value, 4),
            "onset_frame": int(onset),
            "frames_sustained": frames_sustained,
        },
        severity=severity,
        confidence=round(conf, 4),
        onset_frame=int(onset),
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    """
    Find the start index and length of the longest consecutive True run.

    Returns (onset_index, run_length). Returns (0, 0) if no True values.
    """
    best_start = 0
    best_len = 0
    current_start = 0
    current_len = 0

    for i, val in enumerate(mask):
        if val:
            if current_len == 0:
                current_start = i
            current_len += 1
            if current_len > best_len:
                best_len = current_len
                best_start = current_start
        else:
            current_len = 0

    return best_start, best_len
