"""
temporal_smoother.py — phase-aware EMA smoothing + outlier rejection.

PR-7_REVIEW_RESPONSE.md Constraint 6 (HARD BLOCK at code review):
  The `phase` argument is REQUIRED on smoothing functions. No fixed
  global alpha. The smoothing config dict per spec v3 §10 PHASE_CONFIG
  is what each call indexes — this enforces phase-aware behavior
  structurally, not by convention.

Outlier rejection: if frame-to-frame motion exceeds
`outlier_ratio` × scale_reference, treat the raw value as an outlier
and reuse the previous smoothed value instead.

EMA formula:
    smoothed = α * raw + (1 - α) * prev
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class SmoothResult:
    """One frame's smoother output + diagnostics."""
    smoothed: Optional[list[float]]
    was_outlier: bool
    alpha_used: float
    outlier_ratio_used: float


def _euclid_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(min(len(a), len(b)))))


def smooth_keypoint_phase_aware(
    raw: Optional[list[float]],
    prev_smoothed: Optional[list[float]],
    *,
    phase: str,
    smoothing_config: dict[str, dict[str, float]],
    scale_reference: Optional[float] = None,
    prev_raw_for_outlier_check: Optional[list[float]] = None,
) -> SmoothResult:
    """
    Apply phase-aware EMA + outlier rejection to one keypoint at one
    frame.

    Args:
        raw:                this frame's raw 3D keypoint, or None if
                             upstream had nothing for this joint.
        prev_smoothed:      last frame's smoothed value, or None on
                             first valid frame for this joint.
        phase:              REQUIRED per Constraint 6. Must be a key in
                             smoothing_config. NO default.
        smoothing_config:   plugin-provided dict
                             {"setup": {"alpha": 0.20, "outlier_ratio": 0.15},
                              "backswing": {"alpha": 0.30, ...}, ...}
        scale_reference:    distance baseline for outlier scaling. Typical
                             choices: setup-baseline body width (~0.5 m).
                             If None, outlier rejection is skipped (just
                             EMA, no reject).
        prev_raw_for_outlier_check:
                             Previous frame's raw value (or whichever
                             value the orchestrator wants the outlier
                             check to compare against). Recommended:
                             prev RAW (sensor-noise measurement). If
                             None, falls back to prev_smoothed for the
                             outlier check — which is unsafe (causes
                             cascading rejections when prev_smoothed gets
                             stuck after a single rejection) but
                             preserved for callers that haven't been
                             updated yet.

    Returns: SmoothResult with the smoothed value + diagnostics.

    Raises:
        KeyError if `phase` is not in smoothing_config (caller bug —
        plugin must declare every phase its phase_detector emits).
    """
    cfg = smoothing_config[phase]   # KeyError surfaces caller bug
    alpha = cfg["alpha"]
    outlier_ratio = cfg.get("outlier_ratio", 0.0)

    # No raw input → carry forward.
    if raw is None:
        return SmoothResult(
            smoothed=list(prev_smoothed) if prev_smoothed is not None else None,
            was_outlier=False,
            alpha_used=alpha,
            outlier_ratio_used=outlier_ratio,
        )

    # First frame for this joint → seed from raw.
    if prev_smoothed is None:
        return SmoothResult(
            smoothed=list(raw),
            was_outlier=False,
            alpha_used=alpha,
            outlier_ratio_used=outlier_ratio,
        )

    # Outlier check measures FRAME-TO-FRAME RAW MOTION when prev_raw is
    # supplied — this is the right signal for sensor-spike detection.
    # Falls back to prev_smoothed for back-compat, but that mode causes
    # the smoother to get permanently stuck after a single spike (every
    # subsequent frame's motion is measured against the frozen value,
    # which only grows as the subject continues to move).
    outlier_reference = (
        prev_raw_for_outlier_check
        if prev_raw_for_outlier_check is not None
        else prev_smoothed
    )
    if scale_reference is not None and outlier_ratio > 0:
        motion = _euclid_distance(raw, outlier_reference)
        threshold = outlier_ratio * scale_reference
        if motion > threshold:
            # Treat raw as outlier; keep prev value.
            return SmoothResult(
                smoothed=list(prev_smoothed),
                was_outlier=True,
                alpha_used=alpha,
                outlier_ratio_used=outlier_ratio,
            )

    # EMA blend.
    smoothed = [
        alpha * raw[i] + (1.0 - alpha) * prev_smoothed[i]
        for i in range(len(raw))
    ]
    return SmoothResult(
        smoothed=smoothed,
        was_outlier=False,
        alpha_used=alpha,
        outlier_ratio_used=outlier_ratio,
    )


def smooth_frame_phase_aware(
    raw_frame: dict[str, Optional[list[float]]],
    prev_smoothed_frame: dict[str, Optional[list[float]]],
    *,
    phase: str,
    smoothing_config: dict[str, dict[str, float]],
    scale_reference: Optional[float] = None,
    prev_raw_frame_for_outlier_check: Optional[
        dict[str, Optional[list[float]]]
    ] = None,
) -> tuple[dict[str, Optional[list[float]]], dict[str, SmoothResult]]:
    """
    Vectorise smoothing over all keypoints in a frame.

    Returns: (smoothed_frame, per_joint_diagnostics).
    Same `phase` applies to every joint at this frame.

    Pass `prev_raw_frame_for_outlier_check` to compare frame-to-frame
    raw motion (sensor-spike detection) instead of comparing to
    accumulated smoothed history (which can get permanently stuck).
    """
    smoothed_frame: dict[str, Optional[list[float]]] = {}
    diagnostics: dict[str, SmoothResult] = {}
    prev_raw_lookup = prev_raw_frame_for_outlier_check or {}
    # Iterate over the union of joint names so missing joints in prev
    # are still seeded from raw and vice versa.
    for name in raw_frame:
        result = smooth_keypoint_phase_aware(
            raw=raw_frame.get(name),
            prev_smoothed=prev_smoothed_frame.get(name),
            phase=phase,
            smoothing_config=smoothing_config,
            scale_reference=scale_reference,
            prev_raw_for_outlier_check=prev_raw_lookup.get(name),
        )
        smoothed_frame[name] = result.smoothed
        diagnostics[name] = result
    return smoothed_frame, diagnostics


# ─────────────────────────────────────────────────────────────────────
# PR-7a.3: Bidirectional EMA — port of pose_timeline._bidirectional_ema_1d
# (commit aaec479) adapted to 3D + phase-aware + Finding H outlier check.
#
# Two-pass forward+backward EMA across the full timeline, averaged per
# frame. Zero phase delay — eliminates the ~100-200 ms causal lag that
# the forward-only smoother introduced. Per spec v3 §0 (offline
# pipeline) this is the default; production streaming (PR-7c) will use
# the forward-only `smooth_frame_phase_aware` path for causality.
#
# Null-aware: a None resets both passes (preserves gap-spanning
# semantics from PR-5.9). When only one pass has a value at a position
# (the other side hit a null), the available side's value is used as-is.
#
# Outlier check (per Finding H): compares 3D-Euclidean motion against
# the PREV RAW value in the iteration direction — forward pass uses
# raw[i-1], backward pass uses raw[i+1]. This prevents the smoother
# from getting stuck after a single sensor spike.
# ─────────────────────────────────────────────────────────────────────

def _smooth_one_pass_3d(
    raw_seq: list[Optional[list[float]]],
    phases: list[str],
    *,
    smoothing_config: dict[str, dict[str, float]],
    scale_reference: Optional[float],
    direction: str,
) -> tuple[list[Optional[list[float]]], list[bool]]:
    """
    One-direction phase-aware EMA over a per-joint 3D timeline.

    Args:
        raw_seq:           list of [x, y, z] | None per frame for ONE joint.
        phases:            phase name per frame (same len as raw_seq).
        smoothing_config:  plugin's PHASE_CONFIG dict.
        scale_reference:   body-width baseline for outlier scaling.
        direction:         "forward" or "backward".

    Returns: (smoothed_seq, was_outlier_seq) both same len as raw_seq.
    Smoothed is in INPUT INDEX ORDER even when direction="backward".
    """
    n = len(raw_seq)
    if n == 0:
        return [], []
    smoothed: list[Optional[list[float]]] = [None] * n
    outliers: list[bool] = [False] * n
    indices = range(n) if direction == "forward" else range(n - 1, -1, -1)

    prev_smoothed: Optional[list[float]] = None
    prev_raw: Optional[list[float]] = None
    for i in indices:
        raw = raw_seq[i]
        cfg = smoothing_config[phases[i]]   # KeyError surfaces caller bug
        alpha = cfg["alpha"]
        outlier_ratio = cfg.get("outlier_ratio", 0.0)

        if raw is None:
            # Per PR-5.9: a None in the input writes None at this position
            # AND resets both prev pointers (no blending spans a gap).
            # The other-direction pass fills this position when its data
            # is present (handled by the average step in the caller).
            smoothed[i] = None
            prev_smoothed = None
            prev_raw = None
            continue

        if prev_smoothed is None:
            # Seed the pass from raw.
            smoothed[i] = list(raw)
            prev_smoothed = list(raw)
            prev_raw = list(raw)
            continue

        # Outlier check (Finding H): compare to prev RAW in this direction.
        if (scale_reference is not None
                and outlier_ratio > 0
                and prev_raw is not None):
            motion = math.sqrt(
                sum((raw[k] - prev_raw[k]) ** 2 for k in range(3))
            )
            if motion > outlier_ratio * scale_reference:
                # Reject this frame in this pass — reuse prev smoothed.
                smoothed[i] = list(prev_smoothed)
                outliers[i] = True
                prev_raw = list(raw)   # still advance prev_raw
                continue

        # EMA blend.
        new_v = [
            alpha * raw[k] + (1.0 - alpha) * prev_smoothed[k]
            for k in range(3)
        ]
        smoothed[i] = new_v
        prev_smoothed = new_v
        prev_raw = list(raw)

    return smoothed, outliers


def smooth_timeline_bidirectional(
    raw_frame_keypoints: list[dict[str, Optional[list[float]]]],
    phase_per_frame: list[str],
    *,
    smoothing_config: dict[str, dict[str, float]],
    scale_reference: Optional[float] = None,
) -> tuple[
    list[dict[str, Optional[list[float]]]],
    list[dict[str, bool]],
    list[dict[str, float]],
]:
    """
    Whole-timeline bidirectional EMA. Zero phase delay.

    Args:
        raw_frame_keypoints: list of {joint_name: [x,y,z]|None} per frame
                             (post-anatomical-offset; what the previous
                             forward-only smoother got per frame).
        phase_per_frame:     phase name per frame, same length.
        smoothing_config:    plugin's PHASE_CONFIG dict.
        scale_reference:     body-width baseline for outlier scaling.

    Returns: (smoothed_frames, outlier_marks, alpha_used_per_frame)
        smoothed_frames    — per-frame {joint_name: [x,y,z]|None}
        outlier_marks      — per-frame {joint_name: bool} (True iff BOTH
                              forward AND backward passes flagged this
                              joint as an outlier at this frame)
        alpha_used_per_frame — per-frame {joint_name: alpha} (per-axis
                              alpha is constant across axes; this just
                              records the phase's α for diagnostics)
    """
    n = len(raw_frame_keypoints)
    if n == 0:
        return [], [], []
    if len(phase_per_frame) != n:
        raise ValueError(
            f"phase_per_frame length {len(phase_per_frame)} != "
            f"frames {n}"
        )
    smoothed_frames: list[dict[str, Optional[list[float]]]] = [
        {} for _ in range(n)
    ]
    outlier_marks: list[dict[str, bool]] = [{} for _ in range(n)]
    alpha_used: list[dict[str, float]] = [{} for _ in range(n)]

    # Union of joint names so missing-in-some-frames joints still flow.
    all_names: set[str] = set()
    for kp in raw_frame_keypoints:
        all_names.update(kp.keys())

    for name in all_names:
        seq = [kp.get(name) for kp in raw_frame_keypoints]
        fwd, fwd_out = _smooth_one_pass_3d(
            seq, phase_per_frame,
            smoothing_config=smoothing_config,
            scale_reference=scale_reference,
            direction="forward",
        )
        bwd, bwd_out = _smooth_one_pass_3d(
            seq, phase_per_frame,
            smoothing_config=smoothing_config,
            scale_reference=scale_reference,
            direction="backward",
        )
        for i in range(n):
            f, b = fwd[i], bwd[i]
            if f is None and b is None:
                smoothed_frames[i][name] = None
            elif f is None:
                smoothed_frames[i][name] = list(b)
            elif b is None:
                smoothed_frames[i][name] = list(f)
            else:
                smoothed_frames[i][name] = [
                    (f[k] + b[k]) / 2.0 for k in range(3)
                ]
            # Outlier only counts as such if BOTH passes rejected the
            # frame — single-side rejection still contributes one valid
            # estimate that the average pulls toward.
            outlier_marks[i][name] = bool(fwd_out[i] and bwd_out[i])
            alpha_used[i][name] = (
                smoothing_config[phase_per_frame[i]]["alpha"]
            )
    return smoothed_frames, outlier_marks, alpha_used
