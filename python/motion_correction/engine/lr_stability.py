"""
lr_stability.py — left/right identity guard.

Pose estimators (WHAM included) occasionally swap L/R labels for body
parts during fast motion or when the far side is occluded. This module
detects swaps post-hoc and corrects the labeling — purely a label
operation, the underlying 3D coords don't change.

PR-7a.1 Fix 2: single-signal (x-coord vs pelvis) swap detection caused
visible thrashing on DTL clips where body rotation makes x alone
insufficient. Now combines two signals:

  (a) X-COORD: left.x and right.x must be on opposite sides of pelvis.x
      (existing logic).
  (b) LIMB-CHAIN: for shoulders, distance(left_shoulder, left_elbow)
      should be smaller than distance(left_shoulder, right_elbow);
      analogous for hips with knees.

Both signals must agree before a swap is even considered. The caller
(orchestrator) then applies cross-frame HYSTERESIS — requires the
"swap-needed" verdict to hold for 3 consecutive frames before flipping
the applied state. This prevents per-frame thrashing when WHAM's raw
output is jittering across the decision boundary.

Generic engine logic; phase context (whether to require strict lock)
still comes from the plugin via the `requires_lock` kwarg.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


def is_swapped_pair(
    left: Optional[list[float]],
    right: Optional[list[float]],
    reference_center: Optional[list[float]],
    *,
    axis_index: int = 0,
) -> bool:
    """
    Check if a (left, right) keypoint pair is swapped relative to a
    reference center.

    Args:
        left, right:      candidate 3D keypoints (camera frame).
        reference_center: e.g., pelvis position; defines the axis.
        axis_index:       0=x, 1=y, 2=z; the coord that should differ
                           in sign for left vs right. Default x (image
                           horizontal in camera frame).

    Returns: True if the pair appears swapped (both on same side of
    center). False if pair is fine or input insufficient.
    """
    if left is None or right is None or reference_center is None:
        return False
    if (len(left) <= axis_index or len(right) <= axis_index
            or len(reference_center) <= axis_index):
        return False

    left_offset = left[axis_index] - reference_center[axis_index]
    right_offset = right[axis_index] - reference_center[axis_index]

    # Both on same side relative to center → likely swapped.
    # Use strict same-sign with non-zero magnitudes to avoid noise at
    # the centerline.
    epsilon = 0.01  # meters
    if abs(left_offset) < epsilon or abs(right_offset) < epsilon:
        return False
    return (left_offset > 0) == (right_offset > 0)


def _euclid(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def limb_chain_says_swapped(
    left_3d: Optional[list[float]],
    right_3d: Optional[list[float]],
    left_chain_3d: Optional[list[float]],
    right_chain_3d: Optional[list[float]],
) -> Optional[bool]:
    """
    Signal (b): does limb-chain continuity say left/right are swapped?

    For shoulders we pass left/right ELBOWS as chain points; for hips
    we pass left/right KNEES. The anatomically correct labeling has
    distance(left_joint, left_chain) + distance(right_joint, right_chain)
    SMALLER than the cross-pair distances. If the cross-pair sum is
    smaller, the labels are swapped.

    Returns:
        True  — limb chain says SWAPPED (apply swap to fix).
        False — limb chain says fine.
        None  — insufficient data, no signal this frame.
    """
    for p in (left_3d, right_3d, left_chain_3d, right_chain_3d):
        if p is None or len(p) != 3 or any(c is None for c in p):
            return None
    same  = _euclid(left_3d, left_chain_3d) + _euclid(right_3d, right_chain_3d)
    swap  = _euclid(left_3d, right_chain_3d) + _euclid(right_3d, left_chain_3d)
    # Require swap to be MEANINGFULLY shorter (avoid noise-triggered flips).
    return swap < same * 0.9


# Default chain mapping for the limb-continuity signal. Plugin can
# override by passing its own mapping; the engine just uses what it
# gets. (Sport-agnostic naming — H36M joints used by every WHAM-based
# plugin to date.)
DEFAULT_LIMB_CHAIN_MAP: dict[tuple[str, str], tuple[str, str]] = {
    ("left_shoulder", "right_shoulder"): ("left_elbow", "right_elbow"),
    ("left_hip",      "right_hip"):       ("left_knee",  "right_knee"),
}


def analyze_pair(
    left_3d: Optional[list[float]],
    right_3d: Optional[list[float]],
    reference_center: Optional[list[float]],
    left_chain_3d: Optional[list[float]],
    right_chain_3d: Optional[list[float]],
) -> tuple[bool, Optional[bool]]:
    """
    Compute the two swap signals for one pair this frame.

    Returns:
      (x_signal_says_swap, limb_signal_says_swap)
      - x_signal: bool (False if insufficient data)
      - limb_signal: True/False/None (None if data insufficient)
    """
    x_sig    = is_swapped_pair(left_3d, right_3d, reference_center)
    limb_sig = limb_chain_says_swapped(
        left_3d, right_3d, left_chain_3d, right_chain_3d,
    )
    return x_sig, limb_sig


@dataclass
class PairHysteresisState:
    """Per-pair cross-frame state for the 3-frame swap hysteresis."""
    applied_swap: bool = False            # currently in swap-applied state?
    pending_count: int = 0                # consecutive frames signaling toggle
    pending_target: bool = False          # what the pending toggle would land on
    transitions: int = 0                  # count of applied-swap state changes


def hysteretic_decision(
    state: PairHysteresisState,
    x_signal: bool,
    limb_signal: Optional[bool],
    *,
    consecutive_required: int = 3,
) -> bool:
    """
    Combine signals + cross-frame hysteresis. Returns the swap state
    to APPLY this frame (True = swap labels, False = leave as-is).
    Mutates `state` in place.

    Rule:
      - If limb_signal is None (no data), trust x_signal alone but DO
        NOT toggle applied_swap unless we already had 3 consecutive
        matching frames in pending.
      - If both signals agree on a verdict, compare to applied_swap.
        - If verdict == applied_swap → reset pending; nothing to do.
        - If verdict != applied_swap → increment pending toward verdict.
          When pending_count >= consecutive_required, flip applied_swap
          and record a transition.
      - If signals disagree → reset pending (ambiguous frame).
    """
    if limb_signal is None:
        combined_verdict = x_signal
        signals_agree = True
    elif x_signal == limb_signal:
        combined_verdict = x_signal
        signals_agree = True
    else:
        signals_agree = False
        combined_verdict = None

    if not signals_agree:
        # Ambiguous frame: reset pending counter (don't accumulate).
        state.pending_count = 0
        return state.applied_swap

    if combined_verdict == state.applied_swap:
        # Already in the right state.
        state.pending_count = 0
        return state.applied_swap

    # Verdict differs from applied state. Build up pending consecutive count.
    if combined_verdict == state.pending_target:
        state.pending_count += 1
    else:
        state.pending_target = combined_verdict
        state.pending_count = 1

    if state.pending_count >= consecutive_required:
        state.applied_swap = combined_verdict
        state.pending_count = 0
        state.transitions += 1

    return state.applied_swap


def correct_lr_swap(
    keypoints_3d: dict[str, Optional[list[float]]],
    *,
    pair_names: list[tuple[str, str]],
    reference_keypoint: str = "pelvis",
    requires_lock: bool = True,
    chain_map: Optional[dict[tuple[str, str], tuple[str, str]]] = None,
    hysteresis_state: Optional[dict[tuple[str, str], PairHysteresisState]] = None,
) -> tuple[dict[str, Optional[list[float]]], bool]:
    """
    Per-pair L/R swap correction with two-signal + cross-frame hysteresis
    (PR-7a.1 Fix 2). Mutates `hysteresis_state` in place; caller is
    expected to thread the same dict across frames.

    Args:
        keypoints_3d:       the frame's joint dict.
        pair_names:         L/R pair names. Plugin-chosen.
        reference_keypoint: name of joint used as x-coord reference center.
        requires_lock:      if False, permissive phase — no-op.
        chain_map:          {(left_name, right_name): (left_chain, right_chain)}
                              for the limb-continuity signal. Defaults to
                              shoulder→elbow, hip→knee.
        hysteresis_state:   per-pair PairHysteresisState dict, mutated.
                              If None, falls back to per-frame application
                              with no hysteresis (legacy path).

    Returns: (possibly-swapped frame, swap_happened_bool).
    """
    if not requires_lock:
        return dict(keypoints_3d), False

    if chain_map is None:
        chain_map = DEFAULT_LIMB_CHAIN_MAP

    out = dict(keypoints_3d)
    center = keypoints_3d.get(reference_keypoint)

    any_swap = False
    for left_name, right_name in pair_names:
        left_3d  = out.get(left_name)
        right_3d = out.get(right_name)
        chain    = chain_map.get((left_name, right_name))
        left_chain_3d  = out.get(chain[0]) if chain else None
        right_chain_3d = out.get(chain[1]) if chain else None
        x_sig, limb_sig = analyze_pair(
            left_3d, right_3d, center, left_chain_3d, right_chain_3d,
        )
        if hysteresis_state is not None:
            state = hysteresis_state.setdefault(
                (left_name, right_name), PairHysteresisState(),
            )
            apply = hysteretic_decision(state, x_sig, limb_sig)
        else:
            # Legacy path: single-frame x-only (used by old callers /
            # unit tests).
            apply = x_sig
        if apply:
            out[left_name], out[right_name] = right_3d, left_3d
            any_swap = True
    return out, any_swap
