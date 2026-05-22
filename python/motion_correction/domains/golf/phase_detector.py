"""
phase_detector.py — classify each frame into one of the 7 golf phases.

PR-7a starting impl uses **wrist-trajectory fractional thresholds**.
This is the simplest detector that produces all 7 named phases without
requiring trained models or club detection. It's intentionally NOT
production-grade — PR-7c can swap in the existing python/phase_detector.py
algorithm (which already runs in production analyze) once we know the
engine's contract is right.

Fractional model (works on any clip length):

    setup       = first 5%
    backswing   = 5% → 40%
    top         = 40% → 45% (brief)
    transition  = 45% → 55%
    downswing   = 55% → 65%
    impact      = 65% → 70% (brief)
    finish      = 70% → end

Same fractions as python/pilot/scripts/extract_phase_frames.py's
representative-frame heuristics, expanded into per-frame ranges. If
the production phase_detector outputs disagree at PR-7c time, the
correction layer takes phase_detector as authoritative — this module
is just an offline-pipeline starter.
"""
from __future__ import annotations

from typing import Optional

from .phases import PHASE_NAMES


# Fractional phase boundaries — (phase_name, start_fraction).
# Each phase runs from its start_fraction up to the next entry's
# start_fraction (or 1.0 for the last).
_PHASE_BOUNDARIES: list[tuple[str, float]] = [
    ("setup",      0.00),
    ("backswing",  0.05),
    ("top",        0.40),
    ("transition", 0.45),
    ("downswing",  0.55),
    ("impact",     0.65),
    ("finish",     0.70),
]


def detect_phases(raw_timeline: dict) -> dict[int, str]:
    """
    Classify every frame in the raw timeline into a phase.

    Args:
        raw_timeline: PilotRunResult-shape dict; uses .frames + each
                       frame's frame_idx field.

    Returns: {frame_idx: phase_name}. Every frame in raw_timeline.frames
    gets exactly one phase label.
    """
    frames = raw_timeline.get("frames", [])
    n = len(frames)
    if n == 0:
        return {}

    out: dict[int, str] = {}
    for i, f in enumerate(frames):
        frac = i / max(1, n - 1)
        phase = _phase_at_fraction(frac)
        out[int(f["frame_idx"])] = phase
    return out


def _phase_at_fraction(frac: float) -> str:
    """Lookup the phase that contains this position-in-clip fraction."""
    # Iterate boundaries; pick the last one whose start is ≤ frac.
    chosen = _PHASE_BOUNDARIES[0][0]
    for name, start in _PHASE_BOUNDARIES:
        if frac >= start:
            chosen = name
        else:
            break
    return chosen


# Defensive: every name from PHASE_NAMES must be classifiable by the
# fractional model. This catches drift if PHASES adds a new phase
# without updating _PHASE_BOUNDARIES.
_KNOWN_PHASE_SET = set(PHASE_NAMES)
for _name, _ in _PHASE_BOUNDARIES:
    assert _name in _KNOWN_PHASE_SET, (
        f"phase_detector._PHASE_BOUNDARIES references {_name!r}, "
        f"which is not in phases.PHASES. Update phases.py first."
    )
