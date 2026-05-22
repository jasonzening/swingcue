"""
phases.py — golf swing phase taxonomy.

Per spec v3 §5 GolfPlugin.phase_taxonomy. Listed in chronological
order. `is_static=True` phases (setup, top) feed into setup_baseline
window detection; the engine averages anatomy over the longest
contiguous run of frames in a static phase.

`requires_lr_lock` gates left/right identity correction. Setup +
backswing have known L/R orientations (golfer hasn't crossed body
yet); transition / downswing / impact / finish are noisier and the
guard is relaxed.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PhaseSpec:
    """A golf swing phase definition."""
    name: str
    is_static: bool
    typical_duration_s: float
    requires_lr_lock: bool


PHASES: tuple[PhaseSpec, ...] = (
    PhaseSpec("setup",      is_static=True,  typical_duration_s=2.00, requires_lr_lock=True),
    PhaseSpec("backswing",  is_static=False, typical_duration_s=0.80, requires_lr_lock=True),
    PhaseSpec("top",        is_static=True,  typical_duration_s=0.15, requires_lr_lock=False),
    PhaseSpec("transition", is_static=False, typical_duration_s=0.20, requires_lr_lock=False),
    PhaseSpec("downswing",  is_static=False, typical_duration_s=0.25, requires_lr_lock=False),
    PhaseSpec("impact",     is_static=False, typical_duration_s=0.05, requires_lr_lock=False),
    PhaseSpec("finish",     is_static=False, typical_duration_s=1.00, requires_lr_lock=False),
)

PHASE_NAMES: tuple[str, ...] = tuple(p.name for p in PHASES)

# Convenience filters.
STATIC_PHASES: tuple[str, ...] = tuple(p.name for p in PHASES if p.is_static)
LR_LOCK_PHASES: tuple[str, ...] = tuple(p.name for p in PHASES if p.requires_lr_lock)

# L/R pairs the plugin asks the engine to guard. Limited to torso
# pairs that matter for coaching analysis — wrists/elbows move too
# fast and the guard adds more noise than signal during fast phases.
LR_PAIR_NAMES: list[tuple[str, str]] = [
    ("left_shoulder", "right_shoulder"),
    ("left_hip",      "right_hip"),
]
