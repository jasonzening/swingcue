"""
src/judgment/__init__.py — convenience re-exports.
"""

from .rules import (
    FaultDetection,
    bone_length_sentinel,
    r1_loss_of_posture,
    r2_hip_toward_ball,
)
from .root_cause import RootCauseEngine, RootCauseResult
from .output import CoachingOutput, CoachingOutputResult

__all__ = [
    "FaultDetection",
    "bone_length_sentinel",
    "r1_loss_of_posture",
    "r2_hip_toward_ball",
    "RootCauseEngine",
    "RootCauseResult",
    "CoachingOutput",
    "CoachingOutputResult",
]
