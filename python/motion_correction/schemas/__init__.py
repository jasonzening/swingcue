"""
motion_correction.schemas — dataclass shapes for the corrected timeline,
coaching anchors, analysis metrics, and ground-truth label records.

Choice of dataclasses over pydantic:
  - Pure-stdlib (no extra dep)
  - Matches existing python/pilot/runners/_base.py PilotRunResult pattern
  - No runtime validation overhead (offline pipeline; inputs come from
    our own producers, not user input)
  - asdict() + json.dumps() handle serialization

Per Constraint 4 (PR-7a no production cutover) the schemas don't need
to be DB-write-compatible yet. PR-7c may bolt pydantic on top if the
production analyze pipeline benefits from validation at the JSONB
write boundary.
"""

from .corrected_timeline import (
    CorrectedFrame,
    CorrectedTimeline,
    CorrectionDiagnostics,
    SetupBaseline,
)
from .ground_truth import GroundTruthLabel

__all__ = [
    "CorrectedFrame",
    "CorrectedTimeline",
    "CorrectionDiagnostics",
    "SetupBaseline",
    "GroundTruthLabel",
]
