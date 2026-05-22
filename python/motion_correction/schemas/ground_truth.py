"""
ground_truth.py — dataclass mirror of the labeler JSON shape from
docs/files/PR-7_MOTION_CORRECTION_PLATFORM_SPEC_v3.md §7.

This is what PR-7b's sweep harness reads back. Definition lives in
schemas/ so engine + plugin code can import without cycling through
the pilot scripts package.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GroundTruthLabel:
    """One ground-truth label record per spec v3 §7."""
    schema_version: str
    sport: str
    video_id: str
    phase: str
    frame_idx: int
    view: str                           # "face_on" | "down_the_line"
    video_width: int
    video_height: int
    # {keypoint_name: {"x": int, "y": int}}
    labels: dict[str, dict[str, int]]
    labeler_version: str
    labeled_at: str                     # ISO-8601 UTC

    @classmethod
    def from_file(cls, path: Path) -> "GroundTruthLabel":
        d = json.loads(path.read_text())
        return cls(
            schema_version=d["schema_version"],
            sport=d["sport"],
            video_id=d["video_id"],
            phase=d["phase"],
            frame_idx=d["frame_idx"],
            view=d["view"],
            video_width=d["video_width"],
            video_height=d["video_height"],
            labels=d["labels"],
            labeler_version=d["labeler_version"],
            labeled_at=d["labeled_at"],
        )

    def keypoint(self, name: str) -> Optional[tuple[int, int]]:
        """Return (x, y) for a labeled keypoint, or None if missing."""
        kp = self.labels.get(name)
        if kp is None:
            return None
        return (int(kp["x"]), int(kp["y"]))
