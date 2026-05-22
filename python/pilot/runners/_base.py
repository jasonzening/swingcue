"""
_base.py — shared runner contract + I/O schema for the Phase 2 pilot.

Mirrors the design of python/benchmark/runner.py (PR-6.0 Phase 1B):
  - one dataclass per output element (FrameJoints + PilotRunResult)
  - to_json() for direct JSON write
  - save() helper for python/pilot/output/<runner>/<video_id>/

Schema deliberately distinct from production pose_timeline_2d (per
PHASE_2_BONE_CENTER_PILOT_SPEC_v2 §4): bone-center 3D joint centers are
NOT the same logical thing as 2D surface keypoints, and merging them
into one envelope risks silent confusion. Pilot output is its own
schema; PR-7 will design the production version after the pilot
identifies a winner.

The 20-joint subset covers golf-coaching anatomy. SMPL has 24; we
drop hand/toe joints (rarely useful for swing analysis). Phase2c can
re-include if a runner shows them useful.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Joint name canonicalisation
# ---------------------------------------------------------------------------

# 20 joints (axial spine 6 + arms 6 + legs 8). SMPL/SMPL-X joint name
# convention. Used as the canonical key set in joint_centers_3d /
# joint_centers_2d_projected dicts.
PILOT_JOINT_NAMES: tuple[str, ...] = (
    # axial spine
    "pelvis",
    "spine1", "spine2", "spine3",
    "neck", "head",
    # arms
    "left_shoulder",  "left_elbow",  "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
    # legs (foot center = mid-tarsal, distinct from ankle = talus)
    "left_hip",  "left_knee",  "left_ankle",  "left_foot",
    "right_hip", "right_knee", "right_ankle", "right_foot",
)


# ---------------------------------------------------------------------------
# Per-frame container
# ---------------------------------------------------------------------------

@dataclass
class FrameJoints:
    """One sampled frame of pilot output."""
    ts: float
    frame_idx: int
    # 3D world-frame coords (meters). Dict keyed by name above.
    # [x, y, z] floats; None when the runner failed to estimate
    # a particular joint (rare for SMPL-family but possible for
    # iterative methods that don't converge).
    joint_centers_3d: dict[str, Optional[list[float]]]
    # 2D back-projection onto the original video frame (pixel coords).
    # [x_px, y_px, depth_m] — depth_m carries the camera-z so the
    # frontend can do back-to-front rendering. Optional because some
    # runners don't have camera extrinsics (yet).
    joint_centers_2d_projected: Optional[dict[str, Optional[list[float]]]] = None
    # SMPL betas (10 shape params) + pose (72 axis-angle floats for
    # SMPL or 165 for SMPL-X). Stored verbatim for downstream re-fit
    # or visualisation. None for non-SMPL libraries.
    smpl_betas: Optional[list[float]] = None
    smpl_pose: Optional[list[float]] = None


# ---------------------------------------------------------------------------
# Per-video result
# ---------------------------------------------------------------------------

@dataclass
class PilotRunResult:
    """Full bone-center timeline for one (video, runner) pair."""
    video_id: str
    runner: str
    video_width: int
    video_height: int
    fps_native: float
    fps_sampled: float
    duration_sec: float
    frames: list[FrameJoints] = field(default_factory=list)
    # Camera extrinsics — pinhole intrinsics typically estimated by the
    # SLAM stage (WHAM) or assumed for non-SLAM libraries. Shape:
    #   {"rotation": 3x3 row-major, "translation": [x,y,z], "focal_px": float}
    camera: Optional[dict] = None
    # Library + version + notable defaults — same role as the benchmark
    # RunResult.notes field. Free-form strings the visual evaluator can
    # scan to spot config drift.
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "video_id":     self.video_id,
            "runner":       self.runner,
            "video_width":  self.video_width,
            "video_height": self.video_height,
            "fps_native":   self.fps_native,
            "fps_sampled":  self.fps_sampled,
            "duration_sec": self.duration_sec,
            "frames":       [asdict(f) for f in self.frames],
            "camera":       self.camera,
            "notes":        self.notes,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2))
