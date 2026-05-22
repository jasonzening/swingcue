"""
corrected_timeline.py — dataclass shapes for the engine's output.

Mirrors spec v3 §6 `pose_timeline_3d_corrected` envelope. Sport-agnostic
container; plugin-namespaced anchor + metric dicts go inside.

Per PR-7_REVIEW_RESPONSE.md Constraint 3: keeps
`keypoints_3d_corrected` (analysis) and `coaching_anchors_2d` (visual
overlay) STRUCTURALLY SEPARATED even though PR-7a initial values are
identical projections. The separation gates future PR-7.x divergence
without a schema migration.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


# 3D coord = [x, y, z] meters in camera frame (or world frame after SLAM
# grounding). 2D coord = [u, v] pixel.
XYZ = list  # really list[float | None] with len 3
UV = list   # really list[float | None] with len 2


@dataclass
class SetupBaseline:
    """
    Per spec v3 §6: stable baseline values auto-detected at the setup
    phase. Used by the engine as a denominator for downstream relative
    measurements (e.g., disc scale = current_shoulder_width / baseShoulderWidth).

    Setting these once at setup avoids per-frame jitter polluting the
    derived metric. Sport plugin chooses which fields are meaningful.

    PR-7a.1 Fix 1: additionally carries the LOCKED body-local basis and
    a MEDIAN-pose snapshot derived from the entire setup window. The
    orchestrator uses these to freeze the offset-vector transform during
    setup frames, eliminating per-frame basis jitter.
    """
    setup_frame_idx: int
    setup_ts: float
    base_shoulder_width: float          # meters (3D)
    base_hip_width: float                # meters
    base_spine_length: float             # meters, pelvis → neck
    base_stance_width: float             # meters, ankle-to-ankle
    base_spine_angle_deg: Optional[float] = None  # may be None for non-golf domains
    # PR-7a.1 Fix 1 — setup-lock fields:
    setup_window_start: Optional[int] = None    # inclusive frame_idx
    setup_window_end: Optional[int] = None      # exclusive frame_idx
    # Locked body-local basis (camera-frame), used during setup phase
    # for offset-vector transform. Stored as 3 unit vectors:
    #   [horizontal_xyz, spine_up_xyz, body_forward_xyz]
    locked_basis: Optional[list[list[float]]] = None
    # Median 3D keypoint positions across the setup window (joint name → [x,y,z]).
    # Used as the "frozen" reference pose for the entire setup phase.
    locked_pose_3d: Optional[dict[str, list[float]]] = None


@dataclass
class CorrectionDiagnostics:
    """Per-frame diagnostic info — engine emits, doesn't consume."""
    outlier_rejected_joints: list[str] = field(default_factory=list)
    lr_swapped: bool = False
    confidence_avg: float = 0.0
    smoothing_alpha_used: float = 0.0   # the actual α applied this frame
                                          # (per spec v3 PHASE_CONFIG, must
                                          # be phase-aware — Constraint 6)
    smoothing_outlier_ratio_used: float = 0.0


@dataclass
class CorrectedFrame:
    """One sampled frame after correction pipeline."""
    ts: float
    frame_idx: int
    phase: str                          # per plugin.phase_taxonomy
    # Analysis-grade joint positions (sport-agnostic SMPL joint names).
    keypoints_3d_corrected: dict[str, Optional[XYZ]]
    # 2D pinhole projection of the above (camera-frame XYZ → image
    # pixel coords using the pinhole intrinsics in CorrectedTimeline.camera).
    keypoints_2d_projected: dict[str, Optional[UV]]
    # Visual-overlay anchors, plugin-derived. Constraint 3: initial impl
    # may equal keypoints_2d_projected; later PRs may diverge without
    # schema migration. Keys come from plugin.coaching_anchor_namespace.
    coaching_anchors_2d: dict[str, Optional[UV]]
    diagnostics: CorrectionDiagnostics = field(default_factory=CorrectionDiagnostics)


@dataclass
class CorrectedTimeline:
    """Full corrected timeline for one (video, plugin, view) tuple."""
    version: int = 1
    sport: str = ""                     # "golf" in PR-7a
    domain_plugin_version: str = ""     # e.g., "golf_v1"
    pose_backbone: str = ""             # "wham_vit_w_3dpw_v1" today
    view: str = ""                      # "face_on" | "down_the_line"
    video_id: str = ""
    video_width: int = 0
    video_height: int = 0
    fps_native: float = 0.0
    fps_sampled: float = 0.0
    duration_sec: float = 0.0
    # Pinhole intrinsics used for projection (so anyone re-projecting
    # later gets the same coords). Defaults filled by engine.projection
    # if not provided by upstream.
    camera_intrinsics: dict[str, float] = field(default_factory=dict)
    setup_baseline: Optional[SetupBaseline] = None
    # Configuration values actually applied (per-view offsets, per-phase
    # smoothing α's). Locked at runtime; recorded here for reproducibility.
    correction_config_used: dict = field(default_factory=dict)
    frames: list[CorrectedFrame] = field(default_factory=list)
    # Plugin-namespaced summary metrics (e.g., golf hip-shoulder sep).
    analysis_metrics: dict[str, float] = field(default_factory=dict)
    summary_stats: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        # asdict handles nested dataclasses; no further massaging needed
        # since all leaf types are JSON-serialisable.
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
