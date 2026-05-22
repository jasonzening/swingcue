"""
plugin.py — GolfCorrectionPlugin concrete class.

Per PR-7_REVIEW_RESPONSE.md Constraint 1: NO ABC infrastructure in
PR-7a. This is a standalone concrete class. The orchestrator
duck-types against the plugin contract documented in
engine/orchestrator.py.

If/when a second domain plugin (tennis, ski) lands, refactor up to
an ABC at that time — not before. Per spec v3 §0: "the architecture
is in service of golf shipping, not the reverse."

Caller usage:

    from motion_correction.domains.golf.plugin import GolfCorrectionPlugin
    from motion_correction.engine.orchestrator import correct_timeline

    plugin = GolfCorrectionPlugin()
    corrected = correct_timeline(
        raw_input_path=Path("python/pilot/output/wham/<id>/joint_centers_3d.json"),
        plugin=plugin,
        view="face_on",
    )
    corrected.save(Path("docs/PR-7a_OFFLINE_OUTPUT/<id>_face_on_corrected.json"))
"""
from __future__ import annotations

from typing import Any, Optional

from . import (
    analysis_metrics as _analysis_metrics,
    coaching_anchors as _coaching_anchors,
    config as _config,
    phase_detector as _phase_detector,
    phases as _phases,
)


class GolfCorrectionPlugin:
    """
    Golf-domain plugin (first reference implementation).

    Public attributes consumed by engine.orchestrator (duck-typed
    contract, no ABC base class):

      sport_name              str
      plugin_version          str
      static_phases           tuple[str, ...]
      lr_lock_phases          tuple[str, ...]
      lr_pair_names           list[(str, str)]
      compute_spine_angle     bool
      offset_configs          dict[view → dict[offset_key → coef]]
      smoothing_config        dict[phase → dict[alpha, outlier_ratio]]
      coaching_anchor_namespace tuple[str, ...]
      analysis_metric_namespace tuple[str, ...]

    Methods:
      detect_phases(raw_timeline)                → {frame_idx → phase}
      compute_coaching_anchors(kp_2d, kp_3d)     → {anchor → [u, v]}
      compute_analysis_metrics(corrected_timeline)→ {metric → float}
    """

    # ── Identity ───────────────────────────────────────────────────
    sport_name: str = "golf"
    plugin_version: str = "golf_v1"

    # ── Phase taxonomy declarations ────────────────────────────────
    static_phases: tuple[str, ...] = _phases.STATIC_PHASES
    lr_lock_phases: tuple[str, ...] = _phases.LR_LOCK_PHASES
    lr_pair_names: list[tuple[str, str]] = _phases.LR_PAIR_NAMES
    compute_spine_angle: bool = True   # golf cares about spine angle baseline

    # ── Engine-facing configs ──────────────────────────────────────
    offset_configs: dict[str, dict[str, float]] = _config.ANATOMICAL_OFFSETS
    smoothing_config: dict[str, dict[str, float]] = _config.PHASE_CONFIG

    # ── Plugin namespace declarations ──────────────────────────────
    coaching_anchor_namespace: tuple[str, ...] = _coaching_anchors.COACHING_ANCHOR_NAMES
    analysis_metric_namespace: tuple[str, ...] = _analysis_metrics.METRIC_NAMES

    # ── Methods ────────────────────────────────────────────────────

    def detect_phases(self, raw_timeline: dict) -> dict[int, str]:
        """Delegate to module-level function (testable independently)."""
        return _phase_detector.detect_phases(raw_timeline)

    def compute_coaching_anchors(
        self,
        keypoints_2d_projected: dict[str, Optional[list[float]]],
        keypoints_3d_corrected: dict[str, Optional[list[float]]],
    ) -> dict[str, Optional[list[float]]]:
        return _coaching_anchors.derive(
            keypoints_2d_projected, keypoints_3d_corrected,
        )

    def compute_analysis_metrics(self, corrected_timeline: Any) -> dict[str, float]:
        return _analysis_metrics.compute_all(corrected_timeline)

    # ── Convenience repr ───────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"GolfCorrectionPlugin(sport={self.sport_name!r}, "
            f"version={self.plugin_version!r}, "
            f"views={list(self.offset_configs.keys())}, "
            f"phases={list(self.smoothing_config.keys())})"
        )
