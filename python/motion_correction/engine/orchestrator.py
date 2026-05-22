"""
orchestrator.py — assemble engine + plugin into one raw → corrected
pipeline call.

Pipeline order (per spec v3 §2 Layer 2 + Layer 3 composition):

    raw WHAM joint_centers_3d.json
        ↓
    plugin.detect_phases  →  per-frame phase labels
        ↓
    engine.setup_baseline →  SetupBaseline
        ↓  (loop per frame)
    engine.lr_stability   →  swap-corrected raw kp
        ↓
    engine.anatomical_offset (with view-aware coefficients)
        ↓
    engine.temporal_smoother (phase-aware — Constraint 6)
        ↓
    engine.projection 3D → 2D pixel coords
        ↓
    plugin.compute_coaching_anchors → coaching_anchors_2d
        ↓  (end of loop)
    plugin.compute_analysis_metrics → analysis_metrics dict
        ↓
    CorrectedTimeline ready to write JSON

The orchestrator is THE engine's only entry point. Callers in
production / CLI / tests should use this, not the individual modules.

Plugin contract (duck-typed; no ABC per Constraint 1):

    plugin.sport_name              : str — sport identifier
    plugin.plugin_version          : str — versioned plugin identifier
    plugin.static_phases           : tuple[str, ...] — for setup window
    plugin.lr_lock_phases          : tuple[str, ...] — phases needing L/R guard
    plugin.lr_pair_names           : list[(str, str)] — pairs to check
    plugin.compute_spine_angle     : bool
    plugin.offset_configs          : dict[view → dict[offset_key → coef]]
    plugin.smoothing_config        : dict[phase → dict[alpha, outlier_ratio]]
    plugin.coaching_anchor_namespace : list[str]  — names plugin will emit
    plugin.detect_phases(raw_timeline) → dict[int → phase_name]
    plugin.compute_coaching_anchors(kp_2d, kp_3d_corrected) → dict[name → [u, v]]
    plugin.compute_analysis_metrics(corrected_timeline) → dict[name → float]
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schemas.corrected_timeline import (
    CorrectedFrame,
    CorrectedTimeline,
    CorrectionDiagnostics,
)
from . import (
    anatomical_offset,
    lr_stability,
    projection,
    setup_baseline,
    temporal_smoother,
    view_aware,
)


def load_raw_timeline(raw_input_path: Path) -> dict:
    """Read the PilotRunResult-shape JSON emitted by wham_runner."""
    return json.loads(raw_input_path.read_text())


def correct_timeline(
    raw_input_path: Path,
    plugin: Any,
    *,
    view: str,
    bidirectional: bool = True,
) -> CorrectedTimeline:
    """
    Run the full correction pipeline on one (video, plugin, view).

    Args:
        raw_input_path: path to wham_runner's joint_centers_3d.json.
        plugin:         a concrete plugin instance (see docstring contract).
        view:           "face_on" | "down_the_line" (or future view).
        bidirectional:  if True (default — offline PR-7a output), use
                          forward+backward bidirectional EMA on the full
                          timeline (zero phase delay). If False, use the
                          legacy per-frame forward-only smoother
                          (causal, suited for PR-7c real-time streaming).

    Returns: fully-populated CorrectedTimeline ready to .save().
    """
    raw = load_raw_timeline(raw_input_path)
    raw_frames: list[dict] = raw.get("frames", [])

    # ── Per-view offset config selection (Constraint 2) ─────────────
    offset_config = view_aware.select_offset_config(
        plugin.offset_configs, view,
    )

    # ── Phase detection (plugin) ────────────────────────────────────
    # plugin.detect_phases returns {frame_idx → phase_name}.
    phase_by_idx: dict[int, str] = plugin.detect_phases(raw)
    # Default phase for any frame not classified (defensive).
    default_phase = plugin.static_phases[0] if plugin.static_phases else "setup"

    def phase_of_frame(frame_idx: int) -> str:
        return phase_by_idx.get(frame_idx, default_phase)

    # ── Setup baseline (engine, fed by plugin's static-phase declaration) ──
    baseline = setup_baseline.extract_baseline(
        raw_frames,
        phase_of_frame=phase_of_frame,
        static_phases=plugin.static_phases,
        compute_spine_angle=getattr(plugin, "compute_spine_angle", False),
    )

    # Scale reference for temporal smoother outlier rejection — use
    # baseline shoulder width (a stable distance) when available.
    scale_ref = baseline.base_shoulder_width if baseline else None

    # ── Camera intrinsics for 2D projection ──────────────────────────
    video_w = int(raw.get("video_width", 0))
    video_h = int(raw.get("video_height", 0))
    intrinsics = projection.default_intrinsics(video_w, video_h)

    # ── Phase A: per-frame deterministic ops (L/R + anatomical offset).
    # Result: parallel lists (kp_offset_per_frame, lr_swap_per_frame,
    # phase_per_frame) over `raw_frames`. NO temporal smoothing yet —
    # that's Phase B and may run forward+backward for offline outputs.
    lr_hysteresis: dict[tuple[str, str], lr_stability.PairHysteresisState] = {}
    kp_offset_per_frame: list[dict[str, Any]] = []
    raw_kp_per_frame:    list[dict[str, Any]] = []
    lr_swap_per_frame:   list[bool] = []
    phase_per_frame:     list[str] = []
    lr_swap_total = 0

    for f in raw_frames:
        frame_idx = int(f["frame_idx"])
        phase = phase_of_frame(frame_idx)
        raw_kp_3d: dict[str, Any] = f.get("joint_centers_3d", {})

        requires_lock = phase in getattr(plugin, "lr_lock_phases", ())
        kp_lr, lr_swapped = lr_stability.correct_lr_swap(
            raw_kp_3d,
            pair_names=list(plugin.lr_pair_names),
            reference_keypoint="pelvis",
            requires_lock=requires_lock,
            hysteresis_state=lr_hysteresis,
        )
        if lr_swapped:
            lr_swap_total += 1

        # Setup-phase basis lock (Fix 1).
        basis_override = None
        if (phase in getattr(plugin, "static_phases", ())
                and baseline is not None
                and baseline.locked_basis is not None):
            lb = baseline.locked_basis
            basis_override = (list(lb[0]), list(lb[1]), list(lb[2]))
        kp_offset = anatomical_offset.apply_offset_to_frame(
            kp_lr, offset_config,
            basis_override=basis_override,
            phase=phase,
        )

        kp_offset_per_frame.append(kp_offset)
        raw_kp_per_frame.append(raw_kp_3d)
        lr_swap_per_frame.append(lr_swapped)
        phase_per_frame.append(phase)

    # ── Phase B: temporal smoothing (PR-7a.3 bidirectional by default).
    if bidirectional:
        (
            smoothed_per_frame,
            outlier_marks_per_frame,
            alpha_used_per_frame,
        ) = temporal_smoother.smooth_timeline_bidirectional(
            kp_offset_per_frame,
            phase_per_frame,
            smoothing_config=plugin.smoothing_config,
            scale_reference=scale_ref,
        )
        # Build a parallel outlier_ratio_used list for diagnostics. The
        # bidirectional smoother doesn't return it; recompute from config.
        outlier_ratio_used_per_frame: list[float] = [
            plugin.smoothing_config[ph].get("outlier_ratio", 0.0)
            for ph in phase_per_frame
        ]
    else:
        # Legacy forward-only (per-frame causal). Used by PR-7c streaming.
        smoothed_per_frame = []
        outlier_marks_per_frame = []
        alpha_used_per_frame = []
        outlier_ratio_used_per_frame = []
        prev_smoothed: dict[str, Any] = {}
        prev_raw_offset: dict[str, Any] = {}
        for i, kp_offset in enumerate(kp_offset_per_frame):
            ph = phase_per_frame[i]
            kp_smoothed, smooth_diag = temporal_smoother.smooth_frame_phase_aware(
                kp_offset, prev_smoothed,
                phase=ph,
                smoothing_config=plugin.smoothing_config,
                scale_reference=scale_ref,
                prev_raw_frame_for_outlier_check=prev_raw_offset,
            )
            prev_smoothed = kp_smoothed
            prev_raw_offset = kp_offset
            smoothed_per_frame.append(kp_smoothed)
            outlier_marks_per_frame.append(
                {name: d.was_outlier for name, d in smooth_diag.items()}
            )
            any_diag = next(iter(smooth_diag.values()), None)
            alpha_used_per_frame.append(
                {name: (any_diag.alpha_used if any_diag else 0.0)
                 for name in kp_smoothed}
            )
            outlier_ratio_used_per_frame.append(
                any_diag.outlier_ratio_used if any_diag else 0.0
            )

    # ── Phase C: per-frame projection + anchors + diagnostics assembly.
    corrected_frames: list[CorrectedFrame] = []
    outlier_total = 0
    drift_after_px: list[float] = []

    for i, f in enumerate(raw_frames):
        frame_idx = int(f["frame_idx"])
        ts = float(f["ts"])
        phase = phase_per_frame[i]
        raw_kp_3d = raw_kp_per_frame[i]
        kp_smoothed = smoothed_per_frame[i]
        outlier_marks = outlier_marks_per_frame[i]
        outliers_this_frame = [n for n, was in outlier_marks.items() if was]
        outlier_total += len(outliers_this_frame)

        raw_2d = projection.project_keypoint_dict(raw_kp_3d, intrinsics)
        corrected_2d = projection.project_keypoint_dict(kp_smoothed, intrinsics)
        for name, raw_uv in raw_2d.items():
            cor_uv = corrected_2d.get(name)
            if raw_uv is None or cor_uv is None:
                continue
            d = ((raw_uv[0] - cor_uv[0]) ** 2 + (raw_uv[1] - cor_uv[1]) ** 2) ** 0.5
            drift_after_px.append(d)

        coaching_anchors = plugin.compute_coaching_anchors(
            corrected_2d, kp_smoothed,
        )

        avg_conf = 0.0
        # alpha_used for this frame: bidirectional returns a per-joint dict;
        # take the first available (all joints share phase α anyway).
        alpha_per_joint = alpha_used_per_frame[i] if alpha_used_per_frame else {}
        alpha_used = (next(iter(alpha_per_joint.values()), 0.0)
                      if alpha_per_joint else 0.0)
        outlier_ratio_used = outlier_ratio_used_per_frame[i]

        corrected_frames.append(CorrectedFrame(
            ts=ts,
            frame_idx=frame_idx,
            phase=phase,
            keypoints_3d_corrected=kp_smoothed,
            keypoints_2d_projected=corrected_2d,
            coaching_anchors_2d=coaching_anchors,
            diagnostics=CorrectionDiagnostics(
                outlier_rejected_joints=outliers_this_frame,
                lr_swapped=lr_swap_per_frame[i],
                confidence_avg=avg_conf,
                smoothing_alpha_used=alpha_used,
                smoothing_outlier_ratio_used=outlier_ratio_used,
            ),
        ))

    # ── Plugin-computed analysis metrics ─────────────────────────────
    timeline = CorrectedTimeline(
        version=1,
        sport=plugin.sport_name,
        domain_plugin_version=plugin.plugin_version,
        pose_backbone=str(raw.get("runner", "")),
        view=view,
        video_id=str(raw.get("video_id", "")),
        video_width=video_w,
        video_height=video_h,
        fps_native=float(raw.get("fps_native", 0.0)),
        fps_sampled=float(raw.get("fps_sampled", 0.0)),
        duration_sec=float(raw.get("duration_sec", 0.0)),
        camera_intrinsics=intrinsics,
        setup_baseline=baseline,
        correction_config_used={
            "offset_values_used": offset_config,
            "smoothing_config":   plugin.smoothing_config,
        },
        frames=corrected_frames,
        summary_stats={
            "frame_count":                 float(len(corrected_frames)),
            "outlier_rejection_count":     float(outlier_total),
            "outlier_rejection_rate":      (
                outlier_total / (len(corrected_frames) * max(1, len(plugin.coaching_anchor_namespace)))
                if corrected_frames else 0.0
            ),
            "lr_swap_corrections":         float(lr_swap_total),
            # PR-7a.1 Fix 2: per-pair transition counts. Each entry is
            # the number of times that pair's applied-swap state flipped
            # across the timeline (target ≤ 4 per 100 frames).
            "lr_swap_thrash_count":        float(
                sum(s.transitions for s in lr_hysteresis.values())
            ),
            "avg_raw_vs_corrected_drift_px": (
                sum(drift_after_px) / len(drift_after_px) if drift_after_px else 0.0
            ),
        },
        notes=[
            (
                f"pipeline: lr_stability → anatomical_offset → temporal_smoother "
                f"({'bidirectional EMA — zero phase delay' if bidirectional else 'forward-only causal'}) "
                f"→ projection"
            ),
            f"view={view}  offset_keys={list(offset_config.keys())}",
            f"plugin={plugin.sport_name} ({plugin.plugin_version})",
        ],
    )
    timeline.analysis_metrics = plugin.compute_analysis_metrics(timeline)
    return timeline
