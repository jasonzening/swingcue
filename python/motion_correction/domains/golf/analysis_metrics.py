"""
analysis_metrics.py — golf-specific summary metrics computed once
per swing from the corrected timeline.

Per spec v3 §5 GolfPlugin.analysis_metric_namespace. PR-7a ships a
small starter set with the math wired up; PR-7c can extend without
schema migration (just add to METRIC_NAMES + add a function below).

These are SCALAR per-swing values — frame-by-frame derivatives stay
in the timeline as keypoints, not here. Front-end coaching uses
these scalars for the verdict text.
"""
from __future__ import annotations

import math
from typing import Any, Optional


# What this plugin claims to emit. Engine validates against this.
METRIC_NAMES: tuple[str, ...] = (
    "hip_shoulder_separation_at_top_deg",
    "hip_turn_at_impact_deg",
    "spine_tilt_change_setup_to_impact_deg",
    "lateral_head_drift_setup_to_impact_m",
)


def _angle_xz_deg(
    p1: Optional[list[float]],
    p2: Optional[list[float]],
) -> Optional[float]:
    """
    Angle of the (p1 → p2) line in the camera-frame X-Z plane (the
    horizontal-axial plane in a face-on view). Returns degrees, with
    0° = pointing along +x and 90° = along +z. Used as the proxy for
    "shoulder line" or "hip line" rotation.
    """
    if p1 is None or p2 is None or len(p1) != 3 or len(p2) != 3:
        return None
    dx = p2[0] - p1[0]
    dz = p2[2] - p1[2]
    if dx == 0 and dz == 0:
        return None
    return math.degrees(math.atan2(dz, dx))


def _angle_diff_deg(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Signed difference (a - b) wrapped to (-180, 180]."""
    if a is None or b is None:
        return None
    d = a - b
    while d > 180:
        d -= 360
    while d <= -180:
        d += 360
    return d


def _frame_at_phase(corrected_timeline: Any, phase_name: str) -> Optional[dict]:
    """Return the FIRST frame whose phase matches, or None."""
    for f in corrected_timeline.frames:
        if f.phase == phase_name:
            return f
    return None


def compute_all(corrected_timeline: Any) -> dict[str, float]:
    """
    Compute every metric in METRIC_NAMES from the corrected timeline.

    Returns: {metric_name: float}. Missing metrics (insufficient frame
    coverage, missing keypoints) are omitted from the output dict
    rather than set to None — keeps the schema clean.
    """
    out: dict[str, float] = {}

    setup = _frame_at_phase(corrected_timeline, "setup")
    top = _frame_at_phase(corrected_timeline, "top")
    impact = _frame_at_phase(corrected_timeline, "impact")

    # 1. Hip-shoulder separation at top (degrees).
    if top is not None:
        kp = top.keypoints_3d_corrected
        sh_angle = _angle_xz_deg(kp.get("left_shoulder"), kp.get("right_shoulder"))
        hip_angle = _angle_xz_deg(kp.get("left_hip"), kp.get("right_hip"))
        diff = _angle_diff_deg(sh_angle, hip_angle)
        if diff is not None:
            out["hip_shoulder_separation_at_top_deg"] = round(abs(diff), 2)

    # 2. Hip turn at impact (degrees vs setup).
    if setup is not None and impact is not None:
        setup_hip_angle = _angle_xz_deg(
            setup.keypoints_3d_corrected.get("left_hip"),
            setup.keypoints_3d_corrected.get("right_hip"),
        )
        impact_hip_angle = _angle_xz_deg(
            impact.keypoints_3d_corrected.get("left_hip"),
            impact.keypoints_3d_corrected.get("right_hip"),
        )
        diff = _angle_diff_deg(impact_hip_angle, setup_hip_angle)
        if diff is not None:
            out["hip_turn_at_impact_deg"] = round(diff, 2)

    # 3. Spine tilt change setup → impact (degrees).
    if (corrected_timeline.setup_baseline is not None
            and corrected_timeline.setup_baseline.base_spine_angle_deg is not None
            and impact is not None):
        kp = impact.keypoints_3d_corrected
        pelvis = kp.get("pelvis")
        neck = kp.get("neck") or kp.get("head")
        if pelvis is not None and neck is not None:
            dx = neck[0] - pelvis[0]
            dy = neck[1] - pelvis[1]
            dz = neck[2] - pelvis[2]
            horizontal = math.sqrt(dx * dx + dz * dz)
            vertical = abs(dy)
            if vertical > 1e-6:
                impact_spine_angle_deg = math.degrees(math.atan2(horizontal, vertical))
                delta = impact_spine_angle_deg - corrected_timeline.setup_baseline.base_spine_angle_deg
                out["spine_tilt_change_setup_to_impact_deg"] = round(delta, 2)

    # 4. Lateral head drift setup → impact (meters).
    if setup is not None and impact is not None:
        setup_head = setup.keypoints_3d_corrected.get("head")
        impact_head = impact.keypoints_3d_corrected.get("head")
        if setup_head is not None and impact_head is not None:
            # Lateral = x-axis displacement in camera frame.
            drift = abs(impact_head[0] - setup_head[0])
            out["lateral_head_drift_setup_to_impact_m"] = round(drift, 4)

    return out
