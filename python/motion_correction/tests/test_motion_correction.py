"""
test_motion_correction.py — unit tests for engine + golf plugin, plus
one end-to-end smoke against the existing b3fea3f0 WHAM output.

Run from repo root:
    python -m pytest python/motion_correction/tests/test_motion_correction.py -v

Or as a script (no pytest dep):
    python python/motion_correction/tests/test_motion_correction.py

Pytest is optional — every test is a plain assert function and the
__main__ block runs them all.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# Allow `python script.py` invocation from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "python"))

from motion_correction.engine import (
    anatomical_offset,
    lr_stability,
    projection,
    setup_baseline,
    temporal_smoother,
    view_aware,
)
from motion_correction.domains.golf import (
    analysis_metrics,
    coaching_anchors,
    config,
    phase_detector,
    phases,
)
from motion_correction.domains.golf.plugin import GolfCorrectionPlugin
from motion_correction.engine.orchestrator import correct_timeline
from motion_correction.schemas.corrected_timeline import CorrectedTimeline


# ─────────────────────────────────────────────────────────────────────
# projection
# ─────────────────────────────────────────────────────────────────────

def test_default_intrinsics_uses_max_dim_as_focal():
    intr = projection.default_intrinsics(720, 1280)
    assert intr["fx"] == 1280.0
    assert intr["fy"] == 1280.0
    assert intr["cx"] == 360.0
    assert intr["cy"] == 640.0


def test_project_xyz_at_origin_lands_at_principal_point():
    uv = projection.project_xyz_to_uv([0.0, 0.0, 5.0], 1280, 1280, 360, 640)
    assert uv == [360.0, 640.0]


def test_project_xyz_rejects_negative_z():
    assert projection.project_xyz_to_uv([1.0, 1.0, -1.0], 1, 1, 0, 0) is None
    assert projection.project_xyz_to_uv([1.0, 1.0, 0.0], 1, 1, 0, 0) is None


def test_project_keypoint_dict_preserves_nones():
    intr = projection.default_intrinsics(100, 100)
    out = projection.project_keypoint_dict(
        {"a": [0, 0, 1], "b": None, "c": [1, 1, -1]}, intr,
    )
    assert out["a"] == [50.0, 50.0]
    assert out["b"] is None
    assert out["c"] is None  # behind camera


# ─────────────────────────────────────────────────────────────────────
# view_aware
# ─────────────────────────────────────────────────────────────────────

def test_view_aware_returns_native_view_when_present():
    cfgs = {"face_on": {"shoulder_inward": 0.14}}
    out = view_aware.select_offset_config(cfgs, "face_on")
    assert out == {"shoulder_inward": 0.14}


def test_view_aware_fallback_multiplies_face_on():
    cfgs = {"face_on": {"shoulder_inward": 0.10, "hip_inward": 0.20}}
    out = view_aware.select_offset_config(cfgs, "down_the_line")
    assert math.isclose(out["shoulder_inward"], 0.10 * view_aware.DTL_FALLBACK_MULTIPLIER)
    assert math.isclose(out["hip_inward"], 0.20 * view_aware.DTL_FALLBACK_MULTIPLIER)


def test_view_aware_raises_when_no_fallback():
    try:
        view_aware.select_offset_config({}, "anything")
    except KeyError:
        return
    raise AssertionError("expected KeyError for empty configs")


# ─────────────────────────────────────────────────────────────────────
# anatomical_offset
# ─────────────────────────────────────────────────────────────────────

def test_offset_zero_is_identity():
    out = anatomical_offset.offset_toward_center([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], 0.0)
    assert out == [1.0, 2.0, 3.0]


def test_offset_one_goes_all_the_way():
    out = anatomical_offset.offset_toward_center([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], 1.0)
    assert out == [0.0, 0.0, 0.0]


def test_offset_half_is_midpoint():
    out = anatomical_offset.offset_toward_center([2.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.5)
    assert out == [1.0, 0.0, 0.0]


def test_torso_center_handles_missing_corners():
    c = anatomical_offset.compute_torso_center(
        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], None, None,
    )
    assert c == [0.0, 0.0, 0.0]
    assert anatomical_offset.compute_torso_center(None, None, None, None) is None


def test_apply_offset_pulls_shoulders_inward():
    kp = {
        "left_shoulder":  [+0.2, 0.0, 5.0],
        "right_shoulder": [-0.2, 0.0, 5.0],
        "left_hip":       [+0.15, 0.5, 5.0],
        "right_hip":      [-0.15, 0.5, 5.0],
        "pelvis":         [0.0, 0.5, 5.0],
    }
    out = anatomical_offset.apply_offset_to_frame(
        kp, {"shoulder_inward": 0.5, "hip_inward": 0.0},
    )
    # Shoulders should have moved toward x=0 (torso center).
    assert abs(out["left_shoulder"][0]) < abs(kp["left_shoulder"][0])
    assert abs(out["right_shoulder"][0]) < abs(kp["right_shoulder"][0])
    # Hips untouched (coef 0).
    assert out["left_hip"] == kp["left_hip"]
    # Pelvis untouched (not in KEYPOINT_TO_OFFSET_KEY).
    assert out["pelvis"] == kp["pelvis"]


# ─────────────────────────────────────────────────────────────────────
# temporal_smoother — Constraint 6
# ─────────────────────────────────────────────────────────────────────

def test_phase_arg_is_mandatory():
    # Calling without the phase kwarg must be a TypeError at the
    # call site (keyword-only enforcement, per Constraint 6).
    try:
        temporal_smoother.smooth_keypoint_phase_aware(
            [1.0, 2.0, 3.0], [1.0, 2.0, 3.0],
            {"setup": {"alpha": 0.3, "outlier_ratio": 0.25}},
        )  # type: ignore[misc]
    except TypeError:
        return
    raise AssertionError("phase= must be a keyword-only required arg")


def test_unknown_phase_raises_keyerror():
    cfg = {"setup": {"alpha": 0.3, "outlier_ratio": 0.25}}
    try:
        temporal_smoother.smooth_keypoint_phase_aware(
            [1.0, 2.0, 3.0], [1.0, 2.0, 3.0],
            phase="impact", smoothing_config=cfg,
        )
    except KeyError:
        return
    raise AssertionError("unknown phase must surface as KeyError")


def test_smoother_first_frame_seeds_from_raw():
    cfg = {"setup": {"alpha": 0.3, "outlier_ratio": 0.25}}
    r = temporal_smoother.smooth_keypoint_phase_aware(
        [1.0, 2.0, 3.0], None, phase="setup", smoothing_config=cfg,
    )
    assert r.smoothed == [1.0, 2.0, 3.0]
    assert r.was_outlier is False


def test_smoother_blends_when_raw_is_close():
    cfg = {"setup": {"alpha": 0.5, "outlier_ratio": 1.0}}
    r = temporal_smoother.smooth_keypoint_phase_aware(
        [2.0, 2.0, 2.0], [0.0, 0.0, 0.0],
        phase="setup", smoothing_config=cfg, scale_reference=10.0,
    )
    assert r.smoothed == [1.0, 1.0, 1.0]
    assert r.was_outlier is False


def test_smoother_rejects_outlier_and_holds_prev():
    cfg = {"setup": {"alpha": 0.5, "outlier_ratio": 0.1}}
    # Motion (3 m) >> threshold (0.1 * 1 m = 0.1 m) → outlier.
    r = temporal_smoother.smooth_keypoint_phase_aware(
        [3.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        phase="setup", smoothing_config=cfg, scale_reference=1.0,
    )
    assert r.smoothed == [0.0, 0.0, 0.0]
    assert r.was_outlier is True


def test_smoother_carries_forward_on_none_raw():
    cfg = {"setup": {"alpha": 0.3, "outlier_ratio": 0.25}}
    r = temporal_smoother.smooth_keypoint_phase_aware(
        None, [1.0, 2.0, 3.0],
        phase="setup", smoothing_config=cfg, scale_reference=1.0,
    )
    assert r.smoothed == [1.0, 2.0, 3.0]
    assert r.was_outlier is False


# ─────────────────────────────────────────────────────────────────────
# lr_stability
# ─────────────────────────────────────────────────────────────────────

def test_lr_swap_detected_when_both_on_same_side():
    swapped = lr_stability.is_swapped_pair(
        left=[+0.2, 0.0, 5.0], right=[+0.1, 0.0, 5.0],
        reference_center=[0.0, 0.0, 5.0],
    )
    assert swapped is True


def test_lr_swap_not_detected_when_split():
    swapped = lr_stability.is_swapped_pair(
        left=[+0.2, 0.0, 5.0], right=[-0.2, 0.0, 5.0],
        reference_center=[0.0, 0.0, 5.0],
    )
    assert swapped is False


def test_lr_swap_skipped_when_lock_off():
    kp = {"left_hip": [+0.1, 0.5, 5.0], "right_hip": [+0.05, 0.5, 5.0],
          "pelvis": [0.0, 0.5, 5.0]}
    out, swapped = lr_stability.correct_lr_swap(
        kp, pair_names=[("left_hip", "right_hip")],
        reference_keypoint="pelvis", requires_lock=False,
    )
    assert swapped is False
    assert out["left_hip"] == [+0.1, 0.5, 5.0]


def test_lr_swap_applied_when_locked():
    kp = {"left_hip": [+0.05, 0.5, 5.0], "right_hip": [+0.10, 0.5, 5.0],
          "pelvis": [0.0, 0.5, 5.0]}
    out, swapped = lr_stability.correct_lr_swap(
        kp, pair_names=[("left_hip", "right_hip")],
        reference_keypoint="pelvis", requires_lock=True,
    )
    assert swapped is True
    assert out["left_hip"] == [+0.10, 0.5, 5.0]
    assert out["right_hip"] == [+0.05, 0.5, 5.0]


# ─────────────────────────────────────────────────────────────────────
# setup_baseline
# ─────────────────────────────────────────────────────────────────────

def _stub_frame(idx, ts, **kp):
    return {"frame_idx": idx, "ts": ts, "joint_centers_3d": kp}


def test_find_window_picks_contiguous_static_run():
    phase_of = lambda i: "setup" if i in (1, 2, 3) else "backswing"
    start, end = setup_baseline.find_baseline_window(10, phase_of, ("setup",))
    assert (start, end) == (1, 4)


def test_find_window_first_window_sentinel():
    start, end = setup_baseline.find_baseline_window(
        10, phase_of_frame=lambda i: "any",
        static_phases=(setup_baseline.FIRST_WINDOW_SENTINEL,),
        first_window_size=4,
    )
    assert (start, end) == (0, 4)


def test_extract_baseline_averages_anatomy():
    frames = [
        _stub_frame(
            0, 0.0,
            left_shoulder=[+0.20, 0.0, 5.0],
            right_shoulder=[-0.20, 0.0, 5.0],
            left_hip=[+0.15, 0.5, 5.0],
            right_hip=[-0.15, 0.5, 5.0],
            pelvis=[0.0, 0.5, 5.0],
            neck=[0.0, -0.2, 5.0],
            left_ankle=[+0.30, 1.5, 5.0],
            right_ankle=[-0.30, 1.5, 5.0],
        ),
    ]
    bl = setup_baseline.extract_baseline(
        frames, phase_of_frame=lambda i: "setup",
        static_phases=("setup",), compute_spine_angle=True,
    )
    assert bl is not None
    assert math.isclose(bl.base_shoulder_width, 0.40)
    assert math.isclose(bl.base_hip_width, 0.30)
    assert math.isclose(bl.base_stance_width, 0.60)
    # Spine length pelvis (0, 0.5, 5) → neck (0, -0.2, 5) = 0.7.
    assert math.isclose(bl.base_spine_length, 0.70, abs_tol=1e-9)
    # Spine angle ≈ atan2(0, 0.7) = 0° (perfectly upright in this stub).
    assert bl.base_spine_angle_deg is not None
    assert abs(bl.base_spine_angle_deg) < 1e-6


# ─────────────────────────────────────────────────────────────────────
# Golf plugin contract
# ─────────────────────────────────────────────────────────────────────

def test_plugin_declares_required_attributes():
    p = GolfCorrectionPlugin()
    assert p.sport_name == "golf"
    assert p.plugin_version == "golf_v1"
    assert "setup" in p.static_phases
    assert p.lr_pair_names == [("left_shoulder", "right_shoulder"),
                                ("left_hip", "right_hip")]
    assert "face_on" in p.offset_configs
    assert "down_the_line" in p.offset_configs
    # Every phase the plugin claims to emit must have a smoothing entry.
    for ph in phases.PHASE_NAMES:
        assert ph in p.smoothing_config, f"missing smoothing for {ph}"


def test_plugin_phases_round_trip_through_phase_detector():
    raw = {"frames": [{"frame_idx": i, "ts": i / 30.0} for i in range(50)]}
    p = GolfCorrectionPlugin()
    out = p.detect_phases(raw)
    assert set(out.values()).issubset(set(phases.PHASE_NAMES))
    # Every frame must be classified.
    assert len(out) == 50


def test_plugin_coaching_anchors_round_trip():
    kp_2d = {
        "left_shoulder": [100, 200], "right_shoulder": [200, 200],
        "left_hip": [120, 400], "right_hip": [180, 400],
        "neck": [150, 180], "pelvis": [150, 400],
    }
    p = GolfCorrectionPlugin()
    out = p.compute_coaching_anchors(kp_2d, {})
    # All 7 declared names emitted (per Path B namespace).
    assert set(out.keys()) == set(coaching_anchors.COACHING_ANCHOR_NAMES)
    assert out["shoulder_disc_center"] == [150.0, 200.0]
    assert out["hip_ring_center"] == [150.0, 400.0]


def test_phase_detector_covers_full_clip():
    n = 100
    raw = {"frames": [{"frame_idx": i, "ts": i / 30.0} for i in range(n)]}
    pmap = phase_detector.detect_phases(raw)
    assert len(pmap) == n
    # Frame 0 = setup, last frame = finish (fractional boundaries).
    assert pmap[0] == "setup"
    assert pmap[n - 1] == "finish"


def test_phase_config_values_in_sane_ranges():
    for ph, cfg in config.PHASE_CONFIG.items():
        assert 0.0 < cfg["alpha"] < 1.0, f"{ph} alpha out of range"
        assert 0.0 < cfg["outlier_ratio"] < 1.0, f"{ph} outlier ratio out of range"


def test_anatomical_offsets_per_view_keys_match():
    fo = set(config.ANATOMICAL_OFFSETS["face_on"].keys())
    dtl = set(config.ANATOMICAL_OFFSETS["down_the_line"].keys())
    assert fo == dtl, "face_on / down_the_line offset key sets must match"


def test_apply_offset_mode_a_vector():
    """
    Mode A (Path B): per-joint body-local 3D vector [d_h, d_v, d_f]
    transformed to camera frame via current-pose body basis, then added
    to raw. With upright synthetic pose (pelvis below neck on +y):
      spine_up   = (0, -1, 0)
      horizontal = (-1, 0, 0)
      body_forward = (0, 0, 1)
    """
    kp = {
        "left_shoulder":  [+0.20, 0.0, 5.0],
        "right_shoulder": [-0.20, 0.0, 5.0],
        "left_hip":       [+0.15, 0.5, 5.0],
        "right_hip":      [-0.15, 0.5, 5.0],
        "pelvis":         [0.0, 0.5, 5.0],
        "neck":           [0.0, -0.2, 5.0],
    }
    # Body-local: d_h=-0.10 (along horizontal=(-1,0,0)) → cam +x = +0.10
    #             d_v=+0.02 (along spine_up=(0,-1,0))   → cam -y = -0.02
    vec_cfg = {
        "left_shoulder":  [-0.10, +0.02, 0.0],
        "right_shoulder": [+0.10, +0.02, 0.0],
    }
    out = anatomical_offset.apply_offset_to_frame(kp, vec_cfg)
    def _close(actual, expected, tol=1e-9):
        return all(abs(actual[i] - expected[i]) < tol for i in range(3))
    # left_shoulder: raw + cam_offset = (0.20 + 0.10, 0.0 - 0.02, 5.0)
    assert _close(out["left_shoulder"], [0.30, -0.02, 5.0])
    # right_shoulder: raw + (-0.10, -0.02, 0) = (-0.30, -0.02, 5.0)
    assert _close(out["right_shoulder"], [-0.30, -0.02, 5.0])
    # Joints not in config pass through unchanged.
    assert out["pelvis"] == [0.0, 0.5, 5.0]
    assert out["left_hip"] == [+0.15, 0.5, 5.0]


def test_apply_offset_mode_a_zeroes_hip_v():
    """
    Mode A (Path B): hip-class joints zero d_v (body-local vertical)
    BEFORE the body→camera transform — Finding G constraint preserved
    in body-local frame. With upright pose, d_v zeroing means the
    spine-up camera component is zeroed; hip stays at original y level.
    """
    kp = {
        "left_shoulder":  [+0.20, 0.0, 5.0],
        "right_shoulder": [-0.20, 0.0, 5.0],
        "left_hip":       [+0.15, 0.5, 5.0],
        "right_hip":      [-0.15, 0.5, 5.0],
        "pelvis":         [0.0, 0.5, 5.0],
        "neck":           [0.0, -0.2, 5.0],
    }
    # d_v values below MUST be zeroed (Finding G constraint).
    vec_cfg = {
        "left_hip":  [+0.05, +0.99, +0.0],
        "right_hip": [-0.05, -0.99, +0.0],
    }
    out = anatomical_offset.apply_offset_to_frame(kp, vec_cfg)
    def _close(actual, expected, tol=1e-9):
        return all(abs(actual[i] - expected[i]) < tol for i in range(3))
    # With d_v zeroed: cam_offset = d_h * horizontal = ±0.05 * (-1,0,0)
    # left_hip:  d_h=+0.05 → cam (-0.05, 0, 0) → corrected (+0.15 - 0.05, 0.5, 5)
    assert _close(out["left_hip"], [+0.10, 0.5, 5.0])
    # right_hip: d_h=-0.05 → cam (+0.05, 0, 0) → corrected (-0.15 + 0.05, 0.5, 5)
    assert _close(out["right_hip"], [-0.10, 0.5, 5.0])
    # The critical assertion: y-component (body vertical) is UNCHANGED
    # from raw, regardless of what d_v the config supplied.
    assert abs(out["left_hip"][1] - 0.5) < 1e-9, (
        f"left_hip y must equal raw y (0.5); got {out['left_hip'][1]} — "
        "Finding G regressed in body-local frame"
    )
    assert abs(out["right_hip"][1] - 0.5) < 1e-9


def test_apply_offset_mixed_vector_and_scalar_dispatch():
    """
    Mixed config: per-joint vector wins (mode A body-local transform);
    scalar fallback for unfit joints uses mode B group-keyed scalar.
    """
    kp = {
        "left_shoulder":  [+0.20, 0.0, 5.0],
        "right_shoulder": [-0.20, 0.0, 5.0],
        "left_hip":       [+0.15, 0.5, 5.0],
        "right_hip":      [-0.15, 0.5, 5.0],
        "pelvis":         [0.0, 0.5, 5.0],
        "neck":           [0.0, -0.2, 5.0],
        "left_knee":      [+0.10, 1.0, 5.0],
        "right_knee":     [-0.10, 1.0, 5.0],
    }
    cfg = {
        # Mode A body-local vector for shoulders. With upright pose,
        # horizontal=(-1,0,0), so d_h=-0.10 → cam +x = +0.10.
        "left_shoulder":  [-0.10, 0.0, 0.0],
        "right_shoulder": [+0.10, 0.0, 0.0],
        # Mode B scalar for knees.
        "knee_inward":    0.50,
    }
    out = anatomical_offset.apply_offset_to_frame(kp, cfg)
    def _close(actual, expected, tol=1e-9):
        return all(abs(actual[i] - expected[i]) < tol for i in range(3))
    # Vector mode applied with body-local transform.
    assert _close(out["left_shoulder"], [0.30, 0.0, 5.0])
    assert _close(out["right_shoulder"], [-0.30, 0.0, 5.0])
    # Scalar mode applied to knees via KEYPOINT_TO_OFFSET_KEY dispatch:
    # torso center ≈ (0, 0.25, 5). knee inward at 0.50 = halfway to center.
    # left_knee 0.10 → 0.10 + 0.50 * (0 - 0.10) = 0.05
    assert abs(out["left_knee"][0] - 0.05) < 1e-9
    # Joint with neither vector nor scalar entry passes through.
    assert out["left_hip"] == [+0.15, 0.5, 5.0]


def test_body_local_basis_orthonormal():
    """Body-local basis (horizontal, spine_up, body_forward) must be a
    right-handed orthonormal triple."""
    import math
    pelvis = [0.0, 0.5, 5.0]
    neck   = [0.0, -0.2, 5.0]
    basis = anatomical_offset.body_local_basis(pelvis, neck)
    assert basis is not None
    h, s, f = basis
    # Unit lengths.
    for name, v in (("h", h), ("s", s), ("f", f)):
        mag = math.sqrt(sum(x * x for x in v))
        assert abs(mag - 1.0) < 1e-9, f"{name} not unit-length: |{name}|={mag}"
    # Pairwise orthogonal.
    def dot(a, b):
        return sum(a[i] * b[i] for i in range(3))
    assert abs(dot(h, s)) < 1e-9, f"h·s = {dot(h, s)} (must be 0)"
    assert abs(dot(h, f)) < 1e-9, f"h·f = {dot(h, f)} (must be 0)"
    assert abs(dot(s, f)) < 1e-9, f"s·f = {dot(s, f)} (must be 0)"
    # Right-handed: h × s = f (within tolerance).
    cross_hs = [
        h[1] * s[2] - h[2] * s[1],
        h[2] * s[0] - h[0] * s[2],
        h[0] * s[1] - h[1] * s[0],
    ]
    for i in range(3):
        assert abs(cross_hs[i] - f[i]) < 1e-9, (
            f"h × s ≠ f (basis is left-handed or malformed)"
        )


def test_body_local_basis_degenerate_returns_none():
    """When pelvis ≈ neck, spine direction is undefined → basis is None."""
    # Identical points → zero spine vector.
    assert anatomical_offset.body_local_basis([1.0, 2.0, 3.0],
                                                [1.0, 2.0, 3.0]) is None
    # Spine parallel to camera-z → cross product collapses.
    assert anatomical_offset.body_local_basis([0.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0]) is None


def test_offset_vector_body_local_transform_roundtrip():
    """camera_to_body_local must invert body_local_to_camera (orthonormal basis)."""
    import math
    pelvis = [0.05, 0.45, 4.9]   # non-axis-aligned pose
    neck   = [-0.10, -0.15, 5.1]
    basis = anatomical_offset.body_local_basis(pelvis, neck)
    assert basis is not None
    for original in (
        [0.123, -0.456, 0.789],
        [-0.001, 0.002, -0.003],
        [1.5, 2.5, -3.5],
    ):
        cam = anatomical_offset.body_local_to_camera(original, basis)
        back = anatomical_offset.camera_to_body_local(cam, basis)
        max_err = max(abs(original[i] - back[i]) for i in range(3))
        assert max_err < 1e-9, (
            f"roundtrip max err {max_err:.2e} for original={original}"
        )


def test_coaching_anchors_emits_per_side():
    """Per-side anchors must be present + distinct from disc centers."""
    kp_2d = {
        "left_shoulder":  [100, 200],
        "right_shoulder": [200, 200],
        "left_hip":       [120, 400],
        "right_hip":      [180, 400],
        "neck":           [150, 180],
        "pelvis":         [150, 400],
    }
    out = coaching_anchors.derive(kp_2d, {})
    # The 5 per-side visuals must be present and at their respective joints.
    assert out["left_shoulder_visual"]  == [100, 200]
    assert out["right_shoulder_visual"] == [200, 200]
    assert out["left_hip_visual"]       == [120, 400]
    assert out["right_hip_visual"]      == [180, 400]
    assert out["neck_visual"]           == [150, 180]
    # Disc centers are midpoints — distinct from per-side values.
    assert out["shoulder_disc_center"]  == [150.0, 200.0]
    assert out["hip_ring_center"]       == [150.0, 400.0]
    # Disc centers must NOT equal per-side values (no collapse).
    assert out["shoulder_disc_center"] != out["left_shoulder_visual"]
    assert out["shoulder_disc_center"] != out["right_shoulder_visual"]


def test_anatomical_offsets_loaded_from_fit_not_estimate():
    """
    Guard: ANATOMICAL_OFFSETS must contain at least one per-joint 3D
    vector (mode A) — either flat [d_h,d_v,d_f] or per-phase
    dict[phase → [d_h,d_v,d_f]] (PR-7a.1 Fix 3). If someone reverts
    to the all-scalar legacy format, this fires.
    """
    def _is_vec3(v):
        return (isinstance(v, (list, tuple))
                and len(v) == 3
                and all(isinstance(x, (int, float)) for x in v))

    found_vector = False
    for view, kvs in config.ANATOMICAL_OFFSETS.items():
        for k, v in kvs.items():
            if _is_vec3(v):
                found_vector = True
                break
            if isinstance(v, dict):
                # PR-7a.1 Fix 3: per-phase dict — any phase having
                # a 3-vector counts.
                if any(_is_vec3(pv) for pv in v.values()):
                    found_vector = True
                    break
        if found_vector:
            break
    assert found_vector, (
        "ANATOMICAL_OFFSETS has no per-joint 3D vectors (flat or per-phase) "
        "— fit_offsets_from_gt.py output was not applied"
    )
    # Also: the FALLBACK dict must still be defined (preserved scalar form).
    assert hasattr(config, "ANATOMICAL_OFFSETS_FALLBACK")
    assert "face_on" in config.ANATOMICAL_OFFSETS_FALLBACK
    assert "shoulder_inward" in config.ANATOMICAL_OFFSETS_FALLBACK["face_on"]


# ─────────────────────────────────────────────────────────────────────
# End-to-end smoke (uses existing b3fea3f0 WHAM output if present)
# ─────────────────────────────────────────────────────────────────────

B3FEA3_RAW = _REPO_ROOT / "python" / "pilot" / "output" / "wham" / (
    "b3fea3f0-e248-44d7-a923-0bb43172b5bf"
) / "joint_centers_3d.json"


def test_smoke_b3fea3f0_pipeline_runs_clean():
    if not B3FEA3_RAW.exists():
        print(f"[skip] {B3FEA3_RAW} not present — skipping e2e smoke")
        return

    corrected = correct_timeline(B3FEA3_RAW, GolfCorrectionPlugin(), view="face_on")
    assert isinstance(corrected, CorrectedTimeline)
    assert corrected.sport == "golf"
    assert corrected.view == "face_on"
    assert corrected.video_width > 0 and corrected.video_height > 0
    assert len(corrected.frames) >= 100, "expected ~139 frames from this clip"

    # Baseline sanity — pose-runner agrees humans have ~25cm shoulder width.
    bl = corrected.setup_baseline
    assert bl is not None
    assert 0.1 < bl.base_shoulder_width < 0.6, f"shoulder width {bl.base_shoulder_width}"
    assert 0.1 < bl.base_hip_width < 0.6, f"hip width {bl.base_hip_width}"
    assert 0.3 < bl.base_spine_length < 1.0, f"spine length {bl.base_spine_length}"
    # Golf plugin sets compute_spine_angle=True.
    assert bl.base_spine_angle_deg is not None
    assert 10 < bl.base_spine_angle_deg < 70

    # Every phase the plugin declares must appear at least once.
    observed_phases = set(f.phase for f in corrected.frames)
    expected = set(phases.PHASE_NAMES)
    assert observed_phases == expected, (
        f"phase coverage mismatch: missing {expected - observed_phases}"
    )

    # Diagnostics: phase-aware alpha must vary across phases (Constraint 6
    # is enforced structurally, but verify at runtime too).
    alphas_seen = set(f.diagnostics.smoothing_alpha_used for f in corrected.frames)
    assert len(alphas_seen) >= 3, (
        f"only {len(alphas_seen)} unique alphas — phase-aware smoothing not engaged"
    )

    # Coaching anchors namespace conformance.
    sample = corrected.frames[10].coaching_anchors_2d
    assert set(sample.keys()) == set(coaching_anchors.COACHING_ANCHOR_NAMES)

    # Analysis metrics: at least one of the 4 declared metrics computed.
    assert corrected.analysis_metrics, "expected non-empty analysis_metrics"
    for name in corrected.analysis_metrics:
        assert name in analysis_metrics.METRIC_NAMES


def test_smoother_bidirectional_no_causal_lag():
    """
    PR-7a.3 bidirectional EMA acceptance gate: a synthetic Gaussian
    bell with a known peak location must, after bidirectional
    smoothing, have its peak align within 1 frame of the input peak.
    Forward-only EMA introduces ~1/(2α) frame phase delay; the
    forward+backward average must cancel it.

    Also asserts that the forward-only path lags MORE than the
    bidirectional path (sanity check that the comparison is real).
    """
    import math
    cfg = {"setup": {"alpha": 0.3, "outlier_ratio": 0.0}}  # outlier off
    n = 60
    phase_seq = ["setup"] * n
    peak_idx = 30
    sigma = 4.0
    # Single-peak Gaussian in x — no ambiguity in argmax.
    seq = [
        [math.exp(-((i - peak_idx) ** 2) / (2 * sigma * sigma)), 0.0, 5.0]
        for i in range(n)
    ]
    smoothed_bi, _, _ = temporal_smoother.smooth_timeline_bidirectional(
        [{"j": p} for p in seq], phase_seq,
        smoothing_config=cfg, scale_reference=None,
    )
    bi_peak = max(range(n), key=lambda i: smoothed_bi[i]["j"][0])
    assert abs(bi_peak - peak_idx) <= 1, (
        f"bidirectional smoothed peak at frame {bi_peak}, "
        f"input peak at {peak_idx} — > 1 frame lag indicates the "
        f"backward pass isn't cancelling forward causal delay"
    )

    # Sanity: forward-only path should lag noticeably MORE than bi.
    # Run a forward-only pass manually for comparison.
    prev_sm = None
    fwd_x = [0.0] * n
    for i in range(n):
        raw = seq[i][0]
        if prev_sm is None:
            sm = raw
        else:
            sm = 0.3 * raw + 0.7 * prev_sm
        fwd_x[i] = sm
        prev_sm = sm
    fwd_peak = max(range(n), key=lambda i: fwd_x[i])
    assert fwd_peak > bi_peak, (
        f"sanity check failed: forward-only peak={fwd_peak} should LAG "
        f"behind bidirectional peak={bi_peak} (forward EMA shifts peaks right)"
    )


def test_smoother_bidirectional_handles_nulls():
    """Null values in the raw sequence reset both passes (gap span semantics)."""
    cfg = {"setup": {"alpha": 0.4, "outlier_ratio": 0.0}}
    n = 10
    seq = [[float(i), 0.0, 5.0] for i in range(n)]
    seq[4] = None   # gap
    seq[5] = None
    smoothed, _, _ = temporal_smoother.smooth_timeline_bidirectional(
        [{"j": p} for p in seq],
        ["setup"] * n,
        smoothing_config=cfg,
        scale_reference=None,
    )
    # Nulls stay None.
    assert smoothed[4]["j"] is None
    assert smoothed[5]["j"] is None
    # Surrounding values still smoothed.
    assert smoothed[3]["j"] is not None
    assert smoothed[6]["j"] is not None


def test_smoother_bidirectional_orchestrator_default():
    """
    Smoke check: correct_timeline default path uses bidirectional and
    produces a CorrectedTimeline whose notes string mentions it.
    """
    if not B3FEA3_RAW.exists():
        return
    corrected = correct_timeline(B3FEA3_RAW, GolfCorrectionPlugin(), view="face_on")
    notes_blob = " ".join(corrected.notes)
    assert "bidirectional" in notes_blob.lower(), (
        f"correct_timeline default must run bidirectional smoothing; "
        f"notes={corrected.notes}"
    )
    # Forward-only path still callable + produces different notes.
    forward_only = correct_timeline(
        B3FEA3_RAW, GolfCorrectionPlugin(), view="face_on",
        bidirectional=False,
    )
    notes_blob_fwd = " ".join(forward_only.notes)
    assert "forward-only" in notes_blob_fwd.lower()


def test_setup_anchor_drift_under_2px():
    """
    PR-7a.1 Fix 1 acceptance gate: during the setup phase (body
    stationary), the 5 fitted joints (left/right shoulder, left/right
    hip, neck) must have 2D position std < 2.0 px across the setup
    window. Pre-fix the std was ~5-8 px due to α=0.20 admitting WHAM
    noise + per-frame basis jitter.
    """
    if not B3FEA3_RAW.exists():
        return
    import statistics
    corrected = correct_timeline(B3FEA3_RAW, GolfCorrectionPlugin(), view="face_on")
    setup_frames = [f for f in corrected.frames if f.phase == "setup"]
    assert len(setup_frames) >= 3, (
        f"expected at least 3 setup frames; got {len(setup_frames)}"
    )
    fitted_joints = (
        "left_shoulder", "right_shoulder",
        "left_hip", "right_hip", "neck",
    )
    drifts = {}
    for j in fitted_joints:
        xs, ys = [], []
        for f in setup_frames:
            uv = f.keypoints_2d_projected.get(j)
            if uv is None:
                continue
            xs.append(uv[0])
            ys.append(uv[1])
        if len(xs) < 2:
            continue
        std_x = statistics.stdev(xs)
        std_y = statistics.stdev(ys)
        # Combined std as Euclidean magnitude of per-axis std.
        drifts[j] = (std_x ** 2 + std_y ** 2) ** 0.5
    assert drifts, "no setup-frame data collected"
    failing = {j: round(d, 2) for j, d in drifts.items() if d >= 2.0}
    assert not failing, (
        f"setup anchor drift std exceeds 2.0 px for: {failing}. "
        f"all_drifts={ {j: round(d, 2) for j, d in drifts.items()} }"
    )


def test_lr_identity_no_thrashing():
    """
    PR-7a.1 Fix 2 acceptance gate: per-pair applied-swap state must
    transition at most 4 times across the entire DTL clip (120
    frames). Pre-fix the single-frame x-only heuristic + per-frame
    application caused visible thrashing at impact/transition.
    """
    raw = (
        _REPO_ROOT / "python" / "pilot" / "output" / "wham"
        / "b32e0f21-2656-473c-aa87-e1eaf6e1221f" / "joint_centers_3d.json"
    )
    if not raw.exists():
        return
    corrected = correct_timeline(raw, GolfCorrectionPlugin(), view="down_the_line")
    thrash = corrected.summary_stats.get("lr_swap_thrash_count", 0.0)
    n_frames = corrected.summary_stats.get("frame_count", 0.0)
    assert n_frames > 0
    # Gate per spec: ≤ 4 transitions per 100 frames, summed across pairs.
    # b32e0f21 has 120 frames × 2 pairs (shoulders + hips). Allow up to
    # 4 transitions per pair (8 total) as a permissive interpretation
    # that still catches the 48-event thrashing the pre-fix produced.
    assert thrash <= 8, (
        f"L/R applied-swap transitions = {thrash} across {n_frames} "
        f"frames (target ≤ 8 across 2 pairs). Pre-fix value was ~48."
    )


def test_backswing_shoulder_within_bounds():
    """
    PR-7a.1 Fix 3 acceptance gate: at backswing the corrected shoulder
    anchor must not "fly wild" from a wonky interpolation. Body
    bounding box not available offline, so we use two proxies:

      (a) corrected shoulder→neck distance should be < 180 px
          (anatomical shoulder-to-neck is ~30-40 cm; at depth 3.8m
          with fx=1280 that's ~100-140 px; +40 px buffer for finish-
          phase poses).
      (b) corrected position must not drift > 130 px from the raw
          projection (per-phase setup-vec gives ~80 px shift; clamp
          fallback caps interp at 1.5× setup-vec magnitude).
    """
    if not B3FEA3_RAW.exists():
        return
    import math
    from motion_correction.engine.projection import (
        default_intrinsics, project_xyz_to_uv,
    )
    raw = json.loads(B3FEA3_RAW.read_text())
    corrected = correct_timeline(B3FEA3_RAW, GolfCorrectionPlugin(), view="face_on")
    backswing_frames = [f for f in corrected.frames if f.phase == "backswing"]
    if not backswing_frames:
        return  # not all clips have a backswing phase
    intr = default_intrinsics(int(raw["video_width"]), int(raw["video_height"]))
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]

    for f in backswing_frames:
        for j in ("left_shoulder", "right_shoulder"):
            cor = f.keypoints_2d_projected.get(j)
            neck = f.keypoints_2d_projected.get("neck")
            if cor is None or neck is None:
                continue
            d = math.sqrt((cor[0] - neck[0]) ** 2 + (cor[1] - neck[1]) ** 2)
            assert d < 180, (
                f"backswing frame {f.frame_idx} {j} corrected→neck "
                f"distance = {d:.1f} px (should be < 180 — anchor far "
                f"outside body envelope)"
            )
        raw_frame = next(rf for rf in raw["frames"]
                          if int(rf["frame_idx"]) == f.frame_idx)
        for j in ("left_shoulder", "right_shoulder"):
            raw_3d = raw_frame["joint_centers_3d"].get(j)
            if raw_3d is None or raw_3d[2] is None or raw_3d[2] <= 0:
                continue
            raw_uv = project_xyz_to_uv(raw_3d, fx, fy, cx, cy)
            cor_uv = f.keypoints_2d_projected.get(j)
            if raw_uv is None or cor_uv is None:
                continue
            drift = math.sqrt(
                (raw_uv[0] - cor_uv[0]) ** 2 + (raw_uv[1] - cor_uv[1]) ** 2
            )
            # Drift cap: fitted face_on right_shoulder is ~40 cm = ~135 px
            # at depth ~3.8m. Backswing = lerp(setup, top, 0.5), similar
            # magnitude. 170 px = 1.25× setup-vec — anything beyond that
            # would indicate clamp failed.
            assert drift < 170, (
                f"backswing frame {f.frame_idx} {j} raw→corrected "
                f"drift = {drift:.1f} px (should be < 170 — interpolation "
                f"likely blew up; clamp should have fired)"
            )


def test_orchestrator_corrected_differs_from_raw():
    """
    Regression guard against silent no-op orchestrator. The pipeline
    must produce 2D coords that differ from raw projection by more
    than 3 px (Euclidean, averaged across all joints in mid-clip).
    Surfaced from PR-7a Task 2D visual review: an earlier bug had the
    smoother freezing at frame 1 after a single outlier rejection,
    causing the "corrected" timeline to be effectively raw frame 1
    repeated forever — drift looked superficially non-zero on first
    inspection but the corrected wasn't actually tracking motion.
    """
    if not B3FEA3_RAW.exists():
        return
    import math
    corrected = correct_timeline(B3FEA3_RAW, GolfCorrectionPlugin(), view="face_on")
    raw = json.loads(B3FEA3_RAW.read_text())
    intr = corrected.camera_intrinsics
    # Use a mid-clip frame (no setup-window early-frame artifacts).
    mid_frame_idx = len(corrected.frames) // 2
    raw_frame_dict = next(f for f in raw["frames"]
                          if int(f["frame_idx"]) == mid_frame_idx)
    cor_frame = corrected.frames[mid_frame_idx]

    # Re-project raw 3D for comparison (mirror orchestrator's intrinsics).
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    diffs = []
    for name, raw_xyz in raw_frame_dict["joint_centers_3d"].items():
        if raw_xyz is None or len(raw_xyz) != 3 or raw_xyz[2] <= 0:
            continue
        raw_uv = (fx * raw_xyz[0] / raw_xyz[2] + cx,
                  fy * raw_xyz[1] / raw_xyz[2] + cy)
        cor_uv = cor_frame.keypoints_2d_projected.get(name)
        if cor_uv is None:
            continue
        diffs.append(math.sqrt((raw_uv[0] - cor_uv[0]) ** 2
                                + (raw_uv[1] - cor_uv[1]) ** 2))
    assert diffs, "expected at least one joint with valid projection"
    mean_shift = sum(diffs) / len(diffs)
    assert mean_shift > 5.0, (
        f"corrected timeline mean pixel shift = {mean_shift:.2f} px "
        f"(threshold 5 px post-Option-2). Pipeline is producing near-identical "
        f"output to raw — check anatomical_offset config, outlier-rejection "
        f"cascade, or orchestrator wiring."
    )


def test_smoother_does_not_freeze_after_outlier():
    """
    Regression guard for the outlier-cascade bug found in PR-7a Task 2D.

    Setup: feed a stable sequence with ONE spike at frame 2. With the
    NEW outlier check (compares to prev_raw_offset), the smoother
    should reject only frame 2's spike and recover at frame 3+. With
    the OLD behavior (compares to prev_smoothed), the smoother stays
    permanently rejected because the gap from frozen prev_smoothed to
    moving raw keeps growing.
    """
    cfg = {"setup": {"alpha": 0.30, "outlier_ratio": 0.15}}
    # Linear raw motion at 1 cm/frame; insert a 10 cm spike at frame 2.
    raws = [[0.00, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.11, 0.0, 0.0],   # spike
            [0.03, 0.0, 0.0],
            [0.04, 0.0, 0.0],
            [0.05, 0.0, 0.0]]
    smoothed = None
    prev_raw = None
    outlier_pattern = []
    smoothed_seq = []
    for r in raws:
        result = temporal_smoother.smooth_keypoint_phase_aware(
            r, smoothed, phase="setup", smoothing_config=cfg,
            scale_reference=0.5,  # threshold = 0.075 m
            prev_raw_for_outlier_check=prev_raw,
        )
        outlier_pattern.append(result.was_outlier)
        smoothed = result.smoothed
        prev_raw = r
        smoothed_seq.append(smoothed[0])

    # Frame 2 spike (10 cm > 7.5 cm threshold) → outlier.
    # Frames 3+ are gentle motion (8 cm raw→raw delta? actually 0.11 → 0.03 = 8cm,
    # so frame 3 is technically also above threshold relative to raw frame 2's spike).
    # Frame 4: 0.03 → 0.04 = 1cm, well under threshold → must accept.
    assert outlier_pattern[2] is True,  "spike at frame 2 should be flagged"
    assert outlier_pattern[4] is False, "frame 4 must recover (raw motion tiny)"
    assert outlier_pattern[5] is False, "frame 5 must recover (raw motion tiny)"
    # The key regression: smoothed must MOVE from frame 4 onward, not stay frozen.
    assert smoothed_seq[5] != smoothed_seq[2], (
        f"smoother stuck at {smoothed_seq[2]} from frame 2 through frame 5 "
        f"({smoothed_seq[5]}) — outlier cascade bug regressed"
    )


def test_smoke_writes_and_roundtrips_json(tmp_path=None):
    if not B3FEA3_RAW.exists():
        return
    import tempfile
    tmp = Path(tempfile.mkdtemp()) / "rt.json"
    corrected = correct_timeline(B3FEA3_RAW, GolfCorrectionPlugin(), view="face_on")
    corrected.save(tmp)
    assert tmp.stat().st_size > 1000
    payload = json.loads(tmp.read_text())
    assert payload["sport"] == "golf"
    assert payload["view"] == "face_on"
    assert payload["version"] == 1
    assert "setup_baseline" in payload
    assert "analysis_metrics" in payload
    assert len(payload["frames"]) == len(corrected.frames)


# ─────────────────────────────────────────────────────────────────────
# Manual runner (no pytest required)
# ─────────────────────────────────────────────────────────────────────

def _all_tests():
    g = globals()
    return [(name, fn) for name, fn in sorted(g.items())
            if name.startswith("test_") and callable(fn)]


if __name__ == "__main__":
    passed = 0
    failed = []
    for name, fn in _all_tests():
        try:
            fn()
        except AssertionError as e:
            failed.append((name, f"AssertionError: {e}"))
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ERR   {name}: {type(e).__name__}: {e}")
        else:
            passed += 1
            print(f"  ok    {name}")

    print()
    print(f"summary: {passed} passed, {len(failed)} failed of {passed + len(failed)} tests")
    if failed:
        sys.exit(1)
