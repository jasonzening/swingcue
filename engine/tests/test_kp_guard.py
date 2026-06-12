"""
engine/tests/test_kp_guard.py
Unit tests for the keypoint validity guard and head_ref_v2.

Covers:
  - (0,0) case that caused the fo-ok-2 +72.5% bug
  - Boundary / off-frame coordinates
  - Low confidence rejection
  - head_ref_v2 priority order (ears > nose > eyes)
  - Single-ear flag
  - no_ear_nose_only flag
  - degraded (eyes-only) flag
  - All-invalid → None
"""
import pytest
from engine.a_measurement.kp_guard import kp_guard, head_ref_v2, HeadRef


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_kp(x, y, score):
    return {"x": float(x), "y": float(y), "score": float(score)}


def kps(**kwargs):
    """Build a keypoint dict from name=>(x,y,score) triples."""
    return {name: make_kp(*v) for name, v in kwargs.items()}


# ---------------------------------------------------------------------------
# kp_guard tests
# ---------------------------------------------------------------------------

class TestKpGuard:

    def test_valid_returns_xy(self):
        d = kps(left_hip=(300, 500, 0.9))
        assert kp_guard(d, "left_hip") == pytest.approx((300.0, 500.0))

    def test_zero_zero_rejected(self):
        """THE fo-ok-2 BUG: right_ear=(0,0,0.339) must be rejected."""
        d = kps(right_ear=(0.0, 0.0, 0.339))
        assert kp_guard(d, "right_ear") is None

    def test_zero_x_rejected(self):
        d = kps(left_ear=(0.0, 250.0, 0.8))
        assert kp_guard(d, "left_ear") is None

    def test_zero_y_rejected(self):
        d = kps(nose=(320.0, 0.0, 0.8))
        assert kp_guard(d, "nose") is None

    def test_low_score_rejected(self):
        d = kps(left_hip=(300, 500, 0.25))
        assert kp_guard(d, "left_hip") is None

    def test_score_exactly_at_threshold_passes(self):
        d = kps(left_hip=(300, 500, 0.30))
        assert kp_guard(d, "left_hip") == pytest.approx((300.0, 500.0))

    def test_score_just_below_threshold_fails(self):
        d = kps(left_hip=(300, 500, 0.299))
        assert kp_guard(d, "left_hip") is None

    def test_missing_key_returns_none(self):
        d = {}
        assert kp_guard(d, "left_hip") is None

    def test_frame_width_bound(self):
        d = kps(left_hip=(720, 400, 0.9))
        assert kp_guard(d, "left_hip", frame_w=720) is None   # x >= frame_w
        assert kp_guard(d, "left_hip", frame_w=721) is not None

    def test_frame_height_bound(self):
        d = kps(left_hip=(300, 1280, 0.9))
        assert kp_guard(d, "left_hip", frame_h=1280) is None  # y >= frame_h
        assert kp_guard(d, "left_hip", frame_h=1281) is not None

    def test_custom_threshold(self):
        d = kps(left_hip=(300, 500, 0.45))
        assert kp_guard(d, "left_hip", thr=0.5) is None
        assert kp_guard(d, "left_hip", thr=0.4) is not None

    def test_negative_x_rejected(self):
        d = kps(left_hip=(-5, 400, 0.9))
        assert kp_guard(d, "left_hip") is None

    def test_negative_y_rejected(self):
        d = kps(left_hip=(200, -1, 0.9))
        assert kp_guard(d, "left_hip") is None

    def test_small_positive_xy_passes(self):
        d = kps(left_hip=(0.1, 0.1, 0.9))
        result = kp_guard(d, "left_hip")
        assert result == pytest.approx((0.1, 0.1))


# ---------------------------------------------------------------------------
# head_ref_v2 tests
# ---------------------------------------------------------------------------

class TestHeadRefV2:

    def test_both_ears_source_and_no_flag(self):
        d = kps(left_ear=(200, 400, 0.9), right_ear=(260, 398, 0.85))
        hr = head_ref_v2(d)
        assert hr is not None
        assert hr.source == "ears_both"
        assert hr.flag == ""
        assert hr.x == pytest.approx((200 + 260) / 2)
        assert hr.y == pytest.approx((400 + 398) / 2)
        assert hr.is_clean

    def test_left_ear_only_single_ear_flag(self):
        d = kps(left_ear=(200, 400, 0.9), right_ear=(0, 0, 0.9))  # right_ear=(0,0)
        hr = head_ref_v2(d)
        assert hr.source == "ear_left"
        assert hr.flag == "single_ear"
        assert hr.x == pytest.approx(200)

    def test_right_ear_only_single_ear_flag(self):
        d = kps(left_ear=(0, 0, 0.9), right_ear=(260, 398, 0.85))
        hr = head_ref_v2(d)
        assert hr.source == "ear_right"
        assert hr.flag == "single_ear"

    def test_zero_zero_ear_falls_through_to_nose(self):
        """Reproduces fo-ok-2 fr75: right_ear=(0,0,0.339), should use nose."""
        d = kps(
            left_ear=(0, 0, 0.3),       # invalid (0,0)
            right_ear=(0.0, 0.0, 0.339), # THE BUG: (0,0) with score 0.339
            nose=(262.9, 553.3, 0.915),
        )
        hr = head_ref_v2(d)
        assert hr is not None
        assert hr.source == "nose"
        assert hr.flag == "no_ear_nose_only"
        assert hr.x == pytest.approx(262.9)
        assert hr.y == pytest.approx(553.3)

    def test_nose_fallback_when_no_ears(self):
        d = kps(nose=(300, 500, 0.8))
        hr = head_ref_v2(d)
        assert hr.source == "nose"
        assert hr.flag == "no_ear_nose_only"
        assert not hr.is_clean

    def test_eyes_fallback_degraded(self):
        d = kps(left_eye=(280, 450, 0.8), right_eye=(320, 448, 0.75))
        hr = head_ref_v2(d)
        assert hr.source == "eyes"
        assert hr.flag == "degraded"
        assert hr.needs_human

    def test_single_eye_fallback_degraded(self):
        d = kps(left_eye=(280, 450, 0.8))
        hr = head_ref_v2(d)
        assert hr.source == "eye_left"
        assert hr.flag == "degraded"

    def test_all_invalid_returns_none(self):
        d = kps(
            left_ear=(0, 0, 0.9),
            right_ear=(0, 0, 0.9),
            nose=(0, 0, 0.9),
            left_eye=(0, 0, 0.9),
            right_eye=(0, 0, 0.9),
        )
        assert head_ref_v2(d) is None

    def test_empty_dict_returns_none(self):
        assert head_ref_v2({}) is None

    def test_low_score_ears_fall_through_to_nose(self):
        d = kps(
            left_ear=(200, 400, 0.1),   # below threshold
            right_ear=(260, 398, 0.2),  # below threshold
            nose=(300, 500, 0.9),
        )
        hr = head_ref_v2(d)
        assert hr.source == "nose"

    def test_is_clean_property(self):
        d = kps(left_ear=(200, 400, 0.9), right_ear=(260, 398, 0.9))
        assert head_ref_v2(d).is_clean is True

    def test_needs_human_property(self):
        d = kps(nose=(300, 500, 0.8))
        assert head_ref_v2(d).needs_human is True    # no_ear_nose_only: ears unavailable → flag human
        d2 = kps(left_eye=(280, 450, 0.8))
        assert head_ref_v2(d2).needs_human is True   # degraded is needs_human

    def test_custom_threshold(self):
        d = kps(left_ear=(200, 400, 0.45), right_ear=(260, 398, 0.45))
        assert head_ref_v2(d, thr=0.5) is None    # both fail thr=0.5
        hr = head_ref_v2(d, thr=0.4)
        assert hr.source == "ears_both"

    def test_frame_bounds_applied(self):
        """Ear within score threshold but outside frame → falls through to nose."""
        d = kps(
            left_ear=(720, 400, 0.9),   # x == frame_w, invalid
            right_ear=(760, 400, 0.9),  # x > frame_w, invalid
            nose=(300, 500, 0.9),
        )
        hr = head_ref_v2(d, frame_w=720)
        assert hr.source == "nose"

    def test_pt_property(self):
        d = kps(left_ear=(200, 400, 0.9), right_ear=(260, 398, 0.9))
        hr = head_ref_v2(d)
        x, y = hr.pt
        assert x == pytest.approx(230)
        assert y == pytest.approx(399)
