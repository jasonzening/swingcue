"""
7 synthetic unit tests for the judgment core (D + E + F layers).

All tests use synthetic numpy arrays — zero video/GPU dependency.
Each test maps to the spec's test table (Section 4 of JUDGMENT_CORE_SPEC.md).

T1  Normal swing          -> no fault, positive feedback
T2  Typical EE            -> Likely early_extension
T3  Pure posture loss     -> loss_of_posture, no EE attribution
T4  Timing reversed       -> independent faults, NOT early_extension
T5  Noise frames          -> sentinel excludes them, rule NOT triggered
T6  Low confidence        -> rule skipped entirely
T7  Boundary (exactly 8°) -> triggers at boundary (>=), mild severity
"""

from __future__ import annotations

import numpy as np
import pytest

from src.judgment.rules import (
    FaultDetection,
    bone_length_sentinel,
    r1_loss_of_posture,
    r2_hip_toward_ball,
)
from src.judgment.root_cause import RootCauseEngine
from src.judgment.output import CoachingOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_phases(n_address=10, n_takeaway=10, n_backswing=10,
                n_top=5, n_transition=10, n_downswing=15, n_impact=5,
                n_follow=10):
    """Build a phase_labels list of total frames."""
    return (
        ["address"] * n_address
        + ["takeaway"] * n_takeaway
        + ["backswing"] * n_backswing
        + ["top"] * n_top
        + ["transition"] * n_transition
        + ["downswing"] * n_downswing
        + ["impact"] * n_impact
        + ["follow_through"] * n_follow
    )


def frame_count(phases):
    return len(phases)


engine = RootCauseEngine()
coaching = CoachingOutput()


# ---------------------------------------------------------------------------
# T1: Normal swing — no fault triggered
# ---------------------------------------------------------------------------

class TestT1NormalSwing:
    """
    T1: spine delta stays ±3°, hip stays ±2% -> no faults, positive feedback.
    """

    def setup_method(self):
        self.phases = make_phases()
        n = frame_count(self.phases)
        self.spine = np.random.uniform(-3, 3, n)
        self.hip = np.random.uniform(-0.02, 0.02, n)
        self.conf = np.ones(n)

    def test_r1_not_triggered(self):
        result = r1_loss_of_posture(self.spine, self.phases, self.conf)
        assert result is None, "R1 should not trigger on normal swing"

    def test_r2_not_triggered(self):
        result = r2_hip_toward_ball(self.hip, self.phases, self.conf)
        assert result is None, "R2 should not trigger on normal swing"

    def test_root_cause_none(self):
        rc = engine.analyze([])
        assert rc.root_cause == "none"
        assert rc.certainty == "none"

    def test_output_positive_feedback(self):
        rc = engine.analyze([])
        out = coaching.generate(rc)
        assert "没有检出" in out.one_liner or "不错" in out.one_liner


# ---------------------------------------------------------------------------
# T2: Typical Early Extension — R2 onset early, R1 follows
# ---------------------------------------------------------------------------

class TestT2TypicalEarlyExtension:
    """
    T2: hip shifts +8% starting at transition (onset frame ~35),
        spine delta rises +12° starting at downswing (onset frame ~45).
        Expected: Likely early_extension.
    """

    def setup_method(self):
        self.phases = make_phases()
        n = frame_count(self.phases)
        self.spine = np.zeros(n)
        self.hip = np.zeros(n)
        self.conf = np.ones(n)

        # R2: hip starts at transition (frame 35), stays high through impact
        # transition starts at 10+10+10+5 = 35
        for i in range(35, n):
            self.hip[i] = 0.08   # 8% > 5% threshold

        # R1: spine starts at downswing (frame 45), stays high through impact
        # downswing starts at 35+10 = 45
        for i in range(45, n):
            self.spine[i] = 12.0  # 12° (significant)

    def test_r1_triggered(self):
        r1 = r1_loss_of_posture(self.spine, self.phases, self.conf)
        assert r1 is not None
        assert r1.fault_type == "loss_of_posture"

    def test_r2_triggered(self):
        r2 = r2_hip_toward_ball(self.hip, self.phases, self.conf)
        assert r2 is not None
        assert r2.fault_type == "hip_toward_ball"

    def test_r2_onset_before_r1(self):
        r1 = r1_loss_of_posture(self.spine, self.phases, self.conf)
        r2 = r2_hip_toward_ball(self.hip, self.phases, self.conf)
        assert r2.onset_frame <= r1.onset_frame, "R2 should start before R1"

    def test_early_extension_likely(self):
        r1 = r1_loss_of_posture(self.spine, self.phases, self.conf)
        r2 = r2_hip_toward_ball(self.hip, self.phases, self.conf)
        rc = engine.analyze([r1, r2])
        assert rc.root_cause == "early_extension"
        assert rc.certainty == "likely"

    def test_posture_loss_marked_as_result_not_root(self):
        """loss_of_posture should be in causal chain (as result), not independent."""
        r1 = r1_loss_of_posture(self.spine, self.phases, self.conf)
        r2 = r2_hip_toward_ball(self.hip, self.phases, self.conf)
        rc = engine.analyze([r1, r2])
        assert "loss_of_posture" in rc.causal_chain
        assert rc.root_cause != "loss_of_posture"

    def test_output_contains_ee_message(self):
        r1 = r1_loss_of_posture(self.spine, self.phases, self.conf)
        r2 = r2_hip_toward_ball(self.hip, self.phases, self.conf)
        rc = engine.analyze([r1, r2])
        out = coaching.generate(rc)
        assert "Early Extension" in out.one_liner
        assert "很可能" in out.one_liner


# ---------------------------------------------------------------------------
# T3: Pure posture loss — spine rises, hip steady -> loss_of_posture, no EE
# ---------------------------------------------------------------------------

class TestT3PurePostureLoss:
    """
    T3: spine delta +10° in downswing, hip displacement stable.
        Expected: loss_of_posture, NO early_extension.
    """

    def setup_method(self):
        self.phases = make_phases()
        n = frame_count(self.phases)
        self.spine = np.zeros(n)
        self.hip = np.zeros(n)
        self.conf = np.ones(n)

        # downswing starts at frame 45
        for i in range(45, n):
            self.spine[i] = 10.0  # 10° mild-significant boundary

        # hip stays flat
        self.hip[:] = 0.01

    def test_r1_triggered(self):
        r1 = r1_loss_of_posture(self.spine, self.phases, self.conf)
        assert r1 is not None

    def test_r2_not_triggered(self):
        r2 = r2_hip_toward_ball(self.hip, self.phases, self.conf)
        assert r2 is None

    def test_root_cause_is_loss_of_posture_not_ee(self):
        r1 = r1_loss_of_posture(self.spine, self.phases, self.conf)
        rc = engine.analyze([r1])
        assert rc.root_cause == "loss_of_posture"
        assert rc.root_cause != "early_extension"

    def test_output_no_ee_mention(self):
        r1 = r1_loss_of_posture(self.spine, self.phases, self.conf)
        rc = engine.analyze([r1])
        out = coaching.generate(rc)
        assert "Early Extension" not in out.one_liner


# ---------------------------------------------------------------------------
# T4: Timing reversed — spine rises FIRST, hip follows -> NOT EE
# ---------------------------------------------------------------------------

class TestT4TimingReversed:
    """
    T4: spine onset at frame 45, hip onset at frame 55 (> R1 + tolerance).
        Expected: independent faults, NOT early_extension.
    """

    def setup_method(self):
        self.phases = make_phases()
        n = frame_count(self.phases)
        self.spine = np.zeros(n)
        self.hip = np.zeros(n)
        self.conf = np.ones(n)

        # R1 spine starts at downswing (frame 45)
        for i in range(45, n):
            self.spine[i] = 12.0

        # R2 hip starts well after R1 (frame 55) — reversed timing
        for i in range(55, n):
            self.hip[i] = 0.08

    def test_both_rules_triggered(self):
        r1 = r1_loss_of_posture(self.spine, self.phases, self.conf)
        r2 = r2_hip_toward_ball(self.hip, self.phases, self.conf)
        assert r1 is not None
        assert r2 is not None

    def test_r1_onset_before_r2(self):
        r1 = r1_loss_of_posture(self.spine, self.phases, self.conf)
        r2 = r2_hip_toward_ball(self.hip, self.phases, self.conf)
        assert r1.onset_frame < r2.onset_frame, "Spine should precede hip in this test"

    def test_not_early_extension(self):
        """With timing reversed, root cause must NOT be early_extension."""
        r1 = r1_loss_of_posture(self.spine, self.phases, self.conf)
        r2 = r2_hip_toward_ball(self.hip, self.phases, self.conf)
        rc = engine.analyze([r1, r2])
        assert rc.root_cause != "early_extension", (
            f"Reversed timing should not produce EE, got {rc.root_cause}"
        )

    def test_independent_faults_reported(self):
        r1 = r1_loss_of_posture(self.spine, self.phases, self.conf)
        r2 = r2_hip_toward_ball(self.hip, self.phases, self.conf)
        rc = engine.analyze([r1, r2])
        # Both faults should surface
        all_faults = rc.supporting_evidence + rc.independent_faults
        fault_types = [f.fault_type for f in all_faults]
        assert "loss_of_posture" in fault_types
        assert "hip_toward_ball" in fault_types


# ---------------------------------------------------------------------------
# T5: Noise frames — bone sentinel excludes them, rule NOT triggered
# ---------------------------------------------------------------------------

class TestT5NoiseFrames:
    """
    T5: Two spurious frames in downswing where bone length spikes and
        spine_delta spikes to 40°. Sentinel should exclude them, so
        the remaining <3 consecutive clean frames don't trigger R1.
    """

    def setup_method(self):
        self.phases = make_phases()
        n = frame_count(self.phases)
        self.spine = np.zeros(n)
        self.hip = np.zeros(n)
        self.conf = np.ones(n)

        # Only frames 45-46 (2 noise frames) have high spine delta
        # downswing starts at 45
        self.spine[45] = 40.0
        self.spine[46] = 40.0
        # All other frames are fine

        # Bone length ratios: those two frames have 50% deviation
        bone_ratios = {"torso": np.ones(n)}
        bone_ratios["torso"][45] = 1.55   # +55% deviation -> unreliable
        bone_ratios["torso"][46] = 1.60   # +60% deviation -> unreliable
        self.bone_ratios = bone_ratios

    def test_sentinel_flags_noise_frames(self):
        unreliable = bone_length_sentinel(self.bone_ratios, change_threshold=0.20)
        assert unreliable[45], "Frame 45 should be flagged"
        assert unreliable[46], "Frame 46 should be flagged"

    def test_sentinel_does_not_flag_normal_frames(self):
        unreliable = bone_length_sentinel(self.bone_ratios, change_threshold=0.20)
        assert not np.any(unreliable[:45]), "Normal frames should not be flagged"

    def test_r1_not_triggered_after_sentinel(self):
        """With only 2 noisy frames excluded, fewer than 3 consecutive valid
        frames exceed threshold -> R1 should NOT trigger."""
        unreliable = bone_length_sentinel(self.bone_ratios, change_threshold=0.20)
        r1 = r1_loss_of_posture(
            self.spine, self.phases, self.conf,
            unreliable_mask=unreliable,
        )
        assert r1 is None, "R1 should not trigger when only noise frames exceed threshold"


# ---------------------------------------------------------------------------
# T6: Low joint confidence — rule should be skipped
# ---------------------------------------------------------------------------

class TestT6LowConfidence:
    """
    T6: Spine delta genuinely exceeds threshold, but joint confidence < 0.4.
        Rule should not output a fault.
    """

    def setup_method(self):
        self.phases = make_phases()
        n = frame_count(self.phases)
        self.spine = np.zeros(n)
        self.hip = np.zeros(n)

        # Genuine R1 condition in downswing
        for i in range(45, n):
            self.spine[i] = 15.0

        # Low confidence throughout the entire R2+R1 active window
        # R2 window starts at transition (frame 35), R1 at downswing (frame 45)
        # Set low confidence from frame 35 onward to cover both windows fully
        self.conf_low = np.ones(n) * 0.9
        self.conf_low[35:] = 0.3   # below 0.4 threshold, covers transition+downswing+impact

    def test_r1_skipped_low_confidence(self):
        r1 = r1_loss_of_posture(self.spine, self.phases, self.conf_low)
        assert r1 is None, "R1 should be skipped when joint confidence < 0.4"

    def test_r2_skipped_low_confidence(self):
        for i in range(35, len(self.hip)):
            self.hip[i] = 0.08
        r2 = r2_hip_toward_ball(self.hip, self.phases, self.conf_low)
        assert r2 is None, "R2 should be skipped when joint confidence < 0.4"


# ---------------------------------------------------------------------------
# T7: Boundary — delta exactly 8.0° should trigger (>= threshold)
# ---------------------------------------------------------------------------

class TestT7BoundaryThreshold:
    """
    T7: spine_delta exactly = 8.0° for >=3 consecutive frames in downswing.
        Should trigger R1 at boundary (>= operator), severity = mild.
    """

    def setup_method(self):
        self.phases = make_phases()
        n = frame_count(self.phases)
        self.spine = np.zeros(n)
        self.conf = np.ones(n)

        # Exactly 3 frames at exactly 8.0° in downswing (frames 45-47)
        self.spine[45] = 8.0
        self.spine[46] = 8.0
        self.spine[47] = 8.0

    def test_r1_triggered_at_boundary(self):
        r1 = r1_loss_of_posture(self.spine, self.phases, self.conf)
        assert r1 is not None, "R1 should trigger at exactly 8.0° (boundary inclusive)"

    def test_severity_is_mild_at_boundary(self):
        r1 = r1_loss_of_posture(self.spine, self.phases, self.conf)
        assert r1.severity == "mild", f"Expected mild, got {r1.severity}"

    def test_certainty_cap_at_likely(self):
        """
        Even if we construct more evidence, certainty must not exceed 'likely'
        with just 2 evidences (spec rule: >=4 evidences needed for Confirmed).
        """
        self.hip = np.zeros(len(self.spine))
        # R2: add hip displacement starting at transition (frame 35)
        for i in range(35, len(self.hip)):
            self.hip[i] = 0.08
        conf = np.ones(len(self.spine))
        phases = self.phases

        r1 = r1_loss_of_posture(self.spine, phases, conf)
        r2 = r2_hip_toward_ball(self.hip, phases, conf)
        assert r1 is not None and r2 is not None

        rc = engine.analyze([r1, r2])
        assert rc.certainty in ("possible", "likely"), (
            f"Certainty must not exceed 'likely' with 2 evidences, got {rc.certainty}"
        )
        assert rc.certainty != "confirmed", (
            "Confirmed requires >=4 evidences (step 2), should not be reached here"
        )
