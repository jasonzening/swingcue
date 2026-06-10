"""
E-layer Root Cause Engine — minimal version (one causal chain).

Takes list of FaultDetection objects from D-layer, applies the Early Extension
root cause graph, and returns a structured diagnosis result.

Certainty caps (from spec):
  - 1 evidence  -> Possible
  - 2 evidences + timing ok -> Likely  (max in first version)
  - 4+ evidences  -> Confirmed  (only after step-2 feature enrichment)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .rules import FaultDetection


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class RootCauseResult:
    """Structured diagnosis from E-layer root cause engine."""
    root_cause: str                          # e.g. "early_extension", "loss_of_posture", "none"
    certainty: str                           # "possible" | "likely" | "confirmed"
    supporting_evidence: list[FaultDetection] = field(default_factory=list)
    causal_chain: list[str] = field(default_factory=list)
    independent_faults: list[FaultDetection] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "root_cause": self.root_cause,
            "certainty": self.certainty,
            "causal_chain": self.causal_chain,
            "supporting_evidence": [f.to_dict() for f in self.supporting_evidence],
            "independent_faults": [f.to_dict() for f in self.independent_faults],
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# Root Cause Engine
# ---------------------------------------------------------------------------

class RootCauseEngine:
    """
    E-layer root cause engine, first closed loop.

    Parameters
    ----------
    causal_timing_tolerance : int
        Frame tolerance for R2 onset <= R1 onset check.
        (spec default: 3 frames at 30 fps ≈ 0.1 s)
    min_confidence_for_likely : float
        Both fault confidences must exceed this for "likely" certainty.
    """

    def __init__(
        self,
        causal_timing_tolerance: int = 3,
        min_confidence_for_likely: float = 0.6,
    ):
        self.causal_timing_tolerance = causal_timing_tolerance
        self.min_confidence_for_likely = min_confidence_for_likely

    # ------------------------------------------------------------------

    def analyze(self, faults: list[FaultDetection]) -> RootCauseResult:
        """
        Perform root cause analysis on a list of D-layer fault detections.

        Causal chain for early_extension (first closed loop):
          R2 (hip_toward_ball) -> R1 (loss_of_posture)
          R2.onset_frame must be <= R1.onset_frame + tolerance

        Returns
        -------
        RootCauseResult
        """
        r1 = _find_fault(faults, "loss_of_posture")
        r2 = _find_fault(faults, "hip_toward_ball")
        other_faults = [
            f for f in faults
            if f.fault_type not in ("loss_of_posture", "hip_toward_ball")
        ]

        # Case 1: Both R1 and R2 present
        if r1 is not None and r2 is not None:
            return self._handle_both(r1, r2, other_faults)

        # Case 2: Only R1 (pure loss of posture)
        if r1 is not None:
            return self._handle_r1_only(r1, other_faults)

        # Case 3: Only R2
        if r2 is not None:
            return self._handle_r2_only(r2, other_faults)

        # Case 4: No faults
        return RootCauseResult(
            root_cause="none",
            certainty="none",
            note="no_fault_detected",
        )

    # ------------------------------------------------------------------
    # Internal case handlers
    # ------------------------------------------------------------------

    def _handle_both(
        self,
        r1: FaultDetection,
        r2: FaultDetection,
        other_faults: list[FaultDetection],
    ) -> RootCauseResult:
        """R1 + R2 both present: check causal timing."""
        r2_onset = r2.onset_frame if r2.onset_frame is not None else 0
        r1_onset = r1.onset_frame if r1.onset_frame is not None else 0

        timing_ok = r2_onset <= r1_onset + self.causal_timing_tolerance

        if not timing_ok:
            # Timing reversed: spine straightened well before hip moved —
            # do NOT force early_extension attribution, report independently.
            return RootCauseResult(
                root_cause="independent",
                certainty="possible",
                supporting_evidence=[r1, r2],
                independent_faults=[r1, r2] + other_faults,
                causal_chain=[],
                note="timing_reversed_no_ee_attribution",
            )

        # Timing ok: Early Extension diagnosis
        # Certainty capped at "likely" with 2 evidences (spec rule)
        both_confident = (
            r1.confidence >= self.min_confidence_for_likely
            and r2.confidence >= self.min_confidence_for_likely
        )
        certainty = "likely" if both_confident else "possible"

        # With more evidence (future step 2), certainty could reach "confirmed"
        # but with only 2 it's always capped at "likely"
        evidence_count = 2 + len(other_faults)
        if evidence_count >= 4 and certainty == "likely":
            # Spec: >=4 evidences -> Confirmed. But first version cannot reach it.
            # Leave at "likely" until knee/head features are added.
            pass  # certainty stays "likely" in first version

        return RootCauseResult(
            root_cause="early_extension",
            certainty=certainty,
            supporting_evidence=[r2, r1],   # causal order: hip first
            causal_chain=["hip_toward_ball", "loss_of_posture"],
            independent_faults=other_faults,
            note="r2_causes_r1",
        )

    def _handle_r1_only(
        self,
        r1: FaultDetection,
        other_faults: list[FaultDetection],
    ) -> RootCauseResult:
        """Only R1 (spine straightening without hip thrust)."""
        # Certainty based on severity: significant -> likely, mild -> possible
        certainty = "likely" if r1.severity == "significant" else "possible"
        return RootCauseResult(
            root_cause="loss_of_posture",
            certainty=certainty,
            supporting_evidence=[r1],
            causal_chain=["loss_of_posture"],
            independent_faults=other_faults,
            note="standalone_posture_loss",
        )

    def _handle_r2_only(
        self,
        r2: FaultDetection,
        other_faults: list[FaultDetection],
    ) -> RootCauseResult:
        """Only R2 (hip thrust without observed posture loss yet).

        Spec: mild hip-only -> below report threshold, do not output.
        """
        if r2.severity == "mild":
            return RootCauseResult(
                root_cause="none",
                certainty="none",
                note="r2_mild_only_below_report_threshold",
            )
        # Significant hip displacement without R1 -> possible early extension tendency
        return RootCauseResult(
            root_cause="early_extension",
            certainty="possible",
            supporting_evidence=[r2],
            causal_chain=["hip_toward_ball"],
            independent_faults=other_faults,
            note="r2_only_possible_ee_tendency",
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _find_fault(
    faults: list[FaultDetection],
    fault_type: str,
) -> Optional[FaultDetection]:
    """Return first fault matching fault_type, or None."""
    for f in faults:
        if f.fault_type == fault_type:
            return f
    return None
