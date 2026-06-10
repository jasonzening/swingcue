"""
F-layer Coaching Output Engine.

Converts RootCauseResult into:
  - diagnosis_json: machine-readable structured dict
  - one_liner: human-readable Chinese coaching sentence
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .root_cause import RootCauseResult


# ---------------------------------------------------------------------------
# Output templates (from spec Section 3)
# ---------------------------------------------------------------------------

_TEMPLATES: dict[tuple[str, str], str] = {
    ("early_extension", "likely"): (
        "你很可能出现了 Early Extension:"
        "髋部过早向球方向顶,导致身体失去前倾角(起身)。"
        "优先修正:保持髋部在后方,不要向球顶。"
    ),
    ("early_extension", "possible"): (
        "可能有 Early Extension 的倾向:"
        "髋部向球方向移动,需留意身体前倾角的保持。"
        "优先修正:保持髋部在后方,不要向球顶。"
    ),
    ("loss_of_posture", "likely"): (
        "下杆时身体有明显起身,前倾角没保持住。"
        "试着保持 address 时的前倾,头部高度不变。"
    ),
    ("loss_of_posture", "possible"): (
        "下杆时身体有些起身,前倾角没保持住。"
        "试着保持 address 时的前倾,头部高度不变。"
    ),
    ("independent", "possible"): (
        "检测到起身和髋部前移,但两者时序不典型。"
        "建议关注前倾角保持与髋部稳定。"
    ),
    ("none", "none"): (
        "这一杆姿势保持得不错,没有检出明显问题。"
    ),
}

_UNRELIABLE_MSG = (
    "测量质量不足,无法给出可靠诊断。请确保拍摄角度为侧面(DTL),光线充足,全身可见。"
)


# ---------------------------------------------------------------------------
# CoachingOutput
# ---------------------------------------------------------------------------

@dataclass
class CoachingOutputResult:
    diagnosis_json: dict
    one_liner: str

    def to_dict(self) -> dict:
        return {
            "diagnosis_json": self.diagnosis_json,
            "one_liner": self.one_liner,
        }

    def __str__(self) -> str:
        return self.one_liner


class CoachingOutput:
    """
    F-layer output engine.

    Takes a RootCauseResult and produces structured + human-readable output.
    """

    def generate(
        self,
        result: RootCauseResult,
        unreliable_frame_ratio: float = 0.0,
        unreliable_threshold: float = 0.5,
    ) -> CoachingOutputResult:
        """
        Parameters
        ----------
        result : RootCauseResult
            Output from RootCauseEngine.analyze().
        unreliable_frame_ratio : float
            Fraction of frames flagged by bone_length_sentinel.
            If >= unreliable_threshold, return failure message.
        unreliable_threshold : float
            Ratio above which measurement quality is considered too low.

        Returns
        -------
        CoachingOutputResult
        """
        # Check measurement quality gate
        if unreliable_frame_ratio >= unreliable_threshold:
            return CoachingOutputResult(
                diagnosis_json={
                    "status": "measurement_unreliable",
                    "unreliable_frame_ratio": round(unreliable_frame_ratio, 3),
                    "root_cause": None,
                    "certainty": None,
                },
                one_liner=_UNRELIABLE_MSG,
            )

        key = (result.root_cause, result.certainty)
        one_liner = _TEMPLATES.get(key, self._fallback(result))

        diagnosis_json = {
            "status": "ok",
            "root_cause": result.root_cause,
            "certainty": result.certainty,
            "causal_chain": result.causal_chain,
            "supporting_evidence": [e.to_dict() for e in result.supporting_evidence],
            "independent_faults": [f.to_dict() for f in result.independent_faults],
            "note": result.note,
            "one_liner": one_liner,
        }

        return CoachingOutputResult(
            diagnosis_json=diagnosis_json,
            one_liner=one_liner,
        )

    @staticmethod
    def _fallback(result: RootCauseResult) -> str:
        """Fallback message for unexpected diagnosis combinations."""
        if result.root_cause == "none":
            return _TEMPLATES[("none", "none")]
        return (
            f"检测到 {result.root_cause}({result.certainty})。"
            "建议关注相关动作细节。"
        )
