"""
engine/layer0/perception_gate.py
=================================
Layer 0 Perception Gate — VLM-based video quality and content check.

Usage
-----
  from engine.layer0.perception_gate import PerceptionGate, GateResult

  gate = PerceptionGate()
  result = gate.check(video_path, angle_hint="auto")  # calls VLM
  gate.save(result)          # persists to records/
  gate.assert_pass(stem)     # raises RuntimeError if no PASS record

Hard gate in run_pipeline.py:
  PerceptionGate().assert_pass(video_stem)  # call before any other processing
"""

from __future__ import annotations
import json, math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJ         = Path(__file__).resolve().parents[2]
RECORDS_DIR  = PROJ / "engine/layer0/records"
RECORDS_DIR.mkdir(parents=True, exist_ok=True)

PASS_THRESHOLD = 4   # frames out of 5 that must pass hard criteria
N_SAMPLE       = 5   # uniform sample frames per video


@dataclass
class FrameResult:
    frame_idx:   int
    q1_golf:     bool        # real human doing/preparing golf swing
    q2_persons:  int         # number of people visible
    q3_angle:    str         # "DTL" | "face-on" | "other"
    q4_fullbody: bool        # full body visible (head to feet)
    q5_desc:     str         # one-sentence scene description
    hard_pass:   bool = False  # computed: q1 AND q2==1 AND q4

    def __post_init__(self):
        self.hard_pass = self.q1_golf and self.q2_persons == 1 and self.q4_fullbody


@dataclass
class GateResult:
    video_stem:  str
    verdict:     str         # "PASS" | "REJECT" | "needs_human"
    angle:       str         # "DTL" | "face-on" | "other" | "inconsistent"
    reason:      str
    frames:      list[FrameResult] = field(default_factory=list)
    sh_ratio:    Optional[float]   = None   # shoulder-width ratio cross-check
    sh_angle:    Optional[str]     = None   # angle from shoulder-ratio method
    angle_conflict: bool           = False  # VLM angle vs sh_ratio conflict

    def to_dict(self) -> dict:
        return {
            "video_stem":    self.video_stem,
            "verdict":       self.verdict,
            "angle":         self.angle,
            "reason":        self.reason,
            "sh_ratio":      self.sh_ratio,
            "sh_angle":      self.sh_angle,
            "angle_conflict": self.angle_conflict,
            "frames": [
                {
                    "frame_idx":   f.frame_idx,
                    "q1_golf":     f.q1_golf,
                    "q2_persons":  f.q2_persons,
                    "q3_angle":    f.q3_angle,
                    "q4_fullbody": f.q4_fullbody,
                    "q5_desc":     f.q5_desc,
                    "hard_pass":   f.hard_pass,
                }
                for f in self.frames
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GateResult":
        frames = [
            FrameResult(
                frame_idx=f["frame_idx"], q1_golf=f["q1_golf"],
                q2_persons=f["q2_persons"], q3_angle=f["q3_angle"],
                q4_fullbody=f["q4_fullbody"], q5_desc=f["q5_desc"],
            )
            for f in d.get("frames", [])
        ]
        return cls(
            video_stem=d["video_stem"], verdict=d["verdict"],
            angle=d["angle"], reason=d["reason"],
            frames=frames,
            sh_ratio=d.get("sh_ratio"),
            sh_angle=d.get("sh_angle"),
            angle_conflict=d.get("angle_conflict", False),
        )


class PerceptionGate:
    """
    Layer 0 gate.  Two modes:
      1. check(video_path)   — extract frames + call VLM + compute verdict
      2. ingest(result_dict) — record a pre-computed VLM result (from delegation)
      3. assert_pass(stem)   — raise RuntimeError if no PASS record on disk
    """

    def __init__(
        self,
        sh_ratio_faceon_thr: float = 0.35,
        sh_ratio_dtl_thr:    float = 0.20,
    ):
        self.sh_ratio_faceon_thr = sh_ratio_faceon_thr
        self.sh_ratio_dtl_thr    = sh_ratio_dtl_thr

    # ── public API ───────────────────────────────────────────────────────────

    def ingest(self, stem: str, vlm_result: dict,
               sh_ratio: Optional[float] = None,
               sh_angle: Optional[str]   = None) -> GateResult:
        """
        Build a GateResult from pre-computed VLM output (dict with 'frames',
        'verdict', 'angle', 'reason') and save it to disk.

        Parameters
        ----------
        stem       : video stem (filename without .mp4)
        vlm_result : dict from the VLM analysis (frame-level + per-video)
        sh_ratio   : shoulder x-span / torso_h from RTMPose (optional cross-check)
        sh_angle   : angle from shoulder-ratio method (optional cross-check)
        """
        frames = [
            FrameResult(
                frame_idx=f["fr"],
                q1_golf=f["q1_golf"],
                q2_persons=f["q2_persons"],
                q3_angle=f["q3_angle"],
                q4_fullbody=f["q4_fullbody"],
                q5_desc=f["q5_desc"],
            )
            for f in vlm_result.get("frames", [])
        ]

        raw_verdict = vlm_result.get("verdict", "REJECT")
        vlm_angle   = vlm_result.get("angle", "other")
        reason      = vlm_result.get("reason", "")

        # Cross-check angle if sh_ratio available
        angle_conflict = False
        if sh_ratio is not None and sh_angle is not None:
            # If VLM and ratio disagree on DTL vs face-on → flag conflict
            vlm_a   = vlm_angle if vlm_angle in ("DTL", "face-on") else None
            ratio_a = sh_angle  if sh_angle  in ("DTL", "face-on") else None
            if vlm_a and ratio_a and vlm_a != ratio_a:
                angle_conflict = True
                if raw_verdict == "PASS":
                    raw_verdict = "needs_human"
                    reason += " | VLM angle vs shoulder-ratio conflict → needs_human"

        # For multi-angle videos (like test-dwontheline with mixed angles),
        # set final angle to VLM angle
        result = GateResult(
            video_stem=stem, verdict=raw_verdict, angle=vlm_angle,
            reason=reason, frames=frames,
            sh_ratio=sh_ratio, sh_angle=sh_angle,
            angle_conflict=angle_conflict,
        )
        self.save(result)
        return result

    def save(self, result: GateResult) -> None:
        path = RECORDS_DIR / f"{result.video_stem}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    def load(self, stem: str) -> Optional[GateResult]:
        path = RECORDS_DIR / f"{stem}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return GateResult.from_dict(json.load(f))

    def assert_pass(self, stem: str) -> GateResult:
        """
        Raise RuntimeError if no PASS record exists for this video.
        Called as a hard gate in run_pipeline.py before any processing.
        """
        rec = self.load(stem)
        if rec is None:
            raise RuntimeError(
                f"Layer 0 gate: no record for '{stem}'. "
                "Run PerceptionGate analysis first."
            )
        if rec.verdict != "PASS":
            raise RuntimeError(
                f"Layer 0 gate: '{stem}' verdict={rec.verdict!r} — {rec.reason}"
            )
        return rec

    def summary_table(self, stems: list[str]) -> str:
        """Return a markdown table of results for a list of stems."""
        rows = ["| Video | Verdict | Angle | Reason |",
                "|---|---|---|---|"]
        for stem in stems:
            rec = self.load(stem)
            if rec is None:
                rows.append(f"| {stem} | NO_RECORD | — | no gate record on disk |")
            else:
                reason_short = rec.reason[:80] + "..." if len(rec.reason) > 80 else rec.reason
                rows.append(f"| {stem} | {rec.verdict} | {rec.angle} | {reason_short} |")
        return "\n".join(rows)
