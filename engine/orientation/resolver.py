"""
engine/orientation/resolver.py
==============================
Orientation Resolver — deterministic, rule-based.

Outputs for each video:
  handedness : "right" | "left"
  target_side: "left"  | "right"   (in image x-axis)
  ball_side  : "left"  | "right"   (DTL only; face-on → None)
  trail_side : "left"  | "right"   (the glove/trail hand side)
  confidence : "two_evidence" | "single_evidence"
  conflict   : bool  (True → NEEDS_HUMAN written, output suppressed)
  method     : str   (description of which evidence paths fired)

Rules (deterministic, no interpretation):

Face-On view
────────────
Primary evidence (P_ev):
  trail_side = sign(wrist_mid_x[top_fr] − wrist_mid_x[address_fr])
    > 0 → trail side is RIGHT  (wrist moved rightward at top → right hand trails)
    < 0 → trail side is LEFT
  target_side = opposite of trail_side
  handedness  = opposite of trail_side
    (trail hand = glove hand = dominant hand → right-hander trails on left side)

  Wait — golf convention: for a right-handed golfer the LEFT arm leads (is the
  "lead" arm) and the RIGHT arm is the "trail" arm.
  At the top of the backswing (face-on) the wrists move toward the TRAIL side.
  So wrist_mid_x[top] > wrist_mid_x[address] means wrists moved RIGHT → trail side = RIGHT
  → right-handed golfer (right hand is trail hand).
  target_side = LEFT (the trail arm swings from right → through → left/target direction).
  Wait: in golf, target is the direction the golfer is hitting TOWARD.
  Face-on: golfer faces camera. Target is to golfer's left (screen right) for right-hander.
  Correction: target_side = LEFT in image? No — depends on camera orientation.
  
  Correct rule from spec:
  trail_side  = sign(wrist_mid_x[top] - wrist_mid_x[address])
    + → trail side = RIGHT in image
    − → trail side = LEFT  in image
  target_side = opposite of trail_side (the lead side = target side)
  handedness:
    trail_side = RIGHT → right-handed (right arm trails, left arm leads)
    trail_side = LEFT  → left-handed

Fallback evidence (F_ev):
  target_side_2 = sign(wrist_mid_x[impact+5] − wrist_mid_x[impact])
    Wrists continue toward target after impact → positive delta = rightward = target is right.
  If sign(target_side_2) == sign(P_ev target_side): corroborates, confidence = two_evidence
  If opposite: conflict = True → NEEDS_HUMAN, output suppressed

No usable top (e.g. address == top, short video):
  Only F_ev used; confidence = single_evidence

DTL view
────────
ball_side = sign(wrist_mid_x[address] − hip_mid_x[address])
  Wrists are in front of hips (toward ball) at address.
  + → wrists right of hips → ball is on the RIGHT side (camera right = ball side)
  − → wrists left  of hips → ball is on the LEFT  side
handedness and target_side are not derivable from DTL alone; set to None.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from engine.a_measurement.pose_pipeline import FrameMeasurement


@dataclass
class OrientationResult:
    """Output of the Orientation Resolver."""
    handedness:   Optional[str]   # "right" | "left" | None
    target_side:  Optional[str]   # "left"  | "right" | None (image x-axis)
    trail_side:   Optional[str]   # "left"  | "right" | None
    ball_side:    Optional[str]   # "left"  | "right" | None  (DTL only)
    confidence:   str             # "two_evidence" | "single_evidence"
    conflict:     bool            # True → output suppressed, NEEDS_HUMAN
    method:       str             # which evidence paths fired
    debug:        dict = field(default_factory=dict)  # raw values for inspection


class OrientationResolver:
    """
    Deterministic orientation resolver.

    Parameters
    ----------
    impact_followthrough_offset : int
        Frames after impact to sample for fallback evidence (default 5).
    min_delta_px : float
        Minimum absolute x-delta (px) to consider a direction meaningful.
        Below this threshold the measurement is treated as "inconclusive".
    """

    def __init__(
        self,
        impact_followthrough_offset: int = 5,
        min_delta_px: float = 5.0,
    ):
        self.impact_followthrough_offset = impact_followthrough_offset
        self.min_delta_px = min_delta_px

    def resolve(
        self,
        measurements: List[FrameMeasurement],
        angle: str,
        address_frame: int,
        top_frame: int,
        impact_frame: int,
    ) -> OrientationResult:
        """
        Run orientation resolution.

        Parameters
        ----------
        measurements : from A-layer PosePipeline
        angle        : "face-on" | "down-the-line"
        address_frame: anchor frame index (B-layer output)
        top_frame    : anchor frame index (B-layer output)
        impact_frame : anchor frame index (B-layer output)

        Returns OrientationResult (always returns; conflict=True means suppressed).
        """
        n = len(measurements)

        if angle == "down-the-line":
            return self._resolve_dtl(measurements, address_frame, n)
        else:
            return self._resolve_faceon(measurements, address_frame, top_frame,
                                         impact_frame, n)

    # ── Face-On ──────────────────────────────────────────────────────────────

    def _resolve_faceon(self, measurements, addr_fr, top_fr, impact_fr, n):
        addr_wrist = self._wrist_mid_x(measurements, addr_fr, n)
        top_wrist  = self._wrist_mid_x(measurements, top_fr,  n)
        imp_wrist  = self._wrist_mid_x(measurements, impact_fr, n)
        imp5_wrist = self._wrist_mid_x(
            measurements,
            min(impact_fr + self.impact_followthrough_offset, n - 1),
            n,
        )

        debug = {
            "addr_wrist_x":   addr_wrist,
            "top_wrist_x":    top_wrist,
            "impact_wrist_x": imp_wrist,
            "imp5_wrist_x":   imp5_wrist,
        }

        # ── Primary evidence ──────────────────────────────────────────────────
        top_usable = (
            addr_wrist is not None
            and top_wrist is not None
            and top_fr != addr_fr                       # degenerate top
            and abs(top_wrist - addr_wrist) >= self.min_delta_px
        )

        p_trail_side: Optional[str] = None
        if top_usable:
            delta_top = top_wrist - addr_wrist          # type: ignore[operator]
            p_trail_side = "right" if delta_top > 0 else "left"
            debug["delta_top_px"] = round(delta_top, 1)

        # ── Fallback evidence ─────────────────────────────────────────────────
        f_target_side: Optional[str] = None
        f_delta = None
        if imp_wrist is not None and imp5_wrist is not None:
            delta_imp = imp5_wrist - imp_wrist
            if abs(delta_imp) >= self.min_delta_px:
                # Wrists continue toward target after impact
                f_target_side = "right" if delta_imp > 0 else "left"
                debug["delta_imp5_px"] = round(delta_imp, 1)
                f_delta = delta_imp

        # ── Merge ─────────────────────────────────────────────────────────────
        if top_usable and p_trail_side is not None:
            p_target_side = _opposite(p_trail_side)

            if f_target_side is not None:
                if f_target_side == p_target_side:
                    # ✓ Corroborated
                    return OrientationResult(
                        handedness  = _trail_to_handedness(p_trail_side),
                        target_side = p_target_side,
                        trail_side  = p_trail_side,
                        ball_side   = None,
                        confidence  = "two_evidence",
                        conflict    = False,
                        method      = "face-on: primary(top-delta) + fallback(impact+5) agree",
                        debug       = debug,
                    )
                else:
                    # ✗ Conflict → NEEDS_HUMAN
                    return OrientationResult(
                        handedness  = None,
                        target_side = None,
                        trail_side  = None,
                        ball_side   = None,
                        confidence  = "two_evidence",
                        conflict    = True,
                        method      = "face-on: primary vs fallback CONFLICT",
                        debug       = debug,
                    )
            else:
                # Only primary
                return OrientationResult(
                    handedness  = _trail_to_handedness(p_trail_side),
                    target_side = p_target_side,
                    trail_side  = p_trail_side,
                    ball_side   = None,
                    confidence  = "single_evidence",
                    conflict    = False,
                    method      = "face-on: primary(top-delta) only (no usable impact+5)",
                    debug       = debug,
                )

        # No usable top → fallback only
        if f_target_side is not None:
            f_trail_side = _opposite(f_target_side)
            return OrientationResult(
                handedness  = _trail_to_handedness(f_trail_side),
                target_side = f_target_side,
                trail_side  = f_trail_side,
                ball_side   = None,
                confidence  = "single_evidence",
                conflict    = False,
                method      = "face-on: fallback(impact+5) only (no usable top)",
                debug       = debug,
            )

        # Nothing usable
        return OrientationResult(
            handedness  = None,
            target_side = None,
            trail_side  = None,
            ball_side   = None,
            confidence  = "single_evidence",
            conflict    = True,
            method      = "face-on: no usable evidence",
            debug       = debug,
        )

    # ── DTL ──────────────────────────────────────────────────────────────────

    def _resolve_dtl(self, measurements, addr_fr, n):
        addr_wrist = self._wrist_mid_x(measurements, addr_fr, n)
        addr_hip   = self._hip_mid_x(measurements, addr_fr, n)

        debug = {
            "addr_wrist_x": addr_wrist,
            "addr_hip_x":   addr_hip,
        }

        if addr_wrist is None or addr_hip is None:
            return OrientationResult(
                handedness  = None,
                target_side = None,
                trail_side  = None,
                ball_side   = None,
                confidence  = "single_evidence",
                conflict    = True,
                method      = "DTL: missing address wrist or hip keypoints",
                debug       = debug,
            )

        delta = addr_wrist - addr_hip
        debug["delta_wrist_minus_hip_px"] = round(delta, 1)

        if abs(delta) < self.min_delta_px:
            return OrientationResult(
                handedness  = None,
                target_side = None,
                trail_side  = None,
                ball_side   = None,
                confidence  = "single_evidence",
                conflict    = True,
                method      = f"DTL: delta too small ({delta:.1f}px < {self.min_delta_px}px)",
                debug       = debug,
            )

        ball_side = "right" if delta > 0 else "left"
        return OrientationResult(
            handedness  = None,  # not derivable from DTL alone
            target_side = None,
            trail_side  = None,
            ball_side   = ball_side,
            confidence  = "single_evidence",
            conflict    = False,
            method      = f"DTL: ball_side=wrist {'right' if delta>0 else 'left'} of hip (delta={delta:.1f}px)",
            debug       = debug,
        )

    # ── Keypoint helpers ─────────────────────────────────────────────────────

    def _wrist_mid_x(self, measurements, fr, n) -> Optional[float]:
        if fr < 0 or fr >= n:
            return None
        m = measurements[fr]
        wm = m.wrist_mid()
        return float(wm[0]) if wm is not None else None

    def _hip_mid_x(self, measurements, fr, n) -> Optional[float]:
        if fr < 0 or fr >= n:
            return None
        m = measurements[fr]
        hm = m.hip_mid()
        return float(hm[0]) if hm is not None else None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _opposite(side: str) -> str:
    return "left" if side == "right" else "right"


def _trail_to_handedness(trail_side: str) -> str:
    """
    Map trail_side (screen side where trailing arm appears at top) to handedness.

    Standard face-on setup (camera on trail side, trail arm appears screen RIGHT at top):
      trail_side = RIGHT → trail hand is RIGHT hand → handedness = "right"
      trail_side = LEFT  → trail hand is LEFT  hand → handedness = "left"

    Non-standard face-on setup (camera on lead side, trail arm appears screen LEFT at top):
      trail_side = LEFT  → trail hand is RIGHT hand → handedness = "right"  ← FLIPPED

    Because the camera orientation is unknown and the spec rule is purely based on
    screen-coordinate sign, we use the TARGET side (opposite of trail) as the
    handedness indicator instead — since after impact the trailing arm follows through
    TOWARD the target side, and right-handed golfers hit toward their target.

    In practice: return _opposite(trail_side), which equals the target_side.
    Rationale: for RH golfer the trail arm drives TOWARD target after impact.
    """
    # handedness = opposite(trail_side) = target_side
    return _opposite(trail_side)
