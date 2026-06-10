"""
engine/b_phase/swing_phase.py
B Layer — Swing Phase Engine  (v3.3 — speed-gated impact, ordering guard)

Key change v3.3 vs v3.2:
  _detect_anchors now passes real_impacts (speed-filtered list from _detect_swing_count)
  to _detect_impact, which uses real_impacts[0] directly when it falls within the
  expected swing window [address+2, n_eff].

  This fixes the 201015 bug:
    - real_impacts[0] = fr59 (high-speed, correctly identified by swing counter)
    - Previous code searched only [top+2, n_eff] for impact; on 201015 top=fr68
      (the follow-through high), so the search window [70, 133] skipped fr59 entirely,
      fell back to argmax of plateau, and landed on fr132 (wrong).

  Ordering guard:
    After all anchors are found, validate address < top < impact < finish.
    If impact < top (ordering violation), swap or fallback to real_impacts[0].

Multi-swing detection algorithm:
  1. Find wrist-Y maxima with prominence≥20px, min_dist=1.5s
  2. Speed filter: max wrist speed in 20fr before peak >= max(p60(speed), 25px/fr)
  3. real_impacts = speed-filtered peaks, sorted chronologically
  4. swing_count = len(real_impacts); first_swing_end = midpoint[swing1,swing2]
  5. All anchor detection runs in [0, first_swing_end] only

Confidence formula (3-factor):
  sig_score (50%): peak_prominence / (wrist_Y_range × 0.25)
  ambiguity (30%): 1 − 0.25×(swing_count−1), min 0.2
  joint_qual(20%): mean RTMPose quality ±3fr around impact

GT rule: frame numbers are ESTIMATES. GT only from human annotation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from scipy.signal import savgol_filter, find_peaks

from engine.a_measurement.pose_pipeline import FrameMeasurement

PHASE_NAMES = [
    "address", "takeaway", "backswing", "top",
    "transition", "downswing", "impact", "follow_through"
]

IMPACT_WINDOW = 3   # ± frames around impact keyframe → "impact" phase


@dataclass
class PhaseAnnotation:
    frame_idx: int
    phase: str
    phase_confidence: float


@dataclass
class AnchorFrames:
    address: int
    top: int
    impact: int
    finish: int
    impact_conf: float = 0.0
    top_conf: float = 0.0
    swing_count: int = 1
    first_swing_end: int = -1


class SwingPhaseEngine:

    def __init__(self, sg_window_ms: int = 200):
        self.sg_window_ms = sg_window_ms
        self.swing_count: int = 1
        self.first_swing_end: int = -1

    def run(self, measurements: List[FrameMeasurement], fps: float,
            angle: str = "auto") -> tuple[List[PhaseAnnotation], AnchorFrames]:
        n = len(measurements)
        xs, ys = self._extract_wrist(measurements, n)
        anchors = self._detect_anchors(measurements, xs, ys, fps, angle)
        fse = anchors.first_swing_end if anchors.first_swing_end >= 0 else n
        annotations = self._assign_phases(n, anchors, measurements, fse)
        return annotations, anchors

    # ── wrist track ──────────────────────────────────────────────────────────
    def _extract_wrist(self, measurements, n):
        xs = np.full(n, np.nan); ys = np.full(n, np.nan)
        for m in measurements:
            fi = m.frame_idx
            wm = m.wrist_mid()
            if wm is not None:
                xs[fi], ys[fi] = wm
        idx = np.arange(n)
        for arr in (xs, ys):
            nans = np.isnan(arr)
            if not nans.all():
                arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
        return xs, ys

    def _sg(self, arr, fps):
        w = max(7, int(fps * self.sg_window_ms / 1000)) | 1
        return savgol_filter(arr, w, 3)

    # ── multi-swing detection ─────────────────────────────────────────────────
    def _detect_swing_count(self, ys_s: np.ndarray, spd: np.ndarray,
                             fps: float, n: int) -> tuple[int, list[int], int]:
        """
        Detect real swings using wrist-Y maxima + speed filter.

        A candidate peak is a real swing impact if max wrist speed in the
        20 frames before the peak exceeds speed_filter_thr.
        This discriminates real swings (spd ~60-70 px/fr) from rest plateaus
        (spd ~10-15 px/fr) which have similar or higher wrist-Y.

        Returns (swing_count, sorted_real_impacts, first_swing_end)
        """
        min_dist = max(int(fps * 1.5), 10)
        peaks, _ = find_peaks(ys_s, prominence=20, distance=min_dist)
        valid = (peaks >= int(n * 0.05)) & (peaks <= int(n * 0.95))
        peaks = peaks[valid]

        if len(peaks) == 0:
            self.swing_count = 1; self.first_swing_end = n
            return 1, [], n

        # Speed filter: max speed in 20fr before peak
        speed_thr = max(float(np.percentile(spd, 60)), 25.0)
        real_impacts = []
        for pk in peaks:
            lo = max(0, int(pk) - 20)
            if spd[lo:int(pk)+1].max() >= speed_thr:
                real_impacts.append(int(pk))

        if len(real_impacts) == 0:
            # Fallback: keep original peaks list
            real_impacts = peaks.tolist()

        swing_count = len(real_impacts)

        if swing_count == 1:
            first_end = min(n - 1, real_impacts[0] + int(fps * 2.5))
        else:
            midpoint = (real_impacts[0] + real_impacts[1]) // 2
            first_end = min(midpoint, real_impacts[0] + int(fps * 2.5))

        self.swing_count = swing_count
        self.first_swing_end = int(first_end)
        return swing_count, real_impacts, int(first_end)

    # ── anchor detection ─────────────────────────────────────────────────────
    def _detect_anchors(self, measurements, xs, ys, fps, angle) -> AnchorFrames:
        n = len(xs)
        xs_s = self._sg(xs, fps)
        ys_s = self._sg(ys, fps)
        dx = np.diff(xs_s, prepend=xs_s[0])
        dy = np.diff(ys_s, prepend=ys_s[0])
        spd = self._sg(np.sqrt(dx**2 + dy**2), fps)

        # ── multi-swing boundary ──────────────────────────────────────────────
        swing_count, real_impacts, first_swing_end = self._detect_swing_count(
            ys_s, spd, fps, n)
        n_eff = max(first_swing_end, 30) if first_swing_end < n else n

        # ── address ──────────────────────────────────────────────────────────
        static_thr = max(np.percentile(spd[:n_eff], 20) * 3.0, 1.0)
        address = 2
        for i in range(2, int(n_eff * 0.50)):
            if spd[i] < static_thr:
                address = i
            elif spd[i] > static_thr * 3 and i > address + 5:
                break

        # ── top ──────────────────────────────────────────────────────────────
        top_end = int(n_eff * 0.82)
        ys_region = ys_s[address:top_end]
        peaks_l, pp = find_peaks(-ys_region, prominence=30, distance=int(fps * 0.25))
        if len(peaks_l) == 0:
            top = address + int(np.argmin(ys_region))
            left_h  = ys_region[0]  - ys_region.min()
            right_h = ys_region[-1] - ys_region.min()
            top_prom = float(min(left_h, right_h))
        else:
            top = address + peaks_l[0]
            top_prom = float(pp["prominences"][0])

        ys_range = float(ys_s[:n_eff].max() - ys_s[:n_eff].min())
        ys_range = max(ys_range, 30.0)
        top_conf = float(np.clip(top_prom / (ys_range * 0.70), 0.0, 1.0))

        # ── impact (with speed-gate and ordering guard) ───────────────────────
        # Pass real_impacts[0] as a hint — the speed-filtered first-swing peak.
        # _detect_impact will prefer this over a re-detected peak when it is
        # within the expected window and obeys ordering (address < hint < n_eff).
        swing1_hint = real_impacts[0] if real_impacts else None
        impact, impact_conf = self._detect_impact(
            xs_s, ys_s, spd, fps, angle, address, top, n_eff,
            measurements=measurements,
            swing_count=swing_count,
            ys_range=ys_range,
            swing1_hint=swing1_hint,
        )

        # ── ordering guard: address < top < impact < finish ──────────────────
        # If impact <= top (e.g. top is the follow-through high, not backswing top),
        # reset top to the most credible value before impact and try again.
        if impact <= top:
            # Top was misidentified (likely follow-through high).
            # Best estimate: search for wrist-Y minimum in [address, impact)
            pre_impact = ys_s[address:impact]
            if len(pre_impact) > 0:
                top = address + int(np.argmin(pre_impact))
                # Re-score top_conf with updated top_prom
                new_prom = float(ys_s[impact] - ys_s[top])  # rough
                top_conf = float(np.clip(new_prom / (ys_range * 0.70), 0.0, 1.0))

        # ── finish ───────────────────────────────────────────────────────────
        settle_thr = static_thr * 1.5
        settle_win = max(int(fps * 0.35), 4)
        ft_region = ys_s[impact:min(impact + int(fps * 2), n_eff)]
        ft_peaks, _ = find_peaks(-ft_region, prominence=15, distance=int(fps * 0.15))
        ft_top = (impact + ft_peaks[0]) if len(ft_peaks) > 0 else (impact + int(fps * 0.3))

        finish = min(n_eff - 1, ft_top + settle_win)
        found_settle = False
        for i in range(ft_top + 1, n_eff - settle_win):
            if np.all(spd[i:i + settle_win] < settle_thr):
                finish = i + settle_win // 2
                found_settle = True
                break
        if not found_settle:
            finish = (ft_top + int(np.argmin(spd[ft_top:n_eff]))) if ft_top < n_eff else n_eff - 1

        # Clamp finish after impact
        finish = max(finish, impact + 1)
        finish = min(finish, n_eff - 1)

        return AnchorFrames(
            address=int(address), top=int(top),
            impact=int(impact), finish=int(finish),
            impact_conf=round(impact_conf, 3),
            top_conf=round(top_conf, 3),
            swing_count=int(swing_count),
            first_swing_end=int(n_eff),
        )

    def _detect_impact(self, xs_s, ys_s, spd, fps, angle, address, top, n_eff,
                       measurements=None, swing_count: int = 1,
                       ys_range: float = 150.0,
                       swing1_hint: Optional[int] = None) -> tuple[int, float]:
        """
        Impact detection with speed-gate and swing1_hint.

        Strategy:
          1. If swing1_hint is in [address+2, n_eff) AND speed before it passes
             the speed filter → use it directly.  This covers the case where
             impact happens before top (fast start from address) and the
             standard [top+2, ...] search would miss it.
          2. Otherwise fall back to first-peak search in [top+2, n_eff].

        DTL:     first local MAX of wrist-X after top (hands most forward)
        Face-on: first local MAX of wrist-Y after top (hands lowest)
        """
        speed_thr = max(float(np.percentile(spd[:n_eff], 60)), 25.0)

        # ── Test hint first ───────────────────────────────────────────────────
        if swing1_hint is not None and address + 2 <= swing1_hint < n_eff:
            lo = max(0, swing1_hint - 20)
            hint_spd = float(spd[lo:swing1_hint+1].max())
            if hint_spd >= speed_thr:
                # Hint passes speed gate — use it
                return self._score_impact(
                    swing1_hint, xs_s, ys_s, spd, fps, angle, address, top,
                    n_eff, measurements, swing_count, ys_range,
                    method="hint"
                )

        # ── Standard search from top+2 ────────────────────────────────────────
        signal = xs_s if angle == "down-the-line" else ys_s
        prom_min = 10 if angle == "down-the-line" else 15

        search_start = top + 2
        search_end   = min(n_eff - 1, top + int(fps * 4.5))
        cap_region   = signal[search_start:search_end + 1]

        if len(cap_region) == 0:
            # No search window at all — fall back to hint or top+1
            fb = swing1_hint if swing1_hint else top + 1
            return self._score_impact(
                fb, xs_s, ys_s, spd, fps, angle, address, top,
                n_eff, measurements, swing_count, ys_range, method="fallback"
            )

        full_peaks, full_props = find_peaks(
            cap_region, prominence=prom_min,
            distance=max(int(fps * 0.1), 2),
        )

        if len(full_peaks) == 0:
            # Speed-gate the argmax fallback
            best_fr = search_start + int(np.argmax(cap_region))
            lo = max(0, best_fr - 20)
            if spd[lo:best_fr+1].max() < speed_thr and swing1_hint is not None:
                # argmax failed speed gate — prefer hint even if it's before top
                return self._score_impact(
                    swing1_hint, xs_s, ys_s, spd, fps, angle, address, top,
                    n_eff, measurements, swing_count, ys_range, method="hint_fallback"
                )
            impact = best_fr
        else:
            # First chronological peak — also speed-gate it
            best_fr = search_start + int(full_peaks[0])
            lo = max(0, best_fr - 20)
            if spd[lo:best_fr+1].max() < speed_thr and swing1_hint is not None:
                # Peak fails speed gate — prefer hint
                return self._score_impact(
                    swing1_hint, xs_s, ys_s, spd, fps, angle, address, top,
                    n_eff, measurements, swing_count, ys_range, method="hint_speedgate"
                )
            impact = best_fr

        return self._score_impact(
            impact, xs_s, ys_s, spd, fps, angle, address, top,
            n_eff, measurements, swing_count, ys_range, method="peak"
        )

    def _score_impact(self, impact: int, xs_s, ys_s, spd, fps, angle,
                      address, top, n_eff, measurements,
                      swing_count, ys_range, method: str = "peak") -> tuple[int, float]:
        """Compute confidence for a given impact frame."""
        signal = xs_s if angle == "down-the-line" else ys_s

        # Peak prominence: compare signal[impact] to nearby minimum
        win = max(int(fps * 0.5), 5)
        lo_w = max(0, impact - win); hi_w = min(n_eff, impact + win + 1)
        region = signal[lo_w:hi_w]
        peak_prom = float(signal[impact] - region.min()) if len(region) > 0 else 0.0

        # Factor 1: signal significance
        norm_base = max(ys_range * 0.25, 30.0)
        sig_score = float(np.clip(peak_prom / norm_base, 0.0, 1.0))

        # Factor 2: multi-swing ambiguity
        ambiguity = float(np.clip(1.0 - 0.25 * (swing_count - 1), 0.20, 1.0))

        # Factor 3: joint quality near impact
        quality = 1.0
        if measurements is not None:
            lo = max(0, impact - 3); hi = min(len(measurements), impact + 4)
            qscores = []
            for m in measurements[lo:hi]:
                q = (1.0 if m.measurement_quality == "ok"
                     else 0.5 if m.measurement_quality == "degraded"
                     else 0.1)
                qscores.append(q)
            quality = float(np.mean(qscores)) if qscores else 0.5

        # Penalise hint-fallback slightly (less certain than a clean peak)
        method_factor = 0.85 if "fallback" in method else 1.0

        impact_conf = float((sig_score * 0.50 + ambiguity * 0.30 + quality * 0.20)
                            * method_factor)
        return int(impact), round(impact_conf, 3)

    # ── phase assignment ─────────────────────────────────────────────────────
    def _assign_phases(self, n, anchors: AnchorFrames,
                       measurements: List[FrameMeasurement],
                       first_swing_end: int = -1) -> List[PhaseAnnotation]:
        if first_swing_end < 0:
            first_swing_end = n
        A = anchors.address; TOP = anchors.top
        IMP = anchors.impact; FIN = anchors.finish

        takeaway_end   = A + max(1, int((TOP - A) * 0.25))
        top_start      = TOP - 1
        top_end        = TOP + 2
        transition_end = TOP + max(3, int((IMP - TOP) * 0.15))
        impact_lo      = IMP - IMPACT_WINDOW
        impact_hi      = IMP + IMPACT_WINDOW

        # Guard: if top >= impact (ordering violation survived), collapse transition window
        if TOP >= IMP:
            top_start = max(0, IMP - 5); top_end = IMP - 1
            transition_end = IMP - 1

        annotations = []
        for fi in range(n):
            m = measurements[fi]
            q_factor = (1.0 if m.measurement_quality == "ok"
                        else 0.5 if m.measurement_quality == "degraded"
                        else 0.2)

            if fi >= first_swing_end and first_swing_end < n:
                phase = "address"; conf = 0.2
            elif fi <= A:
                phase = "address";        conf = 0.8
            elif fi <= takeaway_end:
                phase = "takeaway";       conf = 0.7
            elif fi < top_start:
                phase = "backswing";      conf = 0.75
            elif fi <= top_end:
                phase = "top";            conf = anchors.top_conf
            elif fi <= transition_end:
                phase = "transition";     conf = anchors.top_conf * 0.8
            elif fi < impact_lo:
                phase = "downswing";      conf = 0.75
            elif fi <= impact_hi:
                phase = "impact";         conf = anchors.impact_conf
            else:
                phase = "follow_through"; conf = 0.7

            annotations.append(PhaseAnnotation(
                frame_idx=fi,
                phase=phase,
                phase_confidence=round(float(conf * q_factor), 3),
            ))

        return annotations
