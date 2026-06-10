"""
engine/b_phase/swing_phase.py
B Layer — Swing Phase Engine

Input:  List[FrameMeasurement] from A layer, fps, camera angle
Output: List[PhaseAnnotation] — one per frame, labeling which of 8 phases it belongs to

8 phases:
  address → takeaway → backswing → top → transition → downswing → impact → follow_through

Method:
  1. Detect 4 anchor keyframes (address / top / impact / follow_through end)
     using verified logic from SwingPhaseDetector + hands-forward impact correction
  2. Assign all frames to phases by interval:
     [0, address]           → address (pre-swing)
     (address, top*0.3]     → takeaway
     (top*0.3, top]         → backswing
     [top, top+transition]  → top (small window around top)
     (top+transition, impact-window] → downswing
     (impact-window, impact+window]  → impact (±window frames)
     (impact, finish]       → follow_through

Phase confidence:
  - anchor frames with high detector confidence → adjacent frames inherit
  - degraded measurement frames get phase_confidence *= 0.5
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

IMPACT_WINDOW = 3   # ± frames around impact keyframe = "impact" phase


@dataclass
class PhaseAnnotation:
    frame_idx: int
    phase: str                # one of PHASE_NAMES
    phase_confidence: float   # 0.0–1.0


@dataclass
class AnchorFrames:
    address: int
    top: int
    impact: int
    finish: int
    impact_conf: float = 0.0
    top_conf: float = 0.0


class SwingPhaseEngine:

    def __init__(self, sg_window_ms: int = 200):
        self.sg_window_ms = sg_window_ms

    def run(self, measurements: List[FrameMeasurement], fps: float,
            angle: str = "auto") -> tuple[List[PhaseAnnotation], AnchorFrames]:
        """
        angle: "face-on" | "down-the-line" | "auto"
        Returns (annotations, anchors)
        """
        n = len(measurements)
        xs, ys = self._extract_wrist(measurements, n)
        anchors = self._detect_anchors(measurements, xs, ys, fps, angle)
        annotations = self._assign_phases(n, anchors, measurements)
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

    # ── anchor detection ─────────────────────────────────────────────────────
    def _detect_anchors(self, measurements, xs, ys, fps, angle) -> AnchorFrames:
        n = len(xs)
        win = max(7, int(fps * self.sg_window_ms / 1000)) | 1
        xs_s = self._sg(xs, fps); ys_s = self._sg(ys, fps)
        dx = np.diff(xs_s, prepend=xs_s[0])
        dy = np.diff(ys_s, prepend=ys_s[0])
        spd = self._sg(np.sqrt(dx**2 + dy**2), fps)

        # ── address ──────────────────────────────────────────────────────────
        static_thr = max(np.percentile(spd, 20) * 3.0, 1.0)
        address = 2
        for i in range(2, int(n * 0.50)):
            if spd[i] < static_thr:
                address = i
            elif spd[i] > static_thr * 3 and i > address + 5:
                break

        # ── top ──────────────────────────────────────────────────────────────
        top_end = int(n * 0.82)
        ys_region = ys_s[address:top_end]
        peaks_l, pp = find_peaks(-ys_region, prominence=30, distance=int(fps*0.25))
        if len(peaks_l) == 0:
            top = address + int(np.argmin(ys_region))
            left_h  = ys_region[0]  - ys_region.min()
            right_h = ys_region[-1] - ys_region.min()
            top_prom = float(min(left_h, right_h))
        else:
            top = address + peaks_l[0]
            top_prom = float(pp["prominences"][0])
        top_conf = float(np.clip(top_prom / 150.0, 0, 1))

        # ── impact (angle-aware) ─────────────────────────────────────────────
        impact, impact_conf = self._detect_impact(
            xs_s, ys_s, spd, fps, angle, address, top, n, measurements=measurements)

        # ── finish ───────────────────────────────────────────────────────────
        settle_thr = static_thr * 1.5
        settle_win = max(int(fps * 0.35), 4)
        # ft_top: first ys minimum after impact
        ft_region = ys_s[impact:min(impact + int(fps*2), n)]
        ft_peaks, _ = find_peaks(-ft_region, prominence=15, distance=int(fps*0.15))
        ft_top = (impact + ft_peaks[0]) if len(ft_peaks) > 0 else (impact + int(fps*0.3))

        finish = min(n-1, ft_top + settle_win)
        for i in range(ft_top+1, n - settle_win):
            if np.all(spd[i:i+settle_win] < settle_thr):
                finish = i + settle_win // 2; break
        else:
            finish = ft_top + int(np.argmin(spd[ft_top:])) if ft_top < n else n-1

        return AnchorFrames(
            address=int(address), top=int(top),
            impact=int(impact), finish=int(min(finish, n-1)),
            impact_conf=impact_conf, top_conf=top_conf,
        )

    def _detect_impact(self, xs_s, ys_s, spd, fps, angle, address, top, n,
                       measurements=None) -> tuple[int, float]:
        """
        DTL:     local MAX of wrist-X after top (hands most forward)
        Face-on: local MAX of wrist-Y after top (hands lowest)

        Strategy: find peaks in the full signal from top+2 onward (so that
        prominence is not truncated by a short search window), then select the
        highest-prominence peak within a time window after top.

        Window: 4.5s for face-on, 3.0s for DTL.
        The real swing impact peak is more energetic (higher prominence) than
        any practice-swing or waggle peak, so highest-prominence is more robust
        than first-peak for multi-swing videos.
        """
        if angle == "down-the-line":
            max_search_seconds = 3.0
            signal = xs_s
        else:  # face-on or auto
            max_search_seconds = 4.5
            signal = ys_s

        search_start = top + 2
        # Find peaks in the FULL signal from search_start to end
        # (so prominence isn't truncated by a short window)
        full_region = signal[search_start:]
        peaks_full, props_full = find_peaks(
            full_region,
            prominence=10 if angle == "down-the-line" else 15,
            distance=int(fps * 0.1)
        )

        # Filter to those within max_search_seconds of top
        search_end = min(n - 1, top + int(fps * max_search_seconds))
        mask = (peaks_full + search_start) <= search_end
        peaks = peaks_full[mask]
        prominences = props_full["prominences"][mask]

        if len(peaks) == 0:
            # Fallback: argmax of capped region
            cap_region = signal[search_start:search_end + 1]
            if len(cap_region) == 0:
                return top + 1, 0.3
            impact = search_start + int(np.argmax(cap_region))
            peak_prom = float(np.max(cap_region) - np.min(cap_region))
        else:
            # Highest-prominence peak in the window = the real (most energetic) swing
            best_idx = int(np.argmax(prominences))
            impact = search_start + peaks[best_idx]
            peak_prom = float(prominences[best_idx])

        # Impact confidence: peak_prominence relative to torso height
        torso = 150.0  # rough fallback
        if measurements is not None:
            addr_m = measurements[address] if address < len(measurements) else None
            if addr_m is not None:
                th = addr_m.torso_height()
                if th > 0:
                    torso = th
        conf = float(np.clip(peak_prom / (torso * 0.5), 0, 1))
        return int(impact), conf

    # ── phase assignment ─────────────────────────────────────────────────────
    def _assign_phases(self, n, anchors: AnchorFrames,
                       measurements: List[FrameMeasurement]) -> List[PhaseAnnotation]:
        A = anchors.address; TOP = anchors.top
        IMP = anchors.impact; FIN = anchors.finish

        # Key boundaries
        takeaway_end = A + max(1, int((TOP - A) * 0.25))  # first 25% of backswing
        top_start    = TOP - 1           # top phase starts 1 frame before TOP
        top_end      = TOP + 2           # top phase ends 2 frames after TOP
        transition_end = TOP + max(3, int((IMP - TOP) * 0.15))  # ~15% of downswing, min 3 frames
        impact_lo = IMP - IMPACT_WINDOW
        impact_hi = IMP + IMPACT_WINDOW

        annotations = []
        for fi in range(n):
            m = measurements[fi]
            q_factor = 1.0 if m.measurement_quality == "ok" else (
                       0.5 if m.measurement_quality == "degraded" else 0.2)

            if fi <= A:
                phase = "address"; conf = 0.8
            elif fi <= takeaway_end:
                phase = "takeaway"; conf = 0.7
            elif fi < top_start:
                phase = "backswing"; conf = 0.75
            elif fi <= top_end:
                phase = "top"; conf = anchors.top_conf
            elif fi <= transition_end:
                phase = "transition"; conf = anchors.top_conf * 0.8
            elif fi < impact_lo:
                phase = "downswing"; conf = 0.75
            elif fi <= impact_hi:
                phase = "impact"; conf = anchors.impact_conf
            else:
                phase = "follow_through"; conf = 0.7

            annotations.append(PhaseAnnotation(
                frame_idx=fi,
                phase=phase,
                phase_confidence=round(float(conf * q_factor), 3),
            ))

        return annotations
