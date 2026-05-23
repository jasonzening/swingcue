'use client';

/**
 * CoachingAnchorOverlay — PR-7c-frontend-v3 visual refinement.
 *
 * v2 (frozen): 5 magenta dots tracking nose + shoulders + hips from
 * production MediaPipe pose_timeline_2d. Confirmed working post the
 * frameAtTime hotfix.
 *
 * v3 elevates "debug dot tracker" to "coaching visual system":
 *   1. Head dot → stable head halo derived from torso anchors.
 *      Rationale: MediaPipe nose drifts on head turn at top phase.
 *      Halo center = shoulder midpoint shifted up 15% of torso length.
 *      Ellipse rx/ry sized to shoulder half-width.
 *   2. Per-anchor confidence gating (< 0.5 hide).
 *      Non-setup phases additionally fall back to disc/ring center
 *      + horizontal guide line when EITHER L/R confidence drops.
 *   3. Visual hierarchy:
 *      - Individual dots: small (r=5), opacity × 0.7 (secondary)
 *      - Halo + disc/ring centers + guide lines: full phase opacity (primary)
 *
 * Architecture (unchanged from v2):
 *   - SVG viewBox in video native px, preserveAspectRatio xMidYMid meet
 *   - Per-frame imperative setAttribute via rAF loop
 *   - One-shot syncs on loadedmetadata + seeked
 *   - Phase mapping: getCurrentPhase (single source of truth with badge)
 *   - Frame lookup: interpolatedFrame (smooth lerp)
 *
 * SVG element count: 8 max per frame
 *   - 4 dots (L/R shoulder, L/R hip)
 *   - 1 head halo (ellipse, stroke only)
 *   - 2 fallback markers (shoulder disc center, hip ring center)
 *   - 2 guide lines (shoulder horizontal, hip horizontal)
 */

import { useEffect, useRef } from 'react';
import type { PhaseMarkers, PoseTimeline } from '@/types/analysis';
import { getCurrentPhase } from '@/lib/overlay/playerSync';
import {
  poseRawAnchorsAtTime,
  computeShoulderCenter,
  computeHipCenter,
  computeHeadHalo,
  HIGH_CONFIDENCE_THRESHOLD,
  type RawKeypoint,
} from '@/lib/coaching/poseTimelineAnchors';
import { computeAnchorOpacity } from '@/lib/coaching/phaseOpacity';

type Props = {
  poseTimeline: PoseTimeline;
  phaseMarkers: PhaseMarkers;
  videoEl: HTMLVideoElement | null;
};

const MAGENTA = '#FF00FF';
const DOT_STROKE = 'rgba(0,0,0,0.5)';
const DOT_RADIUS = 5;
const CENTER_RADIUS = 6;
const HALO_STROKE_WIDTH = 2;
const DOT_STROKE_WIDTH = 1;
const GUIDE_LINE_WIDTH = 1;

// Opacity multipliers applied on top of phaseOpacity:
const DOT_OPACITY_MULT    = 0.7;   // secondary visual
const HALO_OPACITY_MULT   = 0.8;   // primary visual (slight cap)
const CENTER_OPACITY_MULT = 1.0;   // primary, full phase opacity
const GUIDE_OPACITY_MULT  = 0.6;   // subtle reference

type DotName = 'left_shoulder' | 'right_shoulder' | 'left_hip' | 'right_hip';
const DOT_NAMES: readonly DotName[] = [
  'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip',
];

export function CoachingAnchorOverlay({
  poseTimeline,
  phaseMarkers,
  videoEl,
}: Props) {
  // 4 dot refs
  const dotRefs = useRef<Partial<Record<DotName, SVGCircleElement | null>>>({});
  // Halo ref (ellipse)
  const haloRef = useRef<SVGEllipseElement | null>(null);
  // Fallback marker refs
  const shoulderCenterRef = useRef<SVGCircleElement | null>(null);
  const hipCenterRef      = useRef<SVGCircleElement | null>(null);
  // Guide line refs
  const shoulderGuideRef = useRef<SVGLineElement | null>(null);
  const hipGuideRef      = useRef<SVGLineElement | null>(null);

  const videoWidth = poseTimeline.video_width;
  const videoHeight = poseTimeline.video_height;

  useEffect(() => {
    if (!videoEl) return;

    const draw = () => {
      const t = videoEl.currentTime;
      const duration = videoEl.duration;
      const safeDuration =
        Number.isFinite(duration) && duration > 0 ? duration : 1;

      const phase = getCurrentPhase(phaseMarkers, t, safeDuration);
      const phaseOp = computeAnchorOpacity({
        phase,
        ts: t,
        durationSec: safeDuration,
        finishStartTs: phaseMarkers.finishTime,
      });
      const isSetup = phase === 'setup';

      const anchors = poseRawAnchorsAtTime(poseTimeline, t);
      if (!anchors) {
        // Empty timeline / before first sample — hide everything.
        hideAll();
        return;
      }

      const shL = anchors.left_shoulder;
      const shR = anchors.right_shoulder;
      const hipL = anchors.left_hip;
      const hipR = anchors.right_hip;

      // ── Decide visibility ────────────────────────────────────
      // Individual dot: visible iff non-null AND (setup OR conf >= 0.5).
      const showShL = isVisibleDot(shL, isSetup);
      const showShR = isVisibleDot(shR, isSetup);
      const showHipL = isVisibleDot(hipL, isSetup);
      const showHipR = isVisibleDot(hipR, isSetup);

      // Fallback: when not setup AND EITHER side fails confidence,
      // hide BOTH dots and show disc/ring center + guide line instead.
      const shoulderEitherLowConf =
        shL.confidence < HIGH_CONFIDENCE_THRESHOLD
        || shR.confidence < HIGH_CONFIDENCE_THRESHOLD;
      const hipEitherLowConf =
        hipL.confidence < HIGH_CONFIDENCE_THRESHOLD
        || hipR.confidence < HIGH_CONFIDENCE_THRESHOLD;

      const showShoulderFallback = !isSetup && shoulderEitherLowConf;
      const showHipFallback      = !isSetup && hipEitherLowConf;

      // Suppress individual dots when fallback is active.
      const finalShowShL = showShL && !showShoulderFallback;
      const finalShowShR = showShR && !showShoulderFallback;
      const finalShowHipL = showHipL && !showHipFallback;
      const finalShowHipR = showHipR && !showHipFallback;

      // ── Dots ─────────────────────────────────────────────────
      applyDot(dotRefs.current.left_shoulder,  shL,  finalShowShL,  phaseOp);
      applyDot(dotRefs.current.right_shoulder, shR,  finalShowShR,  phaseOp);
      applyDot(dotRefs.current.left_hip,       hipL, finalShowHipL, phaseOp);
      applyDot(dotRefs.current.right_hip,      hipR, finalShowHipR, phaseOp);

      // ── Head halo ────────────────────────────────────────────
      // Per spec: hide entirely if BOTH shoulder confidences < 0.5.
      // Halo geometry also requires all 4 torso coords non-null.
      const haloEl = haloRef.current;
      if (haloEl) {
        const bothShouldersLowConf =
          shL.confidence < HIGH_CONFIDENCE_THRESHOLD
          && shR.confidence < HIGH_CONFIDENCE_THRESHOLD;
        const halo =
          bothShouldersLowConf
            ? null
            : computeHeadHalo(shL, shR, hipL, hipR);
        if (halo === null) {
          haloEl.setAttribute('visibility', 'hidden');
        } else {
          haloEl.setAttribute('cx', String(halo.cx));
          haloEl.setAttribute('cy', String(halo.cy));
          haloEl.setAttribute('rx', String(halo.rx));
          haloEl.setAttribute('ry', String(halo.ry));
          haloEl.setAttribute('opacity', String(phaseOp * HALO_OPACITY_MULT));
          haloEl.setAttribute('visibility', 'visible');
        }
      }

      // ── Shoulder fallback (disc center + guide line) ─────────
      const shCenter = showShoulderFallback
        ? computeShoulderCenter(shL, shR)
        : null;
      applyCenter(shoulderCenterRef.current, shCenter, phaseOp);
      applyGuideLine(
        shoulderGuideRef.current, shCenter, videoWidth, phaseOp,
      );

      // ── Hip fallback (ring center + guide line) ──────────────
      const hipCenter = showHipFallback
        ? computeHipCenter(hipL, hipR)
        : null;
      applyCenter(hipCenterRef.current, hipCenter, phaseOp);
      applyGuideLine(
        hipGuideRef.current, hipCenter, videoWidth, phaseOp,
      );
    };

    const hideAll = () => {
      for (const name of DOT_NAMES) {
        const el = dotRefs.current[name];
        if (el) el.setAttribute('visibility', 'hidden');
      }
      haloRef.current?.setAttribute('visibility', 'hidden');
      shoulderCenterRef.current?.setAttribute('visibility', 'hidden');
      hipCenterRef.current?.setAttribute('visibility', 'hidden');
      shoulderGuideRef.current?.setAttribute('visibility', 'hidden');
      hipGuideRef.current?.setAttribute('visibility', 'hidden');
    };

    // Continuous rAF loop — mirrors SkeletonOverlay / v2.
    let raf = 0;
    const loop = () => {
      draw();
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    // One-shot syncs (PR-5 §5.5 pattern).
    videoEl.addEventListener('loadedmetadata', draw);
    videoEl.addEventListener('seeked', draw);
    draw();

    return () => {
      cancelAnimationFrame(raf);
      videoEl.removeEventListener('loadedmetadata', draw);
      videoEl.removeEventListener('seeked', draw);
    };
  }, [videoEl, poseTimeline, phaseMarkers, videoWidth, videoHeight]);

  return (
    <svg
      className="coaching-anchor-overlay"
      viewBox={`0 0 ${videoWidth} ${videoHeight}`}
      preserveAspectRatio="xMidYMid meet"
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
      }}
    >
      {/* Guide lines — render before centers so dots/circles draw above them */}
      <line
        ref={(el) => { shoulderGuideRef.current = el; }}
        stroke={MAGENTA}
        strokeWidth={GUIDE_LINE_WIDTH}
        visibility="hidden"
      />
      <line
        ref={(el) => { hipGuideRef.current = el; }}
        stroke={MAGENTA}
        strokeWidth={GUIDE_LINE_WIDTH}
        visibility="hidden"
      />

      {/* Head halo — stroke-only ellipse */}
      <ellipse
        ref={(el) => { haloRef.current = el; }}
        fill="none"
        stroke={MAGENTA}
        strokeWidth={HALO_STROKE_WIDTH}
        visibility="hidden"
      />

      {/* Fallback center markers */}
      <circle
        ref={(el) => { shoulderCenterRef.current = el; }}
        r={CENTER_RADIUS}
        fill={MAGENTA}
        stroke={DOT_STROKE}
        strokeWidth={DOT_STROKE_WIDTH}
        visibility="hidden"
      />
      <circle
        ref={(el) => { hipCenterRef.current = el; }}
        r={CENTER_RADIUS}
        fill={MAGENTA}
        stroke={DOT_STROKE}
        strokeWidth={DOT_STROKE_WIDTH}
        visibility="hidden"
      />

      {/* 4 individual anchor dots */}
      {DOT_NAMES.map((name) => (
        <circle
          key={name}
          ref={(el) => { dotRefs.current[name] = el; }}
          r={DOT_RADIUS}
          fill={MAGENTA}
          stroke={DOT_STROKE}
          strokeWidth={DOT_STROKE_WIDTH}
          visibility="hidden"
        />
      ))}
    </svg>
  );
}

// ── Helpers (pure, no React) ─────────────────────────────────────

function isVisibleDot(kp: RawKeypoint, isSetup: boolean): boolean {
  if (!kp.xy) return false;
  if (isSetup) return true;  // setup: null-check only, no conf gate
  return kp.confidence >= HIGH_CONFIDENCE_THRESHOLD;
}

function applyDot(
  el: SVGCircleElement | null | undefined,
  kp: RawKeypoint,
  show: boolean,
  phaseOp: number,
): void {
  if (!el) return;
  if (!show || !kp.xy) {
    el.setAttribute('visibility', 'hidden');
    return;
  }
  el.setAttribute('cx', String(kp.xy[0]));
  el.setAttribute('cy', String(kp.xy[1]));
  el.setAttribute('opacity', String(phaseOp * DOT_OPACITY_MULT));
  el.setAttribute('visibility', 'visible');
}

function applyCenter(
  el: SVGCircleElement | null,
  center: { x: number; y: number } | null,
  phaseOp: number,
): void {
  if (!el) return;
  if (!center) {
    el.setAttribute('visibility', 'hidden');
    return;
  }
  el.setAttribute('cx', String(center.x));
  el.setAttribute('cy', String(center.y));
  el.setAttribute('opacity', String(phaseOp * CENTER_OPACITY_MULT));
  el.setAttribute('visibility', 'visible');
}

function applyGuideLine(
  el: SVGLineElement | null,
  center: { x: number; y: number } | null,
  videoWidth: number,
  phaseOp: number,
): void {
  if (!el) return;
  if (!center) {
    el.setAttribute('visibility', 'hidden');
    return;
  }
  el.setAttribute('x1', '0');
  el.setAttribute('y1', String(center.y));
  el.setAttribute('x2', String(videoWidth));
  el.setAttribute('y2', String(center.y));
  el.setAttribute('opacity', String(phaseOp * GUIDE_OPACITY_MULT));
  el.setAttribute('visibility', 'visible');
}
