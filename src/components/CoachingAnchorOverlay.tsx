'use client';

/**
 * CoachingAnchorOverlay — PR-7c-frontend-v4 simplification.
 *
 * Final visual: 5 magenta dots only.
 *   - head    (derived: midpoint of shoulders, shifted up 10% of torso)
 *   - left_shoulder, right_shoulder, left_hip, right_hip (MediaPipe direct)
 *
 * v3 elements REMOVED in v4: head halo ellipse, shoulder/hip occlusion
 * fallback (2 fallback center dots + 2 horizontal guide lines).
 * Per Jason's review of v3 production b32e0f21:
 *   - Halo should be a single dot anchored at cervical spine
 *     (golf coaching = detect head/neck stability)
 *   - Occlusion fallback read as visual clutter, not coaching value
 *
 * Visibility (unified, no phase-dependent branching):
 *   - 4 direct dots: hide if MediaPipe `coord.xy === null` OR
 *     `confidence < 0.3` (ANCHOR_DOT_CONFIDENCE_MIN)
 *   - Head dot: hide if any of the 4 torso source coords is null
 *     OR if both shoulder confidences < 0.5
 *
 * Architecture (unchanged from v2/v3):
 *   - SVG viewBox in video native px, preserveAspectRatio xMidYMid meet
 *   - Per-frame imperative setAttribute via rAF loop
 *   - One-shot syncs on loadedmetadata + seeked
 *   - Phase mapping: getCurrentPhase (single source of truth with badge)
 *   - Frame lookup: interpolatedFrame (smooth lerp)
 */

import { useEffect, useRef } from 'react';
import type { PhaseMarkers, PoseTimeline } from '@/types/analysis';
import { getCurrentPhase } from '@/lib/overlay/playerSync';
import {
  poseRawAnchorsAtTime,
  computeNeckCenter,
  ANCHOR_DOT_CONFIDENCE_MIN,
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
const DOT_STROKE_WIDTH = 1;

// Per spec §1: "opacity × 0.7" for the dot style.
const DOT_OPACITY_MULT = 0.7;

type DotName =
  | 'head'
  | 'left_shoulder'
  | 'right_shoulder'
  | 'left_hip'
  | 'right_hip';

const DOT_NAMES: readonly DotName[] = [
  'head',
  'left_shoulder',
  'right_shoulder',
  'left_hip',
  'right_hip',
];

export function CoachingAnchorOverlay({
  poseTimeline,
  phaseMarkers,
  videoEl,
}: Props) {
  const dotRefs = useRef<Partial<Record<DotName, SVGCircleElement | null>>>({});

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

      const anchors = poseRawAnchorsAtTime(poseTimeline, t);
      if (!anchors) {
        for (const name of DOT_NAMES) {
          const el = dotRefs.current[name];
          if (el) el.setAttribute('visibility', 'hidden');
        }
        return;
      }

      // 4 direct dots: hide if null OR conf < 0.3.
      applyDot(dotRefs.current.left_shoulder,  anchors.left_shoulder,  phaseOp);
      applyDot(dotRefs.current.right_shoulder, anchors.right_shoulder, phaseOp);
      applyDot(dotRefs.current.left_hip,       anchors.left_hip,       phaseOp);
      applyDot(dotRefs.current.right_hip,      anchors.right_hip,      phaseOp);

      // Head dot: derived from torso. Hide if geometry uncomputable OR
      // both shoulders < 0.5 confidence (derived midpoint unreliable).
      const headEl = dotRefs.current.head;
      if (headEl) {
        const bothShouldersLowConf =
          anchors.left_shoulder.confidence < HIGH_CONFIDENCE_THRESHOLD
          && anchors.right_shoulder.confidence < HIGH_CONFIDENCE_THRESHOLD;
        const headCenter = bothShouldersLowConf
          ? null
          : computeNeckCenter(
              anchors.left_shoulder,
              anchors.right_shoulder,
              anchors.left_hip,
              anchors.right_hip,
            );
        if (!headCenter) {
          headEl.setAttribute('visibility', 'hidden');
        } else {
          headEl.setAttribute('cx', String(headCenter.x));
          headEl.setAttribute('cy', String(headCenter.y));
          headEl.setAttribute('opacity', String(phaseOp * DOT_OPACITY_MULT));
          headEl.setAttribute('visibility', 'visible');
        }
      }
    };

    // Continuous rAF loop.
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

/**
 * Render a single direct anchor dot (shoulder/hip).
 * Hides when MediaPipe coord is null OR confidence below the v4 gate.
 */
function applyDot(
  el: SVGCircleElement | null | undefined,
  kp: RawKeypoint,
  phaseOp: number,
): void {
  if (!el) return;
  if (!kp.xy || kp.confidence < ANCHOR_DOT_CONFIDENCE_MIN) {
    el.setAttribute('visibility', 'hidden');
    return;
  }
  el.setAttribute('cx', String(kp.xy[0]));
  el.setAttribute('cy', String(kp.xy[1]));
  el.setAttribute('opacity', String(phaseOp * DOT_OPACITY_MULT));
  el.setAttribute('visibility', 'visible');
}
