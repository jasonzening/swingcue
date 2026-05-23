'use client';

/**
 * CoachingAnchorOverlay — PR-7c-frontend-v7 body-axis geometric shift.
 *
 * Final visual: 5 magenta dots.
 *   - head           = MediaPipe nose direct (v7 — was derived neck in v4)
 *   - left_shoulder  = MediaPipe glenohumeral, shifted up+out toward acromion
 *   - right_shoulder = MediaPipe glenohumeral, shifted up+out toward acromion
 *   - left_hip       = MediaPipe hip socket, shifted up+out toward outer hip
 *   - right_hip      = MediaPipe hip socket, shifted up+out toward outer hip
 *
 * MediaPipe COCO shoulder/hip keypoints are anatomical interior joint
 * centers — by definition inside the visible body silhouette. v7
 * applies a deterministic, phase-invariant geometric correction in a
 * body-axis-relative frame (see `computeVisualAnchors` in
 * poseTimelineAnchors.ts) so the dots land on the outer silhouette
 * coaches reason about.
 *
 * Tuning: 4 ratios + 1 stability threshold in `VISUAL_ANCHOR_CONFIG`
 * (poseTimelineAnchors.ts) — single source of truth.
 *
 * Visibility (unified, no per-dot special-case):
 *   - All 5 dots use the same `applyDot` helper:
 *     hide if `RawKeypoint.xy === null` OR `confidence < 0.3`.
 *
 * Architecture (unchanged from v2..v6):
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
  computeVisualAnchors,
  ANCHOR_DOT_CONFIDENCE_MIN,
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

      // PR-7c-frontend-v7: shift MediaPipe interior joints (glenohumeral,
      // hip socket) outward toward the visible body silhouette via a
      // body-axis-relative geometric correction. All 5 anchors now flow
      // through the same `applyDot` helper with the < 0.3 confidence
      // gate. Head = direct MediaPipe nose (no longer derived).
      // Tuning happens in VISUAL_ANCHOR_CONFIG (poseTimelineAnchors.ts).
      const visual = computeVisualAnchors(anchors);
      applyDot(dotRefs.current.left_shoulder,  visual.left_shoulder,  phaseOp);
      applyDot(dotRefs.current.right_shoulder, visual.right_shoulder, phaseOp);
      applyDot(dotRefs.current.left_hip,       visual.left_hip,       phaseOp);
      applyDot(dotRefs.current.right_hip,      visual.right_hip,      phaseOp);
      applyDot(dotRefs.current.head,           visual.head,           phaseOp);
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
