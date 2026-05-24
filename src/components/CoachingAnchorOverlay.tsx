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
  findClosestFrameIdx,
  ANCHOR_DOT_CONFIDENCE_MIN,
  type RawKeypoint,
  type VisualAnchorConfigOverride,
} from '@/lib/coaching/poseTimelineAnchors';
import { computeAnchorOpacity } from '@/lib/coaching/phaseOpacity';

type Props = {
  /** PR-7c-frontend-v9: video id (UUID from /result/[id] route).
   * Used to look up per-video keyframes in VIDEO_KEYFRAMES. */
  videoId: string;
  poseTimeline: PoseTimeline;
  phaseMarkers: PhaseMarkers;
  videoEl: HTMLVideoElement | null;
  /** PR-7c-frontend-v8 tune-mode: override production ratios with live
   * slider state. Single-set across all frames during tune (matches
   * v8.1 behavior). Production mode omits this prop and lets v9
   * keyframe interpolation drive per-frame ratios. */
  tuningRatios?: VisualAnchorConfigOverride;
  /** PR-7c-frontend-v8 tune-mode: render 4 raw MediaPipe shoulder/hip
   * dots in green at lower opacity as a "before" reference. */
  showRawDots?: boolean;
};

const MAGENTA = '#FF00FF';
// PR-7c-frontend-v8: raw "before" dots in tune mode use a contrasting
// neon green so the magenta shifted dots remain the primary visual.
const GREEN_RAW = '#00FF88';
// PR-7c-frontend-v8.1: white stroke (was black 50%) for high contrast
// on dark video frames + bright skin tones. Bigger radius + magenta
// drop-shadow glow ensure dots remain visible above tuning-panel area.
const DOT_STROKE = 'rgba(255,255,255,0.9)';
const DOT_RADIUS = 7;
const DOT_STROKE_WIDTH = 2;
const DOT_GLOW_FILTER = 'drop-shadow(0 0 3px rgba(255,0,255,0.7))';
const RAW_DOT_GLOW_FILTER = 'drop-shadow(0 0 2px rgba(0,255,136,0.5))';

// Per spec §1: "opacity × 0.7" for the dot style.
const DOT_OPACITY_MULT = 0.7;
// PR-7c-frontend-v8: raw dots render at half the magenta opacity so
// they read as a secondary reference, not the primary signal.
const RAW_DOT_OPACITY_MULT = 0.35;

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

/** PR-7c-frontend-v8: raw "before" dots in tune mode show only the 4
 * torso joints (head has no raw-vs-shifted distinction — it's nose
 * direct in both raw and visual). */
type RawDotName =
  | 'left_shoulder'
  | 'right_shoulder'
  | 'left_hip'
  | 'right_hip';

const RAW_DOT_NAMES: readonly RawDotName[] = [
  'left_shoulder',
  'right_shoulder',
  'left_hip',
  'right_hip',
];

export function CoachingAnchorOverlay({
  videoId,
  poseTimeline,
  phaseMarkers,
  videoEl,
  tuningRatios,
  showRawDots,
}: Props) {
  const dotRefs = useRef<Partial<Record<DotName, SVGCircleElement | null>>>({});
  // PR-7c-frontend-v8: separate refs for the optional raw "before" dots.
  // Only populated when showRawDots is true (tune mode).
  const rawDotRefs = useRef<Partial<Record<RawDotName, SVGCircleElement | null>>>({});

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
        if (showRawDots) {
          for (const name of RAW_DOT_NAMES) {
            const el = rawDotRefs.current[name];
            if (el) el.setAttribute('visibility', 'hidden');
          }
        }
        return;
      }

      // PR-7c-frontend-v7: shift MediaPipe interior joints (glenohumeral,
      // hip socket) outward toward the visible body silhouette via a
      // body-axis-relative geometric correction. All 5 anchors now flow
      // through the same `applyDot` helper with the < 0.3 confidence
      // gate. Head = direct MediaPipe nose (no longer derived).
      // v9: ratios source = per-frame keyframe interpolation from
      // VIDEO_KEYFRAMES[videoId], or DEFAULT_RATIOS fallback. Tune mode
      // (tuningRatios !== undefined) overrides with a single live set
      // for the current dragging session.
      const frameIdx = findClosestFrameIdx(poseTimeline, t);
      const visual = computeVisualAnchors(anchors, frameIdx, videoId, tuningRatios);
      applyDot(dotRefs.current.left_shoulder,  visual.left_shoulder,  phaseOp);
      applyDot(dotRefs.current.right_shoulder, visual.right_shoulder, phaseOp);
      applyDot(dotRefs.current.left_hip,       visual.left_hip,       phaseOp);
      applyDot(dotRefs.current.right_hip,      visual.right_hip,      phaseOp);
      applyDot(dotRefs.current.head,           visual.head,           phaseOp);

      // PR-7c-frontend-v8: raw "before" dots — green, lower opacity.
      // Same < 0.3 confidence gate as the magenta dots. Only renders
      // in tune mode.
      if (showRawDots) {
        const rawOp = phaseOp * (RAW_DOT_OPACITY_MULT / DOT_OPACITY_MULT);
        applyDot(rawDotRefs.current.left_shoulder,  anchors.left_shoulder,  rawOp);
        applyDot(rawDotRefs.current.right_shoulder, anchors.right_shoulder, rawOp);
        applyDot(rawDotRefs.current.left_hip,       anchors.left_hip,       rawOp);
        applyDot(rawDotRefs.current.right_hip,      anchors.right_hip,      rawOp);
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
  }, [videoEl, poseTimeline, phaseMarkers, videoWidth, videoHeight, tuningRatios, showRawDots, videoId]);

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
        // PR-7c-frontend-v8.1: above the tuning panel (z=10) so dots
        // remain visible behind/around the panel. pointer-events:none
        // means clicks still reach the panel sliders.
        zIndex: 20,
      }}
    >
      {/* Raw "before" dots rendered FIRST so the magenta shifted dots
          paint on top in tune mode (showRawDots only). */}
      {showRawDots && RAW_DOT_NAMES.map((name) => (
        <circle
          key={`raw-${name}`}
          ref={(el) => { rawDotRefs.current[name] = el; }}
          r={DOT_RADIUS}
          fill={GREEN_RAW}
          stroke={DOT_STROKE}
          strokeWidth={DOT_STROKE_WIDTH}
          style={{ filter: RAW_DOT_GLOW_FILTER }}
          visibility="hidden"
        />
      ))}
      {DOT_NAMES.map((name) => (
        <circle
          key={name}
          ref={(el) => { dotRefs.current[name] = el; }}
          r={DOT_RADIUS}
          fill={MAGENTA}
          stroke={DOT_STROKE}
          strokeWidth={DOT_STROKE_WIDTH}
          style={{ filter: DOT_GLOW_FILTER }}
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
