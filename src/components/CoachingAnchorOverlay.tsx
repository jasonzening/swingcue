'use client';

/**
 * CoachingAnchorOverlay — PR-7c-frontend-v2 enhanced overlay.
 *
 * Renders 5 magenta coaching anchors from production MediaPipe
 * pose_timeline_2d (the PoseTimeline already loaded by SwingPlayer
 * for the legacy SkeletonOverlay path). Mounted by SwingPlayer when
 * poseTimeline is available (= virtually all PR-4-analyzed videos).
 *
 * Data source migration (v1 → v2): MCP audit showed PR-7a corrected
 * anchors 54-91 px off MediaPipe at top phase on b32e0f21 frame 30.
 * MediaPipe coords land at visually correct shoulder per Jason GT
 * annotation. Until model swap (PR-7d NLF), MediaPipe-direct is the
 * working visual. See docs/decisions/PR-7c_REFRAME_OPTION_I.md.
 *
 * Architecture mirrors SkeletonOverlay:
 *   - SVG viewBox in video native pixel space; preserveAspectRatio
 *     "xMidYMid meet" letterboxes the same way as `<video object-fit:contain>`.
 *     MediaPipe coords from poseTimeline are in those same native
 *     pixels — no manual scale math.
 *   - Per-frame ref.setAttribute updates via requestAnimationFrame loop;
 *     no React state during playback.
 *   - One-shot syncs on loadedmetadata + seeked (defense-in-depth, PR-5 §5.5).
 *   - Phase-aware opacity via getCurrentPhase (the production helper used
 *     by the phase badge — single source of truth, no badge/anchor desync).
 *   - interpolatedFrame for smooth lerp between MediaPipe samples (no
 *     per-frame stutter on fast motion).
 *
 * Renders exactly 5 anchors (per PR-7c constitution):
 *   - head            ← MediaPipe `nose`  (higher conf than head_crown)
 *   - left_shoulder   ← MediaPipe `left_shoulder`
 *   - right_shoulder  ← MediaPipe `right_shoulder`
 *   - left_hip        ← MediaPipe `left_hip`
 *   - right_hip       ← MediaPipe `right_hip`
 *
 * Per-anchor confidence gate: hide if MediaPipe conf < 0.3 (same
 * threshold as PR-5.9 Task 6's confidenceOpacity lower bound).
 *
 * Visual style (matches offline probe convention):
 *   - filled magenta circle (#FF00FF), radius 12 native video pixels
 *   - 1 px black stroke for contrast against bright/dark backgrounds
 *   - opacity from phaseOpacity (full at setup/top, fading at fast motion)
 */

import { useEffect, useRef } from 'react';
import type { PhaseMarkers, PoseTimeline } from '@/types/analysis';
import { getCurrentPhase } from '@/lib/overlay/playerSync';
import {
  VISUAL_ANCHOR_NAMES,
  type VisualAnchorName,
  poseAnchorsAtTime,
} from '@/lib/coaching/poseTimelineAnchors';
import { computeAnchorOpacity } from '@/lib/coaching/phaseOpacity';

type Props = {
  poseTimeline: PoseTimeline;
  phaseMarkers: PhaseMarkers;
  videoEl: HTMLVideoElement | null;
};

const ANCHOR_FILL = '#FF00FF';       // magenta, matches probe convention
const ANCHOR_STROKE = 'rgba(0,0,0,0.6)';
const ANCHOR_RADIUS = 12;             // native video pixels
const ANCHOR_STROKE_WIDTH = 1;

export function CoachingAnchorOverlay({
  poseTimeline,
  phaseMarkers,
  videoEl,
}: Props) {
  const circleRefs = useRef<Partial<Record<VisualAnchorName, SVGCircleElement | null>>>({});

  useEffect(() => {
    if (!videoEl) return;

    const draw = () => {
      const t = videoEl.currentTime;
      const duration = videoEl.duration;
      // Guard NaN/0 duration before metadata loads — getCurrentPhase
      // returns 'setup' in that case (normT = 0).
      const safeDuration = Number.isFinite(duration) && duration > 0 ? duration : 1;

      // Production phase mapping: same helper that drives the phase
      // badge → anchor opacity stays in sync with the badge text.
      const phase = getCurrentPhase(phaseMarkers, t, safeDuration);
      const opacity = computeAnchorOpacity({
        phase,
        ts: t,
        durationSec: safeDuration,
        finishStartTs: phaseMarkers.finishTime,
      });

      const anchors = poseAnchorsAtTime(poseTimeline, t);
      if (!anchors) {
        // Empty timeline or t before first sample → hide all.
        for (const name of VISUAL_ANCHOR_NAMES) {
          const el = circleRefs.current[name];
          if (el) el.setAttribute('visibility', 'hidden');
        }
        return;
      }

      for (const name of VISUAL_ANCHOR_NAMES) {
        const el = circleRefs.current[name];
        if (!el) continue;
        const anchor = anchors[name];
        if (!anchor.xy) {
          // null x/y or low confidence — hide this one anchor; others
          // continue to render.
          el.setAttribute('visibility', 'hidden');
          continue;
        }
        const [x, y] = anchor.xy;
        el.setAttribute('cx', String(x));
        el.setAttribute('cy', String(y));
        el.setAttribute('opacity', String(opacity));
        el.setAttribute('visibility', 'visible');
      }
    };

    // Continuous rAF loop — mirrors SkeletonOverlay.
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
  }, [videoEl, poseTimeline, phaseMarkers]);

  return (
    <svg
      className="coaching-anchor-overlay"
      viewBox={`0 0 ${poseTimeline.video_width} ${poseTimeline.video_height}`}
      preserveAspectRatio="xMidYMid meet"
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
      }}
    >
      {VISUAL_ANCHOR_NAMES.map((name) => (
        <circle
          key={name}
          ref={(el) => {
            circleRefs.current[name] = el;
          }}
          r={ANCHOR_RADIUS}
          fill={ANCHOR_FILL}
          stroke={ANCHOR_STROKE}
          strokeWidth={ANCHOR_STROKE_WIDTH}
          visibility="hidden"
        />
      ))}
    </svg>
  );
}
