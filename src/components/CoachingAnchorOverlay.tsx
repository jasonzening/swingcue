'use client';

/**
 * CoachingAnchorOverlay — PR-7c-frontend enhanced overlay.
 *
 * Renders 5 magenta coaching anchors per the offline motion_correction
 * CorrectedTimeline JSON (PR-7a deliverable shipped as `corrected-timelines/`
 * Supabase Storage bucket). Mounted by SwingPlayer only when a corrected
 * JSON is present for the current video.
 *
 * Architecture mirrors SkeletonOverlay:
 *   - SVG viewBox in video native pixel space; preserveAspectRatio
 *     "xMidYMid meet" letterboxes the same way as `<video object-fit:contain>`.
 *     Anchor coords from corrected_timeline.frames[i].coaching_anchors_2d
 *     are in those same native pixels — no manual scale math.
 *   - Per-frame ref.setAttribute updates via requestAnimationFrame loop;
 *     no React state during playback.
 *   - One-shot syncs on loadedmetadata + seeked (defense-in-depth, copied
 *     from SkeletonOverlay PR-5 §5.5).
 *   - Phase-aware opacity per phaseOpacity helper.
 *
 * Renders exactly 5 anchors (per PR-7c constitution):
 *   - left_shoulder_visual, right_shoulder_visual (per-side)
 *   - left_hip_visual, right_hip_visual (per-side)
 *   - neck_visual (single)
 *
 * The CorrectedTimeline schema also emits shoulder_disc_center and
 * hip_ring_center (midpoint-derived disc anchors). These are NOT
 * rendered in PR-7c-frontend — out of scope per the constitution.
 *
 * Visual style (matches offline probe rendering convention):
 *   - filled magenta circle (#FF00FF), radius 12 in native video pixels
 *   - 1 px black stroke for contrast against bright/dark backgrounds
 *   - opacity from phaseOpacity (full at setup/top, fading at fast motion)
 */

import { useEffect, useRef } from 'react';
import {
  type CorrectedTimeline,
  type CoachingAnchorName,
  COACHING_ANCHOR_NAMES_RENDER,
  frameAtTime,
  findFinishStartTs,
} from '@/lib/coaching/correctedTimeline';
import { computeAnchorOpacity } from '@/lib/coaching/phaseOpacity';

type Props = {
  timeline: CorrectedTimeline;
  videoEl: HTMLVideoElement | null;
};

const ANCHOR_FILL = '#FF00FF';       // magenta, matches probe convention
const ANCHOR_STROKE = 'rgba(0,0,0,0.6)';
const ANCHOR_RADIUS = 12;             // native video pixels
const ANCHOR_STROKE_WIDTH = 1;

export function CoachingAnchorOverlay({ timeline, videoEl }: Props) {
  const circleRefs = useRef<Partial<Record<CoachingAnchorName, SVGCircleElement | null>>>({});

  useEffect(() => {
    if (!videoEl) return;

    // PR-7c-frontend hotfix: inline these inside the effect to
    // eliminate dep-array churn risk on parent re-renders. Recomputed
    // exactly when the effect itself re-fires (videoEl or timeline
    // identity changes).
    const durationSec = timeline.duration_sec;
    const finishStartTs = findFinishStartTs(timeline);

    const draw = () => {
      const t = videoEl.currentTime;
      // PR-7c-frontend hotfix: frameAtTime is now stateless binary
      // search (no hint). Cannot get poisoned across calls.
      const lookup = frameAtTime(timeline, t);
      if (lookup === null) {
        // Empty timeline (shouldn't happen — validated at fetch time)
        // → hide all anchors defensively.
        for (const name of COACHING_ANCHOR_NAMES_RENDER) {
          const el = circleRefs.current[name];
          if (el) el.setAttribute('visibility', 'hidden');
        }
        return;
      }
      const frame = lookup.frame;
      const opacity = computeAnchorOpacity({
        phase: frame.phase,
        ts: t,
        durationSec,
        finishStartTs: finishStartTs ?? undefined,
      });

      for (const name of COACHING_ANCHOR_NAMES_RENDER) {
        const el = circleRefs.current[name];
        if (!el) continue;
        const uv = frame.coaching_anchors_2d[name];
        if (!uv) {
          el.setAttribute('visibility', 'hidden');
          continue;
        }
        const [x, y] = uv;
        if (typeof x !== 'number' || typeof y !== 'number') {
          el.setAttribute('visibility', 'hidden');
          continue;
        }
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

    // One-shot syncs (PR-5 §5.5 pattern copied from SkeletonOverlay).
    videoEl.addEventListener('loadedmetadata', draw);
    videoEl.addEventListener('seeked', draw);
    draw();

    return () => {
      cancelAnimationFrame(raf);
      videoEl.removeEventListener('loadedmetadata', draw);
      videoEl.removeEventListener('seeked', draw);
    };
  }, [videoEl, timeline]);

  return (
    <svg
      className="coaching-anchor-overlay"
      viewBox={`0 0 ${timeline.video_width} ${timeline.video_height}`}
      preserveAspectRatio="xMidYMid meet"
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
      }}
    >
      {COACHING_ANCHOR_NAMES_RENDER.map((name) => (
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
