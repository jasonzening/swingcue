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
  const hintIdxRef = useRef<number>(0);

  // Cache once per timeline — re-computed only when timeline identity changes.
  const finishStartTs = findFinishStartTs(timeline);
  const durationSec = timeline.duration_sec;

  useEffect(() => {
    if (!videoEl) return;

    const draw = () => {
      const t = videoEl.currentTime;
      const lookup = frameAtTime(timeline, t, hintIdxRef.current);
      if (lookup === null) {
        // ts before first sample (e.g. paused at 0 with offset clip) →
        // hide all anchors; they'll reappear on the next frame in range.
        for (const name of COACHING_ANCHOR_NAMES_RENDER) {
          const el = circleRefs.current[name];
          if (el) el.setAttribute('visibility', 'hidden');
        }
        return;
      }
      hintIdxRef.current = lookup.idx;
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
  }, [videoEl, timeline, durationSec, finishStartTs]);

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
