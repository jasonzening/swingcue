'use client';

/**
 * SkeletonOverlay — debug + demo visualisation of the 17-COCO timeline.
 *
 * PR-4's only visible deliverable. Renders SVG circles for keypoints
 * and SVG lines for the canonical skeleton edges, synced to the video's
 * currentTime via requestAnimationFrame.
 *
 * Architecture notes:
 *   - SVG viewBox is in video native pixel space (matches the v1 JSON
 *     `video_width × video_height`). preserveAspectRatio="xMidYMid meet"
 *     letterboxes the same way as the <video object-fit:contain>, so
 *     keypoint coords land in the right visual spot with zero manual
 *     scale math.
 *   - Per-frame updates go through ref.setAttribute() — no React state,
 *     no re-render during playback.
 *   - Carry-forward: if nearestFrame() returns null (zero-frame timeline)
 *     we hide everything. If a frame's individual keypoint is null
 *     (outlier rejected / low conf), only that dot/edge hides.
 *
 * Coordinate convention: video native pixels. Locked across all
 * future overlay PRs (PR-5+).
 */

import { useEffect, useRef } from 'react';
import type {
  CocoKeypointName,
  PoseFrame,
  PoseTimeline,
} from '@/types/analysis';
import { COCO_KEYPOINT_NAMES, COCO_SKELETON_EDGES } from '@/lib/skeleton/coco';

type Props = {
  timeline: PoseTimeline;
  videoEl: HTMLVideoElement | null;
};

const HIGH_CONF = 0.7;
const COLOR_HIGH = '#CCCCCC';      // bright grey — confident keypoint
const COLOR_MID  = '#666666';      // dim grey — lower confidence
const COLOR_EDGE = '#999999';      // skeleton bone

export function SkeletonOverlay({ timeline, videoEl }: Props) {
  const dotRefs = useRef<Partial<Record<CocoKeypointName, SVGCircleElement | null>>>({});
  const edgeRefs = useRef<Array<SVGLineElement | null>>([]);
  const lastValidFrameRef = useRef<PoseFrame | null>(null);

  useEffect(() => {
    if (!videoEl) return;

    // Single draw step — used by both the continuous rAF loop AND the
    // PR-5 §5.5 one-shot syncs (mount + loadedmetadata + seeked). Extracted
    // so we have exactly one source of truth for "render the SVG at the
    // current video time" regardless of who's triggering it.
    const draw = () => {
      const t = videoEl.currentTime;
      const candidate = nearestFrame(timeline.frames, t);
      const frame = candidate ?? lastValidFrameRef.current;
      if (candidate) lastValidFrameRef.current = candidate;
      if (!frame) return;

      // Dots
      for (const name of COCO_KEYPOINT_NAMES) {
        const dot = dotRefs.current[name];
        if (!dot) continue;
        const kp = frame.keypoints[name];
        const [x, y, conf] = kp;
        if (x === null || y === null) {
          dot.setAttribute('visibility', 'hidden');
        } else {
          dot.setAttribute('cx', String(x));
          dot.setAttribute('cy', String(y));
          dot.setAttribute('fill', conf >= HIGH_CONF ? COLOR_HIGH : COLOR_MID);
          dot.setAttribute('visibility', 'visible');
        }
      }
      // Edges
      COCO_SKELETON_EDGES.forEach(([from, to], i) => {
        const line = edgeRefs.current[i];
        if (!line) return;
        const a = frame.keypoints[from];
        const b = frame.keypoints[to];
        if (a[0] === null || a[1] === null || b[0] === null || b[1] === null) {
          line.setAttribute('visibility', 'hidden');
        } else {
          line.setAttribute('x1', String(a[0]));
          line.setAttribute('y1', String(a[1]));
          line.setAttribute('x2', String(b[0]));
          line.setAttribute('y2', String(b[1]));
          line.setAttribute('visibility', 'visible');
        }
      });
    };

    // Continuous rAF loop (existing behaviour).
    let raf = 0;
    const loop = () => {
      draw();
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    // PR-5 §5.5: defense-in-depth one-shot syncs. The continuous rAF
    // covers the playing case; these handle the edge cases where the
    // overlay is mounted after the video already has a meaningful
    // currentTime (e.g. user toggled the overlay on while the video was
    // ended, or seeked to a phase while paused). Without these, the
    // first paint may show stale / 0,0 positions until the next rAF.
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
      className="skeleton-overlay"
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
      {COCO_SKELETON_EDGES.map((_, i) => (
        <line
          key={`edge-${i}`}
          ref={el => { edgeRefs.current[i] = el; }}
          stroke={COLOR_EDGE}
          strokeWidth={2}
          opacity={0.7}
          visibility="hidden"
        />
      ))}
      {COCO_KEYPOINT_NAMES.map(name => (
        <circle
          key={`dot-${name}`}
          ref={el => { dotRefs.current[name] = el; }}
          r={5}
          stroke="rgba(0,0,0,0.5)"
          strokeWidth={1}
          visibility="hidden"
        />
      ))}
    </svg>
  );
}

/**
 * Nearest-frame lookup. Linear scan — fine for ~30–150 frames at 10 fps.
 * Returns null only when frames array is empty.
 */
function nearestFrame(frames: PoseFrame[], t: number): PoseFrame | null {
  if (frames.length === 0) return null;
  if (t <= frames[0].ts) return frames[0];
  if (t >= frames[frames.length - 1].ts) return frames[frames.length - 1];
  let best = frames[0];
  let bestDist = Math.abs(t - best.ts);
  for (const f of frames) {
    const d = Math.abs(t - f.ts);
    if (d < bestDist) {
      best = f;
      bestDist = d;
    }
  }
  return best;
}
