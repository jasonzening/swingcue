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
  Keypoint,
  PoseFrame,
  PoseTimeline,
} from '@/types/analysis';
import { COCO_KEYPOINT_NAMES, COCO_SKELETON_EDGES } from '@/lib/skeleton/coco';
// PR-5.8A: render-time SwingCue coaching-anchor expansion.
import {
  expandShoulders,
  expandHips,
  SHOULDER_EXPAND_DEFAULT,
  HIP_EXPAND_DEFAULT,
  type Point2D,
} from '@/lib/skeleton/coachingAnchors';
// PR-5.9 Task 3: linear-interpolated time lookup (replaces nearestFrame).
import { interpolatedFrame } from '@/lib/disc/frameAt';

type Props = {
  timeline: PoseTimeline;
  videoEl: HTMLVideoElement | null;
  // PR-5.8A: outward expansion factors along the shoulder/hip line.
  // Optional with defaults from coachingAnchors. See module docstring.
  shoulderExpand?: number;
  hipExpand?: number;
  // PR-5.9 Task 5: debug overlay mode. When `'pose'` AND the active
  // frame has `raw_keypoints` (v1.5+ pose_timeline_2d), render a
  // second dot set in blue so the smoothing effect is visible. Silently
  // disabled when raw_keypoints is absent (legacy v1 videos).
  debugMode?: 'pose';
};

// PR-5.8A: keypoints whose draw position becomes the expanded value.
// All other keypoints render raw. Edges from shoulder→elbow and
// hip→knee use the expanded proximal + the raw distal.
const EXPANDED_NAMES = new Set<CocoKeypointName>([
  'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip',
]);

const COLOR_HIGH = '#CCCCCC';      // bright grey — confident keypoint
const COLOR_MID  = '#666666';      // dim grey — lower confidence
const COLOR_EDGE = '#999999';      // skeleton bone
// PR-5.9 Task 5: raw debug dot style.
const COLOR_RAW  = '#5599FF';
const RAW_DOT_RADIUS = 3;
const RAW_DOT_OPACITY = 0.7;

// PR-5.9 Task 6: 4-tier confidence-based fade. Replaces the prior
// binary HIGH_CONF threshold. Edge opacity uses the min of its two
// endpoint opacities — so an edge with one low-conf endpoint fades
// proportionally rather than disappearing in lockstep with the dot.
function confidenceOpacity(conf: number): number {
  if (conf >= 0.7) return 1.0;
  if (conf >= 0.5) return 0.7;
  if (conf >= 0.3) return 0.4;
  return 0; // < 0.3 → hidden
}

export function SkeletonOverlay({
  timeline,
  videoEl,
  shoulderExpand = SHOULDER_EXPAND_DEFAULT,
  hipExpand = HIP_EXPAND_DEFAULT,
  debugMode,
}: Props) {
  const dotRefs = useRef<Partial<Record<CocoKeypointName, SVGCircleElement | null>>>({});
  const edgeRefs = useRef<Array<SVGLineElement | null>>([]);
  // PR-5.9 Task 5: parallel raw-dot refs, mounted only when debugMode active.
  const rawDotRefs = useRef<Partial<Record<CocoKeypointName, SVGCircleElement | null>>>({});
  const lastValidFrameRef = useRef<PoseFrame | null>(null);
  const debugOn = debugMode === 'pose';

  useEffect(() => {
    if (!videoEl) return;

    // Single draw step — used by both the continuous rAF loop AND the
    // PR-5 §5.5 one-shot syncs (mount + loadedmetadata + seeked). Extracted
    // so we have exactly one source of truth for "render the SVG at the
    // current video time" regardless of who's triggering it.
    const draw = () => {
      const t = videoEl.currentTime;
      // PR-5.9 Task 3: interpolate between bracketing samples instead of
      // snapping to nearest. Removes the per-frame freeze visible
      // between pose samples on fast motion.
      const candidate = interpolatedFrame(timeline, t);
      const frame = candidate ?? lastValidFrameRef.current;
      if (candidate) lastValidFrameRef.current = candidate;
      if (!frame) return;

      // PR-5.8A: compute expanded shoulder + hip pairs once per draw.
      // Resolved at lookup time below for both dots and edges. When a
      // pair has any null coord, the resolver falls back to raw — the
      // existing null-handling branches below still hide the element.
      const expandedShoulder = expandPairOrNull(
        frame.keypoints.left_shoulder,
        frame.keypoints.right_shoulder,
        shoulderExpand,
        expandShoulders,
      );
      const expandedHip = expandPairOrNull(
        frame.keypoints.left_hip,
        frame.keypoints.right_hip,
        hipExpand,
        expandHips,
      );
      const resolveKp = (
        name: CocoKeypointName,
      ): readonly [number | null, number | null, number] => {
        if (!EXPANDED_NAMES.has(name)) return frame.keypoints[name];
        if (name === 'left_shoulder'  && expandedShoulder)
          return [expandedShoulder.left.x,  expandedShoulder.left.y,  frame.keypoints[name][2]];
        if (name === 'right_shoulder' && expandedShoulder)
          return [expandedShoulder.right.x, expandedShoulder.right.y, frame.keypoints[name][2]];
        if (name === 'left_hip'       && expandedHip)
          return [expandedHip.left.x,       expandedHip.left.y,       frame.keypoints[name][2]];
        if (name === 'right_hip'      && expandedHip)
          return [expandedHip.right.x,      expandedHip.right.y,      frame.keypoints[name][2]];
        return frame.keypoints[name];
      };

      // Per-keypoint opacity (PR-5.9 Task 6) — cached for edge endpoint
      // lookups below.
      const opacityByName: Partial<Record<CocoKeypointName, number>> = {};
      // Dots
      for (const name of COCO_KEYPOINT_NAMES) {
        const dot = dotRefs.current[name];
        if (!dot) continue;
        const kp = resolveKp(name);
        const [x, y, conf] = kp;
        const op = confidenceOpacity(conf);
        opacityByName[name] = op;
        if (x === null || y === null || op === 0) {
          dot.setAttribute('visibility', 'hidden');
        } else {
          dot.setAttribute('cx', String(x));
          dot.setAttribute('cy', String(y));
          dot.setAttribute('fill', conf >= 0.7 ? COLOR_HIGH : COLOR_MID);
          dot.setAttribute('opacity', String(op));
          dot.setAttribute('visibility', 'visible');
        }
      }
      // Edges — each endpoint resolved through resolveKp so the four
      // edges touching a shoulder/hip endpoint pick up the expanded
      // value; the elbow/wrist/knee/ankle distal points stay raw.
      COCO_SKELETON_EDGES.forEach(([from, to], i) => {
        const line = edgeRefs.current[i];
        if (!line) return;
        const a = resolveKp(from);
        const b = resolveKp(to);
        // PR-5.9 Task 6: edge opacity = min of two endpoint opacities.
        const edgeOp = Math.min(
          opacityByName[from] ?? 0,
          opacityByName[to]   ?? 0,
        );
        if (a[0] === null || a[1] === null || b[0] === null || b[1] === null || edgeOp === 0) {
          line.setAttribute('visibility', 'hidden');
        } else {
          line.setAttribute('x1', String(a[0]));
          line.setAttribute('y1', String(a[1]));
          line.setAttribute('x2', String(b[0]));
          line.setAttribute('y2', String(b[1]));
          line.setAttribute('opacity', String(0.7 * edgeOp));
          line.setAttribute('visibility', 'visible');
        }
      });
      // PR-5.9 Task 5: raw dots (debug mode only, when raw_keypoints
      // present on the interpolated frame). Rendered without PR-5.8A
      // expansion so the comparison is to truly-untouched data.
      if (debugOn) {
        // PR-5.9 Vercel fixup: PoseFrame.raw_keypoints is already
        // optional + correctly typed (Record<CocoKeypointName, Keypoint>).
        // The previous inline cast widened it to a looser shape and
        // dropped `readonly`, which TS strict mode rejects.
        const raw = frame.raw_keypoints;
        for (const name of COCO_KEYPOINT_NAMES) {
          const rd = rawDotRefs.current[name];
          if (!rd) continue;
          const rk = raw?.[name];
          if (!rk || rk[0] === null || rk[1] === null) {
            rd.setAttribute('visibility', 'hidden');
          } else {
            rd.setAttribute('cx', String(rk[0]));
            rd.setAttribute('cy', String(rk[1]));
            rd.setAttribute('visibility', 'visible');
          }
        }
      }
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
  }, [videoEl, timeline, shoulderExpand, hipExpand, debugOn]);

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
      {/* PR-5.9 Task 5: raw debug dots — only mounted when ?debug=pose. */}
      {debugOn && COCO_KEYPOINT_NAMES.map(name => (
        <circle
          key={`raw-${name}`}
          ref={el => { rawDotRefs.current[name] = el; }}
          r={RAW_DOT_RADIUS}
          fill={COLOR_RAW}
          opacity={RAW_DOT_OPACITY}
          visibility="hidden"
        />
      ))}
    </svg>
  );
}

/**
 * PR-5.8A: apply an expansion helper to an L/R Keypoint pair, returning
 * null when either side has a null coord (so the caller can fall back
 * to the raw value and let the existing null-handling branches hide
 * the element). Confidence is read from the raw value at the call site.
 */
function expandPairOrNull(
  L: Keypoint,
  R: Keypoint,
  factor: number,
  fn: (l: Point2D, r: Point2D, f: number) => { left: Point2D; right: Point2D },
): { left: Point2D; right: Point2D } | null {
  const [lx, ly] = L;
  const [rx, ry] = R;
  if (lx === null || ly === null || rx === null || ry === null) return null;
  return fn({ x: lx, y: ly }, { x: rx, y: ry }, factor);
}

/**
 * @deprecated PR-5.9 Task 3 — replaced by `interpolatedFrame` from
 * `@/lib/disc/frameAt`. Left here for one more PR cycle so any
 * out-of-tree consumer doesn't break; safe to delete after.
 *
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
