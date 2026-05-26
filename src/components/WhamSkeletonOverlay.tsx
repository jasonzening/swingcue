'use client';

/**
 * WhamSkeletonOverlay — PR-8d.1 bone-center full skeleton from WHAM.
 *
 * Reads per-frame keypoints_2d_projected from wham_pose_timeline rows
 * (fetched by the result page from Supabase), in ABSOLUTE source-video
 * pixels (top-left origin). SVG viewBox in source-video px +
 * preserveAspectRatio="xMidYMid meet" matches CSS object-fit:contain
 * automatically — same letterboxing as the browser's <video>.
 *
 * Distinct from CoachingAnchorOverlay (PR-7c-frontend, MediaPipe-derived
 * magenta anchors) — that path stays available for legacy/MediaPipe
 * mode. WHAM mode picks this overlay instead in SwingPlayer.
 *
 * Architecture mirrors CoachingAnchorOverlay:
 *   - rAF loop driven by videoEl.currentTime
 *   - imperative setAttribute on SVG refs (no React re-render per frame)
 *   - frame lookup by closest frame_timestamp_ms (binary search)
 *
 * Per PR-8d.1 R2: NO per-keypoint confidence field exists in WHAM
 * output. Visibility is per-FRAME via fit_ok. If fit_ok=false, hide the
 * entire skeleton for that frame. Otherwise full opacity.
 *
 * Per PR-8d.1 R7: single SVG with stable refs; no React state change
 * per frame; ~17 setAttribute calls per joint + ~14 per edge → ~64k/sec
 * at 30fps. Browser handles this without GC pressure.
 *
 * Palette locked in docs/decisions/PR-8d_PALETTE.md:
 *   #00C2FF cyan   — anatomical LEFT side
 *   #FFB000 amber  — anatomical RIGHT side
 *   #FFFFFF white  — centerline (spine + crossbody)
 */

import { useEffect, useMemo, useRef } from 'react';

// PR-8d.1: shape that result page assembles from wham_video_meta +
// wham_pose_timeline. Compatible with the raw row shape (1:1 fields)
// plus image_width/image_height for the SVG viewBox.
export type WhamPoseFrame = {
  frame_idx: number;
  frame_timestamp_ms: number | null;
  fit_ok: boolean;
  keypoints_2d_projected: Record<string, { x: number; y: number } | null> | null;
};

export type WhamPoseTimelineForOverlay = {
  image_width: number;
  image_height: number;
  processed_fps: number;
  frames: WhamPoseFrame[];   // pre-sorted by frame_idx ASC
};

type Props = {
  timeline: WhamPoseTimelineForOverlay;
  videoEl: HTMLVideoElement | null;
};

// Locked palette — see docs/decisions/PR-8d_PALETTE.md.
const COLOR_LEFT   = '#00C2FF';
const COLOR_RIGHT  = '#FFB000';
const COLOR_CENTER = '#FFFFFF';
const STROKE       = 'rgba(0, 0, 0, 0.45)';

// 17 joint inventory (per PR-8d.1 R1, verified via SQL on 401eb63b).
const LEFT_JOINTS  = ['left_shoulder', 'left_elbow', 'left_wrist',
                       'left_hip', 'left_knee', 'left_ankle'] as const;
const RIGHT_JOINTS = ['right_shoulder', 'right_elbow', 'right_wrist',
                       'right_hip', 'right_knee', 'right_ankle'] as const;
const CENTER_JOINTS = ['head_crown', 'head', 'neck', 'spine1', 'pelvis'] as const;

type LeftJoint   = typeof LEFT_JOINTS[number];
type RightJoint  = typeof RIGHT_JOINTS[number];
type CenterJoint = typeof CENTER_JOINTS[number];
type JointName   = LeftJoint | RightJoint | CenterJoint;

const ALL_JOINTS: readonly JointName[] = [
  ...LEFT_JOINTS, ...RIGHT_JOINTS, ...CENTER_JOINTS,
];

// PR-8e.1: when the backend (PR-8e.0) provides anatomical surface
// landmarks via SMPL mesh vertices, render shoulder/hip dots at those
// positions instead of the SMPL joint centers. SMPL pelvis ≈ L5
// vertebra (above the visible hip surface) and SMPL shoulder = the
// internal glenohumeral joint (medial to the visible acromion) — so
// SMPL-joint dots sit a few cm off the actual body surface. The
// anatomical landmark names are L/R-swapped by the backend to match
// our image-orientation convention, so swapping `left_shoulder` →
// `acromion_left` keeps the cyan dot on the same side it was before.
//
// Backward compatibility: older WHAM rows (pre-PR-8e.0) don't include
// the anatomical keys at all. The resolver falls back to the original
// SMPL joint when the override key is missing or null. Old videos
// render identically to PR-8d.1.
//
// Elbow + ankle: PROBE inventory includes lateral_epicondyle_l/r and
// lateral_malleolus_l/r, and PR-8e.0 emits them. They're not in this
// override map for MVP — the visual difference vs SMPL elbow/ankle
// is small (those H36M joints sit close to the surface already). Add
// entries here in a future PR if user testing shows they matter.
const ANATOMICAL_OVERRIDE: Partial<Record<JointName, string>> = {
  left_shoulder:  'acromion_left',
  right_shoulder: 'acromion_right',
  left_hip:       'greater_trochanter_left',
  right_hip:      'greater_trochanter_right',
};

function resolveJointPos(
  kp: Record<string, { x: number; y: number } | null>,
  name: JointName,
): { x: number; y: number } | null {
  const overrideKey = ANATOMICAL_OVERRIDE[name];
  if (overrideKey) {
    const anatomical = kp[overrideKey];
    if (anatomical) return anatomical;
    // override key present in schema but null this frame (e.g., Z<0.1
    // guard fired for the vertex) — fall through to SMPL joint
  }
  return kp[name] ?? null;
}

// Edges. Color of an edge follows side: L=cyan, R=amber, centerline=white.
// Crossbody (l_shoulder<->r_shoulder, l_hip<->r_hip) is white per spec.
type Edge = readonly [JointName, JointName];

const L_EDGES: readonly Edge[] = [
  ['left_shoulder', 'left_elbow'],
  ['left_elbow',    'left_wrist'],
  ['left_hip',      'left_knee'],
  ['left_knee',     'left_ankle'],
];
const R_EDGES: readonly Edge[] = [
  ['right_shoulder', 'right_elbow'],
  ['right_elbow',    'right_wrist'],
  ['right_hip',      'right_knee'],
  ['right_knee',     'right_ankle'],
];
const CENTER_EDGES: readonly Edge[] = [
  // Crossbody white per spec.
  ['left_shoulder', 'right_shoulder'],
  ['left_hip',      'right_hip'],
  // Spine chain (centerline).
  ['head_crown', 'head'],
  ['head',       'neck'],
  ['neck',       'spine1'],
  ['spine1',     'pelvis'],
];

const POINT_R       = 7;     // points radius 6–8px per spec
const POINT_STROKE  = 2;     // 2px contrast stroke
const BONE_WIDTH    = 3.5;   // 3–4px per spec
const FRAME_OPACITY = 0.85;  // 0.75–0.85 per spec


/** Binary search for the frame with the largest frame_timestamp_ms ≤ t_ms. */
function findFrameAtMs(frames: WhamPoseFrame[], tMs: number): WhamPoseFrame | null {
  if (frames.length === 0) return null;
  if (tMs <= (frames[0].frame_timestamp_ms ?? 0)) return frames[0];
  if (tMs >= (frames[frames.length - 1].frame_timestamp_ms ?? Number.POSITIVE_INFINITY)) {
    return frames[frames.length - 1];
  }
  let lo = 0;
  let hi = frames.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >>> 1;
    const midTs = frames[mid].frame_timestamp_ms ?? Number.POSITIVE_INFINITY;
    if (midTs <= tMs) lo = mid;
    else hi = mid - 1;
  }
  return frames[lo];
}


export function WhamSkeletonOverlay({ timeline, videoEl }: Props) {
  const circleRefs = useRef<Partial<Record<JointName, SVGCircleElement | null>>>({});
  const edgeRefs   = useRef<Partial<Record<string, SVGLineElement | null>>>({});
  const groupRef   = useRef<SVGGElement | null>(null);

  // Stable edge key — `${a}__${b}` joins each pair so the ref keys
  // are unique even if a joint appears in multiple edges.
  const allEdges = useMemo(
    () => [
      ...L_EDGES.map((e) => ({ edge: e, color: COLOR_LEFT })),
      ...R_EDGES.map((e) => ({ edge: e, color: COLOR_RIGHT })),
      ...CENTER_EDGES.map((e) => ({ edge: e, color: COLOR_CENTER })),
    ],
    [],
  );

  useEffect(() => {
    if (!videoEl || !timeline) return;

    const draw = () => {
      const tMs = videoEl.currentTime * 1000;
      const frame = findFrameAtMs(timeline.frames, tMs);

      const group = groupRef.current;
      if (!frame || !frame.fit_ok || !frame.keypoints_2d_projected) {
        // Hide whole skeleton — frame missing or WHAM didn't fit.
        if (group) group.setAttribute('visibility', 'hidden');
        return;
      }
      if (group) group.setAttribute('visibility', 'visible');

      const kp = frame.keypoints_2d_projected;

      // Update joint circles. PR-8e.1: resolveJointPos picks the
      // anatomical surface landmark when available (acromion for
      // shoulder, greater trochanter for hip), else SMPL joint.
      for (const name of ALL_JOINTS) {
        const el = circleRefs.current[name];
        if (!el) continue;
        const p = resolveJointPos(kp, name);
        if (!p) {
          el.setAttribute('visibility', 'hidden');
          continue;
        }
        el.setAttribute('cx', String(p.x));
        el.setAttribute('cy', String(p.y));
        el.setAttribute('visibility', 'visible');
      }

      // Update edge lines. Same resolveJointPos for endpoints so the
      // bone connects the rendered dots (not a mix of anatomical-dot +
      // SMPL-endpoint).
      for (const { edge } of allEdges) {
        const [a, b] = edge;
        const el = edgeRefs.current[`${a}__${b}`];
        if (!el) continue;
        const pa = resolveJointPos(kp, a);
        const pb = resolveJointPos(kp, b);
        if (!pa || !pb) {
          el.setAttribute('visibility', 'hidden');
          continue;
        }
        el.setAttribute('x1', String(pa.x));
        el.setAttribute('y1', String(pa.y));
        el.setAttribute('x2', String(pb.x));
        el.setAttribute('y2', String(pb.y));
        el.setAttribute('visibility', 'visible');
      }
    };

    // rAF loop drives the per-frame redraw against videoEl.currentTime.
    let raf = 0;
    const loop = () => {
      draw();
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    // One-shot syncs on metadata + seeked so the overlay matches even
    // if the rAF loop hasn't ticked yet (e.g., paused at frame 0).
    videoEl.addEventListener('loadedmetadata', draw);
    videoEl.addEventListener('seeked',         draw);
    draw();

    return () => {
      cancelAnimationFrame(raf);
      videoEl.removeEventListener('loadedmetadata', draw);
      videoEl.removeEventListener('seeked',         draw);
    };
  }, [videoEl, timeline, allEdges]);

  return (
    <svg
      className="wham-skeleton-overlay"
      viewBox={`0 0 ${timeline.image_width} ${timeline.image_height}`}
      preserveAspectRatio="xMidYMid meet"
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        // Above the canvas + MediaPipe overlay if both happen to be in
        // the same .sp-vw stack. SwingPlayer hides the MediaPipe path
        // when whamPoseTimeline is present so there's nothing to compete
        // with in practice, but keep z-index explicit.
        zIndex: 25,
        opacity: FRAME_OPACITY,
      }}
    >
      <g ref={groupRef}>
        {/* Edges first so circles paint on top. */}
        {allEdges.map(({ edge, color }) => {
          const [a, b] = edge;
          const key = `${a}__${b}`;
          return (
            <line
              key={key}
              ref={(el) => { edgeRefs.current[key] = el; }}
              stroke={color}
              strokeWidth={BONE_WIDTH}
              strokeLinecap="round"
              visibility="hidden"
            />
          );
        })}
        {/* Joint circles. */}
        {(LEFT_JOINTS as readonly JointName[]).map((name) => (
          <circle
            key={name}
            ref={(el) => { circleRefs.current[name] = el; }}
            r={POINT_R}
            fill={COLOR_LEFT}
            stroke={STROKE}
            strokeWidth={POINT_STROKE}
            visibility="hidden"
          />
        ))}
        {(RIGHT_JOINTS as readonly JointName[]).map((name) => (
          <circle
            key={name}
            ref={(el) => { circleRefs.current[name] = el; }}
            r={POINT_R}
            fill={COLOR_RIGHT}
            stroke={STROKE}
            strokeWidth={POINT_STROKE}
            visibility="hidden"
          />
        ))}
        {(CENTER_JOINTS as readonly JointName[]).map((name) => (
          <circle
            key={name}
            ref={(el) => { circleRefs.current[name] = el; }}
            r={POINT_R}
            fill={COLOR_CENTER}
            stroke={STROKE}
            strokeWidth={POINT_STROKE}
            visibility="hidden"
          />
        ))}
      </g>
    </svg>
  );
}
