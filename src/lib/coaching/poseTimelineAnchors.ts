/**
 * PR-7c-frontend-v3 — MediaPipe-direct anchor lookup + geometry
 * helpers for the enhanced coaching overlay.
 *
 * v2 (frozen): exposed gated anchors only (`poseAnchorsAtTime`).
 * v3: caller (CoachingAnchorOverlay) needs RAW coords + confidence
 * to make per-frame visibility decisions (occlusion fallback) AND
 * needs derived geometry (head halo, shoulder/hip midpoints) for the
 * coaching visual layer. Two-tier exposure: raw lookup + geometry
 * helpers; gating happens at the call site.
 *
 * Data source: production `pose_timeline_2d` envelope (PoseTimeline
 * from @/types/analysis), already loaded by SwingPlayer.
 *
 * Frame lookup: `interpolatedFrame` from @/lib/disc/frameAt — smooth
 * lerp between bracketing MediaPipe samples.
 *
 * Phase mapping: caller uses `getCurrentPhase` from
 * @/lib/overlay/playerSync — single source of truth with phase badge.
 *
 * Pure helpers — no React, no DOM, no I/O.
 */

import type {
  CocoKeypointName,
  PoseTimeline,
} from "@/types/analysis";
import { interpolatedFrame } from "@/lib/disc/frameAt";

/**
 * The 5 MediaPipe keypoints v3 reads. "nose" is included for
 * back-compat / debug; v3's coaching layer derives head from
 * shoulders+hips instead (more stable than nose under head turn).
 */
export const MEDIAPIPE_ANCHOR_NAMES = [
  "nose",
  "left_shoulder",
  "right_shoulder",
  "left_hip",
  "right_hip",
] as const;

export type MediaPipeAnchorName = (typeof MEDIAPIPE_ANCHOR_NAMES)[number];

/**
 * One raw MediaPipe keypoint. `xy` is null only when the source
 * keypoint is null/missing — caller still inspects `confidence`
 * separately to decide rendering.
 */
export interface RawKeypoint {
  xy: readonly [number, number] | null;
  confidence: number;
}

/** Confidence below which v3 hides individual dots in non-setup phases. */
export const HIGH_CONFIDENCE_THRESHOLD = 0.5;

/**
 * Resolve all 5 MediaPipe-source keypoints at video time `t`.
 * Returns raw coords + confidence with NO internal gating — the
 * caller (CoachingAnchorOverlay) applies the v3 visibility matrix
 * which depends on phase context the helper doesn't know about.
 *
 * Uses `interpolatedFrame` for smooth lerp between samples.
 * Returns null only when the timeline is empty or `t` is before the
 * first sample.
 */
export function poseRawAnchorsAtTime(
  timeline: PoseTimeline,
  t: number,
): Record<MediaPipeAnchorName, RawKeypoint> | null {
  const frame = interpolatedFrame(timeline, t);
  if (!frame) return null;
  const out = {} as Record<MediaPipeAnchorName, RawKeypoint>;
  for (const name of MEDIAPIPE_ANCHOR_NAMES) {
    const cocoName: CocoKeypointName = name;
    const kp = frame.keypoints[cocoName];
    if (!kp) {
      out[name] = { xy: null, confidence: 0 };
      continue;
    }
    const [x, y, conf] = kp;
    if (
      x === null
      || y === null
      || typeof x !== "number"
      || typeof y !== "number"
    ) {
      out[name] = { xy: null, confidence: conf };
      continue;
    }
    out[name] = { xy: [x, y], confidence: conf };
  }
  return out;
}

// ── Geometry helpers ────────────────────────────────────────────

export interface Center {
  x: number;
  y: number;
}

export interface HaloGeom {
  cx: number;
  cy: number;
  rx: number;
  ry: number;
}

/**
 * Midpoint of L/R shoulders. Returns null when either side has no
 * coord (can't midpoint a single point — caller handles single-side
 * fallback if needed).
 *
 * Confidence is NOT gated here — caller decides whether a low-conf
 * coord is still usable as input to a midpoint.
 */
export function computeShoulderCenter(
  L: RawKeypoint, R: RawKeypoint,
): Center | null {
  if (!L.xy && !R.xy) return null;
  if (!L.xy) return { x: R.xy![0], y: R.xy![1] };
  if (!R.xy) return { x: L.xy[0], y: L.xy[1] };
  return {
    x: (L.xy[0] + R.xy[0]) / 2,
    y: (L.xy[1] + R.xy[1]) / 2,
  };
}

/** Symmetric to computeShoulderCenter for hips. */
export function computeHipCenter(
  L: RawKeypoint, R: RawKeypoint,
): Center | null {
  if (!L.xy && !R.xy) return null;
  if (!L.xy) return { x: R.xy![0], y: R.xy![1] };
  if (!R.xy) return { x: L.xy[0], y: L.xy[1] };
  return {
    x: (L.xy[0] + R.xy[0]) / 2,
    y: (L.xy[1] + R.xy[1]) / 2,
  };
}

/**
 * Compute v3 head halo geometry — a stable face/head visual derived
 * from torso anchors (shoulders + hips), NOT from MediaPipe nose
 * (which drifts under head turn at top phase).
 *
 *   shoulder_mid     = midpoint(L_sh, R_sh)
 *   hip_mid          = midpoint(L_hip, R_hip)
 *   torso_length     = |shoulder_mid.y - hip_mid.y|   (vertical-only)
 *   halo.cx          = shoulder_mid.x
 *   halo.cy          = shoulder_mid.y - torso_length * 0.15
 *   half_sh_width    = |L_sh.x - R_sh.x| / 2
 *   halo.rx          = half_sh_width * 0.7
 *   halo.ry          = half_sh_width * 0.9
 *
 * Returns null when any of the 4 source coords is missing OR when
 * shoulder width collapses to ~0 (degenerate at face-on top, looks
 * wrong as ellipse). Caller hides halo when null.
 */
export function computeHeadHalo(
  shL: RawKeypoint, shR: RawKeypoint,
  hipL: RawKeypoint, hipR: RawKeypoint,
): HaloGeom | null {
  if (!shL.xy || !shR.xy || !hipL.xy || !hipR.xy) return null;
  const shMidX = (shL.xy[0] + shR.xy[0]) / 2;
  const shMidY = (shL.xy[1] + shR.xy[1]) / 2;
  const hipMidY = (hipL.xy[1] + hipR.xy[1]) / 2;
  const torsoLength = Math.abs(shMidY - hipMidY);
  const halfShoulderWidth = Math.abs(shL.xy[0] - shR.xy[0]) / 2;
  if (halfShoulderWidth < 2) return null;   // degenerate
  return {
    cx: shMidX,
    cy: shMidY - torsoLength * 0.15,
    rx: halfShoulderWidth * 0.7,
    ry: halfShoulderWidth * 0.9,
  };
}
