/**
 * PR-7c-frontend-v4 — MediaPipe-direct anchor lookup + neck-center
 * helper for the simplified 5-dot coaching overlay.
 *
 * v2: gated lookup (poseAnchorsAtTime, removed in v3)
 * v3: raw lookup + halo/center/guide helpers (computeHeadHalo,
 *     computeShoulderCenter, computeHipCenter — removed in v4)
 * v4: just raw lookup + neck-center helper. 5 dots, no fallback,
 *     no derived geometry beyond a single midpoint shift for neck.
 *
 * Data source: production `pose_timeline_2d` envelope (PoseTimeline
 * from @/types/analysis), already loaded by SwingPlayer.
 *
 * Frame lookup: `interpolatedFrame` from @/lib/disc/frameAt — smooth
 * lerp between bracketing MediaPipe samples.
 *
 * Pure helpers — no React, no DOM, no I/O.
 */

import type {
  CocoKeypointName,
  PoseTimeline,
} from "@/types/analysis";
import { interpolatedFrame } from "@/lib/disc/frameAt";

/**
 * The 4 MediaPipe-direct keypoints v4 reads. Head/neck is derived
 * from these (not a MediaPipe keypoint), so it's not in this list.
 * "nose" no longer used (v3 dropped it for halo; v4 uses derived neck).
 */
export const MEDIAPIPE_ANCHOR_NAMES = [
  "left_shoulder",
  "right_shoulder",
  "left_hip",
  "right_hip",
] as const;

export type MediaPipeAnchorName = (typeof MEDIAPIPE_ANCHOR_NAMES)[number];

/**
 * One raw MediaPipe keypoint. `xy` is null when the source keypoint
 * is null/missing — caller still inspects `confidence` separately.
 */
export interface RawKeypoint {
  xy: readonly [number, number] | null;
  confidence: number;
}

/**
 * Confidence below which v4 hides individual dots. Applied uniformly
 * to the 4 direct anchors. Head/neck uses a separate rule (see
 * `computeNeckCenter` docstring + the overlay).
 */
export const ANCHOR_DOT_CONFIDENCE_MIN = 0.3;

/**
 * Higher confidence threshold used by the head/neck visibility check.
 * Hide neck dot when BOTH shoulder confidences fall below this — the
 * derived midpoint becomes unreliable. 0.5 is intentionally stricter
 * than 0.3 because a derived anchor is sensitive to BOTH inputs.
 */
export const HIGH_CONFIDENCE_THRESHOLD = 0.5;

/**
 * Resolve all 4 MediaPipe-source keypoints at video time `t`.
 * Returns raw coords + confidence with NO internal gating — the
 * caller (CoachingAnchorOverlay) applies the v4 < 0.3 rule.
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

export interface Center {
  x: number;
  y: number;
}

/**
 * Compute v4 neck/head center — single dot at cervical spine top,
 * derived from torso anchors. Replaces v3's stroke ellipse halo.
 *
 *   shoulder_mid_x   = (L_sh.x + R_sh.x) / 2
 *   shoulder_mid_y   = (L_sh.y + R_sh.y) / 2
 *   hip_mid_y        = (L_hip.y + R_hip.y) / 2
 *   torso_height_y   = |shoulder_mid_y - hip_mid_y|
 *   neck.x           = shoulder_mid_x
 *   neck.y           = shoulder_mid_y - torso_height_y * 0.10
 *
 * Returns null when any of the 4 torso source coords is null — caller
 * hides the neck dot in that case. Confidence check (BOTH shoulders
 * < 0.5 → hide) is applied separately at the call site.
 *
 * Rationale: in golf, head/neck stability pre-impact is a key
 * coaching signal. Anchoring at cervical spine (not face center, not
 * head crown) gives a stable reference that doesn't drift on head
 * turn or hair/hat occlusion.
 */
export function computeNeckCenter(
  shL: RawKeypoint, shR: RawKeypoint,
  hipL: RawKeypoint, hipR: RawKeypoint,
): Center | null {
  if (!shL.xy || !shR.xy || !hipL.xy || !hipR.xy) return null;
  const shMidX = (shL.xy[0] + shR.xy[0]) / 2;
  const shMidY = (shL.xy[1] + shR.xy[1]) / 2;
  const hipMidY = (hipL.xy[1] + hipR.xy[1]) / 2;
  const torsoHeight = Math.abs(shMidY - hipMidY);
  return {
    x: shMidX,
    y: shMidY - torsoHeight * 0.10,
  };
}
