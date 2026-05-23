/**
 * PR-7c-frontend-v7 — body-axis-relative geometric correction.
 *
 * The 4 MediaPipe shoulder/hip COCO keypoints are interior anatomical
 * joint centers (glenohumeral, hip socket) — by definition they sit
 * INSIDE the visible body silhouette. Rendered raw, dots fall on
 * chest / crotch / interior body instead of the outer silhouette
 * coaches reason about (acromion peak, outer hip line).
 *
 * v7 introduces `computeVisualAnchors`: a deterministic, phase-
 * invariant geometric shift in a body-axis-relative frame. 4 ratios
 * centralized in VISUAL_ANCHOR_CONFIG drive the entire correction.
 * Shifts are applied per-frame, oriented to current body axis, so
 * they auto-rotate with pose — no learned matrix, no per-phase
 * variance (avoids the PR-7a failure mode).
 *
 * v6 → v7 changes:
 *   - MEDIAPIPE_ANCHOR_NAMES: added "nose" (head = nose direct, v4 neck
 *     derivation deleted)
 *   - REMOVED: Center type, computeNeckCenter helper,
 *     HIGH_CONFIDENCE_THRESHOLD constant (no derived anchor anymore)
 *   - ADDED: VISUAL_ANCHOR_CONFIG, VisualAnchors interface,
 *     computeVisualAnchors helper
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
 * The 5 MediaPipe keypoints v7 reads. Head/nose is read directly
 * (no longer derived from shoulders/hips). The 4 torso joints feed
 * the body-axis shift in `computeVisualAnchors`.
 */
export const MEDIAPIPE_ANCHOR_NAMES = [
  "left_shoulder",
  "right_shoulder",
  "left_hip",
  "right_hip",
  "nose",
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
 * Confidence below which v7 hides individual dots. Applied uniformly
 * to all 5 anchors (4 shifted torso joints + head/nose).
 */
export const ANCHOR_DOT_CONFIDENCE_MIN = 0.3;

/**
 * PR-7c-frontend-v7: visual-anchor shift configuration.
 *
 * All tuning happens here. Ratios are relative to shoulder-to-hip
 * distance (the natural per-frame body scale) so a single set of
 * constants works across subject sizes and view angles (face-on vs
 * down-the-line). To iterate visuals, change the 4 ratio numbers
 * below — the algorithm itself is fixed.
 *
 * Defaults are the v7 first-pass values (tuned against b32e0f21
 * frame-by-frame visual review on 2026-05-23).
 */
export const VISUAL_ANCHOR_CONFIG = {
  // SHOULDERS — MediaPipe glenohumeral (interior) → visible acromion peak.
  // Up component dominates (the main correction is upward to the bony
  // peak). Out component small (lateral correction is minor — joint
  // sits roughly under the visible shoulder horizontally).
  SHOULDER_UP_RATIO:  0.10,
  SHOULDER_OUT_RATIO: 0.05,

  // HIPS — MediaPipe hip socket (interior) → outer visible hip silhouette
  // at waistband level. Out component dominates (the main correction is
  // lateral). Small up component pulls toward waistband, not crotch.
  HIP_UP_RATIO:  0.04,
  HIP_OUT_RATIO: 0.06,

  // HEAD — direct MediaPipe nose. v7 ships this; halo fallback for
  // side-profile or bent-over poses deferred to v8 if needed.
  HEAD_USE_NOSE: true,

  // STABILITY — below this body axis length (in image px), the pose
  // is degenerate (collapsed at finish, occluded torso, mis-detection).
  // Return raw anchors without shift in that case to avoid amplifying
  // a small-noise body-axis into wild visual drift.
  MIN_BODY_AXIS_LEN_PX: 30,
} as const;

/**
 * The output of `computeVisualAnchors`. All 5 fields are RawKeypoint
 * so the overlay can use a single uniform `applyDot` helper for all
 * dots — head no longer needs a special-case branch.
 */
export interface VisualAnchors {
  left_shoulder:  RawKeypoint;
  right_shoulder: RawKeypoint;
  left_hip:       RawKeypoint;
  right_hip:      RawKeypoint;
  head:           RawKeypoint;
}

/**
 * Resolve all 5 MediaPipe-source keypoints at video time `t`.
 * Returns raw coords + confidence with NO internal gating — the
 * caller (CoachingAnchorOverlay) applies the v7 < 0.3 rule.
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

/**
 * PR-7c-frontend-v7: shift MediaPipe interior joint centers outward
 * toward the visible body silhouette in a body-axis-relative frame.
 *
 * Algorithm (per-frame, phase-invariant by construction):
 *
 *   1. shoulder_mid = midpoint(left_shoulder, right_shoulder)
 *      hip_mid      = midpoint(left_hip, right_hip)
 *      spine_vec    = hip_mid - shoulder_mid          (current orientation)
 *      spine_len    = |spine_vec|
 *
 *   2. body_axis = spine_vec / spine_len              (down along spine)
 *      up_axis   = -body_axis                         (toward head)
 *      perp_axis = 90° CW rotation of body_axis       (lateral)
 *
 *   3. For each shoulder/hip joint at point p (with mid = shoulder_mid
 *      or hip_mid respectively):
 *        v        = p - mid
 *        proj     = v · perp_axis              (which side of midline?)
 *        out_sign = sign(proj)                 (preserves L/R direction)
 *        shifted  = p + up_axis  * (spine_len * up_ratio)
 *                     + perp_axis * out_sign * (spine_len * out_ratio)
 *
 * Edge cases:
 *   - any of 4 torso joints xy === null     → return raw torso + nose
 *     (don't synthesize a silhouette we can't compute honestly)
 *   - spine_len < MIN_BODY_AXIS_LEN_PX      → degenerate pose, skip
 *     shift (return raw to avoid amplifying small-vector noise)
 *
 * Head: returned as raw MediaPipe nose. Not derived. Confidence
 * gating at the overlay (`< 0.3 → hide`) applies the same as the
 * 4 shifted joints.
 *
 * Phase invariance: body_axis is computed from CURRENT pose every
 * frame. Shifts rotate with the body automatically. No learned
 * per-phase offsets → cannot replicate PR-7a's phase-overshoot bug.
 */
export function computeVisualAnchors(
  raw: Record<MediaPipeAnchorName, RawKeypoint>,
): VisualAnchors {
  const { left_shoulder, right_shoulder, left_hip, right_hip, nose } = raw;

  // Edge case 1: any torso joint missing — pass-through (head = nose).
  if (
    !left_shoulder.xy || !right_shoulder.xy
    || !left_hip.xy || !right_hip.xy
  ) {
    return { left_shoulder, right_shoulder, left_hip, right_hip, head: nose };
  }

  const sh_mid_x  = (left_shoulder.xy[0] + right_shoulder.xy[0]) / 2;
  const sh_mid_y  = (left_shoulder.xy[1] + right_shoulder.xy[1]) / 2;
  const hip_mid_x = (left_hip.xy[0]      + right_hip.xy[0]     ) / 2;
  const hip_mid_y = (left_hip.xy[1]      + right_hip.xy[1]     ) / 2;

  const spine_x = hip_mid_x - sh_mid_x;
  const spine_y = hip_mid_y - sh_mid_y;
  const spine_len = Math.hypot(spine_x, spine_y);

  // Edge case 2: degenerate body axis — pass-through.
  if (spine_len < VISUAL_ANCHOR_CONFIG.MIN_BODY_AXIS_LEN_PX) {
    return { left_shoulder, right_shoulder, left_hip, right_hip, head: nose };
  }

  const body_x = spine_x / spine_len;
  const body_y = spine_y / spine_len;
  const up_x = -body_x;
  const up_y = -body_y;
  // 90° CW rotation: (x, y) → (-y, x)
  const perp_x = -body_y;
  const perp_y = body_x;

  const shift = (
    px: number, py: number,
    mid_x: number, mid_y: number,
    up_ratio: number, out_ratio: number,
  ): readonly [number, number] => {
    const vx = px - mid_x;
    const vy = py - mid_y;
    const proj = vx * perp_x + vy * perp_y;
    const out_sign = proj >= 0 ? 1 : -1;
    const up_shift  = spine_len * up_ratio;
    const out_shift = spine_len * out_ratio;
    return [
      px + up_x * up_shift + perp_x * out_sign * out_shift,
      py + up_y * up_shift + perp_y * out_sign * out_shift,
    ];
  };

  const C = VISUAL_ANCHOR_CONFIG;
  return {
    left_shoulder: {
      xy: shift(
        left_shoulder.xy[0], left_shoulder.xy[1],
        sh_mid_x, sh_mid_y,
        C.SHOULDER_UP_RATIO, C.SHOULDER_OUT_RATIO,
      ),
      confidence: left_shoulder.confidence,
    },
    right_shoulder: {
      xy: shift(
        right_shoulder.xy[0], right_shoulder.xy[1],
        sh_mid_x, sh_mid_y,
        C.SHOULDER_UP_RATIO, C.SHOULDER_OUT_RATIO,
      ),
      confidence: right_shoulder.confidence,
    },
    left_hip: {
      xy: shift(
        left_hip.xy[0], left_hip.xy[1],
        hip_mid_x, hip_mid_y,
        C.HIP_UP_RATIO, C.HIP_OUT_RATIO,
      ),
      confidence: left_hip.confidence,
    },
    right_hip: {
      xy: shift(
        right_hip.xy[0], right_hip.xy[1],
        hip_mid_x, hip_mid_y,
        C.HIP_UP_RATIO, C.HIP_OUT_RATIO,
      ),
      confidence: right_hip.confidence,
    },
    head: nose,
  };
}
