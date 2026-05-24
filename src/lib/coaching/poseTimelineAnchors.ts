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
 * PR-7c-frontend-v8.1: visual-anchor shift configuration.
 *
 * Per-anchor independent ratios — L/R shoulders and L/R hips each have
 * their own UP + OUT pair (8 paired ratios). In DTL view the near vs
 * far shoulder need different offsets, so coupling them like v7/v8 did
 * was a footgun.
 *
 * Head has bipolar UP + OUT (signed shifts from nose, no out_sign
 * computation). Because the head sits near body midline, computing
 * out_sign from a proj that's ~0 causes frame-to-frame sign flips and
 * visible lateral wobble during playback. Bipolar means user-controlled
 * direction: negative HEAD_OUT shifts one way, positive the other.
 *
 * Ratios are still relative to shoulder-to-hip distance (auto-scales
 * across subject sizes + view angles). Algorithm is fixed; only the
 * 10 ratios + 1 stability threshold + 1 toggle below are tunable.
 *
 * TS-safety note (lesson from v8.0.2): declared via an EXPLICIT
 * VisualAnchorConfig interface (not `as const`) so fields are general
 * `number` / `boolean` types. Avoids the literal-type narrowing class
 * of bugs when slider values flow into VISUAL_ANCHOR_CONFIG-shaped
 * objects.
 */
export interface VisualAnchorConfig {
  // SHOULDERS — per-anchor UP + OUT. UP shifts toward acromion peak;
  // OUT shifts laterally away from spine midline.
  LEFT_SHOULDER_UP:    number;
  LEFT_SHOULDER_OUT:   number;
  RIGHT_SHOULDER_UP:   number;
  RIGHT_SHOULDER_OUT:  number;

  // HIPS — per-anchor UP + OUT. UP shifts toward waistband; OUT
  // shifts laterally toward outer visible hip silhouette.
  LEFT_HIP_UP:         number;
  LEFT_HIP_OUT:        number;
  RIGHT_HIP_UP:        number;
  RIGHT_HIP_OUT:       number;

  // HEAD — BIPOLAR UP + OUT (signed, no out_sign computation).
  // Negative UP = down toward chin; positive UP = up toward forehead.
  // Negative OUT = lateral one side; positive OUT = other side.
  // 0.00 + 0.00 → head dot sits exactly at MediaPipe nose.
  HEAD_UP:             number;
  HEAD_OUT:            number;

  // META — v7 still honors these.
  HEAD_USE_NOSE:        boolean;  // future-iteration knob; currently unused
                                  // (v7 always uses nose as head base)
  MIN_BODY_AXIS_LEN_PX: number;   // degenerate-pose threshold
}

/**
 * PR-7c-frontend-v8.1: override type used by the in-browser tuning
 * panel + the overlay in tune mode. Production callers omit it and
 * use VISUAL_ANCHOR_CONFIG defaults.
 *
 * Now a simple `Partial<VisualAnchorConfig>` — the interface above
 * uses general types so Partial works as expected (the v8.0.2
 * literal-narrowing workaround is no longer needed).
 */
export type VisualAnchorConfigOverride = Partial<VisualAnchorConfig>;

export const VISUAL_ANCHOR_CONFIG: VisualAnchorConfig = {
  // SHOULDERS — v7 first-pass values, same UP/OUT for L+R (v8.1 default;
  // user tunes per-anchor in the panel).
  LEFT_SHOULDER_UP:    0.10,
  LEFT_SHOULDER_OUT:   0.05,
  RIGHT_SHOULDER_UP:   0.10,
  RIGHT_SHOULDER_OUT:  0.05,

  // HIPS — v7 first-pass values.
  LEFT_HIP_UP:         0.04,
  LEFT_HIP_OUT:        0.06,
  RIGHT_HIP_UP:        0.04,
  RIGHT_HIP_OUT:       0.06,

  // HEAD — bipolar, default 0 + 0 → head dot = MediaPipe nose direct.
  HEAD_UP:             0.00,
  HEAD_OUT:            0.00,

  HEAD_USE_NOSE:        true,
  MIN_BODY_AXIS_LEN_PX: 30,
};

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
  // PR-7c-frontend-v8: optional ratio overrides for the in-browser
  // tuning panel (?tune=anchors). Production callers omit this arg and
  // use VISUAL_ANCHOR_CONFIG. The tuning panel + overlay in tune mode
  // both pass live slider values via this arg, so the shifted dots
  // animate in real time as the user drags ratios.
  //
  // v8.1: override type is now a simple Partial<VisualAnchorConfig>
  // (the interface uses general types, no `as const` literal narrowing).
  overrideConfig?: VisualAnchorConfigOverride,
): VisualAnchors {
  const C = overrideConfig
    ? { ...VISUAL_ANCHOR_CONFIG, ...overrideConfig }
    : VISUAL_ANCHOR_CONFIG;
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
  if (spine_len < C.MIN_BODY_AXIS_LEN_PX) {
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

  // v8.1: bipolar head shift — applies HEAD_UP / HEAD_OUT directly
  // from nose in body-axis frame, with NO out_sign computation. Avoids
  // the lateral-wobble bug that proj-sign-based shift would create at
  // head positions sitting near the body midline (proj ≈ 0).
  // Negative HEAD_OUT → one direction, positive → the other.
  // HEAD_UP=HEAD_OUT=0 → head dot = MediaPipe nose exactly.
  const headXY: readonly [number, number] | null = nose.xy
    ? [
        nose.xy[0]
          + up_x   * spine_len * C.HEAD_UP
          + perp_x * spine_len * C.HEAD_OUT,
        nose.xy[1]
          + up_y   * spine_len * C.HEAD_UP
          + perp_y * spine_len * C.HEAD_OUT,
      ]
    : null;

  return {
    left_shoulder: {
      xy: shift(
        left_shoulder.xy[0], left_shoulder.xy[1],
        sh_mid_x, sh_mid_y,
        C.LEFT_SHOULDER_UP, C.LEFT_SHOULDER_OUT,
      ),
      confidence: left_shoulder.confidence,
    },
    right_shoulder: {
      xy: shift(
        right_shoulder.xy[0], right_shoulder.xy[1],
        sh_mid_x, sh_mid_y,
        C.RIGHT_SHOULDER_UP, C.RIGHT_SHOULDER_OUT,
      ),
      confidence: right_shoulder.confidence,
    },
    left_hip: {
      xy: shift(
        left_hip.xy[0], left_hip.xy[1],
        hip_mid_x, hip_mid_y,
        C.LEFT_HIP_UP, C.LEFT_HIP_OUT,
      ),
      confidence: left_hip.confidence,
    },
    right_hip: {
      xy: shift(
        right_hip.xy[0], right_hip.xy[1],
        hip_mid_x, hip_mid_y,
        C.RIGHT_HIP_UP, C.RIGHT_HIP_OUT,
      ),
      confidence: right_hip.confidence,
    },
    head: { xy: headXY, confidence: nose.confidence },
  };
}
