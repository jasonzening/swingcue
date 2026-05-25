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
 * centralized in DEFAULT_RATIOS / VIDEO_KEYFRAMES drive the entire correction.
 * Shifts are applied per-frame, oriented to current body axis, so
 * they auto-rotate with pose — no learned matrix, no per-phase
 * variance (avoids the PR-7a failure mode).
 *
 * v6 → v7 changes:
 *   - MEDIAPIPE_ANCHOR_NAMES: added "nose" (head = nose direct, v4 neck
 *     derivation deleted)
 *   - REMOVED: Center type, computeNeckCenter helper,
 *     HIGH_CONFIDENCE_THRESHOLD constant (no derived anchor anymore)
 *   - ADDED: DEFAULT_RATIOS / VIDEO_KEYFRAMES, VisualAnchors interface,
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
 * of bugs when slider values flow into DEFAULT_RATIOS / VIDEO_KEYFRAMES-shaped
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
 * fall back to per-video keyframe interpolation (v9) or DEFAULT_RATIOS.
 *
 * Simple `Partial<VisualAnchorConfig>` — the interface uses general
 * types so Partial works as expected.
 */
export type VisualAnchorConfigOverride = Partial<VisualAnchorConfig>;

/**
 * PR-7c-frontend-v9: production fallback ratios. Used when a video has
 * no entry in VIDEO_KEYFRAMES (= every video except the few Jason has
 * hand-tuned). Matches v8.1's single-set defaults.
 */
export const DEFAULT_RATIOS: VisualAnchorConfig = {
  // SHOULDERS — v7 first-pass values.
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
 * PR-7c-frontend-v9: one keyframe = a saved (frame_idx, ratios) pair.
 * Production interpolation lerps between bracketing keyframes per frame.
 */
export interface AnchorKeyframe {
  frame_idx: number;
  ratios: VisualAnchorConfig;
}

/**
 * PR-7c-frontend-v9: hand-tuned per-video keyframes. Indexed by sampled
 * `frame_idx` (the position in poseTimeline.frames, matches the
 * AnchorTuningPanel's findClosestFrameIdx output that Jason copies in
 * snapshot blocks).
 *
 * Why per-video instead of phase-invariant ratios:
 *   v8.1 assumed ratios would generalize across phases via body-axis-
 *   relative scaling. Jason's 5-snapshot tuning data (b32e0f21,
 *   2026-05-23) showed ratios varying 3-8x across phases
 *   (LEFT_SHOULDER_UP 0.030 → 0.105, RIGHT_HIP_UP 0.020 → 0.160).
 *   Body-axis frame is not invariant enough — perspective + arm
 *   occlusion + body deformation across the swing require per-frame
 *   tuning. v9 lets Jason save N keyframes and interpolates between.
 *
 * Videos NOT in this map fall through to DEFAULT_RATIOS (single-set,
 * matches v8.1 production behavior — no regression).
 */
export const VIDEO_KEYFRAMES: Record<string, AnchorKeyframe[]> = {
  // b32e0f21 — Jason's final refined 9 keyframes (v9.3, 2026-05-24).
  // Closes PR-7c-frontend. Production overlay lerps these 9 points
  // across the full swing for smooth phase-aware coaching anchors.
  //
  // Key refinements from v9.2 baseline:
  //   - f=21:  L_SH_OUT 0.145 → 0.195, R_HIP_UP 0.015 → 0.100
  //   - f=34:  L_HIP_UP 0.135 → 0.015 (major), R_HIP_OUT 0.065 → 0.015
  //   - f=45:  L_SH_UP 0 → 0.050, L_HIP_UP 0.185 → 0.205
  //   - f=47:  R_SH_OUT 0 → 0.055, L_HIP_OUT 0 → 0.175 (major)
  //   - f=49:  most values reduced for smoother lerp into f=54
  //   - f=54:  R_HIP swap (UP 0.080 → 0.010, OUT 0.030 → 0.065)
  'b32e0f21-2656-473c-aa87-e1eaf6e1221f': [
    {
      frame_idx: 0,
      ratios: {
        ...DEFAULT_RATIOS,
        LEFT_SHOULDER_UP:  0.070, LEFT_SHOULDER_OUT:  0.000,
        RIGHT_SHOULDER_UP: 0.000, RIGHT_SHOULDER_OUT: 0.000,
        LEFT_HIP_UP:       0.090, LEFT_HIP_OUT:       0.000,
        RIGHT_HIP_UP:      0.025, RIGHT_HIP_OUT:      0.000,
        HEAD_UP:           0.310, HEAD_OUT:           0.185,
      },
    },
    {
      frame_idx: 21,
      ratios: {
        ...DEFAULT_RATIOS,
        LEFT_SHOULDER_UP:  0.000, LEFT_SHOULDER_OUT:  0.195,
        RIGHT_SHOULDER_UP: 0.000, RIGHT_SHOULDER_OUT: 0.055,
        LEFT_HIP_UP:       0.115, LEFT_HIP_OUT:       0.000,
        RIGHT_HIP_UP:      0.100, RIGHT_HIP_OUT:      0.035,
        HEAD_UP:           0.385, HEAD_OUT:           0.155,
      },
    },
    {
      frame_idx: 34,
      ratios: {
        ...DEFAULT_RATIOS,
        LEFT_SHOULDER_UP:  0.055, LEFT_SHOULDER_OUT:  0.055,
        RIGHT_SHOULDER_UP: 0.010, RIGHT_SHOULDER_OUT: 0.080,
        LEFT_HIP_UP:       0.015, LEFT_HIP_OUT:       0.000,
        RIGHT_HIP_UP:      0.015, RIGHT_HIP_OUT:      0.015,
        HEAD_UP:           0.250, HEAD_OUT:           0.055,
      },
    },
    {
      frame_idx: 45,
      ratios: {
        ...DEFAULT_RATIOS,
        LEFT_SHOULDER_UP:  0.050, LEFT_SHOULDER_OUT:  0.030,
        RIGHT_SHOULDER_UP: 0.000, RIGHT_SHOULDER_OUT: 0.000,
        LEFT_HIP_UP:       0.205, LEFT_HIP_OUT:       0.000,
        RIGHT_HIP_UP:      0.005, RIGHT_HIP_OUT:      0.000,
        HEAD_UP:           0.390, HEAD_OUT:           0.090,
      },
    },
    {
      frame_idx: 47,
      ratios: {
        ...DEFAULT_RATIOS,
        LEFT_SHOULDER_UP:  0.095, LEFT_SHOULDER_OUT:  0.165,
        RIGHT_SHOULDER_UP: 0.000, RIGHT_SHOULDER_OUT: 0.055,
        LEFT_HIP_UP:       0.170, LEFT_HIP_OUT:       0.175,
        RIGHT_HIP_UP:      0.080, RIGHT_HIP_OUT:      0.040,
        HEAD_UP:           0.415, HEAD_OUT:           0.085,
      },
    },
    {
      frame_idx: 49,
      ratios: {
        ...DEFAULT_RATIOS,
        LEFT_SHOULDER_UP:  0.050, LEFT_SHOULDER_OUT:  0.045,
        RIGHT_SHOULDER_UP: 0.000, RIGHT_SHOULDER_OUT: 0.000,
        LEFT_HIP_UP:       0.140, LEFT_HIP_OUT:       0.125,
        RIGHT_HIP_UP:      0.035, RIGHT_HIP_OUT:      0.015,
        HEAD_UP:           0.415, HEAD_OUT:           0.095,
      },
    },
    {
      frame_idx: 54,
      ratios: {
        ...DEFAULT_RATIOS,
        LEFT_SHOULDER_UP:  0.110, LEFT_SHOULDER_OUT:  0.100,
        RIGHT_SHOULDER_UP: 0.000, RIGHT_SHOULDER_OUT: 0.000,
        LEFT_HIP_UP:       0.140, LEFT_HIP_OUT:       0.125,
        RIGHT_HIP_UP:      0.010, RIGHT_HIP_OUT:      0.065,
        HEAD_UP:           0.360, HEAD_OUT:          -0.025,
      },
    },
    {
      frame_idx: 55,
      ratios: {
        ...DEFAULT_RATIOS,
        LEFT_SHOULDER_UP:  0.065, LEFT_SHOULDER_OUT:  0.125,
        RIGHT_SHOULDER_UP: 0.000, RIGHT_SHOULDER_OUT: 0.025,
        LEFT_HIP_UP:       0.060, LEFT_HIP_OUT:       0.095,
        RIGHT_HIP_UP:      0.035, RIGHT_HIP_OUT:      0.070,
        HEAD_UP:           0.290, HEAD_OUT:          -0.095,
      },
    },
    {
      frame_idx: 58,
      ratios: {
        ...DEFAULT_RATIOS,
        LEFT_SHOULDER_UP:  0.085, LEFT_SHOULDER_OUT:  0.035,
        RIGHT_SHOULDER_UP: 0.000, RIGHT_SHOULDER_OUT: 0.000,
        LEFT_HIP_UP:       0.110, LEFT_HIP_OUT:       0.100,
        RIGHT_HIP_UP:      0.045, RIGHT_HIP_OUT:      0.075,
        HEAD_UP:           0.155, HEAD_OUT:          -0.200,
      },
    },
  ],
};

/**
 * PR-7c-frontend-v9: find the sampled-frame index whose `ts` is closest
 * to time `t`. O(n) linear scan — pose timelines are <200 frames so
 * this is trivial. Shared between CoachingAnchorOverlay (rAF) and
 * AnchorTuningPanel (debug readout) so production interpolation indexes
 * match what the panel surfaces (and what Jason copied as `frame_idx`
 * in snapshot blocks).
 */
export function findClosestFrameIdx(timeline: PoseTimeline, t: number): number {
  const frames = timeline.frames;
  if (frames.length === 0) return 0;
  let bestIdx = 0;
  let bestDelta = Math.abs(frames[0].ts - t);
  for (let i = 1; i < frames.length; i++) {
    const d = Math.abs(frames[i].ts - t);
    if (d < bestDelta) {
      bestDelta = d;
      bestIdx = i;
    }
  }
  return bestIdx;
}

/**
 * PR-7c-frontend-v9: linearly interpolate the 10 ratios between
 * bracketing keyframes at `frameIdx`. Non-numeric fields (HEAD_USE_NOSE
 * boolean, etc.) snap to the lower keyframe's value (no fractional
 * boolean semantics).
 *
 * Edge cases:
 *   - empty keyframes array → DEFAULT_RATIOS (matches videos with no
 *     hand-tuning)
 *   - frameIdx before first keyframe → first keyframe's ratios (clamp)
 *   - frameIdx after last keyframe → last keyframe's ratios (clamp)
 *   - frameIdx exactly at a keyframe → that keyframe's ratios
 *
 * Keyframes are sorted defensively (caller may pass unsorted arrays
 * from the tuning panel's edit/delete operations).
 */
export function getRatiosAtFrame(
  frameIdx: number,
  keyframes: AnchorKeyframe[],
): VisualAnchorConfig {
  if (keyframes.length === 0) return DEFAULT_RATIOS;
  const sorted = [...keyframes].sort((a, b) => a.frame_idx - b.frame_idx);
  if (frameIdx <= sorted[0].frame_idx) return sorted[0].ratios;
  if (frameIdx >= sorted[sorted.length - 1].frame_idx) {
    return sorted[sorted.length - 1].ratios;
  }
  for (let i = 0; i < sorted.length - 1; i++) {
    if (sorted[i + 1].frame_idx > frameIdx) {
      const k1 = sorted[i];
      const k2 = sorted[i + 1];
      const t = (frameIdx - k1.frame_idx) / (k2.frame_idx - k1.frame_idx);
      // Typed lerp — Object.fromEntries + single cast at the boundary.
      const entries = (Object.keys(k1.ratios) as Array<keyof VisualAnchorConfig>)
        .map((key): [keyof VisualAnchorConfig, number | boolean] => {
          const v1 = k1.ratios[key];
          const v2 = k2.ratios[key];
          if (typeof v1 === 'number' && typeof v2 === 'number') {
            return [key, v1 + (v2 - v1) * t];
          }
          return [key, v1];
        });
      // Cast through unknown — Object.fromEntries returns generic
      // Record<string, ...>; we know structurally we've covered every
      // VisualAnchorConfig key because `entries` was built from
      // `Object.keys(k1.ratios) as Array<keyof VisualAnchorConfig>`.
      return Object.fromEntries(entries) as unknown as VisualAnchorConfig;
    }
  }
  return sorted[sorted.length - 1].ratios;
}

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
 *
 * v10 clamp audit (2026-05-24): grep-verified ZERO ratio-value clamps
 * exist in this file. Math.abs in findClosestFrameIdx is a distance
 * compare; Math.hypot computes spine_len; spine_len < MIN_BODY_AXIS_LEN_PX
 * is a degenerate-pose stability guard (kept by spec); out_sign is a
 * direction toggle for paired anchors (algorithm correctness, not
 * magnitude clamp). Slider values 0 → 1.20 propagate end-to-end via
 * `spine_len * C.X_RATIO` to coordinate shifts with no internal cap.
 */
export function computeVisualAnchors(
  raw: Record<MediaPipeAnchorName, RawKeypoint>,
  // PR-7c-frontend-v9: per-frame keyframe interpolation.
  //
  // Production (overrideConfig undefined): look up VIDEO_KEYFRAMES[videoId]
  // and interpolate via getRatiosAtFrame(frameIdx, ...). Videos with
  // no keyframes fall back to DEFAULT_RATIOS.
  //
  // Tune mode (overrideConfig defined): use the override directly as
  // a single set across all frames — matches v8.1 tune-mode UX. The
  // panel's keyframes array is the source of saved snapshots; the
  // tune-mode overlay reflects what the user is currently dragging.
  frameIdx: number,
  videoId: string,
  overrideConfig?: VisualAnchorConfigOverride,
): VisualAnchors {
  const C: VisualAnchorConfig = overrideConfig
    ? { ...DEFAULT_RATIOS, ...overrideConfig }
    : getRatiosAtFrame(frameIdx, VIDEO_KEYFRAMES[videoId] ?? []);
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
