/**
 * PR-5: disc geometry types.
 *
 * `DiscParams` is the output of computeShoulderDisc / computeHipDisc
 * and the input to the canvas drawDisc helper in SwingPlayer.tsx.
 *
 * Coordinate convention: cx/cy/rx/ry are in **video native pixel space**
 * (matches pose_timeline_2d.video_width / video_height). SwingPlayer
 * scales to canvas display dims at draw time.
 *
 * Angle convention: see PR-5_DESIGN.md §3 (NORMATIVE) —
 *   angleRad = atan2(L.y - R.y, L.x - R.x), image-y-down, range [-π, +π].
 */

export interface DiscParams {
  /** Center x in video native pixel space (matches pose_timeline_2d.video_width). */
  cx: number;
  /** Center y in video native pixel space. */
  cy: number;
  /**
   * Semi-major axis (along shoulder/hip line), RAW value (= dist / 2).
   * PR-5.1: SwingPlayer overrides this with the per-video DiscAnchor.rx
   * before drawing — disc size is locked to the setup baseline so it
   * doesn't shrink during rotation (only the angle changes).
   */
  rx: number;
  /** Semi-minor axis (perspective foreshortening). */
  ry: number;
  /** Rotation angle in radians, CCW positive in image-y-down coords. See PR-5_DESIGN.md §3 + PR-5.1 §3.A. */
  angleRad: number;
  /** Min confidence of the two endpoint keypoints (0-1). */
  confidence: number;
}

/**
 * PR-5.1: per-video disc size anchor.
 *
 * Initialised once from the earliest valid setup-phase frame (ts < 0.8s
 * and all 4 of shoulders+hips kp confidences ≥ 0.5). Held across the
 * whole video in a SwingPlayer useRef; drawDisc overrides each frame's
 * `rx` with the anchor value so the disc keeps its setup size during
 * rotation. See PR-5.1_DESIGN.md §3.C.
 */
export interface DiscAnchor {
  /** Semi-major axis for the shoulder disc, in video native pixel space. */
  shoulderRx: number;
  /** Semi-major axis for the hip disc, in video native pixel space. */
  hipRx: number;
}
