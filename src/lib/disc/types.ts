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
  /** Semi-major axis (along shoulder/hip line). */
  rx: number;
  /** Semi-minor axis (perspective foreshortening). */
  ry: number;
  /** Rotation angle in radians, CCW positive in image-y-down coords. See PR-5_DESIGN.md §3. */
  angleRad: number;
  /** Min confidence of the two endpoint keypoints (0-1). */
  confidence: number;
}
