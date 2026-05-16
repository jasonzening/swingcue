/**
 * SAM 3D Body MHR keypoint indices and row types.
 *
 * Mirrors python/sam3d/keypoints.py — body anchor values (7-20, 21, 42, 63-69)
 * are verified 2026-05-16 via scripts/fal_test.py + scripts/fal_inspect.py.
 * Head-cluster indices (0-6) are listed here for completeness per the 70-kp
 * MHR layout but were not separately verified; only the body anchors below
 * are used by sparsePhaseOverlay in this PR.
 */

export const SAM3D_KP = {
  // Head + face cluster (0-6) — not separately verified, listed for parity
  NOSE: 0,
  HEAD_TOP: 1,
  CHIN: 5,
  NECK_BASE: 6,

  // Shoulders (verified — Z-disambiguated in finish frame)
  LEFT_SHOULDER: 7,   // acromion, target-side  (Z=+0.041 in finish test)
  RIGHT_SHOULDER: 8,  // acromion, trail-side   (Z=-0.238 in finish test)

  // Hips
  LEFT_HIP: 9,
  RIGHT_HIP: 10,

  // Knees, ankles, feet
  LEFT_KNEE: 11,
  RIGHT_KNEE: 12,
  LEFT_ANKLE: 13,
  RIGHT_ANKLE: 14,
  LEFT_TOE: 15,
  LEFT_TOE_OUTER: 16,
  LEFT_HEEL: 17,
  RIGHT_TOE: 18,
  RIGHT_TOE_OUTER: 19,
  RIGHT_HEEL: 20,

  // Wrists (assumed = hand cluster origins; validate later)
  LEFT_WRIST: 21,
  RIGHT_WRIST: 42,

  // Extra body detail
  LEFT_DELTOID: 63,
  RIGHT_DELTOID: 64,
  LEFT_CLAVICLE: 65,
  RIGHT_CLAVICLE: 66,
  NECK: 67,
  STERNUM: 68,
  THROAT: 69,
} as const;

export type PhaseName = 'setup' | 'top' | 'transition' | 'impact' | 'finish';
export type FalStatus = 'uploaded' | 'processing' | 'completed' | 'failed';

/**
 * One row of pose_3d_phases (Supabase). Null fields use `| null` (not
 * optional) so destructuring forces explicit null checks — `x !== undefined`
 * would silently miss a JSON-null coming back from PostgREST.
 */
export type PoseRow = {
  phase_name: PhaseName;
  fal_status: FalStatus;
  frame_idx: number;
  frame_timestamp_ms: number | null;

  keypoints_2d: number[][];   // length 70, each [x, y] in source-image px
  keypoints_3d: number[][];   // length 70, each [x, y, z] in MHR camera space
  focal_length: number;
  bbox: [number, number, number, number] | null;
  mhr_params: Record<string, unknown> | null;
  glb_url: string | null;

  image_width: number;
  image_height: number;

  shoulder_left_x:  number | null;
  shoulder_left_y:  number | null;
  shoulder_right_x: number | null;
  shoulder_right_y: number | null;
  hip_left_x:       number | null;
  hip_left_y:       number | null;
  hip_right_x:      number | null;
  hip_right_y:      number | null;
};

/** Canonical phase ordering for sorting / iteration. */
export const PHASE_ORDER: readonly PhaseName[] = [
  'setup',
  'top',
  'transition',
  'impact',
  'finish',
] as const;
