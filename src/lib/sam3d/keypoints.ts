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

/**
 * COCO 17 keypoint indices (anatomical surface landmarks).
 *
 * Mirrors python/yolo/keypoints.py. These are the canonical body indices
 * for YOLO11-pose, MoveNet, RTMPose, ViTPose, and every COCO-trained model.
 * Unlike SAM 3D Body's MHR anchors, these land on the true acromion / hip
 * joint / etc. — they are annotated on photographs, not derived from a
 * mesh.
 */
export const COCO_KP = {
  NOSE: 0,
  LEFT_EYE: 1,
  RIGHT_EYE: 2,
  LEFT_EAR: 3,
  RIGHT_EAR: 4,
  LEFT_SHOULDER: 5,    // acromion
  RIGHT_SHOULDER: 6,
  LEFT_ELBOW: 7,
  RIGHT_ELBOW: 8,
  LEFT_WRIST: 9,
  RIGHT_WRIST: 10,
  LEFT_HIP: 11,        // hip joint
  RIGHT_HIP: 12,
  LEFT_KNEE: 13,
  RIGHT_KNEE: 14,
  LEFT_ANKLE: 15,
  RIGHT_ANKLE: 16,
} as const;

/**
 * Per-keypoint confidence threshold below which the disc builder ignores
 * the YOLO value and falls back to SAM materialised anchors. Mirrors
 * python/yolo/keypoints.py MIN_CONFIDENCE.
 */
export const MIN_CONFIDENCE = 0.3;

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

  // PR-3: YOLO11-pose anatomical surface landmarks (COCO 17). Preferred
  // over SAM anchors when present + confident. Stays null when YOLO did
  // not run or failed for that phase, in which case the builder falls back
  // to the SAM shoulder_*/hip_* columns above.
  yolo_keypoints_2d: number[][] | null;   // 17 × [x, y, conf]
  yolo_model:        string | null;        // e.g. "yolo11m-pose"
  yolo_inference_ms: number | null;
};

/** Canonical phase ordering for sorting / iteration. */
export const PHASE_ORDER: readonly PhaseName[] = [
  'setup',
  'top',
  'transition',
  'impact',
  'finish',
] as const;
