/**
 * COCO 17 keypoint name list + canonical skeleton edges.
 *
 * Mirrors python/pose_timeline.py COCO_NAMES (same order, same spelling).
 * Used by:
 *   - src/components/SkeletonOverlay.tsx — render dots + edges
 *   - future PR-5+ overlays that need to iterate over the 17 names
 */

import type { CocoKeypointName } from '@/types/analysis';

/**
 * All 17 names in canonical COCO order. Order matters when iterating —
 * matches the Python-side COCO_NAMES tuple in python/pose_timeline.py.
 */
export const COCO_KEYPOINT_NAMES: readonly CocoKeypointName[] = [
  'nose',
  'left_eye', 'right_eye',
  'left_ear', 'right_ear',
  'left_shoulder', 'right_shoulder',
  'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist',
  'left_hip', 'right_hip',
  'left_knee', 'right_knee',
  'left_ankle', 'right_ankle',
] as const;

/**
 * Canonical skeleton edges (16 lines). Each pair is (from, to) keypoint
 * names. Chosen to look clean on screen rather than mirror COCO's
 * official skeleton verbatim:
 *   - Head: ear—eye—nose pairs (4 lines)
 *   - Torso: shoulders—hips quad (4 lines: shoulder line, hip line,
 *     2 shoulder→hip connectors)
 *   - Arms: 4 lines
 *   - Legs: 4 lines
 */
export const COCO_SKELETON_EDGES: readonly (readonly [CocoKeypointName, CocoKeypointName])[] = [
  // Head
  ['left_eye', 'nose'], ['right_eye', 'nose'],
  ['left_ear', 'left_eye'], ['right_ear', 'right_eye'],
  // Torso
  ['left_shoulder', 'right_shoulder'],
  ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'],
  ['left_hip', 'right_hip'],
  // Arms
  ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'],
  // Legs
  ['left_hip', 'left_knee'], ['left_knee', 'left_ankle'],
  ['right_hip', 'right_knee'], ['right_knee', 'right_ankle'],
] as const;
