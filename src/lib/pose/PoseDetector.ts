/**
 * PoseDetector.ts
 *
 * Defines the data structures for pose keypoints and overlay rendering.
 * Real detection is performed server-side via Python/MediaPipe (Phase 2).
 * This module is the client-side consumer of that data.
 */

export interface Keypoint {
  x: number;        // normalized 0-1 (relative to video width)
  y: number;        // normalized 0-1 (relative to video height)
  score?: number;   // confidence 0-1
  name?: string;
}

export interface PoseFrame {
  time: number;           // seconds into video
  keypoints: Keypoint[];  // 17 MoveNet/MediaPipe points
}

export type DetectionStatus = 'idle' | 'loading' | 'analyzing' | 'ready' | 'error';

/**
 * MoveNet keypoint indices (same as MediaPipe Pose Lite 17-point model)
 */
export const KP = {
  NOSE:         0,
  L_EYE:        1,  R_EYE:        2,
  L_EAR:        3,  R_EAR:        4,
  L_SHOULDER:   5,  R_SHOULDER:   6,
  L_ELBOW:      7,  R_ELBOW:      8,
  L_WRIST:      9,  R_WRIST:      10,
  L_HIP:       11,  R_HIP:        12,
  L_KNEE:      13,  R_KNEE:       14,
  L_ANKLE:     15,  R_ANKLE:      16,
} as const;

/**
 * Interpolate between two pose frames
 */
export function interpolatePose(a: PoseFrame, b: PoseFrame, frac: number): Keypoint[] {
  return a.keypoints.map((kpA, i) => {
    const kpB = b.keypoints[i];
    if (!kpB) return kpA;
    return {
      x: kpA.x + (kpB.x - kpA.x) * frac,
      y: kpA.y + (kpB.y - kpA.y) * frac,
      score: (kpA.score ?? 1) * 0.6 + (kpB.score ?? 1) * 0.4,
      name: kpA.name,
    };
  });
}

/**
 * Find the keypoints at a given video time from stored frames.
 */
export function getPoseAt(frames: PoseFrame[], time: number): Keypoint[] | null {
  if (!frames.length) return null;
  if (time <= frames[0].time) return frames[0].keypoints;
  if (time >= frames[frames.length - 1].time) return frames[frames.length - 1].keypoints;

  for (let i = 0; i < frames.length - 1; i++) {
    const a = frames[i], b = frames[i + 1];
    if (time >= a.time && time <= b.time) {
      const frac = (time - a.time) / Math.max(0.001, b.time - a.time);
      return interpolatePose(a, b, frac);
    }
  }
  return frames[frames.length - 1].keypoints;
}
