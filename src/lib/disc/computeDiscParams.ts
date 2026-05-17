/**
 * PR-5: per-frame shoulder + hip disc geometry from PR-4 keypoints.
 *
 * Pure functions — no React, no DOM. Called from the canvas rAF loop
 * in SwingPlayer.tsx (and may be reused by future PR-5b/c features).
 *
 * Angle convention (NORMATIVE, see PR-5_DESIGN.md §3):
 *   angleRad = atan2(L.y - R.y, L.x - R.x)
 *   Setup  (L right of R, level):       angle ≈  0
 *   Top    (L above R, right-hand RH):  angle ≈ -π/2
 *   Finish (L below R):                 angle ≈ +π/2
 *
 * Coordinate output: cx/cy/rx/ry are in video native pixel space.
 */

import type { PoseFrame } from '@/types/analysis';
import type { DiscParams } from './types';

const MIN_CONFIDENCE = 0.3;
const PERSPECTIVE_RY_RATIO = 0.2; // ry = rx * 0.2 (v1; PR-5b may modulate by angle)

/**
 * Build a DiscParams from two endpoint keypoints. Returns null when
 * either endpoint is missing, below MIN_CONFIDENCE, or so close that the
 * disc would degenerate (extreme DTL view — handled by PR-7).
 */
function ellipseFromPair(
  L: readonly [number | null, number | null, number] | undefined,
  R: readonly [number | null, number | null, number] | undefined,
): DiscParams | null {
  if (!L || !R) return null;
  const [lx, ly, lConf] = L;
  const [rx_, ry_, rConf] = R;
  if (lx === null || ly === null || rx_ === null || ry_ === null) return null;
  const conf = Math.min(lConf, rConf);
  if (conf < MIN_CONFIDENCE) return null;

  const cx = (lx + rx_) / 2;
  const cy = (ly + ry_) / 2;
  const dx = lx - rx_;
  const dy = ly - ry_;
  const dist = Math.sqrt(dx * dx + dy * dy);
  if (dist < 1) return null; // degenerate; PR-7 will handle DTL extremes

  const angleRad = Math.atan2(dy, dx); // see §3 convention
  const rx = dist / 2;
  const ry = rx * PERSPECTIVE_RY_RATIO;
  return { cx, cy, rx, ry, angleRad, confidence: conf };
}

/** Shoulder disc from a PoseFrame's left_shoulder + right_shoulder kp pair. */
export function computeShoulderDisc(frame: PoseFrame): DiscParams | null {
  return ellipseFromPair(frame.keypoints.left_shoulder, frame.keypoints.right_shoulder);
}

/** Hip disc from a PoseFrame's left_hip + right_hip kp pair. */
export function computeHipDisc(frame: PoseFrame): DiscParams | null {
  return ellipseFromPair(frame.keypoints.left_hip, frame.keypoints.right_hip);
}
