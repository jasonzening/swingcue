/**
 * PR-5.1: per-frame shoulder + hip disc geometry from PR-4 keypoints,
 * with PR-5.1 anatomical correction + distance-ratio rotation.
 *
 * Pure functions — no React, no DOM. Called from the canvas rAF loop
 * in SwingPlayer.tsx.
 *
 * Three changes vs. the PR-5 original:
 *   (A) Rotation magnitude derives from shoulder-distance perspective
 *       compression (acos of dist/baseline) instead of pure atan2 —
 *       atan2 alone is clipped to ±90° because MediaPipe's left/right
 *       labels are IMAGE-space, not anatomical, so L.x − R.x stays
 *       positive throughout a swing and atan2 can never exceed |π/2|.
 *       acos(ratio) sees the foreshortening and recovers the true
 *       rotation magnitude up to 90°. See PR-5.1_DESIGN.md §3.A.
 *   (B) Disc center is lifted from MediaPipe's clavicle/abdomen
 *       midpoint toward the anatomical acromion (shoulder) or hip
 *       joint via head-anchored / shoulder-anchored offsets. See
 *       PR-5.1_DESIGN.md §3.B.
 *   (C) `rx` here still returns dist/2 (raw); SwingPlayer overrides
 *       it with the per-video DiscAnchor.rx. See §3.C.
 *
 * Angle convention (NORMATIVE, PR-5_DESIGN.md §3 unchanged):
 *   angle sign matches image-y-down rotation direction. Setup ≈ 0,
 *   Top (RH golfer) ≈ +angle (L drops below R in image), Finish ≈ -angle.
 *
 * Coordinate output: cx/cy/rx/ry are in video native pixel space.
 */

import type { PoseFrame, Keypoint } from '@/types/analysis';
import type { DiscParams } from './types';

const MIN_CONFIDENCE = 0.3;
const PERSPECTIVE_RY_RATIO = 0.2;

/**
 * Distance-ratio rotation. When `baselineDist` is null/invalid, falls back
 * to PR-5's original atan2 behaviour so the helper stays usable before
 * the SwingPlayer anchor is initialised.
 *
 * Algorithm: see PR-5.1_DESIGN.md §3.A. Calibration on test video
 * `7cd23f91-...` at top peak (dist=32, baseline=107) yields acos(0.30) ≈
 * 72.5° — matches Jason's expectation of 70-80° rotation magnitude.
 */
function rotationFromGeometry(
  dx: number, dy: number, dist: number,
  baselineDist: number | null,
): number {
  if (!baselineDist || baselineDist <= 1) {
    return Math.atan2(dy, dx);
  }
  const ratio = Math.min(1, dist / baselineDist);
  const magnitude = Math.acos(ratio);                  // [0, π/2]
  // Sign: in profile (dist small) dy magnitude reveals direction; in
  // face-on (dist big) atan2 sign reveals direction.
  const sign = (Math.abs(dy) > Math.abs(dx))
    ? Math.sign(dy)
    : Math.sign(Math.atan2(dy, dx));
  return sign * magnitude;
}

/**
 * Validate + unpack a Keypoint into a tuple of (x, y, conf) where x/y
 * are guaranteed numeric. Returns null on missing-pair, low confidence,
 * or null coords. Coalesces the four PR-5.1 / PR-5 fail-modes.
 */
function validPair(
  L: Keypoint | undefined, R: Keypoint | undefined,
): { lx: number; ly: number; rx: number; ry: number; conf: number } | null {
  if (!L || !R) return null;
  const [lx, ly, lConf] = L;
  const [rx, ry, rConf] = R;
  if (lx === null || ly === null || rx === null || ry === null) return null;
  const conf = Math.min(lConf, rConf);
  if (conf < MIN_CONFIDENCE) return null;
  return { lx, ly, rx, ry, conf };
}

/**
 * PR-5.1 §3.B — lift shoulder midpoint from MediaPipe's clavicle
 * position toward the visual acromion. Preferred path uses the
 * (nose, ear) triangle to derive a 0.5× ear-distance offset along
 * the mid→nose direction; fallback heuristic is a 15% upward shift
 * by the shoulder-pair distance.
 */
function correctShoulderMidpoint(
  rawMidX: number, rawMidY: number,
  shoulderDist: number,
  nose: Keypoint | undefined,
  leftEar: Keypoint | undefined,
  rightEar: Keypoint | undefined,
): { cx: number; cy: number } {
  if (nose && leftEar && rightEar
      && nose[0] !== null && nose[1] !== null
      && leftEar[0] !== null && leftEar[1] !== null
      && rightEar[0] !== null && rightEar[1] !== null
      && leftEar[2] > MIN_CONFIDENCE && rightEar[2] > MIN_CONFIDENCE) {
    const earDx = leftEar[0] - rightEar[0];
    const earDy = leftEar[1] - rightEar[1];
    const earDist = Math.sqrt(earDx * earDx + earDy * earDy);
    const liftAmount = earDist * 0.5;
    const dx = nose[0] - rawMidX;
    const dy = nose[1] - rawMidY;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len > 1) {
      return {
        cx: rawMidX + (dx / len) * liftAmount,
        cy: rawMidY + (dy / len) * liftAmount,
      };
    }
  }
  // Fallback: straight up by 15% of shoulder distance.
  return { cx: rawMidX, cy: rawMidY - shoulderDist * 0.15 };
}

/**
 * PR-5.1 §3.B — lift hip midpoint from MediaPipe's abdomen position
 * toward the corrected shoulder midpoint by 10% of hip-pair distance.
 * Returns the raw midpoint when no shoulder reference is available.
 */
function correctHipMidpoint(
  rawMidX: number, rawMidY: number,
  hipDist: number,
  shoulderMid: { cx: number; cy: number } | null,
): { cx: number; cy: number } {
  if (!shoulderMid) return { cx: rawMidX, cy: rawMidY };
  const dx = shoulderMid.cx - rawMidX;
  const dy = shoulderMid.cy - rawMidY;
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len <= 1) return { cx: rawMidX, cy: rawMidY };
  const liftAmount = hipDist * 0.1;
  return {
    cx: rawMidX + (dx / len) * liftAmount,
    cy: rawMidY + (dy / len) * liftAmount,
  };
}

/**
 * Shoulder disc from a PoseFrame.
 *
 * @param frame         PR-4 PoseFrame with COCO 17 keypoints.
 * @param baselineDist  Setup-frame shoulder pair distance (from
 *                      SwingPlayer's `discAnchorRef.shoulderRx * 2`).
 *                      Null until the anchor is initialised — in that
 *                      case angle falls back to plain atan2.
 */
export function computeShoulderDisc(
  frame: PoseFrame,
  baselineDist: number | null,
): DiscParams | null {
  const pair = validPair(
    frame.keypoints.left_shoulder,
    frame.keypoints.right_shoulder,
  );
  if (!pair) return null;
  const dx = pair.lx - pair.rx;
  const dy = pair.ly - pair.ry;
  const dist = Math.sqrt(dx * dx + dy * dy);
  if (dist < 1) return null;

  const rawMidX = (pair.lx + pair.rx) / 2;
  const rawMidY = (pair.ly + pair.ry) / 2;
  const corrected = correctShoulderMidpoint(
    rawMidX, rawMidY, dist,
    frame.keypoints.nose,
    frame.keypoints.left_ear,
    frame.keypoints.right_ear,
  );

  const angleRad = rotationFromGeometry(dx, dy, dist, baselineDist);

  const rx = dist / 2;
  return {
    cx: corrected.cx,
    cy: corrected.cy,
    rx,
    ry: rx * PERSPECTIVE_RY_RATIO,
    angleRad,
    confidence: pair.conf,
  };
}

/**
 * Hip disc from a PoseFrame.
 *
 * @param frame         PR-4 PoseFrame with COCO 17 keypoints.
 * @param baselineDist  Setup-frame hip pair distance (from SwingPlayer's
 *                      `discAnchorRef.hipRx * 2`). See computeShoulderDisc.
 * @param shoulderMid   Already-corrected shoulder midpoint (cx, cy) for
 *                      this same frame, so the hip anatomical lift points
 *                      toward the right place. Pass null when the
 *                      shoulder disc could not be computed (rare).
 */
export function computeHipDisc(
  frame: PoseFrame,
  baselineDist: number | null,
  shoulderMid: { cx: number; cy: number } | null,
): DiscParams | null {
  const pair = validPair(
    frame.keypoints.left_hip,
    frame.keypoints.right_hip,
  );
  if (!pair) return null;
  const dx = pair.lx - pair.rx;
  const dy = pair.ly - pair.ry;
  const dist = Math.sqrt(dx * dx + dy * dy);
  if (dist < 1) return null;

  const rawMidX = (pair.lx + pair.rx) / 2;
  const rawMidY = (pair.ly + pair.ry) / 2;
  const corrected = correctHipMidpoint(rawMidX, rawMidY, dist, shoulderMid);

  const angleRad = rotationFromGeometry(dx, dy, dist, baselineDist);

  const rx = dist / 2;
  return {
    cx: corrected.cx,
    cy: corrected.cy,
    rx,
    ry: rx * PERSPECTIVE_RY_RATIO,
    angleRad,
    confidence: pair.conf,
  };
}
