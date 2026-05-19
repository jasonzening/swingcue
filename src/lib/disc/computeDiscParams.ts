/**
 * PR-5.3: per-frame shoulder + hip disc geometry from PR-4 keypoints,
 * with distance-ratio rotation. PR-5.1 §3.B anatomical-midpoint
 * correction has been **reverted** in this PR — see commit message
 * (browser-measured ground-truth vs. skeleton dots showed the shoulder
 * lift over-corrected by 25-32 native px onto the neck, hip lift was
 * visually negligible at ~2px).
 *
 * Pure functions — no React, no DOM. Called from the canvas rAF loop
 * in SwingPlayer.tsx.
 *
 * Surviving changes vs. the PR-5 original:
 *   (A) Rotation magnitude derives from shoulder-distance perspective
 *       compression (acos of dist/baseline) instead of pure atan2 —
 *       atan2 alone is clipped to ±90° because MediaPipe's left/right
 *       labels are IMAGE-space, not anatomical, so L.x − R.x stays
 *       positive throughout a swing and atan2 can never exceed |π/2|.
 *       acos(ratio) sees the foreshortening and recovers the true
 *       rotation magnitude up to 90°. See PR-5.1_DESIGN.md §3.A.
 *   (C) `rx` here still returns dist/2 (raw); SwingPlayer overrides
 *       it with the per-video DiscAnchor.rx. See PR-5.1_DESIGN.md §3.C.
 *
 * Removed (PR-5.3):
 *   (B) Anatomical midpoint lift. Disc center is now the raw midpoint
 *       of the left/right shoulder (or hip) keypoint pair — same point
 *       SkeletonOverlay renders the dots on. Restores visual
 *       skeleton/disc agreement.
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

/**
 * PR-5.4: disc radius is 80% of shoulder/hip kp distance (rx),
 * so disc width = 1.6 × kp distance, extending ~30% beyond
 * the lateral keypoints. Combined with perspective foreshortening
 * and 3D tilt, the disc envelopes the upper/lower torso visually
 * while the disc center remains exactly on the kp midpoint.
 *
 * Exported so SwingPlayer can apply the same ratio when overriding
 * the per-frame rx with the per-video DiscAnchor (PR-5.1 §3.C). The
 * anchor stores the setup-baseline distance/2, which must then be
 * scaled identically here for the locked size to match.
 */
export const DISC_RX_RATIO = 1.6;

/**
 * PR-5.4: 3D plane depth ratio (was 0.20 pre-PR-5.4). After the
 * camera-below-plane skew in SwingPlayer.drawTiltedDisc multiplies
 * this by cos(25°) ≈ 0.906, the on-screen ry/rx settles around 0.29.
 * Exported for the same SwingPlayer override-path reason as
 * DISC_RX_RATIO.
 */
export const PERSPECTIVE_RY_RATIO = 0.32;

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
 * or null coords.
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
 * Shoulder disc from a PoseFrame.
 *
 * Center = midpoint of (left_shoulder, right_shoulder), after PR-5.8A
 * coaching-anchor expansion (default 0; SwingPlayer passes the URL-
 * sourced shoulderExpand factor). With expansion=0 this matches the
 * PR-5.3 behaviour of anchoring exactly on the raw kp midpoint; with
 * expansion>0 the disc anchors on the SwingCue coaching shoulder line
 * (humerus-to-body visual junction) and the chord endpoints
 * (cx ± rx along the rotated axis) span the expanded shoulder span.
 *
 * @param frame           PR-4 PoseFrame with COCO 17 keypoints.
 * @param baselineDist    Setup-frame shoulder pair distance (from
 *                        SwingPlayer's `discAnchorRef.shoulderRx * 2`).
 *                        Null until the anchor is initialised — in
 *                        that case angle falls back to plain atan2.
 * @param shoulderExpand  PR-5.8A render-time outward expansion factor
 *                        applied to L and R independently before any
 *                        downstream geometry. Default 0 = behaviour
 *                        unchanged. See lib/skeleton/coachingAnchors.
 */
export function computeShoulderDisc(
  frame: PoseFrame,
  baselineDist: number | null,
  shoulderExpand: number = 0,
): DiscParams | null {
  const pair = validPair(
    frame.keypoints.left_shoulder,
    frame.keypoints.right_shoulder,
  );
  if (!pair) return null;
  // PR-5.8A: expand both endpoints outward along the shoulder line
  // before computing dx/dy/dist/cx/cy. Mirror of the math in
  // lib/skeleton/coachingAnchors.ts (single source of truth).
  const { lx, ly, rx: rrx, ry: rry } = applyExpand(pair, shoulderExpand);
  const dx = lx - rrx;
  const dy = ly - rry;
  const dist = Math.sqrt(dx * dx + dy * dy);
  if (dist < 1) return null;

  const cx = (lx + rrx) / 2;
  const cy = (ly + rry) / 2;
  const angleRad = rotationFromGeometry(dx, dy, dist, baselineDist);
  // PR-5.4: rx widened past the lateral kp so the disc visually
  // envelops the upper/lower torso. Center stays on the (expanded)
  // kp midpoint; only the radius scales.
  const rx = (dist * DISC_RX_RATIO) / 2;

  return {
    cx,
    cy,
    rx,
    ry: rx * PERSPECTIVE_RY_RATIO,
    angleRad,
    confidence: pair.conf,
  };
}

/**
 * Hip disc from a PoseFrame.
 *
 * Center = midpoint of (left_hip, right_hip) after PR-5.8A coaching-
 * anchor expansion (default 0). See computeShoulderDisc for the
 * rationale.
 *
 * The third parameter (`_shoulderMid`) is retained from the PR-5.1
 * signature so existing SwingPlayer call sites still type-check
 * without modification, but it is no longer used (PR-5.1 §3.B revert).
 * Removable in a future cleanup PR alongside the call-site update.
 *
 * @param frame         PR-4 PoseFrame with COCO 17 keypoints.
 * @param baselineDist  Setup-frame hip pair distance.
 * @param _shoulderMid  Deprecated (PR-5.3). Kept for caller compatibility.
 * @param hipExpand     PR-5.8A render-time outward expansion factor
 *                      (default 0; see computeShoulderDisc).
 */
export function computeHipDisc(
  frame: PoseFrame,
  baselineDist: number | null,
  _shoulderMid: { cx: number; cy: number } | null,
  hipExpand: number = 0,
): DiscParams | null {
  const pair = validPair(
    frame.keypoints.left_hip,
    frame.keypoints.right_hip,
  );
  if (!pair) return null;
  const { lx, ly, rx: rrx, ry: rry } = applyExpand(pair, hipExpand);
  const dx = lx - rrx;
  const dy = ly - rry;
  const dist = Math.sqrt(dx * dx + dy * dy);
  if (dist < 1) return null;

  const cx = (lx + rrx) / 2;
  const cy = (ly + rry) / 2;
  const angleRad = rotationFromGeometry(dx, dy, dist, baselineDist);
  // PR-5.4: rx widened past the lateral kp so the disc visually
  // envelops the upper/lower torso. Center stays on the (expanded)
  // kp midpoint; only the radius scales.
  const rx = (dist * DISC_RX_RATIO) / 2;

  return {
    cx,
    cy,
    rx,
    ry: rx * PERSPECTIVE_RY_RATIO,
    angleRad,
    confidence: pair.conf,
  };
}

/**
 * PR-5.8A: apply outward expansion along the L↔R line, returning the
 * same field names the rest of this module expects (`lx, ly, rx, ry`
 * — note `rx`/`ry` here are right.x/right.y, NOT disc radii). Keeps
 * the math co-located with the existing pair shape so callers don't
 * have to juggle Point2D vs validPair output.
 *
 * factor=0 → returns the input unchanged (no-op, no allocation cost
 * difference worth optimising). factor>0 → both sides move outward
 * symmetrically by `factor` × current half-span. See PR-5.8A spec.
 */
function applyExpand(
  pair: { lx: number; ly: number; rx: number; ry: number; conf: number },
  factor: number,
): { lx: number; ly: number; rx: number; ry: number } {
  if (factor === 0) return { lx: pair.lx, ly: pair.ly, rx: pair.rx, ry: pair.ry };
  const midX = (pair.lx + pair.rx) / 2;
  const midY = (pair.ly + pair.ry) / 2;
  const k = 1 + factor;
  return {
    lx: midX + (pair.lx - midX) * k,
    ly: midY + (pair.ly - midY) * k,
    rx: midX + (pair.rx - midX) * k,
    ry: midY + (pair.ry - midY) * k,
  };
}
