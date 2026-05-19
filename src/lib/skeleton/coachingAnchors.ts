/**
 * PR-5.8A — Render-time SwingCue coaching-anchor expansion.
 *
 * Pure helpers — no React, no DOM, no I/O. Read by SkeletonOverlay
 * and computeDiscParams (PR-5.8A commits 2 + 3) and orchestrated by
 * the result page (commit 1).
 *
 * DEFINITION
 *   SwingCue coaching shoulder anchor = the visible point where the
 *   humerus connects to the body in the rendered overlay. After
 *   expansion the shoulder→elbow skeleton line falls on the upper-arm
 *   visual midline and the shoulder disc anchors there, not on the
 *   inner MediaPipe/COCO points (acromion / sternum-side clavicle).
 *   Same idea applies to hip → side-of-torso.
 *
 * MATH (single source of truth — keep in sync with PR-5.8A spec)
 *   mid       = (L + R) / 2
 *   expandedL = mid + (L - mid) * (1 + factor)
 *   expandedR = mid + (R - mid) * (1 + factor)
 *
 *   factor=0.40 expands EACH side outward by 40% of its current
 *   offset-from-midpoint. Total L↔R width also grows by 40% (because
 *   both sides expand symmetrically), NOT by 80%. The (1 + factor)
 *   form keeps the algebra obvious at the cost of a 1-line comment.
 *
 * CALIBRATION (defaults)
 *   Shoulder 0.40, Hip 0.25 — first-cut on the b3fea3f0 face-on setup
 *   frame. Override at runtime via URL params:
 *       ?shoulderExpand=0.40&hipExpand=0.25
 *   Out-of-range or non-finite values fall back to the default.
 *
 * SCOPE (PR-5.8A — strict)
 *   Render-time only. Raw values in pose_timeline_2d are NEVER
 *   mutated. Confidence (`v`) passes through unchanged.
 */

export const SHOULDER_EXPAND_DEFAULT = 0.40;
export const HIP_EXPAND_DEFAULT = 0.25;

const EXPAND_MIN = 0;
const EXPAND_MAX = 1.5;

export type Point2D = { x: number; y: number; v?: number };

function expand(
  L: Point2D,
  R: Point2D,
  factor: number,
): { left: Point2D; right: Point2D } {
  const midX = (L.x + R.x) / 2;
  const midY = (L.y + R.y) / 2;
  const k = 1 + factor;
  return {
    left:  { x: midX + (L.x - midX) * k, y: midY + (L.y - midY) * k, v: L.v },
    right: { x: midX + (R.x - midX) * k, y: midY + (R.y - midY) * k, v: R.v },
  };
}

export function expandShoulders(L: Point2D, R: Point2D, factor: number) {
  return expand(L, R, factor);
}

export function expandHips(L: Point2D, R: Point2D, factor: number) {
  return expand(L, R, factor);
}

function parseFactor(raw: string | null, fallback: number): number {
  if (raw === null) return fallback;
  const n = Number.parseFloat(raw);
  if (!Number.isFinite(n)) return fallback;
  if (n < EXPAND_MIN || n > EXPAND_MAX) return fallback;
  return n;
}

export function readExpandFactorsFromURL(
  searchParams: URLSearchParams,
): { shoulder: number; hip: number } {
  return {
    shoulder: parseFactor(searchParams.get('shoulderExpand'), SHOULDER_EXPAND_DEFAULT),
    hip:      parseFactor(searchParams.get('hipExpand'),      HIP_EXPAND_DEFAULT),
  };
}
