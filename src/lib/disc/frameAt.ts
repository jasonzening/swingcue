/**
 * Pose-timeline time lookups.
 *
 * PR-5 hotfix introduced `frameAt` (binary-search nearest). PR-5.9 Task 3
 * adds `interpolatedFrame` (linear-lerp between bracketing frames) — the
 * preferred lookup for any new render path, because it animates dots
 * continuously between pose samples instead of freezing them.
 *
 * `frameAt` is left in place, marked DEPRECATED, for any caller that
 * specifically needs nearest-neighbour semantics. New code: use
 * `interpolatedFrame`.
 */

import type { Keypoint, PoseFrame, PoseTimeline } from '@/types/analysis';

/**
 * @deprecated PR-5.9 Task 3. Use `interpolatedFrame` instead — nearest-
 * neighbour lookup causes a visible dot freeze between samples,
 * especially noticeable on fast motion (downswing/transition/impact).
 *
 * Look up the pose frame whose `ts` is closest to `t`.
 * O(log n), no allocation, rAF-safe.
 *
 * Frames are assumed sorted by ts ascending (PR-4 build guarantee).
 * Returns null on empty / missing timeline.
 */
export function frameAt(t: number, pose: PoseTimeline | null | undefined): PoseFrame | null {
  if (!pose?.frames?.length) return null;
  const frames = pose.frames;
  let lo = 0;
  let hi = frames.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].ts < t) lo = mid + 1;
    else hi = mid;
  }
  // After loop lo == hi == first idx with ts >= t (or last idx if t > all ts).
  if (lo > 0) {
    const a = frames[lo - 1];
    const b = frames[lo];
    return (t - a.ts < b.ts - t) ? a : b;
  }
  return frames[lo];
}

/**
 * PR-5.9 Task 3: linear-interpolated lookup. Locates the two frames
 * bracketing `t` via binary search, then per-keypoint lerps x and y.
 * Confidence becomes `min(c_before, c_after)` (conservative — if either
 * sample is uncertain, the interpolated value inherits that doubt).
 *
 * Null-aware: if one side has a null coord for a given keypoint, the
 * non-null side's value is used as-is for that keypoint. If both sides
 * are null, the result is null.
 *
 * raw_keypoints (PR-5.9 Task 4) is interpolated identically when both
 * bracketing frames carry it. Returns a synthesised PoseFrame with
 * `interpolated: true` and `frame_idx` attributed to the earlier of
 * the two source frames.
 *
 * Edge cases:
 *   - empty/missing timeline → null
 *   - t <= frames[0].ts      → frames[0] (no synth)
 *   - t >= frames[last].ts   → frames[last] (no synth)
 *   - bracket frames have equal ts (degenerate) → before (no synth)
 */
export function interpolatedFrame(
  pose: PoseTimeline | null | undefined,
  t: number,
): PoseFrame | null {
  if (!pose?.frames?.length) return null;
  const frames = pose.frames;
  if (t <= frames[0].ts) return frames[0];
  if (t >= frames[frames.length - 1].ts) return frames[frames.length - 1];
  // Binary search: first idx with ts >= t. Guaranteed > 0 by edge checks.
  let lo = 0;
  let hi = frames.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].ts < t) lo = mid + 1;
    else hi = mid;
  }
  const after = frames[lo];
  const before = frames[lo - 1];
  if (before.ts === after.ts) return before;
  const ratio = (t - before.ts) / (after.ts - before.ts);

  const keypoints = lerpKeypointSets(before.keypoints, after.keypoints, ratio);
  const beforeRaw = (before as PoseFrame & { raw_keypoints?: Record<string, Keypoint> }).raw_keypoints;
  const afterRaw  = (after  as PoseFrame & { raw_keypoints?: Record<string, Keypoint> }).raw_keypoints;
  const synth: PoseFrame & { raw_keypoints?: Record<string, Keypoint> } = {
    ts: t,
    frame_idx: before.frame_idx,
    interpolated: true,
    keypoints: keypoints as PoseFrame['keypoints'],
  };
  if (beforeRaw && afterRaw) {
    synth.raw_keypoints = lerpKeypointSets(beforeRaw, afterRaw, ratio);
  }
  return synth;
}

/**
 * Per-keypoint linear lerp. Operates on the loose Record shape (string
 * key) so the function works for both the COCO-typed `keypoints` map
 * and the PR-5.9 `raw_keypoints` sibling (which carries the same
 * 17 + head_crown keys but isn't strict-typed on PoseFrame).
 *
 * For each name present in `a`:
 *   - if `b` is missing the name → use a's value as-is
 *   - if either side's coord is null → use the non-null side
 *   - else → linearly interpolate x and y, take min of confidences
 */
function lerpKeypointSets(
  a: Record<string, Keypoint>,
  b: Record<string, Keypoint>,
  ratio: number,
): Record<string, Keypoint> {
  const out: Record<string, Keypoint> = {};
  for (const name of Object.keys(a)) {
    const ka = a[name];
    const kb = b[name];
    if (!kb) {
      out[name] = ka;
      continue;
    }
    const [x1, y1, c1] = ka;
    const [x2, y2, c2] = kb;
    if (x1 === null || y1 === null) {
      out[name] = (x2 === null || y2 === null)
        ? ([null, null, Math.min(c1, c2)] as Keypoint)
        : ([x2, y2, c2] as Keypoint);
    } else if (x2 === null || y2 === null) {
      out[name] = [x1, y1, c1] as Keypoint;
    } else {
      out[name] = [
        x1 + (x2 - x1) * ratio,
        y1 + (y2 - y1) * ratio,
        Math.min(c1, c2),
      ] as Keypoint;
    }
  }
  return out;
}
