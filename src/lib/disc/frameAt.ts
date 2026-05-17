/**
 * PR-5 hotfix: ts-based binary search.
 *
 * Previous version used `Math.round(t * fps_sampled)` and relied on
 * `pose_timeline_2d.fps_sampled` metadata. The PR-4 backend
 * (python/pose_timeline.py) currently writes `fps_sampled = 10` while
 * the actual sample rate is ~14 fps, causing a ~40% lookup offset.
 *
 * Switching to ts-based binary search makes this helper resilient to
 * any future metadata drift — `fps_sampled` is no longer read.
 *
 * Backend fix is out of scope for PR-5; see PR-4.1 / PR-6.
 */

import type { PoseFrame, PoseTimeline } from '@/types/analysis';

/**
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
