/**
 * PR-5: currentTime → PoseFrame lookup.
 *
 * Uses pose_timeline_2d.fps_sampled to map seconds → frame index, then
 * clamps to the valid range. Designed to be called from a 60Hz rAF loop;
 * O(1) so allocation-free.
 */

import type { PoseFrame, PoseTimeline } from '@/types/analysis';

/**
 * Look up the pose frame closest to a given video timestamp.
 *
 * @param t      Current video time in seconds.
 * @param pose   The pose_timeline_2d payload, or null/undefined when the
 *               video has no timeline (predates PR-4, or validation failed).
 * @returns      The selected PoseFrame, or null when the timeline is empty
 *               or absent.
 */
export function frameAt(t: number, pose: PoseTimeline | null | undefined): PoseFrame | null {
  if (!pose || !pose.frames || pose.frames.length === 0) return null;
  const fps = pose.fps_sampled || 10;
  const idx = Math.round(t * fps);
  const clamped = Math.max(0, Math.min(pose.frames.length - 1, idx));
  return pose.frames[clamped] || null;
}
