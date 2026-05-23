/**
 * PR-7c-frontend-v2 — MediaPipe-direct anchor lookup for the enhanced
 * coaching overlay.
 *
 * Supersedes the prior corrected-timeline-from-Supabase-Storage approach
 * (`correctedTimeline.ts`, deleted). MCP audit confirmed PR-7a fitted
 * offset was 54-91 px off MediaPipe at top phase on b32e0f21 frame 30,
 * because the 15-sample GT-fit has high per-phase variance. Until model
 * swap (PR-7d NLF research path), MediaPipe-direct is the visually
 * accurate path.
 *
 * Data source: production `pose_timeline_2d` envelope (PoseTimeline
 * from @/types/analysis), already loaded by SwingPlayer for the
 * existing SkeletonOverlay path.
 *
 * Frame lookup: `interpolatedFrame` from @/lib/disc/frameAt — the
 * proven helper SkeletonOverlay uses. Linear lerp between bracketing
 * samples, smooth animation at rAF rate.
 *
 * Phase mapping: `getCurrentPhase` from @/lib/overlay/playerSync —
 * the production helper used by the phase badge + disc. 5 phases:
 * setup/top/transition/impact/finish. Single source of truth — anchor
 * opacity ramp stays in sync with phase badge.
 *
 * Pure helpers — no React, no DOM, no I/O.
 */

import type {
  CocoKeypointName,
  PoseTimeline,
} from "@/types/analysis";
import { interpolatedFrame } from "@/lib/disc/frameAt";

/**
 * The 5 visual anchor names rendered by CoachingAnchorOverlay. Maps
 * 1:1 to MediaPipe keypoints per VISUAL_ANCHOR_TO_COCO below.
 */
export const VISUAL_ANCHOR_NAMES = [
  "head",
  "left_shoulder",
  "right_shoulder",
  "left_hip",
  "right_hip",
] as const;

export type VisualAnchorName = (typeof VISUAL_ANCHOR_NAMES)[number];

/**
 * Visual anchor → MediaPipe COCO keypoint. "head" uses nose (not
 * head_crown) per PR-7c-frontend-v2 spec — nose has higher confidence
 * in practice and is closer to the visually expected "head" landmark
 * on a face-on view.
 */
const VISUAL_ANCHOR_TO_COCO: Record<VisualAnchorName, CocoKeypointName> = {
  head:           "nose",
  left_shoulder:  "left_shoulder",
  right_shoulder: "right_shoulder",
  left_hip:       "left_hip",
  right_hip:      "right_hip",
};

/**
 * Hide an anchor when MediaPipe confidence on that keypoint falls
 * below this threshold. Same value used by PR-5.9 Task 6's
 * `confidenceOpacity` lower bound — anything < 0.3 is treated as
 * "not visible" rather than "low quality".
 */
export const ANCHOR_CONFIDENCE_MIN = 0.3;

/**
 * One anchor's render-ready state for a given video time.
 * - `xy` is null when the source keypoint is null/low-conf/missing.
 * - `confidence` is forwarded for diagnostics; caller already gates
 *   on ANCHOR_CONFIDENCE_MIN.
 */
export interface AnchorPoint {
  xy: readonly [number, number] | null;
  confidence: number;
}

/**
 * Resolve all 5 visual anchors at video time `t` from a PoseTimeline.
 * Uses `interpolatedFrame` so anchor positions animate smoothly between
 * source samples (PR-5.9 Task 3 pattern). Returns null only when the
 * timeline is empty or `t` is before the first sample (defensive — the
 * frame helper guards both cases internally).
 */
export function poseAnchorsAtTime(
  timeline: PoseTimeline,
  t: number,
): Record<VisualAnchorName, AnchorPoint> | null {
  const frame = interpolatedFrame(timeline, t);
  if (!frame) return null;
  const out = {} as Record<VisualAnchorName, AnchorPoint>;
  for (const visualName of VISUAL_ANCHOR_NAMES) {
    const cocoName = VISUAL_ANCHOR_TO_COCO[visualName];
    const kp = frame.keypoints[cocoName];
    if (!kp) {
      out[visualName] = { xy: null, confidence: 0 };
      continue;
    }
    const [x, y, conf] = kp;
    if (
      x === null
      || y === null
      || typeof x !== "number"
      || typeof y !== "number"
      || conf < ANCHOR_CONFIDENCE_MIN
    ) {
      out[visualName] = { xy: null, confidence: conf };
      continue;
    }
    out[visualName] = { xy: [x, y], confidence: conf };
  }
  return out;
}
