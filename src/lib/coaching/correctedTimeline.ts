/**
 * PR-7c-frontend — fetch + lookup helper for offline CorrectedTimeline
 * JSON published to Supabase Storage by the motion_correction offline
 * pipeline.
 *
 * Schema source: python/motion_correction/schemas/corrected_timeline.py
 * (Python dataclasses serialized via json.dumps).
 *
 * Storage layout (public bucket, set up via Supabase MCP):
 *   <SUPABASE_URL>/storage/v1/object/public/corrected-timelines/<video_id>.json
 *
 * Demo mode: a video has an "enhanced" overlay iff a JSON exists at that
 * path. Missing JSON → fetch returns null → caller falls back to
 * existing MediaPipe overlay. No errors logged for the 404 path; it's
 * the expected behavior for the majority of videos until PR-7d wires
 * WHAM into production analyze.
 */

/**
 * The 5 anchor names PR-7c-frontend renders. The motion_correction
 * schema emits 7 (5 per-side + 2 disc-center midpoints); we skip the
 * midpoints per the PR-7c constitution.
 */
export const COACHING_ANCHOR_NAMES_RENDER = [
  "left_shoulder_visual",
  "right_shoulder_visual",
  "left_hip_visual",
  "right_hip_visual",
  "neck_visual",
] as const;

export type CoachingAnchorName = (typeof COACHING_ANCHOR_NAMES_RENDER)[number];

/** A 2D pixel coord in the corrected timeline's native video frame. */
export type UV = readonly [number, number];

/**
 * One frame's worth of corrected output. Mirrors the Python
 * `CorrectedFrame` dataclass but with only the fields the frontend
 * consumes — engine-internal fields (keypoints_3d, diagnostics) are
 * left unparsed.
 */
export interface CorrectedFrame {
  ts: number;            // seconds from clip start
  frame_idx: number;
  phase: string;         // setup | backswing | top | transition | downswing | impact | finish
  coaching_anchors_2d: Partial<Record<string, UV | null>>;
}

/**
 * Top-level corrected timeline shape. Other fields exist on disk but
 * the frontend doesn't need them — declared loosely as `unknown` so
 * future schema additions don't break parsing.
 */
export interface CorrectedTimeline {
  video_id: string;
  video_width: number;
  video_height: number;
  duration_sec: number;
  fps_native: number;
  frames: CorrectedFrame[];
}

const SUPABASE_PUBLIC_STORAGE =
  "https://ciofgtwwcgyzfafmbjxu.supabase.co/storage/v1/object/public";
const BUCKET = "corrected-timelines";

function urlFor(videoId: string): string {
  return `${SUPABASE_PUBLIC_STORAGE}/${BUCKET}/${videoId}.json`;
}

/**
 * Best-effort fetch. Returns null on 404, parse error, or abort — the
 * caller treats any of these as "no enhanced overlay for this video"
 * and falls back to the existing MediaPipe path. Never throws.
 */
export async function fetchCorrectedTimeline(
  videoId: string,
  signal?: AbortSignal,
): Promise<CorrectedTimeline | null> {
  try {
    const res = await fetch(urlFor(videoId), {
      signal,
      // Don't send credentials — bucket is public-read.
      credentials: "omit",
      cache: "default",
    });
    if (!res.ok) return null;
    const data = (await res.json()) as Partial<CorrectedTimeline>;
    if (
      typeof data?.video_id !== "string" ||
      !Array.isArray(data.frames) ||
      typeof data.video_width !== "number" ||
      typeof data.video_height !== "number" ||
      typeof data.duration_sec !== "number"
    ) {
      return null;
    }
    return data as CorrectedTimeline;
  } catch {
    return null;
  }
}

/**
 * Find the frame at (or just before) the given ts via binary search.
 *
 * Stateless O(log n) lookup — mirrors `interpolatedFrame` in
 * src/lib/disc/frameAt.ts (the proven helper SkeletonOverlay uses).
 * No carried hint state — cannot poison itself across calls.
 *
 * Earlier (broken) implementation used a forward-only linear scan
 * from a stateful hintIdxRef. If the first lookup happened at a ts
 * past the last frame (e.g. video element preloaded at
 * currentTime=duration before component mount), subsequent lookups
 * scanned forward from `last`, found no frame matching, and fell
 * through to return `frames[last]` permanently → anchors frozen.
 *
 * Edge cases:
 *   - empty timeline → null
 *   - ts <= frames[0].ts → frames[0]
 *   - ts >= frames[last].ts → frames[last]
 */
export function frameAtTime(
  timeline: CorrectedTimeline,
  ts: number,
): { frame: CorrectedFrame; idx: number } | null {
  const frames = timeline.frames;
  if (frames.length === 0) return null;
  if (ts <= frames[0].ts) return { frame: frames[0], idx: 0 };
  const last = frames.length - 1;
  if (ts >= frames[last].ts) return { frame: frames[last], idx: last };
  // Binary search: first idx whose ts is > queried ts. Guaranteed
  // to land at lo > 0 by the edge checks above.
  let lo = 0;
  let hi = last;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].ts <= ts) lo = mid + 1;
    else hi = mid;
  }
  // We want the frame BEFORE the first ts > queried.
  const idx = lo - 1;
  return { frame: frames[idx], idx };
}

/**
 * Return the ts of the first `finish`-phase frame, or null if the
 * timeline never enters `finish`. Used by phaseOpacity to compute the
 * linear fade tail.
 */
export function findFinishStartTs(timeline: CorrectedTimeline): number | null {
  const first = timeline.frames.find((f) => f.phase === "finish");
  return first ? first.ts : null;
}
