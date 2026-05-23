/**
 * PR-7c-frontend — phase-aware opacity ramp for coaching anchors.
 *
 * Source: docs/decisions/PR-7c_READINESS_CHECKLIST.md §2.
 *
 * Purpose: soften anchor opacity at fast-motion phases (impact / finish)
 * where WHAM tracking quality is the data-driven ceiling. The opacity
 * ramp is the user-facing answer to residual drift — NOT an attempt
 * to correct it (PR-7a.5 PROBE FAIL confirmed the architectural ceiling
 * is hit). Less visual prominence at low-confidence frames.
 *
 * Pure function — no React, no DOM, no I/O. Testable in isolation.
 */

/**
 * Base opacity per phase. setup/backswing/top render at full opacity;
 * transition/downswing softens; impact/finish are noticeably faded.
 * The `finish` value is the starting point of a linear ramp to 0 at
 * clip end (see `computeAnchorOpacity` finish branch).
 */
const PHASE_OPACITY_BASE: Record<string, number> = {
  setup: 1.0,
  backswing: 1.0,
  top: 1.0,
  transition: 0.85,
  downswing: 0.65,
  impact: 0.45,
  finish: 0.30,
};

export interface AnchorOpacityArgs {
  /** Phase name at the current frame (from CorrectedFrame.phase). */
  phase: string;
  /** Current playback time in seconds. */
  ts: number;
  /** Clip duration in seconds (from CorrectedTimeline.duration_sec). */
  durationSec: number;
  /**
   * ts of the first `finish`-phase frame, used as the start of the
   * linear fade-to-0 tail. If undefined, no fade is applied — finish
   * frames render at their base 0.30 opacity.
   *
   * Compute once per timeline via `findFinishStartTs(timeline)`.
   */
  finishStartTs?: number;
}

/**
 * Compute the opacity an anchor should render at given the current
 * frame's phase + ts. Returns a value in [0, 1].
 *
 * Behavior:
 * - Non-`finish` phases: return the base opacity for that phase.
 * - `finish` phase + `finishStartTs` known: linear ramp from
 *   PHASE_OPACITY_BASE.finish (0.30) at finishStartTs down to 0 at
 *   durationSec.
 * - `finish` phase without `finishStartTs`: hold at 0.30.
 * - Unknown phase: return 1.0 (defensive — render rather than vanish).
 */
export function computeAnchorOpacity(args: AnchorOpacityArgs): number {
  const { phase, ts, durationSec, finishStartTs } = args;
  const base = PHASE_OPACITY_BASE[phase] ?? 1.0;

  if (phase === "finish" && finishStartTs !== undefined && durationSec > finishStartTs) {
    const tailFrac = Math.min(
      1,
      Math.max(0, (ts - finishStartTs) / (durationSec - finishStartTs)),
    );
    return base * (1 - tailFrac);
  }
  return base;
}

/**
 * Lookup the base opacity for a phase without the finish-tail ramp.
 * Useful for static-frame contexts (debug rendering, screenshots).
 */
export function basePhaseOpacity(phase: string): number {
  return PHASE_OPACITY_BASE[phase] ?? 1.0;
}
