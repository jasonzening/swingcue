import type { VideoMetaForAnnotation } from '@/lib/types/annotation';

/**
 * Phase-frame derivation — single source of truth shared between the
 * annotation workbench (where Jason clicks at these frame_idx values)
 * and the validation review page (where rendered overlays must land on
 * those EXACT same frame_idx values).
 *
 * Extracted from src/app/admin/annotate/[videoId]/AnnotateWorkbench.tsx
 * (originally introduced in commit 0a419ba). Behavior preserved bit-
 * for-bit so the review page renders overlays at the same coordinate
 * the workbench captured — re-deriving in two places risks drift.
 *
 * Frame mapping rules (PR-7A.1 reduction from v1's 7 phases to 5):
 *   setup       = round(setupTime * fps)
 *   takeaway    = round((setupTime + 0.25 * (topTime - setupTime)) * fps)
 *   top         = round(topTime * fps)
 *   transition  = round(transitionTime * fps)
 *                   fallback: round((topTime + 0.4 * (impactTime - topTime)) * fps)
 *   impact      = round(impactTime * fps)
 *
 * Returns null when any of the 4 "primary" markers (setup/top/impact/
 * finish) is missing — callers route to manual calibration in that case.
 */

export type PhaseFrames = {
  setup:      number;
  takeaway:   number;
  top:        number;
  transition: number;
  impact:     number;
};

export function timeToFrame(timeSec: number, fps: number): number {
  return Math.max(0, Math.round(timeSec * fps));
}

export function derivePhaseFrames(
  meta: VideoMetaForAnnotation,
): PhaseFrames | null {
  const pm = meta.phaseMarkers;
  // Hard requirement: the 4 "primary" markers must exist (transition is
  // derivable from top+impact). Without any of the four primaries, route
  // to manual calibration.
  if (
    pm.setupTime == null || pm.topTime == null ||
    pm.impactTime == null || pm.finishTime == null
  ) return null;

  const transitionTime = pm.transitionTime ??
    (pm.topTime + 0.4 * (pm.impactTime - pm.topTime));

  return {
    setup:      timeToFrame(pm.setupTime, meta.fps),
    takeaway:   timeToFrame(pm.setupTime + 0.25 * (pm.topTime - pm.setupTime), meta.fps),
    top:        timeToFrame(pm.topTime, meta.fps),
    transition: timeToFrame(transitionTime, meta.fps),
    impact:     timeToFrame(pm.impactTime, meta.fps),
  };
}
