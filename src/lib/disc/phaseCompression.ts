/**
 * phaseCompression.ts — PR-5.7
 *
 * Phase-aware visual compression curve for the SwingCue coaching
 * disc. Returns a 0.4-1.0 scalar that SwingPlayer multiplies against
 * the median baseline rx to produce the final disc width.
 *
 * Conceptual model (NORMATIVE, per Jason + ChatGPT lock-in):
 *
 *   final_rx = baseline_rx × phase_compression × (1 + micro_correction)
 *              \_______/   \________________/    \_________________/
 *               PR-5.6        primary signal       ±10% currentDist
 *               anchor       (this module)         nudge (this module)
 *
 * `phase_compression` is the DOMINANT signal — coaching readability
 * over raw 2D foreshortening. Shoulder + hip have independent
 * compression curves because the body opens differently at each.
 * `micro_correction` is a small (±10%) nudge derived from live kp
 * distance so the disc still "breathes" with the body even when the
 * phase plateau hasn't moved — but it cannot dominate.
 *
 * Phase boundaries are interpolated via smoothstep (3t² − 2t³), not
 * linear, so disc width doesn't snap at phase thresholds. Smoothstep
 * has zero first-derivative at both endpoints — phase transitions
 * read as eased, not abrupt.
 *
 * Fallback: if phaseMarkers is non-monotonic (out-of-order ts),
 * returns 1.0 unconditionally so the disc gracefully degrades to
 * PR-5.6 (baseline-locked) behaviour instead of producing garbage.
 */

import type { PhaseMarkers } from '@/types/analysis';

export type DiscLayer = 'shoulder' | 'hip';

/**
 * Per-phase, per-layer compression values. Jason's draft table
 * (PR-5.7), informed by Chrome MCP measurements on b3fea3f0 and
 * the competitor reference visual.
 *
 * Keys are the same field names used in `PhaseMarkers` for symmetry
 * with the time anchors. Values are dimensionless multipliers
 * applied to baseline_rx; 1.0 = full setup width.
 */
export const DEFAULT_COMPRESSION: Record<DiscLayer, Record<keyof PhaseMarkers, number>> = {
  shoulder: {
    setupTime:      1.00,
    topTime:        0.50,  // shoulders maximally closed at top of backswing
    transitionTime: 0.55,  // begin opening
    impactTime:     0.70,  // mostly open by impact
    finishTime:     0.60,  // settle into follow-through pose
  },
  hip: {
    setupTime:      1.00,
    topTime:        0.78,  // hip turn lags shoulder
    transitionTime: 0.78,  // hold at top until downswing fires
    impactTime:     0.65,  // hips open earlier than shoulders into impact
    finishTime:     0.70,  // settle
  },
};

/** Smoothstep01: 3t² − 2t³, clamped to [0,1]. */
function smoothstep01(t: number): number {
  const x = Math.max(0, Math.min(1, t));
  return 3 * x * x - 2 * x * x * x;
}

/** Validate phase markers are monotonically non-decreasing. */
function isMonotonic(p: PhaseMarkers): boolean {
  return p.setupTime <= p.topTime
      && p.topTime <= p.transitionTime
      && p.transitionTime <= p.impactTime
      && p.impactTime <= p.finishTime;
}

/**
 * Visual disc compression at video timestamp `ts`, for a given body
 * layer (shoulder or hip). Interpolates smoothstep-style between
 * adjacent phase anchors. Clamps to endpoint values outside the
 * setup→finish window. Returns 1.0 unconditionally if phase markers
 * are non-monotonic (degraded mode).
 *
 * Pure function — no React, no DOM, safe to call from the rAF loop.
 */
export function getPhaseCompression(
  ts: number,
  phases: PhaseMarkers,
  layer: DiscLayer,
  table: typeof DEFAULT_COMPRESSION = DEFAULT_COMPRESSION,
): number {
  if (!isMonotonic(phases)) return 1.0;
  const c = table[layer];
  const anchors: Array<[number, number]> = [
    [phases.setupTime,      c.setupTime],
    [phases.topTime,        c.topTime],
    [phases.transitionTime, c.transitionTime],
    [phases.impactTime,     c.impactTime],
    [phases.finishTime,     c.finishTime],
  ];
  if (ts <= anchors[0][0]) return anchors[0][1];
  if (ts >= anchors[4][0]) return anchors[4][1];
  for (let i = 0; i < 4; i++) {
    const [t0, v0] = anchors[i];
    const [t1, v1] = anchors[i + 1];
    if (ts >= t0 && ts <= t1) {
      const span = t1 - t0;
      const u = span > 0 ? (ts - t0) / span : 0;
      return v0 + (v1 - v0) * smoothstep01(u);
    }
  }
  return 1.0;
}

/**
 * Micro-correction in [-0.1, +0.1] derived from live kp distance vs
 * setup baseline. Mapping (per Jason's PR-5.7 spec):
 *
 *   turnRatio = currentDist / baselineDist
 *   k         = clamp(turnRatio, 0.5, 1.0)
 *   micro     = (k − 0.75) × 0.4
 *
 * Resulting micro range:
 *   currentDist = 0.5× baseline → k=0.5 → micro = −0.10
 *   currentDist = 0.75× baseline → k=0.75 → micro = 0.00
 *   currentDist = 1.0× baseline (or larger) → k=1.0 → micro = +0.10
 *
 * Note the asymmetric mapping: shoulders/hips foreshortening below
 * baseline produces negative micro (slight extra shrink on top of
 * phase compression); a wider-than-baseline kp pair (camera moved
 * in) produces positive micro. The intent is to add subtle
 * "breathing" without letting kp foreshortening dominate phase
 * compression — hence the ±10% clamp.
 *
 * Returns 0 if baselineDist is non-positive (defensive).
 */
export function computeMicroCorrection(
  currentDist: number,
  baselineDist: number,
): number {
  if (baselineDist <= 0) return 0;
  const turnRatio = currentDist / baselineDist;
  const k = Math.max(0.5, Math.min(1.0, turnRatio));
  return (k - 0.75) * 0.4;
}
