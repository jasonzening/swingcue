/**
 * PR-5 hotfix: angle unwrap for continuous disc rotation.
 *
 * `Math.atan2` returns values in [-π, π]. When shoulder/hip rotation
 * crosses the ±π boundary between adjacent frames, the disc would
 * visually jump 360°. `unwrapAngle` keeps the running angle on a
 * continuous real line.
 *
 * If the time delta between consecutive samples is large (seek / phase
 * jump > SEEK_THRESHOLD_S), we treat it as a fresh start and do NOT
 * unwrap — otherwise a scrub would accumulate stale offsets that
 * compound over time.
 *
 * Stateless by design. Caller (SwingPlayer) is responsible for storing
 * `(prevAngle, prevT)` between rAF ticks via useRef.
 */

const SEEK_THRESHOLD_S = 0.2;

export function unwrapAngle(
  curr: number,
  prev: number | null,
  prevT: number | null,
  currT: number,
): number {
  if (prev === null || prevT === null) return curr;
  if (Math.abs(currT - prevT) > SEEK_THRESHOLD_S) return curr;
  let diff = curr - prev;
  while (diff > Math.PI)  diff -= 2 * Math.PI;
  while (diff < -Math.PI) diff += 2 * Math.PI;
  return prev + diff;
}
