# PR-5.7 audit — phase signal availability + accuracy

**Date**: 2026-05-18
**Scope**: Read-only investigation of whether the frontend phase
signal is wired up and accurate enough to drive PR-5.7's
phase-driven compression curve. No code changes, no PR.

---

## §1 Phase signal source — how SwingPlayer knows the current phase

### Wiring (frontend, top → bottom)

1. **DB read** (`src/app/result/[id]/page.tsx:85`):
   ```ts
   const pm: PhaseMarkers = (ana.phase_markers_json as PhaseMarkers | null) ?? {
     setupTime: 0,
     topTime: dur * 0.50,
     transitionTime: dur * 0.62,
     impactTime: dur * 0.75,
     finishTime: dur * 0.92,
   };
   setPhases(pm);
   ```
   The `swing_analysis.phase_markers_json` JSONB column is the
   source of truth. Missing → proportional fallback (50%/62%/75%/92%
   of duration).

2. **Prop passed** (`src/app/result/[id]/page.tsx:190`):
   ```ts
   <SwingPlayer videoUrl={…} timeline={overlayTimeline} phases={phases} duration={meta.durationSec} … />
   ```

3. **Per-frame lookup** (`src/components/SwingPlayer.tsx:256`,
   inside the rAF `renderTick`):
   ```ts
   setPhase(getCurrentPhase(phases, t, d));
   ```
   where `t = v.currentTime` (HTML5 video clock), `d = v.duration
   || dur || 1`.

4. **`getCurrentPhase`** (`src/lib/overlay/playerSync.ts:40-58`):
   ```ts
   export function getCurrentPhase(
     phases: PhaseMarkers,
     currentTime: number,
     duration: number,
   ): 'setup' | 'top' | 'transition' | 'impact' | 'finish' {
     const normT = duration > 0 ? currentTime / duration : 0;
     const p = {
       setup:      phases.setupTime / duration,
       top:        phases.topTime / duration,
       transition: phases.transitionTime / duration,
       impact:     phases.impactTime / duration,
       finish:     phases.finishTime / duration,
     };
     if (normT >= p.finish)     return 'finish';
     if (normT >= p.impact)     return 'impact';
     if (normT >= p.transition) return 'transition';
     if (normT >= p.top)        return 'top';
     return 'setup';
   }
   ```

### Two more parallel implementations of the same logic

| Site | File:line | Notes |
|---|---|---|
| `getCurrentPhase` | `src/lib/overlay/syncSpec.ts:232-241` | Same threshold cascade. No normalisation — compares raw seconds. Unused by SwingPlayer but present in spec/test surface. |
| `computePhase` (private) | `src/lib/overlay/templates/index.ts:179-194` | Same cascade, used by `generateDenseOverlayTimeline` at build time to stamp `frame.phase` on each `OverlayFrame`. |

Three implementations, one algorithm. Any PR-5.7 helper should
reuse — or share an interpolating variant with — `playerSync.getCurrentPhase`.

---

## §2 b3fea3f0 phase data — actual values

### ⚠ Cannot query from this sandbox

No `NEXT_PUBLIC_SUPABASE_URL` / `SUPABASE_SERVICE_ROLE_KEY` in env.
Same blocker as the PR-3.1 audit (§3 of that doc). I can read code
and prior docs, but cannot run SQL against the live DB. **Request:**
run this from your machine via Supabase MCP and paste the output:

```sql
SELECT
  v.id AS video_id,
  v.status,
  v.created_at,
  (v.metadata_json ->> 'durationSec')::float AS duration_sec,
  a.phase_markers_json
FROM swing_videos v
LEFT JOIN swing_analysis a ON a.video_id = v.id
WHERE v.id = 'b3fea3f0-e248-44d7-a923-0bb43172b5bf';
```

### Indirect signals from existing docs

Two cached references in repo, partially conflicting:

| Source | Claim | Likely interpretation |
|---|---|---|
| `docs/PR-3.1_POSE_DATA_AUDIT.md:56-57` | "setup phase, which is at ts≈0.3 in the user's video" | b3fea3f0 `setupTime ≈ 0.3s` |
| `docs/PR-5_DISC_OFFSET_AUDIT.md:261-275` | `currentTime=3.286s` used as a worked example for `frameAt()` | Not necessarily `setupTime` — looks like a Jason scrub-position during testing, but the exact mapping is unstated. Worth confirming whether 3.286 is a phase boundary in this video. |

If PR-3.1's note is accurate (`setupTime ≈ 0.3s`) AND your prompt's
description holds (6s video, real swing ~1.5s, top/impact
mis-labelled), the most likely DB shape is something like:

```jsonc
// HYPOTHESIS — verify with SQL above
{
  "setupTime":      0.30,
  "topTime":        2.50,  // detected on a noise peak during static setup
  "transitionTime": 3.10,
  "impactTime":     3.80,
  "finishTime":     5.40
}
```

Real swing on this hypothesis lives at ts ≈ [4.5s, 6.0s] but the
labelled `topTime` and `impactTime` sit ~1.5–2s **before** real top.
**This is the failure mode you described** and the algorithm
inspection in §6 below shows exactly why it happens.

---

## §3 Phase transition logic — hard switch or interpolated?

**Hard switch, everywhere.** All three implementations
(`playerSync.getCurrentPhase`, `syncSpec.getCurrentPhase`,
`templates.computePhase`) use the same cascading `>=` pattern:

```
if (t >= finish)     → finish
else if (t >= impact)     → impact
else if (t >= transition) → transition
else if (t >= top)        → top
else                      → setup
```

At `t = topTime` the return value flips **instantly** from `setup`
→ `top` on the very frame whose `currentTime` crosses the threshold.
No fade, no lerp, no anticipatory window.

### Confirmed: no interpolation infrastructure exists anywhere

- Grepped `interpolat|lerp|smooth.*phase` across `src/lib/overlay/*`,
  `src/components/*` — zero matches related to phase smoothing.
- The only interpolation in the system is keypoint-level in
  `gap_fill_linear` (backend) and frame-level in
  `generateDenseOverlayTimeline` (overlay densification by time),
  neither of which deals with phase semantics.
- `OverlayTimeline.frames[].phase` (from `templates.computePhase`)
  is a string label per pre-rendered frame, also hard-bucketed at
  build time.

There is no existing "what fraction of `top` are we in" signal that
PR-5.7 can consume.

---

## §4 PR-5.7 compression curve — implementation path

### Why hard switch is unacceptable for PR-5.7

With per-phase compression values like
`{setup: 1.0, top: 0.5, transition: 0.7, impact: 0.85, finish: 0.5}`:

- At `t = topTime - 1ms`, compression = `1.0` (still setup).
- At `t = topTime + 1ms`, compression = `0.5` (now top).
- Disc rx flips from `baseline × 1.0` to `baseline × 0.5` in **one
  rAF frame (16ms)** — a visual snap, not a smooth shrink.

For a player-perceived smooth coaching plane, this is a regression
even versus PR-5.6's constant-size disc.

### Recommended: phase-segment linear interpolation

Implement a sibling to `getCurrentPhase` — proposed name
`getPhaseCompression()` — that returns a continuous scalar:

```ts
// proposal — NOT committed
const COMPRESSION_BY_PHASE = {
  setup:      1.0,
  top:        0.5,   // values TBD by Jason
  transition: 0.7,
  impact:     0.85,
  finish:     0.5,
} as const;

export function getPhaseCompression(
  phases: PhaseMarkers,
  currentTime: number,
): number {
  const pts: Array<[keyof PhaseMarkers, number, number]> = [
    ['setupTime',      phases.setupTime,      COMPRESSION_BY_PHASE.setup],
    ['topTime',        phases.topTime,        COMPRESSION_BY_PHASE.top],
    ['transitionTime', phases.transitionTime, COMPRESSION_BY_PHASE.transition],
    ['impactTime',     phases.impactTime,     COMPRESSION_BY_PHASE.impact],
    ['finishTime',     phases.finishTime,     COMPRESSION_BY_PHASE.finish],
  ];
  // Clamp at endpoints
  if (currentTime <= pts[0][1]) return pts[0][2];
  if (currentTime >= pts[4][1]) return pts[4][2];
  // Find segment & lerp
  for (let i = 0; i < 4; i++) {
    const [, t0, v0] = pts[i];
    const [, t1, v1] = pts[i + 1];
    if (currentTime >= t0 && currentTime <= t1) {
      const span = t1 - t0;
      const u = span > 0 ? (currentTime - t0) / span : 0;
      return v0 + (v1 - v0) * u;
    }
  }
  return 1.0; // unreachable but type-safe fallback
}
```

Then SwingPlayer rAF block:
```ts
const compression = getPhaseCompression(phases, t);
const fixedRx = (discAnchorRef.current?.shoulderRx ?? shoulder.rx) * compression;
```

Cost: O(5) per rAF tick → trivial. No re-render trigger (pure-function in render path).

### "Micro ±10% currentDist" detail from your spec

If PR-5.7 also wants to add a small currentDist modulation on top
of the phase compression, the natural form is:

```ts
const liveDist = …;             // current kp pair distance
const baselineDist = anchor.shoulderRx * 2 / DISC_RX_RATIO;
const micro = clamp(liveDist / baselineDist, 0.9, 1.1);
const finalRx = anchor.shoulderRx * compression * micro;
```

This keeps phase as the dominant signal, with kp dist as a ±10%
secondary nudge — matches "phase-driven (主控), keypoints (anchor),
currentDist (micro ±10%)".

---

## §5 Fallback when phase signal is missing or wrong

### Three known failure modes + what happens today

| Failure mode | Current behaviour | Adequate for PR-5.7? |
|---|---|---|
| **DB row has `phase_markers_json = null`** | `page.tsx:85` falls back to **proportional** `{ setupTime: 0, topTime: dur*0.5, transitionTime: dur*0.62, impactTime: dur*0.75, finishTime: dur*0.92 }`. SwingPlayer is unaware — it sees a valid `PhaseMarkers`. | Good enough — proportional values are monotonic and span the duration. PR-5.7 will produce smooth (but not swing-correct) compression curve. |
| **DB row exists but phases are non-monotonic** (e.g. `topTime < setupTime`, or backend misdetection puts `impactTime < topTime`) | **No validation anywhere.** `getCurrentPhase` cascade still fires — but the cascade assumes `setup ≤ top ≤ transition ≤ impact ≤ finish`. Out-of-order phases produce phase-label glitches AND would break the §4 lerp helper's segment search. | **NOT adequate.** PR-5.7 must validate monotonicity. |
| **Backend phase detection mis-labels a setup-heavy video** (per your prompt — actual swing is the last 1.5s but `topTime/impactTime` got assigned to noise peaks within the long setup) | DB stores plausible-looking but semantically wrong phase markers. Frontend has **no signal** that detection failed. | Hardest case — looks the same as a correct row. PR-5.7's compression curve fires at the wrong times. See §6. |

### Recommended fallback policy for PR-5.7

In the new `getPhaseCompression()` helper, add a validation step at
the top:

```ts
function arePhasesMonotonic(p: PhaseMarkers): boolean {
  return p.setupTime <= p.topTime
      && p.topTime <= p.transitionTime
      && p.transitionTime <= p.impactTime
      && p.impactTime <= p.finishTime;
}

export function getPhaseCompression(phases: PhaseMarkers, t: number): number {
  if (!arePhasesMonotonic(phases)) return 1.0;  // safe — disc stays at baseline
  // …rest…
}
```

Console-warn once per session if non-monotonic (helps debugging
without spamming).

For the "detection silently wrong" mode (third row above), **no
frontend fix is possible** without an out-of-band signal. The
correct response is at the analysis layer:
- Have `phase_detector.py` emit a `phaseConfidence` score, OR
- Sanity-check derived metrics (top→impact interval < 0.5s ⇒ likely
  mis-detected).
Out of scope for PR-5.7 itself.

---

## §6 Blockers & flags

### B1 — `phase_detector.py` has a hard 75% search window

`python/phase_detector.py:82` caps the `topTime` search to the first
75% of sampled frames:

```python
search_end = max(1, int(len(wrist_y_smooth) * 0.75))
top_idx_raw = int(np.argmin(wrist_y_smooth[:search_end]))
```

For your "6s video, swing only in last 1.5s" scenario this is
**catastrophic**: if real top happens at `t = 4.8s` in a 6s video,
the corresponding frame index lives at ~80% — **outside the search
window**. The detector necessarily picks some `wrist_y` jitter peak
within the first 4.5s of static setup as "top", then derives
`setupTime ≈ 0.3` (capped) and `impactTime ≈ top + (dur-top)*0.55`
which lands further away from the real impact.

This is the algorithmic root of your "top/impact 标错" complaint.
**Fix is a backend ticket** (PR-3.2? PR-7? not PR-5.7) — needs to
either widen the search window or use an entirely different
detector (e.g., velocity-peak-based or club-head-tracked).

### B2 — No phase data validation at any layer

- Backend writes whatever `detect_phases` returns; no monotonicity
  check, no plausibility check.
- Frontend reads it raw at `page.tsx:85`; no validation.
- `getCurrentPhase` consumes it; cascade relies on ordering it
  doesn't verify.

PR-5.7's smooth lerp helper amplifies the consequences of bad data
(now visible as smoothly-wrong rather than discretely-wrong). Even
the §5 monotonicity guard helps only with the obvious cases.

### B3 — Three duplicate `getCurrentPhase` / `computePhase`
implementations

`playerSync.getCurrentPhase`, `syncSpec.getCurrentPhase`, and
`templates.computePhase` all encode the same threshold cascade.
Adding `getPhaseCompression()` as a fourth (and the prior three
silently retain hard-switch semantics) further diverges them. Worth
considering a single `src/lib/overlay/phaseSignal.ts` module that
exports both the discrete and continuous variants, and migrating
the other two callers — but that's a follow-up cleanup, not a
PR-5.7 blocker.

### B4 — `phase` React state in SwingPlayer triggers re-renders per phase change

`SwingPlayer.tsx:256` calls `setPhase(...)` inside the rAF loop.
React schedules a re-render on every phase transition (currently 4
per swing — top, transition, impact, finish). This is harmless
today because hard switches only fire once per phase. But if PR-5.7
ever needs the current compression value visible in JSX (e.g., to
display a "60%" badge), be careful not to put the continuous
compression in React state — keep it strictly in the canvas draw
path. Otherwise you'd trigger a re-render every rAF tick (~60Hz).

---

## §7 Recommendations summary

1. **Verify §2 b3fea3f0 phase data** with the SQL query above
   before committing to a PR-5.7 compression table. If phases are
   indeed mis-labelled (B1), PR-5.7 lerp will smoothly distribute
   wrong values across wrong intervals — i.e., the disc will
   *still* shrink at the wrong moments, just gracefully.
2. **Build PR-5.7 on a phase-segment lerp** (§4), not a hard
   switch.
3. **Validate monotonicity in the new helper** (§5); fall back to
   `compression = 1.0` when phases are invalid.
4. **File a separate backend ticket** for the 75% search window bug
   (B1) — PR-5.7 can ship usefully without this fix on
   well-detected videos but will under-deliver on setup-heavy ones
   until the upstream detector is fixed.
5. Keep `getCurrentPhase` (discrete) and `getPhaseCompression`
   (continuous) **co-located** in one module so the third-party
   migration (B3) is a single-file follow-up later.
