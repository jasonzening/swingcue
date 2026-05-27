# PR-8d.2 Part 2 — Result-page state UX spec (DRAFT)

Status: **DRAFT — no implementation.** Spec only. Implementation
gated on Jason sign-off.

Scope: design the user-visible UX for all 5 lifecycle states of a
swing-analysis result page. Replaces the placeholder PR-8d.0 screens
with a coherent set that respects the body-width limitation
documented in PR-8h.0.

## State inventory & data-source mapping

| State name (spec) | `whamUiState.kind` | Trigger condition | Current PR-8d.0/8d.1 behavior |
|---|---|---|---|
| **processing** | `processing` | `wham_status='processing'`, polled every 2s | `ProcessingScreen` with spinner + ETA |
| **ready** | `ready` + timeline object | `wham_status='ready'` AND `wham_pose_timeline` rows fetched | SwingPlayer + WhamSkeletonOverlay + PR-8d.2 Part 1 disclaimer |
| **failed_preprocessing** | `failed_preprocessing` | `wham_failure_stage IN ('preprocessing','timeout')` with user-recoverable reason | `FailedScreen` with reason copy + Try again |
| **failed_system** | `failed_other` (code name) | `wham_failure_stage IN ('dispatch','download','slam_init','inference','postprocess','unknown')` | `FailedScreen` generic + Try again + support ref |
| **legacy_absent** | `absent` | No `wham_status` field on row (pre-WHAM upload era) | Falls through to MediaPipe placeholder UI |

**Note on terminology.** The code uses `failed_other`; the spec uses
`failed_system` because that's more user-charitable when surfaced in
the code comments / support reference. Code rename can be deferred or
bundled with implementation. For now, both names refer to the same
branch.

## Cross-state design principles

1. **Honesty over polish.** Never imply more precision than WHAM
   delivers. The PR-8h.0 audit established that body alignment is
   approximate. UX copy and CTAs respect that.
2. **Reversibility default.** Every error state offers a path back
   (Try again → upload, or ← History). No dead ends.
3. **Never expose internals.** Python tracebacks, Modal call IDs,
   raw stage names — none leak to the user. Support reference (short
   hash) is the only diagnostic surface.
4. **Inline disclaimer, not modal.** Limitation messaging is part of
   the screen, not a popup. Modals are reserved for actions.
5. **One-eyed-friend tone.** Direct, plain, honest. No hype copy
   ("powered by AI", "advanced analytics"). No emoji.
6. **Palette consistency.** Uses the existing locked palette from
   `docs/decisions/PR-8d_PALETTE.md` — accent `#a8f040`, muted text
   `#7a8a72`, deeper muted `#5a6a55`, base `#050805`.

---

## State 1 — `processing`

### When

Polling the `swing_analysis.video_metadata_json` until `wham_status`
changes from `'processing'` to `'ready'` / `'failed_*'`. Poll cadence:
2s with exponential backoff cap 8s (PR-8d.0 R3). Maximum elapsed
shown: no cap, but message escalates at 30s past `expectedSeconds`
and again at 300s.

### Layout hierarchy (top to bottom)

```
┌─────────────────────────────────────────────────┐
│ ← History          [back button, top-left]      │
│                                                  │
│            ╭───────────╮                         │
│            │ stage anim│  [animated progress     │
│            ╰───────────╯   indicator, ~120px]   │
│                                                  │
│       Building your 3D analysis                  │  [headline, 22px bold]
│                                                  │
│       Step 2 of 4 · Pose detection               │  [stage hint, 14px muted]
│                                                  │
│   Usually takes about 60 seconds                 │  [detail, 14px subtle]
│                                                  │
│         Elapsed: 23s · target 60s                │  [counter, 12px mono]
│                                                  │
│                                                  │
│  ─────────────────────────────────────────────  │
│  We can't show analysis quality in advance.      │
│  When it loads, body alignment will be           │
│  approximate (coaching anchor, not precise).     │  [pre-disclaimer, 11px]
└─────────────────────────────────────────────────┘
```

### Exact copy

**Headline** (changes with elapsed time):

| Condition | Headline |
|---|---|
| `elapsed ≤ expectedSeconds + 30s` | `Building your 3D analysis` |
| `expectedSeconds + 30s < elapsed ≤ 300s` | `Still analyzing…` |
| `elapsed > 300s` | `Analysis is taking too long` |

**Stage hint** (NEW — see "Stage estimation" below):

5-stage time-based bucketization. Stages: `Uploading` (already done,
not shown) · `Preparing` · `Detecting pose` · `Analyzing motion` ·
`Rendering`. Frontend shows `Step N of 4 · <stage>` based on
`elapsed / expectedSeconds` fraction. Stages map to fraction ranges:

| Fraction of expectedSeconds | Stage label shown |
|---|---|
| `[0, 0.10)` | `Step 1 of 4 · Preparing` |
| `[0.10, 0.45)` | `Step 2 of 4 · Pose detection` |
| `[0.45, 0.85)` | `Step 3 of 4 · Analyzing motion` |
| `[0.85, ∞)` | `Step 4 of 4 · Rendering` |

After `expectedSeconds + 30s` elapsed → freeze on `Step 4 of 4 ·
Finishing up` (don't roll back, don't claim Step 5). After `300s`
elapsed → hide stage hint entirely (just show "taking too long"
escalation).

**Detail line** (changes with elapsed):

| Condition | Detail |
|---|---|
| `elapsed ≤ expectedSeconds` | `Usually takes about Ns.` (N = expectedSeconds) |
| `elapsed > expectedSeconds + 30s` | `Taking longer than expected — hang tight.` |
| `elapsed > 300s` | `We're looking into it. You can retry from the upload page.` |

**Counter line:**

```
Elapsed: 23s · target 60s
```

Always shown. After `elapsed > expectedSeconds + 30s`, drop the
"target" segment — only `Elapsed: 91s`.

**Pre-disclaimer** (NEW — bottom):

```
We can't show analysis quality in advance. When it loads,
body alignment will be approximate (a coaching anchor,
not a precise measurement).
```

Style: 11px, `#5a6a55`, centered, max-width 320px, with a horizontal
hairline divider above it. Purpose: set expectations BEFORE the
skeleton overlay loads so the ready-state disclaimer isn't a surprise.

### CTAs

- **`← History`** (top-left): jumps to `/history`. Always present.
- **No retry CTA** during processing. The job is in flight; retry
  would create a duplicate Modal call.
- After `elapsed > 300s`: detail line points to upload page as
  manual recovery, but no auto-redirect. User stays in control.

### ETA behavior

`expectedSeconds` is computed at upload time as
`max(60, int(durationSec * 13))` (PR-8c.1 formula). Stored in
`swing_analysis.video_metadata_json.wham_expected_seconds`. Frontend
reads it once on processing-state entry and uses it for both the
"target" counter and the stage-bucketization fractions.

### Animation

Replaces the current solid spinner ring with a 4-segment progress
ring that fills clockwise as stage progresses. Segments are visually
distinct (gap between each). Active segment pulses subtly (1s
breathe). Completed segments are solid `#a8f040`. Unfilled are
`rgba(168,240,64,0.12)`.

CSS-only animation. No JS animation loop. Targets ~60fps.

### Fixtures

| Fixture | When state fires |
|---|---|
| Any new upload | Hits processing branch ~60-120s during Modal infer_video call |
| `wham_status='processing'` directly forced via SQL | Manual test mode |

There is no permanent processing fixture in the DB. Live-test only on
next fresh upload.

---

## State 2 — `ready`

### When

`wham_status='ready'` AND `wham_pose_timeline` fetched AND `wham_video_meta` fetched
successfully. The hot path for the user — when this lands, MVP value
delivered.

### Layout (unchanged from current PR-8d.1 / Part 1)

```
┌─────────────────────────────────────────────────┐
│ ← SwingCue            + New                      │  [header bar]
├─────────────────────────────────────────────────┤
│                                                  │
│            [video player + WHAM skeleton]       │  [SwingPlayer]
│                                                  │
│                                                  │
├─────────────────────────────────────────────────┤
│  Body alignment is approximate — used as         │  [Part 1 disclaimer]
│  a coaching anchor.                              │
└─────────────────────────────────────────────────┘
```

### Copy

**Disclaimer** (Part 1 shipped):
```
Body alignment is approximate — used as a coaching anchor.
```

No other state-specific copy. The SwingPlayer + WhamSkeletonOverlay
deliver the content.

### CTAs

- **`←`** (header back): `/history`
- **`+ New`** (header right): `/upload`
- **No mid-page CTA.** The video itself is the interactive surface.

### Edge case: `ready + null` timeline (R5)

`wham_status='ready'` but `wham_pose_timeline` rows empty (race during
Modal write). Show soft "preparing" screen with Refresh button. This
was shipped in PR-8d.1 — keep as-is.

```
[spinner]
Analysis data is still preparing
Please refresh in a moment.
[Refresh button]
```

### Edge case: `ready + undefined` timeline (fetch in flight)

Light spinner with `Loading 3D analysis…` text. Shipped in PR-8d.1.
Keep as-is.

### Fixtures

| Fixture | Notes |
|---|---|
| `a3f7b0d8-99d0-4f0e-84f3-41abd95ceaea` | T1 face-on far, full skeleton |
| `7e49a385-740d-4704-ba03-bd59426ed704` | Latest 1-5miao-1pr-face reupload |
| `39dab3eb-993c-4425-88a8-b04e07ba7ab9` | Jason self-record test_swing |

### Acceptance

Disclaimer visible. WhamSkeletonOverlay renders cyan/amber/white
17-joint skeleton. No MediaPipe placeholder. No "Head Movement"
template coaching cue.

---

## State 3 — `failed_preprocessing`

### When

Modal's `_preflight_check_video()` rejected the video before any
inference work started. Reasons stored in `wham_failure_stage`:
`'preprocessing'` (scene-cut detected OR duration < 3s) or
`'timeout'` (rare — preflight took too long). User-recoverable —
the user can rerecord and try again.

### Layout

```
┌─────────────────────────────────────────────────┐
│ ← History          [back button]                 │
│                                                  │
│              [!]   [yellow warning icon, 64px]   │
│                                                  │
│       Couldn't analyze this video                │  [headline, 22px bold]
│                                                  │
│   Video too short — needs at least 3 seconds.    │  [reason, 16px]
│                                                  │
│   What to try:                                   │  [help section, 14px]
│   • Record a longer clip (4-8 seconds works     │
│     best)                                        │
│   • Use a single continuous take                 │
│   • Avoid scene cuts or zoom changes             │
│                                                  │
│              [Try again →]                       │  [primary CTA]
└─────────────────────────────────────────────────┘
```

### Copy variants by `wham_failure_stage`

| Failure detail | Reason copy |
|---|---|
| `duration < 3s` | `Video too short — needs at least 3 seconds.` |
| `scene cut detected` | `Multi-scene video detected — please upload a single continuous swing.` |
| `preprocessing timeout` | `Preprocessing timed out — please try a shorter clip.` |
| unknown preprocessing reason | `Couldn't process this video. Try a shorter, single-take clip.` |

Reason comes from `swing_analysis.video_metadata_json.wham_user_message`
(set by Modal in PR-8c.4). NEVER show raw exception text. NEVER show
Python traceback.

**"What to try" section** (NEW — replaces just a generic reason):

```
What to try:
• Record a longer clip (4-8 seconds works best)
• Use a single continuous take
• Avoid scene cuts or zoom changes
```

Static — same for all preprocessing failures. Removes ambiguity
about what "Try again" means.

### CTAs

- **`Try again →`** (primary, accent button): navigates to `/upload`
- **`← History`** (corner): `/history`
- No "Report a bug" — preprocessing failures are user-fixable, not
  bugs.

### Fixtures

| Fixture | Failure detail |
|---|---|
| `0e9153db-65a8-4ee0-9100-71f9d4fee65b` | "Video too short" |
| `b353a7ca-...` | "Multi-scene detected" |

---

## State 4 — `failed_system`

### When

Any non-preprocessing failure stage: `dispatch`, `download`,
`slam_init`, `inference`, `postprocess`, `unknown`. NOT user-
recoverable in a meaningful way — the user did nothing wrong. The
system failed.

### Layout

```
┌─────────────────────────────────────────────────┐
│ ← History          [back button]                 │
│                                                  │
│              [!]   [red error icon, 64px]        │
│                                                  │
│            Analysis failed                       │  [headline, 22px bold]
│                                                  │
│   Something went wrong on our end while          │  [message, 16px]
│   analyzing your swing. This isn't your video    │
│   — we're already looking into it.               │
│                                                  │
│   You can:                                       │  [help section, 14px]
│   • Try uploading again (sometimes it just       │
│     works the second time)                       │
│   • Send the error reference below if it keeps   │
│     failing                                      │
│                                                  │
│              [Try again →]                       │  [primary CTA]
│                                                  │
│   Error reference: A7F3-K2                       │  [hash, 12px mono]
└─────────────────────────────────────────────────┘
```

### Copy

**Headline** (constant): `Analysis failed`

**Message** (constant):
```
Something went wrong on our end while analyzing your swing.
This isn't your video — we're already looking into it.
```

The "this isn't your video" line is intentional — distinguishes
this failure from `failed_preprocessing` where the video IS the
problem.

**Help section** (NEW — gives the user agency):
```
You can:
• Try uploading again (sometimes it just works the second time)
• Send the error reference below if it keeps failing
```

**Error reference** (existing R6 behavior, kept):
```
Error reference: A7F3-K2
```

Format: `shortHash(videoId + failureStage)` — stable per
(video, stage) tuple so support can match logs. 7-character
alphanumeric, separator for readability.

**Never shown:** the actual `wham_error_message` field. May contain
Python traceback, Modal call ID, internal paths, etc. R6 of PR-8d.0.

### CTAs

- **`Try again →`** (primary): `/upload`
- **`← History`** (corner): `/history`
- No "Contact support" CTA (no support flow exists yet). Reference
  hash is the contact handoff.

### Fixtures

| Fixture | failure_stage |
|---|---|
| `b8d9e821-...` | `inference` |
| `88ef44a5-...` | `null` (very early failure, no stage assigned) |

---

## State 5 — `legacy_absent`

### When

`wham_status` field doesn't exist on the row (pre-WHAM era uploads).
These rows have MediaPipe-based `pose_timeline_2d` data only. No 3D
skeleton, no WHAM analysis.

### Current behavior (broken from a user-perspective)

Falls through to the SwingPlayer, which renders the legacy MediaPipe
placeholder dots + the static "Head Movement / Keep your head
centered" coaching cue. No indication to the user that this is OLD
analysis. Confusing.

### Proposed layout (NEW)

Two design options — pick one. Both replace the silent fallback.

#### Option A: Banner over legacy view

Keep the SwingPlayer rendering the legacy MediaPipe view, but add a
dismissible banner at the top:

```
┌─────────────────────────────────────────────────┐
│ ← SwingCue            + New                      │
├─────────────────────────────────────────────────┤
│  Older analysis — limited features.              │  [banner, 11px]
│  [Re-upload for 3D analysis →]                   │  [inline CTA]
├─────────────────────────────────────────────────┤
│           [video + MediaPipe placeholder]        │
│                                                  │
│  ⚡  Head Movement                                │  [legacy cue, unchanged]
│  "Keep your head centered through impact"        │
└─────────────────────────────────────────────────┘
```

Banner CSS: `background: rgba(168,240,64,0.06)`, `border-bottom:
1px solid rgba(168,240,64,0.15)`, `padding: 8px 14px`. Sticks below
header. Not dismissible (gentle reminder, not a popup).

Pros: User sees their old video. Cheap to implement.
Cons: Still shows imprecise MediaPipe dots that look broken to a
new-WHAM-era user.

#### Option B: Replacement screen with re-upload CTA

Don't render the video at all. Treat legacy as a "this is old data"
state:

```
┌─────────────────────────────────────────────────┐
│ ← History          [back button]                 │
│                                                  │
│              [📁]  [archive icon, 64px]          │
│                                                  │
│       Older swing — no 3D analysis               │  [headline, 22px]
│                                                  │
│   This swing was analyzed before SwingCue        │
│   added 3D body tracking. The original video    │
│   is still available below.                      │
│                                                  │
│              [Re-upload for 3D →]                │  [primary CTA]
│                                                  │
│   ─────  Original video below  ─────             │  [divider]
│                                                  │
│          [video player, no overlay]              │
└─────────────────────────────────────────────────┘
```

The original video plays without any pose overlay (no MediaPipe
dots, no skeleton). Pure playback.

Pros: Clear honest state. No "broken-looking" overlay.
Cons: User loses the existing MediaPipe placeholder analysis
entirely. Some users may have built workflows around it.

**Recommendation: Option B.** The MediaPipe placeholder doesn't
deliver real value (it's a few dots over the video), and showing it
now in the post-WHAM era invites comparison the legacy data loses.
A clean "this is old, here's a clear path forward" UX is more
honest.

### Copy (Option B)

**Headline:** `Older swing — no 3D analysis`

**Message:**
```
This swing was analyzed before SwingCue added 3D body tracking.
The original video is still available below.
```

**Divider:** `─── Original video below ───` — 12px muted text,
centered, with hairlines on each side.

### CTAs (Option B)

- **`Re-upload for 3D →`** (primary): `/upload` — passes a query
  param `?from=legacy&original=<videoId>` so the upload page could
  optionally show "Replacing your older video" context. Out-of-scope
  for Part 2, but reserve the query-param contract.
- **`← History`** (corner): `/history`

### Fixtures

| Fixture | Notes |
|---|---|
| `b32e0f21-...` | Pre-WHAM upload, MediaPipe-only |
| `9f0d9c6a-...` | Same |
| `5bbcfbc8-49b9-4fc4-8b0e-a34c5427aa62` | Pre-PR-8e WHAM but has `wham_status='ready'` (NOT legacy_absent — gets ready treatment with SMPL fallback) |

The distinction matters: `wham_status` field absent vs present-but-failed.

---

## Implementation phases (proposed)

Spec is one document; implementation can be split.

### Phase 2A (smallest viable change) — ~1 hour

- Stage hint bucketization in `ProcessingScreen`
- "Pre-disclaimer" footer on processing screen
- "What to try" section on `failed_preprocessing`
- "You can" + sharpened message on `failed_system`

No new screens. Just enrich existing PR-8d.0 components.

### Phase 2B (legacy_absent screen) — ~1 hour

- New `LegacyAbsentScreen` component (Option B above)
- Wire into existing `whamUiState.kind === 'absent'` fall-through
- Update fixture table

### Phase 2C (animated 4-segment progress ring) — ~1 hour

- Replace solid spinner with segmented ring
- CSS-only animation
- Visual polish

Phases 2A → 2B → 2C in order. 2A delivers most user-perceived
value at lowest cost.

## What's NOT in this spec

- **Real-time backend stage updates.** Stage hints are time-based
  estimation, not actual backend signal. A future PR could add
  `wham_processing_stage` field updates from Modal every few
  seconds, but it requires backend changes and is not in PR-8d.2.
- **Phase detection on the ready video** (setup/backswing/top/
  impact/finish auto-detection). That's PR-8d.3+ scope.
- **Real WHAM-derived coaching cues.** "Head Movement / Keep your
  head centered" template is still hidden in ready mode per PR-8d.1
  R4 — no new cue is being added here.
- **Per-user calibration input** to fix body width. PR-8h.1 is DEAD
  per PR-8h.0 audit.

## Open questions for sign-off

1. **Option A vs Option B for `legacy_absent`?** Spec recommends B
   (clean state, re-upload CTA, no MediaPipe overlay). User may
   prefer A (banner over legacy view) for backward sentiment.
2. **Stage hint copy:** `Pose detection` vs `Detecting body`? `Analyzing
   motion` vs `3D reconstruction`? Spec uses simpler/user-facing terms.
   Pick a vocabulary and lock it.
3. **"What to try" + "You can" sections — keep as bullets, or
   convert to short paragraph?** Spec uses bullets for scanability.
4. **Phase rollout** — ship 2A → 2B → 2C as separate commits, or
   bundle into one PR?
5. **Pre-disclaimer on processing screen** — is the "we can't show
   quality in advance" copy too defensive? Could be softened to
   just match the post-load disclaimer ("Body alignment will be
   approximate — used as a coaching anchor").

## Acceptance gates (when implementation ships)

Per state, verify on Vercel deploy:

- [ ] `processing`: trigger a fresh upload, observe ETA + stage
      hint sequence + pre-disclaimer. Headline/detail transitions
      at 30s and 300s elapsed.
- [ ] `ready`: 3 existing ready fixtures still render skeleton +
      Part 1 disclaimer. No regression.
- [ ] `failed_preprocessing`: 2 fixtures show new "What to try"
      section. Reason copy matches the failure type.
- [ ] `failed_system`: 2 fixtures show new "You can" section.
      Error reference present and stable per (video, stage).
- [ ] `legacy_absent` (Option B): 2 fixtures show new screen.
      Original video plays. No MediaPipe placeholder dots.

## Files this spec will touch (when implementing)

- `src/app/result/[id]/page.tsx` — state-screen branches, props
- `src/components/SwingPlayer.tsx` — possibly the legacy-absent
  Option B rendering path (no overlay mode)
- `src/components/*` — maybe a new `LegacyAbsentScreen.tsx` if
  Option B is picked
- Possibly `docs/decisions/PR-8d_PALETTE.md` for any new colors
- No backend changes (stage hints are time-based estimation)
- No DB schema changes
- No Modal changes

## Cross-reference

- PR-8d.0 — wham_status state machine and initial processing/failed
  screens (`docs/decisions/SWING_VIDEOS_STATE_MACHINE.md`)
- PR-8d.1 — WHAM skeleton rendering on ready branch
- PR-8d.2 Part 1 — body-alignment disclaimer (commit `dc9ecd6`)
- PR-8e_CLOSED.md — body-width limitation root cause
- PR-8h_CALIBRATION_AUDIT.md — why per-user correction was rejected
- PR-8d_PALETTE.md — locked colors

---

End of draft. Awaiting sign-off on open questions before
implementation.
