# PR-8d.2 Part 2 — Result-page state UX (canonical spec)

Status: **DRAFT — no implementation.** Spec only. Implementation
gated on Jason sign-off.

Supersedes earlier draft at `PR-8d.2_PART_2_UX_SPEC.md` (deleted in
this commit).

## Goal

Design the result-page UX across all 5 lifecycle states. Every copy
line annotated for honesty. Failure stage mapping made explicit.
Progress mechanism (real backend events vs client-side animation)
chosen and justified. Acceptance fixtures named with expected
behavior on each CTA.

## Constraints (locked)

- No implementation in this PR. Spec only.
- No JSX / CSS / TypeScript code in this doc.
- Match SwingCue product principle:
  **"Fast page load, slow trusted result. Never fast inaccurate
  result."**
- Honor PR-8d.0 / PR-8d.1 invariants:
  - **R4** Visual layer ≠ Analysis layer. Visual screens may use
    estimation; coaching metrics may not. (PR-8h.0 doc.)
  - **R6** Never expose Python tracebacks, Modal call IDs, internal
    paths. Short-hash reference only.
  - **Palette lock** (`docs/decisions/PR-8d_PALETTE.md`): accent
    `#a8f040` lime, muted `#7a8a72`, deeper muted `#5a6a55`, base
    `#050805`. Cyan `#00C2FF` / amber `#FFB000` reserved for skeleton.

## State inventory (5 states)

| # | State | Triggered by |
|---|---|---|
| 1 | `processing` | `wham_status='processing'` on `swing_analysis.video_metadata_json` |
| 2 | `ready` | `wham_status='ready'` AND `wham_pose_timeline` rows exist |
| 3 | `failed_preprocessing` | `wham_status='failed'` AND `wham_failure_stage='preprocessing'` |
| 4 | `failed_system` | `wham_status='failed'` AND `wham_failure_stage ≠ 'preprocessing'` |
| 5 | `legacy_absent` | `wham_status` key absent from `video_metadata_json` |

Code-side `whamUiState.kind` enum maps directly: `processing`,
`ready`, `failed_preprocessing`, `failed_other`, `absent`. The spec
uses `failed_system` and `legacy_absent` for user-charitable naming.
Rename is optional at implementation time.

---

## R2 — Backend stage → UX state mapping (locked)

PR-8c.3 defines 8 backend `wham_failure_stage` values. Collapsed to
2 user-facing failure states:

| Backend `wham_failure_stage` | UX state | Rationale |
|---|---|---|
| `'preprocessing'` | `failed_preprocessing` | Scene-cut or duration < 3s — **user can fix** by recording a different clip. |
| `'dispatch'` | `failed_system` | Modal call failed before reaching the runner. User can't fix. |
| `'download'` | `failed_system` | Modal couldn't fetch the signed URL. Storage / network — user can't fix. |
| `'slam_init'` | `failed_system` | DPVO/SLAM init failed inside Modal. Inference path — user can't fix. |
| `'inference'` | `failed_system` | WHAM forward pass crashed. User can't fix. |
| `'postprocess'` | `failed_system` | Projection / DB write failed. User can't fix. |
| `'timeout'` | `failed_system` | Modal hit job timeout. Conservatively system: most timeouts are inference-side, not preflight. (Future PR could disambiguate by elapsed-at-failure.) |
| `'unknown'` | `failed_system` | Exception didn't match any classifier. User can't fix. |

**Routing rule (one line):** `failed_preprocessing` iff
`wham_failure_stage === 'preprocessing'`; everything else
`failed_system`.

Missing `wham_status` key (truly absent from row) → `legacy_absent`.
Field present but value `null` → treat as `legacy_absent` defensively.

---

## R3 — Progress mechanism (locked)

### Choice: client-side animated, real `wham_status` as override

Backend currently writes `wham_status` only at 2 transitions:
`processing` → `ready` OR `processing` → `failed`. There is no
sub-stage event stream. Adding one would require PR-8c.6+ backend
instrumentation (Modal worker writes `wham_processing_stage` field
every ~3s, frontend polls). That's a separate PR's work and not in
PR-8d.2 scope.

For Part 2, run a **client-side animated progression** over the
expected duration, with **real `wham_status` flips as immediate
overrides**:

```
on processing-screen mount:
  startedAt = nowMs
  expectedSec = swing_analysis.video_metadata_json.wham_expected_seconds || 60
  
  every poll tick (2s, exp-backoff cap 8s per PR-8d.0):
    fetch wham_status
    
    if wham_status === 'ready':
      → exit processing screen, transition to ready state
      → DO NOT WAIT for animation to finish
    
    if wham_status === 'failed':
      → exit immediately to failed_preprocessing or failed_system
      → DO NOT WAIT
    
    if still 'processing':
      → keep current screen, recompute elapsed = (nowMs - startedAt) / 1000
      → update stage hint based on elapsed/expectedSec fraction (table below)
      → update headline + detail per elapsed thresholds
```

### Stage bucketization (timing function)

5 sub-stages mapped to elapsed/expectedSeconds fraction. Stage 1
shows briefly then advances. Stages 4-5 may freeze if backend takes
longer than expected (frontend never claims completion beyond what
backend has signaled).

| Fraction `f = elapsed / expectedSec` | Stage label | What's actually happening backend-side |
|---|---|---|
| `0.00 ≤ f < 0.05` | `Step 1 of 5 · Uploaded` | Modal dispatch ack |
| `0.05 ≤ f < 0.15` | `Step 2 of 5 · Preparing` | Video download + preflight (scene-cut, duration check) |
| `0.15 ≤ f < 0.50` | `Step 3 of 5 · Detecting pose` | YOLO + ViTPose 2D detection per frame |
| `0.50 ≤ f < 0.95` | `Step 4 of 5 · Building 3D` | WHAM transformer forward pass |
| `f ≥ 0.95` | `Step 5 of 5 · Finalizing` | Projection + DB write |
| `f > 1.50` | `Step 5 of 5 · Finalizing` (frozen) | Backend taking longer than expected — don't roll back, don't claim done |

When `elapsed > expectedSec + 30s` → stage hint stays frozen at
Step 5 but the headline/detail escalate (see processing copy below).

When `elapsed > 300s` → stage hint hidden entirely. Only the
elapsed counter remains.

### Why this is honest

Stage labels match real pipeline phases. The bucketization is an
**estimate** of where in the pipeline the backend probably is — not
a real-time signal. The TIMING is approximate; the SEQUENCE is
real. Users learn the right mental model ("there's pose detection,
then 3D analysis") without being lied to about progress percent.

If backend hits an unusual path (e.g., WHAM finishes in 30s instead
of 60s expected), the real `wham_status='ready'` flip overrides the
animation immediately. No "Step 5 of 5" forever-spinning.

### Animation visuals (described, not coded)

A 5-segment ring around a center icon. Active segment pulses (1s
breathe). Completed segments solid `#a8f040`. Unfilled segments
`rgba(168,240,64,0.12)`. CSS-only animation, no JS animation loop.

---

## R1 — Copy honesty annotation legend

Every UI string in this spec carries one of:

| Tag | Meaning |
|---|---|
| `[SHIPPING NOW]` | Copy accurately describes what the current production pipeline does today. Safe to ship verbatim. |
| `[PROMISE]` | Copy describes a future capability or implied behavior. Trust-building OK but flagged for Jason sign-off. Each [PROMISE] line is a small bet on user goodwill. |
| `[FUTURE PR]` | Copy MUST NOT ship until the corresponding analysis PR is merged. Listed so reviewer can spot accidental inclusion. |

Specifically watched (per R1):
- Any "analyzing your body rotation / spine angle / hip movement /
  head stability" copy — those metrics are NOT computed today.
  Tagged `[FUTURE PR]` and excluded from current spec unless Jason
  explicitly approves as `[PROMISE]`.

---

## State 1 — `processing`

### Layout (top to bottom)

```
┌─────────────────────────────────────────────────┐
│ ← History                                        │  back chip, top-left
│                                                  │
│            ╭─────────────╮                       │
│            │             │                       │  5-segment progress ring,
│            │   [ANIM]    │                       │  ~96px, center icon pulses
│            │             │                       │
│            ╰─────────────╯                       │
│                                                  │
│       Building your 3D analysis                  │  headline 22px bold
│                                                  │
│       Step 3 of 5 · Detecting pose               │  stage hint 14px muted
│                                                  │
│   Usually takes about 60 seconds                 │  detail 14px subtle
│                                                  │
│         Elapsed: 23s · target 60s                │  counter 12px mono
│                                                  │
│  ─────────────────────────────────────────────  │  hairline divider
│  Body alignment in the analysis will be          │  pre-disclaimer 11px,
│  approximate — a coaching anchor, not a          │  matches Part 1 disclaimer
│  precise measurement.                            │  tone but inverted-tense
└─────────────────────────────────────────────────┘
```

### Copy (per R1)

| Element | Copy | Tag |
|---|---|---|
| Headline default | `Building your 3D analysis` | [SHIPPING NOW] — WHAM forward pass IS 3D, accurate |
| Headline at `elapsed > expectedSec + 30s` | `Still analyzing…` | [SHIPPING NOW] |
| Headline at `elapsed > 300s` | `Analysis is taking too long` | [SHIPPING NOW] |
| Stage 1 label | `Step 1 of 5 · Uploaded` | [SHIPPING NOW] |
| Stage 2 label | `Step 2 of 5 · Preparing` | [SHIPPING NOW] — preflight (scene-cut + duration) is real |
| Stage 3 label | `Step 3 of 5 · Detecting pose` | [SHIPPING NOW] — YOLO + ViTPose is real 2D detection |
| Stage 4 label | `Step 4 of 5 · Building 3D` | [SHIPPING NOW] — WHAM forward pass is real |
| Stage 5 label | `Step 5 of 5 · Finalizing` | [SHIPPING NOW] — DB write + projection is real. **Intentionally NOT "Preparing coaching cues"** since no cues are computed; that'd be [FUTURE PR]. |
| Detail default | `Usually takes about Ns.` (N = expectedSeconds) | [SHIPPING NOW] |
| Detail at `elapsed > expectedSec + 30s` | `Taking longer than expected — hang tight.` | [SHIPPING NOW] |
| Detail at `elapsed > 300s` | `Still processing — this is taking longer than expected. You can wait or try a different upload.` | [SHIPPING NOW] — locked rewrite. Drops "we're looking into it" which would have implied monitoring/alerting that doesn't exist today. New copy is factual: a long elapsed and a real choice. |
| Counter | `Elapsed: 23s · target 60s` (drop "target" segment after expectedSec+30s) | [SHIPPING NOW] |
| Pre-disclaimer | `Body alignment in the analysis will be approximate — a coaching anchor, not a precise measurement.` | [SHIPPING NOW] — matches PR-8h.0 closure path 1 + PR-8d.2 Part 1 |

### CTAs

| Element | Label | onClick action | When shown |
|---|---|---|---|
| Top-left chip | `← History` | navigate to `/history` | Always |
| (no retry CTA) | — | — | During processing, retry would create a duplicate Modal call. |

### Stage 5 → `ready` transition behavior (sign-off locked)

When animation reaches Stage 5 (`f ≥ 0.95`) but the next poll has
not yet returned `wham_status='ready'`, the screen MUST NOT visually
imply completion. Three options were considered:

| Option | Behavior at f ≥ 0.95, wham_status still 'processing' |
|---|---|
| A (locked) | Stage 5 label stays `Step 5 of 5 · Finalizing` indefinitely. Counter keeps ticking. Ring stays at 4-of-5-filled with segment 5 actively pulsing. Backend signal is the ONLY thing that flips to ready. |
| B (rejected) | Progress bar caps visually at 95% with `Almost done...` subtext. Risk: "stuck at 95%" user confusion. |
| C (rejected) | Drop percentage entirely once at Stage 5, show only spinner. Risk: feels regressed; user loses progress affordance. |

**Locked Option A.** Most honest. Users learn that "Finalizing"
takes a variable amount of time and the backend is the source of
truth. No "almost done" promises the spec can't keep.

### Edge cases

| Scenario | Handling |
|---|---|
| Backend writes `ready` while user mid-stage-3 | Polling fetches `ready`, screen exits immediately. Animation does NOT need to "catch up". User sees normal ready-state transition. |
| Backend writes `failed_*` while user mid-stage | Polling fetches `failed_*`, screen exits to failure UI. |
| `expectedSeconds` missing on row | Default to `60`. Log a warning client-side (console). Don't fail silently — every processing screen should have an ETA. |
| Tab loses focus then refocuses (browser throttles polling) | On `visibilitychange` → fire one immediate poll. Don't wait for next throttled interval. |
| Page refreshes mid-processing | `startedAt` resets to `nowMs`, BUT the original `wham_processing_started_at` is in DB. Show DB-time-based elapsed, not client-mounted elapsed. |
| Network failure during poll | Show last-known state. Retry next tick. Don't surface transient network errors. |
| `wham_status` changes from `processing` → `processing` (no-op) | Continue polling. Common; backend may write same value on retry. |

### R4 fixture — `processing`

**No static fixture in DB.** Live-test via fresh upload.

Test procedure:
1. `/upload` → submit a 4-8 second valid swing video
2. After upload + analyze call returns, navigate to `/result/[newVidId]`
3. State should be `processing`. Screen should show:
   - 5-segment ring animating
   - Headline `Building your 3D analysis`
   - Stage label starting at `Step 1 of 5 · Uploaded` and advancing
   - `Elapsed: Ns · target 60s` counter ticking
   - Pre-disclaimer at bottom
4. Wait ~60-90s. Screen should auto-transition to `ready`.
5. CTA: clicking `← History` navigates to `/history`. Job continues
   in backend regardless.

---

## State 2 — `ready`

### Layout

```
┌─────────────────────────────────────────────────┐
│ ← SwingCue            + New                      │  header bar
├─────────────────────────────────────────────────┤
│                                                  │
│            [video + WHAM skeleton]               │  SwingPlayer +
│                                                  │  WhamSkeletonOverlay
├─────────────────────────────────────────────────┤
│  Body alignment is approximate — used as         │  disclaimer (shipped
│  a coaching anchor.                              │  in PR-8d.2 Part 1)
└─────────────────────────────────────────────────┘
```

### Copy (per R1)

| Element | Copy | Tag |
|---|---|---|
| Header logo | `SwingCue` | [SHIPPING NOW] |
| Disclaimer | `Body alignment is approximate — used as a coaching anchor.` | [SHIPPING NOW] — already shipped, PR-8d.2 Part 1 |

**No other ready-state-specific copy** in this spec. SwingPlayer + skeleton are the content.

### CTAs

| Element | Label | onClick action |
|---|---|---|
| Header back | `←` | `/history` |
| Header `+ New` | `+ New` | `/upload` |
| (no mid-page CTA) | — | Video itself is the interactive surface |

### Edge cases (already shipped, unchanged here)

| Scenario | Handling | Reference |
|---|---|---|
| `wham_status='ready'` but `wham_pose_timeline` rows empty | "Preparing" screen with Refresh button | PR-8d.1 R5 |
| `wham_pose_timeline` fetch in flight | Spinner with `Loading 3D analysis…` text | PR-8d.1 |
| Row has `wham_status='ready'` but no `acromion_left` keys (pre-PR-8e WHAM era, e.g. fixture 5bbcfbc8) | Renders skeleton via SMPL fallback path in `WhamSkeletonOverlay.resolveJointPos` | PR-8e.1 backward compat |

### R4 fixtures — `ready`

| Fixture | Anatomical data | Expected screen | Expected CTAs |
|---|---|---|---|
| `a3f7b0d8-99d0-4f0e-84f3-41abd95ceaea` | yes | WHAM 17-joint skeleton in cyan/amber/white, disclaimer below | `←` → /history, `+ New` → /upload |
| `7e49a385-740d-4704-ba03-bd59426ed704` | yes | same | same |
| `39dab3eb-993c-4425-88a8-b04e07ba7ab9` | yes | same | same |
| `5bbcfbc8-49b9-4fc4-8b0e-a34c5427aa62` | **no** (pre-PR-8e WHAM) | WHAM 17-joint skeleton with SMPL-fallback positions (no acromion override); disclaimer present | same |

Note on `5bbcfbc8`: this row has `wham_status='ready'` so it is
NOT `legacy_absent`. It IS in the `ready` state, just with the
SMPL fallback render path because the row pre-dates PR-8e.0
anatomical landmarks. User-perceived behavior: identical UX to
the 3 anatomical-fixtures except shoulder dots sit at the SMPL
glenohumeral joint instead of the acromion (a few pixels more
medial). Confirmed by SQL prior to writing this spec.

(The R4 spec listing of `5bbcfbc8` under `legacy_absent` is a
fixture-listing error and is reclassified to `ready` here per
the canonical state-mapping rule in R2.)

---

## State 3 — `failed_preprocessing`

### Layout

```
┌─────────────────────────────────────────────────┐
│ ← History                                        │
│                                                  │
│              [!]   yellow warning glyph 64px     │
│                                                  │
│       Couldn't analyze this video                │  headline 22px
│                                                  │
│   <REASON SUBSTITUTION — see copy table>         │  reason 16px
│                                                  │
│   What to try:                                   │  help-row 14px
│   • Record a longer clip (4-8 seconds works     │
│     best)                                        │
│   • Use a single continuous take                 │
│   • Avoid scene cuts or zoom changes             │
│                                                  │
│              [Try again →]                       │  primary CTA
└─────────────────────────────────────────────────┘
```

### Copy (per R1)

| Element | Copy | Tag |
|---|---|---|
| Headline | `Couldn't analyze this video` | [SHIPPING NOW] |
| Reason — duration <3s | `Video too short — needs at least 3 seconds.` | [SHIPPING NOW] — `wham_user_message` populates this verbatim |
| Reason — scene-cut detected | `Multi-scene video detected — please upload a single continuous swing.` | [SHIPPING NOW] |
| Reason — preprocessing timeout | `Preprocessing timed out — please try a shorter clip.` | [SHIPPING NOW] |
| Reason — unknown preprocessing | `Couldn't process this video. Try a shorter, single-take clip.` | [SHIPPING NOW] — fallback for unanticipated wham_user_message values |
| Help-row header | `What to try:` | [SHIPPING NOW] |
| Bullet 1 | `Record a longer clip (4-8 seconds works best)` | [SHIPPING NOW] — current preflight minimum is 3s, 4-8s is comfortable margin |
| Bullet 2 | `Use a single continuous take` | [SHIPPING NOW] — scene-cut detector is real |
| Bullet 3 | `Avoid scene cuts or zoom changes` | [SHIPPING NOW] |
| Primary CTA label | `Try again →` | [SHIPPING NOW] |

**Reason copy SOURCE**: `swing_analysis.video_metadata_json.wham_user_message`
written by Modal in PR-8c.4. Frontend reads verbatim. **Never** show
raw exception text. **Never** show Python traceback. **Never** show
`wham_failure_stage` raw value to user.

### Verbose reason copy preserved (sign-off locked)

PR-8d.0 already ships verbose `wham_user_message` content
(e.g., `"Multi-scene video detected (6 scene cut(s)). SwingCue MVP
only supports single-take swings…"`). This spec **keeps that
verbosity** — does not flatten to a 3-word summary.

Rationale: diagnostic specifics are the highest-value content of a
failure screen. `"Video too short (2.6s) — needs at least 3 seconds"`
gives the user a measurable thing to fix; `"Video too short"` alone
doesn't. As long as the copy is pre-sanitized by Modal (no traceback,
no internals — already true per PR-8c.3 `wham_user_message`
contract), keep the detail.

If a future `wham_user_message` value reveals over-promising or
internal info, fix at the Modal-side message-construction layer
(PR-8c.x), not by post-processing on the frontend.

### CTAs

| Element | Label | onClick action | When shown |
|---|---|---|---|
| Top-left chip | `← History` | navigate to `/history` | Always |
| Primary | `Try again →` | navigate to `/upload` | Always |

No "Report a bug" CTA — preprocessing failures are user-fixable,
not bugs.

### Edge cases

| Scenario | Handling |
|---|---|
| `wham_user_message` is empty/null | Fall through to `Couldn't process this video. Try a shorter, single-take clip.` |
| `wham_failure_stage='preprocessing'` but no `wham_user_message` field at all | Same — fallback copy |
| User clicks Try again, navigates to `/upload`, picks the SAME video again | Backend will fail again with same reason; user experience identical. Out of scope to deduplicate. |

### R4 fixtures — `failed_preprocessing`

| Fixture | Status | Notes |
|---|---|---|
| `0e9153db-65a8-4ee0-9100-71f9d4fee65b` | **NOT IN DB** as of audit (row doesn't exist) | Listed in spec as the canonical T2 "video too short" fixture from prior PR docs. Either create via SQL force or recover from history. |
| `b353a7ca-1700-411b-93e1-25ad8b4d8000` | **NOT IN DB** as of audit | Canonical T1 "multi-scene" fixture from PR-8c.4 docs. Same — recover or recreate. |

**Fixture provisioning (Q6 locked):** create real failures via
genuine bad-clip uploads through a **dedicated test account**
(`test+swingcue-fixtures@…`) so they don't pollute production
user history. Upload:
- A 2-second clip → produces "too short" preprocessing failure
- A multi-scene clip (manually-concat two takes) → produces
  "multi-scene detected" preprocessing failure

Do NOT create stable test fixtures under production user_ids.

Expected behavior on each (whenever they exist):
- Page renders failed_preprocessing screen
- Reason copy matches the failure type, verbose per Q9 (not flattened)
- 3-bullet "What to try" help-row visible
- `Try again →` button → `/upload`
- `← History` → `/history`
- No traceback, no internal info shown

---

## State 4 — `failed_system`

### Layout

```
┌─────────────────────────────────────────────────┐
│ ← History                                        │
│                                                  │
│              [!]   red error glyph 64px          │
│                                                  │
│            Analysis failed                       │  headline 22px
│                                                  │
│   Something went wrong. Please try uploading    │  message 16px
│   again.                                         │
│                                                  │
│              [Try again →]                       │  primary CTA
│                                                  │
│   Error reference: a7f3-k2                       │  hash 12px mono,
│                                                  │  standalone, NO
│                                                  │  instruction text
└─────────────────────────────────────────────────┘
```

Intentionally minimal. The earlier draft proposed a longer message
+ a "You can:" bullet section + a "send the error reference"
instruction. All three were [PROMISE] copy because no monitoring,
no support channel, and no "send to" target exists today. Per
sign-off, the spec now ships only what is true today: a generic
honest message, a working retry CTA, and a stable reference hash
that future support flow (PR-9+) can surface instructions for.

### Copy (per R1)

| Element | Copy | Tag |
|---|---|---|
| Headline | `Analysis failed` | [SHIPPING NOW] |
| Message paragraph | `Something went wrong. Please try uploading again.` | [SHIPPING NOW] — locked rewrite. Dropped the "we're already looking into it" clause (no monitoring system) and the "this isn't your video" framing (overpromises; the rare case where backend stage WAS user-influenced gets called out as preprocessing failure separately). New copy: short, honest, points to the working retry CTA. |
| (help-row removed) | — | — |
| Primary CTA label | `Try again →` | [SHIPPING NOW] |
| Error reference label | `Error reference:` | [SHIPPING NOW] |
| Error reference value format | `a7f3-k2` — 6 alphanumeric chars (kebab-cased 3+3 split) | [SHIPPING NOW] — `shortHash(videoId + failureStage)` already implemented in PR-8d.0. Spec switches casing from `A7F3-K2` → `a7f3-k2` to match standard short-hash convention. Implementation choice. |

**Never shown** (R6 invariant): `wham_error_message` field, Modal
call IDs, internal stage names verbatim, file paths.

### CTAs

| Element | Label | onClick action |
|---|---|---|
| Top-left | `← History` | `/history` |
| Primary | `Try again →` | `/upload` |

No "Contact support" CTA exists today. The error reference hash IS
the contact handoff (a user can include it in any future support
channel).

### Edge cases

| Scenario | Handling |
|---|---|
| `wham_failure_stage='unknown'` | Same screen, same copy. Reference hash uses stage='unknown'. |
| `wham_failure_stage` field missing (very early dispatch failure) | Same screen. Hash uses stage='null'. Stable per video. |
| User clicks Try again multiple times rapidly | Each click creates a new upload; reference hashes differ per attempt. Acceptable — no spam guard needed. |
| Error reference collision (two videos hash to same 7 chars) | Birthday-paradox probability is ~0.0001 at <100 failed videos. Accept. Reference is for support diagnostic, not auth. |

### R4 fixtures — `failed_system`

| Fixture | Status | Notes |
|---|---|---|
| `b8d9e821-c30c-4d40-9bce-bc4f1f717f87` | **NOT IN DB** as of audit | Canonical "inference failed" fixture from PR-8d.0 docs. |
| `88ef44a5-3367-446c-a8df-be3c98edc8c1` | **NOT IN DB** as of audit | Canonical "very early failure, null stage" fixture from PR-8d.0 docs. |

**Fixture provisioning (Q6 locked):** SQL-force ghost rows.
Unlike preprocessing failures which can be reproduced by uploading
bad clips, `failed_system` underlying stages (Modal token error,
network mid-stream, DPVO init crash) aren't cheap to reproduce on
demand. Synthesize via SQL: insert `swing_analysis` rows under the
dedicated test account's user_id with
`video_metadata_json` shapes:
```json
{ "wham_status": "failed",
  "wham_failure_stage": "inference",   /* or 'unknown' for null-stage variant */
  "wham_error_message": "<sanitized synthetic message>" }
```

The `wham_error_message` field is NEVER rendered (R6) — only
present for log-trace symmetry with real rows.

Expected behavior:
- Page renders failed_system screen with SIMPLIFIED copy per Q3/Q4
- `Something went wrong. Please try uploading again.` message
- No "you can:" bullet section (removed)
- Stable error reference per (videoId, stage) pair
- Bare `Error reference: a7f3-k2` line with no instruction text
- `Try again →` → `/upload`
- `← History` → `/history`
- No internal info leaked

---

## State 5 — `legacy_absent`

### When

Row has no `wham_status` key in `swing_analysis.video_metadata_json`.
These are pre-WHAM-integration uploads — they have MediaPipe-based
`pose_timeline_2d` data only. The video plays; there's no 3D
analysis to render.

### Layout (Option B — locked)

Option A (banner over legacy MediaPipe view) is REJECTED at sign-off.
Rationale: a banner invites the question "why would my new upload
also look like this?" → drags the user into the full WHAM trust
model. Legacy gets a clean dedicated screen instead; old videos
are old, no extra cognitive load.


```
┌─────────────────────────────────────────────────┐
│ ← History                                        │
│                                                  │
│              [📁]   archive glyph 64px            │
│                                                  │
│       Older swing — no 3D analysis               │  headline 22px
│                                                  │
│   This swing was analyzed before SwingCue       │  message 16px
│   added 3D body tracking. The original video    │
│   is still available below.                      │
│                                                  │
│              [Re-upload for 3D →]                │  primary CTA
│                                                  │
│   ─────  Original video below  ─────             │  divider 12px
│                                                  │
│          [video player, no overlay]              │  bare video
└─────────────────────────────────────────────────┘
```

**Why Option B (sign-off locked):** legacy MediaPipe placeholder
dots look broken to a post-WHAM user. Replacing them with a clean
"older data" screen + plain video playback is more honest than
silently showing a degraded overlay. Plus: avoids dragging legacy-
viewing users into the full WHAM trust model explanation.

### Copy (per R1, Option B variant)

| Element | Copy | Tag |
|---|---|---|
| Headline | `Older swing — no 3D analysis` | [SHIPPING NOW] |
| Message | `This swing was analyzed before SwingCue added 3D body tracking. The original video is still available below.` | [SHIPPING NOW] — factually accurate |
| Primary CTA label | `Re-upload for 3D →` | [SHIPPING NOW] |
| Divider text | `Original video below` (with hairlines flanking) | [SHIPPING NOW] |

### CTAs

| Element | Label | onClick action |
|---|---|---|
| Top-left | `← History` | `/history` |
| Primary | `Re-upload for 3D →` | `/upload?from=legacy&original=<videoId>` — query params reserve future ability to show "Replacing your older video" context in the upload flow. Implementation of that context is out-of-scope for Part 2 |

### Edge cases

| Scenario | Handling |
|---|---|
| Video URL also missing (very old row, storage_path empty) | Fall back to "video unavailable" inline message + same Re-upload CTA. Still allows user to upload a new clip. |
| User clicks Re-upload, lands on /upload with query params | Upload page silently ignores the params (out of scope here). Future PR can read them. |
| Row exists with `wham_status='processing'` but very stale (>1 hour) | NOT legacy_absent (the field IS set). Routes to processing screen, which after `elapsed > 300s` shows escalated copy. |

### R4 fixtures — `legacy_absent`

| Fixture | Status | Notes |
|---|---|---|
| `b32e0f21-2ed1-44f3-8d2c-e2c83cee1a36` | **NOT IN DB** as of audit | Canonical pre-WHAM fixture from PR-8d.0 docs. |
| `9f0d9c6a-37dd-46aa-aa12-3df0c2c4b317` | **NOT IN DB** as of audit | Same — second pre-WHAM fixture. |
| `5bbcfbc8-49b9-4fc4-8b0e-a34c5427aa62` | **NOT legacy_absent** — has `wham_status='ready'`, reclassified to `ready` state | Per R2 rule. Spec original R4 listing was an error. |

**Fixture provisioning gap.** The 2 valid `legacy_absent` fixture
IDs aren't currently in DB. To verify the screen, either:
- Find a genuinely pre-WHAM row in DB by querying
  `WHERE NOT (video_metadata_json ? 'wham_status')`. Use whatever
  IDs are returned.
- OR synthesize via SQL force: insert a row with
  `video_metadata_json = '{}'::jsonb`.

Expected behavior on each (whenever they exist):
- Page renders legacy_absent screen (Option B)
- Plain video plays below the message
- NO MediaPipe placeholder dots
- NO "Head Movement" coaching cue
- `Re-upload for 3D →` → `/upload?from=legacy&original=<id>`
- `← History` → `/history`

---

## Cross-state design principles

1. **Honesty over polish.** Every [PROMISE] tagged line is a trust
   debt. Don't accumulate without paying it back.
2. **Reversibility.** Every screen has at least one navigation away
   ({`← History`, `Try again →`, `Re-upload →`}). Never a dead end.
3. **Never expose internals.** No tracebacks, Modal call IDs, raw
   stage names. Short-hash reference is the only diagnostic surface.
4. **Inline disclaimer, not modal.** Limitation copy is part of the
   screen, not a popup.
5. **One-eyed-friend tone.** Direct, plain, honest. No "AI-powered",
   no emoji (except glyphs already in the palette).
6. **Palette locked.** Per `docs/decisions/PR-8d_PALETTE.md`.

## Product principle alignment

> "Fast page load, slow trusted result. Never fast inaccurate result."

| State | Alignment |
|---|---|
| `processing` | ✅ Slow result with honest progress; never fakes completion. Real `wham_status` flip is the only "done" signal. |
| `ready` | ✅ Trusted result loaded. Disclaimer prevents over-trust. |
| `failed_preprocessing` | ✅ Fast failure with actionable user guidance. |
| `failed_system` | ✅ Fast honest failure. Doesn't pretend it's user fault. |
| `legacy_absent` | ✅ Fast honest "no 3D here" state. Doesn't degrade to bad-looking MediaPipe dots. |

## Invariant compliance check

| Invariant | This spec |
|---|---|
| PR-8d.0 R4 (visual ≠ analysis layer) | Honored. No coaching metrics implied to be derived from skeleton positions. Disclaimer states "approximate" + "coaching anchor". |
| PR-8d.0 R6 (never expose traceback) | Honored. Error reference is the only diagnostic. `wham_error_message` never rendered. |
| Palette lock | Honored. No new colors introduced (red for error glyph + yellow for warning glyph are existing palette accents). |
| PR-8e_CLOSED limitations #1-#4 | Honored. The disclaimer language in processing pre-disclaimer + ready disclaimer reflects #4 (body-width underfit). |
| PR-8h.0 audit recommendation (path 1: accept + surface) | Honored. Disclaimer is the surface. |

---

## Decisions (sign-off locked 2026-05-27)

All 7 original open questions resolved. 2 additional questions
(Q8, Q9) raised and resolved at sign-off. R1 honesty annotation
process caught 3 real [PROMISE] copy violations on its first use —
all 3 rewritten as honest [SHIPPING NOW] in this revision.

| # | Decision | Notes |
|---|---|---|
| Q1 | **`legacy_absent` = Option B (clean replacement screen)** | Option A (banner-over-legacy) rejected. Avoids dragging legacy-viewing users into the WHAM trust-model explanation. Old videos = old. |
| Q2 | **`processing` 300s+ detail copy rewritten** | Was [PROMISE] `"We're looking into it. You can retry from the upload page."` → now [SHIPPING NOW] `"Still processing — this is taking longer than expected. You can wait or try a different upload."` Drops the implied monitoring system. |
| Q3 | **`failed_system` message rewritten** | Was [PROMISE] long copy with "we're already looking into it" + "this isn't your video" → now [SHIPPING NOW] short copy `"Something went wrong. Please try uploading again."` |
| Q4 | **`failed_system` "Send the error reference" instruction REMOVED** | Was [PROMISE] (no support channel exists). Help-bullet section entirely removed. Hash kept as standalone `Error reference: a7f3-k2` line with NO instruction text. Future support-flow PR (PR-9+) can add an instruction layer when the channel exists. |
| Q5 | **Stage 5 label = `Finalizing`** | NOT `Preparing coaching cues` (would be [FUTURE PR] until real cue computation ships). |
| Q6 | **Fixture provisioning = HYBRID** | `failed_preprocessing` fixtures via (c) genuine bad-clip uploads through a dedicated test account (`test+swingcue-fixtures@…`). `failed_system` fixtures via (b) SQL-force ghost rows (the underlying failure stages — Modal token error, network mid-stream — are not cheap to reproduce). Do NOT create stable test fixtures under production user_ids. |
| Q7 | **Phased rollout = SEPARATE COMMITS** | 2A / 2B / 2C ship as separate commits in sequence. Acceptance surface narrow, bisect-friendly. State-machine work is high-risk per PR-8d.0 history. If any sub-phase grows beyond ~1.5 hours during implementation, split further. |
| Q8 | **Stage 5 → ready transition = Option A** | When animation reaches Stage 5 but `wham_status` is still `processing`, label stays `Finalizing` indefinitely, counter ticks, segment 5 pulses. Backend signal is the only completion trigger. (Sections "Stage 5 → ready transition behavior" above.) |
| Q9 | **`failed_preprocessing` verbose reason copy PRESERVED** | Verbatim `wham_user_message` (e.g. `"Video too short (2.6s) — needs at least 3 seconds"`) — does NOT get flattened. Diagnostic specifics are the highest-value content of a failure screen. Pre-sanitization (no traceback) is enforced at the Modal-side message-construction contract (PR-8c.3). |

## R1 honesty annotation process — first run-through

The annotation legend caught 3 [PROMISE] copy lines on its first
real use:
- `processing` 300s+ "we're looking into it"
- `failed_system` "we're already looking into it"
- `failed_system` "Send the error reference below if it keeps failing"

All 3 implied capabilities (real-time monitoring, support-channel
inbound) that don't exist yet. Rewritten this revision. Pattern
worth recording as SwingCue long-term review-process culture: every
new UX spec gets R1-annotated; every [PROMISE] gets surfaced for
sign-off before ship.

## Implementation phasing (sign-off locked: separate commits)

3 commits in sequence. If any sub-phase grows beyond ~1.5 hours
during implementation, split further (state-machine work is
high-risk per PR-8d.0 history).

| Phase | Scope | Cost | Risk |
|---|---|---|---|
| **2A** | Processing-screen enrichment: 5-stage bucketization logic, stage labels, pre-disclaimer footer, headline/detail escalation thresholds (30s, 300s), Stage-5-stay-on-Finalizing (Q8) behavior. `failed_preprocessing` keeps verbose `wham_user_message` (Q9) + adds 3-bullet "What to try" help-row. `failed_system` SIMPLIFIES per Q3/Q4 (drop "you can" section, drop monitoring-implying copy, keep bare reference line). | ~1.5 hr | Low–Medium — pure JSX edits in existing components, copy updates, time-based stage logic |
| **2B** | `legacy_absent` Option B (Q1): dedicated screen + plain-video render path in SwingPlayer (no-overlay mode). Wire `/upload?from=legacy&original=<id>` query-param contract. | ~1 hr | Medium — touches SwingPlayer's overlay-rendering branch |
| **2C** | Animated 5-segment progress ring (CSS-only). Pure visual polish over 2A's time-based stage logic. | ~1 hr | Low |

2A → 2B → 2C in order. Acceptance verification fixture-by-fixture
at each phase boundary before opening the next.

## Files this spec WILL touch when implementing

- `src/app/result/[id]/page.tsx` — state-screen branches, props,
  copy strings, stage bucketization logic
- `src/components/SwingPlayer.tsx` — possibly the legacy-absent
  Option B no-overlay render path
- `src/components/` — possibly new `LegacyAbsentScreen.tsx` if Option
  B is picked and the code style favors extraction
- No backend changes (stage hints are time-based estimation)
- No DB schema changes
- No Modal changes

## Cross-references

- PR-8d.0 — wham_status state machine and initial processing/failed
  screens
- PR-8d.1 — WHAM skeleton rendering on ready branch (commit `18206ad`)
- PR-8d.2 Part 1 — body-alignment disclaimer (commit `dc9ecd6`)
- PR-8e_CLOSED.md — body-width limitation root cause
- PR-8h_CALIBRATION_AUDIT.md — why per-user correction was rejected
- PR-8d_PALETTE.md — locked colors
- PR-8c.3 — backend stage enum
- PR-8c.4 — preflight (scene-cut + duration check)
- PR-8c.5 — upload pipeline state machine

---

End of canonical spec. Awaiting sign-off on the 7 open questions
above before any 2A/2B/2C implementation.
