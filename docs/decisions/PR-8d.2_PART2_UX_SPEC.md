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
| Detail at `elapsed > 300s` | `We're looking into it. You can retry from the upload page.` | **[PROMISE]** — "we're looking into it" implies real-time monitoring/alerting. No such system wired up today. Either change copy or build monitoring first. Open question for sign-off. |
| Counter | `Elapsed: 23s · target 60s` (drop "target" segment after expectedSec+30s) | [SHIPPING NOW] |
| Pre-disclaimer | `Body alignment in the analysis will be approximate — a coaching anchor, not a precise measurement.` | [SHIPPING NOW] — matches PR-8h.0 closure path 1 + PR-8d.2 Part 1 |

### CTAs

| Element | Label | onClick action | When shown |
|---|---|---|---|
| Top-left chip | `← History` | navigate to `/history` | Always |
| (no retry CTA) | — | — | During processing, retry would create a duplicate Modal call. |

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

**Fixture provisioning gap.** The named failed_preprocessing fixtures
referenced in PR-8d.0 / PR-8d.1 docs don't currently exist as rows.
Either (a) create them by uploading a too-short clip + a multi-scene
clip via UI, OR (b) force-write rows via SQL with synthetic data
matching the expected `wham_failure_stage='preprocessing'` +
`wham_user_message` shape.

Expected behavior on each (whenever they exist):
- Page renders failed_preprocessing screen
- Reason copy matches the failure type
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
│   Something went wrong on our end while          │  message 16px
│   analyzing your swing. This isn't your video    │
│   — we're already looking into it.               │
│                                                  │
│   You can:                                       │  help-row 14px
│   • Try uploading again (sometimes it just       │
│     works the second time)                       │
│   • Send the error reference below if it keeps   │
│     failing                                      │
│                                                  │
│              [Try again →]                       │  primary CTA
│                                                  │
│   Error reference: A7F3-K2                       │  hash 12px mono
└─────────────────────────────────────────────────┘
```

### Copy (per R1)

| Element | Copy | Tag |
|---|---|---|
| Headline | `Analysis failed` | [SHIPPING NOW] |
| Message paragraph | `Something went wrong on our end while analyzing your swing. This isn't your video — we're already looking into it.` | **[PROMISE]** on the "we're already looking into it" clause. There is NO automated alerting / on-call system wired to these failures today. Either build monitoring first OR change copy to drop that implication (e.g., `…analyzing your swing. This isn't your video — please try again.`). Open question for sign-off. |
| Help-row header | `You can:` | [SHIPPING NOW] |
| Bullet 1 | `Try uploading again (sometimes it just works the second time)` | [SHIPPING NOW] — transient Modal / network issues do recover on retry. Honest. |
| Bullet 2 | `Send the error reference below if it keeps failing` | **[PROMISE]** — there's no actual "send to" channel set up (no support email, no support form linked). Either add a contact target OR change copy. Open question. |
| Primary CTA label | `Try again →` | [SHIPPING NOW] |
| Error reference label | `Error reference:` | [SHIPPING NOW] |
| Error reference value format | `A7F3-K2` — 7 alphanumeric chars with hyphen separator | [SHIPPING NOW] — `shortHash(videoId + failureStage)` already implemented in PR-8d.0 |

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

**Same fixture provisioning gap as failed_preprocessing.** Either
recover from history (if they were deleted between sessions) or
synthesize via SQL force on a test row.

Expected behavior:
- Page renders failed_system screen
- Same copy regardless of which underlying stage failed
- Stable error reference per (videoId, stage) pair
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

### Layout (Option B — recommended)

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

**Why Option B:** legacy MediaPipe placeholder dots look broken to
a post-WHAM user; replacing them with a clean "older data" screen +
plain video playback is more honest than silently showing a degraded
overlay.

**Alternative Option A (banner over legacy view):** thin banner
across the top of the existing MediaPipe placeholder UI. Keeps
backward visual sentiment but invites unfair comparison.

Open question (#1 below): A vs B.

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

## Open questions for sign-off

1. **`legacy_absent` Option A vs Option B.** Spec recommends B
   (clean replacement screen). Banner-over-legacy (A) preserves the
   MediaPipe placeholder but invites unfair comparison.
2. **`failed_system` "we're already looking into it" copy
   ([PROMISE]).** No real monitoring/alerting wired up. Three
   options:
   (a) Build basic Sentry / log alerting first, keep copy
   (b) Soften copy: `This isn't your video — please try again.`
   (c) Punt: keep copy, schedule monitoring as a follow-up PR
3. **`failed_system` "Send the error reference below if it keeps
   failing" copy ([PROMISE]).** No support channel exists. Either
   add a contact target (email/form) OR change copy to remove the
   implied "send to" target.
4. **`processing` 300s+ detail copy ([PROMISE]).** Same `we're
   looking into it` issue — same decision.
5. **Stage 5 label — `Finalizing` vs `Preparing coaching cues`?**
   Spec uses `Finalizing` because no cues are computed today.
   Switching to `Preparing coaching cues` would be [FUTURE PR] until
   real cue computation ships (PR-8d.3+). Confirm `Finalizing`.
6. **Fixture provisioning.** 6 of the 10 named R4 fixtures don't
   currently exist as DB rows. Implementation will need:
   (a) Recover the rows from a backup if they were deleted between
   sessions
   (b) OR synthesize via SQL force (insert test rows)
   (c) OR genuinely fail uploads (upload too-short clip, multi-scene
   clip) to populate the failed_* states from real Modal calls
   Picking (c) is the most representative. Decide.
7. **Phased implementation rollout.** Spec contains 3 logical
   chunks (A: enrich existing screens; B: legacy_absent new screen;
   C: animated 5-segment ring). Ship as one PR, or three separate
   commits in sequence?

## Implementation phasing proposal (for sign-off)

Spec is one document; implementation can split:

| Phase | Scope | Cost | Risk |
|---|---|---|---|
| **2A** | Stage hints, pre-disclaimer, What-to-try section, You-can section, copy updates | ~1 hr | Low — pure JSX edits in existing components |
| **2B** | `legacy_absent` dedicated screen + plain-video render path in SwingPlayer (no-overlay mode) | ~1 hr | Medium — touches SwingPlayer's overlay-rendering branch |
| **2C** | Animated 5-segment progress ring (CSS-only) | ~1 hr | Low — purely visual polish |

2A → 2B → 2C delivers most user value at lowest risk first. Bundling
into one PR is also acceptable; ~3 hours total work.

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
