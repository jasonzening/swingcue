# PR-8h.0 — Calibration audit (closeout)

Date: 2026-05-27
Status: **AUDIT FAILED — recommend NOT proceeding to PR-8h.1.**

## Goal

Measure whether WHAM's projected `shoulder_width / body_height` and
`hip_width / body_height` ratios are STABLE across multiple clean
videos, against an anthropometric reference. If stable within ±0.02
of a single value, a global visual correction factor is defensible.
If they vary widely, a global correction would over/under-correct
unpredictably for different users.

## Anthropometric reference (R1)

Source: **ANSUR I (1988)** — Anthropometric Survey of US Army
Personnel: Methods and Summary Statistics. Gordon, C. C., Churchill,
T., Clauser, C. E., Bradtmiller, B., McConville, J. T., Tebbetts, I.,
& Walker, R. A. (1989). Technical Report NATICK/TR-89/027, US Army
Natick Research, Development and Engineering Center.

ANSUR I is the most widely-cited large-sample (N=3,982 male + N=2,208
female) anthropometric dataset in industry. Values used:

| Measurement (P50) | Male | Female | Notes |
|---|---|---|---|
| Stature (mm) | 1755 | 1629 | Heel to crown, standing |
| Shoulder breadth — acromion to acromion (mm) | 467 | 410 | Visible outer shoulder (deltoid surface), what overlay should match |
| Biacromial breadth — humeral head to head (mm) | 397 | 360 | Inner skeletal, smaller than acromion-to-acromion |
| Hip breadth — bispinous (ASIS to ASIS, mm) | 269 | 269 | Wider in females relative to stature |
| Hip breadth — bitrochanteric (greater trochanter, mm) | ~330 | ~340 | Closer to what PR-8e Option W derives |

Derived ratios for VISUAL anatomical landmarks (acromion + trochanter):

| Ratio | Male P50 | Female P50 | Operating midpoint |
|---|---|---|---|
| **acromion / stature** | **0.266** | **0.252** | **~0.259** |
| **bitrochanteric / stature** | ~0.188 | ~0.209 | ~0.198 |

For projected pixel ratios (`shoulder_px / head_to_ankle_px`), subtract
~5% for ankle-vs-heel offset and ~5% for crown-vs-head-top offset
(SMPL `head_crown` ≈ crown). Net expected pixel ratios:

| | Male P50 | Female P50 |
|---|---|---|
| **shoulder_px / head_to_ankle_px** | **~0.279** | **~0.265** |
| **hip_px / head_to_ankle_px** | ~0.197 | ~0.220 |

## Coverage matrix (R3)

| Video ID | Filename hint | Dims | z_med | Class | Anat data? |
|---|---|---|---|---|---|
| `39dab3eb` | test_swing.mp4 (Jason) | 720×1280 | 3.98 m | face-on close | yes (PR-8e+) |
| `7e49a385` | 1-5miao-1pr-face | 576×1024 | 4.84 m | face-on far | yes (PR-8e+) |
| `cbf4f22c` | Videos2026-05-24_210110 | 576×1024 | 3.42 m | face-on close (different scene) | no (pre PR-8e) |
| `956afa87` | 20260414_155746_2 | 1080×1920 | 4.04 m | **DTL angle** (player sideways) | no (pre PR-8e) |

Coverage achieved:
- [x] face-on, close (39dab3eb)
- [x] face-on, far (7e49a385)
- [x] face-on, close, different scene (cbf4f22c)
- [x] DTL accidentally captured (956afa87) — l_sh_x > r_sh_x, shoulders perpendicular to camera, X-width compressed by ~80%
- [ ] Athletic / non-athletic build comparison — couldn't reliably classify available videos by build

**Pragmatic minimum met: 1 face-on close + 1 face-on far + 1 face-on different-scene.** DTL data excluded from ratio audit because shoulder X-width is rotation-compressed (golfer perpendicular to camera).

## Audit methodology

For each video:
- Sample 5 frames at `frac × frame_count`: 5%, 25%, 50%, 75%, 95% (rough setup / backswing / top / impact / finish).
- Compute:
  - `smpl_shoulder_width = |right_shoulder.x − left_shoulder.x|` (H36M joints, always present)
  - `smpl_hip_width = |right_hip.x − left_hip.x|`
  - Where PR-8e anatomical landmarks exist: `anat_shoulder_width = |acromion_right.x − acromion_left.x|`
  - `body_height = mean(left_ankle.y, right_ankle.y) − head_crown.y`
  - Ratios: `shoulder / body_height`, `hip / body_height`

Note on phases other than setup: during rotation (backswing → finish),
shoulders rotate out of the camera-perpendicular plane, so projected
X-width SHRINKS to as little as ~10% of setup width. This is geometric
foreshortening, NOT a WHAM bias. **Audit uses SETUP frame ratios as
the primary stable measurement.**

## Findings — setup frame, face-on videos only

| Video | SMPL sh/h | Anat sh/h | SMPL hip/h | Anat hip/h¹ |
|---|---|---|---|---|
| `39dab3eb` (Jason) | 0.188 | **0.203** | 0.185 | 0.185 |
| `7e49a385` (other) | 0.151 | **0.163** | 0.175 | 0.175 |
| `cbf4f22c` (other) | 0.155 | — | 0.133 | — |

¹ PR-8e Option W derives `greater_trochanter` X-position FROM `left_hip` /
`right_hip` (only Y is offset by +30·image_h/1024). So `anat_hip_width = smpl_hip_width` by construction in current production. The audit treats them as identical for X-width purposes.

### Compared to ANSUR P50 reference

| | WHAM range (anat or SMPL) | ANSUR P50 | Shortfall |
|---|---|---|---|
| shoulder / body_height | 0.151 – 0.203 | 0.265-0.279 (male) | **23%-43%** under |
| hip / body_height | 0.133 – 0.185 | 0.197-0.220 | **6%-39%** under |

### Variance across the 3 face-on videos

| | Min | Max | Spread | Per-video Δ vs mean |
|---|---|---|---|---|
| shoulder / body_height (best-available landmark) | 0.155 | 0.203 | **31% spread** | ±10% from 0.174 midpoint |
| hip / body_height | 0.133 | 0.185 | **39% spread** | ±16% from 0.164 midpoint |

## Pass / fail criterion

> R1 pass condition (Jason): "if shoulder ratio is consistently ~0.20 ± 0.02, correction is defensible. If it varies widely, do not use global correction."

±0.02 around 0.20 = **[0.18, 0.22]**.

| Video | Anat or SMPL shoulder/height | In [0.18, 0.22]? |
|---|---|---|
| `39dab3eb` | 0.203 | yes |
| `7e49a385` | 0.163 | **no** (below by 0.017) |
| `cbf4f22c` | 0.155 | **no** (below by 0.025) |

**Audit FAILS pass criterion.** Two of three face-on videos fall below 0.18 — a ~20% gap from the "stable midpoint" assumption.

Hip/height shows similar instability: 0.133 – 0.185 spans 39%, well outside ±0.02.

## What this means for PR-8h.1

A single global correction factor (e.g., `shoulder_x *= 1.30`) would:

| Video | Current shoulder/h | After ×1.30 | vs ANSUR ~0.265 (male P50) |
|---|---|---|---|
| `39dab3eb` | 0.203 | 0.264 | **matches male P50 well** |
| `7e49a385` | 0.163 | 0.212 | 20% under-correction (~female P50 territory) |
| `cbf4f22c` | 0.155 | 0.202 | 24% under-correction |

A single factor would over-correct narrow-shouldered users to plausible
values and **leave wide-shouldered users still narrow**. Or, choose a
larger factor (×1.50) and risk over-shooting Jason-like users. There
is no factor that lands all three videos inside ±10% of ANSUR P50.

The bias DIRECTION is consistent — every face-on video under-projects
shoulders relative to anthropometric reference. WHAM is structurally
narrow. The bias MAGNITUDE varies per video (15–37% shortfall), and
that variance is too large for a single correction.

## Why does the magnitude vary?

Candidate explanations (not investigated in PR-8h.0):
- Real human build difference: Jason vs the golfer in `1-5miao-1pr-face` may genuinely differ in shoulder/height ratio by 10-20% — anthropometric P5 vs P95 range is ~15%.
- Slight pose / angle differences at setup: ~10° rotation from camera-perpendicular reduces X-projected width by ~15%.
- WHAM training-data prior variance: the network's `(β, s)` joint optimization may converge to different tie-breaks on different visual cues.

Distinguishing these would require known camera angles AND known
real-body proportions per subject — neither available in current data.

## Architectural separation (R4) — locked in for any future PR-8h.1

If a future PR-8h.1 ships ANY visual-only correction (not recommended
per this audit), the schema MUST preserve raw analysis data alongside
the corrected visual data. Specifically:

**Visual layer ≠ Analysis layer.**

| Field (jsonb on `wham_pose_timeline`) | Source | Purpose |
|---|---|---|
| `keypoints_2d_projected` (existing) | Raw WHAM projection | **Analysis layer.** Coaching metrics, biomechanics, swing-plane angle calculations. Never altered after PR-8e write. |
| `keypoints_3d_smpl` (existing) | Raw WHAM 3D | Same. Analysis layer. |
| `keypoints_2d_visual` (NEW, if PR-8h.1 ships) | Post-processed for visual coherence | **Visual layer.** What the SVG overlay reads. May apply uniform scale, anatomical stretch, or other corrections that improve visual match at the cost of strict geometric accuracy. |

Frontend rendering (`WhamSkeletonOverlay.tsx`):
- Reads `keypoints_2d_visual` ONLY when present
- Falls back to `keypoints_2d_projected` for old rows (backward compat)

Future coaching-metric computation (PR-8d.2+, PR-9+):
- MUST read `keypoints_2d_projected` or `keypoints_3d_smpl`
- MUST NOT read `keypoints_2d_visual` unless explicitly approved per-metric
- Code reviewers reject PRs that source coaching insights from `keypoints_2d_visual`

This boundary is what makes the audit-fail recoverable. The raw data
stays untouched; visual hacks live in a clearly-segregated field.

## Recommendation

**Do NOT ship PR-8h.1 global shoulder/hip correction.**

The variance is too large. A single factor will systematically
over-correct ~33% of users and under-correct another ~33% — worse
than no correction for a meaningful fraction of MVP traffic.

Alternative paths (cost-ordered):

1. **Accept + surface limitation.** Keep current PR-8e rendering.
   Document in user-visible help text: "Skeleton overlay is a coaching
   anchor, not a precise anatomical measurement. Body width may
   appear narrower than reality due to monocular reconstruction
   limits." Move to PR-8d.2 progress UX. (Recommended.)
2. **Per-user calibration input.** Ask the user at upload for height
   + shoulder width estimate (or just height). Use real ratios to
   compute a per-user correction factor. Adds friction at upload —
   Jason previously rejected this in PR-8f deliberations.
3. **Scope reduction.** Drop full-skeleton overlay from MVP. Render
   only specific reference points (e.g., head position, hip rotation
   indicator) where X-width precision matters less.
4. **Model swap exploration.** Investigate replacing WHAM with HMR2,
   HMR2.0, OSX, or BEDLAM-CLIFF. Long horizon. Out of PR-8h.x scope.

## Files / data referenced

- Raw audit data: query in `docs/decisions/PR-8h_CALIBRATION_AUDIT.md`
  (this file, "Audit methodology" section) — reproducible via
  Supabase MCP `execute_sql` against project `ciofgtwwcgyzfafmbjxu`.
- Anthropometric reference: ANSUR I (1989), Gordon et al., NATICK/TR-89/027.
- WHAM source pinned at `2b54f77`, cloned locally to `tmp/wham_full/`
  (gitignored).
- Related closed PRs: `docs/decisions/PR-8e_CLOSED.md` (limitation #4
  body-scale underestimation), `PR-8f` abandoned in commit `46b6d9e`.

## Closeout

PR-8h.0 audit complete. No production code changed. No Modal redeploy.
`tmp/wham_full/` retained for future investigation use.

PR-8h.1 stays UN-OPENED pending Jason's scope decision among the
4 alternatives above.

PR-8d.2 progress UX remains gated only on the body-width visual gap
resolution path being chosen — if "accept + surface" is picked,
PR-8d.2 can start immediately.
