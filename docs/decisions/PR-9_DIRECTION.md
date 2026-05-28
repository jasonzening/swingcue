# PR-9 — Direction (stub, awaiting expansion)

Status: **DRAFT placeholder.** Content below is structural scaffolding
so the locked R4 architectural rule has a stable home. Expand the
empty sections inline when ready.

## Context

PR-8 series shipped the WHAM 3D pose pipeline + body-width limitation
acceptance (PR-8h.0). PR-8d.2 closed the result-page state UX. From
here, future investment shifts to **coaching content** — translating
raw WHAM data into actionable swing feedback.

## Future investment shifts to coaching layer

[expansion: scope, sequence, fixture strategy, copy guidelines]

## R4 architectural rule (locked previously in PR-8h_CALIBRATION_AUDIT.md, re-affirmed here)

```
wham_pose_timeline schema:
  ├── keypoints_2d_projected  ← raw WHAM, ANALYSIS LAYER
  ├── keypoints_3d_smpl       ← raw WHAM 3D, ANALYSIS LAYER
  └── keypoints_2d_visual     ← (if ever ships) VISUAL LAYER
```

PR-9 coaching rules (head movement, shoulder-hip rotation, spine
angle, arm structure, weight transfer, etc.) **MUST source metrics
from the analysis layer (raw WHAM 2D/3D), NOT from the
visual-corrected layer.**

Coaching indicators (green ghost lines, ✓/✗ verdicts, body-part
highlights) **ARE the visual layer.** They are rendered FROM the
analysis-layer judgment, not the other way around.

This separation keeps coaching quality independent of any future
visual calibration tweaks. A user-input height calibration shipped
for visual polish should never silently corrupt head_movement metric
computation.

### Concrete enforcement

| Code site | Allowed source | Forbidden source |
|---|---|---|
| Coaching-metric computation (PR-9 head movement, shoulder rotation, etc.) | `wham_pose_timeline.keypoints_2d_projected`, `keypoints_3d_smpl` | `keypoints_2d_visual` |
| Visual overlay rendering (WhamSkeletonOverlay, ghost lines, indicators) | `keypoints_2d_visual` IF present, else `keypoints_2d_projected` fallback | — |
| Phase-detection / swing-event classifiers | `keypoints_3d_smpl` (3D analysis source of truth) | `keypoints_2d_*` |

### Code-review trip-wires (for any PR opening files under `src/lib/coaching/` or similar)

- PR sourcing coaching insights from `keypoints_2d_visual` → reject
  unless explicit per-metric approval is documented in PR body.
- PR adding a new visual correction → MUST NOT touch any
  `coaching/` or `analysis/` file in the same diff.
- PR adding a new coaching rule → MUST cite which raw-layer field
  it reads from in the rule docstring.

## R5 architectural rule (locked 2026-05-28)

**Every coaching rule MUST define these five fields, in both the
rule's docstring AND its implementation:**

1. `required_landmarks` — list of keypoint names the rule reads
2. `min_confidence` — float threshold per landmark per phase
3. `stability_check` — max std/variance allowed in the phase window
4. `fallback_behavior` — `HIDE` or `NEED_CLEARER_VIEW_PLACEHOLDER`
   (there is **no** option for "render verdict anyway")
5. `confidence_label` in output — `high_confidence` |
   `low_confidence` | `insufficient`

### Rationale — trust > accuracy

WHAM raw keypoints are not always anatomically aligned with the
user's real body. Shoulder / elbow / wrist sometimes drift off the
actual arm. Issuing a strong verdict ("Your arm structure is wrong")
based on misaligned keypoints will collapse user trust on first use.

When data is unreliable, **show nothing or show "Need clearer view"
— never confidently wrong.** A blank or muted placeholder costs us
nothing; a confidently-wrong verdict costs us the user.

### Concrete enforcement

| Code site | Required |
|---|---|
| New coaching rule under `src/lib/coaching/` (or per R4: any rule reading from `wham_pose_timeline`) | Declare all 5 fields in docstring + implementation |
| PR-9A indicator design system | Must include visual treatment for `low_confidence` / `insufficient` states (e.g., a muted "Need clearer view" placeholder card in the slot where the indicator would otherwise render) |
| Coaching rule output schema | Must carry the `confidence_label` enum so the indicator layer can branch on it without re-deriving |

### Code-review trip-wire

PR adds a new coaching rule → grep the diff for the 5 fields above.
Any missing → **reject**. A rule that ships without confidence
gating is a regression on the trust contract, regardless of how
accurate the metric appears in fixture footage.

## Sequence

**PR-8j: Color System Rebrand** — green reserved for verdict
semantics only (`--verdict-correct`), electric blue
(`--accent-primary`) becomes the app UI accent, cyan
(`--scan-cyan`) is processing-scan only. Palette lives as CSS
custom properties in `src/app/globals.css`. Ships **before PR-9A**
so the indicator design system inherits a clean color contract:
on the ready page, green appearing in the UI means "✓ correct
verdict" — and nothing else. See `memory/pr_8j_color_system_lock.md`.

**PR-9A: Coaching Indicator Design System** — a reusable indicator
visual language (component prop contracts + coordinate/source lock
+ animation tokens + parametric-target ideal layer + 2 mandatory
mocks for `BoundaryLine` and `RotationDisc`) **MUST ship and be
reviewed BEFORE PR-9.0 Head Movement implementation.** Rationale:
PR-9 is the shift to a coaching-indicator visual language; building
Head Movement first without the shared system guarantees every
later module drifts and reworks. System before instance.

Per R5, PR-9A also defines the **low-confidence / insufficient
placeholder** treatment — the visual that occupies the indicator
slot when a rule's confidence gating prevents a verdict from
rendering. This is mandatory, not optional: a missing
placeholder treatment makes R5 unimplementable downstream.

**PR-9.0: Head Movement** — first coaching rule using the PR-9A
system. Sources from `keypoints_2d_projected` (head_crown vertex)
+ `keypoints_3d_smpl` per R4. Renders BoundaryLine indicator
(threshold) + verdict from PR-9A library. Per R5, the rule
docstring + implementation MUST declare `required_landmarks`,
`min_confidence`, `stability_check`, `fallback_behavior`, and
emit `confidence_label` in output — without those, PR-9.0 cannot
land.

**PR-9.1+** — additional coaching rules, each composed from PR-9A
primitives. Order TBD; likely shoulder-hip rotation next, then
spine angle, arm structure, weight transfer.

[expansion: order of PR-9.1+ rules pending Jason call. Each rule
needs an R4 data-source declaration + which PR-9A indicator types
it composes.]

## Open scope sketches

[expansion: copy guidelines, verdict thresholds, fixture strategy
for coaching ground truth]

## Cross-references

- `PR-8h_CALIBRATION_AUDIT.md` — original R4 lock + body-width
  limitation evidence
- `PR-8e_CLOSED.md` — accepted "approximate body alignment" surface
- `PR-8d.2_PART2_UX_SPEC.md` — Part 1 disclaimer + Part 2 state UX
- `PR-8d_PALETTE.md` — locked body-anatomy palette (joint colors)
- Memory `pr_8j_color_system_lock.md` — locked app-level palette
  (green = verdict, blue = app UI, cyan = scan)
- Memory `r5_confidence_gating_lock.md` — full text + rationale of
  the R5 rule defined above
