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

## Open scope sketches

[expansion: which coaching rules ship in PR-9.0 vs PR-9.1 vs later]

## Cross-references

- `PR-8h_CALIBRATION_AUDIT.md` — original R4 lock + body-width
  limitation evidence
- `PR-8e_CLOSED.md` — accepted "approximate body alignment" surface
- `PR-8d.2_PART2_UX_SPEC.md` — Part 1 disclaimer + Part 2 state UX
- `PR-8d_PALETTE.md` — locked visual palette
