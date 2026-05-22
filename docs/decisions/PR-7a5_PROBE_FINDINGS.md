# PR-7a.5 Probe Findings — SMPL Vertex Sampling Investigation

**Date**: 2026-05-22
**Status**: PROBE FAIL — SMPL vertex sampling abandoned in this architecture
**Outcome**: PR-7a fitted-offset model is the WHAM/SMPL backbone accuracy
ceiling. Further accuracy improvement requires a model swap.

---

## Hypothesis

PR-7a.4 yesterday hypothesized that sampling SMPL mesh vertices at
anatomical landmarks (acromion peak, greater trochanter, lateral
malleolus, etc.) would beat the PR-7a fitted-offset model. Rationale:
direct mesh-surface points eliminate the SMPL-bone-center-vs-skin-
surface mismatch that PR-7a compensates for via trained offset vectors.

Yesterday's probe couldn't run (wham_runner.py dropped pose/trans/verts).
PR-7a.5 patched the upstream serialization (additive, env-flagged via
`--save-smpl-params`), re-ran WHAM on b3fea3f0, and tested vertex
sampling at the 3 GT-labeled phase frames (setup/impact/finish).

## Result: PROBE FAIL

**Acceptance gate**: SMPL vertex must beat PR-7a anchor on ≥2 of 3 frames
for BOTH `acromion_left` and `acromion_right`.

- `acromion_left`: **0/3** new<old → **FAIL**
- `acromion_right`: 2/3 new<old → PASS
- `greater_trochanter_left`: 0/3 (mean: old 4.9 px, new 25.4 px) → off-gate
- `greater_trochanter_right`: 0/3 (mean: old 11.7 px, new 36.2 px) → off-gate
- `throat_midpoint`: 0/3 (mean: old 16.4 px, new 53.1 px) → off-gate

**Gate logic**: Both acromion sides must PASS. `acromion_left` FAILED
0/3, so overall PROBE FAIL regardless of `acromion_right`'s 2/3 PASS.

PR-7a fitted offsets won **11 of 15** total head-to-head distance
comparisons. SMPL vertex sampling won only 2 (acromion_right at impact
and finish — where the PR-7a fit was weakest to begin with).

## Per-frame breakdown

| Frame | Phase | Landmark | PR-7a (px) | SMPL vertex (px) | Δ |
|---|---|---|---|---|---|
| 7 | setup | acromion_left | 2.1 | 16.8 | **−14.6** |
| 7 | setup | acromion_right | 3.6 | 47.0 | **−43.4** |
| 7 | setup | greater_trochanter_left | 4.8 | 25.4 | −20.5 |
| 7 | setup | greater_trochanter_right | 6.3 | 26.7 | −20.4 |
| 7 | setup | throat_midpoint | 10.5 | 37.7 | −27.2 |
| 90 | impact | acromion_left | 10.8 | 27.5 | −16.7 |
| 90 | impact | acromion_right | 29.5 | 16.3 | **+13.2** |
| 90 | impact | greater_trochanter_left | 7.9 | 25.9 | −18.0 |
| 90 | impact | greater_trochanter_right | 14.1 | 31.8 | −17.7 |
| 90 | impact | throat_midpoint | 10.7 | 67.6 | −56.9 |
| 125 | finish | acromion_left | 20.9 | 30.1 | −9.2 |
| 125 | finish | acromion_right | 28.5 | 14.9 | **+13.7** |
| 125 | finish | greater_trochanter_left | 1.9 | 24.9 | −23.0 |
| 125 | finish | greater_trochanter_right | 14.7 | 49.9 | −35.3 |
| 125 | finish | throat_midpoint | 28.0 | 54.0 | −26.0 |

Setup frame is the most damning: PR-7a anchors are within 2-10 px of
GT. SMPL vertex sampling is 16-47 px off. Setup is where the body is
stationary and the SMPL mesh fit should be most accurate.

## Root cause — SMPL canonical vertices don't match Jason's GT labels

Visual inspection of the 3 rendered comparison PNGs confirms the
vertices are anatomically plausible (they land on plausible body
parts), but they consistently miss Jason's specific GT landmarks:

| Landmark | SMPL vertex position vs GT |
|---|---|
| acromion (4721 left / 1238 right) | Lands ~1-2 cm above-and-inside the actual shoulder peak Jason marks. Off by 15-47 px after projection. |
| greater_trochanter (6375 left / 2915 right) | Lands at the **iliac crest** (top of the hip bone, ~10 cm above the trochanter). Jason's "hip" GT is at the trochanter / hip-socket level. |
| throat_midpoint (vertex 444) | Lands at the **chin / upper-face** region. Jason's "neck_center" GT is at the throat-midline (~5-10 cm lower). |
| head_crown (411) | Lands correctly at top of head ✓ |
| lateral_epicondyle / lateral_malleolus | Land plausibly on lateral knee/ankle (no GT to compare) |

The T-pose-derived vertex indices (PR-7a4 probe) are picking real
SMPL canonical anatomical points — they're just **not the same points
Jason labels as ground truth**. The mismatch is intrinsic to SMPL's
canonical-mesh definition, not a bug in vertex index derivation.

## Why fitted offset wins

PR-7a's per-(joint, phase, view) fitted body-local offset vector is
**trained against Jason's exact GT labels**. The fit absorbs the
SMPL-vs-GT bias directly:

- WHAM emits joint center at glenohumeral (~10 cm lower-inside than acromion)
- PR-7a fit learns: "push shoulder up-and-out by [d_h, d_v, d_f] in body-local"
- After PR-7a.2 chirality fix + PR-7a.3 bidirectional EMA, the corrected
  anchor lands within 2-10 px of Jason's GT acromion at setup

SMPL vertex sampling can't do this absorption — it uses canonical mesh
positions defined by the SMPL author's anatomical reference, which
doesn't align with Jason's labeling convention. No vertex index
exists that matches all 5 of Jason's GT label points across the body.

The two approaches sit at fundamentally different points on the
bias-variance trade-off:
- **PR-7a fitted offset**: low bias against THIS labeler's convention
  (15-sample fit minimizes residual), high model complexity
- **SMPL vertex sampling**: zero training, but uses SMPL's canonical
  anatomy as ground truth — which differs from Jason's

## Why this is the WHAM/SMPL backbone ceiling

The mismatch is at the **model level**, not the post-processing level.
Both approaches use the same WHAM-fitted SMPL mesh. The vertices and
the bone centers come from the same underlying SMPL parameterization.
Reshaping the post-process (fitted offset, vertex sampling, hybrid)
can only redistribute residual error — it can't fix the fact that
WHAM's posed mesh has a fixed registration error vs the actual video
body, and that error is what dominates beyond the SMPL-vs-GT bias.

The PR-7a stack already squeezes the post-process accuracy to its
honest limit on this backbone:
- shoulder: 8.1 px (gate <10) GREEN
- hip: 9.3 px (gate <10) GREEN
- head_spine: 8.9 px (gate <12) GREEN

Further gains require a different pose backbone — one that either
(a) regresses skin-surface landmarks directly, or (b) achieves tighter
mesh-to-video registration than WHAM. Neither is a motion_correction
engine problem.

## Recommendation

1. **Abandon PR-7a.5 SMPL vertex sampling** in this architecture.
2. **Ship the PR-7a stack as accuracy ceiling** on the WHAM+SMPL backbone.
3. **Accuracy improvements track**: model swap pilot (PR-7d-research).
   First candidate: **NLF (Neural Localizer Fields, NeurIPS 2024)** —
   query arbitrary anatomical points by 3D coordinate. Currently
   noncommercial license → research pilot only. See
   `docs/decisions/PR-7d_NLF_RESEARCH_PILOT_SPEC.md` for the pilot plan.
4. **Frontend track unblocked**: PR-7c-frontend can proceed knowing
   the 5-anchor PR-7a stack is the ceiling for the foreseeable future.

## Probe artifacts preserved

For replay or future re-evaluation against a different backbone:

- `python/pilot/runners/wham_runner.py` — patched with
  `save_smpl_params` opt-in (additive, gated, zero production cost).
- `python/pilot/scripts/run_wham_one.py` — `--save-smpl-params` flag.
- `python/pilot/scripts/probe_smpl_vertex_landmarks.py` — local
  rendering + acceptance script. Self-contained, reusable.
- `python/pilot/output/wham/b3fea3f0-*/smpl_params.npz` (gitignored,
  10 MB sidecar) — full verts + pose + trans + betas. Regenerable
  via `run_wham_one.py b3fea3f0-... --save-smpl-params` ($0.01 Modal).
- `docs/PR-7a4_PROBE/smpl_landmark_indices.json` — 11 T-pose-derived
  landmark vertex indices.
- `docs/PR-7a5_PROBE/{3 PNGs, distance_to_gt.csv}` — comparison
  renders + numbers.

Future probe of a different SMPL-based model can reuse the same
landmark indices + probe script with a swapped backbone.

## Time elapsed (probe)

~30 min total (under 2-hour budget):
- Phase A (wham_runner patch + smoke): 10 min
- Phase B (Modal cycle): 5 min wall (background)
- Phase C (probe script + render): 10 min
- Phase D (acceptance + analysis): 5 min
