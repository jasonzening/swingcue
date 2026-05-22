# PR-7d NLF Research Pilot Spec

**Date**: 2026-05-22
**Status**: SPEC — pre-implementation; no Modal cycle yet
**Track**: Accuracy research, NOT a production track
**Predecessor**: PR-7a.5 PROBE FAIL (`docs/decisions/PR-7a5_PROBE_FINDINGS.md`)

---

## ⚠️ License gate — model weights are NONCOMMERCIAL RESEARCH ONLY

**Direct quote from NLF README** (`github.com/isarandi/nlf`, verified
2026-05-22 via Chrome MCP):

> "Models for PyTorch and TensorFlow are available for noncommercial
> research use under Releases"

**Interpretation**: the published **model weights** are gated to
noncommercial research use. The training/inference **code** is
separately licensed (typically permissive — to be confirmed at pilot
kickoff by reading `LICENSE` in repo). This split matters because it
opens a future commercial path: train our own weights using the
released code on SwingCue's GT corpus, sidestepping the weights-only
restriction (see Q7 + "Future commercial paths" sections below).

This pilot is **explicitly for accuracy research only** — uses the
released noncommercial weights:

- ❌ NOT for production deployment
- ❌ NOT for paid-product features
- ❌ NOT for the SwingCue commercial pipeline
- ✅ Valid use: comparing model accuracy ceilings to inform whether
  pursuing a commercial-licensed alternative (or training our own
  model) is worth the engineering investment

Any code/weights/data from this pilot stays in `python/pilot/`
(already gitignored output path) and `docs/PR-7d_NLF_PILOT/`. Nothing
NLF-derived enters production runtime, the Railway image, or
commercially-licensed code paths.

If pilot results suggest pursuing NLF-class accuracy, the follow-up
work would be either (a) license negotiation with the authors,
(b) commercial-licensed equivalent (Sapiens, etc.), or (c) training
a SwingCue-licensed model from scratch.

---

## Goal

Validate whether **NLF (Neural Localizer Fields, NeurIPS 2024)** —
specifically its "query arbitrary 3D point" capability — outputs
positions matching Jason's GT labels for the 5 anatomical landmarks
that PR-7a's fitted-offset model handles:

- `left_shoulder` (acromion)
- `right_shoulder` (acromion)
- `left_hip` (greater trochanter / hip socket)
- `right_hip` (greater trochanter / hip socket)
- `neck_center` (throat midpoint)

Per PR-7a.5 findings: SMPL canonical-mesh vertices are 25-50 px from
Jason's GT on these exact landmarks, because SMPL's anatomical
reference differs from Jason's labeling convention. The hypothesis
for this pilot: NLF's design — query 3D points by anatomical reference
position, then localize them in the input image — **may avoid the
canonical-mesh-vs-labeler mismatch entirely**, since the query points
can be specified to match Jason's labeling convention.

---

## Test plan

### Single-frame pilot
- **Video**: b3fea3f0 (face_on)
- **Frame**: 7 (setup phase — golfer stationary, easiest case)
- **GT source**: `docs/PR-7_GROUND_TRUTH/golf/b3fea3f0_setup_face_on.json`
- **Landmarks** (5): the 5 GT-labeled points

### Why setup frame 7
- Body stationary — no motion blur or fast-rotation artifacts
- PR-7a-7a.5 baseline numbers already in the tree:
  - shoulder: 2.1 / 3.6 px (left/right) from PR-7a fitted offset
  - hip: 4.8 / 6.3 px
  - neck: 10.5 px
- Easiest frame for any new model to land cleanly
- If NLF FAILS at setup, no need to test impact/finish — accuracy
  ceiling reached
- If NLF passes setup, expand to impact + finish in a follow-up

### Comparison reference points (already in tree)
| Approach | Setup mean (px from GT) | Source |
|---|---|---|
| Raw WHAM (uncorrected) | 25-100 | shipped PR-7a benchmark numbers |
| SMPL vertex sample (PR-7a.5) | 16-47 | `docs/PR-7a5_PROBE/distance_to_gt.csv` |
| PR-7a fitted offset (current ship) | 2-10 | `docs/PR-7a5_PROBE/distance_to_gt.csv` |
| **NLF pilot (this PR-7d test)** | **TBD** | this spec |

---

## Acceptance criteria

| Criterion | Threshold | Outcome |
|---|---|---|
| NLF prediction within 5 px of Jason GT | ≥ 3 of 5 landmarks | **PASS** — architectural option for future model swap |
| NLF prediction within 5 px of Jason GT | ≤ 2 of 5 landmarks | **FAIL** — ceiling is fundamentally model-quality, not landmark-mapping. PR-7a stack is the achievable accuracy. |

5 px chosen because:
- PR-7a fitted offset achieves 2-10 px at setup. A model that "wins"
  must clearly outperform, not match.
- 5 px is half the spec-§5 class gate (10 px). A model hitting this
  threshold has meaningful headroom for production-grade accuracy.
- At depth ~3.5 m with fx=1280, 5 px ≈ 1.4 cm in 3D — within typical
  pose-estimation noise floor.

Tie-breakers / sanity:
- If NLF lands within 5 px on 3/5 but is wildly wrong (>50 px) on 2/5,
  treat as FAIL — inconsistent model unsuitable even as architecture
  reference.
- Visual sanity check via rendered comparison PNG (red NLF dots, blue
  PR-7a dots, GT) — must be inspectable like PR-7a.5 probe.

---

## Modal cost estimate

| Item | Estimate | Notes |
|---|---|---|
| First Modal Image build (NLF deps) | $0 (free build) | But ~15-30 min wall clock cold-build, similar to WHAM image |
| First inference cold-start | ~$0.10-0.20 | Container init + model load |
| Steady-state per-frame inference | ~$0.05-0.10 | NLF is heavier than WHAM (transformer backbone + dense regression head) |
| Single-frame pilot total | **~$0.30-0.50** | |
| If expanded to 3 frames (setup/impact/finish) | ~$0.50-1.00 | Multi-frame follow-up if pilot passes |

Hard cap: **$2.00 total for the entire NLF research track**, including
re-runs to debug. Surface if costs exceed.

---

## Dependencies (TO BE VERIFIED at pilot kickoff)

### Software
- **NLF repo**: `github.com/isarandi/nlf` (per memory of NeurIPS 2024
  paper authorship — verify URL exact)
- **Model weights**: Likely a separate download (a few hundred MB to
  ~2 GB). Need to confirm hosting + license terms before download.
- **Python deps**: torch + cuda + transformers stack. NLF likely uses
  similar versions to WHAM (torch 1.11+cu113 or newer). Will need
  Modal image rebuild (~10-15 min) on first deploy.
- **Body model**: NLF may or may not require SMPL — verify at kickoff.
  If it does, we already have `local_models/smpl/_extracted/` ready.

### Infrastructure
- **Modal A10G**: same GPU class as WHAM pilot. Verify NLF VRAM fits.
- **Modal Volume**: new volume for NLF weights, separate from WHAM's
  `/models/wham`. Naming: `swingcue-pilot-nlf-models`.
- **Existing pilot scaffolding**: reuse `python/pilot/modal_app.py`
  pattern, add NLF-specific image + function entry.

### Data
- **Input frame**: extract from `python/benchmark/test_videos/b3fea3f0-*.mp4`
  at frame 7. Already done by PR-7a.5 probe (just decode again).
- **GT labels**: existing `docs/PR-7_GROUND_TRUTH/golf/b3fea3f0_setup_face_on.json`.
- **No new GT labels needed for the pilot.**

---

## Open questions to resolve at pilot kickoff

Flagging, not answering:

1. **Does NLF run on a single A10G?** Memory footprint vs WHAM unknown.
   May need A100 ($1.10 → $4.00/hr); changes cost model significantly.

2. **NLF input format**: is it 2D image only, 2D image + bbox, or
   2D image + 3D query points (SMPL canonical)?
   - If pure 2D → straightforward
   - If requires 3D query points → we need to construct them in SMPL
     canonical space (we have `local_models/smpl/_extracted/` from PR-7a4)

3. **Anatomical query specification**: can we query landmarks by name
   (e.g., "left_acromion", "left_greater_trochanter"), or do we
   specify 3D query points in mesh-canonical coords and the model
   localizes them?
   - If by-name: easy alignment with Jason's GT labels
   - If by-3D-query: we re-use the same T-pose-derived indices from
     PR-7a4 + take the resulting NLF output instead of the SMPL vertex

4. **License verification**: where exactly does NLF stand?
   - Apache 2.0?
   - Custom noncommercial?
   - Research-only with explicit commercial-license-on-request?
   - This decides what "future commercial path" looks like.

5. **Output format**: 3D world-frame, camera-frame, or 2D pixel?
   PR-7a probe code expects 3D camera-frame + we project via
   `motion_correction.engine.projection.default_intrinsics`. If NLF
   emits something else, add a thin adapter.

6. **Batch vs single-frame inference**: does NLF support batched
   per-frame inference or only one-shot? Matters for cost projection
   to multi-frame.

7. **Future commercial path — train our own weights?** Per the README,
   NLF training code is provided for both PyTorch and TensorFlow. The
   noncommercial restriction applies to the released weights, not the
   training pipeline. **Long-term commercial path could be: train our
   own NLF-style weights on SwingCue's GT corpus** (15 samples today,
   growing), sidestepping the weights-only restriction. This is not
   in pilot scope but is a real strategic option vs "passively wait
   for license change." Pilot result determines whether this path is
   worth the engineering investment:
   - Pilot PASS → training-our-own becomes a credible 1-2 month
     investment (data collection + training infra + iteration)
   - Pilot FAIL → no point training our own; the architecture itself
     isn't the bottleneck

---

## What this pilot UNBLOCKS

- **Decision: is a model swap worth pursuing?**
  - PASS → start scoping a commercial-licensed path (see "Future
    commercial paths" below).
  - FAIL → PR-7a stack is the achievable ceiling. Frontend track ships
    against it. Accuracy improvements via larger GT corpus / multi-
    labeler consensus, not new pose backbone.
- **Decision: where to invest engineering hours next?**
  - PASS → build commercial-equivalent pipeline (sizeable work).
  - FAIL → frontend rendering polish, more GT labels for fit refinement,
    pilot user testing of the current ceiling.

### Future commercial paths (if pilot PASSES)

Three options, in increasing engineering cost:

1. **Passive wait for NLF license change.** Watch the upstream repo /
   reach out to authors. Zero engineering cost; uncertain timeline;
   may never materialize.
2. **Train our own NLF-style weights on SwingCue GT corpus.** Per Q7
   above — code is released; restriction applies to published weights
   only. SwingCue's GT corpus (15 samples today, expandable) becomes
   the training set. Estimated 1-2 months engineering + GT scaling +
   training infra setup. Outcome: commercial-licensed SwingCue-trained
   weights with NLF-class architecture. Most aligned with long-term
   product control.
3. **Commercial-licensed alternative model.** Sapiens-2 (Meta, custom
   license that's commercially-viable below 700M MAU), custom-trained
   from scratch on different architecture, or licensed NLF via author
   negotiation. Variable cost; depends on which candidate.

Pilot result informs which of these is worth pursuing — failing pilot
means architecture itself isn't the issue, no point investing in any
of them.

## What this pilot does NOT do

- ❌ Replace WHAM in production analyze
- ❌ Wire NLF into `python/main.py /analyze` flow
- ❌ Change PR-7c-frontend scope
- ❌ Modify PR-7a motion_correction engine
- ❌ Modify `wham_runner.py` again
- ❌ Add new GT labels
- ❌ Push to origin

This is **a research-only single-frame benchmark to inform strategy**,
not a buildable feature. Output: one PNG + one CSV + one decision.

---

## Out of scope for this PR-7d spec

The following are intentionally deferred to follow-up PRs once the
single-frame pilot result is known:

- NLF Modal image build (heavy lift if pursued)
- NLF inference function wired into pilot infrastructure
- Multi-frame test expansion (impact + finish + DTL view)
- Comparison vs other model swaps (Sapiens, 4D-Humans, Human3R)
- Production-migration plan
- Commercial license negotiation
- Hybrid architecture (NLF for landmarks + WHAM for full timeline)

---

## Estimated session budget (when pilot runs)

| Phase | Time |
|---|---|
| A. License verification + dependency audit (read repo READMEs, confirm weights URL) | 30-45 min |
| B. Modal image build for NLF (mostly wall-clock during background build) | 30-60 min wall |
| C. Pilot run on b3fea3f0 frame 7 | 5-10 min wall |
| D. Probe rendering + acceptance + writeup | 20-30 min |
| **Total session** | **~2-3 hours** |

If license verification reveals a hard blocker (e.g., NLF weights
require paid license or are gated to academic affiliations), abort
at Phase A and pivot to next model candidate.

---

## File layout (when pilot runs)

```
python/pilot/
  modal_nlf_app.py             # NEW: NLF Modal image + function
  scripts/
    probe_nlf_landmarks.py     # NEW: pilot runner + comparison renderer
docs/
  PR-7d_NLF_PILOT/             # NEW: pilot artifacts
    b3fea3f0_face_on_setup_nlf_compare.png
    nlf_landmark_distances.csv
    license_audit.md           # snapshot of NLF license terms at pilot date
  decisions/
    PR-7d_NLF_RESEARCH_PILOT_SPEC.md   # this doc
    PR-7d_NLF_PILOT_FINDINGS.md        # post-pilot PASS/FAIL report
```

`pilot/output/` and `pilot/scripts/` are already gitignored or
otherwise excluded from the Railway COPY allowlist — zero production
impact.

---

## Recommendation: when to actually kick this off

This spec is ready. Whether to actually run the pilot is a strategy
decision:

- **Run NOW** if: accuracy ceiling matters for product/user testing
  in the next 2 weeks AND you want to know whether to invest in
  model-swap engineering.
- **Defer** if: PR-7c-frontend + early user feedback will inform
  whether the current accuracy ceiling is acceptable. May make NLF
  pilot unnecessary.
- **Skip entirely** if: noncommercial-only is a dealbreaker (no
  realistic path to production), and there's no other practical
  candidate (Sapiens-2, custom-trained, etc.) on the horizon.

Default: **defer until after PR-7c-frontend ships and gets initial
user feedback**. Frontend may make the accuracy question moot if
users find the current ceiling acceptable.
