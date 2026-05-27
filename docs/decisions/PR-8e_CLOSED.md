# PR-8e — CLOSED (MVP anatomical landmark improvement only)

Closeout date: 2026-05-26
Status: shipped (commits `feaa7bc` → `894e873`), Modal deployed,
production rendering on `wham_status='ready'` videos.

## Scope of this PR — explicit

PR-8e addressed exactly one root cause: **SMPL kinematic joint centers
do not coincide with anatomical body-surface landmarks.** The fix
replaces the rendered position of two joint pairs (shoulder, hip) with
landmarks that sit closer to the visible body surface.

**This does NOT solve skeleton-vs-body alignment overall.** Other
sources of visible drift remain in place (see Known limitations).

## Final implementation

### Acromion (shoulder) — vertex-sourced
| Schema field | SMPL mesh vertex | Notes |
|---|---|---|
| `acromion_left` | 4721 | PROBE anat-left; projects image-left for face-on |
| `acromion_right` | 1238 | PROBE anat-right |

Replaces SMPL `left_shoulder` / `right_shoulder` (glenohumeral joint,
medial to acromion) as the cyan/amber render position when the
anatomical key is present in the row.

### Greater trochanter (hip) — derived
| Schema field | Method | Source joint |
|---|---|---|
| `greater_trochanter_left` | `(hip.x, hip.y + 30·image_h/1024)` | `left_hip` |
| `greater_trochanter_right` | same | `right_hip` |

Replaces SMPL `left_hip` / `right_hip` (≈L5 vertebra, above visible
hip surface) as the cyan/amber render position. Two mesh-vertex
PROBE candidates (6375/2915 and 4934/1490) both projected above the
pelvis joint — opposite of anatomical reality — so trochanter is
**derived**, not vertex-sourced. See `meta.anatomical_derived_landmarks`.

### Optional landmarks emitted but NOT rendered
Backend writes them to `keypoints_2d_projected` + `keypoints_3d_smpl`,
frontend doesn't yet pick them up. Available for future PRs:
- `c7` (vertex 414)
- `throat` (vertex 444)
- `lateral_epicondyle_left` / `_right` (4447 / 959)
- `lateral_malleolus_left` / `_right` (6749 / 3348)

### Backward compatibility
Frontend `resolveJointPos` in `WhamSkeletonOverlay.tsx`:
- New row has anatomical key → render at anatomical position
- New row has anatomical key but null this frame (Z-guard fired) → render at SMPL fallback
- Old row (pre-PR-8e.0) missing anatomical key → render at SMPL fallback

Old `wham_status='ready'` videos render identically to PR-8d.1. No
data migration was performed; legacy rows from the PR-8e.0 / 8e.0.1
chirality-wrong era (cbf4f22c, a3f7b0d8) and the 8e.2 step-1 vertex-
wrong era (92a483fc) retain bad anatomical data but are observably
inconsistent — reprocess if you care.

## Known limitations — **do NOT overclaim that these are fixed**

### 1. Occluded-side shoulder ambiguity
WHAM is monocular 3D from a single video. When one shoulder is
occluded by the torso during the backswing/follow-through, WHAM
infers its 3D position from priors, not from visible pixels. The
projected 2D position for that frame is a guess. Renders as the
skeleton "wobbling" through the swing.

**Inherent to WHAM-class monocular reconstruction. Not solvable
without a stronger model or multi-camera input.**

### 2. Mid-swing perspective drift
PR-8b.3 median-z stabilization fixes ~0.8m fake depth "breathing"
on fixed-tripod video, BUT introduces phase-aware distortion:
during the actual swing, true depth deviates from the median by up
to 0.36m (T3: z range 4.33-5.20m, median 4.84m). The skeleton renders
at the median depth, so during mid-swing the projection is slightly
off from where the body actually is in image space.

**Reads as "skeleton looks delayed" relative to the body at peak
swing speed. Not actually a latency bug — it's a depth/scale error
synchronized with swing phase.**

PR-8f scope. Deferred.

### 4. Projected skeleton scale underestimates real body proportions
**Source: clean-video acid test, 2026-05-26.**

Acid test with a user-recorded clean video (close-distance, single
person, face-on, ~6-8s, vid `f97741c0-c6a8-46e5-9ec1-a7dff0acfa61`)
failed. Real shoulder width estimated at 25-30% of frame width
(180-220px on a 720-wide frame); WHAM projects the acromion-to-acromion
distance to ~108px (~15% of frame width). The projected skeleton is
~50% of real body scale.

The PR-8e acromion shift adds ~8px lateral offset per shoulder over
the SMPL glenohumeral joint position — too small to bridge a 70-100px
shortfall in shoulder width. Acromion vertex selection is NOT the
right knob.

**PR-8f attempted (and ABANDONED) — β=0 SMPL override does NOT fix this.**

PR-8f investigation found WHAM's per-video β fit on f97741c0 has
β[1] mean +1.50 (vs population mean 0). Hypothesis: WHAM is fitting
a "narrow build", and forcing β=0 would restore population-mean
shoulder width.

Implementation deployed to Modal, re-ran inspect_pkl on f97741c0,
measured frame-0 world coords pre- vs post-override:

| Measurement | Pre (β-fit) | Post (β=0) | Δ |
|---|---|---|---|
| Shoulder world width | 0.267 m | 0.275 m | +3% |
| Hip world width | 0.271 m | 0.295 m | +9% |
| Head-to-ankle height | ~1.37 m | ~1.36 m | −1% |

Projected pixel equivalents (focal=1469, Z=3.98m, scale 369 px/m):
shoulders ~98px → ~101px. **Target was 160-180px.**

β=0 override is correct engineering but produces ~1/20th of the
required scale change. Setting all β to 0 only changes mesh verts
by a few mm because WHAM's β was already producing a mesh close to
SMPL neutral in overall scale — the narrowness is NOT primarily
β-driven.

**PR-8f abandoned 2026-05-27.** Modal rolled back to pre-PR-8f
state; commit `e28e9e7` (chore inspect_pkl β audit) preserved as a
useful diagnostic for future "narrow / wide / off-proportion"
debugging, but no production code shipped.

**Where the narrowness actually lives — open question for PR-8g**:
Three candidates remain, two of which were ruled "fine" in the
PR-8f preview but should be re-verified more rigorously now that
β is eliminated:
- **(A) CLIFF focal `sqrt(w² + h²)`** — preview accepted, but a
  20-40% focal error in the right direction could account for the
  observed narrowness. Need to compare against WHAM's INTERNAL
  focal (its forward pass uses some focal during inference; ours
  is the SEPARATE projection we do downstream).
- **(B) Z-stabilized trans_z** — preview said range was small,
  but the MAGNITUDE of trans_z (3.98 m median) may not match the
  user's ACTUAL camera distance. If user was at 2.5 m but WHAM
  reports 4 m, projected scale is 0.625× — that's a 37% shortfall
  matching what we see.
- **(D) WHAM model bias** — possible that WHAM's training data
  systematically underestimates shoulder span for this clothing /
  pose / video-class. Not user-fixable; only model swap helps.

**Not magic-constant tunable.** Do not nudge acromion vertex offsets,
trochanter constants, or scale multipliers to compensate; the
deficit is real geometry, not surface-landmark anatomy.

### 3. Trochanter offset is calibrated for a typical case
The `30 * image_h/1024` constant was derived from:
- ~10cm anatomical pelvis-to-trochanter distance
- ~5m typical golfer-to-camera depth
- CLIFF focal `sqrt(w² + h²)`

For users at non-typical depths (closer / farther / different focal
length), the projected anatomical offset varies but the rendered
offset stays constant. Result: hip dot may still appear slightly off
the visible hip surface even though the temporal motion looks smooth.

**Smooth temporal motion does NOT imply anatomically correct
placement.** A deterministic offset eliminates the visual "jitter"
but doesn't pin the dot to the actual anatomical position frame-by-
frame across all camera setups.

## Mitigations available but explicitly NOT implemented

The following could improve perceived accuracy. None shipped in
PR-8e; bookmarked for later.

| Idea | Why deferred |
|---|---|
| Temporal smoothing / EMA on occluded-side joints | Doesn't fix occlusion-ambiguity root cause; can mask issues that need attention |
| Per-joint confidence-based opacity (fade out low-confidence joints) | Would require WHAM to expose per-joint confidence; not in current output |
| Frontend bilateral consistency constraint (force L/R distance from spine) | Adds visual stability at the cost of biomechanical accuracy during asymmetric poses (impact, P-system) |
| 3D-correct trochanter derivation along the pelvis→head_crown axis | Frontend consumes 2D only today; not yet a consumer requirement |
| Re-PROBE SMPL mesh for true greater trochanter vertex | Two attempts already failed; switching to derivation made more progress for less effort |

## Hold rule — do NOT continue tuning the trochanter offset

Further `+/-` adjustments on the 30px base or the `image_h/1024`
denominator become magic-constant tuning. Real improvements require
one of:
- Better WHAM-class model (out of scope)
- Better projection calibration / phase-aware z-stab (PR-8f or beyond)
- Confidence-based rendering decisions (future)

If a future visual report says "trochanter still looks wrong", the
correct response is to look at the **type** of wrong (consistent bias?
phase-aware drift? camera-distance dependent?) and route to the right
follow-up PR — not to nudge the constant.

## Next-step gating

Before starting **PR-8f** (phase-aware z-stab) or **PR-8d.2** (5-stage
progress UX, or any other follow-up), Jason will provide a clean test
video:
- Close-distance, single-person, single-camera, ~6-8s, face-on golf swing
- T3 is a stress-test input, not representative of MVP target use

If the clean video shows visible drift after PR-8e ship, escalate to
**PR-8f phase-aware z-stab**. If it looks fine, the WHAM-skeleton
rendering work is complete for MVP and the next priority is
**PR-8d.2 progress UX**.

**Update 2026-05-26**: clean-video acid test failed, but on a
different root cause than the original PR-8f scope (z-stab phase
distortion). The dominant visible issue is body-scale
underestimation (limitation #4), not phase-aware depth drift.
**PR-8f scope was expanded** from "phase-aware z-stabilization"
to "projection-calibration investigation":

| Sub-question | What it tests | How |
|---|---|---|
| (A) CLIFF focal | Is `sqrt(w² + h²)` the right focal? | Read WHAM/CLIFF source for the focal formula it actually uses internally; compare to ours |
| (B) Z-stab phase distortion | Is trans_z biased high vs reality? | SQL on the clean video's `_trans_z_median_value_m` + `_trans_z_raw_range_m`; back-derive expected scale from a known anatomical reference (e.g., assume adult height ~1.7m, compute pixel head-to-foot, derive implied focal/Z) |
| (C) SMPL β shape | Is WHAM fitting a narrow body? | Read `smpl_shape` β-10 vector via the existing `inspect_pkl` Modal function (reads from per-video cached pkl, no backend edit) |

Output of PR-8f was supposed to be **data, not code**: which of
A/B/C dominates, backed by numbers. Then a targeted fix proposal.

**Update 2026-05-27**: PR-8f closed without shipping any
production fix. Investigation found (C) β fit ~1.5 SD above
population mean on β[1], implemented β=0 override as targeted fix,
but the override produced only 3-9% scale change (limitation #4
above for numbers). β is NOT the dominant contributor despite the
suggestive raw values. Modal rolled back to pre-PR-8f code.

**PR-8g — projection / depth investigation (re-opened)**:
With (C) ruled out empirically, the remaining candidates are
(A) focal, (B) trans_z magnitude (not range), and (D) WHAM
model bias. (A) and (B) need a deeper re-check than the PR-8f
preview gave them — the preview did a quick reasonableness
check, not a calibrated measurement against a reference.

PR-8d.2 progress UX remains gated until PR-8g identifies the
dominant contributor and ships its targeted fix.

## Commit trail

| Commit | What |
|---|---|
| `18206ad` | PR-8d.1 — initial WHAM skeleton renderer (SMPL joints only) |
| `6be94bb` | PR-8e.0 — emit 10 anatomical landmarks via SMPL mesh vertices |
| `feaa7bc` | PR-8e.1 — frontend `resolveJointPos` shoulder→acromion, hip→trochanter |
| `acf3295` | PR-8e.0.1 — fix anatomical L/R chirality (acromion verified correct) |
| `9c5d046` | PR-8e.2 step 1 — try trochanter vertex 4934/1490 (failed, still above pelvis) |
| `894e873` | PR-8e.2 step 2 — Option W, trochanter derived from hip + Y offset |

## Files touched

- `python/pilot/runners/wham_runner.py` — anatomical landmark dict + per-frame extraction/derivation + meta metadata
- `src/components/WhamSkeletonOverlay.tsx` — `resolveJointPos` with anatomical override + SMPL fallback

No DDL, no Supabase migration, no frontend route changes outside the
overlay component.
