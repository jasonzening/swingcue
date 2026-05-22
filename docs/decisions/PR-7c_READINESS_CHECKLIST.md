# PR-7c Readiness Checklist — Frontend Integration of Motion Correction

**PR-7a closure**: commit `5ac60da` on `track2/phase2-bone-center-pilot`
(local, not pushed). 49/49 tests pass; all spec-§5 acceptance gates GREEN.

**PR-7c scope**: wire the offline corrected timeline into the production
frontend and analyze API, then ship.

---

## 1. Frontend integration scope

| File | Change |
|---|---|
| `src/components/SkeletonOverlay.tsx` | Stop rendering raw WHAM skeleton by default. Add a debug toggle (Settings.skeletonDebug) to show raw vs corrected side-by-side for QA. |
| `src/components/SwingPlayer.tsx` | Read `coaching_anchors_2d` per frame (interpolate between sampled frames using ts). Render via the new MagentaAnchor overlay layer. |
| `src/components/MagentaAnchorOverlay.tsx` (new) | 5 magenta anchors (left/right shoulder, left/right hip, spine_mid). Per-side anchors as filled circles r=12 (matches probe rendering convention); spine_mid as smaller hollow ring r=8 stroke=2. |
| `src/lib/skeleton/computeDiscParams.ts` | Replace raw-keypoint disc-scale computation with `setup_baseline.base_shoulder_width` from the corrected timeline. Scale stays constant during the swing (matches PR-7a setup-lock behavior). |
| `src/lib/skeleton/coachingAnchors.ts` (new) | Helper to load `coaching_anchors_2d` per frame, handle nulls (frame skip → null anchor → don't render that frame), per-anchor confidence threshold. |

## 2. Phase-aware rendering strategy (Path E mitigation for residual drift)

Per spec instruction: "soften anchor opacity to mask residual WHAM drift".

| Phase | Anchor opacity | Skeleton edges |
|---|---|---|
| setup | 1.0 (full) | hidden |
| backswing | 1.0 | hidden |
| top | 1.0 | hidden |
| transition | 0.85 | hidden |
| downswing | 0.65 | hidden |
| **impact** | 0.45 (residual WHAM drift acknowledged here) | hidden |
| **finish** | 0.3 fading to 0 at clip end | hidden |

The opacity ramp is the user-facing answer to the WHAM tracking-quality
ceiling identified in PR-7a (top/impact/finish drift, NOT a smoothing
issue per the bidirectional EMA fix in PR-7a.3). Lower opacity at
fast-motion phases trades visual prominence for residual accuracy —
acceptable per Jason's "drift is known limit, ship" instruction.

## 3. Coaching anchor consumption

Schema source: `python/motion_correction/schemas/corrected_timeline.py`
→ `CorrectedFrame.coaching_anchors_2d: dict[str, Optional[UV]]`.

The 7 anchor names emitted by `domains/golf/coaching_anchors.py`:
- `left_shoulder_visual`, `right_shoulder_visual` (per-side)
- `left_hip_visual`, `right_hip_visual` (per-side)
- `neck_visual` (single)
- `shoulder_disc_center`, `hip_ring_center` (midpoints — derived)

Frontend renders 5 filled (per-side + neck) + 2 hollow (disc centers).

## 4. Production analyze API integration

Currently `python/main.py` calls WHAM via `pilot/runners/wham_runner.py`
and writes raw JSON. PR-7c needs:

1. After WHAM completes, invoke `motion_correction.engine.orchestrator.correct_timeline`
   with the GolfCorrectionPlugin + auto-detected view (face_on vs
   down_the_line — already has a detector in production analyze).
2. Replace the WHAM raw JSON written to Supabase Storage with the
   CorrectedTimeline JSON. Schema additions are backward-additive
   (existing `joint_centers_3d` still present) but the frontend should
   read the new `coaching_anchors_2d` + `setup_baseline` fields
   exclusively going forward.
3. Performance: PR-7a.3 bidirectional EMA requires the full timeline
   before write — fine for offline analyze; the bidirectional code path
   is the default. If real-time streaming surfaces later, pass
   `bidirectional=False` to `correct_timeline` (forward-only causal path,
   already implemented and tested).

Cost: motion_correction is pure Python on CPU, ~50 ms per 100 frames
on a single core. No GPU. No Modal. Runs in the existing analyze worker
without infrastructure change.

## 5. PR-7a.5 future ticket — SMPL vertex sampling

Spec: `docs/decisions/PR-7a4_PROBE_FINDINGS.md` for full rationale.
~10-12 hr investment. Prerequisites:
1. Patch `wham_runner.py` to save `smpl_pose` + `smpl_trans` per frame.
2. Re-run WHAM on all 3 test clips to backfill (~$0.03 Modal).
3. Install `smplx` package in `.venv-benchmark`, wire local SMPL
   forward pass.
4. Use the existing `probe_smpl_vertex_landmarks.py` (already built
   this session, awaits verts data).
5. If probe passes acromion-beats-anchor gate on ≥3 of 4 face_on frames,
   swap engine mode A vector lookup with vertex-index lookup.
6. Visual gates on knee/ankle/foot landmarks (new — not currently in
   PR-7a coaching anchors).

Defer until PR-7c user feedback on opacity-ramp strategy. If users
report drift is visually masked acceptably, PR-7a.5 may be unnecessary.

## 6. Parallel research thread — NLF (Neural Localizer Fields)

NeurIPS 2024. Architecturally superior: query arbitrary anatomical
points by 3D coordinate (no fixed joint set, no SMPL bone-center
mismatch). Currently noncommercial license → unsuitable for SwingCue
production. Pilot value: cap on what motion correction can achieve
on better upstream pose data.

**Separate ticket, lower priority, not blocking PR-7c.** Pilot scope:
~$0.50 Modal spend to run NLF on 3 clips, compare landmark accuracy
against PR-7a corrected anchors. If NLF outperforms PR-7a by >50%,
revisit license terms or look for similarly-architected commercial
alternatives.

---

## PR-7c kickoff hard constraints

- NO modifications to `motion_correction/engine/` (locked at PR-7a closure)
- NO Modal pipeline changes
- NO new GT labels needed (the 15 we have are sufficient for PR-7c)
- Frontend changes only + 1 production wiring change in `python/main.py`
- Phase-aware opacity ramp is the user-facing answer to WHAM drift —
  don't try to architecturally fix what's already shipped
