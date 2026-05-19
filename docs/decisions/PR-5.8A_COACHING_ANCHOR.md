# PR-5.8A — Render-time SwingCue Coaching-Anchor Expansion (Shoulder + Hip)

**Date**: 2026-05-19
**Status**: Implemented (branch `pr-5.8a/shoulder-hip-coaching-anchor`) · Pending Jason visual approval
**Scope**: Frontend render only. No backend, no DB, no `pose_timeline_2d` change.
**Relationship to PR-5.8 main spec**: precursor — see §6 below.

---

## 1. Definition

> SwingCue coaching shoulder anchor = the visible point where the humerus connects to the body in the rendered overlay. After expansion:
> - shoulder-line spans actual coaching shoulder width
> - shoulder → elbow line falls on the upper-arm visual midline
> - shoulder disc is anchored on these coaching points, not on the inner MediaPipe/COCO points

Same idea applies to the hip → side-of-torso visual junction. This is a **SwingCue coaching visual correction, NOT an anatomical/medical relabeling.**

---

## 2. The three Jason decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | **Render-time only.** The raw values stored in `pose_timeline_2d` are never mutated. Expansion is applied per frame at draw time. | Keeps the data source pristine; existing analyzed videos do not need re-analyze; PR-5.8 main schema migration remains a clean future move. |
| 2 | **Defaults: shoulder 0.40, hip 0.25.** URL-tunable via `?shoulderExpand=...&hipExpand=...` (clamped to [0, 1.5]). | First-cut calibration on b3fea3f0 face-on setup frame; URL override lets us iterate visually without redeploys. |
| 3 | **Scope: shoulder + hip only.** Elbow, wrist, knee, ankle, head, hand, foot all render raw in this PR. | Smallest surgical fix for the disc-visual-detached problem. The full keypoint reshuffle (head_crown, hand_L/R, foot_L/R) belongs in PR-5.8 proper. |

---

## 3. Math (single source of truth)

```
mid       = (L + R) / 2
expandedL = mid + (L - mid) * (1 + factor)
expandedR = mid + (R - mid) * (1 + factor)
```

The `(1 + factor)` form means a `factor = 0.40` expands EACH side outward by 40% of its current offset from the midpoint — NOT 40% of the total L↔R width. Total width grows by 40% (because both sides expand symmetrically), NOT by 80%.

`factor = 0` is a no-op fast path (callers who don't care about expansion get the legacy behaviour byte-for-byte).

The same math is duplicated in two places:

1. `src/lib/skeleton/coachingAnchors.ts` — pure helper, consumed by SkeletonOverlay.
2. `src/lib/disc/computeDiscParams.ts` — private `applyExpand` co-located with the existing `validPair` tuple shape so callers don't have to juggle Point2D.

These must stay in sync; the math is small enough that a duplicated 5-line helper beats threading a shared utility across two modules with different return-tuple shapes.

---

## 4. URL parameter table

| Param | Default | Range | Behaviour on invalid |
|---|---|---|---|
| `shoulderExpand` | `0.40` | finite number in `[0, 1.5]` | silently falls back to default |
| `hipExpand`      | `0.25` | finite number in `[0, 1.5]` | silently falls back to default |

Parsed once per mount in `src/app/result/[id]/page.tsx` via `readExpandFactorsFromURL(searchParams)` and passed as props to `SwingPlayer` → `SkeletonOverlay` / `computeShoulderDisc` / `computeHipDisc`. No runtime re-read.

Example test URLs (on `b3fea3f0-e248-44d7-a923-0bb43172b5bf`):

```
?shoulderExpand=0.40&hipExpand=0.25   # default
?shoulderExpand=0.30&hipExpand=0.20   # smaller
?shoulderExpand=0.50&hipExpand=0.30   # larger
?shoulderExpand=2.0                   # out of range → falls back to 0.40
```

---

## 5. Acceptance criteria

Visual-only. Run on https://swingcue.ai/result/b3fea3f0-e248-44d7-a923-0bb43172b5bf after the Vercel deploy lands, with the 🦴 skeleton overlay toggled on at the setup frame.

- [ ] **Shoulder dots** visually land on the upper-arm-to-body junction (the visible shoulder ball area), NOT on the chest.
- [ ] **Shoulder→elbow skeleton line** passes through the upper arm visual midline (not through the chest, not along the outer skin edge).
- [ ] **Hip dots** land on the visible hip area at the side of the torso.
- [ ] **Hip→knee line** passes through the thigh midline.
- [ ] All four URL-parameter variations above change the visual in the expected direction.

If any criterion fails, ship a tuning commit on the same branch (adjust defaults) before merge. If multiple criteria fail in ways the math can't fix, escalate to a re-think.

---

## 6. Relationship to PR-5.8 main spec

PR-5.8A is a **precursor** to PR-5.8. It fixes only the visible anchor mismatch using a render-time-only transform on the existing v1 schema's `left_shoulder`/`right_shoulder`/`left_hip`/`right_hip` keypoints.

The full PR-5.8 design (`docs/decisions/PR-5.8_GOLF_17_KEYPOINTS.md`) remains documented as **future work**:

- Schema v2 with the SwingCue-17 anatomical naming (`shoulder_L`/`shoulder_R`, etc.)
- 5 new keypoints not in v1: `head_crown` (derived), `hand_L`, `hand_R`, `foot_L`, `foot_R`
- Python-side `head_crown` derivation in `python/pose_timeline.py`
- `version` bump `1 → 2`, `keypoint_source` re-label
- Frontend rename from `coco.ts` → `swingcue17.ts`

PR-5.8A's defaults (0.40 / 0.25) and the math itself can be re-evaluated when PR-5.8 ships — the v2 keypoints may sit at different anatomical positions and require different expansion factors, or none at all. The two systems will coexist via the `version` field on the timeline JSON: v1 reads → PR-5.8A expansion path; v2 reads → whatever PR-5.8 lands.

**Constraint honored:** this PR does NOT modify `docs/decisions/PR-5.8_GOLF_17_KEYPOINTS.md`. The full spec stays intact as the canonical record of the larger schema migration.

---

## 7. Files touched (frontend render only)

| File | Change | Commit |
|---|---|---|
| `src/lib/skeleton/coachingAnchors.ts` | NEW — helpers + URL parser + defaults | 1 |
| `src/app/result/[id]/page.tsx` | parse URL, pass props to SwingPlayer | 1 |
| `src/components/SwingPlayer.tsx` | accept props (interface only) | 1 |
| `src/components/SwingPlayer.tsx` | destructure with defaults, forward to SkeletonOverlay | 2 |
| `src/components/SkeletonOverlay.tsx` | apply expansion to shoulder/hip dots + edges; helper `expandPairOrNull` | 2 |
| `src/lib/disc/computeDiscParams.ts` | optional expansion param on both compute fns + private `applyExpand` | 3 |
| `src/components/SwingPlayer.tsx` | pass expansion factors to the compute calls; useEffect dep array | 3 |

NOT touched: `python/`, Supabase schema, `pose_timeline_2d` data, elbow/wrist/knee/ankle/head/hand/foot keypoints, `src/lib/disc/phaseCompression.ts`, `docs/decisions/PR-5.8_GOLF_17_KEYPOINTS.md`.

---

## 8. Local verification gap (build-time)

This branch was developed in an environment without `node`/`npm`/`tsc`/`gh` accessible. Vercel build acceptance is the only build-time gate; visual acceptance (§5) is the only product gate. If TS or lint fails on Vercel, ship a fixup commit on this branch.
