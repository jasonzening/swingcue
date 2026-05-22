# PR-7c Reframe — Option I: Offline-only motion correction

**Date**: 2026-05-21
**Status**: Active — supersedes the "wire after WHAM" plan in `PR-7c_READINESS_CHECKLIST.md` §4

---

## What changed

PR-7c_READINESS_CHECKLIST.md §4 assumed motion_correction would wire
into `python/main.py /analyze` immediately after the WHAM step. Audit
of the production analyze flow during PR-7c.1 kickoff revealed:

**WHAM is not in production.** Production `/analyze` (see
`python/main.py:109`) runs:
1. MediaPipe Pose (full-timeline 2D)
2. Phase detection
3. SAM3D pose3d **at 5 phase frames only** (sparse, not a timeline)
4. YOLO11-pose at the same 5 frames
5. PR-4 pose_timeline_2d smoothing (2D)

`motion_correction` was designed against WHAM's 17-joint H36M
full-timeline 3D output (see
`python/pilot/output/wham/<id>/joint_centers_3d.json` schema). Its
input contract does not match what production produces today.

## Decision

**Option I**: defer Python-side integration. Ship motion_correction as
an offline tool only.

- motion_correction module is merged to main (commit 5ac60da) but
  NOT imported by main.py.
- Pre-generate `CorrectedTimeline` JSONs for select videos manually
  using `python/motion_correction/scripts/render_overlay_compare.py`
  + the offline pipeline.
- Upload those JSONs to Supabase Storage at a known prefix (e.g.,
  `corrected_timelines/<video_id>.json`).
- PR-7c proper (renamed PR-7c-frontend) becomes frontend-only:
  fetch the corrected JSON if present, render coaching anchors;
  fall back to raw skeleton if absent.
- WHAM-in-production becomes a separate **PR-7d** ticket (not yet
  scoped). Until PR-7d ships, only videos with a manually-uploaded
  corrected JSON show coaching anchors.

## What ships in PR-7c-frontend (revised scope)

| File | Change |
|---|---|
| `src/components/SwingPlayer.tsx` | Conditionally fetch `corrected_timelines/<video_id>.json` from Supabase Storage. If present, render coaching anchor overlay. If absent, current behavior (raw skeleton). |
| `src/components/MagentaAnchorOverlay.tsx` (new) | 5 per-side anchors (filled magenta r=12) + 2 disc centers (hollow ring r=8 stroke=2). Per `coaching_anchors_2d` schema. |
| `src/lib/skeleton/coachingAnchors.ts` (new) | Frame-by-frame interpolation from sampled CorrectedTimeline frames to per-render-frame anchor positions. Null-aware (skip frame with no anchor). |
| `src/lib/skeleton/computeDiscParams.ts` | Optionally use `setup_baseline.base_shoulder_width` when corrected JSON is present; fall back to existing raw-keypoint computation otherwise. |
| `src/components/SkeletonOverlay.tsx` | Add `Settings.skeletonDebug` toggle for raw vs corrected QA. |

Phase-aware opacity ramp from `PR-7c_READINESS_CHECKLIST.md` §2 still applies.

## What's NOT in scope for PR-7c-frontend

- Production analyze API wiring (deferred to PR-7d)
- Modal-from-Railway invocation (deferred to PR-7d)
- WHAM in production (deferred to PR-7d)
- SMPL vertex sampling (deferred to PR-7a.5)
- New Supabase tables/columns — only Storage uploads at a known prefix

## How users get coaching anchors before PR-7d

1. Pick a video manually.
2. Run offline:
   ```
   .venv-pilot/Scripts/python.exe python/pilot/scripts/run_wham_one.py <video_id>
   .venv-pilot/Scripts/python.exe python/pilot/scripts/normalize_wham_chirality.py
   .venv-benchmark/Scripts/python.exe -c "
       from motion_correction.engine.orchestrator import correct_timeline
       from motion_correction.domains.golf.plugin import GolfCorrectionPlugin
       from pathlib import Path
       cor = correct_timeline(
           Path(f'python/pilot/output/wham/{vid}/joint_centers_3d.json'),
           GolfCorrectionPlugin(), view='face_on',
       )
       cor.save(Path(f'corrected_timelines/{vid}.json'))
   "
   ```
3. Upload to Supabase Storage:
   `supabase storage upload swing-videos/corrected_timelines/<video_id>.json local.json`
4. Frontend auto-detects + renders.

Cost: ~$0.01 Modal per video. Manual labor: ~5 min/video.

## When to revisit (= when to start PR-7d)

PR-7d scope: replace MediaPipe + SAM3D + YOLO with WHAM as the
primary pose backbone in production. Triggers:
- Frontend integration (PR-7c-frontend) ships and users want anchors
  on more videos than manual processing can keep up with
- WHAM-quality threshold proven on >= 50 manually-corrected videos
- Modal cost projection at production volume is acceptable
  (~$0.01/video × N videos/day)

Until any of those, Option I is the holding pattern.
