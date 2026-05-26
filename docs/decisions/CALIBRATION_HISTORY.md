# Calibration history — WHAM-derived 3D pose pipeline

Single-page log of each step in the SwingCue 3D-pose pipeline since
the PR-7 family kicked off. Useful for new readers who need the
narrative ("why are there 4 versions of the focal length?") without
trawling commit messages.

## PR-7 family — research / probes

| PR | Status | Outcome |
|---|---|---|
| PR-7a   | shipped (research baseline) | First WHAM-on-Modal inference. Per-phase fitted offsets (face_on view). Visible overshoot vs MediaPipe at top phase. |
| PR-7a.2 | shipped | Chirality swap (upper-body arm chain L↔R) so WHAM joint labels follow image-orientation convention. |
| PR-7a.5 | probe — FAIL | SMPL vertex sampling vs PR-7a fitted offset. PR-7a fitted offsets won 11/15 on the b32e0f21 ground-truth set. SMPL-X vertex sampling abandoned. |
| PR-7a4  | probe — landmark indices | Derived 6890→landmark mapping via argmax-y over canonical T-pose. `head_crown = vertex 411` (used by PR-8b.1). Other landmarks (throat/c7/acromion/trochanter) saved for future PRs. |
| PR-7c-frontend (v1..v9.3) | shipped | Magenta anchor overlay using MediaPipe `pose_timeline_2d`. Includes per-video keyframe interpolation (`VIDEO_KEYFRAMES`) and in-browser tuning panel (`?tune=anchors`). Frontend layer NOT replaced by WHAM — coexists. |
| PR-7d   | research spec | NLF pilot, deferred indefinitely. |

## PR-8 family — production WHAM pipeline

| PR | Status | Outcome |
|---|---|---|
| PR-8a' | shipped (Supabase) | DDL for `wham_video_meta` (1 row per video) + `wham_pose_timeline` (1 row per frame). Includes Jason's 6 corrections from spec v1. |
| PR-8b   | shipped (Modal) | Promoted `run_wham` local entrypoint to deployed `infer_video` Modal callable. Emits PR-8a' schema with explicit source / joint_type / coordinate_space discriminators. |
| PR-8b.1 | shipped (Modal) | `head_crown` field via SMPL mesh vertex 411 (PR-7a4 probe). Additive — `head` (H36M idx 10, face/nose region) preserved for back-compat. |
| PR-8b.2 | shipped (Modal) | CLIFF focal calibration. Was using `focal = max(w, h)` as fallback; WHAM upstream's `lib/vis/run_vis.py` line 22 uses `focal = sqrt(w² + h²)`. For 720×1280 the difference is 1280 → 1468.6 (~13%), enough to cause ~10–20px lateral drift across the swing. Same bug existed in PR-7a `corrected.json` (camera_intrinsics fx=fy=1280) — 1-year-old inherited bug. |
| PR-8b.3 | shipped (Modal) | Median-z stabilization. WHAM's `trans[:, 2]` (camera-frame depth) varies by 0.81m across a fixed-tripod 4s swing where actual depth change is ~10cm. The remaining ~70cm is monocular-depth-ambiguity bias polluting per-frame projection scale. Fix: replace `trans_z` with `median(trans_z)`, recompute verts with stabilized trans → joints + head_crown both stop "breathing in and out". |
| PR-8b.4 | candidate (deferred) | If PR-8b.3 median-z proves insufficient on handheld videos (where the camera DOES move), parse `slam_results.pth` from WHAM's output folder and use `trans_world` + SLAM camera pose. Not in scope unless a handheld test surfaces it. |
| PR-8c | shipped | Railway → Modal → Supabase write-side integration. After MediaPipe completes on Railway's `/analyze`, a FastAPI `BackgroundTask` synchronously waits on Modal `infer_video` (~55s) and persists results to `wham_video_meta` + `wham_pose_timeline` using service-role. Sets `swing_analysis.video_metadata_json.wham_status='ready'\|'failed'`. MediaPipe response is NOT gated on WHAM — user sees results immediately, WHAM populates in background. Idempotency on `video_id` unique constraint. |
| PR-8c.1 | shipped | UX state machine: synchronously writes `wham_status='processing'` + `wham_started_at` + `wham_expected_completion_seconds` to `swing_analysis.video_metadata_json` before scheduling the background task, so frontend polls never see "WHAM not available" while inference is actively running. Dynamic ETA: `max(60, int(durationSec * 13))` (R1). Non-blocking on failure: synchronous write is best-effort, BackgroundTask retries with 5s backoff (R2). Failure paths converge on `wham_status='failed'` — frontend never stuck at 'processing' (R3). |
| **PR-8b pkl-loader fix** | **shipped (mid-PR-8c.1 hotfix)** | Discovered during PR-8c.1 dual-gate E2E: `_run_wham_pipeline`'s pkl fallback walked the whole `/opt/wham/output` tree and took `found_pkls[0]`. In warm Modal containers, second-and-later videos got the first video's pkl — byte-identical contamination silently. Fix: scope `os.walk` to per-video `out_dir`. Affected PR-8b and PR-8b.3 too but only surfaced when 2+ videos ran back-to-back. |

### PR-8c verification (2026-05-26)

E2E gates verified by `docs/PR-8c_VISUAL_VERIFY/`:

| Gate | Video | Result |
|---|---|---|
| **GATE 1 — DTL** | `956afa87` | ✓ `wham_video_meta.status='completed'`, 119 timeline rows, head_crown=411, _trans_z_median=4.044, wham_status='ready'. Visual MP4 clean. |
| **GATE 2 — FACE-ON** | `5bbcfbc8` | ✓ Same, 201 rows, _trans_z_median=3.979 (distinct from GATE 1 — pkl-loader fix proof). PR-7a.2 chirality swap holds: shoulders/hips NOT reversed in face-on render. |
| **R3 — forced failure** | `88ef44a5` (fresh face-on, bad-JWT URL) | ✓ All 4 spec acceptance gates: `wham_video_meta.status='failed'` + `error_message='HTTP Error 400: Bad Request'` + `wham_pose_timeline count=0` + `swing_analysis.wham_status='failed'` (cleared from 'processing') + `swing_analysis.keypoint_timeline_json` preserved (MediaPipe untouched). Failure detected in 8.5s. |
| PR-8d | next | Frontend `JointSource` abstraction. WHAM output becomes selectable / optional source for `SwingPlayer`. Includes Zod schema for `wham_video_meta` and `wham_pose_timeline` jsonb shapes. |
| PR-8e | next-next | Backfill historical videos. Not blocked by PR-8d. |

## Env vars touched by PR-8c

| Var | Set | Notes |
|---|---|---|
| `SUPABASE_URL` | already | shared with sam3d/yolo Python writers |
| `SUPABASE_SERVICE_ROLE_KEY` | already | same. Spec called for `SUPABASE_SERVICE_KEY` but using the existing var to avoid duplicate config. **Rotation required before production launch — separate ticket.** |
| `MODAL_TOKEN_ID` | **NEW** | `modal token new` output. Put in Railway env. |
| `MODAL_TOKEN_SECRET` | **NEW** | same. |

## Cost note

Modal A10G $1.20/hr × ~55s/video = ~$0.018/warm-start video.
Expected daily Modal spend ≈ $0.02 × N daily uploads. Within ops budget.

If a video fails Modal inference (transient or otherwise), the
BackgroundTask still writes a `wham_video_meta` row with
`status='failed'` and the error message — no orphan state.
