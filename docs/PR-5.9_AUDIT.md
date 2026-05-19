# PR-5.9 — Pose Accuracy Upgrade Pre-implementation Audit

**Date**: 2026-05-19
**Status**: Audit complete, no code changes
**Spec under audit**: TBD (PR-5.9 spec doc not yet written)

Read-only audit of the current pose pipeline. The DB-side `psql` step (verifying actual `fps_sampled` values in production rows) is deferred to Jason. Findings below are sourced from the current `main` branch (HEAD `0d53f1c`); PR-5.8A's render-only changes live on a feature branch and do not affect any of the audit targets in §1–§7.

---

## 1. Sampling rate (`fps_sampled`)

**Where it's set.** Two places agree on the default and one place computes the actual sampling interval:

| Source | File:line | Value |
|---|---|---|
| Request body default | `python/main.py:85` | `sample_fps: float = 4.0` |
| Function param default | `python/analyzer.py:170` | `def analyze_video(..., sample_fps: float = 4.0)` |
| Per-frame stride computation | `python/analyzer.py:195–196` | `video_fps = metadata.fps or 30.0; frame_interval = max(1, int(video_fps / sample_fps))` |
| Envelope persistence | `python/pose_timeline.py:254` | `"fps_sampled": int(round(sample_fps))` |
| End-to-end orchestration | `python/main.py:122–124` | `analyze_video(tmp_path, sample_fps=req.sample_fps)` |

So a request that does NOT include `sample_fps` produces a timeline at **4 fps**. Higher rates require the caller to pass the field explicitly. The API route at `src/app/api/analyze/[id]/route.ts` does not set it (verified via Grep), so production currently runs at 4 fps unless something else upstream overrides.

⚠ **Discrepancy worth flagging.** `src/lib/disc/frameAt.ts:6–7` comment says *"PR-4 backend currently writes `fps_sampled = 10` while the actual sample rate is ~14 fps"*. The current code defaults to 4.0 and writes whatever `sample_fps` it received. Either: (a) the comment is stale (likely — the file was written during the PR-5 hotfix when the default may have been 10), or (b) there is a non-default request path I missed. **Jason's psql check on real rows is the ground truth here.**

**Native fps detection.** Already present at `python/analyzer.py:114–126` (`get_video_metadata`) via `cv2.CAP_PROP_FPS`. Stored in `VideoMetadata.fps`, returned to JS at `python/main.py:271–275`, persisted to `swing_videos.video_metadata_json` by the route at `src/app/api/analyze/[id]/route.ts:267`. So the **source-of-truth native fps already exists in the pipeline**; the analyzer just doesn't currently consume it for sampling.

**Cost of bumping `sample_fps` to native.** Pure sampling-loop change:

- Python side: line 196's `frame_interval` becomes `1` when `sample_fps == video_fps`, so the loop processes every frame.
- Wall time: roughly linear in frame count — 4×–7× slower for 24/30 fps native.
- JSON size: ~7 KB at 10 fps × 3 s × 17 kp grows to ~50 KB at 60 fps × 3 s × 17 kp. Still trivial for JSONB.
- **No downstream frame-count assumptions** worth changing. Frontend lookup uses `frameAt` (ts-based binary search) and `nearestFrame` (linear scan), both fps-agnostic. The only count gate is `validate_timeline` (`pose_timeline.py:415–449`) which checks **ratios** not absolute counts. Outlier rejection uses pixel jumps not frame counts. EMA, gap-fill, YOLO correction all iterate over `frames` without assuming a rate.

---

## 2. Smoothing implementation

**Three smoothing/filter sites total.** None preserves the raw value before overwriting.

| Site | File:line | Algorithm | Causal? | Raw preserved? |
|---|---|---|---|---|
| MediaPipe internal | `python/analyzer.py:201` | `mp.solutions.pose.Pose(..., smooth_landmarks=True, ...)` — MediaPipe's built-in temporal smoothing (LSTM-based, undocumented internals) | Causal | **NO** — applied before our code ever sees the landmark; raw never exposed |
| Backend EMA | `python/pose_timeline.py:147–178` | `smooth_ema(timeline, alpha=0.4)` — per-keypoint trajectory EMA. Null frames reset state so blending never spans a gap. **Mutates `kp[0]` and `kp[1]` in place** at lines 174–175 | Causal | **NO** — overwrites in place; previous value only lives in the local `prev` tuple for one step |
| Orphan utility | `python/analyzer.py:162–167` | `moving_average(arr, window=3)` via `np.convolve` mode=`same` | Non-causal (centered window) | N/A — defined but not called anywhere in the current pipeline |

Frontend has **no value smoothing** for the pose timeline:

- `src/lib/disc/unwrap.ts` — angle unwrap only (continuity fix; not a value filter). Causal, takes `prev/prevT` from caller's `useRef`.
- `src/lib/disc/phaseCompression.ts` — smoothstep curve over phase markers for the disc's visual width signal. **Not a smoother of pose data**; just an interpolation curve on a 5-anchor compression schedule.
- `src/lib/sam3d/poseFetch.ts` — grep for `smooth|EMA|alpha` returned no matches. Pure fetch + cast.

**Where raw COULD be preserved** if we wanted to retain it: between line 244 (`raw_coco_frames.append(...)` in `analyzer.py`) and the first mutating pipeline call in `main.py` (around line 225, the `detect_outliers_and_reject(tl)` call). At that moment `raw_coco_frames` is the untouched MediaPipe-internal-smoothed extract; persisting a deep copy of it alongside the final smoothed `tl` would give us the dual view PR-5.9 task 4 wants. **Note:** MediaPipe's internal `smooth_landmarks=True` is still applied — to get truly raw data we would also need to flip that to `False` (see PR-5.9 task 2 below).

---

## 3. YOLO anchor correction timing

**Per-phase only (5 anchors), NOT per-frame.**

Implementation at `python/pose_timeline.py:267–408`. Algorithm verbatim from the docstring at lines 290–300:

```
1. For each phase in ['setup','top','transition','impact','finish']:
     a. Find MediaPipe frame closest to phase_markers[f"{phase}Time"].
     b. For each of the 17 keypoints, compute offset = yolo_kp - mp_kp.
     c. If either side has conf<min_conf, mark that (phase, kp) None.
2. For each timeline frame:
     a. If ts < first anchor: apply first anchor's offsets.
     b. If ts > last anchor: apply last anchor's offsets.
     c. Otherwise: linear lerp between bracketing anchors per kp.
     d. None anchors are skipped (use neighbouring anchor's offset).
```

Concretely:
- **Per-(phase, keypoint) OFFSET** is computed at the 5 anchor frames (`lines 317–351`). YOLO does NOT replace MediaPipe; it provides a per-keypoint delta vector.
- **Per-frame application** uses linear lerp between bracketing anchors (`lines 358–397`), with the corrected coordinate written as `kp[0] + offset[0]` (`line 396`).
- Confidence gate: `min_conf=0.3` (default at line 271); pairs where either side fails the gate become `None` and are skipped, with neighbours filling in.

Phase markers come from `phase_detector.detect_phases(...)` called in `python/main.py:127`. They are then passed into the correction via the `phase_markers` kwarg at the call site in `main.py` (visible in the broader try/except block around line 219–245).

YOLO 5-phase keypoint source: `yolo_keypoints_per_phase` parameter (line 269), collected in `main.py` from `yolo_summary.results` (the PR-3 ONNX-decoded YOLO output).

**Key implication for PR-5.9.** If we change `sample_fps` (task 1) we are NOT changing how often YOLO runs — YOLO still anchors at 5 phase frames regardless. Only the MediaPipe sampling density between anchors changes.

---

## 4. Schema preservation capacity

**Postgres JSONB.** Practical per-value limit is ~1 GB. Current per-video JSON is on the order of 7–50 KB depending on `sample_fps`. **No size constraint relevant to PR-5.9.**

**Consumers of `pose_timeline_2d` / `PoseTimeline` (Grep across `src/`):**

| File:line | Role |
|---|---|
| `src/types/analysis.ts:272–284` | `PoseTimeline` interface definition. `version: 1` is a literal type — adding v2 requires widening. |
| `src/app/api/analyze/[id]/route.ts:73, 139, 195, 219, 284` | Persists payload as `unknown \| null` and writes to the column. No shape enforcement. |
| `src/app/result/[id]/page.tsx:63–66` | Reads `pose_timeline_2d` via inline cast, validates `Array.isArray(pt.frames) && pt.frames.length > 0` only. Tolerant of unknown extra keys. |
| `src/components/SwingPlayer.tsx:87, 287–289, 405–438` | Reads via `frameAt(t, poseTimeline)` then `frame.keypoints.left_shoulder` etc. Field-by-field access — adding sibling fields to `frame` is safe. |
| `src/components/SkeletonOverlay.tsx:34, 124, 162–176` | Reads `timeline.video_width`, `timeline.video_height`, iterates `timeline.frames`, indexes `frame.keypoints[name]`. Field-by-field again. |
| `src/lib/disc/frameAt.ts:24–41` | Reads `pose.frames`, iterates `ts`. Schema-agnostic beyond `{frames: [{ts: number}]}`. |
| `src/lib/disc/types.ts` | Type re-exports; no runtime reads. |

**Backend validation.** `validate_timeline` (`pose_timeline.py:415–449`) checks frame counts and valid-kp counts only — does NOT validate field names or shape. JSONB writes have no schema constraint.

**Safest design to add `raw_keypoints` alongside `keypoints` per frame:**

```json
"frames": [
  {
    "ts": 0.0,
    "frame_idx": 0,
    "interpolated": false,
    "keypoints": { ... existing final values ... },
    "raw_keypoints": { ... untouched MediaPipe extract ... }
  }
]
```

- All existing readers continue to work (they access `frame.keypoints[name]`).
- Optional on PoseFrame TS type so legacy v1 readers don't break.
- JSONB size roughly doubles per frame — still in single-digit-MB territory at native fps.
- Alternative: top-level `raw_frames: PoseFrame[]` mirror. Same cost. The per-frame sibling approach is preferred because keys stay co-located for debugging.

---

## 5. Face landmarks dropped from output

**Mouth indices are explicitly excluded, not just absent.**

`python/pose_timeline.py:44–54` enumerates exactly 17 MediaPipe-33 indices in `MEDIAPIPE_TO_COCO_IDX`:

```python
MEDIAPIPE_TO_COCO_IDX: dict[str, int] = {
    "nose": 0,
    "left_eye": 2,  "right_eye": 5,
    "left_ear": 7,  "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12,
    ...
    "left_ankle":    27, "right_ankle":    28,
}
```

MediaPipe Pose 33 indices **9 (`mouth_left`)** and **10 (`mouth_right`)** are absent. Same omission in `COCO_NAMES` tuple at lines 30–40.

The extractor loop (`pose_timeline.py:80–91`) is fully generic — it iterates `COCO_NAMES` and looks up each name in `MEDIAPIPE_TO_COCO_IDX`. **Adding mouth would require only:**

1. Two new entries in `MEDIAPIPE_TO_COCO_IDX`: `"mouth_left": 9, "mouth_right": 10`
2. Two new entries in the `COCO_NAMES` tuple (placement matters since downstream order-sensitive code — like YOLO anchor application at line 336 — uses `enumerate(COCO_NAMES)`, but YOLO outputs 17 indices in COCO standard order, so appending mouth at the end would NOT introduce ordering mismatch with YOLO's 17-index output; YOLO would just have no entry to compare against for mouth, and the existing `len(yolo_kps) < 17` check at line 325 already covers this case)

Frontend cost:

3. Add two variants to `CocoKeypointName` union at `src/types/analysis.ts:246–255`.
4. Add two names to `COCO_KEYPOINT_NAMES` at `src/lib/skeleton/coco.ts:16–26`.
5. Optionally add edges for visualization at `coco.ts:38–52` (not strictly required — dots-only is fine).

YOLO anchor correction at `pose_timeline.py:336–347` would silently skip the mouth pair (`yolo_kp[2] < min_conf` would be `False` only if YOLO's 18th+ index has matching data — it doesn't, so mouth would always be `None` for anchor and fall back to raw MediaPipe). **This is the desirable behaviour.**

Total smallest fix: ~6 LoC across 2 files (Python). +4 LoC frontend if we want to render them.

---

## 6. Frontend pose rendering — interpolation vs nearest

**No interpolation anywhere on the frontend.** All lookups are nearest-frame, time-anchored.

| Caller | Lookup | Implementation |
|---|---|---|
| `src/components/SkeletonOverlay.tsx:58` | `nearestFrame(timeline.frames, t)` | `SkeletonOverlay.tsx:162–176` — linear scan over frames, picks min `|t - f.ts|`. Returns null only on empty timeline. |
| `src/components/SwingPlayer.tsx:289` | `frameAt(t, poseTimeline)` | `src/lib/disc/frameAt.ts:24–41` — binary search, picks nearer of the bracket pair. |

`src/lib/disc/computeDiscParams.ts` (`computeShoulderDisc`, `computeHipDisc`) is **pure over the PoseFrame the caller picks**. No internal time arithmetic, no smoothing, no interpolation. Its outputs (`cx`, `cy`, `rx`, `ry`, `angleRad`) come from whichever frame the lookup returned.

The disc pipeline then layers visual smoothing on top:
- `unwrapAngle` (`src/lib/disc/unwrap.ts:20–32`) — angle continuity (math, not value smoothing). Causal.
- `getPhaseCompression` (`src/lib/disc/phaseCompression.ts:85–112`) — smoothstep curve across the 5 phase markers, drives disc visual width. Not a pose smoother.
- `computeMicroCorrection` (`src/lib/disc/phaseCompression.ts:136–144`) — ±10% nudge based on `currentDist / baselineDist`. Not a pose smoother.

**Implication for PR-5.9 task 3 (frontend interpolation).** Both `nearestFrame` and `frameAt` would each grow a sibling `*Interpolated` variant that returns a synthesised PoseFrame whose coords are linearly lerped between the two bracketing frames. Null coord on either side ⇒ skip lerp for that kp (cascade to nearest non-null). Estimated ~40–80 LoC for both helpers plus minor caller-site opt-in.

---

## 7. Video native fps availability

**Available end-to-end, but the frontend currently hardcodes 30.**

| Stage | File:line | What's stored |
|---|---|---|
| OpenCV detection | `python/analyzer.py:116` | `fps = cap.get(cv2.CAP_PROP_FPS) or 30.0` |
| Python dataclass | `python/analyzer.py:86–91` | `VideoMetadata(durationSec, fps, width, height)` |
| API response | `python/main.py:271–275` | `"videoMetadata": {durationSec, fps, width, height}` |
| JS persistence | `src/app/api/analyze/[id]/route.ts:267` | `video_metadata_json: { ...videoMetadata!, dataSource }` |
| Frontend read | `src/app/result/[id]/page.tsx:80–94` | Reads `vmJson?.durationSec ?? 3` and `vmJson?.dataSource ?? 'stub'` from `video_metadata_json` |
| Frontend use | `src/app/result/[id]/page.tsx:94` | `const vm: VideoMetadata = { durationSec: dur, fps: 30, width: 640, height: 360 };` — **fps hardcoded to 30**, width/height also hardcoded |

The video native fps is **already in the DB column** (`swing_videos.video_metadata_json.fps`). The frontend just doesn't read it — line 94 builds `vm` with hardcoded 30 instead of `vmJson?.fps ?? 30`. **Trivial fix**: change one line.

The `meta.fps` value is then passed to `SwingPlayer` indirectly via the `duration` prop only (`page.tsx:191` `duration={meta.durationSec}`). SwingPlayer doesn't currently receive fps at all. If PR-5.9 task 3 (interpolation) needs the native fps for any reason (e.g., adaptive interpolation density), it would be the second hardcoded-vs-real change to make.

---

## 8. PR-5.9 design implications

Difficulty estimates for the 8 tasks in Jason's PR-5.9 scope. References point to the exact lines that would change.

### Task 1 — `sample_fps` upgrade (4 → native 24/30/60)
**LOW.** One-or-two-line change in `python/main.py:85` or `python/analyzer.py:170` (pass the detected `metadata.fps` instead of the hardcoded 4.0). Already-detected native fps at `analyzer.py:116`. Cost: 4×–7× Python wall time, ~10× JSONB size. No downstream code assumes a specific fps (§1 above). Validate that `validate_timeline` (`pose_timeline.py:415–449`) still passes — its ratio gate is fps-independent so it should.

### Task 2 — EMA removal
**LOW.** Delete (or `if False:` guard) the `smooth_ema(tl)` call in `python/main.py` (visible in the try block around lines 219–245). To also remove MediaPipe's internal smoothing, set `smooth_landmarks=False` at `python/analyzer.py:201`. Two-line change across two files. The orphan `moving_average` utility at `analyzer.py:162–167` can also be deleted if we want to clean up.

### Task 3 — Frontend interpolation
**MEDIUM.** Two new helpers: `frameAtInterpolated(t, pose)` next to `frameAt` (`src/lib/disc/frameAt.ts`), and `nearestFrameInterpolated(frames, t)` next to the existing `nearestFrame` (`src/components/SkeletonOverlay.tsx:162–176`). Both return a synthesised `PoseFrame` whose per-keypoint `[x, y, c]` tuples are linearly lerped between the bracketing frames. Null coord on either side ⇒ skip lerp for that specific keypoint and return the non-null side's value (or null if both). Caller-site opt-in via a feature flag or URL param to avoid breaking the existing render path. Estimated ~80–120 LoC total + light test coverage on null handling.

### Task 4 — Raw / final dual storage
**MEDIUM.** Schema-additive: per-frame `raw_keypoints` sibling field (§4 above). Backend changes: in `python/main.py`, deep-copy `raw_coco_frames` into a `raw_by_idx` dict before mutating, then after pipeline runs, walk `tl.frames` and assign `frame["raw_keypoints"] = raw_by_idx[frame["frame_idx"]]["keypoints"]`. Frontend type: extend `PoseFrame` at `src/types/analysis.ts:265–270` with `raw_keypoints?: Record<CocoKeypointName, Keypoint>`. No existing consumer breaks because the field is optional and `frame.keypoints` is untouched. JSONB doubles in size — still fine. Estimated ~30 LoC backend + 5 LoC type.

### Task 5 — Debug overlay (raw vs final visible)
**MEDIUM.** SkeletonOverlay gains a `mode: 'final' | 'raw' | 'both'` prop (or URL param). In `'both'` mode it renders two sets of dots — final in white (current), raw in red. Requires task 4 to have landed so `raw_keypoints` is on the frame. Without task 4, we can only show one set since smoothing is destructive. The render math is identical to the existing draw loop in `SkeletonOverlay.tsx:56–93` — just doubled. Add a URL param at `src/app/result/[id]/page.tsx:25` similar to PR-5.8A's `useSearchParams` pattern. Estimated ~60 LoC.

### Task 6 — Confidence preservation
**LOW.** Already preserved end-to-end. Audited: `analyzer.py:143` preserves `lm.visibility` into `Point2D.confidence`. `pose_timeline.py:83` reads `lm.visibility` and stores it as the third tuple element. EMA at `pose_timeline.py:147–178` does NOT touch `kp[2]` (only `kp[0]` and `kp[1]`). Gap fill writes `0.5` to flag synthesised frames (`pose_timeline.py:226`) — that's intentional and arguably already correct. Outlier rejection writes `0.0` (`pose_timeline.py:131`) — also intentional. **No change needed**; if Jason wants the raw `visibility` preserved separately from any final-confidence overwrite, that becomes a sub-task of task 4 (store original confidence in `raw_keypoints`).

### Task 7 — `head_crown` derivation
**LOW.** Per the PR-5.8 v2 spec, this is Python-side. Implementation needs mouth landmarks (MP indices 9, 10) — currently dropped (§5). Smallest path: add mouth to `MEDIAPIPE_TO_COCO_IDX` (2 lines), add `derive_head_crown(mp_landmarks)` helper computing `(ear_mid + (ear_mid − mouth_mid) × 0.45)` per the PR-5.8 §4 formula, call it inside `extract_coco_subset_from_mediapipe` (`pose_timeline.py:65–92`) and append `out["head_crown"] = [x, y, conf]`. Add `head_crown` to `COCO_NAMES` tuple. Frontend `CocoKeypointName` union gains one variant. Estimated ~30 LoC backend, ~3 LoC frontend type.

### Task 8 — Do-not-touch-disc
**N/A (constraint, not work).** As long as `frame.keypoints.left_shoulder` / `right_shoulder` / `left_hip` / `right_hip` continue to exist and return `[x, y, conf]` tuples, the disc compute path (`src/lib/disc/computeDiscParams.ts:120–198`) is untouched by any of tasks 1–7. Tasks 1, 2, 4, 5, 6, 7 affect data shape OR data semantics — only task 3 (interpolation) could shift what the disc reads if it changes the `frameAt` contract. Task 3 should add a NEW function rather than modify `frameAt` in place to honor this constraint cleanly.

---

## Summary

| Task | Difficulty | Files |
|---|---|---|
| 1 — fps upgrade | LOW | `python/main.py:85`, `python/analyzer.py:170` |
| 2 — EMA removal | LOW | `python/main.py` smooth_ema call, `python/analyzer.py:201` |
| 3 — Frontend interpolation | MEDIUM | `src/lib/disc/frameAt.ts`, `src/components/SkeletonOverlay.tsx` |
| 4 — Raw/final dual storage | MEDIUM | `python/main.py`, `src/types/analysis.ts:265–270` |
| 5 — Debug overlay | MEDIUM | `src/components/SkeletonOverlay.tsx`, `src/app/result/[id]/page.tsx` |
| 6 — Confidence preservation | LOW (no change) | already preserved end-to-end |
| 7 — head_crown derivation | LOW | `python/pose_timeline.py:44–92`, `src/types/analysis.ts:246–255` |
| 8 — Do-not-touch-disc | N/A | constraint, satisfied if tasks 3+4 add rather than replace |

**Two open questions before writing the spec:**

1. **§1 fps mystery:** the current code default is `sample_fps=4.0` but `frameAt.ts` was written with a `10`/`14` assumption. Jason's psql query on real `pose_timeline_2d.fps_sampled` values will clarify whether something at the call layer is bumping it past 4.0 or the frontend comment is stale.
2. **§7 frontend fps hardcoded to 30:** trivial bug — `page.tsx:94` ignores the real fps that's already in `video_metadata_json`. Not a PR-5.9 task per se, but worth a 1-line fix in whatever PR-5.9 commit touches that file.
