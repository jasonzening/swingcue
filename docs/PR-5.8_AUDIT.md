# PR-5.8 — Current-State Audit

**Date**: 2026-05-19
**Status**: Audit complete · Pending Jason review before any implementation
**Spec under audit**: [`docs/decisions/PR-5.8_GOLF_17_KEYPOINTS.md`](decisions/PR-5.8_GOLF_17_KEYPOINTS.md) §11

This document captures the read-only audit Jason requested before any code change. Nothing in the pipeline was modified. The DB-side `psql` check from §11 is deferred to Jason (Supabase SQL editor).

---

## 1. Pose pipeline 实际架构

**Verdict: MediaPipe primary + YOLO 5-phase anchor correction (hybrid).**

`grep mediapipe|MediaPipe|YOLO|yolo|Pose` in `python/pose_timeline.py` returns 35 hits across three concerns:

| Concern | Evidence (line · code) |
|---|---|
| MediaPipe is the per-frame extraction source | `42` — `# MediaPipe pose 33-point indices for the 17 COCO subset.` |
| | `62` — `# 1. Extractor — MediaPipe 33-point → COCO 17 subset (per frame)` |
| | `65` — `def extract_coco_subset_from_mediapipe(...)` |
| | `71` — *"Convert MediaPipe's 33-point pose_landmarks.landmark list into the COCO 17-keypoint subset"* |
| YOLO is 5-phase anchor only (not per-frame) | `12` — pipeline order: `apply_yolo_anchor_correction (when YOLO 5-phase data available)` |
| | `267` — `def apply_yolo_anchor_correction(timeline, yolo_keypoints_per_phase, phase_markers, …)` |
| | `274` — *"Use the 5-phase YOLO COCO-17 keypoints (from PR-3) as ground-truth anchors and linearly interpolate the MP→YOLO offset across the rest of the timeline"* |
| Graceful degradation when YOLO absent | `301-303` — `if not yolo_keypoints_per_phase: timeline["yolo_anchor_correction"] = {"applied": False}; logger.info("…no YOLO data — skipped"); return timeline` |

Pipeline order (from the module docstring at top of file): `build → outlier_reject → smooth_ema → gap_fill → apply_yolo_anchor_correction → validate`.

**Matches spec §11 expected finding:** *"Pipeline includes both `mediapipe` and `yolo` imports / function calls"* ✓

---

## 2. MediaPipe API 变体

**Verdict: LEGACY `mp.solutions.pose` API, not the newer Tasks API.**

`grep -nE "pose_landmarker|PoseLandmarker|mp\.solutions\.pose|mediapipe\.solutions"` across `python/` returns exactly one hit:

```
python\analyzer.py:198:    pose_config = mp.solutions.pose.Pose(
```

Surrounding context (`python/analyzer.py:198-205`):
```python
pose_config = mp.solutions.pose.Pose(
    static_image_mode=False,
    model_complexity=1,  # 0=lite, 1=full, 2=heavy
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
```

Notable:
- `model_complexity=1` → MediaPipe Pose **Full** model (33 landmarks). NOT Lite (model_complexity=0) and NOT Heavy (=2).
- `smooth_landmarks=True` → MediaPipe applies its built-in EMA-like smoothing at the source. **PR-5.8 schema-v2 work should be aware this is on** — applying additional EMA downstream (currently in `pose_timeline.smooth_ema`) is layered on top.
- No use of `mp.tasks.vision.PoseLandmarker` (the newer 2023+ Tasks API).
- No `pose_landmarker` task file / `.task` model bundle anywhere in the tree.

**Matches spec §11 expected finding:** *"MediaPipe model = either `mp.solutions.pose` (legacy) or `pose_landmarker` (new API)"* — confirmed legacy ✓

---

## 3. 当前 `kp_source` metadata 实际值

**Verdict: Field is named `keypoint_source` (NOT `kp_source`) and equals `"mediapipe_pose"`.**

Located at `python/pose_timeline.py:257` inside `build_timeline_from_raw_coco_frames` envelope:

```python
252:    return {
253:        "version": 1,
254:        "fps_sampled": int(round(sample_fps)),
255:        "video_width": video_width,
256:        "video_height": video_height,
257:        "keypoint_source": "mediapipe_pose",
258:        "yolo_anchor_correction": {"applied": False},
259:        "frames": raw_frames,
260:    }
```

**Field-name mismatch with PR-5.8 spec:**
- Spec §6 uses `"kp_source"` (short form).
- Code uses `"keypoint_source"` (long form).
- This is two distinct keys, not just a value change. **PR-5.8 Step 3 implementation must either (a) rename `keypoint_source` → `kp_source` AND change the value, or (b) keep `keypoint_source` and treat `kp_source` as a typo in the spec.** Jason should pick.

**Matches spec §11 expected finding:** *"`kp_source = 'mediapipe_pose'` (stale, per existing memory note) or `'mediapipe_yolo_hybrid_v1'`"* — found the former (`"mediapipe_pose"`), with the field-name caveat above ✓ (mostly)

---

## 4. 当前 `schema_version` 字段是否存在

**Verdict: Field exists but is named `version` (NOT `schema_version`). Current value = `1`.**

`grep -n '"version"|version:'` in `pose_timeline.py` returns one hit:

```
253:        "version": 1,
```

PR-4 design doc (line 124–126) confirms this is the intended v1 envelope shape:
```json
{
  "version": 1,
  "fps_sampled": 10,
  ...
}
```

**Field-name mismatch with PR-5.8 spec:**
- Spec §6 calls it `"schema_version"`.
- Code calls it `"version"`.
- Same situation as `kp_source` vs `keypoint_source` — must be reconciled in Step 3.

If the spec wins both naming choices, the v2 migration is **simultaneously** a schema-version bump (`1` → `2`) AND a field-rename pair (`version` → `schema_version`, `keypoint_source` → `kp_source`). Frontend coexistence logic in §6 says *"Frontend reads `schema_version`"* — it cannot, because today's v1 envelopes use `version`. So either the spec's coexistence rule needs to handle both names (try `schema_version` first, fall back to `version`), or v1 envelopes need a forward-rename pass at read time.

**Matches spec §11 expected finding:** *"Schema version = 1 currently, 17 COCO kp output"* ✓ (value matches; field name differs as noted)

---

## 5. 跟 Claude 记忆 (MediaPipe + YOLO 混合) 是否一致

**Verdict: Consistent on architecture, divergent on three small details.**

### Consistent
- ✅ MediaPipe is the per-frame source — confirmed (analyzer.py:198 + pose_timeline.py:65).
- ✅ YOLO is 5-phase anchor only — confirmed (pose_timeline.py:267).
- ✅ Hybrid pipeline graceful-degrades when YOLO data absent — confirmed (pose_timeline.py:301-304).
- ✅ Output is COCO 17 subset, NOT full MediaPipe 33 — confirmed; mapping table at PR-4 §B (MP `0,2,5,7,8,11-16,23-28` → COCO 17).
- ✅ Data-quality pipeline order matches PR-4 design (`build → outlier → ema → gap → yolo correct → validate`).

### Divergent
- ⚠ **Field name `kp_source` vs `keypoint_source`** — spec §6 short form, code long form. (See §3 above.)
- ⚠ **Field name `schema_version` vs `version`** — spec §6 long form, code short form. (See §4 above.)
- ⚠ **`fps_sampled` value** — spec §6 example shows `14`; PR-4 design says `10`; actual runtime value depends on the `sample_fps` parameter passed from `main.py` (currently 10.0 per PR-4 §B "Current state"). PR-5.8 spec's `14` may be an unintentional value bump, or may signal a sample-rate change is part of v2.

None of these are blocking for the audit, but they need a Jason decision before Step 3 begins.

---

## 6. 对 PR-5.8 spec §7 实施顺序的影响

**Verdict: LOW difficulty for Step 3 (Python schema upgrade). MediaPipe integration already exists — no new model wiring required.**

### Why low difficulty

The audit confirms the most expensive-to-add capability — **per-frame MediaPipe 33-landmark extraction** — is already running in `python/analyzer.py:198-205`. The only thing `pose_timeline.py` currently does on top is **compress 33 → 17 COCO**. PR-5.8 Step 3 needs to **switch the compression target** to the SwingCue-17 mapping (spec §3 table) and **derive `head_crown`** from existing face landmarks (spec §4 formula). No new MediaPipe wiring, no model swap, no Docker/dependency changes.

### Concrete Step 3 work breakdown

| # | Task | File | Effort |
|---|---|---|---|
| 1 | Replace MP→COCO mapping table with MP→SwingCue mapping | `python/pose_timeline.py` (around the `COCO_FROM_MP` indices near L42-62) | ~20 LoC |
| 2 | Implement `derive_head_crown(mp33)` from `mp[7]`, `mp[8]`, `mp[9]`, `mp[10]` | `python/pose_timeline.py` (new helper) | ~15 LoC |
| 3 | Add `hand_L`/`hand_R` (MP indices 19, 20) and `foot_L`/`foot_R` (31, 32) to extraction | `python/pose_timeline.py` (`extract_*` body) | ~10 LoC |
| 4 | Bump envelope `"version": 1` → `"version": 2` (or rename to `schema_version`) | `python/pose_timeline.py:253` | 1 LoC |
| 5 | Update `keypoint_source` → `"mediapipe_pose_33_v2"` (and possibly rename to `kp_source`) | `python/pose_timeline.py:257` | 1 LoC |
| 6 | Update YOLO anchor correction to map by SwingCue index instead of COCO (or keep COCO-indexed and translate at apply time) | `python/pose_timeline.py:267-405` | TBD per Jason decision below |
| 7 | Update validation `min_kp_per_valid_frame` threshold if needed (currently `>= 8` of 17 — same denominator, same threshold should hold) | `python/pose_timeline.py` (validate fn) | 0 LoC likely |

Total backend effort: **~50 LoC of mostly mechanical changes**, plus one design decision (item 6).

### Design decisions blocking Step 3

These need Jason's call before implementation begins. Audit findings make them explicit:

1. **Field names** — `kp_source` vs `keypoint_source`, `schema_version` vs `version`. Pick one set; consistency matters because frontend coexistence (§6 of spec) reads these fields.
2. **YOLO anchor correction during v1→v2 transition** — YOLO outputs COCO-17 keypoints (per `python/yolo/inference.py` and PR-3 spec). PR-5.8 v2 timeline uses SwingCue-17. Options:
   - **Option A**: Keep YOLO COCO output and apply correction only to the 13 SwingCue indices that have a 1:1 COCO equivalent (skip head_crown, hand_L/R, foot_L/R).
   - **Option B**: Re-export YOLO with a SwingCue-17 head, retrain or remap output. Out of scope for PR-5.8.
   - **Option C**: Disable YOLO correction for v2 timelines entirely (simplest; lose some accuracy on the 13 shared keypoints).
   - Spec §2 says *"YOLO is retained as an anchor-correction layer"* but doesn't address the COCO/SwingCue index mismatch. Most likely intent = Option A.
3. **`fps_sampled` target** — spec example shows `14`, current pipeline uses `10`. Confirm whether bumping the sample rate is intended.

### What is NOT in Step 3 scope (preserves spec §9 phasing)

- Joint-center offsets — deferred to Step 2 visual verification per spec §8.
- Club shaft detection, ball position, wrist/spine/X-factor metrics — Phase 2, separate PRs per spec §9.
- Frontend mapping migration — Step 4, not blocked by Step 3 but separate file work (`src/lib/skeleton/coco.ts` → `swingcue17.ts`).

### Conclusion

**Step 3 is "output layer refactor" not "MediaPipe integration".** The infrastructure exists; only the projection of 33 → 17 needs to change. Risk surface is small (single Python module, additive frontend types). Main pre-work is reconciling the field-name mismatches between spec and code (§§3-4 above) before writing any code.
