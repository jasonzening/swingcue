# PR-7 Ground Truth Labels

Hand-marked anatomical keypoint positions on real swing video frames.
These are the **red-dot truth** that PR-7b's empirical offset sweep
calibrates the WHAM bone-center output against. Without these labels
the per-joint offset config in `python/pilot/correction/config.py` has
nothing to optimize for and PR-7b cannot complete.

Source spec: `docs/files/PR-7_GOLF_CORRECTION_LAYER_SPEC_v2.md` §9.

## 1. Purpose

WHAM gives us SMPL bone-center 3D joints. SwingCue needs them aligned
to specific anatomical anchor points that golf coaches reason about
(humeral head, femoral head, talus, etc.). The SMPL-to-coach-anatomy
delta is the **correction layer** — PR-7b learns it from these
hand-marked labels via a least-squares offset sweep.

PR-7c then locks the resulting per-joint offsets into the production
config, and SwingCue's analyzer uses them at inference time to
post-process every WHAM run.

## 2. Body-frame convention

**Always golfer's anatomical L/R, never screen L/R.**

- `left_shoulder` = the shoulder of the golfer's LEFT arm (lead arm
  for a right-handed swing)
- `right_shoulder` = the shoulder of the golfer's RIGHT arm (trail arm)
- Same rule for hip / knee / ankle pairs

For a right-handed golfer recorded from a down-the-line (behind-the-
player) camera, the golfer's left arm appears on the **right side** of
the image. Easy to mis-click; the labeler tool's legend lists each
expected target as you go.

Per spec §9: if this convention proves confusing in practice, the
fallback is `screen_left_*` / `screen_right_*`. Decide before starting
a labeling session; don't mix conventions across files.

## 3. File naming

`<short_video_id>_<phase>.json`

- `short_video_id` = first 8 chars of the video UUID (matches the
  benchmark + pilot output naming convention used elsewhere)
- `phase` = one of `setup` / `top` / `transition` / `impact` / `finish`

Examples:
```
b3fea3f0_setup.json
b3fea3f0_top.json
b3fea3f0_transition.json
b3fea3f0_impact.json
b3fea3f0_finish.json
a735cc7d_setup.json
... (15 total for first pass)
```

## 4. Required samples (first pass)

**Minimum 15 = 3 videos × 5 phases.** Per spec §3.B + ChatGPT-review §B.

Target videos:
- `b3fea3f0-e248-44d7-a923-0bb43172b5bf` (down-the-line, available locally)
- `a735cc7d-...` (UUID to fill — see `python/benchmark/download_videos.py`)
- `5bbcfbc8-...` (UUID to fill)

Below 15 the offset-sweep optimizer is under-constrained and the
PR-7b "lock offsets" step doesn't pass acceptance.

## 5. Phase definitions

| Phase | Identify by |
|---|---|
| `setup` | First frame where player is settled, club grounded behind ball |
| `top` | Club shaft at the highest point of the backswing |
| `transition` | Club shaft approaching parallel to ground on the downswing |
| `impact` | Moment of ball contact (or closest sampled frame; ±1-2 frames OK) |
| `finish` | Club shaft past the lead shoulder, body fully rotated to target |

Production `python/phase_detector.py` already outputs these timestamps
from real analyses. `extract_phase_frames.py` (see §8 below) uses
fractional heuristics — `setup=5%`, `top=40%`, `transition=50%`,
`impact=65%`, `finish=90%` of the video duration — which are good
first-cut suggestions; nudge `--frame` to refine if a suggested frame
doesn't match the desired moment.

## 6. Required labels (first pass — 5 keypoints)

Per spec §9:

1. `left_shoulder` — humeral head position on the LEFT arm (golfer's anatomical)
2. `right_shoulder` — humeral head on the RIGHT arm
3. `left_hip` — femoral head position on the LEFT side
4. `right_hip` — femoral head on the RIGHT side
5. `neck_center` — base of neck where the trapezius dips between the collarbones
   (acceptable fallback: chest_center / sternum if neck is obscured)

These 5 are sufficient for PR-7b's first offset sweep on the torso.
Arms/legs/wrists deferred — limb anchors have higher per-frame motion
and are diagnostic-only at this stage (per spec §5 acceptance gates).

## 7. Optional pass 2 (after first sweep complete)

If PR-7b's torso-offset sweep converges and Jason wants finer tuning,
extend the per-frame `labels` dict with:

6. `head_crown` — top of skull (no hair; bone surface point)
7. `left_knee` — patella center on the LEFT leg
8. `right_knee` — patella center on the RIGHT leg
9. `left_ankle` — talus center on the LEFT foot
10. `right_ankle` — talus center on the RIGHT foot

The labeler tool currently only prompts for the 5 first-pass points;
pass-2 keypoints can be added by re-running with an extended
`KEYPOINT_ORDER` (small code change).

## 8. How to use the tooling

Both scripts live at `python/pilot/scripts/`. Run from repo root with
the `.venv-benchmark` interpreter (has cv2, Pillow, matplotlib via
mediapipe deps).

### Step 1 — extract 5 phase-representative frames from one video

```powershell
.\.venv-benchmark\Scripts\python.exe `
    python/pilot/scripts/extract_phase_frames.py `
    --video-id b3fea3f0-e248-44d7-a923-0bb43172b5bf `
    --video-path python/benchmark/test_videos/b3fea3f0-e248-44d7-a923-0bb43172b5bf.mp4
```

Output: 5 PNG files at `docs/PR-7_GROUND_TRUTH/frames/<short_id>_<phase>_f<idx>.png`.
The script prints copy-pasteable `ground_truth_labeler.py` commands as
its final output — paste each to label each phase.

### Step 2 — label each phase (interactive)

```powershell
.\.venv-benchmark\Scripts\python.exe `
    python/pilot/scripts/ground_truth_labeler.py `
    --video-id b3fea3f0-e248-44d7-a923-0bb43172b5bf `
    --phase setup `
    --frame 6 `
    --image docs/PR-7_GROUND_TRUTH/frames/b3fea3f0_setup_f006.png
```

A matplotlib window opens. Click the 5 points in the order listed in
the legend. Keys:
- `u` undo the last click
- `s` save (only enabled once all 5 collected)
- `q` quit without saving

Output: `docs/PR-7_GROUND_TRUTH/b3fea3f0_setup.json`.

Repeat for each phase × each video. Plan ~3-5 min per phase if you're
careful = **~30-45 min total for the 15-sample first pass**.

### Step 3 — verify (optional)

```powershell
ls docs/PR-7_GROUND_TRUTH/*.json
```

Should show 15 files (3 × 5) when first pass is complete. PR-7b's
`offset_sweep.py` reads them all from this directory.

## 9. What's NOT measured in the first pass

Per spec v2 §5 acceptance gates:

- **wrist / hand** positions are diagnostic-only at this stage. They
  move 10+ px between consecutive frames during swing, so a 1-frame-
  off label has high noise; not worth optimizing against until torso
  anchors lock in.
- **clubhead** is out of scope entirely — that's a separate detection
  problem (PR-?? future).

## 10. Source of truth

If anything in this README disagrees with
`docs/files/PR-7_GOLF_CORRECTION_LAYER_SPEC_v2.md`, the spec wins.
This README is operational guidance for the labeling workflow only.
