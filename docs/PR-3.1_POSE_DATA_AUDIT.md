# PR-3.1 / PR-4.1 pose-data audit — hip keypoint y mis-position

**Date**: 2026-05-18
**Audit scope**: pose-estimation accuracy on
`b3fea3f0-e248-44d7-a923-0bb43172b5bf` setup frame (ts=0.214).
Browser-measured ground truth shows DB hip y ≈ 634 vs visual belt y ≈
770 (off by ~136 px on a 720×1280 video). Skeleton overlay renders DB
data faithfully → rendering is NOT the bug, pose-estimation data IS.
**Read-only.** No code, no commits, no PR.

---

## 1. `kp_source` metadata vs. actual pipeline

### 1.1 What the DB row says

```json
{
  "kp_source": "mediapipe_pose",
  "yolo_anchor_correction": {
    "applied": true,
    "anchor_phases": [...],
    "method": "linear_per_segment"
  }
}
```

### 1.2 What actually runs

The label is **misleading**. Two pose models contribute to every
stored timeline:

| Source | Model | Coverage | File / call site |
|---|---|---|---|
| **Base** (per-frame, all ~70 frames) | MediaPipe Pose v0.10.x, `model_complexity=1` (Full) | every sampled frame | `python/analyzer.py:198` (`mp.solutions.pose.Pose(...)`); `python/analyzer.py:222` (`pose.process(rgb)`); mapped to COCO 17 names by `python/pose_timeline.py:65-92` (`extract_coco_subset_from_mediapipe`) |
| **Anchor correction** (5 frames only) | YOLO11m-pose, exported to ONNX via Ultralytics build stage, runtime via `onnxruntime` | 5 phase frames (setup/top/transition/impact/finish) | `python/yolo/inference.py:73-103` (`_run_sync` → `session.run`); fed into `python/pose_timeline.py:267-408` (`apply_yolo_anchor_correction`) |

`keypoint_source: "mediapipe_pose"` is **hardcoded** at
`python/pose_timeline.py:257` inside `build_timeline_from_raw_coco_frames`.
It does NOT reflect whether YOLO anchor correction has been applied —
that lives in the parallel `yolo_anchor_correction.applied` field.

### 1.3 What the stored data actually represents

**Hybrid: MediaPipe per-frame coordinates + YOLO offset
interpolation at 5 anchors.** Specifically, for every keypoint `k`
in every frame `f`:

```
stored[f][k] = mediapipe[f][k] + interpolated_offset(f.ts, k)
where interpolated_offset(ts, k) is a per-keypoint linear lerp of
(yolo[anchor_i][k] - mediapipe[anchor_i_mp_frame][k]) across the
5 anchor phases.
```

Concretely, frame 3 at ts=0.214 (BEFORE setup phase, which is at
ts≈0.3 in the user's video) falls into the "before-first-anchor"
bracket. Per `python/pose_timeline.py:362-364`:

```python
if ts <= anchor_ts[0]:
    seg_a, seg_b = anchor_offsets[0], anchor_offsets[0]
    t_lerp = 0.0
```

So `stored[3] = mediapipe[3] + setup_anchor_offset`. The setup anchor
offset is `yolo[setup] - mediapipe[setup_mp_frame]`. This IS the YOLO
correction at this frame — applied directly, no lerp.

**Recommendation**: extend the schema for clarity in a follow-up PR
(out of scope here):

```json
{
  "kp_source": "mediapipe_pose+yolo_anchor",  // hybrid label
  "yolo_anchor_correction": { ... }            // unchanged
}
```

---

## 2. `apply_yolo_anchor_correction` — line-by-line audit

### 2.1 Location and method

- **File**: `python/pose_timeline.py`
- **Function**: `apply_yolo_anchor_correction(timeline, yolo_keypoints_per_phase, phase_markers, min_conf=0.3)` — lines 267-408
- **Method string written to DB**: `"linear_per_segment"`
- **Coverage**: ALL 17 COCO keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles). Not just shoulder/hip.

### 2.2 Algorithm step-by-step

```
Step 1 — Build 5 anchor offsets (lines 313-351):
  For each phase in {setup, top, transition, impact, finish}:
    a. Find MediaPipe frame whose ts is closest to phase_markers[phaseTime].
    b. For each of 17 keypoints:
         If both mp_kp.conf >= 0.3 AND yolo_kp.conf >= 0.3:
           offset = (yolo_kp.x - mp_kp.x, yolo_kp.y - mp_kp.y)
         Else:
           offset = None    (this kp skipped for this anchor)

Step 2 — Apply per-frame interpolated offset (lines 358-397):
  For each frame f in timeline.frames (~70 total):
    Locate which segment f.ts falls in:
      If f.ts <= anchor_ts[0]:           use anchor[0] (no lerp)
      Elif f.ts >= anchor_ts[-1]:        use anchor[4] (no lerp)
      Else:                              lerp between anchor[i], anchor[i+1]
                                         with t = (f.ts - anchor_ts[i])
                                                / (anchor_ts[i+1] - anchor_ts[i])

    For each of 17 keypoints:
      Pull anchor offsets at both ends of the segment. If one is None,
      use the other. If both None, skip (no correction this kp this frame).
      Otherwise linear-lerp the two offsets and add to f.keypoints[kp].
```

### 2.3 What "linear_per_segment" means in plain English

"Pull MediaPipe values toward YOLO values at 5 anchor moments, with
smooth interpolation between those moments." There is **no anatomical
reference** — the correction is purely model-vs-model.

### 2.4 Why correction can't fix the hip problem on `b3fea3f0`

The correction nudges MediaPipe toward YOLO. **If both models share
the same hip-position bias, the correction shifts hip a tiny amount
within that bias band but never escapes it.**

- COCO dataset annotation guideline for "hip": "the most pronounced
  part of the hip when standing upright" — ambiguous; in practice
  many COCO annotators land at iliac-crest level or even higher
  (belly-button region), not at the strict femoral head.
- MediaPipe Pose's BlazePose v0.10 hip kp inherits a similar bias
  because its training set is annotation-aligned with COCO-style
  conventions.
- YOLO11m-pose is trained on COCO-Keypoints directly. Same bias.

**Net result**: both models output hip at roughly the same wrong
place. Their offset at setup phase is small. `stored = mediapipe +
small_offset ≈ mediapipe`, which is still wrong.

### 2.5 The geometry doesn't even add up to "anatomical hip"

For `b3fea3f0` setup (user's measurements):

| Landmark | Native y | Source |
|---|---|---|
| shoulder midpoint | 475 | DB (shoulders avg) |
| visual belt | ~770 | Jason's eye-measurement |
| **stored "hip"** | **~634** | DB hip midpoint |

Treating shoulder→belt as ~27% of body height (standard adult
proportion):

```
body_height_px ≈ (770 − 475) / 0.27 ≈ 1093 px
femoral_head_y ≈ shoulder_y + 0.30 × body_height ≈ 475 + 328 ≈ 803
```

So even the **anatomically correct femoral head** would be at y≈803,
**below** the visual belt (770), not above it.

The DB stores y=634, which is **169 px above the femoral head and
136 px above the belt**. This is not at the femoral head, not at the
belt, not at the iliac crest — it's somewhere between belly-button
and lower chest. That's a **model-detection error, not a "wrong
anatomical convention"**.

Likely causes (un-verified, would need a re-run to confirm):
1. **Setup-pose foreshortening**: golfer crouched at setup; torso
   leaned forward. Camera at chest height. Hip projects upward on
   screen relative to a straight-standing pose. MediaPipe may
   over-correct for this.
2. **Clothing / occlusion**: baggy shorts or pants below the belt
   line hide the femoral head visual cue. Model defaults toward
   easier-to-detect features (belly-button, belt buckle).
3. **Both models share this failure mode**: anchor correction can't
   help.

---

## 3. Local re-run — NOT possible in this sandbox

Task #3 ("重新跑这视频的 pose 估计 ... 输出 frames[0..8] 的 raw
vs. corrected") can't be executed here. Concrete blockers:

| Required | Status | Why |
|---|---|---|
| MediaPipe Python package | ❌ not installed | `pip install mediapipe` not run; sandbox has no network/auto-install ability |
| `yolo11m-pose.onnx` model file | ❌ not in workspace | Generated in Dockerfile builder stage on Railway; not committed to repo (only in deployed image) |
| `b3fea3f0` source video file | ❌ no access | Lives in Supabase storage; no `SUPABASE_URL` / `SUPABASE_SERVICE_ROLE_KEY` in this sandbox env (verified: `env \| grep SUPABASE` returns empty) |

What's **available**: `tmp/test_swing.mp4.mp4` — the PR-2B test
artifact. Different video; cannot reproduce the specific `b3fea3f0`
hip mis-position.

### 3.1 Recommended out-of-sandbox re-run path

A debug script the team can run on Railway (or locally on a dev
machine with MediaPipe installed):

```python
# scripts/debug_pose_hip.py (not committed — proposal only)
import sys, json
sys.path.insert(0, 'python')

import cv2
import mediapipe as mp
from pose_timeline import (
    extract_coco_subset_from_mediapipe,
    build_timeline_from_raw_coco_frames,
    detect_outliers_and_reject, smooth_ema, gap_fill_linear,
    apply_yolo_anchor_correction, validate_timeline,
)

def dump(video_path, video_id, phase_markers, yolo_per_phase):
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_frames = []
    with mp.solutions.pose.Pose(model_complexity=1, smooth_landmarks=True) as pose:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if idx % 3 == 0:  # ~10fps sample
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                r = pose.process(rgb)
                if r.pose_landmarks:
                    raw_frames.append({
                        "ts": idx / 30,
                        "frame_idx": idx,
                        "interpolated": False,
                        "keypoints": extract_coco_subset_from_mediapipe(
                            r.pose_landmarks.landmark, w, h
                        ),
                    })
            idx += 1
    cap.release()

    # Snapshot at each stage
    stages = {}
    tl = build_timeline_from_raw_coco_frames(raw_frames, w, h, 10)
    stages["1_raw_mediapipe"] = [
        f["keypoints"]["left_hip"] for f in tl["frames"][:8]
    ]
    tl = detect_outliers_and_reject(tl)
    stages["2_after_outlier"] = [f["keypoints"]["left_hip"] for f in tl["frames"][:8]]
    tl = smooth_ema(tl)
    stages["3_after_ema"] = [f["keypoints"]["left_hip"] for f in tl["frames"][:8]]
    tl = gap_fill_linear(tl)
    stages["4_after_gap_fill"] = [f["keypoints"]["left_hip"] for f in tl["frames"][:8]]
    if yolo_per_phase:
        tl = apply_yolo_anchor_correction(tl, yolo_per_phase, phase_markers)
    stages["5_after_yolo_anchor"] = [
        f["keypoints"]["left_hip"] for f in tl["frames"][:8]
    ]

    print(json.dumps(stages, indent=2))
```

Run this against `b3fea3f0` from Railway (where MediaPipe / ONNX /
storage credentials all exist), and the per-stage hip evolution will
reveal exactly which step introduces or fails to correct the
mis-position. **This is the right next debugging step.**

---

## 4. Proposed fix paths

### Path X — Switch pose model entirely

| Candidate | License | Hip behaviour | Cost |
|---|---|---|---|
| **OpenPose** (CMU) | Academic non-commercial | Hip kp at hip joint; well-documented but ageing | Rebuild PR-3 pipeline (Caffe/Torch deps, ~3-5 GB image) |
| **YOLO v8 pose** | AGPL-3.0 (same as v11) | Trained on COCO-Keypoints — **same bias as current** | Low (drop-in swap in `inference.py`) but unlikely to help |
| **MMPose** (RTMPose) | Apache-2.0 | Highly accurate, supports custom annotation schemes | Heavy: `mmcv`, `mmengine` deps; rebuild Dockerfile yolo-builder stage |
| **MoveNet Thunder** (TF Hub) | Apache-2.0 | COCO bias inherited | Low cost, similar limitations |
| **MediaPipe Holistic** | Apache-2.0 | More keypoints incl. some pelvic detail | Heavier than Pose alone, similar bias risk |

**Risk**: most candidates are COCO-trained and will inherit the same
hip bias. Only OpenPose-CMU and a self-trained MMPose would
demonstrably improve, and both are expensive moves.

**Verdict**: not the cheapest fix; should NOT be the first move.

### Path Y — Hard anatomical constraint on top of existing pose

Add a post-processing pass in `python/pose_timeline.py` after
`apply_yolo_anchor_correction`:

```python
def enforce_anatomical_hip_constraint(timeline: dict) -> dict:
    """
    Override stored hip_y with an anatomical estimate derived from
    shoulder + ankle:
        hip_y_est = shoulder_y + ratio * (ankle_y - shoulder_y)
    Calibrated ratio: 0.32 (anatomical default — shoulder-to-femoral-
    head / shoulder-to-ankle).
    """
    RATIO = 0.32
    for f in timeline["frames"]:
        kp = f["keypoints"]
        for side in ("left", "right"):
            sh = kp[f"{side}_shoulder"]
            an = kp[f"{side}_ankle"]
            if sh[1] is not None and an[1] is not None:
                hip_y_est = sh[1] + RATIO * (an[1] - sh[1])
                # Replace y only when the model's hip diverges > 50 px
                # from anatomical estimate; otherwise trust the model.
                hp = kp[f"{side}_hip"]
                if hp[1] is not None and abs(hp[1] - hip_y_est) > 50:
                    hp[1] = hip_y_est
    return timeline
```

**Pros**:
- Single-file, contained, easy to revert.
- Catches the catastrophic cases (DB hip 169 px off anatomy) without
  disturbing well-detected frames.

**Cons**:
- Magic constant 0.32 doesn't account for **golfer setup pose** —
  hips bent, knees flexed, ankle position shifted forward. Real
  setup-pose ratio could be 0.28-0.36 depending on stance depth.
- Fragile across body types (tall vs short legs, gender, age).
- Bypasses the pose model rather than improving it.
- Side-on / down-the-line camera angles break the assumption that
  shoulder → ankle is roughly vertical.

**Verdict**: viable as a **belt-and-suspenders patch** on top of an
unrelated root-cause fix, but not a standalone fix.

### Path Z — Re-define disc geometric target (proposed, in PR-3.1 scope)

The user-facing complaint is *"hip disc not on belt"*, not *"hip
keypoint anatomically wrong"*. The right fix may be to **decouple
disc center from raw hip kp**:

```python
# proposed (frontend or new backend post-processing field):
disc_hip_y = mediapipe_hip_y + α * (mediapipe_ankle_y - mediapipe_hip_y)
where α ∈ [0.05, 0.20] empirically tuned to land near belt.
```

Or alternatively expose a derived field on `pose_timeline_2d`:

```json
{
  "frames": [{
    "keypoints": { /* raw COCO kp, unchanged */ },
    "disc_anchors": {
      "hip_belt_y": <derived>,
      ...
    }
  }]
}
```

Frontend reads `disc_anchors.hip_belt_y` instead of `keypoints.hip[1]`.

**Pros**:
- Preserves the raw pose data for honest debugging and future ML
  retraining.
- Decouples "where the disc renders" from "where COCO says the hip
  is" — they're different concepts and should be different fields.
- Tunable without touching the pose model.
- No license / Docker risk.
- Skeleton overlay continues to show raw kp (still honest); disc
  uses the derived anchor.

**Cons**:
- Requires schema bump to `pose_timeline_2d` v2 (or new column).
- The empirical constant α has to be calibrated, and may differ for
  shoulders (which appear approximately correct already).

**Verdict**: best long-term path. Honest about the model limitation
while fixing the user-visible symptom.

### Path W — Drop disc rendering, ship only skeleton

Out-of-scope here but worth mentioning: if the disc abstraction
demands a level of anatomical accuracy the underlying data can't
support, simplifying to "just show the 17 skeleton dots + edges"
removes the disc-vs-data mismatch entirely. Skeleton already renders
honestly. This is a product-design decision, not a technical fix.

---

## 5. Recommendation

1. **Immediate**: file follow-up ticket to **run the debug script
   from §3.1 on Railway against `b3fea3f0`**, capture per-stage hip
   evolution. Validates which stage the error is introduced (raw MP
   wrong? YOLO wrong? lerp wrong direction?). ~30 min of work; needed
   evidence before any code change.
2. **If raw MediaPipe alone is wrong**: Path X (model swap) is on the
   table; consider RTMPose if accuracy budget allows the heavier
   container.
3. **If raw MP is wrong AND raw YOLO is wrong in the same direction**:
   neither §1 model is the root cause; both inherit COCO bias. **Path
   Z** is the right fix — derive disc anchor from raw kp + visual
   ratio, ship that on a `disc_anchors` field, leave raw kp honest.
4. **If raw MP is right but YOLO anchor correction pulls it wrong**:
   investigate YOLO export quality; specifically the
   `verify_onnx_export.py` only checks shoulders, hips weren't in the
   acceptance suite.
5. **Path Y** (anatomical constraint) can be slotted on top of any of
   the above as a safety net for catastrophic outliers (>50 px off
   anatomical expectation).

**Out of scope for this audit (and for any PR triggered by it
without further discussion)**:
- Frontend rendering changes (already correct).
- Schema versioning for `pose_timeline_2d` (requires its own design).
- Backend pose model replacement (heavy, multi-PR).

---

## 6. Audit deliverables

- ✅ This document at `docs/PR-3.1_POSE_DATA_AUDIT.md`
- ✅ No code touched
- ✅ No PR opened
- ✅ Frontend untouched (rendering is correct per task constraint)
- ✅ `kp_source` metadata not modified
- ✅ Followup script proposal in §3.1 (not committed)

⏸ Awaiting Jason's verdict: run the debug script, choose Path Z /
X / Y / W, or escalate further.
