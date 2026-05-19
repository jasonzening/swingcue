# PR-5.8 — SwingCue Golf 17 Keypoint Specification

**Date**: 2026-05-19
**Status**: Spec finalized · Pending audit & implementation
**Supersedes**: PR-4 COCO 17 schema (`pose_timeline_2d` v1)
**Authors**: Jason + Claude (chat session 2026-05-19)

---

## 1. Why this exists

PR-5.1 → PR-5.7 spent 8 iterations trying to make the disc visual "feel right" on top of COCO 17 keypoints. The disc kept looking detached from the body. The root cause, discovered at the end of PR-5.7:

1. **COCO 17 keypoints are skin-surface landmarks** (acromion, lateral epicondyle, iliac crest). Lines connecting them sit on the body's *external surface*, not on the *bone midline*. Visual coaching overlays need to run through the body's interior to look anatomically correct.

2. **COCO 17 is missing 5 keypoints critical for golf**: head crown (head sway tracking), hand fingertips × 2 (grip / club anchor), foot tips × 2 (stance direction).

Solution: define a **17-point golf-specific keypoint set**, sourced from MediaPipe Pose Landmarker 33 (already in our pipeline since PR-4), with keypoints chosen to be at **bone endpoints / joint centers** so connecting lines fall on bone midlines.

---

## 2. Decision

SwingCue replaces COCO 17 output with **17 golf-specific keypoints sourced directly from MediaPipe Pose Landmarker 33**. Schema v2 of `pose_timeline_2d`.

YOLO is retained as an anchor-correction layer (per PR-4 architecture) but no longer determines the output schema.

---

## 3. The 17 Keypoints

Indexing: 0 = head_crown (singular center), 1-16 = 8 left/right pairs (anatomical convention — L before R).

| idx | Name | MediaPipe 33 source | Render color (face-on) | Golf use |
|----|------|---------------------|---------------------|----------|
| 0 | head_crown | **DERIVED** from kp[7,8,9,10] | red (center) | Head sway tracking, swing axis stability |
| 1 | shoulder_L | 11 (left_shoulder) | screen-position sorted | Shoulder turn, X-factor upper |
| 2 | shoulder_R | 12 (right_shoulder) | screen-position sorted | Shoulder turn, X-factor upper |
| 3 | elbow_L | 13 (left_elbow) | screen-position sorted | Lead/trail elbow bend |
| 4 | elbow_R | 14 (right_elbow) | screen-position sorted | Lead/trail elbow bend |
| 5 | wrist_L | 15 (left_wrist) | screen-position sorted | Wrist hinge / 立腕 |
| 6 | wrist_R | 16 (right_wrist) | screen-position sorted | Wrist hinge / 立腕 |
| 7 | hand_L | 19 (left_index fingertip) | screen-position sorted | Lead hand grip / club anchor |
| 8 | hand_R | 20 (right_index fingertip) | screen-position sorted | Trail hand grip / club anchor |
| 9 | hip_L | 23 (left_hip) | screen-position sorted | Hip turn, X-factor lower |
| 10 | hip_R | 24 (right_hip) | screen-position sorted | Hip turn, X-factor lower |
| 11 | knee_L | 25 (left_knee) | screen-position sorted | Weight transfer, knee flex |
| 12 | knee_R | 26 (right_knee) | screen-position sorted | Weight transfer, knee flex |
| 13 | ankle_L | 27 (left_ankle) | screen-position sorted | Balance, weight distribution |
| 14 | ankle_R | 28 (right_ankle) | screen-position sorted | Balance, weight distribution |
| 15 | foot_L | 31 (left_foot_index toe) | screen-position sorted | Stance direction, target line |
| 16 | foot_R | 32 (right_foot_index toe) | screen-position sorted | Stance direction, target line |

**L / R suffix = anatomical** (golfer's body left/right), NOT screen position. Rendering layer handles screen-position sorting.

---

## 4. Derived Geometric Helpers (Phase 1, zero cost)

Computed at render time, not stored in `pose_timeline_2d`:

```ts
// Spine line — anchors all rotation-related visuals
spine_top    = midpoint(kp[1], kp[2])    // mid-shoulder
spine_bottom = midpoint(kp[9], kp[10])   // mid-hip
spine_line   = segment(spine_top, spine_bottom)

// Club shaft ray (Phase 1 approximation; Phase 2 replaces with real shaft detection)
club_shaft_ray_L = ray(kp[5], kp[7]).extend(factor: 4.0)   // wrist_L → hand_L → extended
club_shaft_ray_R = ray(kp[6], kp[8]).extend(factor: 4.0)   // wrist_R → hand_R → extended
// For two-handed grip, render midpoint(L,R) extended
club_shaft_midray = ray(midpoint(kp[5], kp[6]), midpoint(kp[7], kp[8])).extend(factor: 4.0)
```

### head_crown derivation formula

MediaPipe Pose does not output skull-crown directly. We derive from face landmarks:

```python
# Python-side derivation during pose extraction (in python/pose_timeline.py)
def derive_head_crown(mp_landmarks):
    """
    Computes skull crown approximation from MediaPipe ear and mouth landmarks.
    Returns [x_px, y_px, confidence] tuple.
    """
    left_ear   = mp_landmarks[7]   # MediaPipe 33 index
    right_ear  = mp_landmarks[8]
    mouth_l    = mp_landmarks[9]
    mouth_r    = mp_landmarks[10]
    
    ear_mid_x   = (left_ear.x  + right_ear.x)  / 2
    ear_mid_y   = (left_ear.y  + right_ear.y)  / 2
    mouth_mid_x = (mouth_l.x   + mouth_r.x)    / 2
    mouth_mid_y = (mouth_l.y   + mouth_r.y)    / 2
    
    # Crown sits roughly one ear-to-mouth distance above ears
    factor = 0.45  # tunable; first-cut calibration on neutral standing setup pose
    crown_x = ear_mid_x + (ear_mid_x - mouth_mid_x) * factor
    crown_y = ear_mid_y + (ear_mid_y - mouth_mid_y) * factor
    
    conf = min(left_ear.visibility, right_ear.visibility, 
               mouth_l.visibility,  mouth_r.visibility) * 0.8
    
    return [crown_x, crown_y, conf]
```

The factor `0.45` is the first-cut. Tunable via a Python module constant `HEAD_CROWN_FACTOR` so it can be changed without re-deploy. Once stable, can be promoted to a config value.

---

## 5. Rendering Convention (Frontend)

Data stores anatomical L/R. Rendering layer applies screen-position visual convention:

```ts
// At each frame, for each L/R pair:
const [yellow, red] = pair.sortByX();  // smaller x → yellow, larger x → red
// head_crown is always rendered red (singular center, no L/R)
```

This makes the rendering view-independent: face-on, behind-the-line, or any angle, the colors stay consistent with screen position.

---

## 6. Schema v2

### v1 schema (actual current state in DB)
Named-key dict per frame, value tuples `[x, y, confidence]`. Verified via direct Supabase query on b3fea3f0 (2026-05-19):

```json
{
  "version": 1,
  "keypoint_source": "mediapipe_pose",
  "fps_sampled": 10,
  "frames": [
    {
      "ts": 0.0,
      "frame_idx": 0,
      "keypoints": {
        "nose": [447.9, 437.6, 1.0],
        "left_eye": [459.2, 422.7, 1.0],
        "right_eye": [433.8, 425.6, 1.0],
        "left_ear": [483.8, 414.2, 1.0],
        "right_ear": [418.3, 423.2, 1.0],
        "left_shoulder": [511.9, 475.1, 1.0],
        "right_shoulder": [405.2, 477.6, 1.0],
        "left_elbow": [497.1, 576.9, 0.985],
        "right_elbow": [411.2, 587.6, 0.964],
        "left_wrist": [464.5, 671.6, 0.962],
        "right_wrist": [432.1, 679.8, 0.937],
        "left_hip": [500.4, 631.1, 1.0],
        "right_hip": [412.6, 626.4, 1.0],
        "left_knee": [529.5, 767.1, 0.991],
        "right_knee": [369.5, 754.8, 0.992],
        "left_ankle": [539.3, 914.1, 0.995],
        "right_ankle": [358.7, 895.5, 0.993]
      },
      "interpolated": false
    }
  ]
}
```

Key facts about v1:
- `keypoints` is a **named-key dict**, NOT an indexed array.
- Each value is `[x_px, y_px, confidence]` — a 3-tuple list, NOT a `{x, y, v}` object.
- No `frame_count` field exists at top level.
- 17 named landmarks following MediaPipe / COCO `left_*` `right_*` naming.
- Each frame has `frame_idx` (integer) and `interpolated` (boolean).

### v2 schema (target for PR-5.8 implementation)
Same overall envelope (named-key dict, `[x, y, c]` tuples, per-frame `frame_idx` + `interpolated`). Three changes:

1. Rename keypoints from MediaPipe `left_*`/`right_*` style to SwingCue anatomical `*_L`/`*_R` suffix (matches §3 spec table).
2. Drop face landmarks (`nose`, `left_eye`, `right_eye`, `left_ear`, `right_ear`) from output. Ears + mouths still extracted internally for `head_crown` derivation, then discarded.
3. Add 5 new keypoints not in v1: `head_crown` (derived), `hand_L`, `hand_R`, `foot_L`, `foot_R`.

`head_crown` is computed **Python-side during pose extraction** (formula in §4), not derived at render time. Frontend reads `keypoints["head_crown"]` as a plain stored value.

```json
{
  "version": 2,
  "keypoint_source": "mediapipe_pose_33_v2",
  "fps_sampled": 10,
  "frames": [
    {
      "ts": 0.0,
      "frame_idx": 0,
      "keypoints": {
        "head_crown":  [451.0,  380.0,  0.92],
        "shoulder_L":  [511.9,  475.1,  1.0],
        "shoulder_R":  [405.2,  477.6,  1.0],
        "elbow_L":     [497.1,  576.9,  0.985],
        "elbow_R":     [411.2,  587.6,  0.964],
        "wrist_L":     [464.5,  671.6,  0.962],
        "wrist_R":     [432.1,  679.8,  0.937],
        "hand_L":      [466.0,  695.0,  0.93],
        "hand_R":      [434.0,  703.0,  0.92],
        "hip_L":       [500.4,  631.1,  1.0],
        "hip_R":       [412.6,  626.4,  1.0],
        "knee_L":      [529.5,  767.1,  0.991],
        "knee_R":      [369.5,  754.8,  0.992],
        "ankle_L":     [539.3,  914.1,  0.995],
        "ankle_R":     [358.7,  895.5,  0.993],
        "foot_L":      [540.0,  935.0,  0.99],
        "foot_R":      [357.0,  920.0,  0.98]
      },
      "interpolated": false
    }
  ]
}
```

### Coexistence with v1
- Existing analyzed videos retain v1. Migration is opt-in via re-analyze.
- Frontend reads `version` field first; v1 → legacy COCO mapping (kept as-is), v2 → new SwingCue 17 mapping.
- New uploads always produce v2.

---

## 7. Implementation Order (PR-5.8)

### Step 1 — Audit current state
Verify Claude's memory that MediaPipe + YOLO hybrid is actually running. See §11 for exact audit commands.

### Step 2 — Visual verification of joint centers
Hypothesis: MediaPipe landmarks for shoulder/hip are already at glenohumeral / femoral head (joint centers), unlike COCO acromion / iliac crest (surface). If true, no offset needed.

Test on `b3fea3f0-e248-44d7-a923-0bb43172b5bf` (test video):
- Render skeleton overlay using raw MediaPipe 11/12 (shoulder) and 23/24 (hip) positions.
- Visually check if shoulder→elbow line falls on humerus midline.
- If yes: no offset needed for shoulders/hips. PR-5.8 complete after schema migration.
- If no: apply joint center offset per original PR-5.8 design (8% of `|shoulder→hip|` toward body center).

### Visual acceptance criteria (must pass before deciding no-offset)
- Skeleton overlay: shoulder → elbow line visibly passes through upper arm midline (not along outer skin edge)
- Skeleton overlay: hip → knee line visibly passes through thigh midline (not along outer skin edge)
- If both pass on b3fea3f0 setup frame: do NOT apply any joint-center offset. Proceed to Step 3 schema migration only.
- If either fails: apply offset per original PR-5.8 design (`?shoulderOffset=0.08&hipOffset=0.08`), URL-tunable, iterate visually on test video until pass.

### Step 3 — Python schema upgrade
Modify `python/pose_timeline.py`:

- Stop dropping the 7 MediaPipe-33 landmarks the v2 schema needs. Specifically, **keep** these MediaPipe 33 indices in addition to currently-kept landmarks:
  - `mouth_left` (9), `mouth_right` (10) — required only for `head_crown` derivation, then discarded
  - `left_index` (19), `right_index` (20) — become `hand_L`, `hand_R`
  - `left_foot_index` (31), `right_foot_index` (32) — become `foot_L`, `foot_R`
- **Drop** these face landmarks from final output (kept only as internal inputs to `head_crown`):
  - `nose` (0), `left_eye` (2), `right_eye` (5), `left_ear` (7), `right_ear` (8)
- **Compute `head_crown`** Python-side using formula in §4, emit at `keypoints["head_crown"]`.
- **Rename** all anatomical landmarks from MediaPipe `left_*`/`right_*` style to SwingCue `*_L`/`*_R` suffix style:
  - `left_shoulder` → `shoulder_L`, `right_shoulder` → `shoulder_R`
  - same pattern for elbow, wrist, hip, knee, ankle
  - `left_index` → `hand_L`, `right_index` → `hand_R`
  - `left_foot_index` → `foot_L`, `right_foot_index` → `foot_R`
- Update `version` field from 1 to 2.
- Update `keypoint_source` field from `"mediapipe_pose"` to `"mediapipe_pose_33_v2"`.
- YOLO anchor correction split for SwingCue 17:
  - 12 anatomical-overlap points (shoulder/elbow/wrist/hip/knee/ankle × 2): retain YOLO correction
  - 5 new points (head_crown, hand_L, hand_R, foot_L, foot_R): use MediaPipe raw value, no YOLO correction (YOLO does not output these landmarks)

### Step 4 — Frontend mapping migration
- Replace `src/lib/skeleton/coco.ts` with `src/lib/skeleton/swingcue17.ts`.
- Update `src/components/SkeletonOverlay.tsx` to render 17 dots + 17 lines.
- Update `src/lib/disc/computeDiscParams.ts` to use new indices (kp[1]/kp[2] for shoulders, kp[9]/kp[10] for hips).
- Add `src/lib/skeleton/derived.ts` for spine_line, club_shaft_ray, head_crown helpers.

### Step 5 — Visual acceptance on b3fea3f0
- Skeleton overlay: 17 dots + 17 lines visible, lines through body interior (bone midlines).
- Disc overlay (existing PR-5.4 visual): re-validate it still works on new keypoint positions.
- Phase markers (PR-3.1 bug still pending, separate issue).

---

## 8. Joint Center Question (deferred to Step 2 verification)

MediaPipe landmarks may be closer to joint centers than COCO surface landmarks, but this must be visually verified before any offset is applied.

**If hypothesis confirmed** (Step 2 visual check passes): no joint-center offset needed. PR-5.8 is purely schema migration + index re-mapping.

**If hypothesis falsified**: apply offset per original PR-5.8 design:
```ts
const SHOULDER_OFFSET = 0.08;  // % of |shoulder→hip|
const HIP_OFFSET      = 0.08;
shoulder_center = shoulder_kp + (midHip - shoulder_kp).normalize() × dist × SHOULDER_OFFSET
hip_center      = hip_kp + (midShoulder - hip_kp).normalize() × dist × HIP_OFFSET
```
URL-parameterize for runtime tuning: `?shoulderOffset=0.08&hipOffset=0.08`.

---

## 9. Phase 2 Deferred Items

Not included in PR-5.8 scope. Each becomes a separate PR.

| Feature | Approach | Estimated work |
|---------|----------|----------------|
| **Club shaft detection** (real, not derived ray) | Object detection model or Hough-line on cropped frame | 1-2 weeks |
| **Ball position** | Small-object detector (golf-ball-specific) or stance-foot-reference heuristic | 1 week |
| **Wrist hinge angle metric** | Compute angle(forearm_vector, hand_vector) per frame, store in derived metrics | 2 days |
| **Spine angle metric** | Compute angle(spine_line, vertical) per frame, store in derived metrics | 2 days |
| **X-factor metric** | abs(shoulder_angle - hip_angle), stored per frame | 1 day |
| **Cervical/thoracic/lumbar spine keypoints** | Extended spine derivation, currently only top/bottom | 3 days |

---

## 10. Prohibitions

- ❌ NEVER mix v1 (COCO 17) and v2 (SwingCue 17) keypoint indexing in the same render frame. Schema version is read once per video, indices are immutable thereafter.
- ❌ NEVER derive head_crown by extrapolating from `nose` alone (kp[0]) — face-on view requires ear+mouth method for stability.
- ❌ NEVER write the joint-center offset directly into stored pose_timeline_2d data. Offset is a render-time transformation, applied per frame in frontend, controllable via URL param.
- ❌ NEVER use anatomical L/R labels in render-output text (always use screen position: yellow=left, red=right).
- ❌ NEVER skip emitting head_crown — if face landmarks are missing, emit kp[0] with v=0 (frontend handles gracefully).

---

## 11. Required Audit Before Implementation

Run these commands and report findings before starting Step 3:

```bash
# Verify pose pipeline is MediaPipe + YOLO hybrid (not YOLO-only)
grep -nE "mediapipe|MediaPipe|YOLO|yolo|Pose" python/pose_timeline.py

# Confirm metadata label and current schema output
grep -nA 3 "keypoint_source\|version\|landmarks" python/pose_timeline.py

# Check what MediaPipe model variant is in use (Pose Landmarker, Pose, etc.)
grep -nE "pose_landmarker|PoseLandmarker|mp\.solutions\.pose|mediapipe\.solutions" python/

# Review PR-3 and PR-4 design docs for any contradictions
cat docs/decisions/PR-3_C_ONNX_DESIGN.md | head -100
cat docs/decisions/PR-4_DESIGN.md | head -100

# Check current schema version in DB
psql -c "SELECT id, jsonb_path_query(pose_timeline_2d, '$.schema_version') AS v FROM swing_videos WHERE pose_timeline_2d IS NOT NULL LIMIT 3;"
```

Expected findings:
- Pipeline includes both `mediapipe` and `yolo` imports / function calls
- `keypoint_source = "mediapipe_pose"` (stale label, fix during PR-5.8)
- `version` field = 1 currently, 17 COCO kp output
- MediaPipe model = either `mp.solutions.pose` (legacy) or `pose_landmarker` (new API)

If audit reveals YOLO-only pipeline (no MediaPipe), revise this spec — PR-5.8 then requires adding MediaPipe integration as additional work.

---

## 12. Test Video for Visual Acceptance

`b3fea3f0-e248-44d7-a923-0bb43172b5bf` — face-on golf swing, 6.8 seconds total, ~1.5s active swing window. Known issues: PR-3.1 phase detection has setup-heavy bug (independent of PR-5.8).

Acceptance criteria for PR-5.8:
1. 17 dots render at MediaPipe-sourced positions (visually verified on body)
2. Skeleton overlay (🦴 toggle) shows 17 lines through body interior — visible "bone midline" feel
3. Disc overlay (PR-5.4 visual) re-validates with new shoulder positions
4. head_crown sits at top of skull (not on forehead, not floating above hair)
5. hand_L/R sit at fingertips when hands holding club (visible in Setup, Top, Finish phases)
6. foot_L/R sit at toe area (visible in Setup phase, often hidden mid-swing)

---

## 13. Related docs

- `docs/decisions/PR-3_C_ONNX_DESIGN.md` — Original YOLO ONNX architecture
- `docs/decisions/PR-4_DESIGN.md` — MediaPipe+YOLO hybrid + pose_timeline_2d v1 schema
- `docs/decisions/API_CLIENT_BOUNDARY.md` — Service-role vs user-session client boundary
- `docs/PR-3_AUDIT.md` — Multi-stage Dockerfile decision audit
