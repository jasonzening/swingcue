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

```ts
function deriveHeadCrown(mp33: MediaPipe33Frame): Keypoint {
  const earMid   = midpoint(mp33[7], mp33[8]);    // left_ear, right_ear
  const mouthMid = midpoint(mp33[9], mp33[10]);   // mouth_left, mouth_right
  // crown sits roughly one "ear-to-mouth" distance above ears
  return {
    x: earMid.x + (earMid.x - mouthMid.x) * 0.45,
    y: earMid.y + (earMid.y - mouthMid.y) * 0.45,
    v: Math.min(mp33[7].v, mp33[8].v, mp33[9].v, mp33[10].v) * 0.8,
  };
}
```

Factor `0.45` calibrated on standing setup pose with neutral head. Tunable via `?headCrownFactor=` URL param during PR-5.8 implementation.

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

New `pose_timeline_2d` JSONB structure:

```json
{
  "schema_version": 2,
  "kp_source": "mediapipe_pose_33_v2",
  "fps_sampled": 14,
  "frame_count": 70,
  "frames": [
    {
      "ts": 0.0,
      "kp": [
        { "x": 412.5, "y":  78.3, "v": 0.85 },
        { "x": 380.1, "y": 195.2, "v": 0.97 },
        ... 17 entries indexed 0..16 ...
      ]
    },
    ...
  ]
}
```

**Coexistence with v1**:
- Existing analyzed videos retain v1 (COCO 17). Migration is opt-in via re-analyze.
- Frontend reads `schema_version`; v1 → uses old COCO mapping (legacy), v2 → uses new SwingCue 17 mapping.
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

### Step 3 — Python schema upgrade
Modify `python/pose_timeline.py`:
- Stop compressing MediaPipe 33 → COCO 17.
- Output schema v2 with 17 SwingCue keypoints + derived head_crown.
- Update `kp_source` metadata to `mediapipe_pose_33_v2` (also fixes the stale `mediapipe_pose` label).

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

MediaPipe Pose documentation describes landmarks 11/12/23/24 as the "shoulder" and "hip" joints (not surface landmarks). Our hypothesis: these are already at glenohumeral / femoral head centers.

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
grep -nA 3 "kp_source\|schema_version\|landmarks" python/pose_timeline.py

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
- `kp_source = "mediapipe_pose"` (stale, per existing memory note) or `"mediapipe_yolo_hybrid_v1"`
- Schema version = 1 currently, 17 COCO kp output
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
