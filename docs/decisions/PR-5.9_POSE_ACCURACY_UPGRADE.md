# PR-5.9 — Pose Accuracy Upgrade v1

**Date**: 2026-05-19
**Status**: Spec finalized · Pending implementation
**Audit ground truth**: docs/PR-5.9_AUDIT.md (commit 59e3629)
**Precursor**: PR-5.8A merged (#2) — render-time anchor expansion working at setup

---

## 1. Why this exists

PR-5.8A fixed setup-phase anchor positions. Verification on b3fea3f0 revealed severe positional drift through every other phase. Root cause is the underlying pose timeline, not the render-time expansion.

Three compounding problems (per audit):

1. **Sampling rate** — `sample_fps = 4.0` (NOT 10 as documented). 250ms between samples. Downswing is ~250ms total, so only 1-2 pose samples cover the most critical motion.
2. **Causal EMA α=0.4** — `pose_timeline.py:147-178`. Each smoothed point is 60% old data, introducing structural 100-200ms lag.
3. **Frontend nearest-neighbor** — `nearestFrame()` and `frameAt()` are both nearest-only. Between samples, dots are frozen while body moves. Frontend also hardcodes `fps: 30` while data is at 4fps, so the lookup window mismatch compounds.

Combined: dots lag body by ~300-450ms during fast motion. Visible as "ghost trail" in Jason's screenshots 3-6.

Accuracy of body keypoints is SwingCue's core product metric. Visual overlays mean nothing if body points drift. This PR is the prerequisite for any further disc / X-factor / swing path work.

---

## 2. Eight tasks

### Task 1 — Native fps sampling

**Current**: `sample_fps = 4.0` hardcoded in Python pipeline.
**Native fps**: already stored in `swing_videos.video_metadata_json.fps` (audit §1/§7).

**Change**:
- Python: `sample_fps = min(video_metadata.fps, 60)`
- Python: store actual rate used in `fps_sampled` field
- Frontend: read `video_metadata_json.fps` instead of hardcoding `fps: 30` at `page.tsx:94`
- Optional URL override `?poseFps=N` for testing (clamped 4-60)

**Cost**: ~7.5× more MediaPipe inference per video (4 → 30 fps). Railway worker may need scale-up. JSONB size grows ~7.5× per video (~30KB → ~225KB, well within limits).

**Difficulty**: LOW (audit §8)

### Task 2 — Bidirectional EMA (replace causal)

**Current**: `smooth_ema(α=0.4)` causal pass at `pose_timeline.py:147-178`. Raw discarded.

**Change**: forward-backward double pass (zero phase delay):

```python
def bidirectional_ema(values, alpha=0.4):
    # Forward pass (causal)
    forward = [values[0]]
    for i in range(1, len(values)):
        forward.append(alpha * values[i] + (1-alpha) * forward[i-1])
    
    # Backward pass (anti-causal)
    backward = [values[-1]]
    for i in range(len(values)-2, -1, -1):
        backward.insert(0, alpha * values[i] + (1-alpha) * backward[0])
    
    # Average — non-causal, no lag
    return [(f + b) / 2 for f, b in zip(forward, backward)]
```

Apply per-keypoint (each of x, y separately).

**Note**: this is offline post-processing. Requires entire timeline in memory before write. SwingCue is offline analysis so this is fine.

**Difficulty**: LOW

### Task 3 — Frontend pose interpolation

**Current**: `nearestFrame()` and `frameAt()` are both nearest-only (audit §6).

**Change** — replace `frameAt` and `nearestFrame` with `interpolatedFrame(t)`:

```typescript
function interpolatedFrame(timeline, t) {
  const before = lastFrameAtOrBefore(timeline, t);
  const after = firstFrameAtOrAfter(timeline, t);
  if (!before) return after;
  if (!after) return before;
  if (before === after) return before;
  
  const ratio = (t - before.ts) / (after.ts - before.ts);
  
  const keypoints = {};
  for (const name of Object.keys(before.keypoints)) {
    const [x1, y1, c1] = before.keypoints[name];
    const [x2, y2, c2] = after.keypoints[name];
    if (x1 === null || x2 === null) {
      // Missing endpoint → use the non-null side
      keypoints[name] = (x1 === null) ? [x2, y2, c2] : [x1, y1, c1];
    } else {
      keypoints[name] = [
        x1 + (x2 - x1) * ratio,
        y1 + (y2 - y1) * ratio,
        Math.min(c1, c2),  // conservative confidence
      ];
    }
  }
  return { ts: t, keypoints, interpolated: true };
}
```

Replace all call sites of `nearestFrame` / `frameAt` in:
- `SkeletonOverlay.tsx`
- `computeShoulderDisc` / `computeHipDisc`
- Any other pose-by-time lookup

**Difficulty**: MEDIUM (audit §8)

### Task 4 — Raw + final dual storage

**Current**: Only smoothed pose stored. Raw discarded (audit §2/§4).

**Schema change** (no version bump — field-presence detection):

```json
{
  "version": 1,
  "keypoint_source": "mediapipe_pose_v1_5",
  "fps_sampled": 30,
  "frames": [
    {
      "ts": 0.0,
      "frame_idx": 0,
      "keypoints": {
        "left_shoulder": [511.9, 475.1, 1.0],
        "head_crown": [451.0, 380.0, 0.92],
        ...
      },
      "raw_keypoints": {
        "left_shoulder": [510.2, 476.8, 1.0],
        "head_crown": [450.5, 381.3, 0.92],
        ...
      },
      "interpolated": false
    }
  ]
}
```

**Python**: preserve raw kp before passing through bidirectional smoothing. Write both into the per-frame dict.

**Frontend compatibility**: 
- v1 videos (no `raw_keypoints` field) — continue to render normally, debug mode silently disabled
- v1.5 videos (has `raw_keypoints`) — debug mode functional

**Difficulty**: MEDIUM (audit §8)

### Task 5 — Debug overlay (raw vs final)

**URL param**: `?debug=pose` toggles debug mode.

**Render** (`SkeletonOverlay.tsx`):
- Final keypoints: existing yellow/red dot style (unchanged)
- Raw keypoints: smaller blue dots, radius 3, color `#5599FF`, opacity 0.7
- Optional: thin grey line connecting each raw → final pair to visualize smoothing effect

**Production**: debug mode hidden behind URL param. No production UI change.

**Difficulty**: MEDIUM

### Task 6 — Confidence-based fade

**Current**: confidence `>= HIGH_CONF (0.7)` shown bright, lower shown dim, no fade gradient.

**Change**:
- Add fade thresholds:
  - `conf >= 0.7`: opacity 1.0
  - `conf >= 0.5`: opacity 0.7
  - `conf >= 0.3`: opacity 0.4
  - `conf < 0.3`: hidden
- Applied to dot rendering and edge rendering (edge opacity = min of two endpoint confidences)

**Rationale**: during occlusion, low-confidence point fades instead of snapping to bad location.

**Difficulty**: LOW

### Task 7 — head_crown derivation

**Current**: Not in v1 output. MediaPipe outputs `mouth_left`, `mouth_right` but they're dropped at `MEDIAPIPE_TO_COCO_IDX` (audit §5, 6 LoC fix).

**Change**:
- Python: extract `mouth_left`, `mouth_right` alongside other landmarks (do NOT add to final keypoints dict, internal only)
- Python: compute `head_crown`:
```python
  HEAD_CROWN_FACTOR = 0.45  # tunable Python module constant
  
  def derive_head_crown(landmarks):
      ear_mid_x = (landmarks.left_ear.x + landmarks.right_ear.x) / 2
      ear_mid_y = (landmarks.left_ear.y + landmarks.right_ear.y) / 2
      mouth_mid_x = (landmarks.mouth_left.x + landmarks.mouth_right.x) / 2
      mouth_mid_y = (landmarks.mouth_left.y + landmarks.mouth_right.y) / 2
      
      crown_x = ear_mid_x + (ear_mid_x - mouth_mid_x) * HEAD_CROWN_FACTOR
      crown_y = ear_mid_y + (ear_mid_y - mouth_mid_y) * HEAD_CROWN_FACTOR
      
      conf = min(landmarks.left_ear.visibility, landmarks.right_ear.visibility,
                 landmarks.mouth_left.visibility, landmarks.mouth_right.visibility) * 0.8
      
      return [crown_x, crown_y, conf]
```
- Python: store as `keypoints["head_crown"]`
- Frontend: render `head_crown` as a single red dot (no L/R pair). Add it to the COCO_KEYPOINT_NAMES iteration with appropriate edge (head_crown → nose, or head_crown → mid_shoulder).

**Tuning**: `HEAD_CROWN_FACTOR = 0.45` first-cut. Adjust via Python redeploy if visual off.

**Difficulty**: LOW

### Task 8 — Do not touch disc

No changes to `phaseCompression.ts`, `computeDiscParams.ts` semantics, PR-5.8A expansion logic. Disc will visually improve as a side effect of better keypoints; we don't tune it.

---

## 3. Implementation order

1. **Python backend** — Tasks 1, 2, 6, 7 + storage layout (Task 4 Python side)
2. **Re-analyze b3fea3f0** with new backend, verify JSONB size + structure
3. **Frontend** — Task 3 (interpolation), Task 4 frontend read, Task 5 (debug overlay), Task 6 (fade)
4. **Vercel preview** — visual acceptance pass

Single PR. Branch: `pr-5.9/pose-accuracy-upgrade`.

---

## 4. Acceptance criteria (b3fea3f0)

For setup / top / transition / impact / finish:

- ✅ Shoulder dots on coaching anchor (PR-5.8A definition)
- ✅ Hip dots on coaching anchor
- ✅ No 100ms+ lag during scrub or playback
- ✅ Dots animate continuously between samples (no freezing)
- ✅ head_crown at top of skull, stable through all phases
- ✅ Debug mode `?debug=pose`: blue raw dots visible alongside final, smoothing visualized

**Hard fail conditions** (block merge):
- Any phase has shoulder/hip dot drift > 30px from coaching anchor
- Dot freeze >50ms visible during playback at 1× speed
- head_crown appearing inside skull / above hair / off head
- Production users see debug dots without `?debug=pose`

---

## 5. Out of scope

- Disc visual tuning (any)
- Model swap to MoveNet/RTMPose (→ PR-6.0 benchmark first)
- SwingCue 17 full schema migration (→ PR-5.8 main spec)
- Foot tip / hand tip keypoints (→ PR-5.8)
- RTS Kalman smoother (→ PR-6.x if bidirectional EMA insufficient)
- 3D pose

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| Railway worker compute cost ↑ 7.5× | Monitor cost after first 10 re-analyzes; scale worker if needed; cap fps at 60 |
| Bidirectional smoothing still has visible lag | Falls back to Task 3 interpolation + lower α; can swap to Savitzky-Golay |
| MediaPipe confidence collapse at occlusion | Task 6 fade hides bad data; PR-6.0 benchmark addresses fundamentally |
| Old videos need re-analyze | Lazy re-analyze on first view if `version < required`, OR batch top N videos manually |
| head_crown wrong in non-face-on views | Tunable constant; future PR can do view-aware derivation |

---

## 7. Related docs

- `docs/PR-5.9_AUDIT.md` — audit ground truth (commit 59e3629)
- `docs/decisions/PR-5.8A_COACHING_ANCHOR.md` — anchor expansion (PR #2)
- `docs/decisions/PR-5.8_GOLF_17_KEYPOINTS.md` — future SwingCue 17 schema (deferred)
- `docs/decisions/PR-6.0_BENCHMARK_SPEC.md` — parallel pose model benchmark
