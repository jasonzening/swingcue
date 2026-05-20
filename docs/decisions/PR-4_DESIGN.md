# PR-4 Design v2: 全身关键点数据基础 + Skeleton 可视化

**Status:** Awaiting approval (Step 2 — design only, no code)
**Date:** 2026-05-16
**Supersedes:** [`PR-4_DESIGN_v1_head_movement_obsolete.md`](./PR-4_DESIGN_v1_head_movement_obsolete.md)
**Scope:** Build the full 17-keypoint frame-level timeline as the data
foundation for all future visualisations. Skeleton overlay is the only
visible artefact in PR-4. Single-issue overlays move to PR-5+.

---

## Section 1 — Product framework (full context for PR-4 through PR-11)

This is the product vision Jason laid out. Recorded here because every
subsequent PR (5, 6, 7, ...) is a rendering layer on top of the PR-4
data foundation. Future implementers reading just one PR should be
able to see the whole shape from any single doc.

### 1.1 The seven visual categories

Every issue SwingCue analyses falls into one of these seven groups.
PR-4's keypoint timeline is the data substrate for **all of them**;
each subsequent PR adds the rendering layer for one group.

| # | Category | Subitems | Best view | Visual primitive |
|---|---|---|---|---|
| 1 | 旋转 (Rotation) | shoulders / hips / X-factor / timing | Face-on / Both | Disc + connector line |
| 2 | 位置 (Position) | head / shoulders / hips / knees sway, spine angle | Face-on / DTL | Marker / angle line |
| 3 | 手臂 (Arms) | triangle, chicken wing, lead arm straight | Face-on + DTL | Triangle wireframe |
| 4 | 手腕 (Wrists) | hinge / lead wrist / release | DTL | Angle arc |
| 5 | 球杆 (Club) | shaft, plane, face, path | DTL | Line + trace |
| 6 | 重心 (CoG) | distribution / shift / trace | Face-on | Marker + trace |
| 7 | 路径 (Paths) | hand path / head trace | Both | Trace line |

### 1.2 Visual language system (universal across categories)

- **Grey / pale yellow** = user's actual position / actual motion
- **Green** = ideal (a tolerance band, not a single point)

This applies to every category. Designers and overlay builders must
use the same colour semantics.

### 1.3 Camera-view classification

Two shooting angles drive which visualisations make sense:

- **Face-on** (camera in front of golfer) — rotation, head sway, weight
  shift readable; club plane is foreshortened.
- **Down-the-line (DTL)** (camera behind golfer along ball line) — club
  plane, wrist angle, spine angle readable; rotation foreshortened.

PR-7 will detect view automatically. PR-4 — PR-6 assume face-on (the
current default).

### 1.4 Core product principles (non-negotiable)

1. **Extreme simplicity** — visual first, text minimum. `cue_text` /
   `drill_text` collapsed by default. Only PR-5+ may surface them.
2. **One overlay at a time** — never stack multiple issue overlays.
   User picks via tab/toggle. Prevents cognitive overload.
3. **Visual density budget** — each overlay must read at a glance on a
   phone screen. If two overlays compete for the same screen region,
   they can't both be on.

---

## Section 2 — PR roadmap (PR-4 → PR-11)

Order is recommended; PR-7 (view detection) and PR-10 (personalisation)
can be reordered if user feedback prioritises one over the other.

| PR | Scope | Depends on |
|---|---|---|
| **PR-4** | **全身 17-keypoint frame-level timeline + skeleton overlay** | — |
| PR-5 | 旋转 (disc + connector line, timing visualisation) | PR-4 data |
| PR-6 | 位置点 (sway / slide markers per joint) | PR-4 data |
| PR-7 | 视角检测 (face-on vs DTL classifier) | PR-4 data |
| PR-8 | 手腕角度 (DTL) | PR-4 data + MediaPipe Hands integration |
| PR-9 | 球杆 (plane / path detection) | New club-detection module |
| PR-10 | 个性化标准 (user body-ratio calibration) | PR-4 — PR-9 visuals |
| PR-11 | 视觉密度控制 / issue tab UI | PR-5+ overlays complete |

PR-4 unlocks PR-5, PR-6, PR-7, PR-10 — they all consume the same
`pose_timeline_2d` column.

---

## Section 3 — PR-4 specifics

### A. Schema (additive only)

#### File

`supabase/migrations/{TS}_add_pose_timeline_2d.sql` — TS generated at
implementation with `Get-Date -Format "yyyyMMddHHmmss"`.

#### Migration SQL

```sql
-- ===========================================================================
-- PR-4: pose_timeline_2d — full 17-keypoint frame-level timeline
-- ===========================================================================
-- Data foundation for PR-5+ (rotation discs, sway markers, paths, etc.).
-- One JSONB column on swing_videos, NULL for pre-PR-4 videos.
--
-- Additive only:
--   - swing_videos columns / RLS / triggers unchanged elsewhere.
--   - No new tables, no FK changes.
--   - Versioned JSON shape (`version: 1`) so future schema bumps don't
--     break existing readers.
-- ===========================================================================

ALTER TABLE public.swing_videos
  ADD COLUMN IF NOT EXISTS pose_timeline_2d JSONB;

COMMENT ON COLUMN public.swing_videos.pose_timeline_2d IS
  'PR-4: 17-keypoint COCO frame-level timeline (see docs/decisions/PR-4_DESIGN.md). NULL for pre-PR-4 videos and for videos where MediaPipe failed to extract any landmarks.';
```

#### JSON shape (v1)

```json
{
  "version": 1,
  "fps_sampled": 10,
  "video_width": 720,
  "video_height": 1280,
  "keypoint_source": "mediapipe_pose",
  "yolo_anchor_correction": {
    "applied": true,
    "anchor_phases": ["setup", "top", "transition", "impact", "finish"],
    "method": "linear_per_segment"
  },
  "frames": [
    {
      "ts": 0.000,
      "frame_idx": 0,
      "interpolated": false,
      "keypoints": {
        "nose":          [360.4, 220.1, 0.94],
        "left_eye":      [368.2, 215.3, 0.91],
        "right_eye":     [352.7, 215.4, 0.89],
        "left_ear":      [380.5, 220.8, 0.78],
        "right_ear":     [340.1, 220.6, 0.81],
        "left_shoulder": [410.0, 290.0, 0.96],
        "right_shoulder":[310.0, 295.0, 0.93],
        "left_elbow":    [445.0, 360.0, 0.88],
        "right_elbow":   [275.0, 365.0, 0.86],
        "left_wrist":    [470.0, 430.0, 0.74],
        "right_wrist":   [255.0, 435.0, 0.71],
        "left_hip":      [395.0, 510.0, 0.92],
        "right_hip":     [325.0, 515.0, 0.90],
        "left_knee":     [400.0, 670.0, 0.84],
        "right_knee":    [330.0, 675.0, 0.83],
        "left_ankle":    [405.0, 820.0, 0.62],
        "right_ankle":   [335.0, 825.0, 0.61]
      }
    }
  ]
}
```

Field semantics:

- **`version: 1`** — schema version. Bump for breaking changes.
- **`fps_sampled`** — frames-per-second the timeline was built from.
  Current MediaPipe loop at `sample_fps=10.0` → `10`. See §I question 2.
- **`video_width`, `video_height`** — source video native pixel dims.
  Frontend SVG `viewBox` uses these so bbox coords map without manual
  scale math.
- **`keypoint_source`** — `"mediapipe_pose"` for PR-4. Reserved values
  for future: `"yolo"`, `"hybrid"` (e.g., wrist from MP, shoulder from
  YOLO).
- **`yolo_anchor_correction`** — present only when applied; describes
  which phases anchored the correction and what algorithm. When
  `applied: false`, the field still present so consumers can rely on
  its shape.
- **`frames[].ts`** — seconds from video start (same time axis as
  `phase_markers_json`).
- **`frames[].frame_idx`** — original frame index in the video stream.
  Lets future PRs cross-reference back to specific frames (e.g., for
  per-frame debugging).
- **`frames[].interpolated`** — `true` if this frame was filled by
  gap-filling (§C step 4) rather than direct detection. Frontend can
  show it differently (e.g., lower opacity) if desired.
- **`frames[].keypoints.{name}`** — `[x, y, conf]` in video native
  pixels. `null` keypoints (any of the 17) appear as `[null, null,
  conf]` where `conf` reflects the rejection reason (low MediaPipe
  visibility = original value < 0.3; outlier-rejected = 0.0).
- **Coordinates** — video native pixels, NOT normalised 0-1.
  Standardises the frontend SVG `viewBox` strategy across overlays.

#### Size estimate

10 fps × 3 s × 17 kp × ~14 bytes per coord triple ≈ **~7 KB / video**.
At 30 fps it's ~20 KB. Both trivial — Supabase JSONB column has no
relevant size pressure.

#### RLS

No new policy. Reads via existing `swing_videos` policy
(`user_id = auth.uid()`).

---

### B. Data-source strategy (MediaPipe primary + YOLO anchor correction)

#### Current state (verified by code exploration)

- `python/analyzer.py:128-160` `extract_landmarks()` — currently
  extracts only NOSE + 12 body joints (shoulders, elbows, wrists, hips,
  knees, ankles). Missing eyes (MP 2, 5) and ears (MP 7, 8).
- `python/analyzer.py:169-222` `analyze_video()` — MediaPipe loop,
  called from `main.py` with `sample_fps=10.0`.
- `python/sam3d/orchestrator.py` + `python/yolo/` — produce YOLO 17
  COCO keypoints for 5 phases (setup/top/transition/impact/finish),
  stored in `pose_3d_phases.yolo_keypoints_2d`.

#### MediaPipe → COCO subset mapping

The MediaPipe pose 33-point model is a superset; we extract the COCO
17 indices:

| COCO name | COCO idx | MediaPipe idx |
|---|---|---|
| nose | 0 | 0 |
| left_eye | 1 | 2 |
| right_eye | 2 | 5 |
| left_ear | 3 | 7 |
| right_ear | 4 | 8 |
| left_shoulder | 5 | 11 |
| right_shoulder | 6 | 12 |
| left_elbow | 7 | 13 |
| right_elbow | 8 | 14 |
| left_wrist | 9 | 15 |
| right_wrist | 10 | 16 |
| left_hip | 11 | 23 |
| right_hip | 12 | 24 |
| left_knee | 13 | 25 |
| right_knee | 14 | 26 |
| left_ankle | 15 | 27 |
| right_ankle | 16 | 28 |

#### YOLO anchor correction (when YOLO 5-phase data is available)

1. **Pick anchor frames** — for each of the 5 phases
   (`setup/top/transition/impact/finish`), find the MediaPipe frame
   whose `ts` is closest to the phase's `phase_markers_json` timestamp.
2. **Compute per-keypoint offset** — for each of the 17 keypoints, at
   each of the 5 anchor frames:
   ```
   offset_phase[i][kp] = yolo_kp[kp] - mediapipe_kp[kp]
   ```
   If either MP or YOLO has low confidence (`< 0.3`) for that
   keypoint at that anchor, the offset for `(phase=i, kp)` is marked
   as `None` and that anchor is skipped for that specific keypoint.
3. **Per-keypoint, per-segment linear lerp**:
   - Sort anchor timestamps `t_0 < t_1 < ... < t_4`.
   - For frames with `ts < t_0`: apply `offset_phase[0][kp]`.
   - For `t_i ≤ ts < t_{i+1}`: linearly interpolate from
     `offset_phase[i][kp]` to `offset_phase[i+1][kp]` by
     `(ts - t_i) / (t_{i+1} - t_i)`.
   - For frames with `ts ≥ t_4`: apply `offset_phase[4][kp]`.
   - If an anchor is `None` for a keypoint, skip it in the
     interpolation (use the neighbouring anchor's offset for the
     full segment).
4. **Apply** — `corrected_kp = mediapipe_kp + interpolated_offset`.
   Confidence value is preserved from MediaPipe; coordinates shift.
5. **Record** in JSON: `keypoint_source: "mediapipe_pose"` (we still
   identify the primary path even with correction), plus the
   `yolo_anchor_correction` block describing what was applied.

**Why linear per-segment**: simplest robust correction. The 5 anchors
are spaced ~0.5–1s apart in a 3s swing; linear lerp between them
produces sub-pixel deviation in the segment interiors that even
trained eyes can't see. Kalman or spline corrections are over-fit for
the budget.

#### Fallback (when YOLO 5-phase data missing)

If `pose_3d_phases` has no `yolo_keypoints_2d` rows for this video
(reanalyze failed, video predates PR-3, etc.):

- Pure MediaPipe path.
- JSON has `yolo_anchor_correction: { "applied": false, ... }`.
- Frontend skeleton overlay still works; just less precise on anchors.

---

### C. Data-quality infrastructure (ordered by impact)

#### 1. Per-keypoint confidence filtering

Apply at extraction time inside `extract_coco_subset_from_mediapipe`:

```
if mediapipe_landmark.visibility < 0.3:
    keypoint = [None, None, visibility]   # reject coords but log conf
else:
    keypoint = [round(x_px, 1), round(y_px, 1), round(visibility, 3)]
```

Frontend treats `[None, None, _]` as "don't draw this kp this frame";
the rest of the frame still renders.

#### 2. Outlier rejection

Per keypoint trajectory, across consecutive frames:

```
MAX_PIXEL_JUMP = max(100, video_width * 0.10)   # adaptive to resolution
for i in range(1, len(timeline)):
    if prev_kp not None and curr_kp not None:
        dx = curr_kp.x - prev_kp.x
        dy = curr_kp.y - prev_kp.y
        if sqrt(dx*dx + dy*dy) > MAX_PIXEL_JUMP:
            curr_kp = [None, None, 0.0]  # reject as outlier
```

Catches MediaPipe identity-swaps (e.g., left/right wrist confusion
during transition phase) without smoothing out legitimate fast motion.

#### 3. EMA smoothing (per keypoint trajectory)

```
alpha = 0.4   # default; see §Section 4 question 5
for each keypoint independently:
    valid_frames = [f for f in timeline if f.kp is not None]
    apply EMA over (x, y) only — confidence values pass through
    null frames are skipped (smoothing does not "fill in")
```

EMA is intentionally lightweight — heavier smoothing (Savitzky-Golay,
Kalman) is deferred to PR-5+ where individual visualisations may want
custom smoothing curves.

#### 4. Gap filling (linear interpolation, short gaps only)

```
MAX_GAP = 5   # frames; with fps=10 this is 0.5 seconds
for each keypoint trajectory:
    find runs of consecutive null frames bracketed by valid frames
    if run length <= MAX_GAP:
        linearly interpolate (x, y) and set conf to a fixed value
        (e.g., 0.5) — flags as "synthesised" data
        mark frame.interpolated = true
    else:
        leave nulls
```

The `interpolated: true` flag lets PR-5+ visualisations decide
per-overlay whether to use synthesised points (e.g., paths probably
want them, position markers probably don't).

#### 5. Final validation (gate before write)

```
def validate_timeline(tl) -> bool:
    if len(tl.frames) == 0:                              # no detections
        return False
    valid_count = sum(
        1 for f in tl.frames
        if sum(1 for kp in f.keypoints.values() if kp[0] is not None) >= 8
    )
    if valid_count / len(tl.frames) < 0.5:               # >50% frames must have
        return False                                      #  >= 8 valid kp
    return True
```

If validation fails, write `pose_timeline_2d = NULL`. Frontend treats
NULL as "skeleton overlay unavailable" and the toggle is disabled with
a tooltip ("Re-analyze the swing to enable skeleton view").

---

### D. Backend implementation

#### New module: `python/pose_timeline.py`

Public functions:

```python
def extract_coco_subset_from_mediapipe(
    mp_pose_landmarks,
    video_width: int,
    video_height: int,
) -> dict[str, list[float | None]]:
    """Returns {coco_name: [x_px, y_px, conf] | [None, None, conf]} for 17 names."""

def build_timeline_from_keypoint_frames(
    keypoint_frames: list[KeypointFrame],
    metadata: VideoMetadata,
    sample_fps: float,
) -> dict:
    """Walks the existing analyze_video output, produces a v1 timeline JSON."""

def apply_yolo_anchor_correction(
    timeline: dict,
    yolo_5_phases: dict[str, dict],     # phase_name -> {keypoints_2d, image_w/h, ...}
    phase_markers: dict[str, float],
) -> dict:
    """Mutates timeline.frames in place; sets yolo_anchor_correction.applied=True."""

def smooth_ema(timeline: dict, alpha: float = 0.4) -> dict:
def detect_outliers_and_reject(timeline: dict, max_pixel_jump_factor: float = 0.10) -> dict:
def gap_fill_linear(timeline: dict, max_gap: int = 5) -> dict:
def validate_timeline(timeline: dict) -> bool:
```

These compose as a pipeline. Order matters:

```
build → outlier_reject → smooth_ema → gap_fill → yolo_anchor_correct → validate
```

Outlier rejection BEFORE EMA so smoothing doesn't blur outliers into
neighbouring frames. Gap fill BEFORE YOLO correction so the
interpolation has clean data.

#### Changes to existing files

- `python/analyzer.py`:
  - Extend `LM` dict at L27 with `LEFT_EYE: 2, RIGHT_EYE: 5, LEFT_EAR: 7, RIGHT_EAR: 8`.
  - Extend `BodyLandmarks` dataclass with `leftEye, rightEye, leftEar, rightEar` (Optional[Point2D]).
  - Extend `extract_landmarks()` to fill the four new fields.
  - **No existing fields removed or renamed.**

- `python/main.py`:
  - After the existing `asyncio.gather(pose3d_for_all_phases, yolo_for_all_phases)` step (where YOLO 5-phase data becomes available), call the `pose_timeline` pipeline:
    ```python
    from pose_timeline import (
        build_timeline_from_keypoint_frames,
        detect_outliers_and_reject, smooth_ema, gap_fill_linear,
        apply_yolo_anchor_correction, validate_timeline,
    )
    tl = build_timeline_from_keypoint_frames(keypoint_frames, metadata, req.sample_fps)
    tl = detect_outliers_and_reject(tl)
    tl = smooth_ema(tl)
    tl = gap_fill_linear(tl)
    if yolo_summary and yolo_summary.get("results"):
        yolo_5_phases = collect_yolo_keypoints_per_phase(yolo_summary, phases)
        tl = apply_yolo_anchor_correction(tl, yolo_5_phases, phases)
    pose_timeline_2d = tl if validate_timeline(tl) else None
    ```
  - Add `poseTimeline2d` to the response payload alongside the existing
    `pose3dSummary` / `yoloSummary`.

- `python/yolo/orchestrator.py`:
  - Minor extension: the `summary["results"]` entries currently include
    `phase`, `status`, optional `error`. Add a `keypoints_2d` field on
    success so main.py can do anchor correction without re-reading the
    DB. Zero extra cost (we already have the data in memory).

#### Where it gets persisted

Same pattern as PR-2/PR-3: Python returns `poseTimeline2d` in the
`/analyze` response; Next.js writes it.

- `src/app/api/analyze/[id]/route.ts`:
  - In the existing `UPDATE swing_videos SET status='completed', ...`
    block, add `pose_timeline_2d: pythonResult.poseTimeline2d ?? null`.

No new service-role usage — the existing user-scoped server client
(already authenticated and ownership-verified earlier in the handler)
is fine for this column.

---

### E. Frontend implementation

#### New types (`src/types/analysis.ts` — augment existing file)

```ts
export type CocoKeypointName =
  | 'nose'
  | 'left_eye' | 'right_eye'
  | 'left_ear' | 'right_ear'
  | 'left_shoulder' | 'right_shoulder'
  | 'left_elbow'    | 'right_elbow'
  | 'left_wrist'    | 'right_wrist'
  | 'left_hip'      | 'right_hip'
  | 'left_knee'     | 'right_knee'
  | 'left_ankle'    | 'right_ankle';

export type Keypoint = readonly [number | null, number | null, number];
// [x_px, y_px, confidence]. x and y are null when rejected.

export type PoseFrame = {
  ts: number;
  frame_idx: number;
  interpolated: boolean;
  keypoints: Record<CocoKeypointName, Keypoint>;
};

export type PoseTimeline = {
  version: 1;
  fps_sampled: number;
  video_width: number;
  video_height: number;
  keypoint_source: 'mediapipe_pose' | 'yolo' | 'hybrid';
  yolo_anchor_correction: {
    applied: boolean;
    anchor_phases?: string[];
    method?: string;
  };
  frames: PoseFrame[];
};
```

`SwingVideoRow` is augmented to include
`pose_timeline_2d: PoseTimeline | null`. Existing codebase pattern is
inline `as X | null` casts at call sites; we follow that pattern (no
canonical row type yet — that's a future refactor).

#### New file: `src/lib/skeleton/coco.ts`

```ts
import type { CocoKeypointName } from '@/types/analysis';

export const COCO_KEYPOINT_NAMES: readonly CocoKeypointName[] = [
  'nose',
  'left_eye', 'right_eye',
  'left_ear', 'right_ear',
  'left_shoulder', 'right_shoulder',
  'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist',
  'left_hip', 'right_hip',
  'left_knee', 'right_knee',
  'left_ankle', 'right_ankle',
] as const;

// 17 standard COCO skeleton edges. Each pair is (from, to) keypoint names.
// Source: cocodataset.org/format-data, simplified for visual cleanliness.
export const COCO_SKELETON_EDGES: readonly [CocoKeypointName, CocoKeypointName][] = [
  // Head
  ['left_eye', 'nose'], ['right_eye', 'nose'],
  ['left_ear', 'left_eye'], ['right_ear', 'right_eye'],
  // Torso
  ['left_shoulder', 'right_shoulder'],
  ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'],
  ['left_hip', 'right_hip'],
  // Arms
  ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'],
  // Legs
  ['left_hip', 'left_knee'], ['left_knee', 'left_ankle'],
  ['right_hip', 'right_knee'], ['right_knee', 'right_ankle'],
];
```

#### New component: `src/components/SkeletonOverlay.tsx`

```tsx
'use client';
import { useEffect, useRef } from 'react';
import type { PoseTimeline, PoseFrame, CocoKeypointName } from '@/types/analysis';
import { COCO_KEYPOINT_NAMES, COCO_SKELETON_EDGES } from '@/lib/skeleton/coco';

type Props = {
  timeline: PoseTimeline;
  videoEl: HTMLVideoElement | null;
};

const HIGH_CONF = 0.7;
const MID_CONF = 0.3;
const COLOR_HIGH = '#CCCCCC';
const COLOR_MID  = '#666666';
const COLOR_EDGE = '#999999';

export function SkeletonOverlay({ timeline, videoEl }: Props) {
  const dotRefs    = useRef<Record<CocoKeypointName, SVGCircleElement | null>>({} as never);
  const edgeRefs   = useRef<Array<SVGLineElement | null>>([]);
  const lastValidFrameRef = useRef<PoseFrame | null>(null);

  useEffect(() => {
    if (!videoEl) return;
    let raf = 0;

    const loop = () => {
      const t = videoEl.currentTime;
      const frame = nearestFrame(timeline.frames, t) ?? lastValidFrameRef.current;
      if (frame) lastValidFrameRef.current = frame;

      if (frame) {
        // Dots
        for (const name of COCO_KEYPOINT_NAMES) {
          const dot = dotRefs.current[name];
          if (!dot) continue;
          const [x, y, conf] = frame.keypoints[name];
          if (x === null || y === null) {
            dot.setAttribute('visibility', 'hidden');
          } else {
            dot.setAttribute('cx', String(x));
            dot.setAttribute('cy', String(y));
            dot.setAttribute('fill', conf >= HIGH_CONF ? COLOR_HIGH : COLOR_MID);
            dot.setAttribute('visibility', 'visible');
          }
        }
        // Edges
        COCO_SKELETON_EDGES.forEach(([from, to], i) => {
          const line = edgeRefs.current[i];
          if (!line) return;
          const a = frame.keypoints[from];
          const b = frame.keypoints[to];
          if (a[0] === null || a[1] === null || b[0] === null || b[1] === null) {
            line.setAttribute('visibility', 'hidden');
          } else {
            line.setAttribute('x1', String(a[0]));
            line.setAttribute('y1', String(a[1]));
            line.setAttribute('x2', String(b[0]));
            line.setAttribute('y2', String(b[1]));
            line.setAttribute('visibility', 'visible');
          }
        });
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [videoEl, timeline]);

  return (
    <svg
      className="skeleton-overlay"
      viewBox={`0 0 ${timeline.video_width} ${timeline.video_height}`}
      preserveAspectRatio="xMidYMid meet"
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
    >
      {COCO_SKELETON_EDGES.map((_, i) => (
        <line
          key={`edge-${i}`}
          ref={el => { edgeRefs.current[i] = el; }}
          stroke={COLOR_EDGE} strokeWidth={2} opacity={0.7}
          visibility="hidden"
        />
      ))}
      {COCO_KEYPOINT_NAMES.map(name => (
        <circle
          key={`dot-${name}`}
          ref={el => { dotRefs.current[name] = el; }}
          r={5}
          stroke="rgba(0,0,0,0.5)" strokeWidth={1}
          visibility="hidden"
        />
      ))}
    </svg>
  );
}

function nearestFrame(frames: PoseFrame[], t: number): PoseFrame | null {
  if (!frames.length) return null;
  // Linear search ~30-150 entries — cheap enough; binary search optional.
  let best = frames[0], bestDist = Math.abs(t - best.ts);
  for (const f of frames) {
    const d = Math.abs(t - f.ts);
    if (d < bestDist) { best = f; bestDist = d; }
  }
  return best;
}
```

**Key design choices**:

- **SVG with `viewBox` in video native pixel space** + `preserveAspectRatio="xMidYMid meet"` — same letterbox semantics as `<video> object-fit:contain`. Coords flow directly from `pose_timeline_2d.frames[].keypoints` to `<rect>` / `<line>` / `<circle>` attributes without any manual scale math. Reusable across PR-5+.
- **rAF + ref `setAttribute`** — no React state in the animation loop; 60 Hz updates with zero re-renders. Same architectural pattern proposed in v1's HeadBoxOverlay.
- **Carry-forward** — `lastValidFrameRef` keeps the last successfully-rendered frame visible if the current frame is `null` (transient gaps). Prevents flicker.

#### Changes to `src/components/SwingPlayer.tsx`

Add a state toggle for skeleton overlay (default off) and a button.
Layer order inside `.sp-vw`:

```
<video>                      (background)
<canvas>                     (PR-3 disc overlay, when timeline present)
<SkeletonOverlay />          (NEW, when toggle on + pose_timeline_2d present)
<sp-badges>                  (top-left labels)
<sp-layer-badge>             (top-right "ALL" badge)
<sp-legend>                  (bottom-right)
<sp-tap />                   (click capture, pointer-events:auto)
```

Toggle button location: small icon-only button in the existing `.sp-layers` row or as a new badge-style toggle in the corner. **Recommend**: place adjacent to the existing layer-toggle bar (Body / Hands / Club / All) so users discover it naturally.

When `pose_timeline_2d === null`, the toggle button is **disabled** with a tooltip: "Re-analyze this swing to enable skeleton view."

#### Changes to `src/app/result/[id]/page.tsx`

- Read `pose_timeline_2d` from `vid` and pass into `<SwingPlayer poseTimeline={...} />`.
- **No other UI changes in PR-4.** Coaching-bar text collapse stays deferred to PR-5 (when single-issue overlays land).

---

### F. UI changes summary (PR-4 only)

| Area | Change |
|---|---|
| Skeleton overlay | NEW; default off; toggle in player controls |
| Disc overlay (PR-3) | Unchanged |
| Phase tabs | Unchanged |
| Coaching bar (issue / cue / drill) | **Unchanged in PR-4** — deferred to PR-5 |
| Badge (data source) | Unchanged (still shows YOLO 11m / SAM 3D / etc.) |

Rationale: PR-4 is data-foundation + one visible debug overlay. UI
density / collapse work belongs with PR-5 because that's when the
single-issue overlays start competing for screen space.

---

### G. Commit split (6 commits, single push at end)

| # | Message | Files |
|---|---|---|
| 1 | `feat(db): add pose_timeline_2d JSONB column to swing_videos` | `supabase/migrations/{TS}_add_pose_timeline_2d.sql` |
| 2 | `feat(analyzer): MediaPipe → COCO 17 subset + per-frame extraction` | `python/analyzer.py` (extend `LM`, `BodyLandmarks`, `extract_landmarks`), `python/pose_timeline.py` (new module — extract + build) |
| 3 | `feat(analyzer): data-quality pipeline (outlier reject, EMA, gap fill, validate)` | `python/pose_timeline.py` (smooth/outlier/gapfill/validate helpers) |
| 4 | `feat(analyzer): YOLO anchor correction + main.py integration` | `python/pose_timeline.py` (apply_yolo_anchor_correction), `python/main.py` (pipeline call + response field), `python/yolo/orchestrator.py` (return keypoints in summary), `src/app/api/analyze/[id]/route.ts` (persist column) |
| 5 | `feat(frontend): PoseTimeline + Keypoint types + COCO skeleton constants` | `src/types/analysis.ts`, new `src/lib/skeleton/coco.ts` |
| 6 | `feat(frontend): SkeletonOverlay component + toggle in SwingPlayer` | new `src/components/SkeletonOverlay.tsx`, `src/components/SwingPlayer.tsx`, `src/app/result/[id]/page.tsx` (pass timeline prop down) |

All 6 push together (matches PR-3 / PR-2 deployment discipline).
Commit 4 is the integration commit — pre-merge testing locally with
the test video confirms end-to-end pipeline before pushing.

---

### H. Risks + mitigations

| # | Risk | Mitigation |
|---|---|---|
| 1 | MediaPipe wrist/ankle precision degrades during fast motion (downswing) | Outlier rejection catches the worst frames; EMA smoothing damps the residual; gap-fill bridges short null runs. |
| 2 | YOLO anchor correction overcorrects (offset large at one phase, small at another) | Per-segment linear lerp keeps the correction bounded by the two surrounding anchors. Tested visually in PR-4 implementation against PR-2B test video. |
| 3 | YOLO 5-phase data missing (re-analysis failed, or video predates PR-3) | Fallback: pure MediaPipe path; `yolo_anchor_correction.applied=false`. Skeleton overlay still renders. |
| 4 | Validation fails → `pose_timeline_2d = NULL` | Frontend toggle disabled with tooltip. No 500, no crash. |
| 5 | JSON size at 30 fps × longer video | 30 fps × 5 s = 150 frames ≈ 35 KB — still fine. Supabase JSONB has no soft cap relevant here. |
| 6 | rAF loop continues after unmount | `useEffect` cleanup `cancelAnimationFrame(raf)` — standard pattern. |
| 7 | YOLO `image_width/height` ≠ MediaPipe `video_width/height` (e.g., YOLO ran on a downscaled frame) | YOLO orchestrator passes through the source PNG dims; `extract_frame` produces native-resolution PNGs (verified earlier). Both pipelines see the same source dims. No scaling needed. **Sanity-check at correction time**: assert YOLO row's `image_width == timeline.video_width`; on mismatch, skip correction for that anchor and log a warning. |
| 8 | Adding eyes/ears to `BodyLandmarks` breaks downstream code that destructures the dataclass | Search `BodyLandmarks` references: only `analyzer.py` and `phase_detector.py` use it; both iterate over known field names. Additive fields safe. |

### Explicit non-risks

- **PR-4 doesn't change disc rendering** — PR-3's canvas overlay stays
  intact and reads from `pose_3d_phases.yolo_keypoints_2d` as before.
- **PR-4 doesn't introduce SVG-vs-canvas migration** — skeleton uses
  SVG, disc still uses canvas. Future PRs may unify.
- **PR-4 doesn't change phase detection** — `phase_markers_json` still
  authored by `phase_detector.py`. Timeline merely consumes it for
  anchor selection.

---

### I. Decision questions for Jason

1. **Skeleton overlay default state** — confirm **off** (toggle to
   enable) as designed? OR show by default on the first render after
   each new analysis?
2. **Sample rate** — keep **10 fps** for MVP, or bump to **30 fps**
   to surface micro-motion (e.g., wrist hinge) at the cost of ~3×
   MediaPipe wall time and ~3× JSON size?
3. **MediaPipe Hands** — defer to **PR-8** (current plan), or add 21
   hand keypoints per side to **PR-4** now? Adds schema complexity
   (`hands_left[21], hands_right[21]`) but PR-8 is just a renderer if
   the data is already there.
4. **YOLO anchor correction in PR-4** — include now (clean
   per-segment lerp implementation, ~50 LoC), or **defer** to a later
   PR for a pure-MediaPipe v1 that ships faster?
5. **EMA alpha = 0.4** — accept, or prefer **0.3** (smoother but more
   lag) / **0.5** (less smooth, more reactive)?
6. **SVG viewBox / SkeletonOverlay coordinate convention** — confirm
   we standardise on **video native pixel space** as the canonical
   coordinate system for all future overlays (skeleton, future
   discs, paths, markers)?

---

## Section 4 — Quick-look decision highlights

For rapid Jason review:

- ✅ **17 COCO keypoints frame-level timeline** persisted on
  `swing_videos.pose_timeline_2d` (JSONB, nullable, additive only).
- ✅ **MediaPipe primary path** (free data — already running for
  phase detection) + **YOLO 5-phase anchor correction** (PR-3 data
  reused, no extra inference cost).
- ✅ **Data-quality pipeline**: confidence filter → outlier reject →
  EMA smooth → gap fill → YOLO correct → validate. Each step
  documented in §C.
- ✅ **SkeletonOverlay** = PR-4's only visible deliverable, behind a
  toggle (default off). SVG-based, rAF-driven, no React re-render
  during playback.
- ✅ **No UI text changes** in PR-4 — coaching bar collapse moves to
  PR-5 where single-issue overlays land.
- ✅ **Additive only**: schema, MediaPipe extraction (4 new fields),
  YOLO orchestrator (1 new field in summary). Zero existing code
  paths changed.
- ✅ **Multi-PR roadmap recorded** (Section 2): PR-4 unlocks PR-5
  through PR-11; future implementers can read this doc and see the
  full shape.
- ✅ **Versioned JSON shape** (`version: 1`) — future PRs can bump
  schema without breaking existing readers.
- ⚠️ **Open decisions**: §Section 3.I lists 6 questions; the biggest
  ones are sample rate (10 vs 30 fps) and whether YOLO anchor
  correction lands in PR-4 or a follow-up.

⏸ **STOP** — awaiting your answers on Section 3.I (or "all good,
proceed to Step 3"), then I'll implement the 6-commit batch with one
push at the end.
