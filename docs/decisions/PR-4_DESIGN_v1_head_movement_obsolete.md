> ⚠️ **OBSOLETE — superseded by [`PR-4_DESIGN.md`](./PR-4_DESIGN.md).**
>
> The original `head_movement`-only scope was re-framed by Jason into
> the broader **"全身关键点数据基础 + skeleton 可视化"** path:
> build the full 17-keypoint frame-level timeline (MediaPipe primary
> path + YOLO 5-phase anchor correction + data-quality infrastructure)
> as PR-4's actual deliverable, with skeleton overlay as the only
> visible artefact. Single-issue overlays (rotation, sway, etc.) move
> to PR-5+, all built on top of the PR-4 timeline.
>
> This document is preserved as a decision-evolution archive — useful
> context for why the v2 design exists and what the earlier
> single-issue framing got right vs. wrong. **Do not implement from
> this file. Read PR-4_DESIGN.md instead.**

---

# PR-4 Design v1 (OBSOLETE): head_movement Visual MVP

**Status:** Superseded by v2
**Date:** 2026-05-16
**Scope:** ONE issue visualisation (`head_movement`) as the template
that future issue-specific overlays will follow. Other issue types are
out of scope for this PR.

---

## 0. Decisions in (already approved by Jason)

| # | Decision |
|---|---|
| 1 | Scope = `head_movement` only; treat as the MVP template. |
| 2 | Architecture = frame-level head bbox timeline persisted on `swing_videos.pose_timeline_2d` (new JSONB column). |
| 3 | UI text = `cue_text` + `drill_text` collapsed by default; expand reveals one ≤60-char summary line (not the full cue/drill text). |
| 4 | Head box geometry = square; side length = `|ear_left.x − ear_right.x| × 1.5`; centred on nose. |
| 5 | Green "ideal" box = locked to head position at `setup` phase. |

---

## A. Schema change (additive only)

### File

`supabase/migrations/{TS}_add_pose_timeline_2d.sql`
TS to be generated at implementation time with `Get-Date -Format "yyyyMMddHHmmss"`.

### Migration SQL

```sql
-- ===========================================================================
-- PR-4: pose_timeline_2d — per-frame head bbox timeline for head_movement
-- ===========================================================================
-- One JSONB column on swing_videos. NULL for videos analysed before PR-4
-- (no migration backfill); frontend treats NULL as "no timeline → no head
-- box overlay" and falls back to the existing coaching-bar UI.
--
-- Additive only:
--   - swing_videos columns / RLS / triggers unchanged elsewhere.
--   - No new tables, no FK changes.
--
-- Reader-side trust: this column is read by the result page through the
-- existing user-scoped browser anon client. The current swing_videos RLS
-- policy (user_id = auth.uid()) is the only access control needed.
-- ===========================================================================

ALTER TABLE public.swing_videos
  ADD COLUMN IF NOT EXISTS pose_timeline_2d JSONB;

COMMENT ON COLUMN public.swing_videos.pose_timeline_2d IS
  'PR-4: per-frame head bbox timeline (see docs/decisions/PR-4_DESIGN.md). NULL for pre-PR-4 videos and for videos where MediaPipe failed to extract any head landmarks.';
```

### JSON shape (version 1)

```json
{
  "version": 1,
  "fps_sampled": 10,
  "video_width": 720,
  "video_height": 1280,
  "frames": [
    { "ts": 0.000, "head_bbox": [330, 180, 390, 240] },
    { "ts": 0.100, "head_bbox": [331, 181, 391, 241] },
    { "ts": 0.200, "head_bbox": null },
    ...
  ]
}
```

- **`version`** — schema version. Future shape changes bump this.
- **`fps_sampled`** — frames-per-second the timeline was built from. With
  the current MediaPipe loop at `sample_fps=10.0`, this is `10`. (See §B
  for the alternative of bumping to 30; not recommended for MVP.)
- **`video_width / video_height`** — source video native pixel dims.
  Used by the frontend SVG `viewBox` so bbox coords are interpreted in
  the same space they were computed.
- **`frames[].ts`** — seconds from video start (matches the same time
  axis used by `phase_markers_json`).
- **`frames[].head_bbox`** — `[x1, y1, x2, y2]` in video native pixels,
  square, computed per §B. **`null`** when any of the 3 source
  landmarks (nose / left_ear / right_ear) was below MediaPipe's 0.5
  visibility threshold for that frame.

### Size estimate

10 fps × 3 s swing × ~40 bytes per row ≈ **1.2 KB / video**. Trivial.

### RLS

No new policy. Reads via `auth.uid() = user_id` already on `swing_videos`.

---

## B. Backend change (`python/`)

### Existing pipeline (verified by exploration)

```
python/analyzer.py:128  extract_landmarks(result, conf_threshold=0.3)
                        # currently fills head = NOSE (idx 0) only
python/analyzer.py:169  analyze_video(video_path, sample_fps=4.0)
                        # MediaPipe loop; main.py calls with sample_fps=10.0
                        # loops at frame_interval = int(video_fps / sample_fps)
                        # pose.process(rgb) on every Nth frame
```

### Change: extend BodyLandmarks + new per-frame head bbox

1. Add two fields to `BodyLandmarks` (the dataclass at `analyzer.py:50`):
   ```python
   leftEar:  Optional[Point2D] = None
   rightEar: Optional[Point2D] = None
   ```
2. Extend `extract_landmarks()` (`analyzer.py:128-160`) with two more
   `pt()` calls using MediaPipe pose indices `7` (LEFT_EAR) and `8`
   (RIGHT_EAR). Add the two entries to the `LM` dict at `analyzer.py:27`.
3. **No existing fields modified.** Pure addition.
4. Per-frame head_bbox computation lives in a new helper:
   ```python
   def compute_head_bbox(
       lm: BodyLandmarks,
       video_width: int,
       video_height: int,
   ) -> Optional[list[float]]:
       """
       Returns [x1, y1, x2, y2] in video native pixel space, or None when
       nose / left_ear / right_ear are not all confidently detected.
       """
       if lm.head is None or lm.leftEar is None or lm.rightEar is None:
           return None
       # MediaPipe coords are normalised 0-1 — convert to pixels.
       lx = lm.leftEar.x  * video_width
       rx = lm.rightEar.x * video_width
       nx = lm.head.x     * video_width
       ny = lm.head.y     * video_height
       ears_dist_px = abs(lx - rx)
       size = ears_dist_px * 1.5
       half = size / 2.0
       # Clamp to image bounds so the SVG doesn't render off-canvas.
       x1 = max(0.0,            nx - half)
       y1 = max(0.0,            ny - half)
       x2 = min(video_width  - 1, nx + half)
       y2 = min(video_height - 1, ny + half)
       return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
   ```

### Where to assemble `pose_timeline_2d`

Add a top-level helper in `analyzer.py` (or new `head_timeline.py`)
that walks `keypoint_frames` post-extraction:

```python
def build_pose_timeline_2d(
    keypoint_frames: list[KeypointFrame],
    metadata: VideoMetadata,
    sample_fps: float,
) -> dict:
    frames = []
    for kf in keypoint_frames:
        bbox = compute_head_bbox(kf.landmarks, metadata.width, metadata.height)
        frames.append({"ts": kf.time, "head_bbox": bbox})
    return {
        "version": 1,
        "fps_sampled": int(round(sample_fps)),
        "video_width": metadata.width,
        "video_height": metadata.height,
        "frames": frames,
    }
```

### Where to plug it into `/analyze`

`python/main.py:80-145` already orchestrates MediaPipe → phase detect
→ SAM/YOLO parallel. The new step is a pure post-processing of the
keypoint_frames we already have:

```python
# After "phases = detect_phases(...)" but before / parallel with the
# pose3d+yolo asyncio.gather:
pose_timeline_2d = build_pose_timeline_2d(
    keypoint_frames, metadata, sample_fps=req.sample_fps,
)
```

Then add to the response:

```python
return {
    ...,
    "poseTimeline2d": pose_timeline_2d,
}
```

### Where it gets written to Supabase

**Next.js**, not Python. The current architecture has Python writing
`pose_3d_phases` directly (service-role httpx) and Next.js writing
`swing_analysis` + `swing_videos.status`. For consistency, pose timeline
follows the swing_videos-owning path — Next.js writes it.

`src/app/api/analyze/[id]/route.ts` currently UPDATEs `swing_videos`
with `status='completed'` after analysis finishes. Same place adds:

```ts
await supabase
  .from('swing_videos')
  .update({
    status: 'completed',
    processing_completed_at: new Date().toISOString(),
    pose_timeline_2d: pythonResult.poseTimeline2d ?? null,
  })
  .eq('id', videoId);
```

This uses the existing user-scoped server client (already authenticated
+ ownership-verified earlier in the handler), no service-role needed.

### Sampling rate — 10 fps stays for MVP

- Bumping to 30 fps would 3× the MediaPipe wall time (currently ~3-5s
  on Railway CPU). That increases /analyze latency proportionally.
- 10 fps → head box visually updates every 100 ms. The head moves
  slowly relative to the wrists; 10 Hz looks smooth.
- If user feedback says it stutters, a future PR can either (a) bump
  the analyzer-wide sample_fps, or (b) interpolate between adjacent
  bboxes in the frontend rAF loop. **Out of scope here.**

### Risk

| Risk | Mitigation |
|---|---|
| MediaPipe ear visibility < 0.5 (profile angle, hat, sun glare) | `head_bbox: null` for that frame; frontend skips rendering on null frames; the previous frame's box stays drawn (carry-forward — see §C). |
| All frames null (e.g., very low resolution video) | `pose_timeline_2d.frames` ends up all-null; frontend treats `setupHeadBbox === null` as "no overlay" and falls back to coaching-bar-only display. |

---

## C. Frontend change (`src/`)

### Files touched

| Path | Change |
|---|---|
| `src/types/analysis.ts` (or new `src/lib/headOverlay/types.ts`) | +`PoseTimeline` type; +`PoseTimelineFrame` type |
| `src/lib/sam3d/keypoints.ts` | (no change — `PoseRow` already covers YOLO setup data) |
| `src/components/HeadBoxOverlay.tsx` | **NEW** — SVG layer with green ideal box + red current box |
| `src/components/SwingPlayer.tsx` | mount `<HeadBoxOverlay>` inside `.sp-vw` above the canvas, only when `issueType === 'head_movement'` and timeline is present |
| `src/app/result/[id]/page.tsx` | collapse cue/drill behind Details button; pass `issueType` + `poseTimeline2d` + `setupHeadBbox` props down to SwingPlayer |
| `src/components/SwingPlayer.tsx` `Props` interface | new optional `headBox?: { timeline: PoseTimeline; setupBbox: [number,number,number,number] | null; videoWidth: number; videoHeight: number }` |

### New types

```ts
// src/lib/headOverlay/types.ts (recommended new file)
export type HeadBbox = readonly [number, number, number, number]; // x1, y1, x2, y2

export type PoseTimelineFrame = {
  ts: number;                  // seconds
  head_bbox: HeadBbox | null;  // null = low confidence this frame
};

export type PoseTimeline = {
  version: 1;
  fps_sampled: number;
  video_width: number;
  video_height: number;
  frames: PoseTimelineFrame[];
};
```

### `HeadBoxOverlay.tsx` design

```tsx
'use client';

import { useEffect, useRef } from 'react';
import type { PoseTimeline, HeadBbox } from '@/lib/headOverlay/types';

type Props = {
  timeline: PoseTimeline;
  videoEl: HTMLVideoElement | null;
  setupHeadBbox: HeadBbox;          // green ideal box
};

export function HeadBoxOverlay({ timeline, videoEl, setupHeadBbox }: Props) {
  const currentRectRef = useRef<SVGRectElement | null>(null);
  const lastValidBoxRef = useRef<HeadBbox | null>(null);

  useEffect(() => {
    if (!videoEl) return;
    let raf = 0;
    const loop = () => {
      const t = videoEl.currentTime;
      const frame = nearestFrame(timeline.frames, t);
      const bbox = frame?.head_bbox ?? lastValidBoxRef.current;
      // Carry-forward: keep last valid box drawn if current frame is null.
      if (frame?.head_bbox) lastValidBoxRef.current = frame.head_bbox;
      const rect = currentRectRef.current;
      if (rect && bbox) {
        const [x1, y1, x2, y2] = bbox;
        rect.setAttribute('x', String(x1));
        rect.setAttribute('y', String(y1));
        rect.setAttribute('width', String(x2 - x1));
        rect.setAttribute('height', String(y2 - y1));
        rect.setAttribute('visibility', 'visible');
      } else if (rect) {
        rect.setAttribute('visibility', 'hidden');
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [videoEl, timeline]);

  const [gx1, gy1, gx2, gy2] = setupHeadBbox;

  return (
    <svg
      className="head-box-overlay"
      viewBox={`0 0 ${timeline.video_width} ${timeline.video_height}`}
      preserveAspectRatio="xMidYMid meet"
      style={{
        position: 'absolute', inset: 0,
        width: '100%', height: '100%',
        pointerEvents: 'none',
      }}
    >
      {/* Green dashed: ideal head position locked from setup */}
      <rect
        x={gx1} y={gy1} width={gx2 - gx1} height={gy2 - gy1}
        stroke="#3cee3c" strokeWidth={3} strokeDasharray="8,6"
        fill="none"
      />
      {/* Red solid: current head position (updated by rAF, not React) */}
      <rect
        ref={currentRectRef}
        stroke="#ff3c3c" strokeWidth={3}
        fill="none"
        visibility="hidden"
      />
    </svg>
  );
}

function nearestFrame(frames: PoseTimelineFrame[], t: number) {
  // Frames are time-ascending; binary search optional. Linear is fine
  // for ~30 frames.
  if (!frames.length) return null;
  if (t <= frames[0].ts) return frames[0];
  if (t >= frames[frames.length - 1].ts) return frames[frames.length - 1];
  let best = frames[0];
  let bestDist = Math.abs(t - best.ts);
  for (const f of frames) {
    const d = Math.abs(t - f.ts);
    if (d < bestDist) { best = f; bestDist = d; }
  }
  return best;
}
```

**Key design choices**:

- **SVG with `viewBox` in video pixel space** + `preserveAspectRatio="xMidYMid meet"` — the SVG auto-letterboxes the same way the `<video>` element does (`object-fit: contain`), so bbox coords map directly to visible pixels without any manual scale-or-pad math. This is **structurally cleaner than the canvas overlay** PR-3 uses; if it works well we may migrate disc rendering to SVG in a future refactor.
- **rAF loop instead of React state** — `videoEl.currentTime` updates ~60Hz during playback. Setting React state at that rate causes ~3% wasted re-renders per second. Using `ref.current.setAttribute()` bypasses React entirely.
- **Carry-forward last valid box** — if a frame has `null` head_bbox (low MediaPipe confidence), keep the last good box drawn. Prevents flicker.
- **No animation, no smoothing for MVP** — head moves slowly; 10 Hz updates look natural. Smoothing/interp can be added later if needed.

### `SwingPlayer.tsx` integration

Inside `.sp-vw`, between `<canvas>` and `.sp-badges`:

```tsx
{headBox && headBox.setupBbox && (
  <HeadBoxOverlay
    timeline={headBox.timeline}
    videoEl={videoRef.current}
    setupHeadBbox={headBox.setupBbox}
  />
)}
```

Layer order (already z-correct, no CSS change needed):
1. `<video>` (background)
2. `<canvas>` (disc overlay — PR-3)
3. `<svg>` (head box overlay — PR-4) ← NEW
4. `.sp-badges`, `.sp-legend`, `.sp-layer-badge` (top labels)
5. `.sp-tap` (click target — `position: absolute, inset: 0, z-index: 2`)

The existing `.sp-tap` has `z-index: 2` for click capture. The new SVG
has `pointer-events: none` so it doesn't steal clicks.

### Coaching bar collapse (`src/app/result/[id]/page.tsx`)

Current (`L186-195`):

```tsx
<div className="coaching-bar">
  <div className="issue-row"><span className="issue-dot">⚡</span><span className="issue-text">{issueLabel}</span></div>
  <div className="cue-row"><span className="cue-quote">&ldquo;{cue}&rdquo;</span></div>
</div>
```

After PR-4:

```tsx
<div className="coaching-bar">
  <div className="issue-row">
    <span className="issue-dot">⚡</span>
    <span className="issue-text">{issueLabel}</span>
    <button
      className="details-toggle"
      onClick={() => setDetailsOpen(o => !o)}
      aria-expanded={detailsOpen}
    >
      Details {detailsOpen ? '▴' : '▾'}
    </button>
  </div>
  {detailsOpen && (
    <div className="details-row">
      <span className="details-text">{shortSummary(cue)}</span>
    </div>
  )}
</div>
```

`shortSummary()` helper:

```ts
function shortSummary(cue: string): string {
  if (!cue) return '';
  if (cue.length <= 60) return cue;
  // Trim to last sentence/clause boundary within 60 chars; fall back to
  // hard slice + ellipsis.
  const slice = cue.slice(0, 60);
  const lastSentence = slice.lastIndexOf('.');
  const lastComma    = slice.lastIndexOf(',');
  const cut = Math.max(lastSentence, lastComma);
  if (cut > 30) return slice.slice(0, cut + 1).trim();
  return slice.trim() + '…';
}
```

Notes:
- `summary_text` and `drill_text` are **not displayed** — they stay in
  `swing_analysis` for future / coaching screens.
- The `cue-quote` styling currently always shows; that gets removed.
- Existing CSS classes `.cue-row` / `.cue-quote` can be repurposed for
  `.details-row` / `.details-text` (no new style block) — keeps the diff
  small.

---

## D. TypeScript types — full surface

### `src/lib/headOverlay/types.ts` (new)

(see §C — `HeadBbox`, `PoseTimelineFrame`, `PoseTimeline`)

### Augment existing `SwingVideo` shape

The codebase doesn't have a canonical `SwingVideoRow` type today
(rows are typed inline via `ana.* as Type | null` casts at the call
sites). We can either:

**(a)** Keep the inline cast pattern — at the result page load:
```ts
const poseTimeline = vid.pose_timeline_2d as PoseTimeline | null;
```

**(b)** Introduce a canonical `SwingVideoRow` type in
`src/types/analysis.ts` matching the actual table:
```ts
export type SwingVideoRow = {
  id: string;
  user_id: string;
  status: 'uploaded' | 'processing' | 'completed' | 'failed';
  storage_path: string;
  view_type: 'face_on' | 'down_the_line';
  // ... existing columns
  pose_timeline_2d: PoseTimeline | null;
};
```

**Recommend (a) for MVP** — matches the existing pattern, no risk of
type drift from missing existing columns. (b) is a future refactor.

---

## E. Setup head_bbox source — two options

### Option 1: Frontend derives from `pose_3d_phases.yolo_keypoints_2d` setup row (recommended)

The frontend already loads YOLO keypoints for the disc overlay via
`fetchPoseRows(videoId)` (PR-2C). The setup-phase row has 17 COCO
keypoints; head landmarks at COCO indices `0` (nose), `3` (left_ear),
`4` (right_ear).

```ts
function deriveSetupHeadBbox(
  yoloKps: number[][],
  imgW: number,
  imgH: number,
  minConf = 0.3,
): HeadBbox | null {
  if (!yoloKps || yoloKps.length < 5) return null;
  const [nx, ny, nc] = yoloKps[0];    // nose
  const [lx, ly, lc] = yoloKps[3];    // left ear
  const [rx, ry, rc] = yoloKps[4];    // right ear
  if (nc < minConf || lc < minConf || rc < minConf) return null;
  const earsDist = Math.abs(lx - rx);
  const size = earsDist * 1.5;
  const half = size / 2;
  return [
    Math.max(0, nx - half),
    Math.max(0, ny - half),
    Math.min(imgW - 1, nx + half),
    Math.min(imgH - 1, ny + half),
  ] as const;
}
```

**Pros**:
- Zero extra backend work — YOLO already provides anatomical head kp.
- More accurate than MediaPipe (COCO ears are sub-pixel-accurate).
- Computed once per session in the React `load()`, cached in state.

**Cons**:
- Depends on YOLO succeeding for the setup row. If `yolo_keypoints_2d`
  is NULL there, this option produces NULL → fall back to Option 2.

### Option 2: Backend averages first N MediaPipe frames before `setup_phase_end`

In `build_pose_timeline_2d`, compute and add a top-level
`setup_ideal_bbox` field by averaging the first frames before
`phaseMarkers.setupTime + 0.5s` (or before the top phase begins).

**Pros**:
- Single source of truth on the backend; frontend reads it directly.
- Works even if YOLO didn't run for setup.

**Cons**:
- Averaging may smear if the golfer adjusts stance during setup window.
- Needs an extra JSON field that the version=1 schema must include.
- MediaPipe head landmarks (which is what we're averaging) are less
  precise than YOLO's.

### Recommendation

**Option 1 with Option 2 as the fallback**. Frontend logic:

```ts
const setupBbox = (
  setupRow?.yolo_keypoints_2d
    ? deriveSetupHeadBbox(setupRow.yolo_keypoints_2d, setupRow.image_width, setupRow.image_height)
    : null
) ?? deriveFromMediaPipeFirstFrames(poseTimeline2d);
```

If both return null, the head box overlay is hidden and the page shows
just the coaching bar. Acceptable degraded state.

`deriveFromMediaPipeFirstFrames` is a small helper inside
`HeadBoxOverlay.tsx` or `headOverlay/utils.ts` — pure frontend, no
backend change required for the fallback.

---

## F. Commit split (5 commits, one push at end)

| # | Message | Files |
|---|---|---|
| 1 | `feat(db): add pose_timeline_2d JSONB column to swing_videos` | `supabase/migrations/{TS}_add_pose_timeline_2d.sql` |
| 2 | `feat(analyzer): extract per-frame head bbox from MediaPipe ears+nose` | `python/analyzer.py` (extend `BodyLandmarks`, `extract_landmarks`, add `compute_head_bbox` + `build_pose_timeline_2d`), `python/main.py` (call new helper, add `poseTimeline2d` to response) |
| 3 | `feat(api): persist pose_timeline_2d on swing_videos completion` | `src/app/api/analyze/[id]/route.ts` (one UPDATE field added) |
| 4 | `feat(frontend): PoseTimeline types + HeadBoxOverlay SVG component` | new `src/lib/headOverlay/types.ts`, new `src/components/HeadBoxOverlay.tsx`, `src/components/SwingPlayer.tsx` (mount overlay when head_movement issue + timeline present) |
| 5 | `feat(frontend): collapse cue/drill behind Details toggle on result page` | `src/app/result/[id]/page.tsx` (state + button + shortSummary helper), CSS class additions |

Push all 5 after Step 3 finishes (PR-3 pattern).

Commit 5 can land independently of 1–4 because the UI change is pure
display logic. But we ship them together to keep the feature coherent
in history.

---

## G. Risks + mitigations

| # | Risk | Mitigation |
|---|---|---|
| 1 | MediaPipe ear visibility < 0.5 on whole video (profile shot, hat) → all `head_bbox: null` | `pose_timeline_2d.frames` still saved; frontend detects "no valid frame" and hides overlay. Coaching bar collapse still works — just no red box. |
| 2 | Setup phase has no YOLO row (YOLO failed for setup) AND first MediaPipe frames also null | Frontend detects null setup bbox → hides green ideal box too. Page renders cleanly with just video + coaching bar. **No 500, no exception.** |
| 3 | `pose_timeline_2d` ends up NULL on old videos | NULL is the expected sentinel; frontend conditional `headBox && headBox.setupBbox &&` guards the render. |
| 4 | JSON size if user uploads 30-second swing instead of 3s | 30 s × 10 fps × 40 B = 12 KB — still well under any practical limit. |
| 5 | rAF loop continues after component unmount | `useEffect` cleanup returns `cancelAnimationFrame(raf)`. Standard React pattern. |
| 6 | Letterbox math drift if `<video>` `object-fit` changes in future CSS work | SVG `viewBox + preserveAspectRatio` independently letterboxes the same way; same change of `<video>` behaviour would require matching SVG `preserveAspectRatio` tweak. Both live in the same component file — refactor will touch both together. |
| 7 | Backend MediaPipe loop already running ~3-5s; new head_bbox compute adds negligible time (it's pure Python over the in-memory list) | No latency budget concern. |
| 8 | Bumping sample_fps later for smoother UX | Versioned JSON (`version: 1`); v2 can change `fps_sampled` or add interpolation hints without breaking existing readers. |

### Explicit non-risks (called out to prevent future second-guessing)

- **Over-engineering**: `head_bbox` is `number[4]` (NOT nested object).
  This was an explicit decision per PR-4 spec.
- **Coordinate space**: video native pixels. NOT normalised 0-1.
  Frontend SVG `viewBox` handles scaling.
- **Smoothing**: deliberately not implemented for MVP. Re-evaluate after
  user testing.

---

## H. Out of scope (explicit non-goals)

- Visualising issues other than `head_movement`. Other issues continue
  to render the existing disc-only overlay.
- Numerical "head moved X pixels" metric in the coaching bar.
- Re-analysing pre-PR-4 videos to backfill `pose_timeline_2d`. NULL
  stays NULL; frontend degrades gracefully.
- Migrating disc overlay from canvas to SVG (interesting follow-up
  refactor but not required for PR-4).
- Adding a `summary_text` / `drill_text` display path. Both still
  exist in `swing_analysis` but neither is rendered after PR-4.

---

## I. Approval questions

Before Step 3 implementation:

1. **JSON shape v1** — locked as shown in §A, or any field you want
   added/renamed?
2. **Sampling rate** — stay at 10 fps for MVP, defer 30 fps to a follow-up?
3. **Setup bbox** — Option 1 (frontend derives from YOLO setup row) +
   Option 2 fallback, OK? Or do you want backend to compute one canonical
   `setup_ideal_bbox` in the JSON?
4. **Coaching bar after collapse** — leave `cue-quote` styling out
   entirely once Details is closed, OR keep a static one-liner visible
   (e.g., the same `shortSummary(cue)` shown by default)?
5. **Inline `SwingVideoRow` cast vs canonical type** — accept the
   inline cast pattern (§D option a)?
6. **Commit 5 (UI collapse)** — does this need to also gate behind
   `issueType === 'head_movement'`, or apply to all issues globally as
   a UX improvement? Spec implies only-when-head_movement; please
   confirm.

⏸ **STOP** — awaiting sign-off on questions 1-6 (or "all good, proceed
to Step 3"), then I'll implement the 5-commit batch.
