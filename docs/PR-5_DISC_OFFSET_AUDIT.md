# PR-5 disc / overlay coordinate-offset audit

**Date**: 2026-05-17
**Audit scope**: position-on-canvas defect ("disc in upper-left, person
in center-bottom") on video `eec305a5-8758-4cf0-ad25-b81e72d3653b`.
**Read-only** — no code touched, no commits, no PR-5.1 opened.
**Code state**: `main` at `1899d76` (PR-5 + PR-5 hotfix `6ca1e31` +
PR-5.1 anatomical correction `1899d76`).

> The PR-5 angle algorithm acceptance (setup −2.2°, top 41.6°, impact
> −4.2°) was verified BEFORE PR-5.1 landed; those values prove the
> stored keypoint values are in pixel space at the algorithm layer
> (dx ≈ 107 px for setup is not a normalized 0-1 quantity). The visual
> offset bug is therefore at the **canvas drawing / coordinate
> transformation** layer, NOT in the keypoint values themselves.

---

## 1. Data flow — coordinate-space chain

| Station | Coord space | Code reference | Notes |
|---|---|---|---|
| **A.** MediaPipe pose | normalized `[0, 1]` relative to **cv2-read frame buffer** (`pose_landmarks.landmark[i].x / .y`) | (MediaPipe SDK) | Native MediaPipe output. |
| **B.** `analyzer.py:get_video_metadata` | reads `cv2.CAP_PROP_FRAME_WIDTH` / `…_HEIGHT` | `python/analyzer.py:114-126` | **No EXIF/rotation handling.** Raw pixel buffer dims, NOT display dims. (Confirmed via repo-wide grep for `rotate\|orientation\|EXIF\|CAP_PROP_ORIENT` returns **zero matches** in `python/`.) |
| **C.** `extract_coco_subset_from_mediapipe` | converts to **video native pixel** by `lm.x * video_width`, `lm.y * video_height` | `python/pose_timeline.py:87-91` | The `video_width` here is the cv2-buffer width from station (B). |
| **D.** `build_timeline_from_raw_coco_frames` | wraps in JSON envelope; stores `video_width`/`video_height` from the same cv2-buffer dims | `python/pose_timeline.py:241-260` | Envelope and keypoint values come from the SAME source → internally consistent. |
| **E.** Pipeline pass-throughs (`detect_outliers_and_reject`, `smooth_ema`, `gap_fill_linear`, `apply_yolo_anchor_correction`) | mutate keypoint x/y in same pixel space | `python/pose_timeline.py:99-408` | YOLO anchor correction (when applied) reads YOLO keypoints from `pose_3d_phases.yolo_keypoints_2d` which the ONNX decoder reverse-letterboxes to source pixel space (`python/yolo/decoder.py:108-112`). Same space — no contamination. |
| **F.** DB write | JSONB on `swing_videos.pose_timeline_2d` | `route.ts` UPDATE | Verbatim. |
| **G.** Result page hydration | `vid.pose_timeline_2d as PoseTimeline \| null`, plumbed to `<SwingPlayer poseTimeline={…} />` | `src/app/result/[id]/page.tsx:63-66, 175-181` | No transformation. |
| **H.** `frameAt(t, pose)` | binary search over `frame.ts` (PR-5 hotfix `6ca1e31`); `fps_sampled` is NEVER READ | `src/lib/disc/frameAt.ts:20-44` | The PR-5 hotfix removed the `Math.round(t * fps_sampled)` lookup. The 10-vs-14-fps metadata drift no longer affects which frame is chosen. ✓ |
| **I.** `computeShoulderDisc` / `computeHipDisc` | reads `frame.keypoints.left_shoulder` (pixel coords), emits `DiscParams { cx, cy, rx, ry, angleRad, confidence }` **in the same pixel space** | `src/lib/disc/computeDiscParams.ts:172-256` | PR-5.1 anatomical correction lifts cx/cy by 15-50 px — small adjustment, would NOT explain a top-left vs center-bottom mis-placement. |
| **J.** `SwingPlayer.drawDisc` | `ctx.translate(p.cx * scaleX, p.cy * scaleY)` then `ellipse(0, 0, rx*scaleX, ry*scaleY, …)` where `scaleX = canvas.width / poseTimeline.video_width` and `scaleY = canvas.height / poseTimeline.video_height` | `SwingPlayer.tsx:68-85` + `159-161` | **Naive multiplicative scale.** Assumes canvas display dims map 1-to-1 onto the cv2-buffer pixel grid. No letterbox compensation. No DPR awareness. |
| **K.** `SwingPlayer.syncCanvas` | `c.width = rect.width`, `c.height = rect.height` from `videoEl.getBoundingClientRect()` | `SwingPlayer.tsx:118-127` | CSS pixels. Canvas buffer sized to the `<video>` element's bounding box. |
| **L.** Video layout CSS | `.sp-vid { width:100%; display:block; object-fit:contain; }`, `.sp-cvs { position:absolute; inset:0; width:100%; height:100% }` | `SwingPlayer.tsx:464-466` | Height is `auto` on `.sp-vid` → element naturally takes the source aspect; `.sp-vw` parent has no explicit height. In the steady state, no letterbox should be needed. |

### Key invariant

**Stations (B)→(C)→(D) all use the SAME cv2-buffer dimensions.** So at
the DB layer, `pose_timeline_2d.video_width` and `keypoints[*]` are
internally consistent — they describe the same coordinate frame.

The bug therefore must live in one of:
- The choice of *which* coordinate frame cv2 returned (station B)
- The choice of *which* coordinate frame the **browser displays**
  the video in (which the canvas math implicitly assumes matches)
- The choice of *which area* the canvas covers (letterbox handling)

---

## 2. Single-most-likely root cause

### 🔴 The cv2 frame buffer and the browser-displayed frame are NOT the same orientation.

**Code evidence**:
- `python/analyzer.py:114-126` reads `cv2.CAP_PROP_FRAME_WIDTH` /
  `CAP_PROP_FRAME_HEIGHT` directly. cv2 does **not** honour the MP4
  display matrix / EXIF rotation by default; it returns the **raw pixel
  buffer** dimensions.
- Grep `rotate|orientation|EXIF|CAP_PROP_ORIENT` across `python/`
  returns **zero matches**. There is no rotation-handling anywhere in
  the analyzer.
- HTML5 `<video>` elements **do** honour the MP4 display matrix —
  iPhone portrait videos display upright even though the underlying
  pixel buffer is landscape.

**Resulting failure mode**:
- Source file: iPhone-encoded portrait → pixel buffer 1920×1080,
  display matrix rotation +90°.
- Backend: cv2 reads 1920×1080 landscape pixels. MediaPipe sees a
  person lying on their side. Keypoints are detected and stored as
  pixel coordinates within the 1920×1080 landscape buffer.
- Backend writes `pose_timeline_2d.video_width = 1920`,
  `video_height = 1080`, with keypoints inside that frame.
- Browser: displays the video portrait (1080×1920 logical). The
  `<video>` element auto-sizes to portrait aspect. Canvas covers that
  element.
- `SwingPlayer.drawDisc` computes `scaleX = canvas.width / 1920`,
  `scaleY = canvas.height / 1080`. These are now **asymmetric** AND
  point at the *wrong* coordinate frame: the cv2 landscape buffer, not
  the visible portrait frame.

**Why it lands in the top-left specifically**: depends on the rotation
direction the iPhone used. For a +90° CW display rotation, a person
displayed at portrait centre maps to the landscape-buffer's right
edge — that's **right-middle of canvas**, not top-left. For a −90° /
+270° rotation, the same display position maps to the landscape
buffer's **left edge / upper-third** depending on the specific
transform direction. The "discs clustered in top-left" pattern is
consistent with a rotation transform that maps the displayed person
to the upper-left corner of the cv2 landscape buffer.

I **cannot prove from the codebase alone** that this is the exact
root cause for `eec305a5-…` without three pieces of empirical
evidence the audit doesn't have access to:

  1. The actual stored `swing_videos.pose_timeline_2d.video_width /
     video_height` for that row.
  2. The MP4 file's display matrix / rotation tag (`ffprobe -v error
     -select_streams v:0 -show_entries stream_tags=rotate,side_data
     <file>` or `ffmpeg -i <file> 2>&1 | grep -E "rotate|displaymatrix"`).
  3. The `videoElement.videoWidth` / `videoHeight` in the user's
     browser (devtools: `document.querySelector('video').videoWidth`).

**If (1) ≠ (3) and (2) shows a non-zero rotation**, the rotation
hypothesis is confirmed and the fix lives in **python/analyzer.py /
pose_timeline.py** (data layer), not PR-5 rendering.

### 🟡 Secondary hypothesis: canvas covers letterbox area while video content doesn't fill it

If for any reason `.sp-vw`'s aspect ≠ source-video's display aspect
(e.g., a parent rule forces a fixed height we haven't seen, or the
video element gets `object-fit: contain` letterboxing because of a
container constraint), the canvas covers the **full container**
(including black bars) while the video content occupies only the
central rectangle. The naive `scaleX/scaleY` in `drawDisc` doesn't
add the letterbox offset, so the disc lands offset by half the
letterbox area in the appropriate dimension.

The SkeletonOverlay's SVG `viewBox` + `preserveAspectRatio="xMidYMid
meet"` **does** handle this case automatically — see §3.

In the steady-state CSS we audited (`.sp-vid { width:100%; height:auto
}` with no parent-height constraint), there *shouldn't* be a
letterbox. But if the user is testing in a layout we don't see (e.g.,
a portrait viewport on mobile where flex parents add a height cap),
this becomes load-bearing.

### 🟢 Ruled-out hypotheses

| Hypothesis | Why not |
|---|---|
| `fps_sampled = 10` vs actual ~14 fps offsets `frameAt` lookup | The PR-5 hotfix `6ca1e31` switched `frameAt` to ts-based binary search; `fps_sampled` is no longer read. (`src/lib/disc/frameAt.ts:20-44`). |
| `lastShoulderRef` / `lastHipRef` lock position | These refs hold only `{ angleRad, ts }` for atan2 unwrap. They don't influence cx/cy. (`SwingPlayer.tsx:109-110, 196-202, 217-224`.) |
| `discAnchorRef` (PR-5.1) locks position | Anchor only overrides `rx` / `ry` (size). It doesn't touch cx/cy. (`SwingPlayer.tsx:116, 205, 225`.) |
| PR-5.1 anatomical correction shifts disc by ~50 px | Lift is 15–50 px in pixel space. Even at worst it cannot produce a "top-left vs center-bottom" displacement which on a 1080×1920 video would be ~500+ px. (`computeDiscParams.ts:81-119`.) |
| YOLO ONNX coordinate space contaminated PR-5 data | YOLO decoder reverse-letterboxes to source pixel space before storing in `pose_3d_phases.yolo_keypoints_2d` (`python/yolo/decoder.py:108-112`). YOLO anchor correction in `pose_timeline.py` uses these (already-source-pixel-space) values to compute offsets, so no contamination. |
| Stale `c.width` due to syncCanvas timing | renderTick reads `canvas.width` afresh each tick. After the initial `loadedmetadata`/`canplay`/`resize` events fire (within ~100ms of mount), the canvas dims are correct. The user is scrubbing AFTER video has loaded. |

---

## 3. SkeletonOverlay vs disc-canvas transform — line-by-line diff

### Skeleton path (`src/components/SkeletonOverlay.tsx:115-126`)

```tsx
<svg
  viewBox={`0 0 ${timeline.video_width} ${timeline.video_height}`}
  preserveAspectRatio="xMidYMid meet"
  style={{ position: 'absolute', inset: 0,
           width: '100%', height: '100%', pointerEvents: 'none' }}
>
  <circle cx={x} cy={y} ... />   // x, y in video-native pixels
</svg>
```

Key properties:
- SVG `viewBox` declares the *internal* coordinate frame as `[0,0,
  video_width, video_height]` in pixel units.
- `preserveAspectRatio="xMidYMid meet"` tells the browser: scale the
  internal frame to fit the element's CSS dimensions, preserving aspect
  ratio, centring with letterbox bars if the aspects mismatch.
- Child elements use pixel coords directly; the browser does the
  scale + letterbox math.

**Result**: keypoint at video-pixel `(540, 1700)` always lands at the
visually correct place on screen regardless of whether the element has
letterbox bars.

### Disc canvas path (`src/components/SwingPlayer.tsx:68-85` + `160-161`)

```ts
function drawDisc(ctx, p, color, scaleX, scaleY) {
  ctx.save();
  ctx.translate(p.cx * scaleX, p.cy * scaleY);
  ctx.rotate(p.angleRad);
  ctx.beginPath();
  ctx.ellipse(0, 0, p.rx * scaleX, p.ry * scaleY, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

// caller:
const scaleX = cw / poseTimeline.video_width;
const scaleY = ch / poseTimeline.video_height;
```

Key properties:
- `scaleX` and `scaleY` are **independent** linear multipliers.
- The translate uses `p.cx * scaleX, p.cy * scaleY` — no letterbox
  offset added.
- The math is correct **iff** canvas dimensions map 1-to-1 onto the
  cv2-stored frame, i.e., the canvas covers exactly the video content
  area with the same aspect ratio.

### Concrete diff

| Aspect | Skeleton (SVG) | Disc canvas |
|---|---|---|
| Coordinate declaration | `viewBox` (browser handles scale) | manual `scaleX/scaleY` multiplication |
| Aspect mismatch handling | `preserveAspectRatio="xMidYMid meet"` → centred with letterbox bars | none — naive linear scale, ignores letterbox |
| DPR handling | implicit via SVG rendering | none (canvas buffer dims = CSS px, no DPR multiplier) |
| Vulnerability to a rotation mismatch | same as disc — both read the same `video_width/height` and same keypoint values | same as skeleton — same data |
| Vulnerability to a letterbox layout | **immune** | **vulnerable** |

**If the user observes the skeleton overlay rendering at the correct
visual position AND the disc rendering in the top-left**, that's
definitive evidence the offset comes from a letterbox / aspect
mismatch between canvas-covered-area and video-content-area, not a
rotation issue (since rotation would affect both equally).

If the user observes **both** skeleton and disc clustered in the same
wrong spot, the cause lives at the data layer (rotation /
keypoint-source mismatch) and PR-5 rendering can't fix it.

---

## 4. Answers to the six audit questions

### 4.1 What coord space does `pose_timeline_2d.frames[i].keypoints[name]` store?

**(B) video native pixel** — relative to whatever cv2 returned for
`CAP_PROP_FRAME_WIDTH/HEIGHT`, which is the raw pixel buffer (no EXIF
rotation handling).

Evidence: `python/pose_timeline.py:87-91`
```python
out[name] = [
    round(float(lm.x) * video_width, 1),
    round(float(lm.y) * video_height, 1),
    conf,
]
```
where `video_width` comes from `python/analyzer.py:114-126`
`get_video_metadata()` → `cap.get(cv2.CAP_PROP_FRAME_WIDTH)`. **Not
(A) 640**, **not (C) normalized**, **not (D) letterbox-padded** — but
the cv2 buffer **may** itself differ from the displayed orientation
when EXIF rotation is present.

### 4.2 What coord space does the SwingPlayer canvas assume?

**Same as (4.1)** — video native pixel as stored in
`poseTimeline.video_width / video_height`. The math at
`SwingPlayer.tsx:160-161` is:
```ts
const scaleX = cw / poseTimeline.video_width;
const scaleY = ch / poseTimeline.video_height;
```

There is **no gap** at this point in the pipeline — frontend reads the
same metadata that backend wrote. The gap is upstream: between cv2's
buffer dims (what we store) and the browser's display dims (what we
implicitly assume the canvas covers).

### 4.3 Are the skeleton + disc transforms the same?

**No.** Skeleton uses SVG `viewBox + preserveAspectRatio` (browser
handles scale + letterbox automatically). Disc uses naive canvas
`ctx.translate(p.cx * scaleX, p.cy * scaleY)` with separate `scaleX/
scaleY` and no letterbox math. See §3 for the line-by-line diff.

### 4.4 `frameAt(currentTime=3.286s, …)` returns which frame?

**Current behaviour** (post-PR-5-hotfix at `6ca1e31`,
`src/lib/disc/frameAt.ts:20-44`): ts-based binary search over
`frame.ts`. **Does NOT read `fps_sampled`**. Returns the frame whose
`ts` is numerically closest to 3.286 — typically the frame `i` such
that `|frames[i].ts − 3.286|` is minimum among all 70 frames.

**Counterfactual — pre-hotfix `Math.round(t * fps_sampled)`**:
- With (incorrect) `fps_sampled = 10` from backend metadata:
  `Math.round(3.286 × 10) = 33` → returns frames[33], which has
  `ts ≈ 33 / 14 ≈ 2.357s` in reality. Off by **~0.93s** = approx 13
  frames of MediaPipe at the true sample rate. Wrong phase entirely.
- With true sample rate 14 fps: `Math.round(3.286 × 14) = 46` → would
  hit the right frame.

**Bottom line**: this is fixed at the frame-index level. It does **not**
contribute to the current position-on-canvas defect.

### 4.5 Does any `useRef` lock a keypoint position?

**No.** The three refs and what each holds:

| Ref | What it holds | Affects |
|---|---|---|
| `lastShoulderRef` | `{ angleRad, ts }` (post-unwrap rolling state) | shoulder disc **angle continuity only** |
| `lastHipRef` | same shape | hip disc **angle continuity only** |
| `discAnchorRef` (PR-5.1) | `{ shoulderRx, hipRx }` | disc **size only** (raw `rx` is overridden in `drawDisc`) |

**None of these affect `cx` or `cy`.** Disc center always comes from
the current `poseFrame` via `computeShoulderDisc/HipDisc`, which reads
`frame.keypoints.{left,right}_shoulder/hip` directly.

### 4.6 Video element vs canvas — what's the relationship?

- `.sp-vid { width: 100%; display: block; object-fit: contain;
  background: #000; }` — width 100% of parent, height defaults to
  `auto`, `object-fit: contain` only matters if a fixed height is
  imposed (which it isn't in this codebase).
- `.sp-cvs { position: absolute; top: 0; left: 0; width: 100%;
  height: 100%; pointer-events: none; }` — covers the `.sp-vw` parent
  100%.
- `.sp-vw { position: relative; width: 100%; }` — no fixed height.

**In the steady state with no fixed parent height**: the video element
auto-sizes its height to match the source video's aspect ratio. The
`.sp-vw` parent then matches that height (because no fixed height). The
canvas covers the same rect. **No letterbox should be present.**

**But**:
- If a parent UPSTREAM (page layout, flexbox) constrains the height of
  `.sp-vw` (e.g., to fit viewport), then `.sp-vid`'s `object-fit:
  contain` activates and we get a real letterbox. The canvas covers
  the FULL constrained rect, the video CONTENT is centred in it, and
  the naive disc math drifts.
- The audit can't determine from the CSS alone whether the user's
  actual viewport layout introduces such a constraint.

There is no DPR (devicePixelRatio) scaling applied — canvas buffer
dims = CSS px from `getBoundingClientRect()`. This makes the canvas
buffer NOT high-DPR-sharp, but does not displace the disc.

---

## 5. Regression risk on the PR-5 angle acceptance

The PR-5 angle algorithm (atan2 + unwrap, plus PR-5.1's distance-ratio
magnitude) reads `frame.keypoints[…][0]` and `[1]` (pixel coords) and
computes `dx, dy, dist, atan2(dy, dx), acos(dist/baselineDist)`.

**The angle math is invariant under any uniform linear scale or
translation of the coordinate frame**: `atan2(k·dy, k·dx) = atan2(dy,
dx)` for any positive `k`, and translation cancels in the `(L − R)`
subtraction.

**The angle math is NOT invariant under**: rotation of the frame
(would rotate atan2 results by the rotation angle) or non-uniform
scaling (would skew atan2). Neither of these is happening in our
pipeline — the cv2 buffer is a single fixed orientation throughout
the swing; MediaPipe runs in that single frame.

**Conclusion**: any fix that corrects the position offset (whether by
honouring video rotation, fixing letterbox compensation in canvas
draw, or switching to SVG viewBox) will **not regress** the
already-passed angle acceptance values (setup −2.2°, top 41.6°, impact
−4.2°). The angle and the position are decoupled at the algorithm
level.

PR-5.1's distance-ratio magnitude (`acos(dist/baselineDist)`) depends
on **dist ratios**, which are also invariant under uniform scale and
translation. Same conclusion: position fixes don't touch the magnitude.

---

## 6. Suggested PR-5.2 scope (minimum-viable)

> **Do not implement until Jason has run the empirical-verification
> steps in §7 and confirmed which hypothesis is the actual cause.**

Three possible PRs depending on which hypothesis the verification step
confirms:

### PR-5.2-A — if rotation is the root cause (most likely)

Affects the **data layer**, not PR-5 rendering. Escalate the ticket.

**Files to touch**:
- `python/analyzer.py` — handle video rotation. Either (1) detect via
  `ffprobe`/`pymediainfo` and rotate frames via `cv2.rotate` before
  feeding MediaPipe, OR (2) record the rotation tag in
  `pose_timeline_2d` so the frontend can apply the inverse transform.
- `python/pose_timeline.py` — extend JSON schema (`version: 1` →
  `version: 2`?) with a `display_rotation` field, OR ensure
  `video_width`/`video_height` always correspond to display
  orientation, not raw buffer.

**Frontend**: no change if backend writes display-oriented coords;
trivial inverse-rotate if frontend gets a `display_rotation` field.

### PR-5.2-B — if letterbox is the root cause

Affects PR-5 rendering layer only. Single-file change.

**Files to touch**:
- `src/components/SwingPlayer.tsx` — replace the naive `scaleX/scaleY`
  in `drawDisc` with letterbox-aware math:
  ```ts
  const videoAspect = poseTimeline.video_width / poseTimeline.video_height;
  const canvasAspect = cw / ch;
  let displayW, displayH, offsetX, offsetY;
  if (videoAspect > canvasAspect) {
    displayW = cw; displayH = cw / videoAspect;
    offsetX = 0; offsetY = (ch - displayH) / 2;
  } else {
    displayH = ch; displayW = ch * videoAspect;
    offsetX = (cw - displayW) / 2; offsetY = 0;
  }
  const scale = displayW / poseTimeline.video_width;  // uniform
  // translate: ctx.translate(offsetX + p.cx * scale, offsetY + p.cy * scale)
  // ellipse:   ctx.ellipse(0, 0, p.rx * scale, p.ry * scale, ...)
  ```
- Or **simpler and more robust**: migrate disc rendering from canvas
  to SVG, reusing the same `viewBox + preserveAspectRatio` pattern as
  SkeletonOverlay. Eliminates the inconsistency from §3 entirely.

### PR-5.2-C — if neither (verification reveals a different root cause)

Pause and re-audit with the new evidence.

---

## 7. Empirical verification — please run these before deciding PR-5.2 scope

Three checks that disambiguate the candidate hypotheses:

### 7.1 Database row inspection

Read the actual stored values for `eec305a5-…`. In a Supabase SQL
editor (user-scoped via Jason's auth):

```sql
SELECT
  id,
  pose_timeline_2d->'video_width'  AS stored_width,
  pose_timeline_2d->'video_height' AS stored_height,
  pose_timeline_2d->'fps_sampled'  AS fps_sampled,
  pose_timeline_2d->'frames'->0->'keypoints'->'left_shoulder'  AS setup_lshoulder,
  pose_timeline_2d->'frames'->0->'keypoints'->'right_shoulder' AS setup_rshoulder,
  pose_timeline_2d->'frames'->0->'keypoints'->'left_hip'       AS setup_lhip
FROM swing_videos
WHERE id = 'eec305a5-8758-4cf0-ad25-b81e72d3653b';
```

**Expected if no bug**: e.g., `stored_width=1080`, `stored_height=1920`,
shoulder values like `[540, 480, 0.9]` (centre-upper of a 1080×1920
frame). If shoulder x < 100 or shoulder y < 200, person was indeed
detected in the upper-left of the cv2 buffer → confirms rotation
hypothesis OR a different bug.

### 7.2 Browser-side check (Chrome MCP or DevTools)

After the result page for `eec305a5-…` is loaded:

```js
(() => {
  const v = document.querySelector('video');
  const c = document.querySelector('canvas.sp-cvs');
  const svg = document.querySelector('svg.skeleton-overlay');
  return {
    video_natural:    { w: v.videoWidth, h: v.videoHeight },
    video_clientRect: v.getBoundingClientRect().toJSON(),
    canvas_buffer:    { w: c.width, h: c.height },
    canvas_clientRect: c.getBoundingClientRect().toJSON(),
    svg_viewBox:      svg?.getAttribute('viewBox'),
    sample_kp: (() => {
      // Read the same data the disc sees:
      const tl = window.__poseTimeline ?? null;  // if exposed; else from React DevTools
      return tl?.frames?.[0]?.keypoints ?? 'not exposed';
    })(),
  };
})()
```

Compare `video_natural.w/h` (what the browser sees) with `canvas_buffer.w/h`
(what the canvas sees). They should be in the same aspect ratio. Compare
`video_natural` with the DB's `stored_width/stored_height` — if they
DIFFER (e.g., DB says 1920×1080 but browser says 1080×1920), the
rotation mismatch is confirmed.

### 7.3 MP4 file inspection

If Jason has access to the raw video file or can pull it from the
Supabase storage signed URL:

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,coded_width,coded_height,r_frame_rate \
  -show_entries stream_tags=rotate \
  -show_entries stream_side_data=rotation \
  -of default=noprint_wrappers=1 \
  <video_file>
```

If the output shows a non-zero `rotate` tag or `displaymatrix` rotation
side-data, **rotation hypothesis confirmed**. The data layer
(PR-5.2-A) is the fix scope.

---

## 8. Audit conclusion

**Single most-likely root cause**: video rotation metadata mismatch
between cv2 (which doesn't honour the MP4 display matrix) and the
HTML5 `<video>` element (which does). `pose_timeline_2d.video_width/
height` describes the cv2 landscape buffer, keypoints live inside it,
canvas math scales using the wrong dimensions because the canvas
covers the browser's *display*-oriented `<video>` element.

**Fix is NOT in PR-5 rendering layer**; it's in the data layer
(`python/analyzer.py` / `python/pose_timeline.py`). Escalate as a
data-layer bug, similar to the earlier PR-4.1 backlog item for
`fps_sampled` metadata.

**Secondary** less-likely cause: letterbox math missing in
`drawDisc` (`SwingPlayer.tsx:68-85`). The SVG skeleton handles this
case; the canvas disc does not. If empirical verification rules out
rotation, this is the next thing to address — and it can be done
entirely in `SwingPlayer.tsx` (PR-5.2-B), with the simplest fix being
to migrate disc rendering to SVG to match SkeletonOverlay's pattern.

**Already-ruled-out**: `fps_sampled` drift (fixed in `frameAt`
hotfix), useRef position locking (refs only touch angle/size, never
position), anatomical correction (too-small magnitude to displace
center-bottom → top-left), YOLO coord-space contamination (decoder
reverse-letterboxes).

**Regression risk on PR-5 angle acceptance from any of the candidate
fixes**: zero — angles are invariant under uniform scale and
translation, which is what every candidate fix preserves.

⏸ **Audit complete. Waiting for Jason to run §7 verification and
decide PR-5.2-A vs PR-5.2-B vs continue audit.**
