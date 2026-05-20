# PR-6.0 Phase 1A — benchmark harness handoff report

**Status**: scaffolding shipped; awaiting Jason to install deps and
run. No production code touched. Phase 1B (run + collect + observe)
is Jason's lane.

---

## 1. What landed

```
python/benchmark/
├── __init__.py
├── README.md                       ← start here
├── requirements_benchmark.txt
├── download_videos.py
├── run_all.py                      ← top-level orchestrator
├── runner.py                       ← Runner base + COCO 17 schema
├── overlay.py                      ← skeleton onto source video
├── compare.py                      ← side-by-side stacker
├── metrics.py                      ← single + cross-runner stats
└── runners/
    ├── __init__.py
    ├── mediapipe_pose.py           ← production mirror (confidence: HIGH)
    ├── mediapipe_tasks.py          ← Tasks-API + heavy model (MEDIUM)
    └── movenet_thunder.py          ← Google MoveNet (MEDIUM)
```

`.gitignore` updated to exclude `python/benchmark/test_videos/`,
`python/benchmark/output/`, and `.venv-benchmark/`. All runner outputs
are local artefacts; nothing the harness creates ends up in git.

---

## 2. Setup decisions

### 2.1 Model picks (one per runner)

| Runner            | Model                                    | Rationale |
|-------------------|------------------------------------------|-----------|
| `mediapipe_pose`  | BlazePose **Full** (`model_complexity=1`) | Bit-exact mirror of production (`python/analyzer.py:198`). Lets us validate the harness against stored `pose_timeline_2d` before reading too much into the other two runners. |
| `mediapipe_tasks` | `pose_landmarker_heavy.task` (float16)   | "Free upgrade" candidate. Heavy ≈ `model_complexity=2` quality, which production has never tried; same family so disagreement = pure capacity, not architecture switch. |
| `movenet_thunder` | `singlepose/thunder/4` (TF Hub)          | Independent third-party model. Thunder > Lightning by ~10% on motion-heavy clips. Single-person — exactly the golf case. |

### 2.2 Sample rate

All runners default to `--sample-fps 10`, matching production's
`sample_fps=10.0` in `python/main.py:88`. Comparable apples-to-apples
with stored `pose_timeline_2d.fps_sampled`.

### 2.3 Output schema

Identical across runners — `RunResult` in `runner.py`:

```json
{
  "video_id":     "b3fea3f0-...",
  "runner":       "mediapipe_pose",
  "video_width":  720,
  "video_height": 1280,
  "fps_native":   30.0,
  "fps_sampled":  10.0,
  "duration_sec": 4.466,
  "frames": [
    {"ts": 0.000, "frame_idx": 0, "keypoints": {"nose": [355.2, 412.1, 0.93], ...}}
  ],
  "notes": ["model_complexity=1", "smooth_landmarks=True", ...]
}
```

Same `[x, y, conf]` triple convention as production `pose_timeline_2d`.
Below-visibility keypoints are stored as `[null, null, conf]` so
downstream code can disambiguate "model didn't see it" from
"coordinate is exactly (0, 0)" — identical to the production
behaviour in `extract_coco_subset_from_mediapipe`.

### 2.4 Coordinate convention

Native video pixel space. Each runner re-derives `video_width` /
`video_height` from cv2 — same source of truth as production.

### 2.5 Single-process, no parallelism

Phase 1A doesn't pipeline runners. Each runner runs serially in
`run_all.py`. Each runner loads its model once and processes all
videos in sequence. This is fine for 3 videos × 3 runners; if/when we
scale to 20 videos, parallelise then.

### 2.6 Heaviness budget

| Component            | Disk    | RAM (run-time) |
|----------------------|---------|----------------|
| MediaPipe wheel      | ~50MB   | ~400MB         |
| Heavy model (.task)  | ~37MB   | +~200MB        |
| TensorFlow + tf-hub  | ~500MB  | ~1.5GB         |
| MoveNet Thunder v4   | ~25MB   | +~300MB        |
| Test videos (3×)     | ~50MB   | n/a            |
| Output (3 vid × 3 runner × overlay+compare) | ~200MB | n/a |

Plan for ~1GB free disk and a 16GB-RAM laptop. CPU-only is fine.

---

## 3. Known risks (ordered by likelihood × impact)

### R1 — `mediapipe_tasks` API drift  (likelihood: HIGH, impact: contained)

**File**: `python/benchmark/runners/mediapipe_tasks.py`

The MediaPipe Tasks API has shifted naming a few times across
0.10.x. The runner targets the 0.10.21 spelling (production version)
but if you've got a different version pinned globally, the imports
might fail. Lines flagged with `# TODO(jason)`:

- L36-37 — `from mediapipe.tasks import python; from mediapipe.tasks.python import vision`. If ImportError, check the [MediaPipe Python API reference](https://ai.google.dev/edge/mediapipe/api/solutions/python) for current spelling.
- L101 — `detect_for_video(mp_image, timestamp_ms)`. Synchronous video-mode call. If it's renamed (e.g. `process_video`), error message will be clear; swap the call.
- L138 — `lm.visibility` for confidence. The Tasks API exposes both `.visibility` and `.presence`; using `.visibility` for apples-to-apples with `mediapipe_pose`. If numbers look weird in Phase 1B, try `.presence`.
- L83 — `mp.ImageFormat.SRGB`. The format enum has held stable but may need `.SRGB_4444` on Apple Silicon with certain TF stacks.

**Fallback if it just won't work**: comment out the `mediapipe_tasks`
import in `run_all.py:_all_runners` and proceed with 2 runners. Better
than nothing.

### R2 — TensorFlow CUDA wheels on CPU-only machines  (HIGH × annoyance)

**File**: `python/benchmark/runners/movenet_thunder.py`

`pip install tensorflow==2.16.2` may pull `nvidia-*` CUDA wheels even
on a CPU-only machine. These wheels are ~500MB each and slow the
install massively. Fix: use `tensorflow-cpu==2.16.2` instead. Edit
`requirements_benchmark.txt` if needed.

A second TF gotcha — sometimes the first import hangs for 30+s while
TF probes for GPUs. Just wait; it eventually proceeds to CPU.

### R3 — MoveNet letterbox un-pad math  (MEDIUM × low)

**File**: `movenet_thunder.py` L131-149

MoveNet outputs `[y, x, conf]` normalised to the **letterboxed 256×256
input**, not the original frame. The un-pad math:

```python
y_px_pad = y_norm * INPUT_SIZE
x_px_pad = x_norm * INPUT_SIZE
y_orig = (y_px_pad - pad_y) / scale
x_orig = (x_px_pad - pad_x) / scale
```

I've sanity-checked this on paper; if MoveNet's overlay looks
systematically offset (e.g. all keypoints shifted left by ~10% in
a portrait video), this is the spot to debug. TODO comment marks
it inline.

### R4 — `download_videos.py` placeholder UUIDs  (LOW × you'll-notice)

Only `b3fea3f0-e248-44d7-a923-0bb43172b5bf` has a full UUID hard-coded.
The other two slots are 8-char prefixes (`a735cc7d`, `5bbcfbc8`)
because I don't have DB access from the sandbox to look them up.

The downloader skips entries shorter than 36 chars with a clear
message. Edit `DEFAULT_VIDEO_IDS` in `download_videos.py` to fill in
full UUIDs, or just run the single calibration video for now:

```bash
python download_videos.py b3fea3f0
```

### R5 — `swing-videos` bucket name  (LOW × easy-fix)

Hard-coded to `swing-videos` based on `src/app/api/analyze/[id]/route.ts:201`.
If your bucket is named differently, edit `BUCKET` at the top of
`download_videos.py`.

### R6 — overlay/compare codec  (LOW × cosmetic)

`cv2.VideoWriter_fourcc(*"mp4v")` is the most-compatible H.264-ish
codec across platforms but produces large files (no real H.264
re-encoding). Comparison videos for 3 runners × 4s @ 30fps will be
~30MB each. Fine for inspection; not what you'd ship.

---

## 4. How to run (5-minute path)

This is the abbreviated version — README.md has the full walkthrough.

```bash
# Repo root.
python3.11 -m venv .venv-benchmark
source .venv-benchmark/bin/activate
pip install -r python/benchmark/requirements_benchmark.txt

export NEXT_PUBLIC_SUPABASE_URL=https://<...>.supabase.co
export SUPABASE_SERVICE_ROLE_KEY=eyJ...   # service-role, NEVER commit

cd python/benchmark
python download_videos.py b3fea3f0        # just the calibration video for now

python -m benchmark.run_all --videos b3fea3f0
```

You'll end up with:

```
python/benchmark/output/
  mediapipe_pose/b3fea3f0-.../{keypoints.json, overlay.mp4, single.metrics.json}
  mediapipe_tasks/b3fea3f0-.../{keypoints.json, overlay.mp4, single.metrics.json}
  movenet_thunder/b3fea3f0-.../{keypoints.json, overlay.mp4, single.metrics.json}
  comparison_b3fea3f0-....mp4               ← THIS is what to watch
  comparison_b3fea3f0-....metrics.json     ← per-kp distance from production
```

---

## 5. Expected runtime (rough — depends heavily on hardware)

On a 2024 M-series Mac, CPU-only, b3fea3f0 (~4.5s video, 60 sampled
frames at 10 fps):

| Stage                              | Time                |
|------------------------------------|---------------------|
| `pip install -r requirements_benchmark.txt` | 5-10 min (one-time) |
| TF first-import (MoveNet only)      | 5-30s (one-time)    |
| `download_videos.py b3fea3f0`       | 2-5s                |
| Runner: `mediapipe_pose`            | ~5s                 |
| Runner: `mediapipe_tasks` (heavy)   | ~8s                 |
| Runner: `movenet_thunder`           | ~12s                |
| Overlay × 3                         | ~9s                 |
| Comparison stacker                  | ~3s                 |
| Metrics                             | <1s                 |
| **Total per video**                 | **~40s**            |

Three videos: ~2 min wall-clock once setup is done.

On CPU-only Intel: roughly 2-3× slower. On older Macs without
hardware-accelerated TF: MoveNet may be 5× slower (still tolerable).

---

## 6. What Phase 1B looks like

Once Jason runs the harness:

1. Watch `comparison_b3fea3f0-*.mp4` end-to-end. Note any visible
   keypoint drift, jitter, or recovery (see README §5).
2. Skim `comparison_b3fea3f0-*.metrics.json` for the per-kp distance
   numbers — the headline is `mean_px` per keypoint per runner, vs the
   `mediapipe_pose` baseline.
3. Repeat for the other two videos (once their full UUIDs are filled in).
4. Write `docs/PR-6.0_PHASE_1B_OBSERVATIONS.md` — narrative notes on
   what looked better/worse, with timestamps where possible. **Claude
   Code can then read the comparison videos + metrics jsons and write
   a verdict report**: which model wins, by how much, on which
   keypoints, and whether the win justifies a production swap.

---

## 7. Constraints honoured (Phase 1A spec)

- No existing `python/` production files touched (`analyzer.py`,
  `pose_timeline.py`, `phase_detector.py`, `main.py`, Dockerfile etc.).
- No `src/` frontend touched.
- No new entries in `python/requirements.txt` — benchmark deps go
  only in `python/benchmark/requirements_benchmark.txt`.
- `.gitignore` got 3 additive entries (test_videos/, output/, venv).
- Nothing pushed without your explicit go.

Single commit, additive only, ready to push direct to `main` when you say.
