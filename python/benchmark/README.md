# PR-6.0 benchmark harness

Compares three pose estimators on three swing videos to decide whether
SwingCue should swap the production pipeline (currently MediaPipe Pose
0.10.x model_complexity=1) for something better.

**Phase 1A** (this PR) — Claude Code delivered the scaffolding.
**Phase 1B** (you, the human) — install deps, download videos, run.

Outputs Claude Code does **NOT** produce:

- `output/<runner>/<video_id>/keypoints.json`
- `output/<runner>/<video_id>/overlay.mp4`
- `output/comparison_<video_id>.mp4`
- `output/<runner>/<video_id>/single.metrics.json`
- `output/comparison_<video_id>.metrics.json`

You produce those by running the harness.

---

## 1. Setup (one-time)

Python 3.11 required. Anything older won't satisfy `tensorflow==2.16.2`.

```bash
# from repo root
python3.11 -m venv .venv-benchmark
source .venv-benchmark/bin/activate     # macOS/Linux
# .venv-benchmark\Scripts\Activate.ps1  # Windows PowerShell

pip install --upgrade pip
pip install -r python/benchmark/requirements_benchmark.txt
```

The install pulls TensorFlow (~500MB), MediaPipe (~50MB), and assorted
runtime libs. Plan for ~10 min of pip on first run.

> **Tip**: if you only want to run the two MediaPipe runners and skip
> MoveNet, comment out the `tensorflow*` lines in
> `requirements_benchmark.txt` first. The harness will skip
> `movenet_thunder` cleanly if the import fails.

---

## 2. Download test videos

The harness expects mp4 files in `python/benchmark/test_videos/`. Three
video IDs are hard-coded — edit `download_videos.py` if you want
different ones.

```bash
# from repo root
export NEXT_PUBLIC_SUPABASE_URL=https://<your-project>.supabase.co
export SUPABASE_SERVICE_ROLE_KEY=eyJ...                 # NEVER commit this

# Windows PowerShell:
# $env:NEXT_PUBLIC_SUPABASE_URL = "https://..."
# $env:SUPABASE_SERVICE_ROLE_KEY = "eyJ..."

cd python/benchmark
python download_videos.py
```

You should end up with:

```
python/benchmark/test_videos/
  b3fea3f0-e248-44d7-a923-0bb43172b5bf.mp4
  a735cc7d-….mp4    # (replace prefix above with full UUID first)
  5bbcfbc8-….mp4
```

`a735cc7d` / `5bbcfbc8` placeholders need their full UUIDs filled in
inside `download_videos.py`'s `DEFAULT_VIDEO_IDS` tuple — Claude Code
didn't have DB access to look them up. If you only care about
`b3fea3f0`, the harness works on a single video.

---

## 3. Run all three runners

```bash
cd python/benchmark
python -m benchmark.run_all
```

This will, for each video × each runner:

1. Run the model → `output/<runner>/<id>/keypoints.json`
2. Render skeleton overlay onto the original video → `overlay.mp4`
3. Assemble side-by-side comparison → `output/comparison_<id>.mp4`
4. Compute per-runner stats + cross-runner distance → `*.metrics.json`

Skip steps with `--skip-overlay`, `--skip-compare`, `--skip-metrics`,
or `--skip-runners` (e.g. if you already have keypoints and just want
to re-render overlays after fixing a bug in `overlay.py`).

Run one video at a time:

```bash
python -m benchmark.run_all --videos b3fea3f0
```

Expected runtime, per video, on a 2024-era M-series Mac:

| Runner            | Sample fps | ~Time per video (4s swing, 60 frames sampled) |
|-------------------|------------|-----------------------------------------------|
| mediapipe_pose    | 10         | ~5s    (mirrors prod)                         |
| mediapipe_tasks   | 10         | ~8s    (heavy model)                          |
| movenet_thunder   | 10         | ~12s   (TF graph startup adds another ~3s once) |

Overlay rendering adds ~2–3s per video per runner. Comparison adds
another ~3s. On CPU-only laptops without GPU, MoveNet may be 2-3× slower.

---

## 4. Individual scripts (debugging)

```bash
# Single runner, single video:
python -m benchmark.runners.mediapipe_pose  test_videos/b3fea3f0.mp4 b3fea3f0
python -m benchmark.runners.mediapipe_tasks test_videos/b3fea3f0.mp4 b3fea3f0
python -m benchmark.runners.movenet_thunder test_videos/b3fea3f0.mp4 b3fea3f0

# Overlay one runner's output:
python -m benchmark.overlay \
    output/mediapipe_pose/b3fea3f0/keypoints.json \
    test_videos/b3fea3f0.mp4 \
    output/mediapipe_pose/b3fea3f0/overlay.mp4

# Compare overlays:
python -m benchmark.compare b3fea3f0 \
    output/mediapipe_pose/b3fea3f0/overlay.mp4 \
    output/mediapipe_tasks/b3fea3f0/overlay.mp4 \
    output/movenet_thunder/b3fea3f0/overlay.mp4

# Cross-runner metrics:
python -m benchmark.metrics compare \
    output/mediapipe_pose/b3fea3f0/keypoints.json \
    output/mediapipe_tasks/b3fea3f0/keypoints.json \
    output/movenet_thunder/b3fea3f0/keypoints.json
```

---

## 5. What to look for in the comparison

When you watch `comparison_<video_id>.mp4`, the three runners play
side-by-side. Things worth eyeballing (priority order):

1. **Wrist crossover at impact** — does the path stay smooth?
   Production sometimes loses wrists at impact (motion blur).
2. **Hip y position at setup** — known to be ~130px above the visual
   belt in production (see `docs/PR-3.1_POSE_DATA_AUDIT.md`). Does
   `mediapipe_tasks` heavy or `movenet_thunder` place it lower?
3. **Top-phase stability** — at the peak of backswing, do keypoints
   stay anchored to the body or jitter onto the club / background?
4. **Leg confidence in finish pose** — production loses lead leg at
   high finish poses. Does any candidate recover it?

`comparison_<video_id>.metrics.json` gives numeric backup for whatever
you eyeball — `mean_px` distance per kp from the production baseline,
plus `agreement_pct_at_30px` so you can see "this runner agrees with
production on 92% of kp within 30px".

---

## 6. Known risks (Phase 1A handoff)

See `docs/PR-6.0_PHASE_1A_REPORT.md` for the full list. Headline:

- **mediapipe_tasks.py**: the Tasks-API calls have been moving across
  MediaPipe 0.10.x patch releases. Inline `# TODO(jason)` comments mark
  the lines most likely to break if you're on a different MediaPipe
  version. Fallback: comment that runner out and proceed with the
  remaining two.
- **movenet_thunder.py**: TF+TF-Hub install can drag in CUDA wheels
  even on CPU-only machines. If TF fails to load on CPU, the inline
  TODO suggests switching to `tensorflow-cpu` in requirements.
- **download_videos.py**: only `b3fea3f0` has its full UUID hard-coded.
  Fill in the other two before running, or run with the explicit
  argument: `python download_videos.py b3fea3f0` to fetch only that one.
