"""
run_all.py — one-shot orchestrator.

For each video in test_videos/, runs all 3 runners → writes
keypoints.json under output/<runner>/<video_id>/. Optionally renders
overlays + comparison side-by-side video + cross-runner metrics.

Skip steps you don't want with --skip flags.

CLI:
    cd python/benchmark
    python run_all.py
    python run_all.py --skip-overlay --skip-compare   # just keypoints + metrics
    python run_all.py --videos b3fea3f0               # one video only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import overlay as overlay_mod
from . import compare as compare_mod
from . import metrics as metrics_mod
from .runner import Runner

BENCH_ROOT = Path(__file__).parent
OUTPUT_DIR = BENCH_ROOT / "output"
VIDEO_DIR  = BENCH_ROOT / "test_videos"


def _all_runners() -> list[Runner]:
    # Imported lazily — heavy ML libs only load when actually instantiated.
    from .runners.mediapipe_pose   import MediaPipePoseRunner
    from .runners.mediapipe_tasks  import MediaPipeTasksRunner
    from .runners.movenet_thunder  import MoveNetThunderRunner
    return [MediaPipePoseRunner(), MediaPipeTasksRunner(), MoveNetThunderRunner()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=None,
                    help="video ID substrings to filter (default: all in test_videos/)")
    ap.add_argument("--sample-fps", type=float, default=10.0)
    ap.add_argument("--skip-runners",  action="store_true")
    ap.add_argument("--skip-overlay",  action="store_true")
    ap.add_argument("--skip-compare",  action="store_true")
    ap.add_argument("--skip-metrics",  action="store_true")
    args = ap.parse_args()

    # Discover videos.
    if not VIDEO_DIR.exists():
        sys.exit(f"[run_all] {VIDEO_DIR} missing — run download_videos.py first")
    all_videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if args.videos:
        all_videos = [v for v in all_videos
                      if any(s in v.stem for s in args.videos)]
    if not all_videos:
        sys.exit(f"[run_all] no videos matched filter — exiting")
    print(f"[run_all] {len(all_videos)} video(s) to process:")
    for v in all_videos:
        print(f"   - {v.name}")

    runners = [] if args.skip_runners else _all_runners()

    # Step 1: run each runner on each video.
    for r in runners:
        print(f"\n[run_all] ===== setting up {r.name} =====")
        r.setup()
        try:
            for video in all_videos:
                video_id = video.stem
                print(f"[run_all]   {r.name} on {video_id} …")
                result = r.run(video, video_id, sample_fps=args.sample_fps)
                kp_out = OUTPUT_DIR / r.name / video_id / "keypoints.json"
                result.save(kp_out)
                print(f"[run_all]     → {kp_out}  ({len(result.frames)} frames)")
        finally:
            r.teardown()

    # Step 2: render per-runner overlay videos.
    if not args.skip_overlay:
        runner_names = [r.name for r in runners] or [
            d.name for d in OUTPUT_DIR.iterdir() if d.is_dir()
        ]
        for video in all_videos:
            video_id = video.stem
            for rn in runner_names:
                kp = OUTPUT_DIR / rn / video_id / "keypoints.json"
                ov = OUTPUT_DIR / rn / video_id / "overlay.mp4"
                if not kp.exists():
                    continue
                print(f"[run_all] overlay {rn}/{video_id} …")
                overlay_mod.render_overlay(kp, video, ov)

    # Step 3: side-by-side comparison.
    if not args.skip_compare:
        for video in all_videos:
            video_id = video.stem
            overlays = []
            for rn_dir in sorted(OUTPUT_DIR.iterdir()):
                if not rn_dir.is_dir():
                    continue
                ov = rn_dir / video_id / "overlay.mp4"
                if ov.exists():
                    overlays.append(ov)
            if len(overlays) < 2:
                print(f"[run_all] compare {video_id}: <2 overlays — skip")
                continue
            out = OUTPUT_DIR / f"comparison_{video_id}.mp4"
            print(f"[run_all] compare {video_id} ({len(overlays)} runners) …")
            compare_mod.make_comparison(video_id, overlays, out)

    # Step 4: per-runner single metrics + cross-runner comparison metrics.
    if not args.skip_metrics:
        for video in all_videos:
            video_id = video.stem
            kp_paths: list[Path] = []
            for rn_dir in sorted(OUTPUT_DIR.iterdir()):
                if not rn_dir.is_dir():
                    continue
                kp = rn_dir / video_id / "keypoints.json"
                if not kp.exists():
                    continue
                kp_paths.append(kp)
                single = metrics_mod.single_runner_metrics(kp)
                (kp.parent / "single.metrics.json").write_text(
                    __import__("json").dumps(single, indent=2)
                )
                print(f"[run_all] metrics {rn_dir.name}/{video_id}: "
                      f"coverage={single['stats'].get('frame_coverage')}")
            if len(kp_paths) >= 2:
                # Baseline = mediapipe_pose if present, else first.
                baseline = next(
                    (p for p in kp_paths if p.parts[-3] == "mediapipe_pose"),
                    kp_paths[0],
                )
                others = [p for p in kp_paths if p != baseline]
                comp = metrics_mod.cross_runner_metrics(baseline, others)
                out = OUTPUT_DIR / f"comparison_{video_id}.metrics.json"
                out.write_text(__import__("json").dumps(comp, indent=2))
                print(f"[run_all] cross-metrics {video_id} → {out}")

    print("\n[run_all] done.")


if __name__ == "__main__":
    sys.exit(main())
