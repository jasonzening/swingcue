"""
extract_phase_frames.py — pull 5 phase-representative frames from a
swing video so Jason can hand-label them for PR-7b ground truth.

Per PR-7_GOLF_CORRECTION_LAYER_SPEC_v2.md §9: minimum 15 ground-truth
samples = 3 videos × 5 phases. This script does the per-video frame
extraction step.

Phase detection: the production pose_timeline_2d JSONB has actual phase
markers, but locally we don't have a copy. This script uses heuristic
fractional timestamps as a first cut — Jason can re-pick `--frame` when
running the labeler if the suggested frames don't match the desired
phase moment.

Default fractional positions:
    setup       =  5%    (player settled, club grounded)
    top         = 40%    (club shaft at highest point)
    transition  = 50%    (mid-downswing, shaft roughly parallel)
    impact      = 65%    (ball contact moment)
    finish      = 90%    (rotation complete)

CLI:
    python -m pilot.scripts.extract_phase_frames \\
        --video-id b3fea3f0-e248-44d7-a923-0bb43172b5bf \\
        --video-path python/benchmark/test_videos/b3fea3f0-e248-44d7-a923-0bb43172b5bf.mp4

Output: PNG files at docs/PR-7_GROUND_TRUTH/frames/<short_id>_<phase>_f<idx>.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2


# Phase → fractional position in video (0.0 = first frame, 1.0 = last).
DEFAULT_PHASE_FRACTIONS: dict[str, float] = {
    "setup":      0.05,
    "top":        0.40,
    "transition": 0.50,
    "impact":     0.65,
    "finish":     0.90,
}

DEFAULT_OUTPUT_DIR = Path("docs/PR-7_GROUND_TRUTH/frames")


def short_id(video_id: str) -> str:
    """First 8 chars of the UUID — matches the naming convention in the
    spec §9 output paths."""
    return video_id.split("-")[0]


def extract_frames(
    video_id: str,
    video_path: Path,
    out_dir: Path,
    phases: list[str],
) -> list[tuple[str, int, Path]]:
    """
    Returns list of (phase, frame_idx, png_path) for each successfully
    extracted phase frame.
    """
    if not video_path.exists():
        raise SystemExit(f"missing video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n_frames / fps if fps > 0 else 0.0
    print(
        f"[extract] {video_path}  {width}x{height} @ {fps:.2f}fps  "
        f"{n_frames} frames  ({duration:.2f}s)"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    sid = short_id(video_id)
    results: list[tuple[str, int, Path]] = []

    for phase in phases:
        if phase not in DEFAULT_PHASE_FRACTIONS:
            print(f"[extract]   ! unknown phase '{phase}', skipping")
            continue
        frac = DEFAULT_PHASE_FRACTIONS[phase]
        # Clamp 0..n_frames-1 so impact at 100% doesn't try to seek
        # past EOF.
        frame_idx = min(int(round(frac * n_frames)), n_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[extract]   ! read failed at frame_idx={frame_idx} for {phase}")
            continue
        png_path = out_dir / f"{sid}_{phase}_f{frame_idx:03d}.png"
        cv2.imwrite(str(png_path), frame)
        sz_kb = png_path.stat().st_size / 1024
        print(
            f"[extract]   {phase:<12} frame_idx={frame_idx:>4}  "
            f"({frac*100:>4.0f}% of clip)  → {png_path}  ({sz_kb:.1f} KB)"
        )
        results.append((phase, frame_idx, png_path))

    cap.release()
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", required=True, help="Full UUID; first 8 chars used for filenames")
    ap.add_argument("--video-path", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument(
        "--phases",
        default="setup,top,transition,impact,finish",
        help="Comma-separated subset of: setup,top,transition,impact,finish",
    )
    args = ap.parse_args()

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    results = extract_frames(args.video_id, args.video_path, args.output_dir, phases)

    if not results:
        raise SystemExit("[extract] no frames written")

    # Tail summary in the format the labeler CLI expects, so Jason can
    # copy-paste straight into the next command.
    print()
    print("[extract] To label these frames, run for each phase:")
    for phase, frame_idx, png_path in results:
        print(
            f"  python -m pilot.scripts.ground_truth_labeler \\\n"
            f"      --video-id {args.video_id} \\\n"
            f"      --phase {phase} \\\n"
            f"      --frame {frame_idx} \\\n"
            f"      --image {png_path}"
        )


if __name__ == "__main__":
    sys.exit(main())
