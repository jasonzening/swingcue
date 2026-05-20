"""
compare.py — assemble side-by-side comparison video.

Reads multiple `overlay.mp4` files (one per runner, all for the same
source video) and produces `comparison_<video_id>.mp4`: each overlay
shrunk to a horizontal strip, all stacked horizontally. The
per-runner label drawn by overlay.py stays visible in each strip.

CLI:
    python -m benchmark.compare <video_id> <overlay1.mp4> <overlay2.mp4> [overlay3.mp4 …]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def make_comparison(
    video_id: str,
    overlay_paths: list[Path],
    out_path: Path,
) -> None:
    if not overlay_paths:
        raise ValueError("need at least one overlay")

    caps = [cv2.VideoCapture(str(p)) for p in overlay_paths]
    for c, p in zip(caps, overlay_paths):
        if not c.isOpened():
            raise RuntimeError(f"could not open {p}")

    fps    = caps[0].get(cv2.CAP_PROP_FPS) or 30.0
    height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    widths = [int(c.get(cv2.CAP_PROP_FRAME_WIDTH)) for c in caps]
    total_w = sum(widths)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (total_w, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer at {out_path}")

    n_frames = 0
    while True:
        frames: list[np.ndarray] = []
        all_ok = True
        for c in caps:
            ok, f = c.read()
            if not ok:
                all_ok = False
                break
            frames.append(f)
        if not all_ok:
            break
        # Concat horizontally. Heights are assumed equal (overlay.py
        # preserves source dims). Pad if any differ.
        if any(f.shape[0] != height for f in frames):
            for i, f in enumerate(frames):
                if f.shape[0] != height:
                    pad = np.zeros((height, f.shape[1], 3), dtype=np.uint8)
                    pad[:f.shape[0]] = f
                    frames[i] = pad
        merged = np.hstack(frames)
        writer.write(merged)
        n_frames += 1

    for c in caps:
        c.release()
    writer.release()
    print(f"[compare] {video_id}: {n_frames} frames → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id")
    ap.add_argument("overlays", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: output/comparison_<video_id>.mp4")
    args = ap.parse_args()
    out = args.out or (Path("output") / f"comparison_{args.video_id}.mp4")
    make_comparison(args.video_id, args.overlays, out)


if __name__ == "__main__":
    sys.exit(main())
