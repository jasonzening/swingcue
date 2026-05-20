"""
overlay.py — render a runner's keypoints onto the original video.

Reads a RunResult JSON and the source mp4; writes an `overlay.mp4` with
the COCO 17 skeleton drawn on each sampled frame. Inputs that fall
between sampled frames carry forward the last sampled keypoints (zero-
order hold) — this matches what the production canvas overlay does, so
the visual quality reflects what the user will actually see.

CLI:
    python -m benchmark.overlay output/<runner>/<video_id>/keypoints.json \\
                                 test_videos/<video_id>.mp4 \\
                                 output/<runner>/<video_id>/overlay.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .runner import COCO_INDEX, COCO_NAMES

# Skeleton edges (COCO 17 standard connectivity).
SKELETON_EDGES: tuple[tuple[str, str], ...] = (
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
)

# Per-side hue assignments for L/R distinction.
COLOR_LEFT  = (255, 200,  60)   # BGR — yellow-cyan
COLOR_RIGHT = ( 60, 200, 255)   # BGR — orange
COLOR_MID   = (220, 220, 220)   # BGR — light gray (midline / face)
KP_RADIUS = 5
EDGE_THICKNESS = 3
TEXT_COLOR = (255, 255, 255)


def _edge_color(a: str, b: str) -> tuple[int, int, int]:
    if a.startswith("left_") or b.startswith("left_"):
        return COLOR_LEFT
    if a.startswith("right_") or b.startswith("right_"):
        return COLOR_RIGHT
    return COLOR_MID


def _kp_color(name: str) -> tuple[int, int, int]:
    if name.startswith("left_"):
        return COLOR_LEFT
    if name.startswith("right_"):
        return COLOR_RIGHT
    return COLOR_MID


def draw_skeleton(
    frame_bgr: np.ndarray,
    kps: dict[str, list[Any]],
) -> None:
    """Mutates frame in place. kps[name] = [x|None, y|None, conf]."""
    # Edges first (under dots).
    for a, b in SKELETON_EDGES:
        pa = kps.get(a)
        pb = kps.get(b)
        if not pa or not pb:
            continue
        if pa[0] is None or pb[0] is None:
            continue
        cv2.line(
            frame_bgr,
            (int(pa[0]), int(pa[1])),
            (int(pb[0]), int(pb[1])),
            _edge_color(a, b),
            EDGE_THICKNESS,
            cv2.LINE_AA,
        )
    # Dots on top.
    for name in COCO_NAMES:
        kp = kps.get(name)
        if not kp or kp[0] is None:
            continue
        cv2.circle(
            frame_bgr,
            (int(kp[0]), int(kp[1])),
            KP_RADIUS,
            _kp_color(name),
            -1,
            cv2.LINE_AA,
        )


def _label_overlay(
    frame_bgr: np.ndarray,
    runner_name: str,
    frame_idx: int,
    ts: float,
) -> None:
    h = frame_bgr.shape[0]
    label = f"{runner_name}  f={frame_idx:>4}  t={ts:>5.2f}s"
    cv2.rectangle(frame_bgr, (8, h - 38), (8 + 9 * len(label), h - 8),
                  (0, 0, 0), -1)
    cv2.putText(
        frame_bgr, label, (12, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA,
    )


def render_overlay(
    run_json: Path,
    video_path: Path,
    out_path: Path,
) -> None:
    data = json.loads(run_json.read_text())
    runner_name: str = data["runner"]
    width:   int = int(data["video_width"])
    height:  int = int(data["video_height"])
    sampled = sorted(data["frames"], key=lambda f: f["ts"])
    if not sampled:
        print(f"[overlay] {runner_name}: empty frames — nothing to render")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if src_w != width or src_h != height:
        print(
            f"[overlay] {runner_name}: video dims ({src_w}x{src_h}) "
            f"≠ run dims ({width}x{height}); kp may not line up. "
            f"Using video dims for canvas, keeping kp at native coords."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (src_w, src_h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer at {out_path}")

    # Zero-order-hold lookup: for each output frame at native fps, find
    # the most recent sampled frame whose ts ≤ current ts.
    sampled_idx = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cur_ts = frame_idx / src_fps
        # Advance the sampled cursor while next sample is still in the past.
        while (sampled_idx + 1 < len(sampled)
               and sampled[sampled_idx + 1]["ts"] <= cur_ts):
            sampled_idx += 1
        s = sampled[sampled_idx]
        draw_skeleton(frame, s["keypoints"])
        _label_overlay(frame, runner_name, s["frame_idx"], s["ts"])
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[overlay] {runner_name}: wrote {frame_idx} frames → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_json", type=Path)
    ap.add_argument("video",    type=Path)
    ap.add_argument("out",      type=Path)
    args = ap.parse_args()
    render_overlay(args.run_json, args.video, args.out)


if __name__ == "__main__":
    sys.exit(main())
