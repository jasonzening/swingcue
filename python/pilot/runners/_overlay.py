"""
_overlay.py — render 2D back-projection of WHAM joints onto source video.

Local-only renderer (no Modal cost). Reads:
  - python/pilot/output/wham/<video_id>/joint_centers_3d.json
  - python/benchmark/test_videos/<video_id>.mp4  (or any local source)

Projects WHAM's camera-frame 3D joint centers onto each video frame via
pinhole projection, draws an H36M-style skeleton, writes overlay.mp4
next to the JSON.

Camera-intrinsics assumption: WHAM's demo.py uses a default focal
length of `max(W, H)` and principal point at image center when no
camera calibration is provided. We mirror that here so the projection
lines up with what WHAM actually saw at inference time.

CLI:
    python -m pilot.runners._overlay <video_id>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# H36M-style skeleton edges (pairs of joint names that should be
# connected by a line). The names match the PilotRunResult keys filled
# in by wham_runner.py via H36M_TO_PILOT_NAME.
SKELETON_EDGES: tuple[tuple[str, str], ...] = (
    # spine chain
    ("pelvis", "spine1"),
    ("spine1", "neck"),
    ("neck", "head"),
    # arms (shoulders → elbows → wrists)
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    # legs (hips → knees → ankles)
    ("pelvis", "left_hip"),
    ("pelvis", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)

# Per-side dot colors (BGR). Matches SwingCue production convention:
#   right_* → orange/yellow
#   left_*  → cyan/blue
#   spine/head/neck/pelvis (midline) → grey
# Edges are uniformly grey (matches the production SkeletonOverlay.tsx
# rendering that uses '#999999' for all bone strokes). The L/R color
# distinction is carried by the joint dots only — colored edges added
# visual noise in the first phase2b cut and Jason called it out.
COLOR_LEFT  = (255, 200,  60)   # BGR — cyan/blue
COLOR_RIGHT = ( 60, 200, 255)   # BGR — orange
COLOR_MID   = (180, 180, 180)   # BGR — neutral grey (midline dots)
COLOR_EDGE  = (153, 153, 153)   # BGR — uniform grey for all skeleton bones
KP_RADIUS = 6
EDGE_THICKNESS = 3


def _dot_color(name: str) -> tuple[int, int, int]:
    """Color one keypoint dot by SwingCue convention."""
    if "left" in name:
        return COLOR_LEFT
    if "right" in name:
        return COLOR_RIGHT
    return COLOR_MID


def _project(joints_3d: dict, fx: float, fy: float, cx: float, cy: float) -> dict:
    """
    Pinhole projection: (X, Y, Z) in camera frame → (u, v) in pixel coords.
    Returns a dict of name → (u, v) for every joint with valid 3D coords.

    WHAM's camera-frame y is "down in image", z is depth from camera.
    Standard pinhole: u = fx*X/Z + cx, v = fy*Y/Z + cy.
    """
    out = {}
    for name, xyz in joints_3d.items():
        if xyz is None or len(xyz) != 3:
            continue
        X, Y, Z = xyz
        if Z is None or Z <= 0:
            continue
        u = fx * X / Z + cx
        v = fy * Y / Z + cy
        out[name] = (int(round(u)), int(round(v)))
    return out


def render_overlay(json_path: Path, video_path: Path, out_path: Path) -> None:
    data = json.loads(json_path.read_text())
    frames = data["frames"]
    if not frames:
        raise SystemExit(f"no frames in {json_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"could not open {video_path}")
    fps_native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_native = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # WHAM default intrinsics — empirically `max(W, H)` focal length +
    # image-center principal point. If your video used a different
    # focal length at WHAM inference time, the projection here will be
    # rescaled accordingly. For phase2b smoke this default is what
    # demo.py used.
    fx = fy = float(max(W, H))
    cx, cy = W / 2.0, H / 2.0
    print(
        f"[_overlay] {W}x{H} @ {fps_native:.2f}fps, {n_native} native frames, "
        f"{len(frames)} WHAM frames; intrinsics fx=fy={fx} cx={cx} cy={cy}"
    )

    # Index WHAM output by frame_idx so we can look up per native frame.
    by_frame_idx: dict[int, dict] = {}
    for f in frames:
        by_frame_idx[int(f["frame_idx"])] = f

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_native, (W, H))
    if not writer.isOpened():
        raise SystemExit(f"could not open writer for {out_path}")

    # Sub-title font for legend overlay.
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    written = 0
    last_drawn_pts: dict | None = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # WHAM may have skipped frames; carry forward last detection.
        fr = by_frame_idx.get(written)
        if fr is not None:
            pts = _project(fr["joint_centers_3d"], fx, fy, cx, cy)
            last_drawn_pts = pts
        else:
            pts = last_drawn_pts or {}

        # Draw skeleton edges first (so dots sit on top). All edges
        # uniform grey per SwingCue production convention.
        for a, b in SKELETON_EDGES:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], COLOR_EDGE, EDGE_THICKNESS, cv2.LINE_AA)

        # Draw dots — colored by L/R/midline.
        for name, (u, v) in pts.items():
            color = _dot_color(name)
            cv2.circle(frame, (u, v), KP_RADIUS, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (u, v), KP_RADIUS + 1, (0, 0, 0), 1, cv2.LINE_AA)

        # Top-left label.
        label = f"WHAM bone-center  frame {written}/{n_native - 1}  ({len(pts)} kp)"
        cv2.rectangle(frame, (8, 8), (8 + 11 * len(label), 36), (0, 0, 0), -1)
        cv2.putText(frame, label, (12, 28), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(frame)
        written += 1

    cap.release()
    writer.release()
    sz_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[_overlay] wrote {written} frames → {out_path} ({sz_mb:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id")
    ap.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help=(
            "Source video file. Defaults to "
            "python/benchmark/test_videos/<video_id>.mp4"
        ),
    )
    ap.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help=(
            "Output mp4. Defaults to "
            "python/pilot/output/wham/<video_id>/overlay.mp4"
        ),
    )
    args = ap.parse_args()

    json_path = Path("python/pilot/output/wham") / args.video_id / "joint_centers_3d.json"
    if not json_path.exists():
        raise SystemExit(f"missing pilot output: {json_path}")

    video_path = args.video_path or Path(
        f"python/benchmark/test_videos/{args.video_id}.mp4"
    )
    if not video_path.exists():
        raise SystemExit(f"missing source video: {video_path}")

    out_path = args.out_path or json_path.with_name("overlay.mp4")
    render_overlay(json_path, video_path, out_path)


if __name__ == "__main__":
    sys.exit(main())
