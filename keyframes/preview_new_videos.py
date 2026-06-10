#!/usr/bin/env python3
"""
preview_new_videos.py
For each video: extract 4 frames (~address/top/impact/finish),
tile into a 2x2 preview, annotate with filename + detected angle.
"""

import cv2
import numpy as np
from pathlib import Path

INPUT = Path("/home/jason/projects/swingcue-postest/input")
DESK  = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview")
DESK.mkdir(parents=True, exist_ok=True)

VIDEOS = sorted(INPUT.glob("Videos2026-06-09*.mp4"))

LABELS = ["address", "top", "impact", "finish"]
# Approximate fractions of video where each phase occurs
FRACS = [0.10, 0.35, 0.60, 0.88]

TILE_W, TILE_H = 480, 360   # each tile size
FONT = cv2.FONT_HERSHEY_DUPLEX


def get_frame(cap, frac):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx   = max(0, min(int(total * frac), total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, f = cap.read()
    return f if ret else None, idx


def detect_angle(cap):
    """
    Rough face-on vs down-the-line heuristic:
    Sample a mid-swing frame, look at horizontal extent of motion relative
    to frame width.  Face-on: golfer faces camera → body occupies ~centre
    horizontally.  DTL: golfer side-on → club/body sweeps horizontally.
    We use a simpler proxy: check aspect ratio of the "motion region" via
    frame difference between two nearby frames.
    Returns "face-on" or "down-the-line".
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30

    # Grab two frames ~1/3 and ~2/3 through
    frames = []
    for frac in (0.30, 0.55):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac))
        ret, f = cap.read()
        if ret:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))

    if len(frames) < 2:
        return "unknown"

    diff = cv2.absdiff(frames[0], frames[1])
    _, mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

    # Bounding box of motion region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "unknown"

    all_pts = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(all_pts)
    aspect = w / (h + 1e-6)

    # DTL: swing arc is wider horizontally → aspect > 1.0
    # Face-on: body stays roughly vertical → aspect ≤ 1.0
    return "down-the-line" if aspect > 1.05 else "face-on"


def make_preview(video_path: Path) -> Path:
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30

    angle = detect_angle(cap)

    tiles = []
    for label, frac in zip(LABELS, FRACS):
        frame, fidx = get_frame(cap, frac)
        if frame is None:
            frame = np.zeros((H, W, 3), dtype=np.uint8)

        tile = cv2.resize(frame, (TILE_W, TILE_H))

        # Label banner
        banner = np.zeros((40, TILE_W, 3), dtype=np.uint8)
        banner[:] = (30, 30, 30)
        cv2.putText(banner, f"{label.upper()}  fr{fidx}",
                    (8, 27), FONT, 0.65, (200, 230, 200), 1, cv2.LINE_AA)
        tile = np.vstack([banner, tile])
        tiles.append(tile)

    cap.release()

    # 2×2 grid
    row0 = np.hstack(tiles[:2])
    row1 = np.hstack(tiles[2:])
    grid = np.vstack([row0, row1])

    # Top header bar
    header_h = 52
    header   = np.zeros((header_h, grid.shape[1], 3), dtype=np.uint8)
    header[:] = (20, 20, 20)

    short_name = video_path.name[:40]
    angle_color = (80, 210, 80) if "face" in angle else (80, 180, 255)
    cv2.putText(header, short_name,
                (10, 22), FONT, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(header, f"[{angle}]  {W}x{H}  {total}fr @{fps:.0f}fps",
                (10, 44), FONT, 0.55, angle_color, 1, cv2.LINE_AA)

    canvas = np.vstack([header, grid])

    stem = video_path.stem[-18:]   # last 18 chars of name
    out  = DESK / f"preview_{stem}.png"
    cv2.imwrite(str(out), canvas)
    print(f"  {video_path.name}  → [{angle}]  saved: {out.name}")
    return out


def main():
    print(f"Processing {len(VIDEOS)} videos...\n")
    for v in VIDEOS:
        make_preview(v)
    print("\nAll previews on desktop.")


if __name__ == "__main__":
    main()
