#!/usr/bin/env python3
"""
export_gt_frames.py
-------------------
Export single frames for human Ground Truth annotation.

Target ranges:
  201015  fr55–fr72   (face-on; approx first impact zone)
  201058  fr180–fr200 (down-the-line; true impact is after fr186)

Output: Desktop/gate1_gt/
  201015_fr055.jpg ... 201015_fr072.jpg
  201058_fr180.jpg ... 201058_fr200.jpg

Each JPEG: original resolution, frame number burned into top-left corner.
Human annotator will mark the true impact frame from these images.

Rule: Ground Truth only from human annotation — this script never auto-labels frames.
"""

import cv2
import sys
from pathlib import Path

INPUT = Path("/home/jason/projects/swingcue-postest/input")
OUT   = Path("/mnt/c/Users/jason/Desktop/gate1_gt")
OUT.mkdir(parents=True, exist_ok=True)

EXPORTS = [
    # (video_name,                       fr_start, fr_end_inclusive, label)
    ("Videos2026-06-09_201015_827.mp4",  55,  72,  "201015"),
    ("Videos2026-06-09_201058_697.mp4", 180, 200,  "201058"),
]

FONT      = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 1.0
THICKNESS  = 2
COLOR_BG   = (10, 10, 10)
COLOR_TEXT = (50, 240, 50)


def burn_label(frame, text):
    """Burn a frame-number label into top-left of frame (in-place)."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
    # Dark background rectangle
    cv2.rectangle(frame, (0, 0), (tw + 16, th + baseline + 12), COLOR_BG, -1)
    cv2.putText(frame, text, (8, th + 8), FONT, FONT_SCALE, COLOR_TEXT, THICKNESS, cv2.LINE_AA)


def export_range(vname, fr_start, fr_end, label):
    vpath = INPUT / vname
    if not vpath.exists():
        print(f"  ERROR: video not found: {vpath}")
        return 0

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        print(f"  ERROR: cannot open {vpath}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fr_end = min(fr_end, total_frames - 1)

    count = 0
    for fr_idx in range(fr_start, fr_end + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  WARNING: could not read frame {fr_idx} from {vname}")
            continue

        burn_label(frame, f"fr{fr_idx:03d}")
        out_path = OUT / f"{label}_fr{fr_idx:03d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        count += 1

    cap.release()
    return count


def main():
    print(f"export_gt_frames.py — exporting to {OUT}")
    total = 0
    for vname, fr_start, fr_end, label in EXPORTS:
        print(f"\n  {label}  fr{fr_start}–fr{fr_end}")
        n = export_range(vname, fr_start, fr_end, label)
        print(f"  -> {n} frames exported")
        total += n

    print(f"\nDone. {total} frames in {OUT}")
    print()
    print("HUMAN ACTION NEEDED:")
    print("  Look through gate1_gt/ and identify the true impact frame for:")
    print("  - 201015: which frame shows hands at ball contact (first swing, fr55-fr72)?")
    print("  - 201058: which frame shows hands at ball contact (fr180-fr200)?")
    print()
    print("GT Rule: these human-identified frame numbers are the ONLY valid Ground Truth.")
    print("         Never use detector output as GT.")


if __name__ == "__main__":
    main()
