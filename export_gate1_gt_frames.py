#!/usr/bin/env python3
"""
export_gate1_gt_frames.py
导出 GT 标注素材:
  - 201058 fr180-fr200 (每帧一张, 原比例, 标帧号)
  - 201015 fr55-fr72   (每帧一张, 原比例, 标帧号)
放桌面 gate1_gt/
"""
import sys, cv2, numpy as np
from pathlib import Path

sys.path.insert(0, "/home/jason/projects/swingcue-postest")

INPUT  = Path("/home/jason/projects/swingcue-postest/input")
DEST   = Path("/mnt/c/Users/jason/Desktop/gate1_gt")
DEST.mkdir(parents=True, exist_ok=True)

EXPORTS = [
    ("Videos2026-06-09_201058_697.mp4", range(180, 201), "201058"),
    ("Videos2026-06-09_201015_827.mp4", range(55, 73),   "201015"),
]

FONT = cv2.FONT_HERSHEY_DUPLEX

def export_frames(vname, frame_range, stem):
    vpath = str(INPUT / vname)
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        print(f"  ERROR: cannot open {vpath}")
        return 0

    out_dir = DEST / stem
    out_dir.mkdir(exist_ok=True)
    count = 0

    for fi in frame_range:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            print(f"  WARNING: fr{fi} read failed")
            continue

        # 原比例, 仅加帧号标注
        h, w = frame.shape[:2]
        label = f"fr{fi:04d}"

        # 黑底白字, 右上角
        lbl_x = w - 120; lbl_y = 36
        cv2.rectangle(frame, (lbl_x - 4, 4), (w - 4, 46), (0, 0, 0), -1)
        cv2.putText(frame, label, (lbl_x, lbl_y), FONT, 1.0, (255, 255, 255), 2)

        out_path = out_dir / f"{stem}_fr{fi:04d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        count += 1

    cap.release()
    return count

print("Exporting GT annotation frames to Desktop/gate1_gt/")
for vname, frame_range, stem in EXPORTS:
    n = export_frames(vname, frame_range, stem)
    print(f"  {stem}: exported {n} frames -> gate1_gt/{stem}/")

print("\nDone. Please annotate the true impact frame in each set.")
print("  201058: which frame in fr180-200 is the actual ball strike?")
print("  201015: which frame in fr55-72 is the actual ball strike?")
print("After annotation, provide the frame number to calibrate gate-1.")
