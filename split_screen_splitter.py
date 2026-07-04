#!/usr/bin/env python3
"""
split_screen_splitter.py — v0.1
分屏对比视频切分 + 勾叉标识识别

Standalone — no engine imports.
VLM results pre-computed from vision analysis of mid-frames.
"""

import json, os
from pathlib import Path
import cv2

INPUT_DIR = Path("/mnt/c/Users/jason/Zening/Swingcue/教学视频/dtl-1")
OUT_BASE  = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/split_check")
OUT_BASE.mkdir(parents=True, exist_ok=True)

# VLM analysis results (hardcoded from vision analysis of mid-frames 2026-07-03)
# split_x_frac: fraction of frame width where vertical split occurs
# Marker convention:  checkmark=✓(green)  cross=✗(red)  OK=O(Korean正确)  none  unknown
VLM_RESULTS = {
    "dtl-1": {
        "split_screen": False, "split_x_frac": None,
        "left_marker": None, "right_marker": None,
        "center_marker": "cross",
        "note": "单面板, 上方中央红圈×标识",
        "verdict": "needs_human",
    },
    "dtl-2": {
        "split_screen": False, "split_x_frac": None,
        "left_marker": None, "right_marker": None,
        "center_marker": "cross",
        "note": "单面板, 上方中央红圈×标识",
        "verdict": "needs_human",
    },
    "dtl-3": {
        "split_screen": True, "split_x_frac": 0.389,
        "left_marker": "checkmark", "right_marker": "cross",
        "note": "左正确(✓/正确) / 右错误(✗/错误), 中文教学",
        "verdict": "PASS",
    },
    "dtl-4": {
        "split_screen": True, "split_x_frac": 0.375,
        "left_marker": "checkmark", "right_marker": "cross",
        "note": "左正确(✓/正确) / 右错误(✗/错误), 中文教学",
        "verdict": "PASS",
    },
    "dtl-5": {
        "split_screen": True, "split_x_frac": 0.381,
        "left_marker": "cross", "right_marker": "checkmark",
        "note": "左错误(✗/错误) / 右正确(✓/正确), 中文教学, 顺序与dtl-3/4相反",
        "verdict": "PASS",
    },
    "dlt-6": {
        "split_screen": True, "split_x_frac": 0.380,
        "left_marker": "cross", "right_marker": "OK",
        "note": "左错误(×) / 右正确(O), KORSA Golf Academy, 韩国O/X标识",
        "verdict": "PASS",
    },
    "dtl-7": {
        "split_screen": True, "split_x_frac": 0.385,
        "left_marker": "cross", "right_marker": "OK",
        "note": "左错误(×) / 右正确(O), KORSA Golf Academy, 韩国O/X标识",
        "verdict": "PASS",
    },
    "dtl-8": {
        "split_screen": False, "split_x_frac": None,
        "left_marker": None, "right_marker": None,
        "center_marker": None,
        "note": "单面板, 无标识, 女性干净单镜头",
        "verdict": "needs_human",
    },
}

FILE_ORDER = ["dtl-1","dtl-2","dtl-3","dtl-4","dtl-5","dlt-6","dtl-7","dtl-8"]


def split_video(video_path, split_x, out_dir):
    cap = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_l = cv2.VideoWriter(str(out_dir / "left.mp4"),  fourcc, fps, (split_x, h))
    out_r = cv2.VideoWriter(str(out_dir / "right.mp4"), fourcc, fps, (w - split_x, h))
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_l.write(frame[:, :split_x])
        out_r.write(frame[:, split_x:])
        count += 1
    cap.release(); out_l.release(); out_r.release()
    print(f"    split done: left={split_x}px right={w-split_x}px  {count}/{total}fr")


def draw_annotated_midframe(video_path, stem, vlm, out_path):
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"    WARN: could not read mid-frame")
        return

    if vlm["split_screen"]:
        sx = int(w * vlm["split_x_frac"])
        # Split line
        cv2.line(frame, (sx, 0), (sx, h), (0, 255, 255), 4)

        lm = vlm.get("left_marker")
        rm = vlm.get("right_marker")

        def badge(img, cx, cy, mtype):
            r = min(55, h // 10)
            col = (0, 180, 0) if mtype in ("checkmark", "OK") else (30, 30, 210)
            cv2.circle(img, (cx, cy), r, col, -1)
            cv2.circle(img, (cx, cy), r, (255,255,255), 3)
            lbl = "OK" if mtype in ("checkmark", "OK") else "X"
            fs = 1.2
            (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, fs, 3)
            cv2.putText(img, lbl, (cx - tw//2, cy + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), 3)

        cy = max(70, h // 12)
        if lm:
            badge(frame, sx // 2, cy, lm)
        if rm:
            badge(frame, sx + (w - sx) // 2, cy, rm)

        cv2.putText(frame, f"split_x={sx}px ({vlm['split_x_frac']:.3f})",
                    (sx + 5, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.putText(frame, "needs_human: no split / single panel",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 3)
        cm = vlm.get("center_marker")
        if cm == "cross":
            cx, cy = w // 2, h // 4
            cv2.circle(frame, (cx, cy), 60, (30, 30, 210), -1)
            cv2.putText(frame, "X", (cx - 22, cy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255,255,255), 4)

    # Bottom bar
    cv2.rectangle(frame, (0, h-55), (w, h), (0,0,0), -1)
    cv2.putText(frame, f"{stem}  verdict={vlm['verdict']}  fr{n//2}/{n}",
                (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

    cv2.imwrite(str(out_path), frame)
    print(f"    annotated: {out_path.name}")


def main():
    results = []
    for stem in FILE_ORDER:
        # Find actual file (handle dlt-6 typo in filename)
        fname = stem + ".mp4"
        video_path = INPUT_DIR / fname
        if not video_path.exists():
            print(f"\nSKIP {fname}: not found")
            continue

        vlm = VLM_RESULTS[stem]
        out_dir = OUT_BASE / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*55}")
        print(f"{stem}: split={vlm['split_screen']}  verdict={vlm['verdict']}")
        print(f"  left_marker={vlm.get('left_marker')}  right_marker={vlm.get('right_marker')}")
        print(f"  note: {vlm['note']}")

        cap_tmp = cv2.VideoCapture(str(video_path))
        w = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_tmp.release()

        left_path = right_path = None
        split_x_px = None

        if vlm["split_screen"]:
            split_x_px = int(w * vlm["split_x_frac"])
            split_video(video_path, split_x_px, out_dir)
            left_path  = str(out_dir / "left.mp4")
            right_path = str(out_dir / "right.mp4")

        ann_path = OUT_BASE / f"{stem}_midframe_annotated.jpg"
        draw_annotated_midframe(video_path, stem, vlm, ann_path)

        results.append({
            "file": fname, "stem": stem,
            "split_screen": vlm["split_screen"],
            "verdict": vlm["verdict"],
            "split_x_px": split_x_px,
            "left_marker":   vlm.get("left_marker"),
            "right_marker":  vlm.get("right_marker"),
            "center_marker": vlm.get("center_marker"),
            "note": vlm["note"],
            "left_path":  left_path,
            "right_path": right_path,
            "annotated_frame": str(ann_path),
        })

    json_path = OUT_BASE / "split_check_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON: {json_path}")

    print("\n\n=== 切分汇报表 ===")
    print(f"{'视频':8s} {'分屏':5s} {'left标识':12s} {'right标识':12s} {'verdict':14s}  说明")
    print("-"*95)
    for r in results:
        ls = r["left_marker"] or r.get("center_marker") or "—"
        rs = r["right_marker"] or "—"
        print(f"{r['stem']:8s} {'是' if r['split_screen'] else '否':5s} "
              f"{ls:12s} {rs:12s} {r['verdict']:14s}  {r['note'][:40]}")

    print(f"\nWindows路径: C:\\Users\\jason\\Desktop\\rtmpose_results\\preview\\split_check\\")

    return results


if __name__ == "__main__":
    main()
