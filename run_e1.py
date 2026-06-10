#!/usr/bin/env python3
"""
run_e1.py — E1 top-level runner

For each of the 5 new videos:
  1. A layer: run RTMPose (or load cached JSON)
  2. B layer: detect 8-phase annotations
  3. Output: phase summary sheet (gate-1 review image)

Gate-1 output: keyframes/gate1_preview/ on desktop
"""

import cv2
import numpy as np
import os
import sys
import json
import time
from pathlib import Path
from typing import List

sys.path.insert(0, "/home/jason/projects/swingcue-postest")

from engine.a_measurement.pose_pipeline import PosePipeline, FrameMeasurement
from engine.b_phase.swing_phase import SwingPhaseEngine, PhaseAnnotation, AnchorFrames, PHASE_NAMES

INPUT   = Path("/home/jason/projects/swingcue-postest/input")
KP_CACHE = Path("/home/jason/projects/swingcue-postest/engine/kp_cache")
GATE1   = Path("/home/jason/projects/swingcue-postest/keyframes/gate1_preview")
DESK    = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/gate1")
KP_CACHE.mkdir(parents=True, exist_ok=True)
GATE1.mkdir(parents=True, exist_ok=True)
DESK.mkdir(parents=True, exist_ok=True)

VIDEOS = {
    "Videos2026-06-09_201015_827.mp4": "face-on",
    "Videos2026-06-09_201039_231.mp4": "face-on",
    "Videos2026-06-09_201047_915.mp4": "face-on",
    "Videos2026-06-09_201054_561.mp4": "down-the-line",
    "Videos2026-06-09_201058_697.mp4": "down-the-line",
}

PHASE_COLORS = {
    "address":      (120, 120, 120),
    "takeaway":     (200, 150,  50),
    "backswing":    (200, 100,  30),
    "top":          (50,  50,  220),
    "transition":   (180,  50, 180),
    "downswing":    (50,  180, 220),
    "impact":       (50,  220,  50),
    "follow_through": (100, 200, 100),
}

FONT = cv2.FONT_HERSHEY_DUPLEX


# ── helpers ──────────────────────────────────────────────────────────────────

def get_frame(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, f = cap.read(); cap.release()
    return f if ret else np.zeros((400, 300, 3), np.uint8)


def phase_summary(annotations: List[PhaseAnnotation]) -> dict:
    """Return {phase: (start_fr, end_fr, frame_count)} for each phase that appears."""
    summary = {}
    for ann in annotations:
        p = ann.phase
        if p not in summary:
            summary[p] = [ann.frame_idx, ann.frame_idx, 0]
        else:
            summary[p][1] = ann.frame_idx
        summary[p][2] += 1
    return {p: (v[0], v[1], v[2]) for p, v in summary.items()}


def representative_frame(annotations: List[PhaseAnnotation], phase: str) -> int:
    """Middle frame of that phase."""
    frames = [a.frame_idx for a in annotations if a.phase == phase]
    if not frames: return 0
    return frames[len(frames)//2]


def make_phase_sheet(video_path: str, stem: str, angle: str,
                     annotations: List[PhaseAnnotation],
                     anchors: AnchorFrames, fps: float) -> np.ndarray:
    """
    Build a phase summary sheet:
      - Top: timeline bar (colored by phase, full width)
      - Middle: 8 thumbnail tiles (one per phase), original aspect ratio scaled
      - Bottom: text table (phase, fr_start, fr_end, duration_ms)
    """
    n = len(annotations)
    SHEET_W = 1440

    # ── Timeline bar ──────────────────────────────────────────────────────────
    bar_h = 50
    timeline = np.zeros((bar_h, SHEET_W, 3), np.uint8); timeline[:] = (20,20,20)
    for ann in annotations:
        x = int(ann.frame_idx / n * SHEET_W)
        color = PHASE_COLORS[ann.phase]
        timeline[:, x:x+2] = color

    # Phase labels on timeline
    summary = phase_summary(annotations)
    for phase in PHASE_NAMES:
        if phase not in summary: continue
        s, e, _ = summary[phase]
        x0 = int(s / n * SHEET_W); x1 = int(e / n * SHEET_W)
        color = PHASE_COLORS[phase]
        cv2.putText(timeline, phase[:4].upper(),
                    (x0+2, bar_h-8), FONT, 0.42, color, 1)

    # Anchor markers
    for name, fr, c in [("A", anchors.address, (180,180,180)),
                         ("T", anchors.top,     (100,100,255)),
                         ("I", anchors.impact,  (80,255,80)),
                         ("F", anchors.finish,  (180,100,180))]:
        x = int(fr / n * SHEET_W)
        cv2.line(timeline, (x,0), (x,bar_h), c, 2)
        cv2.putText(timeline, name, (x+2, 14), FONT, 0.45, c, 1)

    # ── Thumbnail strip ───────────────────────────────────────────────────────
    THUMB_W = SHEET_W // 8
    THUMB_H = int(THUMB_W * 16 / 9)   # 9:16 portrait

    thumbs = []
    for phase in PHASE_NAMES:
        fi = representative_frame(annotations, phase)
        frame = get_frame(video_path, fi)
        # Crop to center (portrait: keep full height, center-crop width)
        fh, fw = frame.shape[:2]
        target_aspect = 16 / 9
        if fh / fw > target_aspect:
            # Too tall: crop height
            new_h = int(fw * target_aspect)
            y0 = (fh - new_h) // 2
            frame = frame[y0:y0+new_h, :]
        thumb = cv2.resize(frame, (THUMB_W, THUMB_H))

        # Color border
        color = PHASE_COLORS[phase]
        cv2.rectangle(thumb, (0,0), (THUMB_W-1, THUMB_H-1), color, 4)

        # Phase label banner
        banner = np.zeros((38, THUMB_W, 3), np.uint8); banner[:] = (15,15,15)
        cv2.putText(banner, phase.upper(), (4,22), FONT, 0.50, color, 1)
        if phase in summary:
            s, e, cnt = summary[phase]
            dur_ms = int((e-s)/fps*1000)
            cv2.putText(banner, f"fr{s}-{e} {dur_ms}ms", (4,34), FONT, 0.38, (160,160,160), 1)

        # Anchor indicator
        if phase in summary:
            s, e, _ = summary[phase]
            anchor_fi = {anchors.address: "ADDR", anchors.top: "TOP",
                         anchors.impact: "IMP", anchors.finish: "FIN"}
            for afr, alabel in anchor_fi.items():
                if s <= afr <= e:
                    ax = int((afr - s) / max(e-s,1) * THUMB_W)
                    cv2.line(thumb, (ax, 0), (ax, THUMB_H), (255,255,255), 1)

        thumbs.append(np.vstack([banner, thumb]))

    strip = np.hstack(thumbs)

    # ── Text table ────────────────────────────────────────────────────────────
    row_h = 28; cols = 5
    table_h = (len(PHASE_NAMES)+2) * row_h
    table = np.zeros((table_h, SHEET_W, 3), np.uint8); table[:] = (18,18,18)

    headers = ["Phase", "Start fr", "End fr", "Frames", "Duration"]
    for ci, h in enumerate(headers):
        cv2.putText(table, h, (12 + ci*(SHEET_W//cols), row_h-6),
                    FONT, 0.52, (200,200,200), 1)
    cv2.line(table, (0,row_h), (SHEET_W, row_h), (60,60,60), 1)

    for ri, phase in enumerate(PHASE_NAMES):
        y = (ri+2)*row_h - 6
        color = PHASE_COLORS[phase]
        if phase in summary:
            s, e, cnt = summary[phase]; dur_ms = int((e-s)/fps*1000)
            vals = [phase, str(s), str(e), str(cnt), f"{dur_ms}ms"]
        else:
            vals = [phase, "-", "-", "0", "-"]
        for ci, v in enumerate(vals):
            cv2.putText(table, v, (12 + ci*(SHEET_W//cols), y), FONT, 0.50, color, 1)

    # ── Header ────────────────────────────────────────────────────────────────
    hdr = np.zeros((54, SHEET_W, 3), np.uint8); hdr[:] = (25,25,25)
    cv2.putText(hdr, f"{stem}  [{angle}]  {n}fr @{fps:.0f}fps",
                (10, 26), FONT, 0.70, (220,220,220), 1)
    cv2.putText(hdr,
                f"Anchors: address=fr{anchors.address}  top=fr{anchors.top}  "
                f"impact=fr{anchors.impact}(conf={anchors.impact_conf:.2f})  "
                f"finish=fr{anchors.finish}",
                (10, 48), FONT, 0.50, (140,140,140), 1)

    return np.vstack([hdr, timeline, strip, table])


# ── main ──────────────────────────────────────────────────────────────────────

def run_video(vname: str, angle: str) -> bool:
    vpath = str(INPUT / vname)
    stem  = Path(vname).stem[-14:]
    cache_path = KP_CACHE / f"{Path(vname).stem}.json"

    print(f"\n{'='*60}")
    print(f"{vname}  [{angle}]")

    # A layer: load from cache or run RTMPose
    pipeline = PosePipeline(device="cuda")
    if cache_path.exists():
        print(f"  A-layer: loading cached keypoints from {cache_path.name}")
        with open(cache_path) as f:
            kp_json = json.load(f)
        measurements, fps = pipeline.run_from_json(kp_json)
    else:
        print(f"  A-layer: running RTMPose...")
        t0 = time.time()
        measurements, fps = pipeline.run(vpath, verbose=True)
        print(f"  A-layer: done in {time.time()-t0:.1f}s")
        # Save cache as standard RTMPose JSON format
        frames_data = []
        for m in measurements:
            kp_dict = {}
            for name in pipeline.JOINT_NAMES if hasattr(pipeline, 'JOINT_NAMES') else []:
                pass
        # Use simplified cache
        _save_kp_cache(measurements, fps, vname, cache_path)

    # B layer
    engine = SwingPhaseEngine()
    annotations, anchors = engine.run(measurements, fps, angle=angle)

    psummary = phase_summary(annotations)
    print(f"  B-layer anchors: addr={anchors.address} top={anchors.top} "
          f"impact={anchors.impact} finish={anchors.finish}")
    for p in PHASE_NAMES:
        if p in psummary:
            s,e,cnt = psummary[p]
            print(f"    {p:16s}: fr{s:4d}–{e:4d}  ({cnt}fr  {(e-s)/fps*1000:.0f}ms)")

    # Gate-1 sheet
    sheet = make_phase_sheet(vpath, stem, angle, annotations, anchors, fps)
    out   = GATE1 / f"gate1_{stem}.jpg"
    desk  = DESK  / f"gate1_{stem}.jpg"
    cv2.imwrite(str(out),  sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(desk), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  Gate-1 image: {out.name}")

    # Fix 4: wrist-Y curve for 201015
    if "201015" in stem:
        generate_wrist_y_curve(measurements, fps, anchors, stem, DESK)

    return True


def generate_wrist_y_curve(measurements, fps, anchors, video_stem, dest_dir):
    """Generate a wrist-Y vs frame plot for diagnostic purposes (Fix 4)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.signal import savgol_filter
    except ImportError:
        print("  matplotlib not available, skipping wrist-Y curve")
        return

    n = len(measurements)
    ys = np.full(n, np.nan)
    for m in measurements:
        fi = m.frame_idx
        wm = m.wrist_mid()
        if wm is not None:
            ys[fi] = wm[1]
    idx = np.arange(n)
    nans = np.isnan(ys)
    if not nans.all():
        ys[nans] = np.interp(idx[nans], idx[~nans], ys[~nans])

    w = max(7, int(fps * 200 / 1000)) | 1
    ys_s = savgol_filter(ys, w, 3)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ys, alpha=0.3, color="steelblue", label="wrist-Y raw")
    ax.plot(ys_s, color="steelblue", linewidth=2, label="wrist-Y smoothed")

    colors = {"address": "gray", "top": "blue", "impact": "green", "finish": "purple"}
    for name, fr, c in [("address", anchors.address, "gray"),
                        ("top",     anchors.top,     "blue"),
                        ("impact",  anchors.impact,  "green"),
                        ("finish",  anchors.finish,  "purple")]:
        ax.axvline(fr, color=c, linestyle="--", linewidth=1.5, label=f"{name} fr{fr}")

    ax.set_xlabel("Frame")
    ax.set_ylabel("Wrist Y (pixels, down=high)")
    ax.set_title(f"Wrist-Y vs Frame — {video_stem}")
    ax.legend(fontsize=8)
    ax.invert_yaxis()  # Y increases downward in image coords
    plt.tight_layout()

    out_path = Path(dest_dir) / f"{video_stem}_wrist_y_curve.png"
    plt.savefig(str(out_path), dpi=120)
    plt.close()
    print(f"  Wrist-Y curve saved: {out_path}")


def _save_kp_cache(measurements: List[FrameMeasurement], fps: float,
                   vname: str, cache_path: Path):
    from engine.a_measurement.pose_pipeline import JOINT_NAMES
    frames = []
    for m in measurements:
        persons = []
        if m.measurement_quality != "bad":
            kps = {}
            for name in JOINT_NAMES:
                pt = m.keypoints.get(name)
                sc = m.confidences.get(name, 0.0)
                kps[name] = {
                    "x": float(pt[0]) if pt else 0.0,
                    "y": float(pt[1]) if pt else 0.0,
                    "score": sc
                }
            persons = [{"person_id": 0, "keypoints": kps}]
        frames.append({"frame": m.frame_idx, "persons": persons})
    data = {
        "model": "RTMPose-x",
        "keypoint_format": "COCO-17",
        "stats": {"source_fps": fps, "video": vname},
        "frames": frames,
    }
    with open(cache_path, "w") as f:
        json.dump(data, f)
    print(f"  Cached: {cache_path.name}")


def main():
    import datetime
    print(f"E1 run_e1.py started {datetime.datetime.now().isoformat()}")

    for vname, angle in VIDEOS.items():
        run_video(vname, angle)

    # Write NEEDS_HUMAN
    needs = Path("/home/jason/projects/swingcue-postest/NEEDS_HUMAN.md")
    needs.write_text(
        "# NEEDS_HUMAN.md\n\n"
        "## Gate-1 v2 ready for review\n\n"
        "B层8阶段标注已完成（v2修复版）。5段视频的阶段摘要图放在：\n"
        "- WSL: ~/projects/swingcue-postest/keyframes/gate1_preview/\n"
        "- 桌面: C:\\Users\\jason\\Desktop\\rtmpose_results\\preview\\gate1\\\n\n"
        "v2修复内容:\n"
        "- Fix1: impact用wrist-Y-max(face-on)/wrist-X-max(DTL)，取第一个峰（非全局最大）\n"
        "- Fix2: transition阶段不再为空，top阶段窗口[TOP-1, TOP+2]\n"
        "- Fix3: impact置信度改用峰值prominence/torso_height\n"
        "- Fix4: 201015_wrist_y_curve.png已生成在桌面\n\n"
        "请核对每张图：\n"
        "1. 8个阶段分得对不对？\n"
        "2. 特别看 address/top/impact 的缩略图是否对应正确的动作\n"
        "3. 如有问题请指出具体视频和阶段\n\n"
        "人验收通过前，E2代码可以继续写但不得用这些帧号做C层计算。\n"
    )
    print(f"\nNEEDS_HUMAN.md written.")

    # Update PROGRESS.log
    prog = Path("/home/jason/projects/swingcue-postest/PROGRESS.log")
    with open(prog, "a") as f:
        f.write(f"{datetime.datetime.now().isoformat()}  E1 complete: A+B layers done, "
                f"5 gate-1 sheets on desktop, waiting gate-1 review\n")

    print("\nAll done. Gate-1 images on desktop. Waiting for review.")


if __name__ == "__main__":
    main()
