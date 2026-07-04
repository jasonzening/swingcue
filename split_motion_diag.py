#!/usr/bin/env python3
"""
split_motion_diag.py — v0.1
分诊:完整挥杆 vs 定格/慢速讲解

Method: frame-to-frame motion proxy
  - Per clip: compute grayscale absolute diff between consecutive frames
  - Focus on lower 65% of frame (body / wrist / club region)
  - Exclude top 35% (UI overlays, text annotations, score displays)
  - Peak detection: scipy.signal.find_peaks on smoothed curve
  - Classification: if max_peak_motion > SWING_THR → full swing
                    else → static/slow-demo

Why not RTMPose wrist KP?
  These are split clips with annotation overlays; frame-motion proxy
  is faster, equally reliable for binary swing/no-swing detection,
  and avoids false KP failures from graphical overlays.

Outputs:
  preview/split_check/motion/<stem>_<side>_motion.png  (curve plots)
  preview/split_check/motion/motion_diag_results.json
"""

import json, sys
from pathlib import Path
import cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter

SPLIT_BASE = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/split_check")
OUT_DIR    = SPLIT_BASE / "motion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLIPS = [
    ("dtl-3", "left"),  ("dtl-3", "right"),
    ("dtl-4", "left"),  ("dtl-4", "right"),
    ("dtl-5", "left"),  ("dtl-5", "right"),
    ("dlt-6", "left"),  ("dlt-6", "right"),
    ("dtl-7", "left"),  ("dtl-7", "right"),
]

# Swing threshold: mean absolute diff per active pixel
# Empirically: real swings produce peaks 15-60+ units; slow demos < 8 units
SWING_THR   = 12.0   # peak must exceed this to qualify as swing
MIN_PEAK_W  = 2      # minimum peak width (frames) to reject single-frame glitches


def compute_motion_curve(video_path: Path, body_frac: float = 0.35):
    """
    body_frac: ignore top fraction of frame (text/UI area).
    Returns (motion_curve, fps, n_frames).
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    y0  = int(h * body_frac)   # skip top portion

    motion = []
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return np.array([]), fps, n

    prev_gray = cv2.cvtColor(prev[y0:], cv2.COLOR_BGR2GRAY).astype(np.float32)
    motion.append(0.0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame[y0:], cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.mean(np.abs(gray - prev_gray))
        motion.append(float(diff))
        prev_gray = gray

    cap.release()
    return np.array(motion), fps, n


def classify(curve: np.ndarray, fps: float):
    """
    Returns: (clip_type, peak_val, peak_fr, peak_count, notes)
    clip_type: "full_swing" | "static_demo" | "slow_demo" | "inconclusive"
    """
    if len(curve) == 0:
        return "inconclusive", 0.0, -1, 0, "empty curve"

    # Smooth
    win = max(5, int(fps * 0.15)) | 1   # ~150ms window
    smooth = savgol_filter(curve, win, 3) if len(curve) > win else curve

    mean_motion = float(np.mean(smooth))
    max_motion  = float(np.max(smooth))
    peak_fr     = int(np.argmax(smooth))

    # Find prominent peaks
    min_dist = max(3, int(fps * 0.1))   # peaks at least 100ms apart
    peaks, props = find_peaks(smooth, height=SWING_THR,
                               prominence=SWING_THR * 0.5,
                               width=MIN_PEAK_W,
                               distance=min_dist)
    peak_count = len(peaks)
    swing_peaks = [p for p in peaks if smooth[p] >= SWING_THR]

    if swing_peaks:
        best_peak_fr  = swing_peaks[np.argmax([smooth[p] for p in swing_peaks])]
        best_peak_val = float(smooth[best_peak_fr])
        if best_peak_val >= SWING_THR * 2.5:
            clip_type = "full_swing"
            notes = f"peak={best_peak_val:.1f} >> thr={SWING_THR}"
        elif best_peak_val >= SWING_THR:
            clip_type = "full_swing"
            notes = f"peak={best_peak_val:.1f} > thr={SWING_THR}"
        else:
            clip_type = "inconclusive"
            notes = f"weak peak={best_peak_val:.1f}"
    else:
        best_peak_fr  = peak_fr
        best_peak_val = max_motion
        if mean_motion < 3.0:
            clip_type = "static_demo"
            notes = f"mean={mean_motion:.1f} max={max_motion:.1f} (near-static)"
        else:
            clip_type = "slow_demo"
            notes = f"mean={mean_motion:.1f} max={max_motion:.1f} no swing peak"

    return clip_type, best_peak_val, best_peak_fr, peak_count, notes


def plot_curve(curve, fps, stem, side, clip_type, peak_fr, peak_val, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n = len(curve)
    t = np.arange(n) / fps   # seconds

    win = max(5, int(fps * 0.15)) | 1
    smooth = savgol_filter(curve, win, 3) if n > win else curve

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, curve, color="#aaaaaa", lw=0.8, alpha=0.6, label="raw motion")
    ax.plot(t, smooth, color="#2c3e50", lw=1.8, label="smoothed")
    ax.axhline(SWING_THR, color="#e74c3c", lw=1.2, ls="--",
               label=f"swing threshold ({SWING_THR})")

    if peak_fr > 0 and peak_fr < n:
        ax.axvline(t[peak_fr], color="#e67e22", lw=1.5, ls=":", alpha=0.8)
        ax.annotate(f"peak={peak_val:.1f}\nfr{peak_fr}",
                    xy=(t[peak_fr], smooth[peak_fr]),
                    xytext=(t[peak_fr] + 0.2, smooth[peak_fr] + 1),
                    fontsize=8, color="#e67e22",
                    arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1))

    color_map = {"full_swing": "#27ae60", "static_demo": "#8e44ad",
                 "slow_demo": "#2980b9", "inconclusive": "#f39c12"}
    type_color = color_map.get(clip_type, "#999999")

    ax.set_title(f"{stem}/{side} — {clip_type}",
                 fontsize=13, fontweight="bold", color=type_color)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frame-motion proxy (mean |ΔI| px)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)


def main():
    results = []

    for stem, side in CLIPS:
        video_path = SPLIT_BASE / stem / f"{side}.mp4"
        print(f"\n{'='*50}")
        print(f"{stem}/{side}")

        if not video_path.exists():
            print(f"  MISSING: {video_path}")
            results.append({
                "clip": f"{stem}/{side}", "stem": stem, "side": side,
                "type": "missing", "peak_val": None,
            })
            continue

        curve, fps, n_frames = compute_motion_curve(video_path)
        duration_s = n_frames / fps if fps > 0 else 0

        clip_type, peak_val, peak_fr, peak_count, notes = classify(curve, fps)
        can_pipeline = clip_type == "full_swing"

        print(f"  {n_frames}fr @{fps:.0f}fps ({duration_s:.1f}s)")
        print(f"  peak={peak_val:.1f} at fr{peak_fr}  peaks_above_thr={peak_count}")
        print(f"  type={clip_type}  can_pipeline={can_pipeline}")
        print(f"  notes: {notes}")

        # Plot
        plot_path = OUT_DIR / f"{stem}_{side}_motion.png"
        plot_curve(curve, fps, stem, side, clip_type, peak_fr, peak_val, plot_path)
        print(f"  plot: {plot_path.name}")

        results.append({
            "clip": f"{stem}/{side}",
            "stem": stem,
            "side": side,
            "n_frames": int(n_frames),
            "fps": float(fps),
            "duration_s": round(duration_s, 1),
            "peak_motion": round(float(peak_val), 1),
            "peak_fr": int(peak_fr),
            "peak_count_above_thr": int(peak_count),
            "type": clip_type,
            "can_pipeline": can_pipeline,
            "notes": notes,
            "plot": str(plot_path),
        })

    # Save JSON
    json_path = OUT_DIR / "motion_diag_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary table
    print("\n\n=== 分诊汇报表 ===")
    print(f"{'片段':14s} {'时长':>7s} {'峰值':>7s} {'类型':>14s} {'可进管线':>8s}")
    print("-"*60)
    for r in results:
        if r["type"] == "missing":
            print(f"{r['clip']:14s}  MISSING")
            continue
        can = "是" if r["can_pipeline"] else "否"
        print(f"{r['clip']:14s} {r['duration_s']:>6.1f}s {r['peak_motion']:>7.1f} "
              f"{r['type']:>14s} {can:>8s}")

    print(f"\nWindows: C:\\Users\\jason\\Desktop\\rtmpose_results\\preview\\split_check\\motion\\")
    print(f"  <stem>_<side>_motion.png (10张曲线图)")
    print(f"  motion_diag_results.json")
    return results


if __name__ == "__main__":
    main()
