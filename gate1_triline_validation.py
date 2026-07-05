#!/usr/bin/env python3
"""
gate1_triline_validation.py
关卡1 — 三线几何测量 + top帧渲染 验收

对 5 对强信号配对(clip_129/016/035/039/041)逐对:
  1. 提取 top帧 shoulder_lateral_tilt (左半 + 右半)
  2. 渲染 top帧 带三线+角度标注
  3. 汇报每对一行: 左半tilt / 右半tilt / |差值|

GT纪律:
  - clip_129 (dlt-6): 已知 GT — right=○(correct), left=✗(wrong)
    来自 split_screen_splitter.py O/X 标识
  - clip_016/035/039/041: 仅报测量值, 不贴对错标签
    (GT需人工目视确认 split 哪侧是正确示范)

输出:
  output/gate1_triline/top_frame_<pid>_left.jpg
  output/gate1_triline/top_frame_<pid>_right.jpg
  output/gate1_triline/gate1_tilt_report.md
"""

import sys, json, math, cv2
from pathlib import Path
import numpy as np

PROJ = Path("/home/jason/projects/swingcue-postest")
sys.path.insert(0, str(PROJ))

from engine.features.triline_geometry import (
    compute_triline_sequence,
    render_triline_frame,
)
from profiler_gate3_fullrun import get_pipeline, extract_sampled_kp

VIDEO_DIR    = Path("/mnt/c/Users/jason/Zening/Swingcue/教学视频")
TEACH_CACHE  = PROJ / "engine/kp_cache/teach"
OUT_DIR      = PROJ / "output/gate1_triline"
OUT_WIN      = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/gate1_triline")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_WIN.mkdir(parents=True, exist_ok=True)

NaN = float("nan")

# ─── pair definitions ─────────────────────────────────────────────────────────
# Each entry: (parent_id, source_file, gt_note)
PAIRS = [
    ("clip_129", "dtl-1/dlt-6.mp4",
     "GT: right=○(correct) left=✗(wrong)  [O/X marker, confirmed]"),
    ("clip_016", "115. 好的旋转方向是：身体按照顺序自然旋转。#高尔夫 #golf #golfswing #高尔夫课程 #高尔夫挥杆 #golflesson #golflife.mp4",
     "GT: 未登记 — 仅报测量值"),
    ("clip_035", "134. 不同杆数的区别#高尔夫球#高尔夫 #高尔夫教学 #高尔夫教练 #又直又远.mp4",
     "GT: 未登记 — 仅报测量值"),
    ("clip_039", "138. 读者一眼就看懂为什么要这么练.mp4",
     "GT: 未登记 — 仅报测量值"),
    ("clip_041", "15. 增加20m后 不击打后地的 挥杆方法.mp4",
     "GT: 未登记 — 仅报测量值"),
]

# ─── extract top-frame kp for one half ────────────────────────────────────────

def get_kp_json_for_half(vp: Path, side: str, pipeline) -> dict:
    """Return full kp_json for left or right half of video."""
    cap = cv2.VideoCapture(str(vp))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    half = w // 2
    x_crop = (0, half) if side == "left" else (half, w)
    # Use 60 samples — enough for reliable top detection
    return extract_sampled_kp(vp, n_samples=60, x_crop=x_crop)


def render_top(vp: Path, side: str, top_idx: int, kp_seq: dict,
               label: str, out_path: Path, pipeline) -> None:
    """Extract the top frame BGR and render triline overlay, save JPEG."""
    frames_kp = kp_seq["frames"]
    if top_idx is None or top_idx >= len(frames_kp):
        print(f"  [{side}] top_idx invalid, skip render")
        return

    cap = cv2.VideoCapture(str(vp))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    nf  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # The kp_json was sampled — map top_idx to actual video frame number
    n_samples = len(frames_kp)
    if n_samples == 0:
        cap.release(); return
    # Sampled frames are evenly spaced
    actual_fr = int(top_idx * (nf - 1) / max(n_samples - 1, 1))
    half       = w // 2
    x0         = 0 if side == "left" else half

    cap.set(cv2.CAP_PROP_POS_FRAMES, actual_fr)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        print(f"  [{side}] could not read frame {actual_fr}")
        return

    # Crop to the half panel
    panel = bgr[:, x0:x0 + half]

    # Get kps for this frame
    fd  = frames_kp[top_idx]
    p   = fd.get("persons", [])
    kps = p[0].get("keypoints", {}) if p else {}
    feat = kp_seq["frames"][top_idx]  # pre-computed features

    annotated = render_triline_frame(panel, kps, feat, label=label)
    cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  [{side}] frame={actual_fr}  tilt={feat.get('shoulder_lateral_tilt', NaN):.2f}°  -> {out_path.name}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    pipeline = get_pipeline()

    report_rows = []

    for pid, src, gt_note in PAIRS:
        print(f"\n{'='*60}")
        print(f"  {pid}  {src[:55]}")
        print(f"  {gt_note}")
        vp = VIDEO_DIR / src

        # ── clip_129: use pre-extracted full kp_cache ─────────────────────────
        if pid == "clip_129":
            kp_l = json.load(open(TEACH_CACHE / "dlt-6_left.json"))
            kp_r = json.load(open(TEACH_CACHE / "dlt-6_right.json"))
            # For rendering, the pre-split left.mp4 / right.mp4 exist
            vp_l = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/split_check/dlt-6/left.mp4")
            vp_r = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/split_check/dlt-6/right.mp4")
        else:
            kp_l = get_kp_json_for_half(vp, "left",  pipeline)
            kp_r = get_kp_json_for_half(vp, "right", pipeline)
            vp_l = vp_r = vp  # original split-screen — we'll crop in render

        seq_l = compute_triline_sequence(kp_l, addr_pct=0.08, front_pct=0.65)
        seq_r = compute_triline_sequence(kp_r, addr_pct=0.08, front_pct=0.65)

        tilt_l = seq_l["top_features"].get("shoulder_lateral_tilt", NaN)
        tilt_r = seq_r["top_features"].get("shoulder_lateral_tilt", NaN)
        n_l    = seq_l["n_valid"]
        n_r    = seq_r["n_valid"]
        top_l  = seq_l["top_idx"]
        top_r  = seq_r["top_idx"]

        diff = abs(tilt_l - tilt_r) if not (math.isnan(tilt_l) or math.isnan(tilt_r)) else NaN

        print(f"  LEFT:  tilt@top={tilt_l:.2f}°  top_idx={top_l}  n_valid={n_l}")
        print(f"  RIGHT: tilt@top={tilt_r:.2f}°  top_idx={top_r}  n_valid={n_r}")
        print(f"  |diff|={diff:.2f}°" if not math.isnan(diff) else "  diff=N/A")

        # ── render top frames ─────────────────────────────────────────────────
        out_l = OUT_DIR / f"top_frame_{pid}_left.jpg"
        out_r = OUT_DIR / f"top_frame_{pid}_right.jpg"

        if pid == "clip_129":
            # Render from pre-split videos (full frame, no crop needed)
            _render_top_direct(vp_l, kp_l, seq_l, f"{pid} LEFT | {gt_note[:40]}", out_l)
            _render_top_direct(vp_r, kp_r, seq_r, f"{pid} RIGHT | {gt_note[:40]}", out_r)
        else:
            render_top(vp, "left",  top_l, seq_l, f"{pid} LEFT", out_l, pipeline)
            render_top(vp, "right", top_r, seq_r, f"{pid} RIGHT", out_r, pipeline)

        # Copy to Windows
        import shutil
        for p_out in [out_l, out_r]:
            if p_out.exists():
                shutil.copy(p_out, OUT_WIN / p_out.name)

        report_rows.append({
            "pid": pid, "src_short": src.split("/")[-1][:45], "gt_note": gt_note,
            "tilt_left": tilt_l, "tilt_right": tilt_r, "diff": diff,
            "n_left": n_l, "n_right": n_r,
        })

    # ── report ────────────────────────────────────────────────────────────────
    md = _build_report(report_rows)
    rep = OUT_DIR / "gate1_tilt_report.md"
    rep.write_text(md, encoding="utf-8")
    import shutil
    shutil.copy(rep, OUT_WIN / "gate1_tilt_report.md")
    print(f"\nReport: {rep}")
    print(md)


def _render_top_direct(vp: Path, kp_json: dict, seq: dict, label: str, out_path: Path):
    """Render top frame directly from a single-panel video (clip_129 halves)."""
    frames_kp = seq["frames"]
    top_idx   = seq["top_idx"]
    if top_idx is None: return

    cap = cv2.VideoCapture(str(vp))
    nf  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_samples = len(frames_kp)
    actual_fr = int(top_idx * (nf - 1) / max(n_samples - 1, 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, actual_fr)
    ok, bgr = cap.read()
    cap.release()
    if not ok: return

    fd  = frames_kp[top_idx]
    p   = fd.get("persons", [])
    kps = p[0].get("keypoints", {}) if p else {}
    feat = seq["frames"][top_idx]

    annotated = render_triline_frame(bgr, kps, feat, label=label)
    cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  frame={actual_fr}  tilt={feat.get('shoulder_lateral_tilt', float('nan')):.2f}°  -> {out_path.name}")


def _build_report(rows):
    lines = []
    lines.append("# gate1_tilt_report.md")
    lines.append("# 关卡1 — 三线几何 shoulder_lateral_tilt@top 验收")
    lines.append("")
    lines.append("> 生成时间: 2026-07-04")
    lines.append("> 纯测量层, 无对错标签.")
    lines.append("> clip_129 GT已知(O/X标识): right=正确, left=错误")
    lines.append("> clip_016/035/039/041: 仅报测量值, GT需人工目视split确认")
    lines.append("")
    lines.append("## 汇总表")
    lines.append("")
    lines.append("| pair | 视频(截断) | tilt_left(°) | tilt_right(°) | |diff|(°) | n_left | n_right |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        tl = f"{r['tilt_left']:+.2f}"  if not math.isnan(r['tilt_left'])  else "N/A"
        tr = f"{r['tilt_right']:+.2f}" if not math.isnan(r['tilt_right']) else "N/A"
        df = f"{r['diff']:.2f}"         if not math.isnan(r['diff'])        else "N/A"
        lines.append(f"| {r['pid']} | {r['src_short']} | {tl} | {tr} | {df} | {r['n_left']} | {r['n_right']} |")
    lines.append("")
    lines.append("## GT 纪律说明")
    lines.append("")
    for r in rows:
        lines.append(f"- **{r['pid']}**: {r['gt_note']}")
    lines.append("")
    lines.append("## 渲染图说明")
    lines.append("")
    lines.append("- 红线 = 肩线 (shoulder line)")
    lines.append("- 蓝线 = 髋线 (pelvis line)")
    lines.append("- 绿线 = 踝线 (ankle line)")
    lines.append("- 橙线 = 躯干轴 (hip_mid→shoulder_mid, 延伸)")
    lines.append("- 白虚线 = 过肩中点的垂直参考线")
    lines.append("- tilt 数值 = 躯干轴 vs 垂直线夹角(°), 正=右倾, 负=左倾")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
