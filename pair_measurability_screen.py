#!/usr/bin/env python3
"""
pair_measurability_screen.py
"正面2D可测性"筛查 — 24对配对

对每对(left半 / right半), 提取骨架并计算三维度几何差异:
  1. sway_range: 骨盆中点x横向移动范围 (归一化/torso_height)
  2. tilt_top:   spine_lateral_tilt 在top帧 (atan2 torso axis vs vertical, deg)
  3. head_range: 鼻尖x横向移动范围 (归一化/torso_height)

注意:
  - 纯几何测量, 禁贴对错标签
  - 不推断哪半是错误 — 只输出 |left_val - right_val| 差异
  - 若三维度差异均 < SMALL_THRESH → 标记 "2D不可测候选"
  - clip_130/131 几何偏DTL → 排除并说明
  - clip_129 (dlt-6) 复用 teach/ kp_cache

输出:
  output/pair_screen_2d/pair_measurability_results.json
  output/pair_screen_2d/pair_measurability_report.md
"""

import sys, json, math, cv2
from pathlib import Path
import numpy as np

PROJ = Path("/home/jason/projects/swingcue-postest")
sys.path.insert(0, str(PROJ))

from profiler_gate3_fullrun import get_pipeline, extract_sampled_kp

VIDEO_DIR  = Path("/mnt/c/Users/jason/Zening/Swingcue/教学视频")
TEACH_CACHE = PROJ / "engine/kp_cache/teach"
OUT_DIR    = PROJ / "output/pair_screen_2d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KP_THR      = 0.30
N_SAMPLES   = 40          # frames per half for measurements
MIN_SHW     = 20          # min shoulder width px for valid face-on frame
SMALL_DIFF  = 0.06        # below this → dimension not discriminating
ALL_SMALL   = 0.05        # ALL three dims below → 2D不可测候选

# ─── pair definitions ────────────────────────────────────────────────────────
# Load from video_profile_full.json
CARDS  = json.load(open(PROJ / "output/video_profile_full.json"))
leaf   = [c for c in CARDS if not c.get("children")]
split_parents = {}
for c in leaf:
    pid = c.get("parent_video")
    if pid:
        split_parents.setdefault(pid, []).append(c)

PAIR_ORDER = [
    "clip_003","clip_005","clip_016","clip_025","clip_026","clip_035",
    "clip_039","clip_041","clip_045","clip_048","clip_054","clip_064",
    "clip_078","clip_099","clip_113","clip_119","clip_121","clip_124",
    "clip_126","clip_127","clip_128","clip_129",
    # clip_130/131 excluded: geometric check both halves sh_lat<0.30 → DTL suspected
]

FLAGGED_EXCLUDE = {"clip_130", "clip_131"}

# clip_129 = dlt-6: use existing teach/ kp_cache (full extraction)
DLT6_CACHE_L = TEACH_CACHE / "dlt-6_left.json"
DLT6_CACHE_R = TEACH_CACHE / "dlt-6_right.json"

# ─── helpers ─────────────────────────────────────────────────────────────────

def safe_pt(kps, name):
    k = kps.get(name, {})
    if k.get("score", 0) >= KP_THR:
        return (k["x"], k["y"])
    return None


def torso_height(kps):
    ls = safe_pt(kps, "left_shoulder");  rs = safe_pt(kps, "right_shoulder")
    lh = safe_pt(kps, "left_hip");       rh = safe_pt(kps, "right_hip")
    if not (ls and rs and lh and rh):
        return None
    sh_w = math.hypot(rs[0]-ls[0], rs[1]-ls[1])
    if sh_w < MIN_SHW:
        return None
    sx = (ls[0]+rs[0])/2; sy = (ls[1]+rs[1])/2
    hx = (lh[0]+rh[0])/2; hy = (lh[1]+rh[1])/2
    th = math.hypot(sx-hx, sy-hy)
    return th if th > 15 else None


def pelvis_cx(kps):
    lh = safe_pt(kps, "left_hip"); rh = safe_pt(kps, "right_hip")
    if not (lh and rh): return None
    return (lh[0]+rh[0])/2


def spine_tilt_deg(kps):
    """Spine lateral tilt in degrees. Positive = right-lean."""
    ls = safe_pt(kps, "left_shoulder");  rs = safe_pt(kps, "right_shoulder")
    lh = safe_pt(kps, "left_hip");       rh = safe_pt(kps, "right_hip")
    if not (ls and rs and lh and rh):
        return None
    sh_w = math.hypot(rs[0]-ls[0], rs[1]-ls[1])
    if sh_w < MIN_SHW:
        return None
    sx = (ls[0]+rs[0])/2; sy = (ls[1]+rs[1])/2
    hx = (lh[0]+rh[0])/2; hy = (lh[1]+rh[1])/2
    dx = sx - hx; dy = sy - hy   # torso vector (hip→shoulder)
    # angle vs image-up (-y direction)
    # image vertical: (0, -1); tilt = atan2(dx, -dy) 
    tilt = math.degrees(math.atan2(dx, -dy))
    return tilt


def nose_cx(kps):
    n = safe_pt(kps, "nose")
    return n[0] if n else None


def wrist_y_min(kps):
    """Min y of wrists (highest wrist position = top of backswing in image coords)."""
    lw = safe_pt(kps, "left_wrist");  rw = safe_pt(kps, "right_wrist")
    vals = [p[1] for p in [lw, rw] if p]
    return min(vals) if vals else None


def frame_kps(frame_dict):
    p = frame_dict.get("persons", [])
    if not p: return None
    return p[0].get("keypoints", {})


# ─── metrics from kp_json ────────────────────────────────────────────────────

def compute_metrics(kp_json):
    """
    Returns dict: sway_range, tilt_top, head_range, n_valid, top_fr_idx
    All values normalized / torso_height.  None if insufficient data.
    """
    frames = kp_json.get("frames", [])
    if not frames:
        return {"sway_range": None, "tilt_top": None, "head_range": None,
                "n_valid": 0, "note": "no frames"}

    torso_h_vals = []
    pelvis_xs    = []
    nose_xs      = []
    wrist_ys     = []
    tilt_vals    = []
    frame_indices = []

    for i, fd in enumerate(frames):
        kps = frame_kps(fd)
        if kps is None: continue
        th = torso_height(kps)
        if th is None: continue
        torso_h_vals.append(th)
        px = pelvis_cx(kps)
        nx = nose_cx(kps)
        wy = wrist_y_min(kps)
        tl = spine_tilt_deg(kps)
        pelvis_xs.append(px)
        nose_xs.append(nx)
        wrist_ys.append(wy)
        tilt_vals.append(tl)
        frame_indices.append(i)

    n_valid = len(torso_h_vals)
    if n_valid < 4:
        return {"sway_range": None, "tilt_top": None, "head_range": None,
                "n_valid": n_valid, "note": f"too few valid frames ({n_valid})"}

    mean_th = np.mean(torso_h_vals)

    # sway_range: range of pelvis_x, normalized
    px_valid = [v for v in pelvis_xs if v is not None]
    sway_range = (max(px_valid)-min(px_valid))/mean_th if px_valid else None

    # head_range: range of nose_x, normalized
    nx_valid = [v for v in nose_xs if v is not None]
    head_range = (max(nx_valid)-min(nx_valid))/mean_th if nx_valid else None

    # tilt_top: find top frame (min wrist_y = highest position)
    # Use front portion of frames (top 65%) to avoid follow-through
    n_front = max(1, int(n_valid * 0.65))
    wy_front = [(wy, i) for i, wy in enumerate(wrist_ys[:n_front]) if wy is not None]
    tilt_top = None
    top_fr_idx = None
    if wy_front:
        min_wy, idx_in_valid = min(wy_front, key=lambda x: x[0])
        # actual tilt at that frame
        t = tilt_vals[idx_in_valid]
        tilt_top = t
        top_fr_idx = frame_indices[idx_in_valid]

    return {
        "sway_range": sway_range,
        "tilt_top":   tilt_top,
        "head_range": head_range,
        "n_valid":    n_valid,
        "top_fr_idx": top_fr_idx,
        "mean_torso_h": float(mean_th),
        "note": "ok",
    }


# ─── extract kp from video half ───────────────────────────────────────────────

def get_kp_for_half(video_path, side, width):
    """Extract sampled kp for left or right half of split-screen video."""
    half = width // 2
    if side == "left":
        x_crop = (0, half)
    else:
        x_crop = (half, width)
    return extract_sampled_kp(video_path, n_samples=N_SAMPLES, x_crop=x_crop)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    pipeline = get_pipeline()  # load RTMPose once

    results = []

    for i, pid in enumerate(PAIR_ORDER):
        children = split_parents.get(pid, [])
        l_card = next((c for c in children if c["side"]=="left"),  None)
        r_card = next((c for c in children if c["side"]=="right"), None)
        if not (l_card and r_card):
            print(f"  [{i+1}/{len(PAIR_ORDER)}] {pid}: missing children, skip")
            continue

        src = l_card.get("source_file", "")
        vp  = VIDEO_DIR / src
        short_title = src.split("/")[-1][:50]
        print(f"\n[{i+1}/{len(PAIR_ORDER)}] {pid}  {short_title}")

        # Special case: clip_129 (dlt-6) — reuse full teach/ kp_cache
        if pid == "clip_129":
            print("  -> reusing teach/ kp_cache (dlt-6 left+right full extraction)")
            kp_l = json.load(open(DLT6_CACHE_L))
            kp_r = json.load(open(DLT6_CACHE_R))
        else:
            # Read video width
            cap = cv2.VideoCapture(str(vp))
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            print(f"  -> extracting {N_SAMPLES} frames per half (w={w})")
            kp_l = get_kp_for_half(vp, "left",  w)
            kp_r = get_kp_for_half(vp, "right", w)

        m_l = compute_metrics(kp_l)
        m_r = compute_metrics(kp_r)
        print(f"  L: sway={m_l['sway_range']}, tilt={m_l['tilt_top']}, head={m_l['head_range']}  n={m_l['n_valid']}")
        print(f"  R: sway={m_r['sway_range']}, tilt={m_r['tilt_top']}, head={m_r['head_range']}  n={m_r['n_valid']}")

        # Compute absolute diffs
        def adiff(a, b):
            if a is None or b is None: return None
            return abs(a - b)

        d_sway = adiff(m_l["sway_range"], m_r["sway_range"])
        d_tilt = adiff(m_l["tilt_top"],   m_r["tilt_top"])
        d_head = adiff(m_l["head_range"], m_r["head_range"])

        # Determine dominant dimension
        dims = {"sway": d_sway, "tilt": d_tilt, "head": d_head}
        valid_dims = {k: v for k, v in dims.items() if v is not None}

        if not valid_dims:
            dominant = "NO_DATA"
            category = "NO_DATA"
        else:
            max_val = max(valid_dims.values())
            dominant = max(valid_dims, key=lambda k: valid_dims[k])
            all_small = all(v < ALL_SMALL for v in valid_dims.values())
            if all_small:
                category = "2D不可测候选"
            elif max_val < SMALL_DIFF:
                category = "差异偏小"
            else:
                category = f"主维度={dominant}"

        print(f"  diffs: sway={d_sway}, tilt={d_tilt}, head={d_head}  → {category}")

        results.append({
            "pair_num":  i + 1,
            "parent_id": pid,
            "source_file": src,
            "left_metrics":  m_l,
            "right_metrics": m_r,
            "diff_sway":  d_sway,
            "diff_tilt":  d_tilt,
            "diff_head":  d_head,
            "dominant_dim": dominant,
            "category":   category,
        })

    # Save JSON
    out_json = OUT_DIR / "pair_measurability_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"pairs": results,
                   "excluded": list(FLAGGED_EXCLUDE),
                   "exclusion_reason": "sh_lat_ratio both halves <0.30 → DTL geometry, needs human visual review"},
                  f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_json}")

    # ─── Generate report ──────────────────────────────────────────────────────
    report = generate_report(results)
    out_md = OUT_DIR / "pair_measurability_report.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved: {out_md}")

    # Copy to Windows
    win_dir = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/pair_screen_2d")
    win_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(out_json, win_dir / "pair_measurability_results.json")
    shutil.copy(out_md,   win_dir / "pair_measurability_report.md")
    print(f"Copied to Windows: {win_dir}")

    return results


def generate_report(results):
    lines = []
    lines.append("# pair_measurability_report.md")
    lines.append("# 正面2D可测性筛查 — 22对配对几何差异分析")
    lines.append("")
    lines.append("> 生成时间: 2026-07-04")
    lines.append("> 维度: sway(骨盆横移) / tilt(脊柱侧倾@top) / head(头部位移)")
    lines.append("> 差异 = |left_val - right_val|, 纯几何, 无对错标签")
    lines.append("> clip_130/131 (dtl-1/dtl-2) 几何偏DTL → 已排除")
    lines.append("")

    # Summary table
    lines.append("## 汇总表 (每对一行)")
    lines.append("")
    lines.append("| # | parent_id | 视频(截断) | sway差 | tilt差(°) | head差 | 最大维度 | 分类 |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for r in results:
        src_short = r["source_file"].split("/")[-1][:35]
        d_sw = f"{r['diff_sway']:.3f}" if r['diff_sway'] is not None else "N/A"
        d_tl = f"{r['diff_tilt']:.1f}" if r['diff_tilt'] is not None else "N/A"
        d_hd = f"{r['diff_head']:.3f}" if r['diff_head'] is not None else "N/A"
        lines.append(f"| {r['pair_num']} | {r['parent_id']} | {src_short} | {d_sw} | {d_tl} | {d_hd} | {r['dominant_dim']} | {r['category']} |")

    lines.append("")

    # Group by category
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[r["category"]].append(r)

    lines.append("## 分组统计")
    lines.append("")
    for cat, items in sorted(groups.items()):
        lines.append(f"### {cat} ({len(items)} 对)")
        for it in items:
            src_short = it['source_file'].split('/')[-1][:50]
            d_sw = f"{it['diff_sway']:.3f}" if it['diff_sway'] is not None else "N/A"
            d_tl = f"{it['diff_tilt']:.1f}" if it['diff_tilt'] is not None else "N/A"
            d_hd = f"{it['diff_head']:.3f}" if it['diff_head'] is not None else "N/A"
            lines.append(f"  - {it['parent_id']}  sway={d_sw}  tilt={d_tl}°  head={d_hd}  |  {src_short}")
        lines.append("")

    # Per-pair detail
    lines.append("## 各对详细数值")
    lines.append("")
    for r in results:
        ml = r["left_metrics"]
        mr = r["right_metrics"]
        src_short = r["source_file"].split("/")[-1][:60]
        lines.append(f"### 对 #{r['pair_num']}: {r['parent_id']}  —  {src_short}")
        lines.append(f"")
        lines.append(f"| 面板 | sway_range | tilt_top(°) | head_range | n_valid |")
        lines.append(f"|---|---|---|---|---|")
        sw_l = f"{ml['sway_range']:.4f}" if ml['sway_range'] is not None else "N/A"
        tl_l = f"{ml['tilt_top']:.2f}"  if ml['tilt_top']  is not None else "N/A"
        hd_l = f"{ml['head_range']:.4f}" if ml['head_range'] is not None else "N/A"
        sw_r = f"{mr['sway_range']:.4f}" if mr['sway_range'] is not None else "N/A"
        tl_r = f"{mr['tilt_top']:.2f}"  if mr['tilt_top']  is not None else "N/A"
        hd_r = f"{mr['head_range']:.4f}" if mr['head_range'] is not None else "N/A"
        lines.append(f"| 左半 | {sw_l} | {tl_l} | {hd_l} | {ml['n_valid']} |")
        lines.append(f"| 右半 | {sw_r} | {tl_r} | {hd_r} | {mr['n_valid']} |")
        d_sw = f"{r['diff_sway']:.3f}" if r['diff_sway'] is not None else "N/A"
        d_tl = f"{r['diff_tilt']:.1f}" if r['diff_tilt'] is not None else "N/A"
        d_hd = f"{r['diff_head']:.3f}" if r['diff_head'] is not None else "N/A"
        lines.append(f"| **差异** | **{d_sw}** | **{d_tl}** | **{d_hd}** | — |")
        lines.append(f"")
        lines.append(f"分类: **{r['category']}**   主维度: {r['dominant_dim']}")
        lines.append(f"")

    # Interpretation guide
    lines.append("---")
    lines.append("")
    lines.append("## 阈值说明")
    lines.append("")
    lines.append(f"- sway/head 单位: 无量纲 (像素/torso_height, 归一化)")
    lines.append(f"- tilt 单位: 度 (°)")
    lines.append(f"- SMALL_DIFF={ALL_SMALL}: 低于此值认为该维度无显著差异")
    lines.append(f"- 2D不可测候选: 三维度差异均 < {ALL_SMALL} → 该对讲的可能是手腕/深度旋转等2D不可测错误")
    lines.append("")
    lines.append("## 排除说明")
    lines.append("")
    lines.append("- clip_130 (dtl-1/dtl-1.mp4): sh_lat_ratio 全帧=0.332, 左半=0.296, 右半=0.293 → 两半均<0.30 (DTL信号)")
    lines.append("- clip_131 (dtl-2/dtl-2.mp4): 同上 (值完全相同, 疑同场景)")
    lines.append("- 上述两对需人工目视确认是否真为分屏face-on; 确认前不纳入分析")

    return "\n".join(lines)


if __name__ == "__main__":
    results = main()
    print("\n=== DONE ===")
