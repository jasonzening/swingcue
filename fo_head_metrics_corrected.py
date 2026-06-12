#!/usr/bin/env python3
"""
fo_head_metrics_corrected.py
修复 head_center() 的 (0,0) bug 并重算 batch2 正面四段头部指标表

Bug原因：
  right_ear 在 fo-ok-2 fr75 的 x=0.0, y=0.0, score=0.339
  score >= 0.3 阈值通过，但 (0,0) 是无效坐标，不应纳入平均
  修复：head_center() 额外过滤 x>0 AND y>0

Correction:
  head_lat = (head_x - addr_head_x) / torso_h * 100  (正=目标侧)
  head_vert = (addr_head_y - head_y) / torso_h * 100  (正=上升)
"""
import sys, json, math
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/jason/projects/swingcue-postest")
from engine.a_measurement.pose_pipeline import PosePipeline
from engine.b_phase.swing_phase import SwingPhaseEngine

KP_DIR = Path("/home/jason/projects/swingcue-postest/engine/kp_cache/batch2")
INPUT  = Path("/home/jason/projects/swingcue-postest/input")
DESK   = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/batch2")
DESK.mkdir(parents=True, exist_ok=True)

FO_VIDEOS = [
    "fo-ok-1",
    "fo-ok-2",
    "fo-wrong-3",
    "fo-wrong-4",
]


def kp_pt(kps, name, thr=0.3):
    if name not in kps: return None
    k = kps[name]
    x, y = float(k["x"]), float(k["y"])
    # FIX: exclude (0,0) — invalid coordinate
    if k["score"] >= thr and x > 0 and y > 0:
        return (x, y)
    return None


def head_center_fixed(kps):
    """Average of valid head keypoints, with (0,0) guard."""
    pts = [kp_pt(kps, nm, 0.3) for nm in ("nose", "left_eye", "right_eye", "left_ear", "right_ear")]
    valid = [p for p in pts if p is not None]
    if not valid:
        return None
    return (sum(p[0] for p in valid) / len(valid),
            sum(p[1] for p in valid) / len(valid))


def head_center_buggy(kps):
    """Original buggy version (allows (0,0))."""
    pts = [kp_pt_buggy(kps, nm, 0.3) for nm in ("nose", "left_eye", "right_eye", "left_ear", "right_ear")]
    valid = [p for p in pts if p is not None]
    if not valid:
        return None
    return (sum(p[0] for p in valid) / len(valid),
            sum(p[1] for p in valid) / len(valid))


def kp_pt_buggy(kps, name, thr=0.3):
    if name not in kps: return None
    k = kps[name]
    return (float(k["x"]), float(k["y"])) if k["score"] >= thr else None


def process(stem):
    kp_path = KP_DIR / f"{stem}.json"
    with open(kp_path) as f:
        kp_json = json.load(f)

    pipeline = PosePipeline(device="cpu")
    measurements, fps = pipeline.run_from_json(kp_json)
    n = len(measurements)

    engine = SwingPhaseEngine()
    annotations, anchors = engine.run(measurements, fps, angle="face-on")
    phase_labels = [a.phase for a in annotations]
    phase_map = {a.frame_idx: a.phase for a in annotations}

    addr_fr   = anchors.address
    impact_fr = anchors.impact
    p5_fr = next((i for i, p in enumerate(phase_labels) if p == "transition"), addr_fr)

    # Address head from kp_json
    fd0   = kp_json["frames"][addr_fr]
    kps0  = fd0["persons"][0]["keypoints"] if fd0["persons"] else {}
    hc0_fixed = head_center_fixed(kps0)
    hc0_buggy = head_center_buggy(kps0)
    addr_hx_f, addr_hy_f = hc0_fixed if hc0_fixed else (360, 200)
    addr_hx_b, addr_hy_b = hc0_buggy if hc0_buggy else (360, 200)

    torso_h = measurements[addr_fr].torso_height() or 200.0

    print(f"\n{stem}:")
    print(f"  addr_fr={addr_fr}  impact_fr={impact_fr}  p5_fr={p5_fr}  torso_h={torso_h:.0f}")
    print(f"  addr head (fixed)={hc0_fixed}  (buggy)={hc0_buggy}")

    # Compute vert window p5→impact with fixed and buggy
    vert_frs = list(range(p5_fr, min(impact_fr + 1, n)))
    lat_frs  = list(range(addr_fr, min(impact_fr + 1, n)))

    def compute_peaks(frs, addr_hx, addr_hy, head_fn):
        lats, verts = [], []
        for fri in lat_frs if frs is lat_frs else frs:
            fd = kp_json["frames"][fri]
            hc = head_fn(fd["persons"][0]["keypoints"]) if fd["persons"] else None
            if hc:
                lats.append((fri, (hc[0] - addr_hx) / torso_h * 100))
            else:
                lats.append((fri, float("nan")))
        for fri in vert_frs:
            fd = kp_json["frames"][fri]
            hc = head_fn(fd["persons"][0]["keypoints"]) if fd["persons"] else None
            if hc:
                verts.append((fri, (addr_hy - hc[1]) / torso_h * 100))
            else:
                verts.append((fri, float("nan")))
        return lats, verts

    lats_f, verts_f = compute_peaks(lat_frs, addr_hx_f, addr_hy_f, head_center_fixed)
    lats_b, verts_b = compute_peaks(lat_frs, addr_hx_b, addr_hy_b, head_center_buggy)

    def peak(arr, direction="max"):
        valid = [(fr, v) for fr, v in arr if not math.isnan(v)]
        if not valid:
            return (None, None)
        return max(valid, key=lambda x: x[1]) if direction == "max" else min(valid, key=lambda x: x[1])

    # Fixed peaks
    lat_neg_f = peak(lats_f, "min")
    lat_pos_f = peak(lats_f, "max")
    vert_pk_f = peak(verts_f, "max")

    # Buggy peaks
    lat_neg_b = peak(lats_b, "min")
    vert_pk_b = peak(verts_b, "max")

    print(f"  FIXED:  head_lat_neg=fr{lat_neg_f[0]}/{lat_neg_f[1]:+.1f}%  "
          f"head_lat_pos=fr{lat_pos_f[0]}/{lat_pos_f[1]:+.1f}%  "
          f"head_vert=fr{vert_pk_f[0]}/{vert_pk_f[1]:+.1f}%")
    print(f"  BUGGY:  head_lat_neg=fr{lat_neg_b[0]}/{lat_neg_b[1]:+.1f}%  "
          f"head_vert=fr{vert_pk_b[0]}/{vert_pk_b[1]:+.1f}%")

    # CW elbow (for completeness)
    def cw_window(impact_f):
        for fr in range(impact_f + 1, min(impact_f + 20, n)):
            fd = kp_json["frames"][fr]
            if not fd["persons"]: continue
            kps = fd["persons"][0]["keypoints"]
            lw = kp_pt(kps, "left_wrist")
            lh = kp_pt(kps, "left_hip")
            if lw and lh and lw[1] < lh[1]:
                return fr
        return min(impact_f + 15, n - 1)

    cw_end = cw_window(impact_fr)
    elbow_angs = []
    for fr in range(impact_fr, cw_end + 1):
        if fr >= n: break
        fd = kp_json["frames"][fr]
        if not fd["persons"]: continue
        kps = fd["persons"][0]["keypoints"]
        sh = kp_pt(kps, "left_shoulder")
        el = kp_pt(kps, "left_elbow")
        wr = kp_pt(kps, "left_wrist")
        if sh and el and wr:
            import math as _m
            ba = (sh[0]-el[0], sh[1]-el[1])
            bc = (wr[0]-el[0], wr[1]-el[1])
            dot = ba[0]*bc[0] + ba[1]*bc[1]
            mag = _m.sqrt(ba[0]**2+ba[1]**2) * _m.sqrt(bc[0]**2+bc[1]**2)
            if mag > 0:
                ang = _m.degrees(_m.acos(max(-1, min(1, dot/mag))))
                elbow_angs.append((fr, ang))
    elbow_min = min(elbow_angs, key=lambda x: x[1]) if elbow_angs else (None, None)

    return {
        "stem": stem,
        "addr": addr_fr, "top": anchors.top, "impact": impact_fr,
        "impact_conf": round(anchors.impact_conf, 3),
        "torso_h": round(torso_h, 1),
        # CORRECTED values
        "head_lat_pos_fr": lat_pos_f[0], "head_lat_pos_pct": lat_pos_f[1],
        "head_lat_neg_fr": lat_neg_f[0], "head_lat_neg_pct": round(lat_neg_f[1], 1),
        "head_vert_fr":    vert_pk_f[0], "head_vert_pct":    round(vert_pk_f[1], 1),
        "elbow_min_fr":    elbow_min[0], "elbow_min_deg":    round(elbow_min[1], 1) if elbow_min[1] else None,
        "cw_window_end":   cw_end,
        # BUGGY values (for correction record)
        "buggy_head_lat_neg_pct": round(lat_neg_b[1], 1) if lat_neg_b[1] else None,
        "buggy_head_vert_pct":    round(vert_pk_b[1], 1) if vert_pk_b[1] else None,
        "buggy_head_vert_fr":     vert_pk_b[0],
    }


rows = []
for stem in FO_VIDEOS:
    row = process(stem)
    rows.append(row)

# Print corrected table
print("\n" + "="*100)
print("batch2 正面四段 头部指标表（修正版 — head_center (0,0) bug 已修复）")
print("="*100)
print(f"{'stem':12s} {'addr':>5} {'top':>5} {'impact':>6} "
      f"{'lat+(fr/%)':>14} {'lat-(fr/%)':>14} {'vert(fr/%)':>14} "
      f"{'elbow_min(fr/°)':>17}")
print("-"*100)
for r in rows:
    la_p = f"fr{r['head_lat_pos_fr']}/{r['head_lat_pos_pct']:+.1f}%" if r['head_lat_pos_fr'] else "—"
    la_n = f"fr{r['head_lat_neg_fr']}/{r['head_lat_neg_pct']:+.1f}%" if r['head_lat_neg_fr'] else "—"
    ve   = f"fr{r['head_vert_fr']}/{r['head_vert_pct']:+.1f}%"       if r['head_vert_fr'] else "—"
    el   = f"fr{r['elbow_min_fr']}/{r['elbow_min_deg']:.0f}°"        if r['elbow_min_fr'] else "—"
    print(f"{r['stem']:12s} {r['addr']:5d} {r['top']:5d} {r['impact']:6d} "
          f"{la_p:>14} {la_n:>14} {ve:>14} {el:>17}")

print("\n修正记录 (fo-ok-2 fr75 right_ear=(0,0) 导致数值失真):")
for r in rows:
    if r["buggy_head_vert_fr"] is not None and r["buggy_head_vert_pct"] != r["head_vert_pct"]:
        print(f"  {r['stem']:12s}: head_vert  旧={r['buggy_head_vert_pct']:+.1f}%(fr{r['buggy_head_vert_fr']}) "
              f"→ 新={r['head_vert_pct']:+.1f}%(fr{r['head_vert_fr']})")
        print(f"  {r['stem']:12s}: head_lat_neg 旧={r['buggy_head_lat_neg_pct']:+.1f}% "
              f"→ 新={r['head_lat_neg_pct']:+.1f}%")

# Save JSON
out_path = DESK / "fo_head_metrics_corrected.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print(f"\nJSON: {out_path}")
