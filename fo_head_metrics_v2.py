#!/usr/bin/env python3
"""
fo_head_metrics_v2.py
head_ref v2: ears-first (双耳中点).  比较 v1(5点平均/含(0,0)bug) vs v2(耳为主).
输出正面四段头部指标表 v1 vs v2 并列对照.
"""
import sys, json, math
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/jason/projects/swingcue-postest")
from engine.a_measurement.pose_pipeline import PosePipeline
from engine.b_phase.swing_phase import SwingPhaseEngine
from engine.a_measurement.kp_guard import kp_guard, head_ref_v2

KP_DIR = Path("/home/jason/projects/swingcue-postest/engine/kp_cache/batch2")
DESK   = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/batch2")
DESK.mkdir(parents=True, exist_ok=True)

FO_VIDEOS = ["fo-ok-1", "fo-ok-2", "fo-wrong-3", "fo-wrong-4"]


def head_center_v1_fixed(kps):
    """v1 fixed (bug patched: x>0 AND y>0 guard, 5-point average)."""
    pts = [kp_guard(kps, nm, 0.3) for nm in
           ("nose", "left_eye", "right_eye", "left_ear", "right_ear")]
    valid = [p for p in pts if p is not None]
    if not valid: return None
    return (sum(p[0] for p in valid)/len(valid),
            sum(p[1] for p in valid)/len(valid)), "5pt_avg"


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

    torso_h = measurements[addr_fr].torso_height() or 200.0

    # Address head — both versions
    kps0 = kp_json["frames"][addr_fr]
    kps0 = kps0["persons"][0]["keypoints"] if kps0["persons"] else {}

    v1_result = head_center_v1_fixed(kps0)
    addr_v1 = v1_result[0] if v1_result else None
    addr_v2_ref = head_ref_v2(kps0)
    addr_v2 = (addr_v2_ref.x, addr_v2_ref.y) if addr_v2_ref else None
    addr_v2_src = addr_v2_ref.source if addr_v2_ref else "none"
    addr_v2_flag = addr_v2_ref.flag if addr_v2_ref else "none"

    lat_frs  = list(range(addr_fr, min(impact_fr + 1, n)))
    vert_frs = list(range(p5_fr,   min(impact_fr + 1, n)))

    def compute_tracks(head_fn):
        lats = []; verts = []
        addr_hx = head_fn(kps0)[0][0] if head_fn(kps0) else 0
        addr_hy = head_fn(kps0)[0][1] if head_fn(kps0) else 0
        for fri in lat_frs:
            fd = kp_json["frames"][fri]
            kps = fd["persons"][0]["keypoints"] if fd["persons"] else {}
            r = head_fn(kps)
            if r:
                lats.append((fri, (r[0][0] - addr_hx) / torso_h * 100))
            else:
                lats.append((fri, float("nan")))
        for fri in vert_frs:
            fd = kp_json["frames"][fri]
            kps = fd["persons"][0]["keypoints"] if fd["persons"] else {}
            r = head_fn(kps)
            if r:
                verts.append((fri, (addr_hy - r[0][1]) / torso_h * 100))
            else:
                verts.append((fri, float("nan")))
        return lats, verts

    def v2_head_fn(kps):
        hr = head_ref_v2(kps)
        if hr is None: return None
        return (hr.pt, hr.source)

    def compute_tracks_v2():
        if addr_v2 is None: return [], []
        addr_hx, addr_hy = addr_v2
        lats = []; verts = []
        for fri in lat_frs:
            fd = kp_json["frames"][fri]
            kps = fd["persons"][0]["keypoints"] if fd["persons"] else {}
            hr = head_ref_v2(kps)
            if hr:
                lats.append((fri, (hr.x - addr_hx) / torso_h * 100, hr.source, hr.flag))
            else:
                lats.append((fri, float("nan"), "none", "none"))
        for fri in vert_frs:
            fd = kp_json["frames"][fri]
            kps = fd["persons"][0]["keypoints"] if fd["persons"] else {}
            hr = head_ref_v2(kps)
            if hr:
                verts.append((fri, (addr_hy - hr.y) / torso_h * 100, hr.source, hr.flag))
            else:
                verts.append((fri, float("nan"), "none", "none"))
        return lats, verts

    lats_v1, verts_v1 = compute_tracks(head_center_v1_fixed)
    lats_v2, verts_v2 = compute_tracks_v2()

    def nanpeak(arr_with_meta, direction="max"):
        # arr_with_meta: list of (fr, val, ...) or (fr, val)
        valid = [(item[0], item[1]) for item in arr_with_meta if not math.isnan(item[1])]
        if not valid: return (None, None)
        return max(valid, key=lambda x: x[1]) if direction=="max" else min(valid, key=lambda x: x[1])

    lat_neg_v1 = nanpeak(lats_v1,  "min")
    lat_pos_v1 = nanpeak(lats_v1,  "max")
    vert_pk_v1 = nanpeak(verts_v1, "max")
    lat_neg_v2 = nanpeak(lats_v2,  "min")
    lat_pos_v2 = nanpeak(lats_v2,  "max")
    vert_pk_v2 = nanpeak(verts_v2, "max")

    # Address head source summary
    v2_src_counts = {}
    for _, _, src, flag in lats_v2:
        v2_src_counts[src] = v2_src_counts.get(src, 0) + 1

    return {
        "stem": stem,
        "addr": addr_fr, "top": anchors.top, "impact": impact_fr,
        "torso_h": round(torso_h, 1),
        "addr_v2_src": addr_v2_src, "addr_v2_flag": addr_v2_flag,
        "v2_src_counts": v2_src_counts,
        # V1 (fixed, 5pt avg)
        "v1_lat_neg_fr": lat_neg_v1[0], "v1_lat_neg_pct": round(lat_neg_v1[1],1) if lat_neg_v1[1] else None,
        "v1_lat_pos_fr": lat_pos_v1[0], "v1_lat_pos_pct": round(lat_pos_v1[1],1) if lat_pos_v1[1] else None,
        "v1_vert_fr":    vert_pk_v1[0], "v1_vert_pct":    round(vert_pk_v1[1],1) if vert_pk_v1[1] else None,
        # V2 (ears-first)
        "v2_lat_neg_fr": lat_neg_v2[0], "v2_lat_neg_pct": round(lat_neg_v2[1],1) if lat_neg_v2[1] else None,
        "v2_lat_pos_fr": lat_pos_v2[0], "v2_lat_pos_pct": round(lat_pos_v2[1],1) if lat_pos_v2[1] else None,
        "v2_vert_fr":    vert_pk_v2[0], "v2_vert_pct":    round(vert_pk_v2[1],1) if vert_pk_v2[1] else None,
    }


rows = []
for stem in FO_VIDEOS:
    print(f"Processing {stem}...")
    row = process(stem)
    rows.append(row)
    print(f"  addr_v2: src={row['addr_v2_src']} flag={row['addr_v2_flag']}")
    print(f"  v2 src distribution: {row['v2_src_counts']}")

# Print comparison table
print("\n" + "="*115)
print("正面四段头部指标 v1(5点平均,已修(0,0)bug) vs v2(耳为主) 对照")
print("="*115)
print(f"{'':12s}  {'版本':4s}  {'addr':>5} {'top':>5} {'impact':>6}  "
      f"{'head_lat-(fr/%)':>18} {'head_lat+(fr/%)':>18} {'head_vert(fr/%)':>18}  备注")
print("-"*115)
for r in rows:
    for ver in ["v1", "v2"]:
        ln  = f"fr{r[f'{ver}_lat_neg_fr']}/{r[f'{ver}_lat_neg_pct']:+.1f}%" if r[f'{ver}_lat_neg_fr'] and r[f'{ver}_lat_neg_pct'] is not None else "—"
        lp  = f"fr{r[f'{ver}_lat_pos_fr']}/{r[f'{ver}_lat_pos_pct']:+.1f}%" if r[f'{ver}_lat_pos_fr'] and r[f'{ver}_lat_pos_pct'] is not None else "—"
        ve  = f"fr{r[f'{ver}_vert_fr']}/{r[f'{ver}_vert_pct']:+.1f}%"       if r[f'{ver}_vert_fr'] and r[f'{ver}_vert_pct'] is not None else "—"
        note = (f"addr_src={r['addr_v2_src']}" if ver == "v2" else "")
        stem_col = r['stem'] if ver == "v1" else ""
        print(f"{stem_col:12s}  [{ver}]  {r['addr']:5d} {r['top']:5d} {r['impact']:6d}  "
              f"{ln:>18} {lp:>18} {ve:>18}  {note}")
    print()

# Diff summary
print("="*80)
print("差值 (v2 - v1) — 仅显示有差异者:")
for r in rows:
    diffs = []
    for key, label in [("lat_neg_pct","lat-"), ("lat_pos_pct","lat+"), ("vert_pct","vert")]:
        v1 = r[f"v1_{key}"]; v2 = r[f"v2_{key}"]
        if v1 is not None and v2 is not None and abs(v2-v1) > 0.5:
            diffs.append(f"{label}: {v1:+.1f}→{v2:+.1f} (Δ{v2-v1:+.1f}%)")
    if diffs:
        print(f"  {r['stem']:12s}: {', '.join(diffs)}")

# Save
import json
out_path = DESK / "fo_head_metrics_v2_comparison.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print(f"\nJSON: {out_path}")
