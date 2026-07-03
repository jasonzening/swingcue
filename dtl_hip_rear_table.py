#!/usr/bin/env python3
"""
dtl_hip_rear_table.py
For all 7 DTL videos:
  - Compute hip_rear_disp via SAM2 mask rear-edge
  - Output comparison table: hip_mid peak vs hip_rear peak per swing
  - Mark dtl-wrong-3 as excluded (invalid anchors)
  - Write JSON to Desktop
"""
import sys, json, math
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/jason/projects/swingcue-postest")

from engine.a_measurement.pose_pipeline import PosePipeline
from engine.b_phase.swing_phase import SwingPhaseEngine
from engine.c_features.feature_extractor import FeatureExtractor
from engine.c_features.hip_rear_extractor import HipRearExtractor
from engine.orientation.resolver import OrientationResolver

PROJ   = Path("/home/jason/projects/swingcue-postest")
INPUT  = PROJ / "input"
KP_DIR = PROJ / "engine/kp_cache"
DESK   = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/batch2")
DESK.mkdir(parents=True, exist_ok=True)

# (stem, kp_path, video_path, group, excluded)
DTL_VIDEOS = [
    ("201054",      KP_DIR/"Videos2026-06-09_201054_561.json",
                    INPUT/"Videos2026-06-09_201054_561.mp4",  "confirmed-fault", False),
    ("201058",      KP_DIR/"Videos2026-06-09_201058_697.json",
                    INPUT/"Videos2026-06-09_201058_697.mp4",  "confirmed-fault", False),
    ("dtl-ok-1",    KP_DIR/"batch2/dtl-ok-1.json",
                    INPUT/"dtl-ok-1.mp4",    "normal",        False),
    ("dtl-ok-2",    KP_DIR/"batch2/dtl-ok-2.json",
                    INPUT/"dtl-ok-2.mp4",    "normal",        False),
    ("dtl-wrong-1", KP_DIR/"batch2/dtl-wrong-1.json",
                    INPUT/"dtl-wrong-1.mp4", "fault-pending", False),
    ("dtl-wrong-2", KP_DIR/"batch2/dtl-wrong-2.json",
                    INPUT/"dtl-wrong-2.mp4", "fault-pending", False),
    ("dtl-wrong-3", KP_DIR/"batch2/dtl-wrong-3.json",
                    INPUT/"dtl-wrong-3.mp4", "excluded",      True),
]

extractor_rear = HipRearExtractor(device="cuda")
rows = []

for stem, kp_path, video_path, group, excluded in DTL_VIDEOS:
    print(f"\n{'='*55}")
    print(f"{stem}  [{group}]{'  ⚠️ EXCLUDED — invalid anchors' if excluded else ''}")

    with open(kp_path) as f:
        kp_json = json.load(f)
    pipeline = PosePipeline(device="cpu")
    measurements, fps = pipeline.run_from_json(kp_json)
    n = len(measurements)

    engine = SwingPhaseEngine()
    annotations, anchors = engine.run(measurements, fps, angle="down-the-line")
    phase_labels = [a.phase for a in annotations]

    # C-layer feature (hip_mid)
    feat = FeatureExtractor().extract(measurements, fps, angle="down-the-line",
                                      address_frame=anchors.address)

    # hip_mid window peak (transition→impact)
    hip_window_phases = {"transition", "downswing", "impact"}
    hip_win_idx = [i for i, p in enumerate(phase_labels) if p in hip_window_phases]
    if hip_win_idx:
        arr = feat.hip_disp[hip_win_idx]
        bi  = int(np.argmax(arr))
        hip_mid_peak_fr  = hip_win_idx[bi]
        hip_mid_peak_val = float(arr[bi])
    else:
        hip_mid_peak_fr = hip_mid_peak_val = None

    row = {
        "stem": stem, "group": group, "excluded": excluded,
        "n_frames": n, "swing_count": anchors.swing_count,
        "addr": anchors.address, "top": anchors.top,
        "top_conf": round(anchors.top_conf, 3),
        "impact": anchors.impact,
        "impact_conf": round(anchors.impact_conf, 3),
        "hip_mid_peak_fr": hip_mid_peak_fr,
        "hip_mid_peak_pct": round(hip_mid_peak_val * 100, 1) if hip_mid_peak_val is not None else None,
        "hip_rear_peak_fr": None,
        "hip_rear_peak_pct": None,
        "hip_rear_addr_rear_x": None,
        "hip_rear_torso_h": None,
        "hip_rear_nan_count": None,
        "hip_rear_note": "",
    }

    if excluded:
        row["hip_rear_note"] = "EXCLUDED: top_conf=0.012 anchors invalid"
        print(f"  SKIP hip_rear: excluded")
    else:
        # Orientation (ball_side)
        resolver = OrientationResolver()
        ori = resolver.resolve(
            measurements=measurements,
            angle="down-the-line",
            address_frame=anchors.address,
            top_frame=anchors.top,
            impact_frame=anchors.impact,
        )
        ball_side = ori.ball_side if ori.ball_side else "right"
        print(f"  orientation: ball_side={ball_side}")

        # hip_rear_disp via SAM2
        try:
            rear_result = extractor_rear.extract(
                str(video_path), measurements, anchors,
                ball_side=ball_side,
                phase_labels=phase_labels,
                kp_json=kp_json,
            )
            # Window peak
            valid_frames = [f for f in rear_result.window_frames
                            if not np.isnan(rear_result.hip_rear_disp[f])]
            if valid_frames:
                vals = [rear_result.hip_rear_disp[f] for f in valid_frames]
                bi   = int(np.argmax(vals))
                rear_peak_fr  = valid_frames[bi]
                rear_peak_val = vals[bi]
                row["hip_rear_peak_fr"]     = rear_peak_fr
                row["hip_rear_peak_pct"]    = round(rear_peak_val * 100, 1)
                row["hip_rear_addr_rear_x"] = round(rear_result.addr_rear_x, 1)
                row["hip_rear_torso_h"]     = round(rear_result.torso_h, 1)
                row["hip_rear_nan_count"]   = rear_result.nan_count
                print(f"  hip_mid:  fr{hip_mid_peak_fr}  {hip_mid_peak_val*100:.1f}%")
                print(f"  hip_rear: fr{rear_peak_fr}  {rear_peak_val*100:.1f}%  nan_count={rear_result.nan_count}")
                print(f"  addr_rear_x={rear_result.addr_rear_x:.0f}  torso_h={rear_result.torso_h:.0f}")
            else:
                row["hip_rear_note"] = "no valid frames in window"
                print("  hip_rear: no valid frames")
        except Exception as e:
            row["hip_rear_note"] = f"ERROR: {e}"
            print(f"  hip_rear ERROR: {e}")

    rows.append(row)

# Print comparison table
print("\n" + "="*110)
print("DTL 七段 hip_mid vs hip_rear 对照表")
print("="*110)
print(f"{'stem':12s} {'group':15s} {'exc':3s} {'addr':>5} {'impact':>6} "
      f"{'hip_mid_fr':>10} {'hip_mid%':>9} {'hip_rear_fr':>11} {'hip_rear%':>10} {'nan':>5}  note")
print("-"*115)
for r in rows:
    exc = "Y" if r["excluded"] else " "
    hm_fr  = f"fr{r['hip_mid_peak_fr']}"   if r["hip_mid_peak_fr"]  is not None else "—"
    hm_pct = f"{r['hip_mid_peak_pct']:+.1f}%" if r["hip_mid_peak_pct"] is not None else "—"
    hr_fr  = f"fr{r['hip_rear_peak_fr']}"  if r["hip_rear_peak_fr"] is not None else "—"
    hr_pct = f"{r['hip_rear_peak_pct']:+.1f}%" if r["hip_rear_peak_pct"] is not None else "—"
    nan_c  = str(r["hip_rear_nan_count"]) if r["hip_rear_nan_count"] is not None else "—"
    note   = r["hip_rear_note"][:30] if r["hip_rear_note"] else ""
    print(f"{r['stem']:12s} {r['group']:15s} {exc:3s} {r['addr']:5d} {r['impact']:6d} "
          f"{hm_fr:>10} {hm_pct:>9} {hr_fr:>11} {hr_pct:>10} {nan_c:>5}  {note}")

# Group distribution (excluding dtl-wrong-3 and using corrected groups)
print("\n" + "="*70)
print("分组分布统计 (修正版)")
print("="*70)
print("正常组 (申报正常, 无已知故意错误): dtl-ok-1, dtl-ok-2")
print("已确认故意错误组: 201054, 201058")
print("故障待GT确认: dtl-wrong-1, dtl-wrong-2")
print("剔除-锚点无效: dtl-wrong-3")
print()

for grp_label, grp_stems in [
    ("normal",          ["dtl-ok-1", "dtl-ok-2"]),
    ("confirmed-fault", ["201054", "201058"]),
    ("fault-pending",   ["dtl-wrong-1", "dtl-wrong-2"]),
]:
    g = [r for r in rows if r["stem"] in grp_stems]
    if not g: continue
    print(f"{grp_label} (n={len(g)}):")
    for r in g:
        hm = f"{r['hip_mid_peak_pct']:+.1f}%" if r["hip_mid_peak_pct"] is not None else "—"
        hr = f"{r['hip_rear_peak_pct']:+.1f}%" if r["hip_rear_peak_pct"] is not None else "—"
        print(f"  {r['stem']:15s}  hip_mid={hm:8s}  hip_rear={hr}")
    hm_vals = [r["hip_mid_peak_pct"] for r in g if r["hip_mid_peak_pct"] is not None]
    hr_vals = [r["hip_rear_peak_pct"] for r in g if r["hip_rear_peak_pct"] is not None]
    if hm_vals:
        print(f"  hip_mid  mean={sum(hm_vals)/len(hm_vals):.1f}%  "
              f"range=[{min(hm_vals):.1f}%,{max(hm_vals):.1f}%]")
    if hr_vals:
        print(f"  hip_rear mean={sum(hr_vals)/len(hr_vals):.1f}%  "
              f"range=[{min(hr_vals):.1f}%,{max(hr_vals):.1f}%]")
    print()

# Save JSON
out_path = DESK / "dtl_hip_rear_table.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print(f"JSON: {out_path}")
