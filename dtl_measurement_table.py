#!/usr/bin/env python3
"""
dtl_measurement_table.py
为 DTL 五段视频生成完整测量表：
  - 锚点 (addr/top/impact/finish, swing_count, top_conf, impact_conf)
  - transition→impact 窗口内 hip_disp 峰值帧/值
  - downswing→impact 窗口内 spine_delta 峰值帧/值
  - D层诊断原文 (R1/R2/root_cause)
输出：打印表格 + 保存 JSON 到 Desktop/rtmpose_results/preview/batch2/dtl_measurement_table.json
"""
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/jason/projects/swingcue-postest")

from engine.a_measurement.pose_pipeline import PosePipeline
from engine.b_phase.swing_phase import SwingPhaseEngine, PHASE_NAMES
from engine.c_features.feature_extractor import FeatureExtractor
from src.judgment.rules import (
    bone_length_sentinel, r1_loss_of_posture, r2_hip_toward_ball
)
from src.judgment.root_cause import RootCauseEngine
from src.judgment.output import CoachingOutput

PROJ   = Path("/home/jason/projects/swingcue-postest")
KP_DIR = PROJ / "engine/kp_cache"
DESK   = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/batch2")
DESK.mkdir(parents=True, exist_ok=True)

DTL_VIDEOS = [
    # (stem, kp_path, group)
    ("201054", KP_DIR / "Videos2026-06-09_201054_561.json", "dtl-ok"),
    ("201058", KP_DIR / "Videos2026-06-09_201058_697.json", "dtl-ok"),
    ("dtl-ok-1",    KP_DIR / "batch2/dtl-ok-1.json",    "dtl-ok"),
    ("dtl-ok-2",    KP_DIR / "batch2/dtl-ok-2.json",    "dtl-ok"),
    ("dtl-wrong-1", KP_DIR / "batch2/dtl-wrong-1.json", "dtl-wrong"),
    ("dtl-wrong-2", KP_DIR / "batch2/dtl-wrong-2.json", "dtl-wrong"),
    ("dtl-wrong-3", KP_DIR / "batch2/dtl-wrong-3.json", "dtl-wrong"),
]

def process(stem, kp_path, group):
    pipeline = PosePipeline(device="cpu")
    with open(kp_path) as f:
        kp_json = json.load(f)
    measurements, fps = pipeline.run_from_json(kp_json)
    n = len(measurements)

    engine = SwingPhaseEngine()
    annotations, anchors = engine.run(measurements, fps, angle="down-the-line")
    phase_labels = [a.phase for a in annotations]

    ext = FeatureExtractor()
    feat = ext.extract(measurements, fps, angle="down-the-line",
                       address_frame=anchors.address)

    # Bone sentinel
    bone_length_ratios = {}
    for bk in ["left_hip_left_knee", "right_hip_right_knee"]:
        lengths = np.array([m.bone_lengths.get(bk, 0.0) for m in measurements])
        med = float(np.median(lengths[lengths > 0])) if np.any(lengths > 0) else 1.0
        if med > 0:
            bone_length_ratios[bk] = lengths / med
    unreliable_mask = bone_length_sentinel(bone_length_ratios)
    unreliable_ratio = float(np.mean(unreliable_mask)) if len(unreliable_mask) > 0 else 0.0

    # Window: transition → impact (hip rule window)
    hip_window_phases = {"transition", "downswing", "impact"}
    hip_window_idx = [i for i, p in enumerate(phase_labels) if p in hip_window_phases]

    hip_peak_val = 0.0
    hip_peak_fr = -1
    if hip_window_idx:
        arr = feat.hip_disp[hip_window_idx]
        best = int(np.argmax(arr))
        hip_peak_val = float(arr[best])
        hip_peak_fr = hip_window_idx[best]

    # Window: downswing → impact (spine rule window)
    spine_window_phases = {"downswing", "impact"}
    spine_window_idx = [i for i, p in enumerate(phase_labels) if p in spine_window_phases]

    spine_peak_val = 0.0
    spine_peak_fr = -1
    if spine_window_idx:
        arr = feat.spine_delta[spine_window_idx]
        best = int(np.argmax(arr))
        spine_peak_val = float(arr[best])
        spine_peak_fr = spine_window_idx[best]

    # D/E/F
    r1 = r1_loss_of_posture(feat.spine_delta, phase_labels,
                             joint_confidences=feat.joint_conf,
                             unreliable_mask=unreliable_mask if len(unreliable_mask)==n else None)
    r2 = r2_hip_toward_ball(feat.hip_disp, phase_labels,
                             joint_confidences=feat.joint_conf,
                             unreliable_mask=unreliable_mask if len(unreliable_mask)==n else None)
    faults = [x for x in [r1, r2] if x is not None]
    rc = RootCauseEngine().analyze(faults)
    coaching = CoachingOutput()
    out_f = coaching.generate(rc, unreliable_frame_ratio=unreliable_ratio)

    r1_str = "none"
    r2_str = "none"
    if r1:
        r1_str = f"{r1.severity} peak={r1.evidence['peak_value']:.2f}° onset=fr{r1.onset_frame}"
    if r2:
        r2_str = f"{r2.severity} peak={r2.evidence['peak_value']:.3f} onset=fr{r2.onset_frame}"

    return {
        "stem": stem,
        "group": group,
        "n_frames": n,
        "swing_count": anchors.swing_count,
        "addr": anchors.address,
        "top": anchors.top,
        "top_conf": round(anchors.top_conf, 3),
        "impact": anchors.impact,
        "impact_conf": round(anchors.impact_conf, 3),
        "finish": anchors.finish,
        "first_swing_end": anchors.first_swing_end,
        "hip_window": "transition→impact",
        "hip_peak_fr": hip_peak_fr,
        "hip_peak_val": round(hip_peak_val, 4),
        "spine_window": "downswing→impact",
        "spine_peak_fr": spine_peak_fr,
        "spine_peak_val": round(spine_peak_val, 2),
        "torso_h": round(feat.torso_h, 1),
        "r1_result": r1_str,
        "r2_result": r2_str,
        "root_cause": rc.root_cause,
        "certainty": rc.certainty,
        "one_liner": out_f.one_liner,
        "unreliable_ratio": round(unreliable_ratio, 3),
    }

rows = []
for stem, kp_path, group in DTL_VIDEOS:
    print(f"Processing {stem} ...", flush=True)
    row = process(stem, kp_path, group)
    rows.append(row)

# Print table
print("\n" + "="*100)
print("DTL 五段完整测量表")
print("="*100)
header = (f"{'stem':12s} {'grp':10s} {'n':>5} {'sc':>2} "
          f"{'addr':>5} {'top':>5} {'tc':>5} {'impact':>6} {'ic':>5} {'finish':>6} "
          f"{'hip_fr':>7} {'hip%':>7} {'sp_fr':>7} {'sp°':>6} "
          f"{'R1':25s} {'R2':25s} {'root_cause':12s}")
print(header)
print("-"*len(header))
for r in rows:
    print(f"{r['stem']:12s} {r['group']:10s} {r['n_frames']:5d} {r['swing_count']:2d} "
          f"{r['addr']:5d} {r['top']:5d} {r['top_conf']:5.3f} "
          f"{r['impact']:6d} {r['impact_conf']:5.3f} {r['finish']:6d} "
          f"fr{r['hip_peak_fr']:4d} {r['hip_peak_val']*100:6.1f}% "
          f"fr{r['spine_peak_fr']:4d} {r['spine_peak_val']:6.2f}° "
          f"{r['r1_result']:25s} {r['r2_result']:25s} {r['root_cause']:12s}")

# Distribution stats by group
print("\n" + "="*60)
print("分组分布统计")
print("="*60)
for grp in ["dtl-ok", "dtl-wrong"]:
    g = [r for r in rows if r["group"] == grp]
    if not g:
        continue
    hips = [r["hip_peak_val"] for r in g]
    spines = [r["spine_peak_val"] for r in g]
    print(f"\n{grp} (n={len(g)}):")
    print(f"  hip_peak:   min={min(hips)*100:.1f}%  max={max(hips)*100:.1f}%  "
          f"mean={sum(hips)/len(hips)*100:.1f}%  median={sorted(hips)[len(hips)//2]*100:.1f}%")
    print(f"  spine_peak: min={min(spines):.2f}°  max={max(spines):.2f}°  "
          f"mean={sum(spines)/len(spines):.2f}°  median={sorted(spines)[len(spines)//2]:.2f}°")
    for r in g:
        print(f"    {r['stem']:15s}  hip={r['hip_peak_val']*100:5.1f}%  "
              f"spine={r['spine_peak_val']:6.2f}°  root_cause={r['root_cause']}")

# Save JSON
out_path = DESK / "dtl_measurement_table.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print(f"\nJSON saved: {out_path}")
