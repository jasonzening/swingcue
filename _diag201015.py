#!/usr/bin/env python3
"""Diagnostic: check 201015 current B-layer output"""
import sys, json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/home/jason/projects/swingcue-postest")
from engine.a_measurement.pose_pipeline import PosePipeline
from engine.b_phase.swing_phase import SwingPhaseEngine, PHASE_NAMES

kp_cache = Path("/home/jason/projects/swingcue-postest/engine/kp_cache/Videos2026-06-09_201015_827.json")
pipeline = PosePipeline(device="cpu")
with open(kp_cache) as f:
    kp_json = json.load(f)
measurements, fps = pipeline.run_from_json(kp_json)
print(f"n={len(measurements)}, fps={fps}")

engine = SwingPhaseEngine()
annotations, anchors = engine.run(measurements, fps, angle="face-on")
print(f"swing_count={anchors.swing_count} first_swing_end={anchors.first_swing_end}")
print(f"addr=fr{anchors.address} top=fr{anchors.top}(tc={anchors.top_conf:.3f}) "
      f"impact=fr{anchors.impact}(ic={anchors.impact_conf:.3f}) finish=fr{anchors.finish}")

phase_frames = defaultdict(list)
for a in annotations:
    phase_frames[a.phase].append(a.frame_idx)

print("\nPhase summary:")
for p in PHASE_NAMES:
    frs = sorted(set(phase_frames[p]))
    if not frs:
        print(f"  {p:16s}: (none)")
        continue
    print(f"  {p:16s}: fr{frs[0]}-{frs[-1]} ({len(frs)}fr) rep=fr{frs[len(frs)//2]}")

# Address sub-ranges
frs = sorted(set(phase_frames["address"]))
runs = []; start = frs[0]
for i in range(1, len(frs)):
    if frs[i] - frs[i-1] > 5:
        runs.append((start, frs[i-1]))
        start = frs[i]
runs.append((start, frs[-1]))
print(f"\naddress sub-ranges: {runs}")
print(f"fr154 is labeled: '{[a.phase for a in annotations if a.frame_idx == 154]}'")
