#!/usr/bin/env python3
"""Diagnostic: all 5 videos B-layer conf check"""
import sys, json
from pathlib import Path
sys.path.insert(0, "/home/jason/projects/swingcue-postest")
from engine.a_measurement.pose_pipeline import PosePipeline
from engine.b_phase.swing_phase import SwingPhaseEngine

VIDEOS = [
    ("Videos2026-06-09_201015_827.json", "face-on"),
    ("Videos2026-06-09_201039_231.json", "face-on"),
    ("Videos2026-06-09_201047_915.json", "face-on"),
    ("Videos2026-06-09_201054_561.json", "down-the-line"),
    ("Videos2026-06-09_201058_697.json", "down-the-line"),
]
KP_CACHE = Path("/home/jason/projects/swingcue-postest/engine/kp_cache")

for fname, angle in VIDEOS:
    with open(KP_CACHE / fname) as f:
        kp_json = json.load(f)
    pipeline = PosePipeline(device="cpu")
    measurements, fps = pipeline.run_from_json(kp_json)
    engine = SwingPhaseEngine()
    annotations, anchors = engine.run(measurements, fps, angle=angle)
    stem = fname[18:24]
    print(f"{stem} [{angle:12s}] swing={anchors.swing_count} "
          f"addr={anchors.address} top={anchors.top}(tc={anchors.top_conf:.3f}) "
          f"impact={anchors.impact}(ic={anchors.impact_conf:.3f}) finish={anchors.finish}")
