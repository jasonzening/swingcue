#!/usr/bin/env python3
"""
Probe fo-ok-2 fr75 discrepancy: +72.5% (report) vs 13.0% (render).
Trace both code paths against the same kp_cache.
"""
import sys, json, math
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/jason/projects/swingcue-postest")
from engine.a_measurement.pose_pipeline import PosePipeline
from engine.b_phase.swing_phase import SwingPhaseEngine

KP_PATH = Path("/home/jason/projects/swingcue-postest/engine/kp_cache/batch2/fo-ok-2.json")
with open(KP_PATH) as f:
    kp_json = json.load(f)

# --- Code path A: via PosePipeline (like _fo_ok_2_jitter.py) ---
pipeline = PosePipeline(device="cpu")
measurements, fps = pipeline.run_from_json(kp_json)
engine = SwingPhaseEngine()
annotations, anchors = engine.run(measurements, fps, angle="face-on")
addr_fr = anchors.address  # 33
impact_fr = anchors.impact  # 77

# Phase map
phase_map = {a.frame_idx: a.phase for a in annotations}
p5_fr = next((f for f in range(addr_fr, len(measurements)) if phase_map.get(f) == "transition"), addr_fr)
print(f"addr_fr={addr_fr}  impact_fr={impact_fr}  p5_fr={p5_fr}")

# --- Code path B: direct kp_json read (like batch2_pipeline.py) ---
def kp_pt(kps, name, thr=0.3):
    if name not in kps: return None
    k = kps[name]
    return (float(k["x"]), float(k["y"])) if k["score"] >= thr else None

def head_center(kps):
    pts = [kp_pt(kps, k, 0.3) for k in ("nose","left_eye","right_eye","left_ear","right_ear")]
    valid = [p for p in pts if p]
    if not valid: return None
    return (sum(p[0] for p in valid)/len(valid), sum(p[1] for p in valid)/len(valid))

# Address head center
fd0 = kp_json["frames"][addr_fr]
kps0 = fd0["persons"][0]["keypoints"] if fd0["persons"] else {}
hc0 = head_center(kps0)
addr_hx, addr_hy = hc0 if hc0 else (360, 200)

# Torso height at address via FrameMeasurement
torso_h = measurements[addr_fr].torso_height() or 200.0
print(f"addr head_center={hc0}  addr_hx={addr_hx:.1f}  addr_hy={addr_hy:.1f}  torso_h={torso_h:.1f}")

# Show fr75 in both paths
fr = 75
print(f"\n=== fr75 in CODE PATH B (direct kp_json) ===")
fd75 = kp_json["frames"][fr]
if fd75["persons"]:
    kps75 = fd75["persons"][0]["keypoints"]
    print("Raw keypoints (name: x/y/score):")
    for nm in ("nose","left_eye","right_eye","left_ear","right_ear"):
        k = kps75.get(nm)
        if k:
            print(f"  {nm:12s}: x={k['x']:.1f}  y={k['y']:.1f}  score={k['score']:.3f}  "
                  f"pass_0.3={'YES' if k['score']>=0.3 else 'NO'}")
    hc75 = head_center(kps75)
    print(f"head_center(0.3 thr): {hc75}")
    if hc75:
        dy_pct = (addr_hy - hc75[1]) / torso_h * 100   # batch2_pipeline formula
        dx_pct = (hc75[0] - addr_hx) / torso_h * 100
        print(f"  head_vert(addr_hy - hc[1]) = ({addr_hy:.1f} - {hc75[1]:.1f}) / {torso_h:.1f} * 100 = {dy_pct:.1f}%")
        print(f"  head_lat (hc[0] - addr_hx) = ({hc75[0]:.1f} - {addr_hx:.1f}) / {torso_h:.1f} * 100 = {dx_pct:.1f}%")

print(f"\n=== fr75 in CODE PATH A (PosePipeline) ===")
m75 = measurements[fr]
nose75 = m75.keypoints.get("nose")
print(f"  nose={nose75}  conf={m75.confidences.get('nose', 0):.3f}")
if nose75:
    dy_a = (addr_hy - nose75[1]) / torso_h * 100
    dx_a = (nose75[0] - addr_hx) / torso_h * 100
    print(f"  head_vert = ({addr_hy:.1f} - {nose75[1]:.1f}) / {torso_h:.1f} * 100 = {dy_a:.1f}%")
    print(f"  head_lat  = ({nose75[0]:.1f} - {addr_hx:.1f}) / {torso_h:.1f} * 100 = {dx_a:.1f}%")

# Full vert window p5→impact via both paths
print(f"\n=== vert window fr{p5_fr}-fr{impact_fr} via CODE PATH B ===")
vert_frs = list(range(p5_fr, min(impact_fr+1, len(kp_json["frames"]))))
vert_b = []
for fri in vert_frs:
    fd = kp_json["frames"][fri]
    if fd["persons"]:
        hc = head_center(fd["persons"][0]["keypoints"])
        val = (addr_hy - hc[1]) / torso_h * 100 if hc else float("nan")
    else:
        val = float("nan")
    vert_b.append((fri, val))

# Find peak
valid_b = [(f, v) for f, v in vert_b if not math.isnan(v)]
if valid_b:
    peak_b = max(valid_b, key=lambda x: x[1])
    print(f"  peak: fr{peak_b[0]}  vert={peak_b[1]:+.1f}%")
    # Print around fr75
    for fri, val in vert_b:
        if abs(fri - 75) <= 3:
            print(f"  fr{fri}: vert={val:+.1f}%")

print(f"\n=== vert window fr{p5_fr}-fr{impact_fr} via CODE PATH A ===")
for fri in vert_frs:
    m = measurements[fri]
    nose = m.keypoints.get("nose")
    if nose:
        val = (addr_hy - nose[1]) / torso_h * 100
        if abs(fri - 75) <= 3:
            print(f"  fr{fri}: nose_y={nose[1]:.1f}  vert={val:+.1f}%")
