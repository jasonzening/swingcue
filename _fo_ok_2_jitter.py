#!/usr/bin/env python3
"""
fo_ok_2_jitter_check.py
核查 fo-ok-2 fr75 跳变: 确认是单帧 keypoint jitter 还是真实头部运动
"""
import sys, json, math
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/jason/projects/swingcue-postest")
from engine.a_measurement.pose_pipeline import PosePipeline

KP_CACHE = Path("/home/jason/projects/swingcue-postest/engine/kp_cache/batch2/fo-ok-2.json")

pipeline = PosePipeline(device="cpu")
with open(KP_CACHE) as f:
    kp_json = json.load(f)
measurements, fps = pipeline.run_from_json(kp_json)
n = len(measurements)
print(f"fo-ok-2: {n} frames, fps={fps:.1f}")

# Extract head positions fr60-fr90
print("\nHead positions fr60-fr90:")
print(f"{'fr':>5} {'nose_x':>8} {'nose_y':>8} {'dx':>8} {'dy':>8} {'quality'}")
prev_x = prev_y = None
for m in measurements:
    fi = m.frame_idx
    if fi < 60 or fi > 90:
        continue
    kps = m.keypoints
    nose = kps.get("nose")
    if nose is None:
        # Try head from mid of left_eye/right_eye
        le = kps.get("left_eye"); re = kps.get("right_eye")
        if le and re:
            hx = (le[0]+re[0])/2; hy = (le[1]+re[1])/2
        else:
            hx = hy = None
    else:
        hx, hy = nose[0], nose[1]

    if hx is not None:
        dx = hx - prev_x if prev_x is not None else 0
        dy = hy - prev_y if prev_y is not None else 0
        flag = " *** JUMP" if (abs(dx) > 30 or abs(dy) > 30) else ""
        print(f"{fi:5d} {hx:8.1f} {hy:8.1f} {dx:8.1f} {dy:8.1f}  {m.measurement_quality}{flag}")
        prev_x, prev_y = hx, hy
    else:
        print(f"{fi:5d} {'None':>8} {'None':>8} {'—':>8} {'—':>8}  {m.measurement_quality}")
        prev_x = prev_y = None

# Also check torso height for normalization at fr75
print("\n--- torso height near fr75 ---")
for m in measurements:
    fi = m.frame_idx
    if fi < 70 or fi > 80:
        continue
    kps = m.keypoints
    lh = kps.get("left_hip"); rh = kps.get("right_hip")
    ls = kps.get("left_shoulder"); rs = kps.get("right_shoulder")
    if lh and rh and ls and rs:
        hip_y = (lh[1]+rh[1])/2
        sh_y  = (ls[1]+rs[1])/2
        torso = hip_y - sh_y
        nose  = kps.get("nose")
        if nose:
            head_disp_pct = (nose[1] - hip_y) / torso * 100 if torso > 0 else 0
            print(f"fr{fi:3d}  torso_h={torso:.1f}  nose_y={nose[1]:.1f}  "
                  f"head_disp={head_disp_pct:.1f}% (neg=above hip)")
