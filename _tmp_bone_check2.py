import json, numpy as np, sys
sys.path.insert(0, ".")
from engine.a_measurement.pose_pipeline import PosePipeline

d = json.load(open("engine/kp_cache/Videos2026-06-09_201058_697.json"))
pipe = PosePipeline(device="cpu")
meas, fps = pipe.run_from_json(d)
n = len(meas)

unreliable = np.zeros(n, dtype=bool)
for bk in ["left_hip_left_knee", "right_hip_right_knee"]:
    ls = np.array([m.bone_lengths.get(bk, 0.0) for m in meas])
    valid = ls[ls > 0]
    if not len(valid):
        print(f"{bk}: NO DATA")
        continue
    med = float(np.median(valid))
    ratios = np.where(ls > 0, ls / med, 1.0)
    bad = int(np.sum(np.abs(ratios - 1.0) > 0.20))
    print(f"{bk}: med={med:.1f} bad={bad}/{n} ({bad/n*100:.0f}%)")
    unreliable |= np.abs(ratios - 1.0) > 0.20

print(f"hip_knee-only unreliable: {unreliable.sum()}/{n} ({unreliable.mean()*100:.1f}%)")

# Also check what the C-layer is computing
from engine.c_features.feature_extractor import FeatureExtractor
fe = FeatureExtractor()
# Need to find address frame
from engine.b_phase.swing_phase import SwingPhaseEngine
eng = SwingPhaseEngine()
ann, anch = eng.run(meas, fps, angle="down-the-line")
feat = fe.extract(meas, fps, "down-the-line", anch.address)
print(f"C-layer unreliable_ratio: {feat.unreliable.mean()*100:.1f}%")
