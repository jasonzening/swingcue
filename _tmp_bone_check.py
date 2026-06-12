import json, sys, numpy as np

sys.path.insert(0, ".")
from engine.a_measurement.pose_pipeline import PosePipeline

with open("engine/kp_cache/Videos2026-06-09_201058_697.json") as f:
    d = json.load(f)

pipeline = PosePipeline(device="cpu")
meas, fps = pipeline.run_from_json(d)
n = len(meas)
print(f"n={n} fps={fps}")

bone_keys = ["left_shoulder_left_elbow", "right_shoulder_right_elbow",
             "left_hip_left_knee", "right_hip_right_knee",
             "left_shoulder_right_shoulder", "left_hip_right_hip"]

for bk in bone_keys:
    lengths = np.array([m.bone_lengths.get(bk, 0.0) for m in meas])
    valid = lengths[lengths > 0]
    if len(valid) == 0:
        print(f"  {bk}: NO DATA")
        continue
    med = float(np.median(valid))
    ratios = np.where(lengths > 0, lengths / med, 1.0)
    bad = int(np.sum(np.abs(ratios - 1.0) > 0.20))
    pct = bad / n * 100
    print(f"  {bk[:40]:40s}: med={med:.1f}  bad={bad}/{n} ({pct:.0f}%)")

# Per-frame unreliable count
unreliable = np.zeros(n, dtype=bool)
for bk in bone_keys:
    lengths = np.array([m.bone_lengths.get(bk, 0.0) for m in meas])
    valid = lengths[lengths > 0]
    if len(valid) == 0: continue
    med = float(np.median(valid))
    ratios = np.where(lengths > 0, lengths / med, 1.0)
    unreliable |= np.abs(ratios - 1.0) > 0.20

print(f"\nTotal unreliable frames: {unreliable.sum()}/{n} ({unreliable.mean()*100:.1f}%)")

# Now check measurement_quality counts
quals = [m.measurement_quality for m in meas]
from collections import Counter
print("Measurement quality:", Counter(quals))
