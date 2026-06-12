import json, numpy as np, sys
sys.path.insert(0, ".")
from engine.a_measurement.pose_pipeline import PosePipeline
from engine.b_phase.swing_phase import SwingPhaseEngine
from engine.c_features.feature_extractor import FeatureExtractor

d = json.load(open("engine/kp_cache/Videos2026-06-09_201058_697.json"))
pipe = PosePipeline(device="cpu")
meas, fps = pipe.run_from_json(d)
eng = SwingPhaseEngine()
ann, anch = eng.run(meas, fps, angle="down-the-line")
fe = FeatureExtractor()
feat = fe.extract(meas, fps, "down-the-line", anch.address)

print("Meta:", feat.meta)
print("Anchors: addr=%d top=%d impact=%d finish=%d" % (anch.address, anch.top, anch.impact, anch.finish))
print()
print("Hip_disp and spine_delta at key frames:")
for label, fr in [("address", anch.address), ("top", anch.top),
                  ("impact-10", anch.impact-10), ("impact", anch.impact),
                  ("impact+10", anch.impact+10)]:
    if 0 <= fr < len(feat.hip_disp):
        print(f"  {label:12s} fr{fr:3d}: hip_disp={feat.hip_disp[fr]:+.4f}  spine_delta={feat.spine_delta[fr]:+.2f}deg")

print()
# Raw hip_x values at those frames
print("Raw hip_x at key frames (vs addr_hip_x=%.1f):" % feat.meta["addr_hip_x"])
for label, fr in [("address", anch.address), ("top", anch.top),
                  ("impact-5", anch.impact-5), ("impact", anch.impact),
                  ("impact+5", anch.impact+5)]:
    if 0 <= fr < len(meas):
        hp = meas[fr].hip_mid()
        if hp:
            print(f"  {label:12s} fr{fr:3d}: hip_x={hp[0]:.1f}  delta_x={hp[0]-feat.meta['addr_hip_x']:+.1f}px")
        else:
            print(f"  {label:12s} fr{fr:3d}: hip=None")
