"""Quick regression check on all 5 videos after swing_phase v3.3"""
import json, sys
sys.path.insert(0, ".")
from engine.a_measurement.pose_pipeline import PosePipeline
from engine.b_phase.swing_phase import SwingPhaseEngine

VIDEOS = [
    ("Videos2026-06-09_201015_827", "face-on",        59),
    ("Videos2026-06-09_201039_231", "face-on",        208),
    ("Videos2026-06-09_201047_915", "face-on",        282),
    ("Videos2026-06-09_201054_561", "down-the-line",  150),
    ("Videos2026-06-09_201058_697", "down-the-line",  186),
]

print(f"{'Video':12s}  {'SC':>3}  {'Addr':>5}  {'Top':>5}  {'Impact':>7}  {'GT':>5}  {'Err':>5}  {'Result':>6}  {'ic':>5}  {'tc':>5}")
print("-"*75)
for vname, angle, gt in VIDEOS:
    d = json.load(open(f"engine/kp_cache/{vname}.json"))
    pipe = PosePipeline(device="cpu")
    meas, fps = pipe.run_from_json(d)
    eng = SwingPhaseEngine()
    ann, anch = eng.run(meas, fps, angle=angle)
    err = anch.impact - gt
    ok = "PASS" if abs(err) <= 2 else "FAIL"
    print(f"{vname[-6:]:12s}  {anch.swing_count:>3}  {anch.address:>5}  "
          f"{anch.top:>5}  {anch.impact:>7}  {gt:>5}  {err:>+5}  "
          f"{ok:>6}  {anch.impact_conf:.3f}  {anch.top_conf:.3f}")
