"""Quick smoke test: run OrientationResolver on all 5 cached videos."""
import json, sys
sys.path.insert(0, ".")
from engine.a_measurement.pose_pipeline import PosePipeline
from engine.b_phase.swing_phase import SwingPhaseEngine
from engine.orientation.resolver import OrientationResolver

VIDEOS = [
    ("Videos2026-06-09_201015_827", "face-on"),
    ("Videos2026-06-09_201039_231", "face-on"),
    ("Videos2026-06-09_201047_915", "face-on"),
    ("Videos2026-06-09_201054_561", "down-the-line"),
    ("Videos2026-06-09_201058_697", "down-the-line"),
]

resolver = OrientationResolver()

print(f"{'Video':12s}  {'angle':14s}  {'handed':8s}  {'target':8s}  {'trail':8s}  {'ball':8s}  {'conf':18s}  {'conflict':8s}")
print("-"*100)
for stem, angle in VIDEOS:
    kp = json.load(open(f"engine/kp_cache/{stem}.json"))
    pipe = PosePipeline(device="cpu")
    meas, fps = pipe.run_from_json(kp)
    eng = SwingPhaseEngine()
    ann, anchors = eng.run(meas, fps, angle=angle)

    result = resolver.resolve(meas, angle, anchors.address, anchors.top, anchors.impact)
    vid_id = stem[-6:]
    print(f"{vid_id:12s}  {angle:14s}  {str(result.handedness):8s}  "
          f"{str(result.target_side):8s}  {str(result.trail_side):8s}  "
          f"{str(result.ball_side):8s}  {result.confidence:18s}  {result.conflict}")
    if result.debug:
        for k, v in result.debug.items():
            print(f"              {k}: {v}")
    print(f"              method: {result.method}")
    print()
