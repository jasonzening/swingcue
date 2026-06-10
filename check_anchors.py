import json, numpy as np, sys
sys.path.insert(0, '/home/jason/projects/swingcue-postest')
from engine.a_measurement.pose_pipeline import PosePipeline
from engine.b_phase.swing_phase import SwingPhaseEngine

KNOWN_ANGLE = {
    "Videos2026-06-09_201015_827": "face-on",
    "Videos2026-06-09_201039_231": "face-on",
    "Videos2026-06-09_201047_915": "face-on",
    "Videos2026-06-09_201054_561": "down-the-line",
    "Videos2026-06-09_201058_697": "down-the-line",
}

GROUND_TRUTH = {
    "Videos2026-06-09_201015_827": 154,   # human-verified (2nd swing impact)
    "Videos2026-06-09_201039_231": 208,
    "Videos2026-06-09_201047_915": 282,
    "Videos2026-06-09_201054_561": 150,
    "Videos2026-06-09_201058_697": 185,   # might need to be later
}

pipeline = PosePipeline(device='cuda')
engine = SwingPhaseEngine()

from pathlib import Path
cache_dir = Path('/home/jason/projects/swingcue-postest/engine/kp_cache')

for stem, angle in KNOWN_ANGLE.items():
    cache = cache_dir / f"{stem}.json"
    with open(cache) as f:
        d = json.load(f)
    measurements, fps = pipeline.run_from_json(d)
    annotations, anchors = engine.run(measurements, fps, angle=angle)
    gt = GROUND_TRUTH[stem]
    delta = anchors.impact - gt
    status = "OK" if abs(delta) <= 5 else ("CLOSE" if abs(delta) <= 15 else "WRONG")
    print(f"{stem[-14:]}: addr={anchors.address} top={anchors.top} impact={anchors.impact} finish={anchors.finish} | GT={gt} delta={delta:+d} [{status}] imp_conf={anchors.impact_conf:.2f}")
