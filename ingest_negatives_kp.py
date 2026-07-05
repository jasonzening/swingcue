#!/usr/bin/env python3
"""
ingest_negatives_kp.py — RTMPose kp_cache for negative control clips

Outputs: engine/kp_cache/negatives/<stem>.json
"""
import sys, json, time
from pathlib import Path

PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ))

from engine.a_measurement.pose_pipeline import PosePipeline, JOINT_NAMES

NEGATIVES_DIR = PROJ / "tests/negatives"
KP_CACHE_OUT  = PROJ / "engine/kp_cache/negatives"
KP_CACHE_OUT.mkdir(parents=True, exist_ok=True)

CLIPS = [
    "fo-eet-1-neg-setup",
    "fo-eet-1-neg-truncated",
]

pipeline = PosePipeline(device="cuda")

for stem in CLIPS:
    video_path = NEGATIVES_DIR / f"{stem}.mp4"
    cache_path = KP_CACHE_OUT / f"{stem}.json"

    if not video_path.exists():
        print(f"  SKIP {stem}: video not found at {video_path}")
        continue

    if cache_path.exists():
        print(f"  {stem}: cache hit, skipping RTMPose")
        continue

    print(f"  {stem}: running RTMPose on {video_path.name} ...")
    t0 = time.time()
    meas, fps = pipeline.run(str(video_path), verbose=False)
    elapsed = time.time() - t0
    print(f"    → {len(meas)} frames  {elapsed:.1f}s  fps={fps:.1f}")

    frames_out = []
    for m in meas:
        persons = []
        if m.measurement_quality != "bad":
            kps = {}
            for name in JOINT_NAMES:
                pt = m.keypoints.get(name)
                sc = m.confidences.get(name, 0.0)
                kps[name] = {
                    "x":     float(pt[0]) if pt else 0.0,
                    "y":     float(pt[1]) if pt else 0.0,
                    "score": float(sc),
                }
            persons.append({"keypoints": kps})
        frames_out.append({"frame_idx": m.frame_idx, "persons": persons})

    kp_json = {"video": str(video_path), "fps": fps, "frames": frames_out}
    with open(cache_path, "w") as f:
        json.dump(kp_json, f)
    print(f"    saved → {cache_path}")

print("\nDone.")
