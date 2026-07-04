#!/usr/bin/env python3
"""extract_dtl_eet1_kp.py — Run RTMPose on dtl-eet-1.mp4 to fill missing cache."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.a_measurement.pose_pipeline import PosePipeline

VIDEO = Path("/mnt/c/Users/jason/Zening/Swingcue/Video/dtl-eet-1.mp4")
OUT   = Path("engine/kp_cache/batch3/dtl-eet-1.json")

if OUT.exists():
    print(f"Cache already exists: {OUT}")
    sys.exit(0)

pipeline = PosePipeline(device="cuda")
print(f"Running RTMPose on {VIDEO.name}...")
t0 = time.time()
meas, fps = pipeline.run(str(VIDEO), verbose=False)
elapsed = time.time() - t0
print(f"Done: {len(meas)} frames in {elapsed:.1f}s @{fps:.1f}fps")

frames_out = []
for m in meas:
    kps = {}
    for name in m.keypoints:
        xy = m.keypoints[name]
        sc = m.confidences.get(name, 0.0)
        if xy is not None:
            kps[name] = {"x": xy[0], "y": xy[1], "score": sc}
        else:
            kps[name] = {"x": 0.0, "y": 0.0, "score": 0.0}
    persons = [{"keypoints": kps}] if kps else []
    frames_out.append({"frame_idx": m.frame_idx, "persons": persons})

kp_json = {
    "model": "rtmpose",
    "keypoint_format": "coco17",
    "stats": {"source_fps": fps, "total_frames": len(meas)},
    "frames": frames_out,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(kp_json, ensure_ascii=False))
print(f"Saved: {OUT}")
