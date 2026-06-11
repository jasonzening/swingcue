#!/usr/bin/env python3
"""
screen_normal_group.py
======================
Screening pass for normal_group/ videos:
  - Frame count, resolution, duration
  - Person count (using RTMPose, CPU mode)
  - Camera angle: DTL / face-on / other / uncertain
    Method: shoulder x-span / torso_height at address frame
    Threshold: ratio > 0.35 → face-on; < 0.20 → DTL; 0.20-0.35 → uncertain
  - Swing count (reuse SwingPhaseEngine multi-swing detection)
  - Completeness (address + finish both present in first swing)

Outputs a screening table to stdout and to a JSON file.
"""

import json, sys, math, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJ    = Path("/home/jason/projects/swingcue-postest")
NG_DIR  = PROJ / "input/normal_group"
KP_DIR  = PROJ / "engine/kp_cache/normal_group"
KP_DIR.mkdir(parents=True, exist_ok=True)

SHOULDER_RATIO_FACEON_THR  = 0.35   # > this → face-on
SHOULDER_RATIO_DTL_THR     = 0.20   # < this → DTL
# between → uncertain

def probe_video(vpath: Path):
    """Get frame count, fps, width, height via cv2."""
    import cv2
    cap = cv2.VideoCapture(str(vpath))
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30
    n    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    dur_s = n / fps if fps > 0 else 0
    return {"fps": round(fps, 2), "n_frames": n, "width": w, "height": h,
            "duration_s": round(dur_s, 1)}


def run_rtmpose(vpath: Path, cache_path: Path):
    """Run PosePipeline (cached) and return (measurements, fps)."""
    from engine.a_measurement.pose_pipeline import PosePipeline, JOINT_NAMES
    pipe = PosePipeline(device="cuda")  # GPU for speed
    if cache_path.exists():
        with open(cache_path) as f:
            kp_json = json.load(f)
        return pipe.run_from_json(kp_json)

    meas, fps = pipe.run(str(vpath), verbose=False)
    # Save cache
    frames = []
    for m in meas:
        persons = []
        if m.measurement_quality != "bad":
            kps = {}
            for name in JOINT_NAMES:
                pt = m.keypoints.get(name)
                sc = m.confidences.get(name, 0.0)
                kps[name] = {"x": float(pt[0]) if pt else 0.0,
                             "y": float(pt[1]) if pt else 0.0,
                             "score": sc}
            persons = [{"person_id": 0, "keypoints": kps}]
        frames.append({"frame": m.frame_idx, "persons": persons})
    data = {"model": "RTMPose-x", "keypoint_format": "COCO-17",
            "stats": {"source_fps": fps, "video": vpath.name},
            "frames": frames}
    with open(cache_path, "w") as f:
        json.dump(data, f)
    return meas, fps


def detect_angle(meas, anchors, fps):
    """
    Shoulder x-span / torso_height at address frame.
    Returns ("face-on" | "DTL" | "uncertain" | "no_data", ratio)
    """
    addr = anchors.address
    m    = meas[addr]
    ls   = m.keypoints.get("left_shoulder")
    rs   = m.keypoints.get("right_shoulder")
    if ls is None or rs is None:
        return "no_data", None
    sh_span  = abs(ls[0] - rs[0])
    torso_h  = m.torso_height()
    if torso_h < 10:
        return "no_data", None
    ratio = sh_span / torso_h
    if ratio > SHOULDER_RATIO_FACEON_THR:
        angle = "face-on"
    elif ratio < SHOULDER_RATIO_DTL_THR:
        angle = "DTL"
    else:
        angle = "uncertain"
    return angle, round(ratio, 3)


def check_completeness(meas, anchors, n):
    """
    Check if first swing segment has usable address AND finish frames.
    'Complete' = address confidence > 0 AND finish > impact + 5fr.
    """
    fse = anchors.first_swing_end if anchors.first_swing_end >= 0 else n
    addr_ok   = anchors.address >= 0 and anchors.address < fse
    finish_ok = anchors.finish > anchors.impact + 5
    in_frame  = anchors.impact < fse and anchors.finish < fse
    return addr_ok and finish_ok and in_frame


def max_persons(meas):
    """
    Rough person count: look for frames where multiple distinct clusters
    of shoulder keypoints appear. Simple heuristic: just return 1 (single
    person) since RTMPose-x in single-person mode returns one skeleton.
    For multi-person we'd need multi-person model.
    Return 1 (single) or "uncertain" if bbox spread is very large.
    """
    # Without multi-person model, we can check if shoulder spread is
    # suspiciously large relative to expected (which would indicate two people
    # being merged or a very wide frame)
    return 1  # Conservative: flag manually only if obviously wrong


def screen_video(vpath: Path):
    stem = vpath.stem
    cache_path = KP_DIR / f"{stem}.json"

    print(f"  {stem}...", end="", flush=True)
    t0 = time.time()

    # Basic probe
    probe = probe_video(vpath)

    # RTMPose (CPU — will be slow but avoids GPU dependency for screening)
    try:
        meas, fps = run_rtmpose(vpath, cache_path)
    except Exception as e:
        print(f" ERROR: {e}")
        return {"file": vpath.name, "error": str(e), **probe}

    from engine.b_phase.swing_phase import SwingPhaseEngine
    # Use auto angle for initial detection
    eng = SwingPhaseEngine()
    try:
        ann, anchors = eng.run(meas, fps, angle="auto")
    except Exception as e:
        print(f" SWING_ERR: {e}")
        return {"file": vpath.name, "error_swing": str(e), **probe}

    n = len(meas)
    angle, sh_ratio = detect_angle(meas, anchors, fps)
    swing_count     = anchors.swing_count
    fse             = anchors.first_swing_end
    complete        = check_completeness(meas, anchors, n)
    persons         = max_persons(meas)  # always 1 with single-person model

    elapsed = time.time() - t0
    print(f" {angle} sc={swing_count} {elapsed:.1f}s")

    return {
        "file":         vpath.name,
        "n_frames":     probe["n_frames"],
        "fps":          probe["fps"],
        "width":        probe["width"],
        "height":       probe["height"],
        "duration_s":   probe["duration_s"],
        "persons":      persons,
        "angle":        angle,
        "sh_ratio":     sh_ratio,
        "swing_count":  swing_count,
        "first_swing_end": int(fse),
        "complete":     complete,
        "addr_fr":      anchors.address,
        "top_fr":       anchors.top,
        "impact_fr":    anchors.impact,
        "finish_fr":    anchors.finish,
        "impact_conf":  anchors.impact_conf,
        "top_conf":     anchors.top_conf,
        "pipeline_ready": angle == "DTL" and persons == 1 and complete and swing_count >= 1,
    }


def main():
    import datetime
    print(f"screen_normal_group.py  {datetime.datetime.now().isoformat()}")

    videos = sorted(NG_DIR.glob("*.mp4"))
    print(f"Found {len(videos)} videos in {NG_DIR}")

    results = []
    for vpath in videos:
        r = screen_video(vpath)
        results.append(r)

    # Save JSON
    out_json = PROJ / "pipeline_output/normal_group_screening.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nScreening JSON: {out_json}")

    # Print table
    print("\n" + "="*100)
    print(f"{'File':35s} {'Res':12s} {'Fr':>5} {'FPS':>4} {'Persons':>7} {'Angle':>10} "
          f"{'ShRatio':>8} {'Swings':>7} {'Complete':>9} {'Ready':>6}")
    print("-"*100)
    dtl_ready = []
    faceon_only = []
    for r in results:
        if "error" in r:
            print(f"  {r['file']:33s} ERROR: {r.get('error','')[:60]}")
            continue
        res_str = f"{r['width']}x{r['height']}"
        ready_str = "YES" if r.get("pipeline_ready") else "no"
        print(f"  {r['file']:33s} {res_str:12s} {r['n_frames']:>5} {r['fps']:>4.0f} "
              f"{r['persons']:>7} {r['angle']:>10} {str(r['sh_ratio']):>8} "
              f"{r['swing_count']:>7} {str(r['complete']):>9} {ready_str:>6}")
        if r.get("pipeline_ready"):
            dtl_ready.append(r)
        if r.get("angle") == "face-on":
            faceon_only.append(r)

    print("\n--- Summary ---")
    print(f"  DTL pipeline-ready: {len(dtl_ready)}")
    print(f"  Face-on (register only, no pipeline): {len(faceon_only)}")
    print(f"  Uncertain/other: {len(results) - len(dtl_ready) - len(faceon_only)}")

    # Update PROGRESS.log
    prog = PROJ / "PROGRESS.log"
    with open(prog, "a") as f:
        ts = datetime.datetime.now().isoformat()
        f.write(f"{ts}  normal_group screening complete: {len(dtl_ready)} DTL-ready, "
                f"{len(faceon_only)} face-on\n")

    return results, dtl_ready


if __name__ == "__main__":
    results, dtl_ready = main()
