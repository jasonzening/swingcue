#!/usr/bin/env python3
"""
batch3_fo_gate1.py — batch3 fo-eet-* Gate-1 + head_ref v2 measurements (Step 4)

Runs:
  A-layer: RTMPose (or load cache)
  B-layer: SwingPhaseEngine (face-on)
  head_ref v2: head_lat / head_vert curves (addr→impact window)
  elbow angle: trailing elbow (right_elbow) at key frames

Output:
  engine/kp_cache/batch3/<stem>.json  (RTMPose cache)
  output/batch3_eet/<stem>_gate1.json  (anchors + measurements)
  printed per-clip summary table
"""

import sys, json, math, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.a_measurement.pose_pipeline import PosePipeline
from engine.b_phase.swing_phase import SwingPhaseEngine
from engine.a_measurement.kp_guard import kp_guard, head_ref_v2

KP_CACHE = Path("engine/kp_cache/batch3")
KP_CACHE.mkdir(parents=True, exist_ok=True)

OUT_DIR = Path("output/batch3_eet")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_DIR = Path("input")
FO_CLIPS = ["fo-eet-1", "fo-eet-2", "fo-eet-3"]


def load_or_run(stem: str, verbose: bool = False):
    video = VIDEO_DIR / f"{stem}.mp4"
    cache = KP_CACHE / f"{stem}.json"
    pipeline = PosePipeline(device="cuda")

    if cache.exists():
        print(f"  [{stem}] A-layer: cache hit")
        with open(cache) as f:
            kp_json = json.load(f)
        return pipeline.run_from_json(kp_json), kp_json
    else:
        print(f"  [{stem}] A-layer: RTMPose running...")
        t0 = time.time()
        meas, fps = pipeline.run(str(video), verbose=verbose)
        print(f"  [{stem}] A-layer: {len(meas)}fr in {time.time()-t0:.1f}s @{fps}fps")
        # Save cache
        from engine.a_measurement.pose_pipeline import JOINT_NAMES
        frames_out = []
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
                persons.append({"keypoints": kps})
            frames_out.append({"frame_idx": m.frame_idx, "persons": persons})
        kp_json = {"video": str(video), "fps": fps, "frames": frames_out}
        with open(cache, "w") as f:
            json.dump(kp_json, f)
        return (meas, fps), kp_json


def elbow_angle(kp_json_frame, side="right"):
    """Angle at elbow: shoulder-elbow-wrist."""
    kps = {}
    if kp_json_frame.get("persons"):
        kps = kp_json_frame["persons"][0]["keypoints"]

    def pt(name):
        k = kps.get(name)
        if k and (k["x"] > 0 or k["y"] > 0) and k.get("score", 0) >= 0.3:
            return (k["x"], k["y"])
        return None

    shoulder = pt(f"{side}_shoulder")
    elbow    = pt(f"{side}_elbow")
    wrist    = pt(f"{side}_wrist")

    if not (shoulder and elbow and wrist):
        return float("nan")

    # vectors from elbow
    v1 = (shoulder[0]-elbow[0], shoulder[1]-elbow[1])
    v2 = (wrist[0]-elbow[0],    wrist[1]-elbow[1])
    cos_a = (v1[0]*v2[0]+v1[1]*v2[1]) / (
        math.hypot(*v1) * math.hypot(*v2) + 1e-9
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))


def process(stem):
    print(f"\n=== {stem} ===")
    (meas, fps), kp_json = load_or_run(stem)
    n = len(meas)

    engine = SwingPhaseEngine()
    annotations, anchors = engine.run(meas, fps, angle="face-on")
    phase_labels = [a.phase for a in annotations]

    addr_fr   = anchors.address
    top_fr    = anchors.top
    impact_fr = anchors.impact
    finish_fr = anchors.finish
    top_conf  = anchors.top_conf
    imp_conf  = anchors.impact_conf
    swing_cnt = anchors.swing_count

    print(f"  Gate-1: addr={addr_fr} top={top_fr}(conf={top_conf:.2f}) "
          f"impact={impact_fr}(conf={imp_conf:.2f}) finish={finish_fr} "
          f"swings={swing_cnt}")

    torso_h = meas[addr_fr].torso_height() or 200.0

    # --- head_ref v2 curve (addr→impact) ---
    frames_data = kp_json["frames"]

    def get_kps(fr):
        fd = frames_data[fr] if fr < len(frames_data) else {"persons": []}
        return fd["persons"][0]["keypoints"] if fd.get("persons") else {}

    addr_ref = head_ref_v2(get_kps(addr_fr))
    addr_x = addr_ref.x if addr_ref else None
    addr_y = addr_ref.y if addr_ref else None

    # head_lat peak (X shift, positive = toward target = left for right-hander)
    # head_vert peak (Y shift, positive = upward = smaller y value = y decreasing)
    head_lat_series  = []  # px relative to addr
    head_vert_series = []  # px relative to addr
    valid_frs = []

    for fr in range(addr_fr, min(impact_fr + 1, n)):
        ref = head_ref_v2(get_kps(fr))
        if ref and addr_x is not None:
            dx = ref.x - addr_x     # +ve = moved toward target (rightward in face-on = toward ball)
            dy = addr_y - ref.y     # +ve = upward (y decreases when going up in image coords)
            head_lat_series.append(dx)
            head_vert_series.append(dy)
            valid_frs.append(fr)

    def pct(px): return round(100.0 * px / torso_h, 1) if torso_h else None

    # Peak values
    if head_lat_series:
        peak_lat_px = max(head_lat_series, key=abs)
        peak_lat_fr = valid_frs[head_lat_series.index(peak_lat_px)]
    else:
        peak_lat_px, peak_lat_fr = float("nan"), -1

    if head_vert_series:
        peak_vert_px = max(head_vert_series, key=abs)
        peak_vert_fr = valid_frs[head_vert_series.index(peak_vert_px)]
    else:
        peak_vert_px, peak_vert_fr = float("nan"), -1

    # Elbow angles at key frames
    def ea(fr, side="right"):
        if fr < 0 or fr >= len(frames_data):
            return float("nan")
        return elbow_angle(frames_data[fr], side)

    right_elbow_addr   = ea(addr_fr)
    right_elbow_top    = ea(top_fr)
    right_elbow_impact = ea(impact_fr)
    left_elbow_impact  = ea(impact_fr, "left")

    nan_count = sum(1 for v in [right_elbow_addr, right_elbow_top,
                                 right_elbow_impact, left_elbow_impact,
                                 peak_lat_px, peak_vert_px]
                    if (isinstance(v, float) and math.isnan(v)))

    result = {
        "stem": stem,
        "fps": fps,
        "n_frames": n,
        "anchors": {
            "address": addr_fr,
            "top": top_fr,
            "top_conf": round(top_conf, 3),
            "impact": impact_fr,
            "impact_conf": round(imp_conf, 3),
            "finish": finish_fr,
            "swing_count": swing_cnt,
        },
        "torso_height_px": round(torso_h, 1),
        "head_lat": {
            "peak_fr": peak_lat_fr,
            "peak_px": round(peak_lat_px, 1) if not math.isnan(peak_lat_px) else None,
            "peak_pct": pct(peak_lat_px) if not math.isnan(peak_lat_px) else None,
        },
        "head_vert": {
            "peak_fr": peak_vert_fr,
            "peak_px": round(peak_vert_px, 1) if not math.isnan(peak_vert_px) else None,
            "peak_pct": pct(peak_vert_px) if not math.isnan(peak_vert_px) else None,
        },
        "elbow_angles": {
            "right_at_address": round(right_elbow_addr, 1) if not math.isnan(right_elbow_addr) else None,
            "right_at_top":     round(right_elbow_top, 1)  if not math.isnan(right_elbow_top)  else None,
            "right_at_impact":  round(right_elbow_impact, 1) if not math.isnan(right_elbow_impact) else None,
            "left_at_impact":   round(left_elbow_impact, 1)  if not math.isnan(left_elbow_impact)  else None,
        },
        "nan_count": nan_count,
    }

    out_path = OUT_DIR / f"{stem}_gate1.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  head_lat:  peak fr{peak_lat_fr} {pct(peak_lat_px) if not math.isnan(peak_lat_px) else 'nan'}%")
    print(f"  head_vert: peak fr{peak_vert_fr} {pct(peak_vert_px) if not math.isnan(peak_vert_px) else 'nan'}%")
    print(f"  right_elbow: addr={right_elbow_addr:.1f}° top={right_elbow_top:.1f}° impact={right_elbow_impact:.1f}°")
    print(f"  left_elbow_impact: {left_elbow_impact:.1f}°")
    print(f"  nan_count: {nan_count}")
    return result


if __name__ == "__main__":
    results = []
    for stem in FO_CLIPS:
        r = process(stem)
        results.append(r)

    print("\n\n=== STEP 4 SUMMARY TABLE (fo-eet-*) ===")
    print(f"{'clip':<12} {'addr':>5} {'top':>5} {'tc':>5} {'impact':>7} {'ic':>5} "
          f"{'head_lat(fr/%)':>15} {'head_vert(fr/%)':>16} {'R_elb@imp':>10} {'L_elb@imp':>10} {'nan':>4}")
    print("-" * 105)
    for r in results:
        a = r["anchors"]
        hl = r["head_lat"]
        hv = r["head_vert"]
        ea = r["elbow_angles"]
        hl_str = f"fr{hl['peak_fr']}/{hl['peak_pct']}%" if hl['peak_fr'] >= 0 else "N/A"
        hv_str = f"fr{hv['peak_fr']}/{hv['peak_pct']}%" if hv['peak_fr'] >= 0 else "N/A"
        re_i = f"{ea['right_at_impact']}°" if ea['right_at_impact'] is not None else "nan"
        le_i = f"{ea['left_at_impact']}°"  if ea['left_at_impact']  is not None else "nan"
        print(f"{r['stem']:<12} {a['address']:>5} {a['top']:>5} {a['top_conf']:>5.2f} "
              f"{a['impact']:>7} {a['impact_conf']:>5.2f} "
              f"{hl_str:>15} {hv_str:>16} {re_i:>10} {le_i:>10} {r['nan_count']:>4}")

    print("\nOutputs written to output/batch3_eet/<stem>_gate1.json")
