#!/usr/bin/env python3
"""
teach_clips_pipeline.py — dtl-4/left|right + dlt-6/left|right (FACE-ON clips)

Step 1: Layer 0 gate records saved (face-on confirmed from VLM, not DTL)
Step 2: A-layer RTMPose keypoints (cached), B-layer Gate-1 8-phase
Step 3: Export address/top/impact thumbnails for human review
Step 4: Orientation (handedness) resolver
Step 5: head_ref_v2 face-on measurements: head_lat, head_vert, elbow angles
        NOTE: hip_mid / hip_rear / spine_delta are DTL-only — NOT computed here
Step 6: Paired diff (错误 − 正确) per pair

Gate note for all clips:
  VLM reads face-on for all 4. "dtl-4" / "dlt-6" are content category labels,
  NOT camera-angle labels. Pipeline proceeds as face-on.
  dlt-6/right: 2 persons visible → gate=needs_human, but NOT blocking;
               reported and entered into pipeline with caveat.
"""

import sys, json, math
from pathlib import Path
import cv2
import numpy as np

PROJ = Path("/home/jason/projects/swingcue-postest")
sys.path.insert(0, str(PROJ))

from engine.a_measurement.pose_pipeline import PosePipeline, JOINT_NAMES
from engine.b_phase.swing_phase import SwingPhaseEngine
from engine.a_measurement.kp_guard import head_ref_v2
from engine.orientation.resolver import OrientationResolver

SPLIT_BASE  = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/split_check")
KP_CACHE    = PROJ / "engine/kp_cache/teach"
KP_CACHE.mkdir(parents=True, exist_ok=True)
GATE_DIR    = PROJ / "engine/layer0/records"
OUT_DIR     = SPLIT_BASE / "teach_pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR   = OUT_DIR / "thumbnails"
THUMB_DIR.mkdir(parents=True, exist_ok=True)

# VLM gate results (pre-computed — see summary in script header)
VLM_GATE = {
    "dtl-4_left":  {
        "verdict": "PASS", "angle": "face-on", "persons_per_frame": 1,
        "reason": "4/5 frames face-on 1-person golf content; fr352 is end-card. "
                  "Angle=face-on (clip label 'dtl-4' is content category, not camera angle)."
    },
    "dtl-4_right": {
        "verdict": "PASS", "angle": "face-on", "persons_per_frame": 1,
        "reason": "4/5 frames face-on 1-person golf content; fr352 is end-card. "
                  "Angle=face-on."
    },
    "dlt-6_left":  {
        "verdict": "PASS", "angle": "face-on", "persons_per_frame": 1,
        "reason": "5/5 frames face-on 1-person golf content. KORSA sim. Angle=face-on."
    },
    "dlt-6_right": {
        "verdict": "needs_human", "angle": "face-on", "persons_per_frame": 2,
        "reason": "5/5 frames face-on; 2 persons detected — primary golfer + partial "
                  "coach/observer at left edge. Non-blocking; entered with caveat."
    },
}

CLIPS = [
    # (parent_stem, side, marker, video_path)
    ("dtl-4", "left",  "checkmark",  SPLIT_BASE / "dtl-4" / "left.mp4"),
    ("dtl-4", "right", "cross",      SPLIT_BASE / "dtl-4" / "right.mp4"),
    ("dlt-6", "left",  "cross",      SPLIT_BASE / "dlt-6" / "left.mp4"),
    ("dlt-6", "right", "OK",         SPLIT_BASE / "dlt-6" / "right.mp4"),
]


# ─── helpers ──────────────────────────────────────────────────────────────────

def save_gate_record(stem_side, vlm):
    rec = {
        "stem": stem_side,
        "vlm_angle": vlm["angle"],
        "vlm_verdict": vlm["verdict"],
        "vlm_reason": vlm["reason"],
        "persons_per_frame": vlm["persons_per_frame"],
        "human_override": None,
        "angle_note": "face-on (not DTL) — hip_mid/hip_rear/spine_delta features N/A",
        "dtl_features_available": False,
    }
    p = GATE_DIR / f"{stem_side}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    return rec


def kp_dict_from_fm(fm):
    """Convert FrameMeasurement → {name: {x,y,score}} dict for head_ref_v2."""
    d = {}
    for name in JOINT_NAMES:
        pt = fm.keypoints.get(name)
        sc = fm.confidences.get(name, 0.0)
        if pt is not None:
            d[name] = {"x": float(pt[0]), "y": float(pt[1]), "score": sc}
        else:
            d[name] = {"x": 0.0, "y": 0.0, "score": 0.0}
    return d


def run_or_cache(stem_side, video_path):
    cache = KP_CACHE / f"{stem_side}.json"
    pipeline = PosePipeline(device="cuda")
    if cache.exists():
        print(f"  A-layer: using cache {cache.name}")
        with open(cache) as f:
            kp_json = json.load(f)
        meas, fps = pipeline.run_from_json(kp_json)
        return meas, fps, kp_json

    print(f"  A-layer: running RTMPose on {video_path.name} ...")
    meas, fps = pipeline.run(str(video_path), verbose=True)

    frames_out = []
    for fm in meas:
        persons = []
        if fm.measurement_quality != "bad":
            kps = {}
            for nm in JOINT_NAMES:
                pt = fm.keypoints.get(nm)
                sc = fm.confidences.get(nm, 0.0)
                kps[nm] = {"x": float(pt[0]) if pt else 0.0,
                           "y": float(pt[1]) if pt else 0.0,
                           "score": sc}
            persons.append({"keypoints": kps})
        frames_out.append({"frame_idx": fm.frame_idx, "persons": persons})

    kp_json = {"video": str(video_path), "fps": fps, "frames": frames_out}
    with open(cache, "w") as f:
        json.dump(kp_json, f)
    print(f"  A-layer: cached → {cache.name}")
    return meas, fps, kp_json


def export_thumbnail(video_path, fr, label, out_path):
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fr_safe = min(max(0, fr), n - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fr_safe)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 55), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"fr{fr} {label}", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.imwrite(str(out_path), frame)


def elbow_angle(fm, side="right"):
    def pt(name):
        p = fm.keypoints.get(name)
        sc = fm.confidences.get(name, 0.0)
        if p is not None and sc >= 0.3:
            return p
        return None
    s = pt(f"{side}_shoulder"); e = pt(f"{side}_elbow"); w = pt(f"{side}_wrist")
    if not (s and e and w):
        return float("nan")
    v1 = (s[0]-e[0], s[1]-e[1]); v2 = (w[0]-e[0], w[1]-e[1])
    denom = math.hypot(*v1) * math.hypot(*v2) + 1e-9
    cos_a = (v1[0]*v2[0] + v1[1]*v2[1]) / denom
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))


def fmt(v, fmt_str=".1f", unit=""):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "nan"
    return f"{v:{fmt_str}}{unit}"


# ─── main per-clip ─────────────────────────────────────────────────────────────

def process_clip(stem, side, marker, video_path):
    stem_side = f"{stem}_{side}"
    print(f"\n{'='*60}")
    print(f"  {stem}/{side}  marker={marker}")

    # Step 1: Gate
    vlm = VLM_GATE[stem_side]
    save_gate_record(stem_side, vlm)
    print(f"  Gate: verdict={vlm['verdict']}  angle={vlm['angle']}")
    if vlm["verdict"] == "needs_human":
        print(f"  NOTE: needs_human — {vlm['reason']}")
    print(f"  NOTE: DTL features (hip_mid/hip_rear/spine_delta) NOT available for face-on")

    # Step 2: A-layer
    meas, fps, kp_json = run_or_cache(stem_side, video_path)
    n = len(meas)
    print(f"  A-layer: {n} frames @ {fps:.0f}fps")

    # B-layer
    engine_b = SwingPhaseEngine()
    annotations, anchors = engine_b.run(meas, fps, angle="face-on")
    print(f"  B-layer: addr={anchors.address} top={anchors.top}(tc={anchors.top_conf:.2f}) "
          f"impact={anchors.impact}(ic={anchors.impact_conf:.2f}) "
          f"finish={anchors.finish} swings={anchors.swing_count}")

    # Step 3: Thumbnails
    thumb_paths = {}
    for phase, fr in [("address", anchors.address),
                       ("top",     anchors.top),
                       ("impact",  anchors.impact)]:
        p = THUMB_DIR / f"{stem_side}_{phase}_fr{fr}.jpg"
        export_thumbnail(video_path, fr, phase, p)
        thumb_paths[phase] = str(p)
        print(f"  thumbnail [{phase}]: {p.name}")

    # Step 4: Orientation
    resolver = OrientationResolver()
    ori = resolver.resolve(
        measurements=meas, angle="face-on",
        address_frame=anchors.address,
        top_frame=anchors.top,
        impact_frame=anchors.impact,
    )
    print(f"  Orientation: handedness={ori.handedness}  target_side={ori.target_side}")

    # Step 5: head_ref_v2 measurements (P5→impact window)
    p5 = anchors.address   # using address as P5 equivalent (face-on)
    imp = anchors.impact

    # Get address reference head position
    addr_kp = kp_dict_from_fm(meas[p5])
    addr_ref = head_ref_v2(addr_kp)
    torso_h = meas[p5].torso_height() if hasattr(meas[p5], 'torso_height') else None
    if torso_h is None or torso_h == 0:
        # Estimate from hip-shoulder distance
        def _ys(fm, nm):
            p = fm.keypoints.get(nm)
            sc = fm.confidences.get(nm, 0.0)
            return p[1] if (p is not None and sc >= 0.3) else None
        l_sh = _ys(meas[p5], "left_shoulder"); r_sh = _ys(meas[p5], "right_shoulder")
        l_hp = _ys(meas[p5], "left_hip");      r_hp = _ys(meas[p5], "right_hip")
        sh_y = np.mean([v for v in [l_sh, r_sh] if v]) if any([l_sh, r_sh]) else None
        hp_y = np.mean([v for v in [l_hp, r_hp] if v]) if any([l_hp, r_hp]) else None
        torso_h = abs(hp_y - sh_y) * 2.0 if (sh_y and hp_y) else 200.0

    head_lat_series, head_vert_series, valid_frs = [], [], []
    if addr_ref:
        for fr in range(p5, min(imp + 1, n)):
            kp = kp_dict_from_fm(meas[fr])
            ref = head_ref_v2(kp)
            if ref:
                head_lat_series.append(ref.x - addr_ref.x)
                head_vert_series.append(addr_ref.y - ref.y)   # positive = upward
                valid_frs.append(fr)

    def peak_abs(series, frs):
        if not series: return float("nan"), -1
        idx = max(range(len(series)), key=lambda i: abs(series[i]))
        return series[idx], frs[idx]

    lat_px, lat_fr  = peak_abs(head_lat_series, valid_frs)
    vert_px, vert_fr = peak_abs(head_vert_series, valid_frs)

    def pct(px):
        if isinstance(px, float) and math.isnan(px): return None
        return round(100.0 * px / torso_h, 1) if torso_h else None

    # Elbow angles
    r_elb_addr   = elbow_angle(meas[anchors.address])
    r_elb_top    = elbow_angle(meas[anchors.top])
    r_elb_impact = elbow_angle(meas[anchors.impact])
    l_elb_impact = elbow_angle(meas[anchors.impact], "left")

    nan_count = sum(1 for v in [lat_px, vert_px, r_elb_addr, r_elb_top,
                                 r_elb_impact, l_elb_impact]
                    if isinstance(v, float) and math.isnan(v))

    print(f"  head_lat  peak: fr{lat_fr}  {fmt(pct(lat_px),'','')}%")
    print(f"  head_vert peak: fr{vert_fr}  {fmt(pct(vert_px),'','')}%")
    print(f"  R_elbow: addr={fmt(r_elb_addr)}° top={fmt(r_elb_top)}° impact={fmt(r_elb_impact)}°")
    print(f"  L_elbow_impact: {fmt(l_elb_impact)}°")
    print(f"  torso_h={torso_h:.1f}px  nan_count={nan_count}")

    result = {
        "clip": f"{stem}/{side}", "stem": stem, "side": side, "marker": marker,
        "gate_verdict": vlm["verdict"], "gate_angle": vlm["angle"],
        "gate_reason": vlm["reason"],
        "anchors": {
            "address": anchors.address, "top": anchors.top,
            "top_conf": round(anchors.top_conf, 3),
            "impact": anchors.impact, "impact_conf": round(anchors.impact_conf, 3),
            "finish": anchors.finish, "swing_count": anchors.swing_count,
        },
        "handedness": ori.handedness, "target_side": ori.target_side,
        "torso_h_px": round(torso_h, 1),
        "head_lat":  {"peak_fr": lat_fr,  "peak_pct": pct(lat_px)},
        "head_vert": {"peak_fr": vert_fr, "peak_pct": pct(vert_px)},
        "elbow": {
            "R_addr": round(r_elb_addr, 1) if not math.isnan(r_elb_addr) else None,
            "R_top":  round(r_elb_top, 1)  if not math.isnan(r_elb_top)  else None,
            "R_impact": round(r_elb_impact, 1) if not math.isnan(r_elb_impact) else None,
            "L_impact": round(l_elb_impact, 1) if not math.isnan(l_elb_impact) else None,
        },
        "dtl_features_available": False,
        "dtl_features_note": "hip_mid/hip_rear/spine_delta N/A — angle=face-on",
        "thumbnails": thumb_paths,
        "nan_count": nan_count,
    }

    out_p = OUT_DIR / f"{stem_side}_result.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def main():
    results = []
    for stem, side, marker, video_path in CLIPS:
        r = process_clip(stem, side, marker, video_path)
        results.append(r)

    # Save combined
    combined = {"clips": results}
    jp = OUT_DIR / "teach_pipeline_results.json"
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    # ── Per-clip report ────────────────────────────────────────────────────────
    print("\n\n" + "="*70)
    print("每片段汇报 (face-on; DTL特征 N/A)")
    print("="*70)
    hdr = (f"{'片段':14s} {'标识':10s} {'gate':12s} {'addr':>5} {'top':>5} "
           f"{'tc':>5} {'imp':>5} {'ic':>5} "
           f"{'head_lat(fr/%)':>15} {'head_vert(fr/%)':>15} "
           f"{'R_elb@imp':>10} {'nan':>4}")
    print(hdr)
    print("-" * 115)
    for r in results:
        a = r["anchors"]
        hl = r["head_lat"]; hv = r["head_vert"]; el = r["elbow"]
        hl_fr = hl["peak_fr"]; hv_fr = hv["peak_fr"]
        hl_s = f"fr{hl_fr}/{hl['peak_pct']}%" if hl_fr >= 0 else "N/A"
        hv_s = f"fr{hv_fr}/{hv['peak_pct']}%" if hv_fr >= 0 else "N/A"
        re_s = f"{el['R_impact']}°" if el["R_impact"] is not None else "nan"
        print(f"{r['clip']:14s} {r['marker']:10s} {r['gate_verdict']:12s} "
              f"{a['address']:>5} {a['top']:>5} {a['top_conf']:>5.2f} "
              f"{a['impact']:>5} {a['impact_conf']:>5.2f} "
              f"{hl_s:>15} {hv_s:>15} {re_s:>10} {r['nan_count']:>4}")

    # ── Paired diff ────────────────────────────────────────────────────────────
    def get_r(stem, side):
        return next(x for x in results if x["stem"] == stem and x["side"] == side)

    print("\n" + "="*70)
    print("配对差值 (face-on特征: head_lat / head_vert / R_elbow_impact)")
    print("注意: hip_mid / hip_rear / spine_delta 均 N/A (机位=face-on)")
    print("="*70)

    for pair_stem, wrong_side, correct_side in [("dtl-4","right","left"),
                                                  ("dlt-6","left","right")]:
        w = get_r(pair_stem, wrong_side)
        c = get_r(pair_stem, correct_side)

        def diff(w_val, c_val):
            if w_val is None or c_val is None: return "N/A"
            return round(w_val - c_val, 1)

        d_hl = diff(w["head_lat"]["peak_pct"],    c["head_lat"]["peak_pct"])
        d_hv = diff(w["head_vert"]["peak_pct"],   c["head_vert"]["peak_pct"])
        d_re = diff(w["elbow"]["R_impact"],        c["elbow"]["R_impact"])

        print(f"\n  {pair_stem}: 错误({wrong_side}) − 正确({correct_side})")
        print(f"    Δhead_lat (% torso)  : {d_hl}")
        print(f"    Δhead_vert (% torso) : {d_hv}")
        print(f"    ΔR_elbow@impact (°)  : {d_re}")

    # ── Thumbnail index ────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("缩略图路径 (每段 address/top/impact 供人工核关键帧):")
    print("Windows: C:\\Users\\jason\\Desktop\\rtmpose_results\\preview\\"
          "split_check\\teach_pipeline\\thumbnails\\")
    for r in results:
        print(f"\n  {r['clip']} ({r['marker']}):")
        for phase, p in r["thumbnails"].items():
            print(f"    [{phase}] {Path(p).name}")

    # ── Gate summary ──────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("感知门汇总:")
    for r in results:
        print(f"  {r['clip']:14s}  {r['gate_verdict']:12s}  angle={r['gate_angle']}  "
              f"persons={VLM_GATE[r['stem']+'_'+r['side']]['persons_per_frame']}")
    print("\nNOTE: 所有4段均为 face-on (非DTL)。clip名称 'dtl-4'/'dlt-6' 为内容")
    print("分类标签,非摄像机角度。dtl特征(hip_mid/hip_rear/spine_delta)不适用。")


if __name__ == "__main__":
    main()
