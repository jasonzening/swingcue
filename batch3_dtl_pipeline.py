#!/usr/bin/env python3
"""
batch3_dtl_pipeline.py — Step 3 + Step 5 for dtl-eet-2 and dtl-eet-3

Step 3: Full A→F pipeline per clip + hip_rear (R2' v1.1) + spine_delta peak
Step 5: hip_rear peak frame + address frame annotated renders → preview/batch3_eet/

Per-clip report line:
  file / anchors / hip_mid_peak(fr/%) / hip_rear_peak(fr/%) /
  spine_peak(fr/°) / R1状态 / R2状态 / diagnosis_text / nan_count
"""

import sys, json, math, shutil, time
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.a_measurement.pose_pipeline import PosePipeline, JOINT_NAMES
from engine.b_phase.swing_phase import SwingPhaseEngine
from engine.c_features.feature_extractor import FeatureExtractor
from engine.c_features.hip_rear_extractor import HipRearExtractor
from engine.orientation.resolver import OrientationResolver
from engine.layer0.perception_gate import PerceptionGate
from src.judgment.rules import bone_length_sentinel, r1_loss_of_posture, r2_hip_toward_ball
from src.judgment.root_cause import RootCauseEngine
from src.judgment.output import CoachingOutput
from engine.a_measurement.kp_guard import kp_guard

PROJ     = Path(__file__).resolve().parent
INPUT    = PROJ / "input"
KP_CACHE = PROJ / "engine/kp_cache/batch3"
KP_CACHE.mkdir(parents=True, exist_ok=True)
OUT_DIR  = PROJ / "output/batch3_eet"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DESK = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/batch3_eet")
DESK.mkdir(parents=True, exist_ok=True)

CLIPS = ["dtl-eet-2", "dtl-eet-3"]


# ── A-layer: load or run RTMPose ──────────────────────────────────────────────

def load_or_run(stem: str, video_path: Path):
    cache = KP_CACHE / f"{stem}.json"
    pipeline = PosePipeline(device="cuda")
    if cache.exists():
        print(f"  A-layer: cache hit ({cache.name})")
        with open(cache) as f:
            kp_json = json.load(f)
        return pipeline.run_from_json(kp_json), kp_json
    print(f"  A-layer: RTMPose running on {video_path.name}...")
    t0 = time.time()
    meas, fps = pipeline.run(str(video_path), verbose=False)
    print(f"  A-layer: {len(meas)}fr in {time.time()-t0:.1f}s @{fps:.1f}fps")
    # save cache
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
    kp_json = {"video": str(video_path), "fps": fps, "frames": frames_out}
    with open(cache, "w") as f:
        json.dump(kp_json, f)
    return (meas, fps), kp_json


# ── Step 5 rendering: annotate a single frame with hip/spine lines ────────────

def draw_annotated_frame(video_path: Path, frame_idx: int,
                          kp_json: dict, anchors,
                          label: str, out_path: Path):
    """Draw RTMPose skeleton + hip_mid line on a specific frame."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"  WARN: could not read fr{frame_idx} from {video_path.name}")
        return

    frames_data = kp_json.get("frames", [])
    kps = {}
    if frame_idx < len(frames_data):
        fd = frames_data[frame_idx]
        if fd.get("persons"):
            kps = fd["persons"][0].get("keypoints", {})

    def pt(name, score_thr=0.3):
        k = kps.get(name)
        if k and k.get("score", 0) >= score_thr and (k["x"] > 0 or k["y"] > 0):
            return (int(k["x"]), int(k["y"]))
        return None

    # Draw skeleton (basic)
    EDGES = [
        ("left_shoulder","right_shoulder"),
        ("left_shoulder","left_elbow"), ("left_elbow","left_wrist"),
        ("right_shoulder","right_elbow"), ("right_elbow","right_wrist"),
        ("left_shoulder","left_hip"), ("right_shoulder","right_hip"),
        ("left_hip","right_hip"),
        ("left_hip","left_knee"), ("left_knee","left_ankle"),
        ("right_hip","right_knee"), ("right_knee","right_ankle"),
    ]
    for a_name, b_name in EDGES:
        a, b = pt(a_name), pt(b_name)
        if a and b:
            cv2.line(frame, a, b, (0, 220, 0), 2)

    # Keypoints
    for name in JOINT_NAMES:
        p = pt(name)
        if p:
            cv2.circle(frame, p, 4, (0, 255, 255), -1)

    # Hip mid line (horizontal reference at address hip level)
    lh = pt("left_hip"); rh = pt("right_hip")
    if lh and rh:
        hip_y = (lh[1] + rh[1]) // 2
        hip_x = (lh[0] + rh[0]) // 2
        cv2.line(frame, (0, hip_y), (frame.shape[1], hip_y), (0, 140, 255), 1)
        cv2.circle(frame, (hip_x, hip_y), 6, (0, 140, 255), -1)

    # Address spine reference line (shoulder_mid → hip_mid)
    addr_fr = anchors.address
    addr_kps = {}
    if addr_fr < len(frames_data):
        fd = frames_data[addr_fr]
        if fd.get("persons"):
            addr_kps = fd["persons"][0].get("keypoints", {})

    def apt(name):
        k = addr_kps.get(name)
        if k and k.get("score", 0) >= 0.3 and (k["x"] > 0 or k["y"] > 0):
            return (int(k["x"]), int(k["y"]))
        return None

    als = apt("left_shoulder"); ars = apt("right_shoulder")
    alh = apt("left_hip");      arh = apt("right_hip")
    if als and ars and alh and arh:
        sh_mid = ((als[0]+ars[0])//2, (als[1]+ars[1])//2)
        hi_mid = ((alh[0]+arh[0])//2, (alh[1]+arh[1])//2)
        cv2.line(frame, sh_mid, hi_mid, (255, 255, 0), 2)  # yellow = address spine

    # Label
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h-50), (w, h), (0,0,0), -1)
    cv2.putText(frame, f"{stem}  fr{frame_idx}  {label}",
                (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.imwrite(str(out_path), frame)
    print(f"  Rendered: {out_path.name}")


# ── Main pipeline per clip ────────────────────────────────────────────────────

extractor_rear = HipRearExtractor(device="cuda")

def run_clip(stem: str):
    global extractor_rear
    video_path = INPUT / f"{stem}.mp4"
    print(f"\n{'='*60}")
    print(f"Step 3: {stem}")

    # Assert PASS gate
    gate = PerceptionGate()
    gr = gate.load(stem)
    if gr is None or gr.verdict not in ("PASS",):
        print(f"  WARN: gate record verdict={gr.verdict if gr else 'missing'} — continuing per human_override")

    # A-layer
    (meas, fps), kp_json = load_or_run(stem, video_path)
    n = len(meas)

    # B-layer
    engine_b = SwingPhaseEngine()
    annotations, anchors = engine_b.run(meas, fps, angle="down-the-line")
    phase_labels = [a.phase for a in annotations]
    print(f"  B-layer: addr={anchors.address} top={anchors.top}(tc={anchors.top_conf:.2f}) "
          f"impact={anchors.impact}(ic={anchors.impact_conf:.2f}) finish={anchors.finish} "
          f"swings={anchors.swing_count}")

    # C-layer
    feat = FeatureExtractor().extract(meas, fps, angle="down-the-line",
                                       address_frame=anchors.address)
    unreliable_ratio = float(np.mean(feat.unreliable))
    print(f"  C-layer: torso_h={feat.torso_h:.0f}px "
          f"addr_spine={feat.meta['addr_spine_angle']:.1f}° "
          f"unreliable={unreliable_ratio:.1%}")

    # spine_delta peak (P5→impact window)
    p5_fr = next((i for i, p in enumerate(phase_labels) if p == "transition"),
                 anchors.address)
    spine_win = list(range(p5_fr, min(anchors.impact + 1, n)))
    if spine_win:
        spine_vals = feat.spine_delta[spine_win]
        bi = int(np.argmax(np.abs(spine_vals)))
        spine_peak_fr  = spine_win[bi]
        spine_peak_deg = float(spine_vals[bi])
    else:
        spine_peak_fr, spine_peak_deg = anchors.impact, float("nan")

    # hip_mid peak (P5→impact)
    hip_win_phases = {"transition", "downswing", "impact"}
    hip_win_idx = [i for i, p in enumerate(phase_labels) if p in hip_win_phases]
    if hip_win_idx:
        arr = feat.hip_disp[hip_win_idx]
        bi = int(np.argmax(arr))
        hip_mid_peak_fr  = hip_win_idx[bi]
        hip_mid_peak_pct = float(arr[bi]) * 100.0
    else:
        hip_mid_peak_fr = anchors.impact
        hip_mid_peak_pct = float("nan")

    # Orientation
    resolver = OrientationResolver()
    ori = resolver.resolve(measurements=meas, angle="down-the-line",
                            address_frame=anchors.address,
                            top_frame=anchors.top,
                            impact_frame=anchors.impact)
    ball_side = ori.ball_side if ori.ball_side else "right"
    print(f"  Orientation: ball_side={ball_side}")

    # R2' hip_rear via SAM2
    hip_rear_peak_fr  = None
    hip_rear_peak_pct = None
    nan_count = 0
    try:
        rear_result = extractor_rear.extract(
            str(video_path), meas, anchors,
            ball_side=ball_side,
            phase_labels=phase_labels,
            kp_json=kp_json,
        )
        valid_frs = [f for f in rear_result.window_frames
                     if not math.isnan(rear_result.hip_rear_disp[f])]
        nan_count = rear_result.nan_count
        if valid_frs:
            vals = [rear_result.hip_rear_disp[f] for f in valid_frs]
            bi = int(np.argmax(vals))
            hip_rear_peak_fr  = valid_frs[bi]
            hip_rear_peak_pct = vals[bi] * 100.0
            print(f"  hip_rear: fr{hip_rear_peak_fr} {hip_rear_peak_pct:+.1f}%  nan_count={nan_count}")
        else:
            print(f"  hip_rear: no valid frames  nan_count={nan_count}")
    except Exception as e:
        print(f"  hip_rear ERROR: {e}")
        nan_count = -1

    # Bone sentinel
    bone_length_ratios = {}
    for bk in ["left_hip_left_knee", "right_hip_right_knee"]:
        lengths = np.array([m.bone_lengths.get(bk, 0.0) for m in meas])
        med = float(np.median(lengths[lengths > 0])) if np.any(lengths > 0) else 1.0
        if med > 0:
            bone_length_ratios[bk] = lengths / med
    unreliable_mask = bone_length_sentinel(bone_length_ratios)

    # D-layer
    faults = []
    r1 = r1_loss_of_posture(feat.spine_delta, phase_labels,
                             joint_confidences=feat.joint_conf,
                             unreliable_mask=unreliable_mask if len(unreliable_mask)==n else None)
    r2 = r2_hip_toward_ball(feat.hip_disp, phase_labels,
                             joint_confidences=feat.joint_conf,
                             unreliable_mask=unreliable_mask if len(unreliable_mask)==n else None)
    if r1: faults.append(r1); print(f"  D R1: {r1.fault_type} {r1.severity} conf={r1.confidence:.3f}")
    if r2: faults.append(r2); print(f"  D R2: {r2.fault_type} {r2.severity} conf={r2.confidence:.3f}")
    if not faults: print("  D-layer: no faults detected")

    r1_state = f"{r1.severity}(conf={r1.confidence:.3f})" if r1 else "none"
    r2_state = f"{r2.severity}(conf={r2.confidence:.3f})" if r2 else "none"

    # E-layer
    engine_e = RootCauseEngine()
    rc = engine_e.analyze(faults)
    print(f"  E-layer: root_cause={rc.root_cause}  certainty={rc.certainty}")

    # F-layer
    coaching = CoachingOutput()
    out_f = coaching.generate(rc, unreliable_frame_ratio=unreliable_ratio)
    diagnosis_text = out_f.one_liner
    print(f"  F-layer: {diagnosis_text}")

    # Save JSON
    diag = {
        "video": stem, "angle": "down-the-line", "fps": fps, "n_frames": n,
        "b_layer": {
            "addr": anchors.address, "top": anchors.top,
            "top_conf": round(anchors.top_conf, 3),
            "impact": anchors.impact,
            "impact_conf": round(anchors.impact_conf, 3),
            "finish": anchors.finish, "swing_count": anchors.swing_count,
        },
        "c_layer": {
            "torso_h": feat.torso_h,
            "addr_spine_deg": feat.meta["addr_spine_angle"],
            "unreliable_ratio": round(unreliable_ratio, 4),
        },
        "hip_mid_peak": {"fr": hip_mid_peak_fr,
                         "pct": round(hip_mid_peak_pct,1) if not math.isnan(hip_mid_peak_pct) else None},
        "hip_rear_peak": {"fr": hip_rear_peak_fr,
                          "pct": round(hip_rear_peak_pct,1) if hip_rear_peak_pct is not None else None},
        "spine_peak": {"fr": spine_peak_fr,
                       "deg": round(spine_peak_deg,1) if not math.isnan(spine_peak_deg) else None},
        "nan_count": nan_count,
        "d_layer_r1": r1.to_dict() if r1 else None,
        "d_layer_r2": r2.to_dict() if r2 else None,
        "e_layer": {"root_cause": rc.root_cause, "certainty": rc.certainty},
        "f_layer": {"one_liner": diagnosis_text},
        **out_f.diagnosis_json,
    }
    json_path = OUT_DIR / f"{stem}_diagnosis.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(diag, jf, indent=2, ensure_ascii=False)
    shutil.copy(str(json_path), str(DESK / json_path.name))

    # ── Step 5: render address + hip_rear peak frames ─────────────────────────
    addr_render = DESK / f"{stem}_addr_annotated.jpg"
    draw_annotated_frame(video_path, anchors.address, kp_json, anchors,
                          f"ADDRESS addr=fr{anchors.address}", addr_render)

    if hip_rear_peak_fr is not None:
        peak_render = DESK / f"{stem}_hiprear_peak_fr{hip_rear_peak_fr}.jpg"
        draw_annotated_frame(video_path, hip_rear_peak_fr, kp_json, anchors,
                              f"HIP_REAR_PEAK {hip_rear_peak_pct:+.1f}% fr{hip_rear_peak_fr}", peak_render)
    else:
        print(f"  Step 5: hip_rear peak unavailable, rendering impact frame instead")
        imp_render = DESK / f"{stem}_impact_fr{anchors.impact}.jpg"
        draw_annotated_frame(video_path, anchors.impact, kp_json, anchors,
                              f"IMPACT fr{anchors.impact}", imp_render)

    return {
        "stem": stem, "n": n,
        "addr": anchors.address, "top": anchors.top,
        "top_conf": anchors.top_conf, "impact": anchors.impact,
        "impact_conf": anchors.impact_conf,
        "hip_mid_peak_fr": hip_mid_peak_fr,
        "hip_mid_peak_pct": hip_mid_peak_pct,
        "hip_rear_peak_fr": hip_rear_peak_fr,
        "hip_rear_peak_pct": hip_rear_peak_pct,
        "spine_peak_fr": spine_peak_fr,
        "spine_peak_deg": spine_peak_deg,
        "r1_state": r1_state,
        "r2_state": r2_state,
        "diagnosis": diagnosis_text,
        "nan_count": nan_count,
    }


if __name__ == "__main__":
    results = []
    for stem in CLIPS:
        r = run_clip(stem)
        results.append(r)

    print("\n\n=== STEP 3 REPORT TABLE (dtl-eet-2 / dtl-eet-3) ===")
    HDR = ("file", "addr", "top", "tc", "impact", "ic",
           "hip_mid(fr/%)", "hip_rear(fr/%)", "spine(fr/°)",
           "R1状态", "R2状态", "nan")
    print(f"{'file':<13} {'addr':>5} {'top':>5} {'tc':>5} {'impact':>7} {'ic':>5} "
          f"{'hip_mid(fr/%)':>14} {'hip_rear(fr/%)':>15} {'spine(fr/°)':>12} "
          f"{'R1状态':>20} {'R2状态':>20} {'nan':>4}")
    print("-"*130)
    for r in results:
        hm = (f"fr{r['hip_mid_peak_fr']}/{r['hip_mid_peak_pct']:+.1f}%"
              if r['hip_mid_peak_fr'] is not None and not math.isnan(r['hip_mid_peak_pct'])
              else "N/A")
        hr = (f"fr{r['hip_rear_peak_fr']}/{r['hip_rear_peak_pct']:+.1f}%"
              if r['hip_rear_peak_fr'] is not None else "N/A")
        sp = (f"fr{r['spine_peak_fr']}/{r['spine_peak_deg']:+.1f}°"
              if not math.isnan(r['spine_peak_deg']) else "N/A")
        print(f"{r['stem']:<13} {r['addr']:>5} {r['top']:>5} {r['top_conf']:>5.2f} "
              f"{r['impact']:>7} {r['impact_conf']:>5.2f} "
              f"{hm:>14} {hr:>15} {sp:>12} "
              f"{r['r1_state']:>20} {r['r2_state']:>20} {r['nan_count']:>4}")

    print("\n--- 诊断原文 ---")
    for r in results:
        print(f"  {r['stem']}: {r['diagnosis']}")

    print(f"\nStep 5 renders → Windows: C:\\Users\\jason\\Desktop\\rtmpose_results\\preview\\batch3_eet\\")
    print(f"Files: <stem>_addr_annotated.jpg + <stem>_hiprear_peak_fr*.jpg")
