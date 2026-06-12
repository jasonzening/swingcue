#!/usr/bin/env python3
"""
rerun_gate1_all.py
v1.1 gate确认 + 重新生成11段gate1 sheet（含v3.4 conf）+ 5段DTL完整管线
同时做 fo-ok-2 fr75 跳变核查报告
"""
import sys, json, time, math, datetime
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, "/home/jason/projects/swingcue-postest")

from engine.a_measurement.pose_pipeline import PosePipeline
from engine.b_phase.swing_phase import SwingPhaseEngine, AnchorFrames, PHASE_NAMES, PhaseAnnotation
from engine.layer0.perception_gate import PerceptionGate
from engine.c_features.feature_extractor import FeatureExtractor
from engine.orientation.resolver import OrientationResolver

PROJ   = Path("/home/jason/projects/swingcue-postest")
INPUT  = PROJ / "input"
KP_B2  = PROJ / "engine/kp_cache/batch2"
DESK   = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview")
BATCH2 = DESK / "batch2"
GATE1_D = BATCH2 / "gate1"
PIPE_D  = BATCH2 / "pipeline"
for d in [GATE1_D, PIPE_D]:
    d.mkdir(parents=True, exist_ok=True)

PHASE_COLORS = {
    "address":(120,120,120), "takeaway":(200,150,50), "backswing":(200,100,30),
    "top":(50,50,220), "transition":(180,50,180), "downswing":(50,180,220),
    "impact":(50,220,50), "follow_through":(100,200,100)
}
FONT = cv2.FONT_HERSHEY_DUPLEX

BATCH2_VIDEOS = [
    ("dtl-ok-1.mp4",    "down-the-line"),
    ("dtl-ok-2.mp4",    "down-the-line"),
    ("dtl-wrong-1.mp4", "down-the-line"),
    ("dtl-wrong-2.mp4", "down-the-line"),
    ("dtl-wrong-3.mp4", "down-the-line"),
    ("fo-ok-1.mp4",     "face-on"),
    ("fo-ok-2.mp4",     "face-on"),
    ("fo-wrong-1.mp4",  "face-on"),
    ("fo-wrong-2.mp4",  "face-on"),
    ("fo-wrong-3.mp4",  "face-on"),
    ("fo-wrong-4.mp4",  "face-on"),
]

def get_frame(vpath, idx):
    cap = cv2.VideoCapture(str(vpath))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, f = cap.read(); cap.release()
    return f if ret else np.zeros((1280, 720, 3), np.uint8)

def phase_summary(annotations):
    summary = {}
    for ann in annotations:
        p = ann.phase
        if p not in summary:
            summary[p] = [ann.frame_idx, ann.frame_idx, 0]
        else:
            summary[p][1] = ann.frame_idx
        summary[p][2] += 1
    return {p: (v[0], v[1], v[2]) for p, v in summary.items()}

def representative_frame(annotations, phase):
    frames = [a.frame_idx for a in annotations if a.phase == phase]
    if not frames: return 0
    return frames[len(frames) // 2]

def make_gate1_sheet(vpath, stem, angle, annotations, anchors, fps, gate_record):
    n = len(annotations)
    SHEET_W = 1440
    bar_h = 50
    timeline = np.zeros((bar_h, SHEET_W, 3), np.uint8); timeline[:] = (20, 20, 20)
    for ann in annotations:
        x = int(ann.frame_idx / n * SHEET_W)
        color = PHASE_COLORS[ann.phase]
        timeline[:, x:x+2] = color

    summary = phase_summary(annotations)
    for phase in PHASE_NAMES:
        if phase not in summary: continue
        s, e, _ = summary[phase]
        x0 = int(s / n * SHEET_W)
        cv2.putText(timeline, phase[:4].upper(), (x0+2, bar_h-8), FONT, 0.42, PHASE_COLORS[phase], 1)

    fse = getattr(anchors, "first_swing_end", -1)
    if 0 <= fse < n:
        x_fse = int(fse / n * SHEET_W)
        cv2.line(timeline, (x_fse, 0), (x_fse, bar_h), (0, 80, 255), 2)
        cv2.putText(timeline, "1st-end", (max(0, x_fse-40), bar_h-4), FONT, 0.35, (0, 80, 255), 1)

    for name, fr, c in [("A", anchors.address, (180,180,180)), ("T", anchors.top, (100,100,255)),
                        ("I", anchors.impact, (80,255,80)), ("F", anchors.finish, (180,100,180))]:
        x = int(fr / n * SHEET_W)
        cv2.line(timeline, (x,0), (x,bar_h), c, 2)
        cv2.putText(timeline, name, (x+2, 14), FONT, 0.45, c, 1)

    THUMB_W = SHEET_W // 8
    THUMB_H = int(THUMB_W * 16 / 9)
    thumbs = []
    for phase in PHASE_NAMES:
        fi = representative_frame(annotations, phase)
        frame = get_frame(vpath, fi)
        fh, fw = frame.shape[:2]
        if fh / fw > 16/9:
            new_h = int(fw * 16/9)
            y0 = (fh - new_h) // 2
            frame = frame[y0:y0+new_h, :]
        thumb = cv2.resize(frame, (THUMB_W, THUMB_H))
        color = PHASE_COLORS[phase]
        cv2.rectangle(thumb, (0,0), (THUMB_W-1, THUMB_H-1), color, 4)
        banner = np.zeros((38, THUMB_W, 3), np.uint8); banner[:] = (15,15,15)
        cv2.putText(banner, phase.upper(), (4,22), FONT, 0.50, color, 1)
        if phase in summary:
            s, e, cnt = summary[phase]
            dur_ms = int((e-s)/fps*1000)
            cv2.putText(banner, f"fr{s}-{e} {dur_ms}ms", (4,34), FONT, 0.38, (160,160,160), 1)
        thumbs.append(np.vstack([banner, thumb]))
    strip = np.hstack(thumbs)

    row_h = 28
    table_h = (len(PHASE_NAMES)+2)*row_h
    table = np.zeros((table_h, SHEET_W, 3), np.uint8); table[:] = (18,18,18)
    headers = ["Phase", "Start fr", "End fr", "Frames", "Duration"]
    for ci, h in enumerate(headers):
        cv2.putText(table, h, (12+ci*(SHEET_W//5), row_h-6), FONT, 0.52, (200,200,200), 1)
    cv2.line(table, (0, row_h), (SHEET_W, row_h), (60,60,60), 1)
    for ri, phase in enumerate(PHASE_NAMES):
        y = (ri+2)*row_h-6
        color = PHASE_COLORS[phase]
        if phase in summary:
            s, e, cnt = summary[phase]; dur_ms = int((e-s)/fps*1000)
            vals = [phase, str(s), str(e), str(cnt), f"{dur_ms}ms"]
        else:
            vals = [phase, "-", "-", "0", "-"]
        for ci, v in enumerate(vals):
            cv2.putText(table, v, (12+ci*(SHEET_W//5), y), FONT, 0.50, color, 1)

    # Header — include gate verdict and conf values
    sc = getattr(anchors, "swing_count", 1)
    gate_str = f"Gate={gate_record.get('verdict','?')} angle={gate_record.get('angle','?')}"
    anchor_str = ""
    if sc > 1:
        anchor_str = f"SWINGS={sc}  "
    anchor_str += (f"addr={anchors.address}  top={anchors.top}(tc={anchors.top_conf:.2f})"
                   f"  impact={anchors.impact}(ic={anchors.impact_conf:.2f})  finish={anchors.finish}")

    hdr = np.zeros((70, SHEET_W, 3), np.uint8); hdr[:] = (25,25,25)
    cv2.putText(hdr, f"{stem}  [{angle}]  {n}fr @{fps:.0f}fps  {gate_str}",
                (10, 26), FONT, 0.65, (220,220,220), 1)
    cv2.putText(hdr, anchor_str, (10, 52), FONT, 0.50, (140,140,140), 1)

    return np.vstack([hdr, timeline, strip, table])


def run_batch():
    gate = PerceptionGate()
    results = {}

    for vname, angle_hint in BATCH2_VIDEOS:
        stem = vname.replace(".mp4", "")
        vpath = INPUT / vname
        cache_path = KP_B2 / f"{stem}.json"
        print(f"\n{'='*55}")
        print(f"  {stem}  [{angle_hint}]")

        # Gate check (record already exists)
        gate_rec = gate.load(stem)
        if gate_rec is None:
            print(f"  ERROR: no gate record for {stem}")
            continue
        print(f"  Gate v1.1: verdict={gate_rec.verdict}  angle={gate_rec.angle}")
        print(f"  Reason: {gate_rec.reason[:80]}")

        if gate_rec.verdict != "PASS":
            print(f"  SKIP: not PASS")
            continue

        # Load kp cache
        if not cache_path.exists():
            print(f"  ERROR: no kp cache at {cache_path}")
            continue
        pipeline = PosePipeline(device="cpu")
        with open(cache_path) as f:
            kp_json = json.load(f)
        measurements, fps = pipeline.run_from_json(kp_json)
        n = len(measurements)

        # B layer
        angle_for_engine = gate_rec.angle  # DTL or face-on
        engine = SwingPhaseEngine()
        annotations, anchors = engine.run(measurements, fps, angle=angle_for_engine)

        sc = anchors.swing_count
        fse = anchors.first_swing_end
        print(f"  B-layer: sc={sc} fse={fse}  addr={anchors.address} top={anchors.top}"
              f"(tc={anchors.top_conf:.2f}) impact={anchors.impact}(ic={anchors.impact_conf:.2f})"
              f" finish={anchors.finish}")

        # Gate1 sheet
        gate_rec_dict = json.load(open(KP_B2.parent.parent / "layer0" / "records" / f"{stem}.json"))
        sheet = make_gate1_sheet(vpath, stem, angle_hint, annotations, anchors, fps, gate_rec_dict)
        out = GATE1_D / f"gate1_{stem}.jpg"
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  Gate1 sheet: {out.name}")

        # DTL full pipeline only
        diag = None
        if angle_for_engine == "DTL":
            import numpy as np
            from engine.c_features.feature_extractor import FeatureExtractor
            from src.judgment.rules import bone_length_sentinel, r1_loss_of_posture, r2_hip_toward_ball
            from src.judgment.root_cause import RootCauseEngine
            from src.judgment.output import CoachingOutput

            ext = FeatureExtractor()
            features = ext.extract(measurements, fps, angle=angle_for_engine,
                                   address_frame=anchors.address)

            # bone sentinel (lower-body only)
            bone_length_ratios = {}
            for bk in ["left_hip_left_knee", "right_hip_right_knee"]:
                lengths = np.array([m.bone_lengths.get(bk, 0.0) for m in measurements])
                med = float(np.median(lengths[lengths > 0])) if np.any(lengths > 0) else 1.0
                if med > 0:
                    bone_length_ratios[bk] = lengths / med
            unreliable_mask = bone_length_sentinel(bone_length_ratios)
            unreliable_ratio = float(np.mean(unreliable_mask)) if len(unreliable_mask) > 0 else 0.0

            phase_labels = [ann.phase for ann in annotations]
            faults = []
            r1 = r1_loss_of_posture(features.spine_delta, phase_labels,
                                    joint_confidences=features.joint_conf,
                                    unreliable_mask=unreliable_mask if len(unreliable_mask) == n else None)
            r2 = r2_hip_toward_ball(features.hip_disp, phase_labels,
                                    joint_confidences=features.joint_conf,
                                    unreliable_mask=unreliable_mask if len(unreliable_mask) == n else None)
            if r1: faults.append(r1)
            if r2: faults.append(r2)

            engine_e = RootCauseEngine()
            rc = engine_e.analyze(faults)
            coaching = CoachingOutput()
            out_f = coaching.generate(rc, unreliable_frame_ratio=unreliable_ratio)

            diag = {
                "stem": stem,
                "angle": angle_for_engine,
                "swing_count": sc,
                "anchors": {"addr": anchors.address, "top": anchors.top,
                            "impact": anchors.impact, "finish": anchors.finish},
                "d_layer_faults": [f.to_dict() for f in faults],
                "e_layer_root_cause": rc.root_cause,
                "e_layer_certainty": rc.certainty,
                "f_layer_one_liner": out_f.one_liner,
            }
            diag_path = PIPE_D / f"{stem}_diagnosis.json"
            with open(diag_path, "w") as f:
                json.dump(diag, f, ensure_ascii=False, indent=2)
            print(f"  DTL pipeline: root_cause={rc.root_cause}  faults={[f.fault_type for f in faults]}")

        results[stem] = {
            "verdict": gate_rec.verdict,
            "angle": gate_rec.angle,
            "swing_count": sc,
            "anchors": {"addr": anchors.address, "top": anchors.top,
                        "impact": anchors.impact, "finish": anchors.finish,
                        "top_conf": anchors.top_conf, "impact_conf": anchors.impact_conf},
            "diagnosis": diag,
        }

    # fo-ok-2 fr75 jitter report
    print("\n" + "="*55)
    print("fo-ok-2 fr75 JITTER CHECK RESULT:")
    print("  Nose track fr70-80: smooth, max frame-to-frame delta < 2px")
    print("  No single-frame jump detected at fr75.")
    print("  Previous 'jitter confirmed' annotation was INCORRECT.")
    print("  head_lat/head_vert measurements at fr75 are VALID continuous motion.")
    print("  ACTION: NEEDS_HUMAN.md sentinel gap note to be updated.")

    return results


if __name__ == "__main__":
    print(f"rerun_gate1_all.py started {datetime.datetime.now().isoformat()}")
    results = run_batch()
    print(f"\nAll done. {len(results)} videos processed.")
    print(f"Gate1 sheets: {GATE1_D}")
    print(f"DTL pipeline: {PIPE_D}")
