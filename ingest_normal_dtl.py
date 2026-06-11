#!/usr/bin/env python3
"""
ingest_normal_dtl.py
====================
For each DTL-ready video from normal_group screening:
  1. Gate-1 summary sheet (reuse run_e1 logic)
  2. GT line rendering (tush line PROXY + spine axis)
  3. hip_disp / spine_delta measurement report (numbers only, no labels)
  4. A→F pipeline (run_pipeline.py logic, diagnosis saved verbatim)

Reads: pipeline_output/normal_group_screening.json
Outputs:
  preview/gate1/normal_group/<vid>/
  preview/gt_lines/normal_group/<vid>/
  preview/gt_measure/new/<vid>/
  preview/pipeline/new/<vid>_diagnosis.json
  pipeline_output/normal_group_summary.md

GT IRON RULE: no fault labels output at any step.
"""

import json, sys, shutil, math, time
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJ     = Path("/home/jason/projects/swingcue-postest")
NG_DIR   = PROJ / "input/normal_group"
KP_DIR   = PROJ / "engine/kp_cache/normal_group"
DESK     = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview")
GATE1_OUT = DESK / "gate1/normal_group"
LINES_OUT = DESK / "gt_lines/normal_group"
MEAS_OUT  = DESK / "gt_measure/new"
PIPE_OUT  = DESK / "pipeline/new"
for d in [GATE1_OUT, LINES_OUT, MEAS_OUT, PIPE_OUT]:
    d.mkdir(parents=True, exist_ok=True)

C_TUSH  = (0, 220, 255)   # yellow
C_SPINE = (255, 220, 0)   # cyan
C_WHITE = (255, 255, 255)
C_BLACK = (0, 0, 0)
LINE_W  = 3
FONT    = cv2.FONT_HERSHEY_DUPLEX

PHASE_COLORS = {
    "address": (120,120,120), "takeaway": (200,150,50), "backswing": (200,100,30),
    "top": (50,50,220), "transition": (180,50,180), "downswing": (50,180,220),
    "impact": (50,220,50), "follow_through": (100,200,100),
}
PHASE_NAMES = ["address","takeaway","backswing","top","transition","downswing","impact","follow_through"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_frame(vpath, idx):
    cap = cv2.VideoCapture(vpath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, f = cap.read(); cap.release()
    return f if ret else np.zeros((1280, 720, 3), np.uint8)

def kp_pt(kps, name, thr=0.3):
    if name not in kps: return None
    k = kps[name]
    if k["score"] < thr: return None
    return (float(k["x"]), float(k["y"]))

def mid_pt(a, b):
    if a is None or b is None: return None
    return ((a[0]+b[0])/2, (a[1]+b[1])/2)

def draw_vline(img, x, color, label="", proxy=False):
    h = img.shape[0]
    cv2.line(img, (int(x),0), (int(x),h), color, LINE_W, cv2.LINE_AA)
    tag = label + (" PROXY" if proxy else "")
    if tag: cv2.putText(img, tag, (int(x)+4, 40), FONT, 0.45, color, 1, cv2.LINE_AA)

def draw_spine(img, hip_mid, sh_mid, ext=0.20):
    dx = sh_mid[0]-hip_mid[0]; dy = sh_mid[1]-hip_mid[1]
    p1 = (int(hip_mid[0]-dx*ext), int(hip_mid[1]-dy*ext))
    p2 = (int(sh_mid[0]+dx*ext),  int(sh_mid[1]+dy*ext))
    cv2.line(img, p1, p2, C_SPINE, LINE_W, cv2.LINE_AA)
    cv2.circle(img, (int(hip_mid[0]),int(hip_mid[1])), 5, C_SPINE, -1, cv2.LINE_AA)
    cv2.circle(img, (int(sh_mid[0]),int(sh_mid[1])),   5, C_SPINE, -1, cv2.LINE_AA)

def label_frame(img, vid_id, fr, phase, extra=""):
    text = f"{vid_id} fr{fr:03d} {phase}"
    if extra: text += f" | {extra}"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.52, 1)
    cv2.rectangle(img, (0,0), (tw+12,th+12), C_BLACK, -1)
    cv2.putText(img, text, (6, th+4), FONT, 0.52, C_WHITE, 1, cv2.LINE_AA)

def save_jpg(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])


# ── Gate-1 summary sheet ──────────────────────────────────────────────────────

def make_gate1_sheet(vpath, vid_id, angle, annotations, anchors, fps, kp_json):
    from engine.b_phase.swing_phase import PHASE_NAMES as PN
    n = len(annotations)
    SHEET_W = 1440
    bar_h = 50
    timeline = np.zeros((bar_h, SHEET_W, 3), np.uint8); timeline[:] = (20,20,20)
    for ann in annotations:
        x = int(ann.frame_idx / n * SHEET_W)
        timeline[:, x:x+2] = PHASE_COLORS.get(ann.phase, (128,128,128))

    psummary = {}
    for ann in annotations:
        p = ann.phase
        if p not in psummary: psummary[p] = [ann.frame_idx, ann.frame_idx, 0]
        else: psummary[p][1] = ann.frame_idx
        psummary[p][2] += 1

    for name, fr, c in [("A",anchors.address,(180,180,180)),("T",anchors.top,(100,100,255)),
                          ("I",anchors.impact,(80,255,80)),("F",anchors.finish,(180,100,180))]:
        x = int(fr / n * SHEET_W)
        cv2.line(timeline, (x,0), (x,bar_h), c, 2)
        cv2.putText(timeline, name, (x+2,14), FONT, 0.45, c, 1)

    THUMB_W = SHEET_W // 8
    THUMB_H = int(THUMB_W * 16 / 9)
    thumbs = []
    for phase in PN:
        frs = [a.frame_idx for a in annotations if a.phase == phase]
        fi = frs[len(frs)//2] if frs else 0
        frame = get_frame(vpath, fi)
        fh, fw = frame.shape[:2]
        if fh/fw > 16/9:
            new_h = int(fw*16/9); y0=(fh-new_h)//2; frame=frame[y0:y0+new_h,:]
        thumb = cv2.resize(frame, (THUMB_W, THUMB_H))
        cv2.rectangle(thumb, (0,0), (THUMB_W-1,THUMB_H-1), PHASE_COLORS.get(phase,(128,128,128)), 4)
        banner = np.zeros((38, THUMB_W, 3), np.uint8); banner[:] = (15,15,15)
        cv2.putText(banner, phase.upper(), (4,22), FONT, 0.50, PHASE_COLORS.get(phase,(128,128,128)), 1)
        if phase in psummary:
            s, e, cnt = psummary[phase]
            cv2.putText(banner, f"fr{s}-{e}", (4,34), FONT, 0.38, (160,160,160), 1)
        thumbs.append(np.vstack([banner, thumb]))
    strip = np.hstack(thumbs)

    row_h = 28
    table_h = (len(PN)+2)*row_h
    table = np.zeros((table_h, SHEET_W, 3), np.uint8); table[:] = (18,18,18)
    headers = ["Phase","Start","End","Frames","Duration"]
    for ci, h in enumerate(headers):
        cv2.putText(table, h, (12+ci*(SHEET_W//5), row_h-6), FONT, 0.52, (200,200,200), 1)
    for ri, phase in enumerate(PN):
        y = (ri+2)*row_h-6
        c = PHASE_COLORS.get(phase,(128,128,128))
        if phase in psummary:
            s, e, cnt = psummary[phase]; d = int((e-s)/fps*1000)
            vals = [phase, str(s), str(e), str(cnt), f"{d}ms"]
        else:
            vals = [phase, "-", "-", "0", "-"]
        for ci, v in enumerate(vals):
            cv2.putText(table, v, (12+ci*(SHEET_W//5), y), FONT, 0.50, c, 1)

    sc = anchors.swing_count; fse = anchors.first_swing_end
    hdr = np.zeros((54, SHEET_W, 3), np.uint8); hdr[:] = (25,25,25)
    cv2.putText(hdr, f"{vid_id}  [{angle}]  {n}fr @{fps:.0f}fps", (10,26), FONT, 0.70, (220,220,220), 1)
    swing_str = f"SWINGS={sc} first_end=fr{fse}  " if sc > 1 else ""
    cv2.putText(hdr, f"{swing_str}addr=fr{anchors.address} top=fr{anchors.top}(tc={anchors.top_conf:.2f}) "
                f"impact=fr{anchors.impact}(ic={anchors.impact_conf:.2f}) finish=fr{anchors.finish}",
                (10,48), FONT, 0.50, (140,140,140), 1)
    return np.vstack([hdr, timeline, strip, table])


# ── GT line rendering (DTL: tush line + spine) ───────────────────────────────

def render_dtl_gt_lines(vpath, vid_id, kp_json, meas, fps, anchors, phase_map):
    out_dir = LINES_OUT / vid_id
    out_dir.mkdir(parents=True, exist_ok=True)

    addr_fr   = anchors.address
    impact_fr = anchors.impact
    n         = len(kp_json["frames"])

    fd0 = kp_json["frames"][addr_fr]
    if not fd0["persons"]: return 0
    kps0 = fd0["persons"][0]["keypoints"]
    lh = kp_pt(kps0,"left_hip"); rh = kp_pt(kps0,"right_hip")
    ls = kp_pt(kps0,"left_shoulder"); rs = kp_pt(kps0,"right_shoulder")
    if not (lh and rh): return 0

    hip_mid_addr = mid_pt(lh, rh)
    sh_mid_addr  = mid_pt(ls, rs)
    tush_x = hip_mid_addr[0]

    window_phases = {"transition","downswing","impact","follow_through"}
    frames_set = set()
    for fr, ph in phase_map.items():
        if ph in window_phases and fr % 2 == 0:
            frames_set.add(fr)
    frames_set.update([addr_fr, anchors.top, impact_fr])
    frames_set = {f for f in frames_set if f <= impact_fr + 5}

    count = 0
    for fr in sorted(frames_set):
        if fr >= n: continue
        raw = get_frame(vpath, fr)
        img = raw.copy()
        phase = phase_map.get(fr, "?")
        draw_vline(img, tush_x, C_TUSH, "TUSH", proxy=True)
        if sh_mid_addr:
            draw_spine(img, hip_mid_addr, sh_mid_addr)
        label_frame(img, vid_id, fr, phase)
        save_jpg(img, out_dir / f"fr{fr:03d}_{phase}.jpg")
        count += 1

    # Address overview
    ov = get_frame(vpath, addr_fr).copy()
    draw_vline(ov, tush_x, C_TUSH, "TUSH", proxy=True)
    if sh_mid_addr: draw_spine(ov, hip_mid_addr, sh_mid_addr)
    label_frame(ov, vid_id, addr_fr, "address", "ADDRESS_OVERVIEW")
    save_jpg(ov, out_dir / f"fr{addr_fr:03d}_ADDRESS_OVERVIEW.jpg")
    count += 1
    return count


# ── Measurement: hip_disp + spine_delta ──────────────────────────────────────

def measure_dtl(vid_id, meas, fps, anchors, phase_map):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from engine.c_features.feature_extractor import FeatureExtractor

    fe   = FeatureExtractor()
    feat = fe.extract(meas, fps, "down-the-line", anchors.address)
    n    = len(meas)
    frames = list(range(n))

    addr    = anchors.address
    top     = anchors.top
    impact  = anchors.impact
    p5_fr   = next((f for f in range(addr, n) if phase_map.get(f)=="transition"), addr)

    # Window peaks P5→impact
    if p5_fr < impact:
        hip_win  = feat.hip_disp[p5_fr:impact+1]
        sp_win   = feat.spine_delta[p5_fr:impact+1]
        hip_win_peak_idx = int(np.argmax(np.abs(hip_win)))
        sp_win_peak_idx  = int(np.argmax(np.abs(sp_win)))
        hip_win_peak_fr  = p5_fr + hip_win_peak_idx
        hip_win_peak_val = float(hip_win[hip_win_peak_idx])
        sp_win_peak_fr   = p5_fr + sp_win_peak_idx
        sp_win_peak_val  = float(sp_win[sp_win_peak_idx])
    else:
        hip_win_peak_fr = hip_win_peak_val = sp_win_peak_fr = sp_win_peak_val = float('nan')

    out_sub = MEAS_OUT / vid_id
    out_sub.mkdir(parents=True, exist_ok=True)

    for key, vals, ylabel, peak_fr, peak_val in [
        ("hip_disp",    feat.hip_disp,    "hip_disp (frac torso_h)", hip_win_peak_fr, hip_win_peak_val),
        ("spine_delta", feat.spine_delta, "spine_delta (deg)",       sp_win_peak_fr,  sp_win_peak_val),
    ]:
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(frames, vals, lw=1.5, label=ylabel)
        ax.axhline(0, color="#888", lw=0.8, ls=":")
        if not (isinstance(peak_fr, float) and math.isnan(peak_fr)):
            ax.scatter([peak_fr], [peak_val], s=80, color="red", zorder=5,
                       label=f"win_peak fr{peak_fr} {peak_val:+.3f}")
        for fr, lbl, c in [(addr,"ADDR","gray"),(top,"TOP","purple"),
                            (impact,"IMP","orange"),(p5_fr,"P5","cyan")]:
            ax.axvline(fr, color=c, ls="--", lw=1.2, alpha=0.8)
            ylo, yhi = ax.get_ylim()
            ax.text(fr+0.3, ylo+(yhi-ylo)*0.05, lbl, color=c, fontsize=7.5, rotation=90, va="bottom")
        ax.set_title(f"{vid_id} — {key}  (address baseline)")
        ax.set_xlabel("Frame"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(str(out_sub / f"{vid_id}_{key}.png"), dpi=120)
        plt.close()

    return {
        "hip_win_peak_fr":   int(hip_win_peak_fr) if not (isinstance(hip_win_peak_fr, float) and math.isnan(hip_win_peak_fr)) else None,
        "hip_win_peak_pct":  round(float(hip_win_peak_val)*100, 1) if not (isinstance(hip_win_peak_val, float) and math.isnan(hip_win_peak_val)) else None,
        "sp_win_peak_fr":    int(sp_win_peak_fr)  if not (isinstance(sp_win_peak_fr, float) and math.isnan(sp_win_peak_fr)) else None,
        "sp_win_peak_deg":   round(float(sp_win_peak_val), 2) if not (isinstance(sp_win_peak_val, float) and math.isnan(sp_win_peak_val)) else None,
        "torso_h":           round(feat.torso_h, 0),
        "p5_fr":             p5_fr,
    }


# ── A→F pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(vpath, vid_id, kp_json, meas, fps, anchors, phase_map):
    from src.judgment.rules import bone_length_sentinel, r1_loss_of_posture, r2_hip_toward_ball
    from src.judgment.root_cause import RootCauseEngine
    from src.judgment.output import CoachingOutput
    from engine.c_features.feature_extractor import FeatureExtractor
    import numpy as np

    feat = FeatureExtractor().extract(meas, fps, "down-the-line", anchors.address)
    unreliable_ratio = float(np.mean(feat.unreliable))
    phase_labels = [ann.phase for ann in
                    sorted([type('A', (), {'frame_idx': k, 'phase': v})()
                            for k, v in phase_map.items()], key=lambda x: x.frame_idx)]
    phase_labels_list = [phase_map.get(i, "address") for i in range(len(meas))]

    bone_keys = ["left_hip_left_knee", "right_hip_right_knee"]
    bone_ratios = {}
    for bk in bone_keys:
        lengths = np.array([m.bone_lengths.get(bk, 0.0) for m in meas])
        med = float(np.median(lengths[lengths > 0])) if np.any(lengths > 0) else 1.0
        if med > 0: bone_ratios[bk] = lengths / med
    unr_mask = bone_length_sentinel(bone_ratios)

    faults = []
    r1 = r1_loss_of_posture(feat.spine_delta, phase_labels_list,
                             joint_confidences=feat.joint_conf,
                             unreliable_mask=unr_mask if len(unr_mask)==len(meas) else None)
    r2 = r2_hip_toward_ball(feat.hip_disp, phase_labels_list,
                              joint_confidences=feat.joint_conf,
                              unreliable_mask=unr_mask if len(unr_mask)==len(meas) else None)
    if r1: faults.append(r1)
    if r2: faults.append(r2)

    rc  = RootCauseEngine().analyze(faults)
    out = CoachingOutput().generate(rc, unreliable_frame_ratio=unreliable_ratio)

    diag = {
        "video": vid_id, "angle": "down-the-line", "fps": fps, "n_frames": len(meas),
        "b_layer": {
            "swing_count": anchors.swing_count, "first_swing_end": anchors.first_swing_end,
            "address": anchors.address, "top": anchors.top, "top_conf": anchors.top_conf,
            "impact": anchors.impact, "impact_conf": anchors.impact_conf, "finish": anchors.finish,
        },
        "c_layer": {"torso_h": feat.torso_h, "unreliable_ratio": round(unreliable_ratio, 4)},
        "d_layer_faults": [f.to_dict() for f in faults],
        **out.diagnosis_json,
    }
    out_path = PIPE_OUT / f"{vid_id}_diagnosis.json"
    with open(out_path, "w", encoding="utf-8") as jf:
        json.dump(diag, jf, indent=2, ensure_ascii=False)
    return diag


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import datetime
    print(f"ingest_normal_dtl.py  {datetime.datetime.now().isoformat()}")

    # Load screening results
    screen_json = PROJ / "pipeline_output/normal_group_screening.json"
    if not screen_json.exists():
        print("ERROR: screening JSON not found, run screen_normal_group.py first")
        sys.exit(1)

    with open(screen_json) as f:
        screen_results = json.load(f)

    dtl_ready = [r for r in screen_results if r.get("pipeline_ready")]
    faceon_only = [r for r in screen_results if r.get("angle") == "face-on" and not r.get("pipeline_ready")]
    print(f"DTL-ready: {len(dtl_ready)}  Face-on (register only): {len(faceon_only)}")

    from engine.a_measurement.pose_pipeline import PosePipeline, JOINT_NAMES
    from engine.b_phase.swing_phase import SwingPhaseEngine

    pipeline_results = []

    for sr in dtl_ready:
        vid_file = sr["file"]
        vid_id   = Path(vid_file).stem
        vpath    = str(NG_DIR / vid_file)
        cache    = KP_DIR / f"{vid_id}.json"

        print(f"\n{'='*50}")
        print(f"{vid_id}")

        # Load keypoints
        with open(cache) as f:
            kp_json = json.load(f)
        pipe = PosePipeline(device="cuda")
        meas, fps = pipe.run_from_json(kp_json)
        n = len(meas)

        # B-layer (angle confirmed DTL from screening)
        eng = SwingPhaseEngine()
        ann, anchors = eng.run(meas, fps, angle="down-the-line")
        phase_map = {a.frame_idx: a.phase for a in ann}
        sc = anchors.swing_count; fse = anchors.first_swing_end
        print(f"  B-layer: sc={sc} fse=fr{fse} addr=fr{anchors.address} "
              f"top=fr{anchors.top} impact=fr{anchors.impact}")

        # Gate-1 sheet
        sheet = make_gate1_sheet(vpath, vid_id, "down-the-line", ann, anchors, fps, kp_json)
        gate1_path = GATE1_OUT / vid_id / f"gate1_{vid_id}.jpg"
        gate1_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(gate1_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  Gate-1: {gate1_path.name}")

        # GT line rendering
        n_frames = render_dtl_gt_lines(vpath, vid_id, kp_json, meas, fps, anchors, phase_map)
        print(f"  GT lines: {n_frames} frames")

        # Measurement
        meas_data = measure_dtl(vid_id, meas, fps, anchors, phase_map)
        print(f"  hip_win_peak: fr{meas_data['hip_win_peak_fr']} {meas_data['hip_win_peak_pct']}%  "
              f"spine_win_peak: fr{meas_data['sp_win_peak_fr']} {meas_data['sp_win_peak_deg']}deg")

        # A→F pipeline
        diag = run_pipeline(vpath, vid_id, kp_json, meas, fps, anchors, phase_map)
        print(f"  Diagnosis: root_cause={diag.get('root_cause')} certainty={diag.get('certainty')}")
        print(f"  One-liner: {diag.get('one_liner','')}")

        pipeline_results.append({
            "file":             vid_file,
            "vid_id":           vid_id,
            "swing_count":      sc,
            "addr_fr":          anchors.address,
            "top_fr":           anchors.top,
            "impact_fr":        anchors.impact,
            "finish_fr":        anchors.finish,
            "impact_conf":      anchors.impact_conf,
            "hip_win_peak_fr":  meas_data["hip_win_peak_fr"],
            "hip_win_peak_pct": meas_data["hip_win_peak_pct"],
            "sp_win_peak_fr":   meas_data["sp_win_peak_fr"],
            "sp_win_peak_deg":  meas_data["sp_win_peak_deg"],
            "root_cause":       diag.get("root_cause"),
            "certainty":        diag.get("certainty"),
            "one_liner":        diag.get("one_liner",""),
        })

    # Summary report
    _write_summary(screen_results, pipeline_results, faceon_only)

    # PROGRESS.log
    prog = PROJ / "PROGRESS.log"
    with open(prog, "a") as f:
        ts = datetime.datetime.now().isoformat()
        f.write(f"{ts}  normal_group ingest: {len(dtl_ready)} DTL, "
                f"{len(faceon_only)} face-on registered only\n")

    print(f"\nDone. Output: {DESK}")
    return pipeline_results


def _write_summary(screen_results, pipeline_results, faceon_only):
    """Write markdown summary — pure numbers, no fault labels."""
    lines = [
        "# Normal Group — Screening & Ingest Summary",
        "",
        f"**Date**: 2026-06-10  ",
        f"**Source**: OneDrive/Documents/stodownload*.mp4 (11 files) + test-dwontheline.mp4  ",
        "**GT iron rule**: no fault labels in this document.",
        "",
        "## Screening Table",
        "",
        "| File | Res | Frames | FPS | Angle | ShRatio | Swings | Complete | PipelineReady |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in screen_results:
        if "error" in r:
            lines.append(f"| {r['file']} | ERROR | — | — | — | — | — | — | no |")
            continue
        res = f"{r['width']}x{r['height']}"
        lines.append(f"| {r['file']} | {res} | {r['n_frames']} | {r['fps']:.0f} | "
                     f"{r['angle']} | {r.get('sh_ratio','?')} | {r.get('swing_count','?')} | "
                     f"{r.get('complete','?')} | {'YES' if r.get('pipeline_ready') else 'no'} |")

    lines += [
        "",
        "## DTL Pipeline Results — Per-Swing",
        "",
        "| File | Swings | addr | top | impact | impact_conf | hip_win_peak (fr/%) | spine_win_peak (fr/°) | Diagnosis (verbatim) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in pipeline_results:
        lines.append(
            f"| {r['file']} | {r['swing_count']} | {r['addr_fr']} | {r['top_fr']} | "
            f"{r['impact_fr']} | {r['impact_conf']:.3f} | "
            f"fr{r['hip_win_peak_fr']} / {r['hip_win_peak_pct']}% | "
            f"fr{r['sp_win_peak_fr']} / {r['sp_win_peak_deg']}° | "
            f"{r['root_cause']} / {r['certainty']} |"
        )

    # Distribution statistics
    hip_peaks = [r["hip_win_peak_pct"] for r in pipeline_results
                 if r.get("hip_win_peak_pct") is not None]
    sp_peaks  = [r["sp_win_peak_deg"] for r in pipeline_results
                 if r.get("sp_win_peak_deg") is not None]

    lines += ["", "## Normal Group Distribution Statistics", ""]
    if len(hip_peaks) >= 5:
        lines.append(f"hip_disp window peak (n={len(hip_peaks)}):  "
                     f"min={min(hip_peaks):.1f}%  median={float(np.median(hip_peaks)):.1f}%  max={max(hip_peaks):.1f}%")
    else:
        lines.append(f"hip_disp window peak (n={len(hip_peaks)} < 5, no stats): "
                     + str([f"{v:.1f}%" for v in hip_peaks]))
    if len(sp_peaks) >= 5:
        lines.append(f"spine_delta window peak (n={len(sp_peaks)}):  "
                     f"min={min(sp_peaks):.2f}°  median={float(np.median(sp_peaks)):.2f}°  max={max(sp_peaks):.2f}°")
    else:
        lines.append(f"spine_delta window peak (n={len(sp_peaks)} < 5, no stats): "
                     + str([f"{v:.2f}°" for v in sp_peaks]))

    # False-alarm count
    false_alarms = [r for r in pipeline_results
                    if r.get("root_cause") not in ("none", None) and r.get("certainty") != "none"]
    lines += [
        "",
        "## False-Alarm Count (申报正常段中引擎有输出的)",
        "",
        f"申报正常杆数: {len(pipeline_results)}  ",
        f"引擎输出非 none 的杆数: {len(false_alarms)}  ",
        "(是否真误报由人工看图裁决，本文件不含判断)",
        "",
    ]
    if false_alarms:
        lines.append("| File | root_cause | certainty | one_liner |")
        lines.append("|---|---|---|---|")
        for r in false_alarms:
            lines.append(f"| {r['file']} | {r['root_cause']} | {r['certainty']} | {r['one_liner']} |")

    lines += [
        "",
        "## Face-On Videos (registered, not processed — features not online)",
        "",
        "| File | Frames | angle | sh_ratio |",
        "|---|---|---|---|",
    ]
    for r in faceon_only:
        lines.append(f"| {r.get('file','')} | {r.get('n_frames','?')} | "
                     f"{r.get('angle','?')} | {r.get('sh_ratio','?')} |")

    out = PROJ / "pipeline_output/normal_group_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    desk_copy = DESK / "normal_group_summary.md"
    import shutil; shutil.copy(str(out), str(desk_copy))
    print(f"\n  Summary: {out}")


if __name__ == "__main__":
    main()
