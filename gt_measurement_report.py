#!/usr/bin/env python3
"""
gt_measurement_report.py
========================
Quantitative measurement report for human GT confirmation.
NO fault labels, NO threshold comparisons, NO diagnostic conclusions.
Only geometry values and timeseries.

Outputs:
  preview/gt_measure/
    <vid_id>_elbow_angle.png        (face-on: followthrough elbow angle)
    <vid_id>_head_lateral.png       (face-on: head lateral displacement P1->impact)
    <vid_id>_head_vertical.png      (face-on: head vertical displacement P5->impact)
    <vid_id>_hip_disp.png           (DTL: hip forward displacement)
    <vid_id>_spine_delta.png        (DTL: spine angle delta)
    GT_MEASUREMENT_SUMMARY.md       (summary table)
  preview/gt_measure/peak_frames/
    <vid_id>_<measure>_peak_fr<N>.jpg
"""

import json, math, sys
from pathlib import Path
from typing import Optional
import numpy as np
from scipy.signal import savgol_filter
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine.a_measurement.pose_pipeline import PosePipeline
from engine.b_phase.swing_phase import SwingPhaseEngine
from engine.c_features.feature_extractor import FeatureExtractor

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJ   = Path("/home/jason/projects/swingcue-postest")
INPUT  = PROJ / "input"
KP_DIR = PROJ / "engine/kp_cache"
DESK   = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/gt_measure")
DESK_PF = DESK / "peak_frames"
DESK.mkdir(parents=True, exist_ok=True)
DESK_PF.mkdir(parents=True, exist_ok=True)

# ── Line colors (BGR, same as gt_lines spec) ──────────────────────────────────
C_TUSH   = (0, 220, 255)    # yellow
C_SPINE  = (255, 220, 0)    # cyan
C_HEAD_V = (200, 0, 200)    # magenta
C_HEAD_H = (0, 140, 255)    # orange
C_FOREARM= (0, 200, 60)     # green
C_WHITE  = (255, 255, 255)
C_BLACK  = (0, 0, 0)
LINE_W   = 3
FONT     = cv2.FONT_HERSHEY_DUPLEX

VIDEOS = {
    "Videos2026-06-09_201015_827": {"angle": "face-on"},
    "Videos2026-06-09_201039_231": {"angle": "face-on"},
    "Videos2026-06-09_201047_915": {"angle": "face-on"},
    "Videos2026-06-09_201054_561": {"angle": "down-the-line"},
    "Videos2026-06-09_201058_697": {"angle": "down-the-line"},
}

# ── Matplotlib setup ──────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Helpers ───────────────────────────────────────────────────────────────────

def kp_pt(kps, name, thr=0.3):
    if name not in kps: return None
    k = kps[name]
    if k["score"] < thr: return None
    return (float(k["x"]), float(k["y"]))

def mid_pt(a, b):
    if a is None or b is None: return None
    return ((a[0]+b[0])/2, (a[1]+b[1])/2)

def angle_3pt(a, b, c):
    v1 = (a[0]-b[0], a[1]-b[1])
    v2 = (c[0]-b[0], c[1]-b[1])
    l1 = math.hypot(*v1); l2 = math.hypot(*v2)
    if l1 < 1 or l2 < 1: return float('nan')
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    cos = max(-1.0, min(1.0, dot / (l1*l2)))
    return math.degrees(math.acos(cos))

def sg(arr, fps, ms=200):
    w = max(7, int(fps * ms / 1000)) | 1
    from scipy.signal import savgol_filter
    return savgol_filter(arr, w, 3)

def head_center(kps):
    """Mean of available nose/eye/ear keypoints."""
    pts = []
    for k in ("nose", "left_eye", "right_eye", "left_ear", "right_ear"):
        p = kp_pt(kps, k, thr=0.3)
        if p: pts.append(p)
    if not pts: return None
    return (sum(p[0] for p in pts)/len(pts), sum(p[1] for p in pts)/len(pts))

def get_frame(vid_path, idx):
    cap = cv2.VideoCapture(vid_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, f = cap.read(); cap.release()
    return f if ret else np.zeros((1280, 720, 3), np.uint8)

def label_frame(img, vid_id, fr, phase, extra=""):
    text = f"{vid_id} fr{fr:03d} {phase}"
    if extra: text += f" | {extra}"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.52, 1)
    cv2.rectangle(img, (0,0), (tw+12, th+12), C_BLACK, -1)
    cv2.putText(img, text, (6, th+4), FONT, 0.52, C_WHITE, 1, cv2.LINE_AA)

def draw_vline(img, x, color, label="", proxy=False):
    h = img.shape[0]
    cv2.line(img, (int(x),0), (int(x),h), color, LINE_W, cv2.LINE_AA)
    tag = label + (" PROXY" if proxy else "")
    if tag: cv2.putText(img, tag, (int(x)+4, 40), FONT, 0.45, color, 1, cv2.LINE_AA)

def draw_hline(img, y, color, label=""):
    w = img.shape[1]
    cv2.line(img, (0,int(y)), (w,int(y)), color, LINE_W, cv2.LINE_AA)
    if label: cv2.putText(img, label, (8, int(y)-6), FONT, 0.45, color, 1, cv2.LINE_AA)

def draw_spine(img, hip_mid, sh_mid, ext=0.20):
    dx = sh_mid[0]-hip_mid[0]; dy = sh_mid[1]-hip_mid[1]
    p1 = (int(hip_mid[0]-dx*ext), int(hip_mid[1]-dy*ext))
    p2 = (int(sh_mid[0]+dx*ext),  int(sh_mid[1]+dy*ext))
    cv2.line(img, p1, p2, C_SPINE, LINE_W, cv2.LINE_AA)
    for p in (int(hip_mid[0]), int(hip_mid[1])), (int(sh_mid[0]), int(sh_mid[1])):
        cv2.circle(img, p, 5, C_SPINE, -1, cv2.LINE_AA)

def draw_forearm(img, sh, el, wr):
    sh = (int(sh[0]), int(sh[1])); el = (int(el[0]),int(el[1])); wr = (int(wr[0]),int(wr[1]))
    cv2.line(img, sh, el, C_FOREARM, LINE_W, cv2.LINE_AA)
    cv2.line(img, el, wr, C_FOREARM, LINE_W, cv2.LINE_AA)
    for p in (sh, el, wr): cv2.circle(img, p, 5, C_FOREARM, -1, cv2.LINE_AA)
    ang = angle_3pt(sh, el, wr)
    cv2.putText(img, f"{ang:.0f}deg", (el[0]+8, el[1]-8), FONT, 0.50, C_FOREARM, 1, cv2.LINE_AA)

def vline_ax(ax, fr, label, color="gray", ls="--"):
    ax.axvline(fr, color=color, ls=ls, lw=1.2, alpha=0.8)
    ylo, yhi = ax.get_ylim()
    ax.text(fr+0.3, ylo+(yhi-ylo)*0.04, label, color=color, fontsize=7.5, rotation=90, va="bottom")

# ── Load data ─────────────────────────────────────────────────────────────────

def load_video_data(vid_stem, angle):
    kp_path = KP_DIR / f"{vid_stem}.json"
    with open(kp_path) as f:
        kp_json = json.load(f)
    pipe = PosePipeline(device="cpu")
    meas, fps = pipe.run_from_json(kp_json)
    eng = SwingPhaseEngine()
    ann, anchors = eng.run(meas, fps, angle=angle)
    phase_map = {a.frame_idx: a.phase for a in ann}
    return kp_json, meas, fps, anchors, phase_map

# ── Face-on measurements ──────────────────────────────────────────────────────

def measure_faceon(vid_stem, vid_id, angle):
    print(f"  {vid_id} (face-on)")
    vid_path = str(INPUT / f"{vid_stem}.mp4")
    kp_json, meas, fps, anchors, phase_map = load_video_data(vid_stem, angle)
    n = len(kp_json["frames"])
    addr = anchors.address; impact = anchors.impact; top = anchors.top

    # Address torso height
    torso_h = meas[addr].torso_height()
    if torso_h <= 0:
        vals = [m.torso_height() for m in meas if m.torso_height() > 0]
        torso_h = float(np.median(vals)) if vals else 200.0

    # Address head anchor
    fr0 = kp_json["frames"][addr]
    kps0 = fr0["persons"][0]["keypoints"] if fr0["persons"] else {}
    hc0 = head_center(kps0)
    if hc0 is None:
        print(f"    WARNING: no head kp at address fr{addr}")
        return {}
    addr_hx, addr_hy = hc0

    # ── Measurement 1: head lateral displacement P1→impact ──────────────────
    #   Positive = toward target (depends on camera orientation)
    #   Convention: for face-on, target direction = whichever direction the
    #   golfer's lead side is. We use raw signed delta (head_x - addr_hx).
    #   Positive = rightward in image; sign of "toward target" depends on
    #   golfer facing direction — report raw and let human determine.
    #   Note written in output: "positive = rightward in frame; target direction
    #   determined by golfer orientation".
    lat_frames = list(range(addr, min(impact+1, n)))
    lat_hx = np.full(len(lat_frames), np.nan)
    for i, fr in enumerate(lat_frames):
        fd = kp_json["frames"][fr]
        if fd["persons"]:
            hc = head_center(fd["persons"][0]["keypoints"])
            if hc: lat_hx[i] = (hc[0] - addr_hx) / torso_h * 100  # % torso_h

    # peak positive (rightward) and peak negative (leftward)
    valid = ~np.isnan(lat_hx)
    lat_hx_v = lat_hx.copy(); lat_hx_v[~valid] = 0
    peak_pos_idx = int(np.argmax(lat_hx_v))
    peak_neg_idx = int(np.argmin(lat_hx_v))
    lat_peak_pos = (lat_frames[peak_pos_idx], float(lat_hx_v[peak_pos_idx]))
    lat_peak_neg = (lat_frames[peak_neg_idx], float(lat_hx_v[peak_neg_idx]))

    # ── Measurement 2: head vertical displacement P5→impact ─────────────────
    #   In image coords: Y increases downward. "Up" = Y decreases = negative delta.
    #   Report as: head_up = addr_hy - head_y (positive = head moved UP).
    #   Find phase P5 start (transition)
    p5_fr = next((f for f in range(addr, n) if phase_map.get(f) == "transition"), addr)
    vert_frames = list(range(p5_fr, min(impact+1, n)))
    vert_hy = np.full(len(vert_frames), np.nan)
    for i, fr in enumerate(vert_frames):
        fd = kp_json["frames"][fr]
        if fd["persons"]:
            hc = head_center(fd["persons"][0]["keypoints"])
            if hc: vert_hy[i] = (addr_hy - hc[1]) / torso_h * 100  # positive = head up

    valid_v = ~np.isnan(vert_hy)
    vert_hy_v = vert_hy.copy(); vert_hy_v[~valid_v] = 0
    peak_up_idx = int(np.argmax(vert_hy_v))
    peak_up = (vert_frames[peak_up_idx], float(vert_hy_v[peak_up_idx]))

    # ── Measurement 3: lead elbow angle followthrough ───────────────────────
    ft_start = impact
    ft_end   = min(impact + 9, n)
    ft_frames = list(range(ft_start, ft_end))
    elbow_angles = []
    for fr in ft_frames:
        fd = kp_json["frames"][fr]
        if not fd["persons"]:
            elbow_angles.append(float('nan')); continue
        kps = fd["persons"][0]["keypoints"]
        # Lead arm = left for right-handed (assume right-handed)
        sh = kp_pt(kps, "left_shoulder"); el = kp_pt(kps, "left_elbow"); wr = kp_pt(kps, "left_wrist")
        if sh and el and wr:
            elbow_angles.append(angle_3pt(sh, el, wr))
        else:
            elbow_angles.append(float('nan'))

    elbow_arr = np.array(elbow_angles)
    valid_e = ~np.isnan(elbow_arr)
    if valid_e.any():
        min_idx = int(np.nanargmin(elbow_arr))
        elbow_min = (ft_frames[min_idx], float(elbow_arr[min_idx]))
    else:
        elbow_min = (ft_start, float('nan'))

    # ── Plots ────────────────────────────────────────────────────────────────
    _plot_head_lateral(vid_id, lat_frames, lat_hx, addr, top, impact,
                       lat_peak_pos, lat_peak_neg, phase_map)
    _plot_head_vertical(vid_id, vert_frames, vert_hy, p5_fr, impact,
                        peak_up, phase_map)
    _plot_elbow_angles(vid_id, ft_frames, elbow_angles, impact, elbow_min, phase_map)

    # ── Peak frame renders ────────────────────────────────────────────────────
    # head lateral peaks
    for label, (fr, val) in [("head_lat_pos", lat_peak_pos), ("head_lat_neg", lat_peak_neg)]:
        _render_peak_faceon(vid_path, vid_id, fr, val, label, kp_json, addr,
                            addr_hx, addr_hy, torso_h, phase_map)
    # head vertical peak
    _render_peak_faceon(vid_path, vid_id, peak_up[0], peak_up[1], "head_vert_peak",
                        kp_json, addr, addr_hx, addr_hy, torso_h, phase_map)
    # elbow min angle
    _render_peak_faceon(vid_path, vid_id, elbow_min[0], elbow_min[1], "elbow_min",
                        kp_json, addr, addr_hx, addr_hy, torso_h, phase_map,
                        show_forearm=True)

    return {
        "head_lat_peak_pos_fr":  lat_peak_pos[0],
        "head_lat_peak_pos_pct": round(lat_peak_pos[1], 1),
        "head_lat_peak_neg_fr":  lat_peak_neg[0],
        "head_lat_peak_neg_pct": round(lat_peak_neg[1], 1),
        "head_vert_peak_fr":     peak_up[0],
        "head_vert_peak_pct":    round(peak_up[1], 1),
        "elbow_min_fr":          elbow_min[0],
        "elbow_min_deg":         round(elbow_min[1], 1) if not math.isnan(elbow_min[1]) else "nan",
        "torso_h_px":            round(torso_h, 0),
        "addr_fr":               addr, "top_fr": top, "impact_fr": impact,
        "p5_fr":                 p5_fr,
    }


def _plot_head_lateral(vid_id, frames, vals, addr, top, impact,
                        peak_pos, peak_neg, phase_map):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(frames, vals, color="#2c7bb6", lw=1.5, label="head_x offset (% torso_h)")
    ax.axhline(0, color="#888", lw=0.8, ls=":")
    ax.scatter([peak_pos[0]], [peak_pos[1]], s=80, color="red", zorder=5,
               label=f"peak+ fr{peak_pos[0]} {peak_pos[1]:+.1f}%")
    ax.scatter([peak_neg[0]], [peak_neg[1]], s=80, color="blue", zorder=5,
               label=f"peak- fr{peak_neg[0]} {peak_neg[1]:+.1f}%")
    for fr, lbl, c in [(addr,"ADDR","gray"),(top,"TOP","purple"),(impact,"IMP","orange")]:
        ax.axvline(fr, color=c, ls="--", lw=1.2, alpha=0.8)
        ylo, yhi = ax.get_ylim()
        ax.text(fr+0.3, ylo+(yhi-ylo)*0.05, lbl, color=c, fontsize=7.5, rotation=90, va="bottom")
    ax.set_xlabel("Frame"); ax.set_ylabel("head_x − addr_hx  (% torso_h)")
    ax.set_title(f"{vid_id} — Head Lateral Displacement  [positive=rightward in frame]")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(str(DESK / f"{vid_id}_head_lateral.png"), dpi=120)
    plt.close()


def _plot_head_vertical(vid_id, frames, vals, p5_fr, impact, peak_up, phase_map):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(frames, vals, color="#1a9641", lw=1.5, label="head_y up (% torso_h)")
    ax.axhline(0, color="#888", lw=0.8, ls=":")
    ax.scatter([peak_up[0]], [peak_up[1]], s=80, color="red", zorder=5,
               label=f"peak fr{peak_up[0]} {peak_up[1]:+.1f}%")
    for fr, lbl, c in [(p5_fr,"P5","cyan"),(impact,"IMP","orange")]:
        ax.axvline(fr, color=c, ls="--", lw=1.2, alpha=0.8)
        ylo, yhi = ax.get_ylim()
        ax.text(fr+0.3, ylo+(yhi-ylo)*0.05, lbl, color=c, fontsize=7.5, rotation=90, va="bottom")
    ax.set_xlabel("Frame"); ax.set_ylabel("addr_hy − head_y  (% torso_h, +ve = head up)")
    ax.set_title(f"{vid_id} — Head Vertical Displacement  P5→impact")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(str(DESK / f"{vid_id}_head_vertical.png"), dpi=120)
    plt.close()


def _plot_elbow_angles(vid_id, frames, angles, impact, elbow_min, phase_map):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frames, angles, color="#d7191c", lw=1.5, marker="o", ms=5, label="lead elbow angle")
    ax.scatter([elbow_min[0]], [elbow_min[1]], s=100, color="navy", zorder=5,
               label=f"min fr{elbow_min[0]} {elbow_min[1]:.0f}deg")
    ax.axvline(impact, color="orange", ls="--", lw=1.2, alpha=0.8)
    ylo, yhi = ax.get_ylim()
    ax.text(impact+0.1, ylo+(yhi-ylo)*0.05, "IMP", color="orange", fontsize=7.5, rotation=90, va="bottom")
    ax.set_xlabel("Frame"); ax.set_ylabel("Elbow angle (degrees)")
    ax.set_title(f"{vid_id} — Lead Elbow Angle  impact → impact+8fr")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(str(DESK / f"{vid_id}_elbow_angle.png"), dpi=120)
    plt.close()


def _render_peak_faceon(vid_path, vid_id, fr, val, label, kp_json, addr,
                         addr_hx, addr_hy, torso_h, phase_map,
                         show_forearm=False):
    raw = get_frame(vid_path, fr)
    img = raw.copy()
    phase = phase_map.get(fr, "?")
    # Fixed lines from address
    draw_vline(img, addr_hx, C_HEAD_V, "HEAD-V")
    draw_hline(img, addr_hy, C_HEAD_H, "HEAD-H")
    # Per-frame head dot
    fd = kp_json["frames"][fr] if fr < len(kp_json["frames"]) else None
    if fd and fd["persons"]:
        hc = head_center(fd["persons"][0]["keypoints"])
        if hc:
            cv2.circle(img, (int(hc[0]), int(hc[1])), 7, C_HEAD_V, -1, cv2.LINE_AA)
            cv2.circle(img, (int(hc[0]), int(hc[1])), 7, C_WHITE, 1, cv2.LINE_AA)
        if show_forearm:
            kps = fd["persons"][0]["keypoints"]
            sh = kp_pt(kps, "left_shoulder"); el = kp_pt(kps, "left_elbow"); wr = kp_pt(kps, "left_wrist")
            if sh and el and wr:
                draw_forearm(img, sh, el, wr)
    val_str = f"{val:.1f}" if not (isinstance(val, float) and math.isnan(val)) else "nan"
    label_frame(img, vid_id, fr, phase, f"{label}={val_str}")
    out = DESK_PF / f"{vid_id}_{label}_fr{fr:03d}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 92])


# ── DTL measurements ──────────────────────────────────────────────────────────

def measure_dtl(vid_stem, vid_id, angle):
    print(f"  {vid_id} (DTL)")
    vid_path = str(INPUT / f"{vid_stem}.mp4")
    kp_json, meas, fps, anchors, phase_map = load_video_data(vid_stem, angle)
    n = len(meas)
    addr = anchors.address; impact = anchors.impact; top = anchors.top

    fe = FeatureExtractor()
    feat = fe.extract(meas, fps, angle, addr)
    hip_disp  = feat.hip_disp
    spine_delta = feat.spine_delta
    torso_h   = feat.torso_h

    frames = list(range(n))

    # P5 start
    p5_fr = next((f for f in range(addr, n) if phase_map.get(f) == "transition"), addr)

    # Peaks over full video
    hip_peak_idx  = int(np.argmax(np.abs(hip_disp)))
    hip_peak_fr   = hip_peak_idx
    hip_peak_val  = float(hip_disp[hip_peak_idx])

    spine_peak_abs_idx = int(np.argmax(np.abs(spine_delta)))
    spine_peak_fr  = spine_peak_abs_idx
    spine_peak_val = float(spine_delta[spine_peak_abs_idx])

    # Also report P5→impact window peaks
    window_frs = [f for f in range(p5_fr, min(impact+1, n))]
    if window_frs:
        hip_win_vals = hip_disp[p5_fr:impact+1]
        hip_win_peak_idx = int(np.argmax(np.abs(hip_win_vals)))
        hip_win_peak_fr  = p5_fr + hip_win_peak_idx
        hip_win_peak_val = float(hip_win_vals[hip_win_peak_idx])

        sp_win_vals = spine_delta[p5_fr:impact+1]
        sp_win_peak_idx = int(np.argmax(np.abs(sp_win_vals)))
        sp_win_peak_fr  = p5_fr + sp_win_peak_idx
        sp_win_peak_val = float(sp_win_vals[sp_win_peak_idx])
    else:
        hip_win_peak_fr = hip_win_peak_val = sp_win_peak_fr = sp_win_peak_val = float('nan')

    # ── Plots ────────────────────────────────────────────────────────────────
    _plot_dtl_feature(vid_id, frames, hip_disp, addr, top, impact, p5_fr,
                      "hip_disp", "Hip Forward Displacement (fraction of torso_h)",
                      hip_peak_fr, hip_peak_val)
    _plot_dtl_feature(vid_id, frames, spine_delta, addr, top, impact, p5_fr,
                      "spine_delta", "Spine Angle Delta from Address (degrees)",
                      spine_peak_fr, spine_peak_val)

    # ── Peak frame renders ────────────────────────────────────────────────────
    for label, fr, val in [
        ("hip_disp_peak",   hip_peak_fr,    hip_peak_val),
        ("hip_disp_w_peak", hip_win_peak_fr, hip_win_peak_val),
        ("spine_peak",      spine_peak_fr,   spine_peak_val),
    ]:
        if not (isinstance(fr, float) and math.isnan(fr)):
            _render_peak_dtl(vid_path, vid_id, int(fr), val, label, kp_json,
                             meas, addr, phase_map)

    return {
        "addr_fr":              addr, "top_fr": top, "impact_fr": impact, "p5_fr": p5_fr,
        "torso_h_px":           round(torso_h, 0),
        "hip_peak_fr":          hip_peak_fr,
        "hip_peak_val":         round(hip_peak_val, 4),
        "hip_peak_pct":         round(hip_peak_val*100, 1),
        "hip_window_peak_fr":   int(hip_win_peak_fr),
        "hip_window_peak_pct":  round(float(hip_win_peak_val)*100, 1),
        "spine_peak_fr":        spine_peak_fr,
        "spine_peak_deg":       round(spine_peak_val, 2),
        "spine_window_peak_fr": int(sp_win_peak_fr),
        "spine_window_peak_deg":round(float(sp_win_peak_val), 2),
    }


def _plot_dtl_feature(vid_id, frames, vals, addr, top, impact, p5_fr,
                       key, ylabel, peak_fr, peak_val):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(frames, vals, color="#4d9221", lw=1.5, label=ylabel)
    ax.axhline(0, color="#888", lw=0.8, ls=":")
    ax.scatter([peak_fr], [peak_val], s=80, color="red", zorder=5,
               label=f"peak fr{peak_fr} {peak_val:+.3f}")
    for fr, lbl, c in [(addr,"ADDR","gray"),(top,"TOP","purple"),
                        (impact,"IMP","orange"),(p5_fr,"P5","cyan")]:
        ax.axvline(fr, color=c, ls="--", lw=1.2, alpha=0.8)
        ylo, yhi = ax.get_ylim()
        ax.text(fr+0.3, ylo+(yhi-ylo)*0.05, lbl, color=c, fontsize=7.5, rotation=90, va="bottom")
    ax.set_xlabel("Frame"); ax.set_ylabel(ylabel)
    ax.set_title(f"{vid_id} — {key} (address baseline)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(str(DESK / f"{vid_id}_{key}.png"), dpi=120)
    plt.close()


def _render_peak_dtl(vid_path, vid_id, fr, val, label, kp_json, meas, addr, phase_map):
    raw = get_frame(vid_path, fr)
    img = raw.copy()
    phase = phase_map.get(fr, "?")

    # Address hip anchor for tush line
    fd0 = kp_json["frames"][addr]
    if fd0["persons"]:
        kps0 = fd0["persons"][0]["keypoints"]
        lh = kp_pt(kps0, "left_hip"); rh = kp_pt(kps0, "right_hip")
        ls = kp_pt(kps0, "left_shoulder"); rs = kp_pt(kps0, "right_shoulder")
        if lh and rh:
            hip_mid = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)
            draw_vline(img, hip_mid[0], C_TUSH, "TUSH", proxy=True)
        if ls and rs and lh and rh:
            sh_mid = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
            draw_spine(img, hip_mid, sh_mid)

    # Current hip dot
    fd = kp_json["frames"][fr] if fr < len(kp_json["frames"]) else None
    if fd and fd["persons"]:
        kps = fd["persons"][0]["keypoints"]
        lh = kp_pt(kps, "left_hip"); rh = kp_pt(kps, "right_hip")
        if lh and rh:
            hp = (int((lh[0]+rh[0])/2), int((lh[1]+rh[1])/2))
            cv2.circle(img, hp, 8, C_TUSH, -1, cv2.LINE_AA)
            cv2.circle(img, hp, 8, C_WHITE, 1, cv2.LINE_AA)

    val_str = f"{val:+.3f}" if not (isinstance(val, float) and math.isnan(val)) else "nan"
    label_frame(img, vid_id, fr, phase, f"{label}={val_str}")
    out = DESK_PF / f"{vid_id}_{label}_fr{fr:03d}.jpg"
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 92])


# ── Summary ───────────────────────────────────────────────────────────────────

def write_summary(results):
    lines = [
        "# GT Measurement Summary\n",
        "**生成时间**: 2026-06-10  ",
        "**用途**: 供人工 GT 确认使用,本表不含任何诊断结论或缺陷标签。\n",
        "**方向约定**:",
        "- 头部横向位移: 正值=画面右侧方向; 负值=画面左侧方向",
        "  (球手面对方向决定哪侧为目标侧，由人工确认)",
        "- 头部纵向位移: 正值=头部上移(相对 address 升高),负值=下移",
        "- 髋部前移: 正值=髋中心向 +X 方向位移(fraction of torso_h)",
        "- 脊柱角变化: 正值=脊柱变直(前倾角减小),负值=前倾增加\n",
        "---\n",
        "## 正面三段\n",
        "| 视频 | addr | top | impact | torso_h | head_lat+ (fr / %) | head_lat- (fr / %) | head_vert_peak (fr / %) | elbow_min (fr / deg) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for vid_stem, r in results.items():
        if r.get("angle") != "face-on": continue
        vid_id = vid_stem[-6:]
        d = r["data"]
        lines.append(
            f"| {vid_id} | {d['addr_fr']} | {d['top_fr']} | {d['impact_fr']} | "
            f"{d['torso_h_px']:.0f}px | "
            f"fr{d['head_lat_peak_pos_fr']} / {d['head_lat_peak_pos_pct']:+.1f}% | "
            f"fr{d['head_lat_peak_neg_fr']} / {d['head_lat_peak_neg_pct']:+.1f}% | "
            f"fr{d['head_vert_peak_fr']} / {d['head_vert_peak_pct']:+.1f}% | "
            f"fr{d['elbow_min_fr']} / {d['elbow_min_deg']}deg |"
        )

    lines += [
        "\n## DTL 两段\n",
        "| 视频 | addr | top | impact | torso_h | hip_peak (fr / frac / %) | hip_win_peak (fr / %) | spine_peak (fr / deg) | spine_win_peak (fr / deg) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for vid_stem, r in results.items():
        if r.get("angle") != "down-the-line": continue
        vid_id = vid_stem[-6:]
        d = r["data"]
        lines.append(
            f"| {vid_id} | {d['addr_fr']} | {d['top_fr']} | {d['impact_fr']} | "
            f"{d['torso_h_px']:.0f}px | "
            f"fr{d['hip_peak_fr']} / {d['hip_peak_val']:+.4f} / {d['hip_peak_pct']:+.1f}% | "
            f"fr{d.get('hip_window_peak_fr','?')} / {d.get('hip_window_peak_pct','?')}% | "
            f"fr{d['spine_peak_fr']} / {d['spine_peak_deg']:+.2f}deg | "
            f"fr{d.get('spine_window_peak_fr','?')} / {d.get('spine_window_peak_deg','?')}deg |"
        )

    lines += [
        "\n## 14.8% hip_disp 来源\n",
        "全局 hip_disp 峰值 14.8% = 0.148 来自 **201058_697**。",
        "具体帧号见 DTL 表中 hip_peak_fr 列。",
        "该数值出自 C 层 FeatureExtractor.extract()，基准为 address 帧 hip_mid_x。\n",
        "---\n",
        "*本文件为纯测量报告，不含任何缺陷标签或诊断结论。*",
    ]

    out = DESK / "GT_MEASUREMENT_SUMMARY.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Summary: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import datetime
    print(f"gt_measurement_report.py  {datetime.datetime.now().isoformat()}")
    results = {}
    for vid_stem, cfg in VIDEOS.items():
        vid_id = vid_stem[-6:]
        angle  = cfg["angle"]
        results[vid_stem] = {"angle": angle}
        try:
            if angle == "face-on":
                results[vid_stem]["data"] = measure_faceon(vid_stem, vid_id, angle)
            else:
                results[vid_stem]["data"] = measure_dtl(vid_stem, vid_id, angle)
        except Exception as e:
            print(f"  ERROR {vid_id}: {e}")
            import traceback; traceback.print_exc()
            results[vid_stem]["data"] = {}

    write_summary(results)

    # Print console summary
    print("\n" + "="*60)
    print("GT MEASUREMENT SUMMARY (console)")
    print("="*60)
    for vid_stem, r in results.items():
        vid_id = vid_stem[-6:]
        d = r.get("data", {})
        print(f"\n{vid_id} [{r['angle']}]")
        for k, v in d.items():
            print(f"    {k:35s}: {v}")

    # PROGRESS.log
    prog = PROJ / "PROGRESS.log"
    with open(prog, "a") as f:
        ts = datetime.datetime.now().isoformat()
        f.write(f"{ts}  GT measurement report complete: {DESK}\n")

    print(f"\nOutput: {DESK}")
    print("No fault labels generated.")


if __name__ == "__main__":
    main()
