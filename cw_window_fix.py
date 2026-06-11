#!/usr/bin/env python3
"""
cw_window_fix.py
================
Chicken Wing window fix per FAULT_VISUAL_STANDARDS v0.2:
  Window = [impact, cutoff_frame]
  cutoff_frame = first frame after impact where wrist_mid_y < hip_mid_y
  (image coords: y↓, so wrist ABOVE hip line = wrist_y < hip_y = hands above hip = exit window)

Outputs:
  - Console: per-video corrected elbow min (fr + deg)
  - gt_measure/peak_frames_v2/: annotated frames for corrected window
  - 201015: fr59-fr63 ALL frames rendered
  - 201047: confirm fr282 in/out of corrected window
"""

import json, math, sys
from pathlib import Path
from typing import Optional
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine.a_measurement.pose_pipeline import PosePipeline
from engine.b_phase.swing_phase import SwingPhaseEngine

PROJ    = Path("/home/jason/projects/swingcue-postest")
INPUT   = PROJ / "input"
KP_DIR  = PROJ / "engine/kp_cache"
OUT_DIR = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/gt_measure/peak_frames_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_HEAD_V = (200, 0, 200)
C_HEAD_H = (0, 140, 255)
C_FOREARM= (0, 200, 60)
C_WHITE  = (255, 255, 255)
C_BLACK  = (0, 0, 0)
LINE_W   = 3
FONT     = cv2.FONT_HERSHEY_DUPLEX

FACEON_VIDEOS = {
    "Videos2026-06-09_201015_827": {"angle": "face-on", "gt_impact": 59},
    "Videos2026-06-09_201039_231": {"angle": "face-on", "gt_impact": 208},
    "Videos2026-06-09_201047_915": {"angle": "face-on", "gt_impact": 282},
}


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
    cos = max(-1.0, min(1.0, dot/(l1*l2)))
    return math.degrees(math.acos(cos))


def head_center(kps):
    pts = []
    for k in ("nose","left_eye","right_eye","left_ear","right_ear"):
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


def draw_vline(img, x, color, label=""):
    h = img.shape[0]
    cv2.line(img, (int(x),0), (int(x),h), color, LINE_W, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (int(x)+4, 40), FONT, 0.45, color, 1, cv2.LINE_AA)


def draw_hline(img, y, color, label=""):
    w = img.shape[1]
    cv2.line(img, (0,int(y)), (w,int(y)), color, LINE_W, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (8, int(y)-6), FONT, 0.45, color, 1, cv2.LINE_AA)


def draw_forearm_chain(img, sh, el, wr):
    sh = (int(sh[0]),int(sh[1])); el=(int(el[0]),int(el[1])); wr=(int(wr[0]),int(wr[1]))
    cv2.line(img, sh, el, C_FOREARM, LINE_W, cv2.LINE_AA)
    cv2.line(img, el, wr, C_FOREARM, LINE_W, cv2.LINE_AA)
    for p in (sh, el, wr):
        cv2.circle(img, p, 5, C_FOREARM, -1, cv2.LINE_AA)
    ang = angle_3pt(sh, el, wr)
    cv2.putText(img, f"{ang:.0f}deg", (el[0]+8, el[1]-8), FONT, 0.50, C_FOREARM, 1, cv2.LINE_AA)


def dynamic_cw_window(kp_json, impact_fr, max_search=20):
    """
    Returns (start, cutoff) where cutoff = first frame after impact
    where wrist_mid_y < hip_mid_y (hands above hip line in image coords).
    If no such frame found within max_search, cutoff = impact + max_search.
    """
    n = len(kp_json["frames"])
    cutoff = min(impact_fr + max_search, n - 1)

    for fr in range(impact_fr + 1, min(impact_fr + max_search + 1, n)):
        fd = kp_json["frames"][fr]
        if not fd["persons"]:
            continue
        kps = fd["persons"][0]["keypoints"]
        # wrist mid
        lw = kp_pt(kps, "left_wrist"); rw = kp_pt(kps, "right_wrist")
        wm = mid_pt(lw, rw) or (lw or rw)
        # hip mid
        lh = kp_pt(kps, "left_hip"); rh = kp_pt(kps, "right_hip")
        hm = mid_pt(lh, rh) or (lh or rh)
        if wm is None or hm is None:
            continue
        # y↓: wrist above hip = wrist_y < hip_y
        if wm[1] < hm[1]:
            cutoff = fr
            break

    return impact_fr, cutoff


def render_frame(vid_path, vid_id, fr, kp_json, addr_hx, addr_hy, phase_map,
                 extra="", in_window=True):
    raw = get_frame(vid_path, fr)
    img = raw.copy()
    phase = phase_map.get(fr, "?")

    # Fixed address lines
    draw_vline(img, addr_hx, C_HEAD_V, "HEAD-V")
    draw_hline(img, addr_hy, C_HEAD_H, "HEAD-H")

    # Per-frame forearm chain
    if fr < len(kp_json["frames"]):
        fd = kp_json["frames"][fr]
        if fd["persons"]:
            kps = fd["persons"][0]["keypoints"]
            sh = kp_pt(kps, "left_shoulder"); el = kp_pt(kps, "left_elbow"); wr = kp_pt(kps, "left_wrist")
            if sh and el and wr:
                draw_forearm_chain(img, sh, el, wr)

            # Head dot
            hc = head_center(kps)
            if hc:
                cv2.circle(img, (int(hc[0]),int(hc[1])), 6, C_HEAD_V, -1, cv2.LINE_AA)

            # Hip line (horizontal, current frame) for window boundary check
            lh = kp_pt(kps, "left_hip"); rh = kp_pt(kps, "right_hip")
            hm = mid_pt(lh, rh)
            if hm:
                # Draw current hip height as dashed reference
                w_img = img.shape[1]
                for x in range(0, w_img, 14):
                    cv2.line(img, (x, int(hm[1])), (min(x+8, w_img), int(hm[1])),
                             (0, 180, 180), 1, cv2.LINE_AA)
                cv2.putText(img, "hip", (8, int(hm[1])-4), FONT, 0.38, (0,180,180), 1, cv2.LINE_AA)

    win_tag = "IN-WIN" if in_window else "OUT-WIN"
    label_frame(img, vid_id, fr, phase, f"{extra} {win_tag}")
    return img


def process_video(vid_stem, cfg):
    vid_id   = vid_stem[-6:]
    vid_path = str(INPUT / f"{vid_stem}.mp4")
    angle    = cfg["angle"]

    kp_path = KP_DIR / f"{vid_stem}.json"
    with open(kp_path) as f:
        kp_json = json.load(f)

    pipe = PosePipeline(device="cpu")
    meas, fps = pipe.run_from_json(kp_json)
    eng = SwingPhaseEngine()
    ann, anchors = eng.run(meas, fps, angle=angle)
    phase_map = {a.frame_idx: a.phase for a in ann}
    n = len(kp_json["frames"])

    addr_fr   = anchors.address
    impact_fr = anchors.impact

    # Address head anchors
    fd0 = kp_json["frames"][addr_fr]
    kps0 = fd0["persons"][0]["keypoints"] if fd0["persons"] else {}
    hc0 = head_center(kps0)
    if hc0 is None:
        print(f"  {vid_id}: WARNING no head kp at address")
        addr_hx, addr_hy = 360, 200
    else:
        addr_hx, addr_hy = hc0

    # Dynamic window
    win_start, win_cutoff = dynamic_cw_window(kp_json, impact_fr)
    print(f"  {vid_id}: impact=fr{impact_fr}  dynamic_window=[fr{win_start}, fr{win_cutoff}]")

    # Elbow angles in window
    elbow_data = []  # (fr, angle_deg)
    for fr in range(win_start, win_cutoff + 1):
        if fr >= n: break
        fd = kp_json["frames"][fr]
        if not fd["persons"]:
            elbow_data.append((fr, float('nan')))
            continue
        kps = fd["persons"][0]["keypoints"]
        sh = kp_pt(kps, "left_shoulder"); el = kp_pt(kps, "left_elbow"); wr = kp_pt(kps, "left_wrist")
        if sh and el and wr:
            elbow_data.append((fr, angle_3pt(sh, el, wr)))
        else:
            elbow_data.append((fr, float('nan')))

    valid_elbows = [(fr, ang) for fr, ang in elbow_data if not math.isnan(ang)]
    if valid_elbows:
        min_fr, min_deg = min(valid_elbows, key=lambda x: x[1])
    else:
        min_fr, min_deg = impact_fr, float('nan')

    print(f"  {vid_id}: elbow_min=fr{min_fr} {min_deg:.1f}deg  "
          f"(prev fixed+8 window was fr{impact_fr}-fr{impact_fr+8})")

    # Per-frame table
    print(f"  {vid_id}: frame-by-frame elbow angles in corrected window:")
    for fr, ang in elbow_data:
        in_w = win_start <= fr <= win_cutoff
        ang_str = f"{ang:.1f}deg" if not math.isnan(ang) else "nan"
        phase = phase_map.get(fr, "?")
        marker = " <-- MIN" if fr == min_fr else ""
        print(f"    fr{fr:03d} [{phase:15s}] elbow={ang_str}{marker}")

    # Render frames
    out_sub = OUT_DIR / vid_id
    out_sub.mkdir(parents=True, exist_ok=True)

    # For 201015: render ALL frames fr59-fr63 regardless of window
    special_range = None
    if "201015" in vid_stem:
        special_range = range(59, min(64, n))
        print(f"  {vid_id}: rendering special range fr59-fr63")

    frames_to_render = set(range(win_start, win_cutoff + 1))
    if special_range:
        frames_to_render.update(special_range)
    # Always render min frame
    frames_to_render.add(min_fr)

    for fr in sorted(frames_to_render):
        if fr >= n: continue
        in_w = win_start <= fr <= win_cutoff
        # get elbow angle for this frame
        ang_here = next((a for f, a in elbow_data if f == fr), float('nan'))
        ang_str = f"elbow={ang_here:.0f}deg" if not math.isnan(ang_here) else "elbow=nan"
        if fr == min_fr:
            ang_str += "_MIN"
        img = render_frame(vid_path, vid_id, fr, kp_json, addr_hx, addr_hy,
                           phase_map, extra=ang_str, in_window=in_w)
        fname = f"fr{fr:03d}_{ang_str}.jpg"
        cv2.imwrite(str(out_sub / fname), img, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Special check for 201047 fr282
    if "201047" in vid_stem:
        is_in = win_start <= 282 <= win_cutoff
        print(f"  {vid_id}: fr282 in corrected window? {is_in} "
              f"(window=[fr{win_start}, fr{win_cutoff}])")

    rendered = len(list(out_sub.glob("*.jpg")))
    print(f"  {vid_id}: {rendered} frames -> {out_sub}")

    return {
        "vid_id": vid_id,
        "impact_fr": impact_fr,
        "win_start": win_start,
        "win_cutoff": win_cutoff,
        "elbow_min_fr": min_fr,
        "elbow_min_deg": round(min_deg, 1) if not math.isnan(min_deg) else "nan",
        "elbow_table": elbow_data,
    }


def main():
    import datetime
    print(f"cw_window_fix.py  {datetime.datetime.now().isoformat()}")
    results = {}
    for vid_stem, cfg in FACEON_VIDEOS.items():
        r = process_video(vid_stem, cfg)
        results[r["vid_id"]] = r

    print("\n=== CORRECTED CHICKEN WING WINDOW SUMMARY ===")
    print(f"{'Video':10s}  {'impact':>8}  {'win_end':>8}  {'elbow_min_fr':>13}  {'elbow_min_deg':>14}")
    print("-"*60)
    for vid_id, r in results.items():
        print(f"{vid_id:10s}  {r['impact_fr']:>8}  {r['win_cutoff']:>8}  "
              f"{r['elbow_min_fr']:>13}  {r['elbow_min_deg']:>14}")

    # Update PROGRESS.log
    prog = Path("/home/jason/projects/swingcue-postest/PROGRESS.log")
    with open(prog, "a") as f:
        ts = datetime.datetime.now().isoformat()
        f.write(f"{ts}  CW window fix complete: corrected elbow measurements in peak_frames_v2/\n")

    return results


if __name__ == "__main__":
    main()
