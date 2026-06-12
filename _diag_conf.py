#!/usr/bin/env python3
"""Conf diagnostic - trace each factor"""
import sys, json, numpy as np
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks

sys.path.insert(0, "/home/jason/projects/swingcue-postest")
from engine.a_measurement.pose_pipeline import PosePipeline
from engine.b_phase.swing_phase import SwingPhaseEngine

VIDEOS = [
    ("Videos2026-06-09_201015_827.json", "face-on"),
    ("Videos2026-06-09_201039_231.json", "face-on"),
    ("Videos2026-06-09_201047_915.json", "face-on"),
    ("Videos2026-06-09_201054_561.json", "down-the-line"),
    ("Videos2026-06-09_201058_697.json", "down-the-line"),
]
KP_CACHE = Path("/home/jason/projects/swingcue-postest/engine/kp_cache")

for fname, angle in VIDEOS:
    with open(KP_CACHE / fname) as f:
        kp_json = json.load(f)
    pipeline = PosePipeline(device="cpu")
    measurements, fps = pipeline.run_from_json(kp_json)
    n = len(measurements)
    stem = fname[18:24]

    engine = SwingPhaseEngine()
    annotations, anchors = engine.run(measurements, fps, angle=angle)

    # Reproduce internal computations
    xs = np.full(n, np.nan); ys = np.full(n, np.nan)
    for m in measurements:
        wm = m.wrist_mid()
        if wm: xs[m.frame_idx], ys[m.frame_idx] = wm
    idx = np.arange(n)
    for arr in (xs, ys):
        nans = np.isnan(arr)
        if not nans.all():
            arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
    w = max(7, int(fps * 200 / 1000)) | 1
    xs_s = savgol_filter(xs, w, 3)
    ys_s = savgol_filter(ys, w, 3)
    dx = np.diff(xs_s, prepend=xs_s[0]); dy = np.diff(ys_s, prepend=ys_s[0])
    spd = savgol_filter(np.sqrt(dx**2 + dy**2), w, 3)

    impact = anchors.impact
    n_eff = anchors.first_swing_end if anchors.first_swing_end < n else n

    signal = xs_s if angle == "down-the-line" else ys_s
    ys_range = float(ys_s[:n_eff].max() - ys_s[:n_eff].min())
    ys_range = max(ys_range, 30.0)

    win = max(int(fps * 0.5), 5)
    lo_w = max(0, impact - win); hi_w = min(n_eff, impact + win + 1)
    region = signal[lo_w:hi_w]
    peak_prom = float(signal[impact] - region.min()) if len(region) > 0 else 0.0
    norm_base = max(ys_range * 0.25, 30.0)
    sig_score = float(np.clip(peak_prom / norm_base, 0.0, 1.0))
    ambiguity = float(np.clip(1.0 - 0.25 * (anchors.swing_count - 1), 0.20, 1.0))

    lo = max(0, impact - 3); hi = min(n, impact + 4)
    qscores = []
    for m in measurements[lo:hi]:
        q = 1.0 if m.measurement_quality == "ok" else 0.5 if m.measurement_quality == "degraded" else 0.1
        qscores.append(q)
    quality = float(np.mean(qscores)) if qscores else 0.5

    raw_conf = sig_score * 0.50 + ambiguity * 0.30 + quality * 0.20

    # Top conf
    address = anchors.address; top = anchors.top
    top_end = int(n_eff * 0.82)
    ys_region = ys_s[address:top_end]
    peaks_l, pp = find_peaks(-ys_region, prominence=30, distance=int(fps * 0.25))
    if len(peaks_l) == 0:
        left_h  = ys_region[0]  - ys_region.min()
        right_h = ys_region[-1] - ys_region.min()
        top_prom = float(min(left_h, right_h))
    else:
        top_prom = float(pp["prominences"][0])
    top_conf_raw = top_prom / (ys_range * 0.70)

    print(f"{stem} [{angle:12s}] sc={anchors.swing_count}")
    print(f"  top: top_prom={top_prom:.1f} ys_range={ys_range:.1f} ratio={top_conf_raw:.3f} → tc={min(1.0,top_conf_raw):.3f}")
    print(f"  imp: peak_prom={peak_prom:.1f} norm_base={norm_base:.1f} sig={sig_score:.3f} ambig={ambiguity:.3f} qual={quality:.3f} → ic={raw_conf:.3f}")
    print()
