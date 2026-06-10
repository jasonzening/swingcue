#!/usr/bin/env python3
"""
detect_keyframes.py  v3
=======================
SwingPhaseDetector — reusable golf swing keyframe detector.
Pure signal processing, no ML.

Usage (CLI):
    python detect_keyframes.py <keypoints.json> [--view {faceon,downtheline,auto}]
                               [--plot] [--verify VIDEO] [--out DIR]

Usage (library):
    from detect_keyframes import SwingPhaseDetector
    det = SwingPhaseDetector()
    result = det.detect("path/to/keypoints.json")
    # result["keyframes"]   → {"address": N, "top": N, "impact": N, "finish": N}
    # result["confidence"]  → {"address": 0.0-1.0, ...}
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter, find_peaks


# ─────────────────────────── signal helpers ──────────────────────────────────

def _make_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def _sg(arr: np.ndarray, window: int, poly: int = 3) -> np.ndarray:
    w = max(_make_odd(min(window, len(arr) - 1)), poly + 2)
    return savgol_filter(arr, w, poly)


def _interp_nan(arr: np.ndarray) -> np.ndarray:
    arr = arr.copy()
    nans = np.isnan(arr)
    if nans.all():
        return arr
    idx = np.arange(len(arr))
    arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
    return arr


def _load_wrist_track(d_frames: list, score_thr: float = 0.4):
    """Extract weighted-average wrist midpoint (x, y) per frame."""
    n = len(d_frames)
    xs = np.full(n, np.nan)
    ys = np.full(n, np.nan)
    for fd in d_frames:
        fi = fd["frame"]
        if not fd["persons"]:
            continue
        kps = fd["persons"][0]["keypoints"]
        lw, rw = kps["left_wrist"], kps["right_wrist"]
        lsc, rsc = lw["score"], rw["score"]
        if max(lsc, rsc) < score_thr:
            continue
        w = lsc + rsc
        xs[fi] = (lw["x"] * lsc + rw["x"] * rsc) / w
        ys[fi] = (lw["y"] * lsc + rw["y"] * rsc) / w
    return _interp_nan(xs), _interp_nan(ys)


def _torso_height(d_frames: list, frame_idx: int) -> float:
    """|shoulder_y - hip_y| at a given frame — body-scale reference."""
    try:
        kps = d_frames[frame_idx]["persons"][0]["keypoints"]
        sh_y  = (kps["left_shoulder"]["y"]  + kps["right_shoulder"]["y"])  / 2
        hip_y = (kps["left_hip"]["y"]       + kps["right_hip"]["y"])       / 2
        return abs(hip_y - sh_y)
    except Exception:
        return 0.0


# ─────────────────────────── SwingPhaseDetector ──────────────────────────────

class SwingPhaseDetector:
    """
    Detects four golf swing keyframes from RTMPose keypoint JSON.

    Parameters
    ----------
    sg_window_ms : int
        Savitzky-Golay smoothing window in milliseconds (default 200).
    static_percentile : float
        Speed percentile used to compute the "static" threshold (default 20).
    static_multiplier : float
        Threshold = percentile_speed × static_multiplier (default 3.0).
    top_prominence_px : float
        Minimum prominence (px) for a wrist-y local minimum to count as TOP (default 30).
    impact_window_s : float
        Search window around speed peak for impact: ±impact_window_s seconds (default 0.4).
        Covers both face-on (impact slightly before peak) and down-the-line
        (impact slightly after peak).
    impact_sanity_frac : float
        Max allowed wrist Y above address as fraction of torso height (default 0.35).
    finish_settle_s : float
        Minimum duration of low-speed window to declare finish (default 0.35 s).
    """

    def __init__(
        self,
        sg_window_ms: int   = 200,
        static_percentile: float = 20,
        static_multiplier: float = 3.0,
        top_prominence_px: float = 30,
        impact_window_s: float   = 0.40,
        impact_sanity_frac: float = 0.35,
        finish_settle_s: float   = 0.35,
    ):
        self.sg_window_ms      = sg_window_ms
        self.static_percentile = static_percentile
        self.static_multiplier = static_multiplier
        self.top_prominence_px = top_prominence_px
        self.impact_window_s   = impact_window_s
        self.impact_sanity_frac = impact_sanity_frac
        self.finish_settle_s   = finish_settle_s

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self, json_path: str) -> dict:
        """
        Run detection on a keypoint JSON file.

        Returns
        -------
        dict with keys:
            keyframes   : {address, top, impact, finish}  (frame indices)
            confidence  : {address, top, impact, finish}  (0.0–1.0)
            signals     : raw/smoothed arrays for plotting
            meta        : fps, n_frames, impact_dist_px, torso_h, etc.
        """
        with open(json_path) as f:
            data = json.load(f)
        d_frames = data["frames"]
        fps = data["stats"].get("source_fps", 30.0)
        return self._run(d_frames, fps)

    def detect_from_dict(self, data: dict) -> dict:
        """Same as detect() but accepts an already-loaded JSON dict."""
        return self._run(data["frames"], data["stats"].get("source_fps", 30.0))

    # ── internals ─────────────────────────────────────────────────────────────

    def _run(self, d_frames: list, fps: float) -> dict:
        n = len(d_frames)
        xs_raw, ys_raw = _load_wrist_track(d_frames)

        # Smooth
        win = _make_odd(max(7, int(fps * self.sg_window_ms / 1000)))
        xs_s = _sg(xs_raw, win)
        ys_s = _sg(ys_raw, win)

        # Speed
        dx = np.diff(xs_s, prepend=xs_s[0])
        dy = np.diff(ys_s, prepend=ys_s[0])
        spd_raw = np.sqrt(dx**2 + dy**2)
        spd_s   = _sg(spd_raw, win)

        # Vertical velocity (vy > 0 → wrist moving DOWN in image)
        vy_s = _sg(np.gradient(ys_s), win)

        # ── ADDRESS ──────────────────────────────────────────────────────────
        static_thr = max(
            np.percentile(spd_s, self.static_percentile) * self.static_multiplier,
            1.0
        )
        search_end = int(n * 0.50)   # look in first 50 %
        address = 2
        for i in range(2, search_end):
            if spd_s[i] < static_thr:
                address = i
            else:
                if spd_s[i] > static_thr * 3 and i > address + 5:
                    break

        # ── TOP ──────────────────────────────────────────────────────────────
        top_end = int(n * 0.82)   # 82%: leaves enough right-base for prominence
        ys_region = ys_s[address:top_end]
        peaks_local, peak_props = find_peaks(
            -ys_region,
            prominence=self.top_prominence_px,
            distance=int(fps * 0.25),
        )
        if len(peaks_local) == 0:
            # Fallback: global minimum; estimate prominence from region edges
            top = address + int(np.argmin(ys_region))
            left_h  = ys_region[0]  - ys_region.min()
            right_h = ys_region[-1] - ys_region.min()
            top_prominence = float(min(left_h, right_h))
        else:
            top = address + peaks_local[0]
            top_prominence = float(peak_props["prominences"][0])

        # ── IMPACT (three-condition rule, view-agnostic) ──────────────────────
        #
        # (1) Find speed peak in downswing [top → 80% of video].
        # (2) Search window: [spd_peak - impact_window_s, spd_peak + impact_window_s]
        #     This covers:
        #       - down-the-line: impact slightly AFTER speed peak (hands return to ball)
        #       - face-on: impact slightly BEFORE speed peak (hands sweep through ball)
        # (3) Within that window, pick the frame whose (x,y) is closest to address anchor.
        # (4) Sanity: wrist must not be more than sanity_frac × torso_h above address.
        #
        addr_x = xs_s[address]
        addr_y = ys_s[address]
        torso_h = _torso_height(d_frames, address)

        down_end = int(n * 0.80)
        down_spd = spd_s[top:down_end]
        if len(down_spd) == 0:
            spd_peak = top + 1
        else:
            pk_local, pk_props = find_peaks(
                down_spd, height=np.percentile(spd_s, 40)
            )
            if len(pk_local) == 0:
                spd_peak = top + int(np.argmax(down_spd))
            else:
                spd_peak = top + pk_local[np.argmax(pk_props["peak_heights"])]

        half_win = max(int(fps * self.impact_window_s), 4)
        i_start = max(top + 1, spd_peak - half_win)
        i_end   = min(n,       spd_peak + half_win + 1)

        dists = np.sqrt(
            (xs_s[i_start:i_end] - addr_x) ** 2 +
            (ys_s[i_start:i_end] - addr_y) ** 2
        )
        impact = i_start + int(np.argmin(dists))
        impact_dist = float(dists.min())

        # Sanity check
        y_diff = addr_y - ys_s[impact]      # >0 → impact wrist above address
        impact_ok = not (torso_h > 0 and y_diff > torso_h * self.impact_sanity_frac)

        # ── FINISH ───────────────────────────────────────────────────────────
        # Find followthrough high point (second ys minimum after impact)
        ft_search_len = min(int(fps * 2.0), n - impact - 1)
        ft_region = ys_s[impact: impact + ft_search_len]
        ft_peaks_local, _ = find_peaks(
            -ft_region, prominence=15, distance=int(fps * 0.15)
        )
        ft_top = (impact + ft_peaks_local[0]) if len(ft_peaks_local) > 0 \
                 else (impact + int(fps * 0.3))

        settle_thr = static_thr * 1.5
        settle_win = max(int(fps * self.finish_settle_s), 4)
        finish = min(n - 1, ft_top + settle_win)
        found_settle = False
        for i in range(ft_top + 1, n - settle_win):
            if np.all(spd_s[i: i + settle_win] < settle_thr):
                finish = i + settle_win // 2
                found_settle = True
                break
        if not found_settle:
            # Fallback: frame of minimum speed in final third of video
            last_third = spd_s[ft_top:]
            finish = ft_top + int(np.argmin(last_third))

        # ── CONFIDENCE SCORES ─────────────────────────────────────────────────
        #
        # address  : 1 - clamp(speed_at_address / (2 * static_thr), 0, 1)
        # top      : clamp(top_prominence / 150, 0, 1)
        # impact   : max(0, 1 - dist_to_addr / (torso_h * 1.5)) * sanity_flag
        # finish   : 1 - clamp(mean_speed_in_settle / settle_thr, 0, 1)
        #
        ref = torso_h if torso_h > 0 else 150.0

        conf_address = float(np.clip(
            1.0 - spd_s[address] / (2.0 * static_thr), 0, 1
        ))
        conf_top = float(np.clip(top_prominence / 150.0, 0, 1))
        conf_impact = float(np.clip(
            1.0 - impact_dist / (ref * 1.5), 0, 1
        )) * (1.0 if impact_ok else 0.3)
        settle_slice = spd_s[finish - settle_win // 2: finish + settle_win // 2 + 1]
        conf_finish = float(np.clip(
            1.0 - settle_slice.mean() / settle_thr, 0, 1
        )) if len(settle_slice) > 0 else 0.5
        if not found_settle:
            conf_finish *= 0.5   # penalise fallback

        keyframes = {
            "address": int(address),
            "top":     int(top),
            "impact":  int(impact),
            "finish":  int(finish),
        }
        confidence = {
            "address": round(conf_address, 3),
            "top":     round(conf_top, 3),
            "impact":  round(conf_impact, 3),
            "finish":  round(conf_finish, 3),
        }
        signals = {
            "xs": xs_raw, "ys": ys_raw,
            "xs_s": xs_s, "ys_s": ys_s,
            "spd_raw": spd_raw, "spd_s": spd_s,
            "vy_s": vy_s,
            "static_thr": float(static_thr),
            "settle_thr": float(settle_thr),
            "spd_peak": int(spd_peak),
            "top_prominence": float(top_prominence),
            "ft_top": int(ft_top),
            "addr_x": float(addr_x), "addr_y": float(addr_y),
            "torso_h": float(torso_h),
            "impact_dist": float(impact_dist),
            "impact_ok": bool(impact_ok),
        }
        return {
            "keyframes":   keyframes,
            "confidence":  confidence,
            "signals":     signals,
            "fps":         fps,
            "n_frames":    n,
        }


# ─────────────────────────── plot ─────────────────────────────────────────────

def make_plot(frames: np.ndarray, result: dict, out_path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sig = result["signals"]
    kf  = result["keyframes"]
    cf  = result["confidence"]

    COLORS = {
        "address": "#27ae60",
        "top":     "#e74c3c",
        "impact":  "#e67e22",
        "finish":  "#2980b9",
    }

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(f"SwingPhaseDetector — {title}", fontsize=13, fontweight="bold")

    def vlines(ax, ylo, yhi):
        for kname, kf_idx in kf.items():
            c = COLORS[kname]
            ax.axvline(kf_idx, color=c, lw=2, ls="--", alpha=0.9)
            ax.text(kf_idx + 0.5, ylo + (yhi - ylo) * 0.05,
                    f"{kname.upper()}\n[{kf_idx}]\nconf={cf[kname]:.2f}",
                    color=c, fontsize=7.5, va="bottom", rotation=90,
                    fontweight="bold", linespacing=1.3)

    # Panel 1 — Wrist height (y)
    ax0 = axes[0]
    ax0.plot(frames, sig["ys"],   alpha=0.30, color="#bdc3c7", lw=1, label="raw y")
    ax0.plot(frames, sig["ys_s"], color="#2c3e50", lw=2, label="smoothed y")
    ax0.scatter([kf["address"]], [sig["ys_s"][kf["address"]]],
                zorder=5, s=60, color=COLORS["address"])
    ax0.invert_yaxis()
    ax0.set_ylabel("Wrist Y (px)", fontsize=10)
    ax0.set_title("Wrist Height  [inverted: higher on chart = higher in frame]",
                  fontsize=9)
    ax0.legend(loc="lower right", fontsize=8)
    ylo0, yhi0 = ax0.get_ylim()
    vlines(ax0, yhi0, ylo0)   # inverted axis

    # Panel 2 — Speed
    ax1 = axes[1]
    ax1.plot(frames, sig["spd_raw"], alpha=0.30, color="#bdc3c7", lw=1, label="raw speed")
    ax1.plot(frames, sig["spd_s"],   color="#8e44ad", lw=2, label="smoothed speed")
    ax1.axvline(sig["spd_peak"], color="#f39c12", lw=1.5, ls="-.",
                alpha=0.9, label=f"spd peak (fr{sig['spd_peak']})")
    ax1.axhline(sig["static_thr"], color="#95a5a6", lw=1, ls=":",
                alpha=0.8, label=f"static thr={sig['static_thr']:.1f}")
    ax1.axhline(sig["settle_thr"], color="#7f8c8d", lw=1, ls=":",
                alpha=0.6, label=f"settle thr={sig['settle_thr']:.1f}")
    ax1.set_ylabel("Speed (px/frame)", fontsize=10)
    ax1.set_title(f"Wrist Speed  |  impact dist_to_addr={sig['impact_dist']:.0f}px  "
                  f"torso_h={sig['torso_h']:.0f}px  sanity={'OK' if sig['impact_ok'] else 'FAIL'}",
                  fontsize=9)
    ax1.legend(loc="upper left", fontsize=7.5, ncol=2)
    ylo1, yhi1 = ax1.get_ylim()
    vlines(ax1, ylo1, yhi1)

    # Panel 3 — Vertical velocity
    ax2 = axes[2]
    ax2.plot(frames, sig["vy_s"], color="#16a085", lw=2, label="vy (smoothed)")
    ax2.axhline(0, color="#7f8c8d", lw=1, alpha=0.5)
    ax2.fill_between(frames, sig["vy_s"], 0,
                     where=(sig["vy_s"] > 0), alpha=0.15,
                     color="#e74c3c", label="downswing (vy>0)")
    ax2.fill_between(frames, sig["vy_s"], 0,
                     where=(sig["vy_s"] < 0), alpha=0.15,
                     color="#3498db", label="back/followthrough (vy<0)")
    ax2.set_ylabel("Vert. vel. (px/frame)", fontsize=10)
    ax2.set_xlabel("Frame", fontsize=10)
    ax2.set_title("Vertical velocity  [>0 = wrist moving DOWN; <0 = UP]", fontsize=9)
    ax2.legend(loc="upper left", fontsize=8, ncol=2)
    ylo2, yhi2 = ax2.get_ylim()
    vlines(ax2, ylo2, yhi2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {out_path}")


# ─────────────────────────── verify video ─────────────────────────────────────

def make_verify_video(video_path: str, result: dict, out_path: str):
    import cv2

    kf  = result["keyframes"]
    cf  = result["confidence"]
    sig = result["signals"]
    fps = result["fps"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: cannot open {video_path}")
        return
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps

    freeze_n = int(src_fps * 1.0)   # 1 s

    all_frames = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        all_frames.append(fr)
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, src_fps, (w, h))

    COLORS_BGR = {
        "address": (50,  180,  50),
        "top":     (40,   60, 220),
        "impact":  (20,  130, 230),
        "finish":  (210, 110,  40),
    }
    LABELS = {
        "address": "ADDRESS",
        "top":     "TOP",
        "impact":  "IMPACT",
        "finish":  "FINISH",
    }

    freeze_at = {}
    for kname, kf_idx in kf.items():
        clamped = max(0, min(kf_idx, len(all_frames) - 1))
        freeze_at[clamped] = kname

    font = cv2.FONT_HERSHEY_DUPLEX
    fs   = h / 280.0
    banner_h = int(h * 0.22)

    total_written = 0
    for i, raw_frame in enumerate(all_frames):
        frame = raw_frame.copy()

        if i in freeze_at:
            kname = freeze_at[i]
            label = LABELS[kname]
            color = COLORS_BGR[kname]

            # Dark banner
            ov = frame.copy()
            cv2.rectangle(ov, (0, h - banner_h), (w, h), (15, 15, 15), -1)
            cv2.addWeighted(ov, 0.70, frame, 0.30, 0, frame)

            # Main label
            tw, th = cv2.getTextSize(label, font, fs * 2.0, 5)[0]
            tx, ty = (w - tw) // 2, h - banner_h // 2 + th // 2
            cv2.putText(frame, label, (tx + 3, ty + 3), font,
                        fs * 2.0, (0, 0, 0), 7, cv2.LINE_AA)
            cv2.putText(frame, label, (tx, ty),         font,
                        fs * 2.0, color,   5, cv2.LINE_AA)

            # Top-left info lines
            info_lines = [f"Frame {i}   conf={cf[kname]:.2f}"]
            if kname == "impact":
                info_lines.append(
                    f"dist_to_addr={sig['impact_dist']:.0f}px  "
                    f"sanity={'OK' if sig['impact_ok'] else 'FAIL'}"
                )
            for li, line in enumerate(info_lines):
                cv2.putText(frame, line, (12, 44 + li * 38), font,
                            fs * 0.80, (220, 220, 220), 2, cv2.LINE_AA)

            for _ in range(freeze_n):
                out.write(frame)
            total_written += freeze_n
        else:
            out.write(frame)
            total_written += 1

    out.release()
    dur = total_written / src_fps
    print(f"  Verify video: {out_path}  ({total_written} frames, {dur:.1f}s)")


# ─────────────────────────── CLI main ─────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="SwingPhaseDetector — golf keyframe detector")
    ap.add_argument("json",  help="RTMPose keypoint JSON")
    ap.add_argument("--plot",   action="store_true")
    ap.add_argument("--verify", metavar="VIDEO")
    ap.add_argument("--out",    default="keyframes")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.json).stem.replace("_keypoints", "")

    print(f"\nSwingPhaseDetector v3")
    print(f"  JSON  : {args.json}")
    print(f"  Output: {out_dir}/")

    det    = SwingPhaseDetector()
    result = det.detect(args.json)

    kf  = result["keyframes"]
    cf  = result["confidence"]
    sig = result["signals"]
    fps = result["fps"]
    n   = result["n_frames"]

    print(f"  Frames: {n}  FPS: {fps:.1f}")
    print()
    print(f"  {'Key':10s} | {'Frame':>5} | {'t(ms)':>7} | {'wrist_y':>8} | {'speed':>6} | {'conf':>5}")
    print(f"  {'-'*10}-+-{'-'*5}-+-{'-'*7}-+-{'-'*8}-+-{'-'*6}-+-{'-'*5}")
    for k in ("address", "top", "impact", "finish"):
        v = kf[k]
        print(f"  {k:10s} | {v:5d} | {v/fps*1000:7.0f} | "
              f"{sig['ys_s'][v]:8.1f} | {sig['spd_s'][v]:6.1f} | {cf[k]:5.3f}")

    if kf["impact"] is not None:
        print()
        print(f"  Impact details:")
        print(f"    spd_peak       : frame {sig['spd_peak']}")
        print(f"    dist_to_addr   : {sig['impact_dist']:.1f} px")
        print(f"    torso_h        : {sig['torso_h']:.1f} px")
        print(f"    sanity         : {'OK' if sig['impact_ok'] else 'FAIL — wrist too high'}")

    # Save JSON
    out_json = out_dir / f"{stem}_keyframes.json"
    with open(out_json, "w") as f:
        json.dump({"keyframes": kf, "confidence": cf,
                   "fps": fps, "n_frames": n}, f, indent=2)
    print(f"\n  Saved: {out_json}")

    # Plot
    if args.plot:
        frames = np.arange(n)
        out_plot = out_dir / f"{stem}_speed_curve.png"
        make_plot(frames, result, str(out_plot), stem)

    # Verify video
    if args.verify:
        out_vid = out_dir / f"{stem}_verify.mp4"
        make_verify_video(args.verify, result, str(out_vid))

    print("\nResult JSON:")
    print(json.dumps({"keyframes": kf, "confidence": cf}, indent=2))


if __name__ == "__main__":
    main()
