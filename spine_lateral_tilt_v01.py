#!/usr/bin/env python3
"""
spine_lateral_tilt_v01.py — spine_lateral_tilt feature v0.1

Feature definition:
  spine_lateral_tilt[t] = angle between torso axis (shoulder_mid→hip_mid) and
  image vertical, in degrees.
  Sign convention: toward target_side = positive, away = negative.
  Shoulder coords normalized by shoulder_width before angle computation.

Inputs: 4 face-on clips from teach_pipeline (dtl-4/left,right + dlt-6/left,right)
Outputs:
  1. Per-clip top-frame tilt value + address baseline
  2. Address→top tilt trajectory curve PNGs
  3. Top-frame render with torso axis + vertical ref + tilt annotation
  4. Paired diff (wrong - correct) per pair

No diagnostic labels. Correct/wrong from split_check_results markers only.
"""

import sys, json, math, os
from pathlib import Path
import numpy as np
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ    = Path("/home/jason/projects/swingcue-postest")
CACHE   = PROJ / "engine/kp_cache/teach"
OUT_WIN = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/face_tilt_v01")
OUT_WIN.mkdir(parents=True, exist_ok=True)
OUT_PROJ = PROJ / "output/face_tilt_v01"
OUT_PROJ.mkdir(parents=True, exist_ok=True)
SPLIT_BASE = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/split_check")

# Clip definitions from previous pipeline results
CLIPS = [
    # (stem_side,       video_path,                           marker,      target_side, addr, top, impact)
    ("dtl-4_left",  SPLIT_BASE/"dtl-4"/"left.mp4",  "checkmark(✓)", "right", 33, 69, 226),
    ("dtl-4_right", SPLIT_BASE/"dtl-4"/"right.mp4", "cross(✗)",     "right", 49, 81, 162),
    ("dlt-6_left",  SPLIT_BASE/"dlt-6"/"left.mp4",  "cross(✗)",     None,     7, 26,  74),
    ("dlt-6_right", SPLIT_BASE/"dlt-6"/"right.mp4", "OK(○)",        "right",  7, 28,  51),
]

KP_THR  = 0.30
MIN_SHW = 20   # minimum shoulder width (px) for valid face-on measurement


# ─── helpers ──────────────────────────────────────────────────────────────────

def safe_pt(kps: dict, name: str):
    k = kps.get(name, {})
    if k.get("score", 0) >= KP_THR and (k.get("x", 0) > 0 or k.get("y", 0) > 0):
        return (k["x"], k["y"])
    return None


def spine_lateral_tilt(kps: dict, target_side: str = "right"):
    """
    Compute spine_lateral_tilt for a single frame.

    Returns (tilt_deg: float, sh_w: float, valid: bool, note: str)
    tilt_deg: positive = toward target_side. NaN if invalid.
    sh_w: shoulder width in px (for audit).
    """
    ls = safe_pt(kps, "left_shoulder")
    rs = safe_pt(kps, "right_shoulder")
    lh = safe_pt(kps, "left_hip")
    rh = safe_pt(kps, "right_hip")

    if not (ls and rs and lh and rh):
        missing = [n for n, p in [("left_shoulder",ls),("right_shoulder",rs),
                                   ("left_hip",lh),("right_hip",rh)] if p is None]
        return float("nan"), 0.0, False, f"kp_guard_fail: {missing}"

    sh_w = math.hypot(rs[0]-ls[0], rs[1]-ls[1])
    if sh_w < MIN_SHW:
        return float("nan"), sh_w, False, f"sh_w={sh_w:.1f}px<{MIN_SHW} (shoulders stacked, DTL-ish pose)"

    # Normalized coordinates (divided by shoulder width)
    sh_mid_x = (ls[0]+rs[0]) / 2
    sh_mid_y = (ls[1]+rs[1]) / 2
    hp_mid_x = (lh[0]+rh[0]) / 2
    hp_mid_y = (lh[1]+rh[1]) / 2

    # Torso vector (hip→shoulder) in shoulder-width units
    dx = (sh_mid_x - hp_mid_x) / sh_w
    dy = (sh_mid_y - hp_mid_y) / sh_w   # negative = shoulder above hip

    # Angle from image vertical (0, up = (0, -1))
    # atan2(lateral_component, vertical_component_upward)
    tilt_raw = math.degrees(math.atan2(dx, -dy))
    # tilt_raw: positive = lean toward +x (right in image)

    if target_side == "right":
        tilt = tilt_raw
    elif target_side == "left":
        tilt = -tilt_raw
    else:
        # target_side unknown — keep image-right convention, flag needs_human
        tilt = tilt_raw

    note = "ok" if target_side in ("right", "left") else "needs_human:target_side=None, raw=image-right"
    return tilt, sh_w, True, note


def get_kps(d: dict, fr_idx: int):
    fr = d["frames"][fr_idx]
    if not fr.get("persons"):
        return {}
    return fr["persons"][0]["keypoints"]


def tilt_series(d: dict, fr_start: int, fr_end: int, target_side):
    """Compute tilt for each frame in [fr_start, fr_end]. Returns (frames, tilts, sh_ws)."""
    frames, tilts, sh_ws = [], [], []
    for fi in range(fr_start, fr_end + 1):
        if fi >= len(d["frames"]):
            break
        kps = get_kps(d, fi)
        t, sw, valid, _ = spine_lateral_tilt(kps, target_side or "right")
        frames.append(fi); tilts.append(t); sh_ws.append(sw)
    return frames, tilts, sh_ws


# ─── curve plot ───────────────────────────────────────────────────────────────

def plot_curve(stem, marker, frames, tilts, addr_fr, top_fr, out_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    valid = [(f, t) for f, t in zip(frames, tilts) if not math.isnan(t)]
    if not valid:
        ax.text(0.5, 0.5, "all NaN", ha="center", va="center", transform=ax.transAxes)
    else:
        vf, vt = zip(*valid)
        ax.plot(vf, vt, "-o", ms=3, lw=1.5, color="#2196F3", label="spine_lateral_tilt")
        # mark top frame
        try:
            ti = frames.index(top_fr)
            if not math.isnan(tilts[ti]):
                ax.axvline(top_fr, color="#E91E63", lw=1.2, linestyle="--", label=f"top fr{top_fr}")
                ax.scatter([top_fr], [tilts[ti]], color="#E91E63", zorder=5, s=60)
        except ValueError:
            pass
        try:
            ai = frames.index(addr_fr)
            if not math.isnan(tilts[ai]):
                ax.axvline(addr_fr, color="#4CAF50", lw=1.0, linestyle=":", label=f"addr fr{addr_fr}")
        except ValueError:
            pass
    ax.axhline(0, color="#888", lw=0.8, linestyle="-")
    ax.set_xlabel("Frame"); ax.set_ylabel("spine_lateral_tilt (°)")
    ax.set_title(f"{stem}  {marker}\naddr→top window | positive=toward_target")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─── top frame render ─────────────────────────────────────────────────────────

def render_top_frame(video_path, top_fr, kps_top, tilt_deg, sh_w, stem, marker, out_path):
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(top_fr, n-1))
    ret, frame = cap.read(); cap.release()
    if not ret:
        print(f"  WARN: could not read fr{top_fr} from {video_path.name}"); return

    ls = safe_pt(kps_top, "left_shoulder")
    rs = safe_pt(kps_top, "right_shoulder")
    lh = safe_pt(kps_top, "left_hip")
    rh = safe_pt(kps_top, "right_hip")

    if ls and rs and lh and rh and not math.isnan(tilt_deg):
        sh_mid = (int((ls[0]+rs[0])/2), int((ls[1]+rs[1])/2))
        hp_mid = (int((lh[0]+rh[0])/2), int((lh[1]+rh[1])/2))

        # Draw torso axis (extended slightly)
        dx = sh_mid[0] - hp_mid[0]; dy = sh_mid[1] - hp_mid[1]
        length = math.hypot(dx, dy)
        extend = 1.2  # extend 20% past shoulder
        ex = int(sh_mid[0] + extend * dx); ey = int(sh_mid[1] + extend * dy)
        bx = int(hp_mid[0] - 0.2 * dx); by = int(hp_mid[1] - 0.2 * dy)
        cv2.line(frame, (bx, by), (ex, ey), (0, 255, 128), 3)  # green torso axis
        cv2.circle(frame, sh_mid, 6, (255, 200, 0), -1)
        cv2.circle(frame, hp_mid, 6, (255, 200, 0), -1)

        # Draw vertical reference line through sh_mid
        vlen = int(length * 1.5)
        cv2.line(frame, (sh_mid[0], sh_mid[1] - vlen), (sh_mid[0], sh_mid[1] + vlen),
                 (255, 255, 255), 1, cv2.LINE_AA)

    # Annotation box
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-130), (w, h), (0,0,0), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    tilt_str = f"{tilt_deg:+.2f}" if not math.isnan(tilt_deg) else "NaN"
    cv2.putText(frame, f"spine_lateral_tilt(top)= {tilt_str}deg", (10, h-95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,128), 2)
    cv2.putText(frame, f"sh_width= {sh_w:.0f}px  fr{top_fr}", (10, h-60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    cv2.putText(frame, f"{stem}  {marker}  +ve=toward_target", (10, h-28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

    cv2.imwrite(str(out_path), frame)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    results = []

    for stem, video_path, marker, target_side, addr_fr, top_fr, impact_fr in CLIPS:
        print(f"\n{'='*55}")
        print(f"  {stem}  {marker}  target_side={target_side}")

        d = json.load(open(CACHE / f"{stem}.json"))

        # Address frame tilt (baseline)
        kps_addr = get_kps(d, addr_fr)
        t_addr, sw_addr, v_addr, note_addr = spine_lateral_tilt(kps_addr, target_side or "right")
        print(f"  address fr{addr_fr}: tilt={t_addr:.2f}°  sw={sw_addr:.0f}px  note={note_addr}")

        # Top frame tilt
        kps_top = get_kps(d, top_fr)
        t_top, sw_top, v_top, note_top = spine_lateral_tilt(kps_top, target_side or "right")
        print(f"  top     fr{top_fr}: tilt={t_top:.2f}°  sw={sw_top:.0f}px  note={note_top}")

        # Trajectory (address→top)
        frames, tilts, sh_ws = tilt_series(d, addr_fr, top_fr, target_side)
        valid_count = sum(1 for t in tilts if not math.isnan(t))
        print(f"  trajectory: {len(frames)}fr, {valid_count} valid")

        # Curve plot
        curve_path = OUT_WIN / f"{stem}_tilt_curve.png"
        plot_curve(stem, marker, frames, tilts, addr_fr, top_fr, curve_path)
        print(f"  curve → {curve_path.name}")

        # Top frame render
        render_path = OUT_WIN / f"{stem}_top_render.jpg"
        render_top_frame(video_path, top_fr, kps_top, t_top, sw_top, stem, marker, render_path)
        print(f"  render → {render_path.name}")

        result = {
            "stem": stem, "marker": marker, "target_side": target_side,
            "addr_fr": addr_fr, "top_fr": top_fr, "impact_fr": impact_fr,
            "addr_tilt": round(t_addr, 2) if not math.isnan(t_addr) else None,
            "addr_sw":   round(sw_addr, 1),
            "addr_note": note_addr,
            "top_tilt":  round(t_top, 2) if not math.isnan(t_top) else None,
            "top_sw":    round(sw_top, 1),
            "top_note":  note_top,
            "trajectory_len": len(frames),
            "trajectory_valid": valid_count,
        }
        results.append(result)

    # ── Paired diff report ──────────────────────────────────────────────────────
    def get_r(pair_stem, side):
        return next(r for r in results if r["stem"] == f"{pair_stem}_{side}")

    print("\n\n" + "="*60)
    print("每片段汇报 (spine_lateral_tilt @ top frame)")
    print("="*60)
    print(f"{'片段':16s} {'标识':12s} {'target_side':11s} {'addr_tilt':>10} {'top_tilt':>10} {'top_sw':>7} {'note'}")
    print("-"*100)
    for r in results:
        a = f"{r['addr_tilt']:+.2f}°" if r["addr_tilt"] is not None else "NaN"
        t = f"{r['top_tilt']:+.2f}°" if r["top_tilt"] is not None else "NaN"
        print(f"{r['stem']:16s} {r['marker']:12s} {str(r['target_side']):11s} "
              f"{a:>10} {t:>10} {r['top_sw']:>6.0f}px  {r['top_note']}")

    print("\n" + "="*60)
    print("配对差值 (错误 − 正确, top_tilt)")
    print("="*60)
    for pair_stem, wrong_side, correct_side in [("dtl-4","right","left"),
                                                  ("dlt-6","left","right")]:
        w = get_r(pair_stem, wrong_side)
        c = get_r(pair_stem, correct_side)
        if w["top_tilt"] is not None and c["top_tilt"] is not None:
            diff = round(w["top_tilt"] - c["top_tilt"], 2)
        else:
            diff = None
        d_str = f"{diff:+.2f}°" if diff is not None else "N/A"
        print(f"\n  {pair_stem}: 错误({wrong_side}, {w['marker']}) − 正确({correct_side}, {c['marker']})")
        print(f"    错误 top_tilt  = {w['top_tilt']}°  (fr{w['top_fr']})")
        print(f"    正确 top_tilt  = {c['top_tilt']}°  (fr{c['top_fr']})")
        print(f"    差值 Δtilt     = {d_str}")
        if pair_stem == "dlt-6" and (w["target_side"] is None or c["target_side"] is None):
            print(f"    NOTE: dlt-6/left target_side=None → 符号以image-right为准, needs_human核实方向")

    # Gate note for dlt-6/left
    print("\n" + "="*60)
    print("needs_human 记录:")
    for r in results:
        if "needs_human" in r.get("top_note","") or r["target_side"] is None:
            print(f"  {r['stem']}: target_side=None → 倾斜角符号待人工确认方向")

    # Save JSON
    out_j = OUT_PROJ / "spine_tilt_v01_results.json"
    with open(out_j, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)
    print(f"\n结果JSON: {out_j}")
    print(f"输出目录(Windows): C:\\Users\\jason\\Desktop\\rtmpose_results\\preview\\face_tilt_v01\\")
    print("  文件: dtl-4_left_tilt_curve.png / _top_render.jpg (×4段)")

    return results


if __name__ == "__main__":
    main()
