#!/usr/bin/env python3
"""
gate3_no_false_positive.py  — v2
关卡3 — 诊断 + 不误报验证

Top 检测 v2 规则:
  1. 搜索区间: [15%, 65%] 排除 setup 和收杆段
  2. 振幅检查: zone 内 wrist_y 振幅 >= MIN_AMP_PX(40px) —— 振幅过小说明无真实上杆
  3. 若上述任一失败 → phase_detection_low_confidence → 沉默 (不硬判)

Confidence Ledger (完整):
  1. camera_view 非 face_on → SILENT
  2. top 不在 [15%,65%] 区间 → SILENT
  3. zone 内 wrist 振幅 < 40px → SILENT (no real backswing detected)
  4. window 有效帧 < MIN_VALID_WINDOW → SILENT
  5. tilt 计算失败(关键点不足) → SILENT
"""

import sys, json, math, statistics
from pathlib import Path

PROJ = Path("/home/jason/projects/swingcue-postest")
sys.path.insert(0, str(PROJ))

from engine.reference_flywheel import load_baseline, query_tilt
from engine.features.triline_geometry import _safe_pt, _lateral_tilt_deg

KP_GUARD          = 0.30
MIN_SHW           = 15.0
WINDOW            = 5        # top ± 5 frames
MIN_VALID_WINDOW  = 3        # need ≥ 3 valid frames in window
ZONE_LO_PCT       = 0.15     # earliest valid top position
ZONE_HI_PCT       = 0.65     # latest valid top position
MIN_AMP_PX        = 40.0     # minimum wrist_y range in zone (px) — no real backswing below this

# ─── keypoint helpers ─────────────────────────────────────────────────────────

def load_kp_json(clip_id: str) -> dict | None:
    for sub in ["batch3", "batch2", "batch1", ""]:
        p = (PROJ / "engine/kp_cache" / sub / f"{clip_id}.json") if sub \
            else (PROJ / "engine/kp_cache" / f"{clip_id}.json")
        if p.exists():
            return json.load(open(p))
    return None

def tilt_from_kps(kp_dict: dict) -> float | None:
    ls = _safe_pt(kp_dict, "left_shoulder"); rs = _safe_pt(kp_dict, "right_shoulder")
    lh = _safe_pt(kp_dict, "left_hip");      rh = _safe_pt(kp_dict, "right_hip")
    if not (ls and rs and lh and rh): return None
    sh_w = math.hypot(rs[0]-ls[0], rs[1]-ls[1])
    if sh_w < MIN_SHW: return None
    sx = (ls[0]+rs[0])/2; sy = (ls[1]+rs[1])/2
    hx = (lh[0]+rh[0])/2; hy = (lh[1]+rh[1])/2
    return _lateral_tilt_deg((hx, hy), (sx, sy))

def wrist_y(kp_dict: dict) -> float | None:
    lw = _safe_pt(kp_dict, "left_wrist"); rw = _safe_pt(kp_dict, "right_wrist")
    ys = [pt[1] for pt in [lw, rw] if pt]
    return min(ys) if ys else None

# ─── Top detection v2 ─────────────────────────────────────────────────────────

def find_top_v2(frames: list) -> tuple[int | None, float, float]:
    """
    Find top-of-backswing frame using v2 rules.

    Returns:
        top_idx:  frame index of top (None if not found or invalid)
        amp_px:   wrist_y range within [15%, 65%] zone (px)
        zone_lo:  start of zone (frame index)
    """
    n = len(frames)
    lo = int(n * ZONE_LO_PCT)
    hi = int(n * ZONE_HI_PCT)

    zone_wys: list[tuple[int, float]] = []
    for i in range(lo, hi + 1):
        p = frames[i].get("persons", [])
        kps = p[0].get("keypoints", {}) if p else {}
        wy = wrist_y(kps)
        if wy is not None:
            zone_wys.append((i, wy))

    if not zone_wys:
        return None, 0.0, float(lo)

    # Amplitude in zone
    wy_vals = [w for _, w in zone_wys]
    amp_px  = max(wy_vals) - min(wy_vals)

    # Minimum wrist_y = highest wrist position = top of backswing
    top_idx, _ = min(zone_wys, key=lambda x: x[1])

    return top_idx, amp_px, float(lo)

def window_tilt_median(frames: list, top_idx: int) -> tuple[float | None, list[float], int]:
    n = len(frames); tilts = []; valid = 0
    for i in range(max(0, top_idx - WINDOW), min(n, top_idx + WINDOW + 1)):
        p = frames[i].get("persons", [])
        if not p: continue
        kps = p[0].get("keypoints", {})
        valid += 1
        t = tilt_from_kps(kps)
        if t is not None:
            tilts.append(t)
    return (statistics.median(tilts) if tilts else None), tilts, valid

# ─── Confidence Ledger ────────────────────────────────────────────────────────

def confidence_ledger_check(
    clip_id: str,
    top_idx: int | None,
    amp_px: float,
    window_valid: int,
) -> tuple[str, str]:
    """
    Returns (status, reason).  status: "RUN" | "SILENT"
    """
    # 1. camera_view gate
    layer0_path = PROJ / "engine/layer0/records" / f"{clip_id}.json"
    if layer0_path.exists():
        rec = json.load(open(layer0_path))
        if rec.get("verdict") == "REJECT":
            return "SILENT", "layer0 verdict=REJECT"
        cam = rec.get("camera_view", "")
        if cam and cam not in ("face_on", "face-on"):
            return "SILENT", f"camera_view={cam} (not face_on)"

    # 2. Top in zone
    if top_idx is None:
        return "SILENT", "phase_detection: no wrist keypoints in zone"

    # 3. Amplitude gate — no real backswing
    if amp_px < MIN_AMP_PX:
        return "SILENT", (f"phase_detection_low_confidence: "
                          f"zone wrist_y amp={amp_px:.1f}px < {MIN_AMP_PX}px "
                          f"(no real backswing detected, top unreliable)")

    # 4. Window valid frames
    if window_valid < MIN_VALID_WINDOW:
        return "SILENT", f"window valid frames={window_valid} < {MIN_VALID_WINDOW}"

    return "RUN", "ok"

# ─── Diagnose one clip ────────────────────────────────────────────────────────

def diagnose_clip(clip_id: str, label: str, baseline: dict) -> dict:
    kp_json = load_kp_json(clip_id)
    if kp_json is None:
        return dict(clip_id=clip_id, label=label, status="SILENT",
                    reason="kp_cache not found", tilt_deg=None, confidence=None,
                    top_idx=None, amp_px=None, window_valid=0)

    frames    = kp_json.get("frames", [])
    n_frames  = len(frames)

    top_idx, amp_px, zone_lo = find_top_v2(frames)

    tilt_med, tilt_window, window_valid = (None, [], 0) if top_idx is None else \
        window_tilt_median(frames, top_idx)

    status, reason = confidence_ledger_check(clip_id, top_idx, amp_px, window_valid)

    confidence = None
    if status == "RUN":
        if tilt_med is None:
            status, reason = "SILENT", "tilt computation failed (no valid window tilts)"
        else:
            confidence = query_tilt(tilt_med, baseline)["confidence"]

    return dict(
        clip_id=clip_id, label=label,
        n_frames=n_frames, top_idx=top_idx,
        amp_px=round(amp_px, 1) if amp_px is not None else None,
        zone_lo=int(zone_lo), zone_hi=int(n_frames * ZONE_HI_PCT),
        tilt_deg=round(tilt_med, 2) if tilt_med is not None else None,
        window_tilts=[round(t, 1) for t in tilt_window],
        window_valid=window_valid,
        status=status, reason=reason,
        confidence=confidence,
    )

# ─── Positive control (clip_016 from gate1 human-confirmed values) ────────────

def diagnose_clip016_controls(baseline: dict) -> list[dict]:
    results = []
    for side, tilt_val, exp in [(  "left_ERROR", +29.1, "Confirmed"),
                                 ("right_OK",     -6.8,  "None")]:
        r = query_tilt(tilt_val, baseline)
        results.append(dict(
            clip_id=f"clip_016/{side}", label="positive_control",
            n_frames="N/A", top_idx="N/A", amp_px="N/A",
            zone_lo="N/A", zone_hi="N/A",
            tilt_deg=tilt_val, window_tilts=[], window_valid="N/A",
            status="RUN",
            reason=f"gate1 human-confirmed, expected={exp}",
            confidence=r["confidence"],
        ))
    return results

# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    baseline = load_baseline()
    band_lo  = baseline["reference_band"]["band_lower_deg"]
    band_hi  = baseline["reference_band"]["band_upper_deg"]
    mu       = baseline["reference_band"]["center_mu_deg"]

    print(f"\n{'='*72}")
    print(f"  关卡3 v2 — 诊断 + 不误报验证 (top检测v2: zone[15%,65%] + amp≥40px)")
    print(f"  Baseline: mu={mu}°  band=[{band_lo}, {band_hi}]°")
    print(f"  Ledger: None≤5° | 5<Possible≤15° | 15<Likely≤28° | Confirmed>28°")
    print(f"{'='*72}\n")

    normal_clips = [
        ("fo-eet-1", "creator_normal"),
        ("fo-eet-2", "creator_normal"),
        ("fo-eet-3", "creator_normal"),
        ("fo-ok-1",  "labeled_ok"),
        ("fo-ok-2",  "labeled_ok"),
    ]

    all_results = []
    false_positives = []

    print("--- 正常杆 (预期: None/Possible/SILENT，不得 Likely+) ---")
    for clip_id, label in normal_clips:
        r = diagnose_clip(clip_id, label, baseline)
        all_results.append(r)
        tilt_s  = f"{r['tilt_deg']:+.2f}°" if r["tilt_deg"] is not None else "  N/A  "
        amp_s   = f"{r['amp_px']}px" if r["amp_px"] is not None else "N/A"
        top_s   = f"fr{r['top_idx']}" if r["top_idx"] is not None else "None"
        diag    = r["confidence"] if r["status"] == "RUN" else f"SILENT({r['reason'][:55]})"
        fp_flag = ""
        if r["status"] == "RUN" and r["confidence"] in ("Likely", "Confirmed"):
            fp_flag = "  ⚠️ FALSE POSITIVE"
            false_positives.append(r)
        print(f"  {clip_id:12s}  top={top_s:>5s}  amp={amp_s:>6s}  tilt={tilt_s}  {diag}{fp_flag}")

    print("\n--- 阳性对照 (clip_016, 预期 ERROR=Confirmed / OK=None) ---")
    ctrl_results = diagnose_clip016_controls(baseline)
    for r in ctrl_results:
        print(f"  {r['clip_id']:25s}  tilt={r['tilt_deg']:+.2f}°  "
              f"confidence={r['confidence']:10s}  {r['reason']}")
        all_results.append(r)

    # ── summary ──────────────────────────────────────────────────────────────
    n_normal  = len(normal_clips)
    rn        = all_results[:n_normal]
    n_silent  = sum(1 for r in rn if r["status"] == "SILENT")
    n_none    = sum(1 for r in rn if r["confidence"] == "None")
    n_poss    = sum(1 for r in rn if r["confidence"] == "Possible")
    n_fp      = len(false_positives)

    ctrl_err = next(r for r in ctrl_results if "ERROR" in r["clip_id"])
    ctrl_ok  = next(r for r in ctrl_results if "right_OK"  in r["clip_id"])

    print(f"\n{'='*72}")
    print("  总结")
    print(f"{'='*72}")
    print(f"  正常杆 {n_normal} 条:")
    print(f"    沉默(SILENT)  = {n_silent}")
    print(f"    带内(None)    = {n_none}")
    print(f"    Possible      = {n_poss}")
    print(f"    误报(Likely+) = {n_fp}  {'✅ 无误报' if n_fp == 0 else '❌ 存在误报，需分析'}")
    err_ok = ctrl_err["confidence"] == "Confirmed"
    ok_ok  = ctrl_ok["confidence"]  == "None"
    print(f"\n  阳性对照:")
    print(f"    clip_016/left_ERROR  → {ctrl_err['confidence']:10s}  {'✅' if err_ok else '❌'}")
    print(f"    clip_016/right_OK    → {ctrl_ok['confidence']:10s}  {'✅' if ok_ok else '❌'}")

    verdict = "PASS" if n_fp == 0 and err_ok and ok_ok else "FAIL"
    print(f"\n  关卡3 v2 总裁决: {verdict}")

    if false_positives:
        print("\n  ⚠️ 误报详情:")
        for r in false_positives:
            print(f"    {r['clip_id']}: tilt={r['tilt_deg']:+.2f}°  conf={r['confidence']}")

    # JSON
    out_path = PROJ / "output/gate3_no_fp/gate3_results_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(
        version="v2",
        top_detection="zone[15%,65%]+amp>=40px",
        summary=dict(
            n_normal=n_normal, n_silent=n_silent, n_none=n_none,
            n_possible=n_poss, n_false_positives=n_fp,
            positive_control_error=ctrl_err["confidence"],
            positive_control_ok=ctrl_ok["confidence"],
            verdict=verdict,
        ),
        results=all_results,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  JSON → {out_path}\n")


if __name__ == "__main__":
    main()
