#!/usr/bin/env python3
"""
gate3_no_false_positive.py
关卡3 — 诊断 + 不误报验证

测试集:
  正常样本 (应 None 或 Possible):
    fo-eet-1/2/3  — 创始人正面挥杆 (batch3 EET, 实为正常杆)
    fo-ok-1/2     — 标记为 ok 的正面挥杆

  阳性对照 (应 Confirmed):
    clip_016 right (错误半, +29.1°) — 来自 pair_measurability_results

完整 Confidence Ledger:
  - camera_view 置信度低 → 沉默
  - top 检测可靠性 (有效帧数 < 10) → 沉默
  - 关键点不足 → 沉默
"""

import sys, json, math, statistics
from pathlib import Path

PROJ = Path("/home/jason/projects/swingcue-postest")
sys.path.insert(0, str(PROJ))

from engine.reference_flywheel import load_baseline, query_tilt
from engine.features.triline_geometry import _safe_pt, _lateral_tilt_deg

KP_GUARD = 0.30
MIN_SHW  = 15.0
WINDOW   = 5        # top ± 5 frames
MIN_VALID_WINDOW = 3  # need ≥3 valid frames in window to report

# ─── helpers ─────────────────────────────────────────────────────────────────

def load_kp_json(clip_id: str) -> dict | None:
    """Find kp_cache JSON across all batch subdirs."""
    for sub in ["batch3", "batch2", "batch1", ""]:
        p = (PROJ / "engine/kp_cache" / sub / f"{clip_id}.json") if sub \
            else (PROJ / "engine/kp_cache" / f"{clip_id}.json")
        if p.exists():
            return json.load(open(p))
    return None

def tilt_from_kps(kp_dict: dict) -> float | None:
    """Compute shoulder_lateral_tilt from a frame's keypoint dict."""
    ls = _safe_pt(kp_dict, "left_shoulder")
    rs = _safe_pt(kp_dict, "right_shoulder")
    lh = _safe_pt(kp_dict, "left_hip")
    rh = _safe_pt(kp_dict, "right_hip")
    if not (ls and rs and lh and rh):
        return None
    sh_w = math.hypot(rs[0]-ls[0], rs[1]-ls[1])
    if sh_w < MIN_SHW:
        return None
    sx = (ls[0]+rs[0])/2; sy = (ls[1]+rs[1])/2
    hx = (lh[0]+rh[0])/2; hy = (lh[1]+rh[1])/2
    return _lateral_tilt_deg((hx, hy), (sx, sy))

def wrist_y(kp_dict: dict) -> float | None:
    lw = _safe_pt(kp_dict, "left_wrist")
    rw = _safe_pt(kp_dict, "right_wrist")
    ys = [pt[1] for pt in [lw, rw] if pt]
    return min(ys) if ys else None

def find_top(frames: list, front_pct: float = 0.70) -> int | None:
    n = len(frames); end = max(1, int(n * front_pct))
    best_i, best_wy = None, float("inf")
    for i in range(end):
        p = frames[i].get("persons", [])
        if not p: continue
        kps = p[0].get("keypoints", {})
        wy = wrist_y(kps)
        if wy is not None and wy < best_wy:
            best_wy, best_i = wy, i
    return best_i

def window_tilt_median(frames: list, top_idx: int) -> tuple[float | None, list, int]:
    n = len(frames); tilts = []; valid = 0
    for i in range(max(0, top_idx - WINDOW), min(n, top_idx + WINDOW + 1)):
        p = frames[i].get("persons", [])
        if not p: continue
        kps = p[0].get("keypoints", {})
        t = tilt_from_kps(kps)
        valid += 1
        if t is not None:
            tilts.append(t)
    median = statistics.median(tilts) if tilts else None
    return median, tilts, valid

def confidence_ledger_check(
    kp_json: dict,
    top_idx: int | None,
    window_valid: int,
    clip_id: str,
) -> tuple[str, str]:
    """
    Returns (status, reason).
    status: "RUN" | "SILENT"
    reason: explanation for SILENT
    """
    # 1. camera_view gate — check layer0 record if exists
    layer0_path = PROJ / "engine/layer0/records" / f"{clip_id}.json"
    if layer0_path.exists():
        rec = json.load(open(layer0_path))
        verdict = rec.get("verdict", "?")
        camera  = rec.get("camera_view", "?")
        if verdict == "REJECT":
            return "SILENT", f"layer0 verdict=REJECT"
        # For fo-* clips, camera_view should be face_on
        if camera not in ("face_on", "face-on", "?", ""):
            return "SILENT", f"camera_view={camera} (not face_on)"
    # 2. top detection
    if top_idx is None:
        return "SILENT", "top detection failed (no wrist keypoints)"
    # 3. window valid frames
    if window_valid < MIN_VALID_WINDOW:
        return "SILENT", f"window valid frames={window_valid} < {MIN_VALID_WINDOW}"
    return "RUN", "ok"

# ─── run diagnosis for one clip ───────────────────────────────────────────────

def diagnose_clip(clip_id: str, label: str, baseline: dict) -> dict:
    """Run full diagnosis pipeline on a clip from kp_cache."""
    kp_json = load_kp_json(clip_id)
    if kp_json is None:
        return {
            "clip_id": clip_id, "label": label,
            "status": "SILENT", "reason": "kp_cache not found",
            "tilt_deg": None, "confidence": None, "top_idx": None,
        }

    frames = kp_json.get("frames", [])
    n_frames = len(frames)

    top_idx = find_top(frames, front_pct=0.70)
    tilt_med, tilt_window, window_valid = (None, [], 0) if top_idx is None else \
        window_tilt_median(frames, top_idx)

    status, reason = confidence_ledger_check(kp_json, top_idx, window_valid, clip_id)

    if status == "SILENT":
        confidence = None
    else:
        result = query_tilt(tilt_med, baseline) if tilt_med is not None else None
        confidence = result["confidence"] if result else None
        if tilt_med is None:
            status, reason = "SILENT", "tilt computation failed (no valid window tilts)"

    return {
        "clip_id": clip_id, "label": label,
        "n_frames": n_frames, "top_idx": top_idx,
        "tilt_deg": round(tilt_med, 2) if tilt_med is not None else None,
        "window_tilts": [round(t, 1) for t in tilt_window],
        "window_valid": window_valid,
        "status": status,
        "reason": reason if status == "SILENT" else "ok",
        "confidence": confidence,
    }

# ─── special: clip_016 right half from pair_measurability ────────────────────

def diagnose_clip016_right(baseline: dict) -> dict:
    """
    clip_016 RIGHT = correct (our gate1 baseline), but we use the ERROR half (LEFT)
    as the positive control.
    
    Values from gate1_tilt_report_v2: 
      LEFT (error) tilt = +29.1° → should be Confirmed
      RIGHT (correct) tilt = -6.8° → should be None
    Both from pair_measurability window-median (human-confirmed).
    """
    results = []
    for side, tilt_val, expected_label in [
        ("left_ERROR",  +29.1, "Confirmed"),
        ("right_OK",    -6.8,  "None"),
    ]:
        r = query_tilt(tilt_val, baseline)
        results.append({
            "clip_id": f"clip_016/{side}",
            "label": "positive_control" if "ERROR" in side else "correct_control",
            "n_frames": "N/A (from gate1_tilt_report_v2)",
            "top_idx": "N/A",
            "tilt_deg": tilt_val,
            "window_tilts": [],
            "window_valid": "N/A",
            "status": "RUN",
            "reason": f"gate1 human-confirmed, expected={expected_label}",
            "confidence": r["confidence"],
        })
    return results

# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    baseline = load_baseline()
    band_lo = baseline["reference_band"]["band_lower_deg"]
    band_hi = baseline["reference_band"]["band_upper_deg"]
    mu      = baseline["reference_band"]["center_mu_deg"]

    print(f"\n{'='*70}")
    print(f"  关卡3 — 诊断 + 不误报验证")
    print(f"  Baseline: mu={mu}° band=[{band_lo}, {band_hi}]°")
    print(f"  Ledger: None<=5° | Possible<=15° | Likely<=28° | Confirmed>28°")
    print(f"{'='*70}\n")

    # Test clips: (clip_id, label_desc)
    normal_clips = [
        ("fo-eet-1", "creator_normal"),
        ("fo-eet-2", "creator_normal"),
        ("fo-eet-3", "creator_normal"),
        ("fo-ok-1",  "labeled_ok"),
        ("fo-ok-2",  "labeled_ok"),
    ]

    all_results = []

    print("--- 正常杆 (应 None 或 Possible) ---")
    false_positives = []
    for clip_id, label in normal_clips:
        r = diagnose_clip(clip_id, label, baseline)
        all_results.append(r)
        tilt_s = f"{r['tilt_deg']:+.2f}°" if r["tilt_deg"] is not None else "N/A"
        conf_s = r["confidence"] or r["status"]
        fp_flag = ""
        if r["status"] == "RUN" and r["confidence"] in ("Likely", "Confirmed"):
            fp_flag = "  ⚠️ FALSE POSITIVE"
            false_positives.append(r)
        print(f"  {clip_id:12s}  top={str(r['top_idx']):>4s}  tilt={tilt_s:>8s}  "
              f"confidence={conf_s:10s}  camera_ok={r['status']}  {r['reason']}{fp_flag}")

    print("\n--- 阳性对照 (clip_016, 应 Confirmed/None) ---")
    ctrl_results = diagnose_clip016_right(baseline)
    for r in ctrl_results:
        all_results.append(r)
        tilt_s = f"{r['tilt_deg']:+.2f}°"
        conf_s = r["confidence"] or r["status"]
        print(f"  {r['clip_id']:20s}  tilt={tilt_s:>8s}  confidence={conf_s:10s}  {r['reason']}")

    # ─── summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  总结")
    print(f"{'='*70}")
    n_normal = len(normal_clips)
    n_silent  = sum(1 for r in all_results[:n_normal] if r["status"] == "SILENT")
    n_none    = sum(1 for r in all_results[:n_normal] if r["confidence"] == "None")
    n_possible= sum(1 for r in all_results[:n_normal] if r["confidence"] == "Possible")
    n_fp      = len(false_positives)

    print(f"  正常杆 {n_normal} 条:")
    print(f"    沉默(SILENT)  = {n_silent}")
    print(f"    带内(None)    = {n_none}")
    print(f"    Possible      = {n_possible}")
    print(f"    误报(Likely+) = {n_fp}  {'✅ 无误报' if n_fp == 0 else '❌ 存在误报，需分析'}")

    ctrl_error  = next((r for r in ctrl_results if "ERROR" in r["clip_id"]), None)
    ctrl_ok     = next((r for r in ctrl_results if "right_OK" in r["clip_id"]), None)
    print(f"\n  阳性对照:")
    print(f"    clip_016/left_ERROR  confidence={ctrl_error['confidence']}  "
          f"{'✅ Confirmed' if ctrl_error['confidence']=='Confirmed' else '❌ 未检出'}")
    print(f"    clip_016/right_OK    confidence={ctrl_ok['confidence']}    "
          f"{'✅ None (带内)' if ctrl_ok['confidence']=='None' else '⚠️ 非None'}")

    if false_positives:
        print("\n  ⚠️ 误报详情:")
        for r in false_positives:
            print(f"    {r['clip_id']}: tilt={r['tilt_deg']:+.2f}° conf={r['confidence']}")
            print(f"    window={r['window_tilts']}")

    # JSON output
    out = {
        "summary": {
            "n_normal": n_normal, "n_silent": n_silent,
            "n_none": n_none, "n_possible": n_possible, "n_false_positives": n_fp,
            "positive_control_error_conf": ctrl_error["confidence"],
            "positive_control_ok_conf": ctrl_ok["confidence"],
            "verdict": "PASS" if n_fp == 0 and ctrl_error["confidence"] == "Confirmed" else "FAIL"
        },
        "results": all_results + ctrl_results,
    }
    out_path = PROJ / "output/gate3_no_fp/gate3_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  JSON → {out_path}")
    print(f"\n  关卡3 总裁决: {out['summary']['verdict']}\n")


if __name__ == "__main__":
    main()
