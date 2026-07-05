#!/usr/bin/env python3
"""
gate3_no_false_positive.py  — v3
关卡3 — 诊断 + 不误报验证 (TOPV3-001)

Top 检测 v3 规则:
  - 单一事实源: 消费 B 层 SwingPhaseEngine 的 top 帧 + top_conf
  - 有效挥杆门: 挥杆段内 wrist_y 振幅 >= 0.8 × shoulder_width (address 帧肩宽)
  - 沉默路径:
      1. camera_view 非 face_on → SILENT (camera_gate)
      2. top_conf < CONF_THR (0.50, provisional) → SILENT (phase_detection_low_confidence)
      3. 挥杆段振幅 < 0.8 × SHW → SILENT (no real backswing, amp_gate)
      4. 窗口有效帧 < MIN_VALID_WINDOW → SILENT (window_too_small)
      5. tilt 计算失败 → SILENT (tilt_failed)
  - DTL 机位: camera_gate 拦截, SKIPPED(camera_gate), 不进 spine_lateral_tilt

已废弃:
  - v1: find_top_v2 zone[15%,65%] + 全程/区间最高腕位法
    原因: zone 假设截断真实挥杆段(fo-eet-1 DIAG-001); 最高腕位误认收杆
  - v2: zone[15%,65%] + amp>=40px 绝对阈值
    原因: 同 v1 zone 截断问题; 40px 绝对值不随体型归一化

Confidence Ledger (完整):
  None      : tilt <= +5.0°  (含等于)
  Possible  : +5.0° < tilt <= +15°
  Likely    : +15° < tilt <= +28°
  Confirmed : tilt > +28°

技术债 (已登记 FACE_ON_TRILINE_GEOMETRY_SPEC.md §9):
  - top_conf 沉默阈值 0.50 为 provisional，正式定版待阴性对照数据确认
  - B 层 0.82 百分比截止窗口与 prominence=30 绝对阈值为已知债务
  - 多挥杆段边界条件: n_eff 截断已处理; 多个 top 候选不实现(见规格 §9)
"""

import sys, json, math, statistics
from pathlib import Path

PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ))

import numpy as np
from scipy.signal import savgol_filter

from engine.b_phase.swing_phase import SwingPhaseEngine
from engine.a_measurement.pose_pipeline import PosePipeline, JOINT_NAMES
from engine.reference_flywheel import load_baseline, query_tilt
from engine.features.triline_geometry import _safe_pt, _lateral_tilt_deg

KP_GUARD          = 0.30
MIN_SHW           = 15.0
WINDOW            = 5         # top ± 5 frames for median tilt
MIN_VALID_WINDOW  = 3
CONF_THR          = 0.50      # provisional — pending negative-control data (Jason 2026-07-05)
AMP_SHW_THR       = 0.8       # swing-segment wrist_y amplitude >= 0.8 × shoulder_width

# ── kp_cache helpers ──────────────────────────────────────────────────────────

def load_kp_json(clip_id: str) -> dict | None:
    search_dirs = ["negatives", "batch3", "batch2", "batch1", "normal_group", ""]
    for sub in search_dirs:
        p = (PROJ / "engine/kp_cache" / sub / f"{clip_id}.json") if sub \
            else (PROJ / "engine/kp_cache" / f"{clip_id}.json")
        if p.exists():
            return json.load(open(p))
    return None


def get_fps(kp: dict) -> float:
    if "fps" in kp:
        return float(kp["fps"])
    stats = kp.get("stats", {})
    return float(stats.get("source_fps", stats.get("fps", 30.0)))


def get_frame_idx(fr: dict) -> int:
    return fr.get("frame_idx", fr.get("frame", 0))


# ── keypoint helpers ──────────────────────────────────────────────────────────

def tilt_from_kps(kp_dict: dict) -> float | None:
    ls = _safe_pt(kp_dict, "left_shoulder");  rs = _safe_pt(kp_dict, "right_shoulder")
    lh = _safe_pt(kp_dict, "left_hip");       rh = _safe_pt(kp_dict, "right_hip")
    if not (ls and rs and lh and rh): return None
    sh_w = math.hypot(rs[0]-ls[0], rs[1]-ls[1])
    if sh_w < MIN_SHW: return None
    sx = (ls[0]+rs[0])/2; sy = (ls[1]+rs[1])/2
    hx = (lh[0]+rh[0])/2; hy = (lh[1]+rh[1])/2
    return _lateral_tilt_deg((hx, hy), (sx, sy))


def shoulder_width_from_kps(kp_dict: dict) -> float | None:
    ls = _safe_pt(kp_dict, "left_shoulder"); rs = _safe_pt(kp_dict, "right_shoulder")
    if not (ls and rs): return None
    w = math.hypot(rs[0]-ls[0], rs[1]-ls[1])
    return w if w >= MIN_SHW else None


def wrist_y_from_kps(kp_dict: dict) -> float | None:
    lw = _safe_pt(kp_dict, "left_wrist"); rw = _safe_pt(kp_dict, "right_wrist")
    ys = [pt[1] for pt in [lw, rw] if pt]
    return min(ys) if ys else None


# ── B-layer wrist trajectory (mirrors swing_phase._extract_wrist) ─────────────

def extract_wrist_ys(frames: list) -> np.ndarray:
    n = len(frames)
    ys = np.full(n, np.nan)
    for fr in frames:
        fi = get_frame_idx(fr)
        if fi >= n: continue
        p = fr.get("persons", [])
        if not p: continue
        kps = p[0].get("keypoints", {})
        wy = wrist_y_from_kps(kps)
        if wy is not None:
            ys[fi] = wy
    idx = np.arange(n)
    nans = np.isnan(ys)
    if not nans.all():
        ys[nans] = np.interp(idx[nans], idx[~nans], ys[~nans])
    return ys


# ── B-layer top detection (v3: use SwingPhaseEngine) ─────────────────────────

class _MinimalMeasurement:
    """Minimal duck-type stand-in for FrameMeasurement, used by SwingPhaseEngine."""
    def __init__(self, fi, wx, wy, quality="ok"):
        self.frame_idx = fi
        self._wx = wx
        self._wy = wy
        self.measurement_quality = quality
        self.keypoints = {}
        self.confidences = {}

    def wrist_mid(self):
        if self._wx is None or self._wy is None:
            return None
        return (self._wx, self._wy)


def run_b_layer(kp_json: dict, camera_angle: str = "face-on") -> tuple:
    """
    Run SwingPhaseEngine on kp_cache, using minimal duck-type measurements.

    Returns:
        top_fr    : int   — top frame index from B layer
        top_conf  : float — top confidence (fixed fallback prominence bug)
        anchors   : AnchorFrames
        n         : int   — total frame count
        fps       : float
    """
    frames = kp_json["frames"]
    fps    = get_fps(kp_json)
    n      = len(frames)

    meas_list = []
    for i, fr in enumerate(frames):
        fi  = get_frame_idx(fr)
        p   = fr.get("persons", [])
        kps = p[0].get("keypoints", {}) if p else {}

        lw = _safe_pt(kps, "left_wrist"); rw = _safe_pt(kps, "right_wrist")
        pts = [pt for pt in [lw, rw] if pt]
        if pts:
            wx = sum(pt[0] for pt in pts) / len(pts)
            wy = sum(pt[1] for pt in pts) / len(pts)
            quality = "ok"
        else:
            wx = wy = None
            quality = "bad"

        meas_list.append(_MinimalMeasurement(fi, wx, wy, quality))

    angle_key = "down-the-line" if camera_angle in ("dtl", "down-the-line") else "face-on"
    engine = SwingPhaseEngine()
    _, anchors = engine.run(meas_list, fps, angle=angle_key)

    return anchors.top, anchors.top_conf, anchors, n, fps


# ── Swing-segment amplitude gate ──────────────────────────────────────────────

def swing_amp_shw(frames: list, anchors, fps: float) -> tuple[float, float]:
    """
    Compute wrist_y amplitude in swing segment [address, n_eff].
    Returns (amp_shw, shw_px) — amplitude in shoulder-width units, raw SHW px.

    SHW = shoulder width at address frame.
    """
    n = len(frames)
    address = anchors.address
    n_eff   = anchors.first_swing_end if (0 < anchors.first_swing_end < n) else n

    # SHW from address frame
    addr_fr = frames[min(address, n-1)]
    p = addr_fr.get("persons", [])
    kps = p[0].get("keypoints", {}) if p else {}
    shw = shoulder_width_from_kps(kps)
    if shw is None:
        # fallback: scan ±5 frames around address
        for di in range(1, 6):
            for off in [di, -di]:
                fi = address + off
                if 0 <= fi < n:
                    p2 = frames[fi].get("persons", [])
                    k2 = p2[0].get("keypoints", {}) if p2 else {}
                    shw = shoulder_width_from_kps(k2)
                    if shw: break
            if shw: break
    if shw is None or shw < MIN_SHW:
        return 0.0, 0.0

    ys = extract_wrist_ys(frames)
    seg = ys[address:n_eff]
    amp_px = float(seg.max() - seg.min()) if len(seg) > 1 else 0.0
    return round(amp_px / shw, 3), round(shw, 1)


# ── Window tilt median ────────────────────────────────────────────────────────

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


# ── Confidence Ledger ─────────────────────────────────────────────────────────

def confidence_ledger_check(
    clip_id: str,
    camera_view: str,
    top_conf: float,
    amp_shw: float,
    window_valid: int,
) -> tuple[str, str]:
    """Returns (status, reason). status: "RUN" | "SILENT"."""

    # 1. Camera view gate — DTL and uncertain both silenced for face-on features
    if camera_view not in ("face_on", "face-on", ""):
        return "SILENT", f"camera_gate: camera_view={camera_view}"

    # 2. top_conf gate (provisional threshold 0.50)
    if top_conf < CONF_THR:
        return "SILENT", (f"phase_detection_low_confidence: "
                          f"top_conf={top_conf:.3f} < {CONF_THR} (provisional)")

    # 3. Swing-segment amplitude gate (0.8 × SHW)
    if amp_shw < AMP_SHW_THR:
        return "SILENT", (f"amp_gate: swing_amp={amp_shw:.3f} SHW < {AMP_SHW_THR} SHW "
                          f"(no real backswing detected)")

    # 4. Window valid frames
    if window_valid < MIN_VALID_WINDOW:
        return "SILENT", f"window_too_small: valid_frames={window_valid} < {MIN_VALID_WINDOW}"

    return "RUN", "ok"


# ── Diagnose one clip ─────────────────────────────────────────────────────────

def diagnose_clip(clip_id: str, label: str, camera_view: str, baseline: dict) -> dict:
    kp_json = load_kp_json(clip_id)
    if kp_json is None:
        return dict(clip_id=clip_id, label=label, camera_view=camera_view,
                    status="SILENT", reason="kp_cache not found",
                    top_idx=None, top_conf=None, b_path=None,
                    amp_shw=None, shw_px=None, tilt_deg=None, confidence=None,
                    window_valid=0)

    frames = kp_json.get("frames", [])

    # B layer
    top_idx, top_conf, anchors, n, fps = run_b_layer(kp_json, camera_view)

    # Swing-segment amplitude (SHW-normalized)
    amp_shw, shw_px = swing_amp_shw(frames, anchors, fps)

    # Tilt window
    tilt_med, tilt_window, window_valid = (
        (None, [], 0) if top_idx is None
        else window_tilt_median(frames, top_idx)
    )

    status, reason = confidence_ledger_check(
        clip_id, camera_view, top_conf, amp_shw, window_valid
    )

    confidence = None
    if status == "RUN":
        if tilt_med is None:
            status, reason = "SILENT", "tilt_failed: no valid window tilts"
        else:
            confidence = query_tilt(tilt_med, baseline)["confidence"]

    return dict(
        clip_id=clip_id, label=label, camera_view=camera_view,
        n_frames=n, top_idx=int(top_idx) if top_idx is not None else None,
        top_conf=round(top_conf, 3) if top_conf is not None else None,
        b_path=("FIND_PEAKS" if (top_conf is not None and top_conf > 0) else "FALLBACK"),
        amp_shw=float(amp_shw), shw_px=float(shw_px),
        tilt_deg=round(tilt_med, 2) if tilt_med is not None else None,
        window_tilts=[round(t, 1) for t in tilt_window],
        window_valid=window_valid,
        status=status, reason=reason,
        confidence=confidence,
    )


# ── Positive control (clip_016 gate1 human-confirmed values) ──────────────────

def diagnose_clip016_controls(baseline: dict) -> list[dict]:
    results = []
    for side, tilt_val, exp in [("left_ERROR", +29.1, "Confirmed"),
                                 ("right_OK",    -6.8,  "None")]:
        r = query_tilt(tilt_val, baseline)
        results.append(dict(
            clip_id=f"clip_016/{side}", label="positive_control",
            camera_view="face-on", n_frames="N/A",
            top_idx="N/A", top_conf="N/A", b_path="N/A",
            amp_shw="N/A", shw_px="N/A",
            tilt_deg=tilt_val, window_tilts=[], window_valid="N/A",
            status="RUN",
            reason=f"gate1 human-confirmed tilt={tilt_val:+.1f}°, expected={exp}",
            confidence=r["confidence"],
        ))
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    baseline = load_baseline()
    band_lo  = baseline["reference_band"]["band_lower_deg"]
    band_hi  = baseline["reference_band"]["band_upper_deg"]
    mu       = baseline["reference_band"]["center_mu_deg"]

    print(f"\n{'='*80}")
    print(f"  关卡3 v3 — 诊断 + 不误报验证")
    print(f"  top检测v3: B层单一事实源 + conf≥{CONF_THR}(provisional) + amp≥{AMP_SHW_THR}SHW")
    print(f"  Baseline: mu={mu}°  band=[{band_lo}, {band_hi}]°")
    print(f"  Ledger: None≤+5° | +5<Possible≤+15° | +15<Likely≤+28° | Confirmed>+28°")
    print(f"{'='*80}\n")

    # ── 1. face-on clips (no EE) ──────────────────────────────────────────────
    fo_clips = [
        ("fo-eet-1",  "eet_faceon",   "face-on"),
        ("fo-eet-2",  "eet_faceon",   "face-on"),
        ("fo-eet-3",  "eet_faceon",   "face-on"),
        ("fo-ok-1",   "labeled_ok",   "face-on"),
        ("fo-ok-2",   "labeled_ok",   "face-on"),
    ]
    # ── 2. DTL clips — camera_gate should fire ─────────────────────────────────
    dtl_clips = [
        ("dtl-eet-2", "eet_dtl", "down-the-line"),
        ("dtl-eet-3", "eet_dtl", "down-the-line"),
    ]
    # ── 3. negative controls ──────────────────────────────────────────────────
    neg_clips = [
        ("fo-eet-1-neg-setup",     "neg_no_swing",    "face-on"),
        ("fo-eet-1-neg-truncated", "neg_top_absent",  "face-on"),
    ]

    all_results = []
    false_positives = []

    # --- face-on clips ---
    print("--- 正面杆 (预期: None / SILENT, 0 误报) ---")
    for clip_id, label, cam in fo_clips:
        r = diagnose_clip(clip_id, label, cam, baseline)
        all_results.append(r)
        _print_row(r)
        if r["status"] == "RUN" and r["confidence"] in ("Likely", "Confirmed"):
            false_positives.append(r)

    # --- DTL clips ---
    print("\n--- DTL 杆 (预期: SKIPPED(camera_gate)) ---")
    for clip_id, label, cam in dtl_clips:
        r = diagnose_clip(clip_id, label, cam, baseline)
        all_results.append(r)
        _print_row(r, dtl=True)
        if r["status"] == "RUN" and r["confidence"] in ("Likely", "Confirmed"):
            false_positives.append(r)

    # --- negative controls ---
    print("\n--- 阴性对照 (预期: 全部 SILENT, 逐段报告拦截门) ---")
    for clip_id, label, cam in neg_clips:
        r = diagnose_clip(clip_id, label, cam, baseline)
        all_results.append(r)
        _print_neg_row(r)

    # --- positive control ---
    print("\n--- 阳性对照 (clip_016, 预期 ERROR=Confirmed / OK=None) ---")
    ctrl_results = diagnose_clip016_controls(baseline)
    for r in ctrl_results:
        print(f"  {r['clip_id']:28s}  tilt={r['tilt_deg']:+.2f}°  "
              f"confidence={r['confidence']:10s}  {r['reason']}")
        all_results.append(r)

    # ── summary ──────────────────────────────────────────────────────────────
    fo_results  = all_results[:len(fo_clips)]
    dtl_results = all_results[len(fo_clips):len(fo_clips)+len(dtl_clips)]
    neg_results = all_results[len(fo_clips)+len(dtl_clips):len(fo_clips)+len(dtl_clips)+len(neg_clips)]

    ctrl_err = next(r for r in ctrl_results if "ERROR" in r["clip_id"])
    ctrl_ok  = next(r for r in ctrl_results if "right_OK" in r["clip_id"])

    n_fo_silent = sum(1 for r in fo_results if r["status"] == "SILENT")
    n_fo_none   = sum(1 for r in fo_results if r["confidence"] == "None")
    n_dtl_gated = sum(1 for r in dtl_results if "camera_gate" in r.get("reason",""))
    n_neg_silent = sum(1 for r in neg_results if r["status"] == "SILENT")
    n_fp = len(false_positives)

    err_ok = ctrl_err["confidence"] == "Confirmed"
    ok_ok  = ctrl_ok["confidence"]  == "None"

    print(f"\n{'='*80}")
    print("  总结")
    print(f"{'='*80}")
    print(f"  正面杆 {len(fo_clips)} 条: SILENT={n_fo_silent}  None={n_fo_none}  误报(Likely+)={n_fp}  {'✅ 无误报' if n_fp==0 else '❌ 存在误报'}")
    print(f"  DTL  杆 {len(dtl_clips)} 条: camera_gate拦截={n_dtl_gated}/{len(dtl_clips)}  {'✅' if n_dtl_gated==len(dtl_clips) else '❌'}")
    print(f"  阴性对照 {len(neg_clips)} 条: 全部SILENT={n_neg_silent}/{len(neg_clips)}  {'✅' if n_neg_silent==len(neg_clips) else '❌'}")
    print(f"\n  阳性对照:")
    print(f"    clip_016/left_ERROR  → {ctrl_err['confidence']:10s}  {'✅' if err_ok else '❌'}")
    print(f"    clip_016/right_OK    → {ctrl_ok['confidence']:10s}  {'✅' if ok_ok else '❌'}")

    # fo-eet-1 specific check (must hit fr185±2)
    fo1 = next((r for r in fo_results if r["clip_id"]=="fo-eet-1"), None)
    fo1_top_ok = (fo1 is not None and fo1["top_idx"] is not None
                  and abs(fo1["top_idx"] - 185) <= 2)
    print(f"\n  fo-eet-1 top GT check: top={fo1['top_idx'] if fo1 else 'N/A'}  "
          f"(GT=fr185, ±2)  {'✅' if fo1_top_ok else '❌'}")

    verdict_items = [
        n_fp == 0,           # no false positives
        fo1_top_ok,          # fo-eet-1 top hit
        err_ok, ok_ok,       # positive controls
        n_dtl_gated == len(dtl_clips),   # DTL gate
        n_neg_silent == len(neg_clips),  # negative controls
    ]
    verdict = "PASS" if all(verdict_items) else "FAIL"
    print(f"\n  关卡3 v3 总裁决: {verdict}")

    if false_positives:
        print("\n  ⚠️ 误报详情:")
        for r in false_positives:
            print(f"    {r['clip_id']}: tilt={r['tilt_deg']:+.2f}°  conf={r['confidence']}")

    # ── detailed negative gate report ─────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  阴性对照详表 (阈值定版依据)")
    print(f"{'='*80}")
    for r in neg_results:
        tc = f"{r['top_conf']:.3f}" if isinstance(r['top_conf'], float) else str(r['top_conf'])
        am = f"{r['amp_shw']:.3f}SHW" if isinstance(r['amp_shw'], float) else str(r['amp_shw'])
        gate = _which_gate(r["reason"])
        print(f"  {r['clip_id']}")
        print(f"    top_conf={tc}  amp={am}  gate拦截={gate}")
        print(f"    reason: {r['reason']}")

    # JSON output
    out_path = PROJ / "output/gate3_no_fp/gate3_results_v3.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(
        version="v3",
        top_detection="B_layer_single_source+conf>=0.50+amp>=0.8SHW",
        conf_thr=CONF_THR, amp_shw_thr=AMP_SHW_THR,
        conf_thr_status="provisional",
        summary=dict(
            n_fo=len(fo_clips), n_fo_silent=n_fo_silent, n_fo_none=n_fo_none,
            n_dtl=len(dtl_clips), n_dtl_camera_gated=n_dtl_gated,
            n_neg=len(neg_clips), n_neg_silent=n_neg_silent,
            n_false_positives=n_fp,
            positive_control_error=ctrl_err["confidence"],
            positive_control_ok=ctrl_ok["confidence"],
            fo1_top=fo1["top_idx"] if fo1 else None,
            fo1_top_gt=185, fo1_top_ok=fo1_top_ok,
            verdict=verdict,
        ),
        results=all_results + ctrl_results,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  JSON → {out_path}\n")


# ── print helpers ─────────────────────────────────────────────────────────────

def _print_row(r: dict, dtl: bool = False):
    tc   = f"{r['top_conf']:.3f}" if isinstance(r['top_conf'], float) else "N/A"
    am   = f"{r['amp_shw']:.3f}" if isinstance(r['amp_shw'], float) else "N/A"
    tilt = f"{r['tilt_deg']:+.2f}°" if r['tilt_deg'] is not None else "  N/A  "
    top  = f"fr{r['top_idx']}" if isinstance(r['top_idx'], int) else "None"
    if dtl:
        diag = f"SKIPPED(camera_gate)" if "camera_gate" in r.get("reason","") else r["reason"]
    else:
        diag = (r["confidence"] if r["status"] == "RUN"
                else f"SILENT({r['reason'][:50]})")
    print(f"  {r['clip_id']:28s}  top={top:>5s}  conf={tc}  "
          f"amp={am}SHW  tilt={tilt}  {diag}")


def _print_neg_row(r: dict):
    tc = f"{r['top_conf']:.3f}" if isinstance(r['top_conf'], float) else "N/A"
    am = f"{r['amp_shw']:.3f}" if isinstance(r['amp_shw'], float) else "N/A"
    gate = _which_gate(r["reason"])
    status_str = "✅ SILENT" if r["status"] == "SILENT" else "❌ RUN (should be SILENT)"
    print(f"  {r['clip_id']:32s}  conf={tc}  amp={am}SHW  "
          f"gate={gate}  {status_str}")


def _which_gate(reason: str) -> str:
    if "camera_gate"   in reason: return "camera_gate"
    if "phase_detection_low_confidence" in reason: return "conf_gate"
    if "amp_gate"      in reason: return "amp_gate"
    if "window_too_small" in reason: return "window_gate"
    if "tilt_failed"   in reason: return "tilt_gate"
    if "kp_cache"      in reason: return "kp_missing"
    return "other"


if __name__ == "__main__":
    main()
