#!/usr/bin/env python3
"""
run_cue_renders.py — CUE-001 关卡B 验收运行脚本

从已有 kp_cache + gate3 v3 诊断结果生成 Reverse Pivot cue 图。
产出:
  output/cue_renders/reverse_pivot/<clip_id>_top_cue.jpg       彩色 cue
  output/cue_renders/reverse_pivot/<clip_id>_top_cue_gray.jpg  灰度自查
  preview/cue_renders/reverse_pivot/  (Windows Desktop 同步)

验收用 clips (关卡B 四项标准):
  1. clip_016/left_ERROR  (Confirmed, +29.1°)  → 完整 cue 图
  2. fo-eet-1/2/3, fo-ok-1/2               → None → 中性帧
  3. fo-eet-1-neg-setup, neg-truncated     → SILENT → 重拍引导
  4. clip_016/right_OK (-6.8°)              → None → 中性帧
"""
import sys, json, math, statistics
from pathlib import Path
import numpy as np
import cv2

PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ))

from engine.b_phase.swing_phase import SwingPhaseEngine
from engine.reference_flywheel import load_baseline, query_tilt
from engine.features.triline_geometry import _safe_pt, _lateral_tilt_deg
from cue_renderer import ReversePivotPayload, render_reverse_pivot_cue

# ── paths ─────────────────────────────────────────────────────────────────────
OUT_DIR  = PROJ / "output/cue_renders/reverse_pivot"
DESK_DIR = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/cue_renders/reverse_pivot")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DESK_DIR.mkdir(parents=True, exist_ok=True)

KP_GUARD  = 0.30
MIN_SHW   = 15.0
WINDOW    = 5
CONF_THR  = 0.50
AMP_SHW_THR = 0.8

# clip_016 source video
CLIP016_VIDEO = Path("/mnt/c/Users/jason/Zening/Swingcue/教学视频") / \
    "115. 好的旋转方向是：身体按照顺序自然旋转。#高尔夫 #golf #golfswing #高尔夫课程 #高尔夫挥杆 #golflesson #golflife.mp4"


# ── kp helpers ────────────────────────────────────────────────────────────────

def load_kp(clip_id: str) -> dict | None:
    for sub in ["negatives", "batch3", "batch2", "batch1", ""]:
        p = (PROJ/"engine/kp_cache"/sub/f"{clip_id}.json") if sub \
            else (PROJ/"engine/kp_cache"/f"{clip_id}.json")
        if p.exists(): return json.load(open(p))
    return None


def get_fps(kp): return float(kp.get("fps", kp.get("stats",{}).get("source_fps", 30.0)))
def get_fi(fr):  return fr.get("frame_idx", fr.get("frame", 0))


# ── minimal measurement for B-layer ──────────────────────────────────────────

class _M:
    def __init__(self, fi, wx, wy, q="ok"):
        self.frame_idx=fi; self._wx=wx; self._wy=wy
        self.measurement_quality=q; self.keypoints={}; self.confidences={}
    def wrist_mid(self):
        return None if (self._wx is None or self._wy is None) else (self._wx, self._wy)


def run_b_layer(kp_json):
    frames = kp_json["frames"]; fps = get_fps(kp_json); n = len(frames)
    ml = []
    for fr in frames:
        fi = get_fi(fr); p = fr.get("persons",[]); kps = p[0].get("keypoints",{}) if p else {}
        lw = _safe_pt(kps,"left_wrist"); rw = _safe_pt(kps,"right_wrist")
        pts = [pt for pt in [lw,rw] if pt]
        if pts:
            wx = sum(pt[0] for pt in pts)/len(pts); wy = sum(pt[1] for pt in pts)/len(pts)
            ml.append(_M(fi,wx,wy,"ok"))
        else:
            ml.append(_M(fi,None,None,"bad"))
    engine = SwingPhaseEngine()
    _, anchors = engine.run(ml, fps, angle="face-on")
    return anchors, n, fps


def shoulder_width_at(frames, fi, n):
    fr = frames[min(fi, n-1)]
    kps = (fr.get("persons",[{}]) or [{}])[0].get("keypoints",{})
    ls = _safe_pt(kps,"left_shoulder"); rs = _safe_pt(kps,"right_shoulder")
    if ls and rs:
        w = math.hypot(rs[0]-ls[0], rs[1]-ls[1])
        return w if w >= MIN_SHW else None
    return None


def swing_amp_shw(frames, anchors, n, fps):
    from scipy.signal import savgol_filter
    ys = np.full(n, np.nan)
    for fr in frames:
        fi = get_fi(fr); p = fr.get("persons",[]); kps = p[0].get("keypoints",{}) if p else {}
        lw = _safe_pt(kps,"left_wrist"); rw = _safe_pt(kps,"right_wrist")
        pts = [pt for pt in [lw,rw] if pt]
        if pts: ys[fi] = min(pt[1] for pt in pts)
    idx = np.arange(n); nans = np.isnan(ys)
    if not nans.all(): ys[nans] = np.interp(idx[nans], idx[~nans], ys[~nans])
    shw = shoulder_width_at(frames, anchors.address, n) or 100.0
    n_eff = anchors.first_swing_end if 0 < anchors.first_swing_end < n else n
    seg = ys[anchors.address:n_eff]
    amp = float(seg.max()-seg.min()) if len(seg)>1 else 0.0
    return amp/shw, shw


def tilt_from_kps(kps):
    ls=_safe_pt(kps,"left_shoulder"); rs=_safe_pt(kps,"right_shoulder")
    lh=_safe_pt(kps,"left_hip");     rh=_safe_pt(kps,"right_hip")
    if not (ls and rs and lh and rh): return None
    shw = math.hypot(rs[0]-ls[0],rs[1]-ls[1])
    if shw < MIN_SHW: return None
    sx=(ls[0]+rs[0])/2; sy=(ls[1]+rs[1])/2
    hx=(lh[0]+rh[0])/2; hy=(lh[1]+rh[1])/2
    return _lateral_tilt_deg((hx,hy),(sx,sy))


def window_tilt_median(frames, top_idx, n):
    tilts=[]; valid=0
    for i in range(max(0,top_idx-WINDOW), min(n,top_idx+WINDOW+1)):
        p=frames[i].get("persons",[]); kps=p[0].get("keypoints",{}) if p else {}
        valid+=1; t=tilt_from_kps(kps)
        if t is not None: tilts.append(t)
    return (statistics.median(tilts) if tilts else None), valid


# ── extract hip_mid and shoulder_mid from top frame ──────────────────────────

def extract_anchors(frames, top_idx, n):
    fr = frames[min(top_idx, n-1)]
    kps = (fr.get("persons",[{}]) or [{}])[0].get("keypoints",{})
    lh=_safe_pt(kps,"left_hip"); rh=_safe_pt(kps,"right_hip")
    ls=_safe_pt(kps,"left_shoulder"); rs=_safe_pt(kps,"right_shoulder")
    hip_mid = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2) if (lh and rh) else None
    sho_mid = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2) if (ls and rs) else None
    return hip_mid, sho_mid, kps


# ── extract top frame BGR from video ─────────────────────────────────────────

def grab_frame_bgr(video_path: Path, frame_idx: int,
                   x_crop: tuple = None) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, bgr = cap.read(); cap.release()
    if not ok: return None
    if x_crop:
        bgr = bgr[:, x_crop[0]:x_crop[1]]
    return bgr


# ── camera_view gate ──────────────────────────────────────────────────────────

def camera_view_ok(clip_id: str) -> bool:
    rec_path = PROJ/"engine/layer0/records"/f"{clip_id}.json"
    if rec_path.exists():
        rec = json.load(open(rec_path))
        if rec.get("verdict") == "REJECT": return False
        cv = rec.get("camera_view","")
        if cv and cv not in ("face_on","face-on"): return False
    return True


# ── diagnose one clip and build payload ───────────────────────────────────────

def diagnose_and_payload(clip_id: str, baseline: dict,
                         video_path: Path = None,
                         x_crop: tuple = None,
                         override_tilt: float = None,
                         override_confidence: str = None) -> ReversePivotPayload | None:
    """
    Build ReversePivotPayload from kp_cache + optional video frame.
    override_tilt/confidence: for clip_016 which uses gate1 human-confirmed values.
    """
    kp_json = load_kp(clip_id)
    if kp_json is None:
        print(f"  {clip_id}: kp_cache not found")
        return None

    frames = kp_json["frames"]; n = len(frames)
    anchors, n, fps = run_b_layer(kp_json)
    top_idx = anchors.top; top_conf = anchors.top_conf

    # Camera gate
    if not camera_view_ok(clip_id):
        print(f"  {clip_id}: camera_gate SILENT")
        confidence = "SILENT"
        top_idx = max(0, n//3)
    elif override_confidence:
        confidence = override_confidence
    elif top_conf < CONF_THR:
        confidence = "SILENT"
    else:
        amp_shw, _ = swing_amp_shw(frames, anchors, n, fps)
        if amp_shw < AMP_SHW_THR:
            confidence = "SILENT"
        else:
            tilt_med, valid = window_tilt_median(frames, top_idx, n)
            if valid < 3 or tilt_med is None:
                confidence = "SILENT"
            else:
                confidence = query_tilt(tilt_med, baseline)["confidence"]

    # Get tilt value
    if override_tilt is not None:
        tilt_deg = override_tilt
    else:
        tilt_med, _ = window_tilt_median(frames, top_idx, n)
        tilt_deg = tilt_med if tilt_med is not None else 0.0

    # Extract geometry
    hip_mid, shoulder_mid, top_kps = extract_anchors(frames, top_idx, n)

    # Fallback geometry if keypoints missing
    h_ref = 720; w_ref = 400
    if hip_mid is None:      hip_mid      = (w_ref//2, int(h_ref*0.55))
    if shoulder_mid is None: shoulder_mid = (w_ref//2, int(h_ref*0.35))

    # Get top frame BGR
    frame_bgr = None
    if video_path and video_path.exists():
        # For full-video clips, compute actual frame from kp frame_idx
        total_frames_video = None
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        if total_frames_video:
            actual_fr = int(top_idx * total_frames_video / max(n-1,1))
        else:
            actual_fr = top_idx
        frame_bgr = grab_frame_bgr(video_path, actual_fr, x_crop)

    if frame_bgr is None:
        # Use a grey placeholder
        frame_bgr = np.full((720,400,3), 40, dtype=np.uint8)

    return ReversePivotPayload(
        fault_id="reverse_pivot",
        confidence=confidence,
        tilt_deg=tilt_deg,
        top_frame_idx=top_idx,
        hip_mid=hip_mid,
        shoulder_mid=shoulder_mid,
        band_lower_deg=-18.8,
        band_upper_deg=+5.0,
        frame_bgr=frame_bgr,
        skeleton_kps=top_kps,
        clip_id=clip_id,
    )


# ── clip_016: split-screen, no kp_cache → use gate1 GT values ────────────────

def make_clip016_payload(side: str, tilt_gt: float, confidence_gt: str,
                         baseline: dict, kp_label: str) -> ReversePivotPayload | None:
    """
    Build payload for clip_016 left/right from gate1 human-confirmed values.
    Extracts the top frame from the split-screen video.
    """
    if not CLIP016_VIDEO.exists():
        print(f"  clip_016/{side}: source video not found")
        return None

    # Open video to get dimensions and frame count
    cap = cv2.VideoCapture(str(CLIP016_VIDEO))
    if not cap.isOpened():
        print(f"  clip_016/{side}: cannot open video")
        return None
    w_full = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nf     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    half = w_full // 2
    x_crop = (0, half) if side == "left" else (half, w_full)
    # Top is ~60-70% into swing; estimate at 55% of clip
    est_top_fr = int(nf * 0.55)
    frame_bgr = grab_frame_bgr(CLIP016_VIDEO, est_top_fr, x_crop)
    if frame_bgr is None:
        frame_bgr = np.full((h, half, 3), 40, dtype=np.uint8)

    fh, fw = frame_bgr.shape[:2]
    # Use approximate anatomical positions (center of frame)
    hip_mid      = (fw//2, int(fh * 0.60))
    shoulder_mid = (fw//2, int(fh * 0.38))
    # Adjust shoulder_mid for tilt: +29.1° means shoulder is ~29° to target side
    # In face-on: target = screen right. shift x proportionally.
    dist = abs(shoulder_mid[1] - hip_mid[1])
    tilt_rad = math.radians(tilt_gt)
    sho_x = int(hip_mid[0] + dist * math.sin(tilt_rad))
    sho_y = int(hip_mid[1] - dist * math.cos(tilt_rad))
    shoulder_mid = (sho_x, sho_y)

    return ReversePivotPayload(
        fault_id="reverse_pivot",
        confidence=confidence_gt,
        tilt_deg=tilt_gt,
        top_frame_idx=est_top_fr,
        hip_mid=hip_mid,
        shoulder_mid=shoulder_mid,
        band_lower_deg=-18.8,
        band_upper_deg=+5.0,
        frame_bgr=frame_bgr,
        skeleton_kps={},
        clip_id=f"clip_016_{side}",
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    baseline = load_baseline()
    VIDEO_DIR = Path("/mnt/c/Users/jason/Zening/Swingcue/Video")

    results = []

    # ── Group 1: face-on EET + OK clips (expect None) ─────────────────────────
    fo_clips = [
        ("fo-eet-1", VIDEO_DIR/"fo-eet-1.mp4"),
        ("fo-eet-2", VIDEO_DIR/"fo-eet-2.mp4"),
        ("fo-eet-3", VIDEO_DIR/"fo-eet-3.mp4"),
        ("fo-ok-1",  VIDEO_DIR/"fo-ok-1.mp4"),
        ("fo-ok-2",  VIDEO_DIR/"fo-ok-2.mp4"),
    ]
    print("\n--- face-on 正常杆 (预期 None → 中性帧) ---")
    for clip_id, vpath in fo_clips:
        p = diagnose_and_payload(clip_id, baseline, vpath)
        if p:
            outs = render_reverse_pivot_cue(p, OUT_DIR)
            for k, path in outs.items():
                import shutil
                shutil.copy2(path, DESK_DIR / path.name)
            print(f"  {clip_id:20s}  conf={p.confidence:12s}  tilt={p.tilt_deg:+.1f}°  → {list(outs.values())[0].name}")
            results.append((clip_id, p.confidence, p.tilt_deg))

    # ── Group 2: negative controls (expect SILENT) ────────────────────────────
    neg_clips = [
        ("fo-eet-1-neg-setup",     Path(PROJ/"tests/negatives/fo-eet-1-neg-setup.mp4")),
        ("fo-eet-1-neg-truncated", Path(PROJ/"tests/negatives/fo-eet-1-neg-truncated.mp4")),
    ]
    print("\n--- 阴性对照 (预期 SILENT → 重拍引导) ---")
    for clip_id, vpath in neg_clips:
        p = diagnose_and_payload(clip_id, baseline, vpath)
        if p:
            outs = render_reverse_pivot_cue(p, OUT_DIR)
            for k, path in outs.items():
                import shutil
                shutil.copy2(path, DESK_DIR / path.name)
            print(f"  {clip_id:28s}  conf={p.confidence:12s}  → {list(outs.values())[0].name}")
            results.append((clip_id, p.confidence, None))

    # ── Group 3: clip_016 left=Confirmed (+29.1°) / right=None (-6.8°) ────────
    print("\n--- clip_016 阳性对照 (left=Confirmed / right=None) ---")
    for side, tilt_gt, conf_gt in [("left",  +29.1, "Confirmed"),
                                    ("right",  -6.8,  "None")]:
        p = make_clip016_payload(side, tilt_gt, conf_gt, baseline, f"clip_016_{side}")
        if p:
            outs = render_reverse_pivot_cue(p, OUT_DIR)
            for k, path in outs.items():
                import shutil
                shutil.copy2(path, DESK_DIR / path.name)
            print(f"  clip_016/{side:6s}  conf={p.confidence:12s}  tilt={p.tilt_deg:+.1f}°  → {list(outs.values())[0].name}")
            results.append((f"clip_016/{side}", p.confidence, p.tilt_deg))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  CUE-001 关卡B 渲染汇总")
    print(f"{'='*70}")
    print(f"  {'clip':<30s}  {'confidence':<12s}  {'tilt':>8s}  预期")
    print(f"  {'-'*62}")
    expected = {
        "fo-eet-1":"None","fo-eet-2":"None","fo-eet-3":"None",
        "fo-ok-1":"None","fo-ok-2":"None",
        "fo-eet-1-neg-setup":"SILENT","fo-eet-1-neg-truncated":"SILENT",
        "clip_016/left":"Confirmed","clip_016/right":"None",
    }
    all_pass = True
    for clip_id, conf, tilt in results:
        exp = expected.get(clip_id, "?")
        ok = "✅" if conf==exp else "❌"
        if conf != exp: all_pass = False
        tilt_s = f"{tilt:+.1f}°" if tilt is not None else "  N/A"
        print(f"  {clip_id:<30s}  {conf:<12s}  {tilt_s:>8s}  {ok} (预期{exp})")

    verdict = "PASS" if all_pass else "FAIL"
    print(f"\n  关卡B 渲染验收: {verdict}")
    print(f"\n  彩色+灰度图输出目录:")
    print(f"    {OUT_DIR}")
    print(f"    Windows: C:\\Users\\jason\\Desktop\\rtmpose_results\\preview\\cue_renders\\reverse_pivot\\")
    print()


if __name__ == "__main__":
    main()
