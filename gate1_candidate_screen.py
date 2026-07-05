#!/usr/bin/env python3
"""
gate1_candidate_screen.py
剩余候选配对筛查 — 全量帧 + 并列渲染

对 medium-signal 候选(clip_099/126/128/124):
  1. 全量帧 RTMPose 提取
  2. 全量帧 top 检测(wrist-Y 最低 = 腕位最高)
  3. top±5 窗口中位数 tilt
  4. 渲染"左半top + 右半top"并列图(2048px宽)

判断标准 (供人工验收):
  - 同一个人 (左右体型/发型/服装一致)?
  - 都是 face-on?
  - top帧是真正的上杆顶点?
  - 对比的是身体侧倾?

注: weak-signal(diff<2°)暂不处理
"""

import sys, json, math, statistics, cv2
from pathlib import Path
import numpy as np

PROJ = Path("/home/jason/projects/swingcue-postest")
sys.path.insert(0, str(PROJ))

from engine.a_measurement.pose_pipeline import PosePipeline
from engine.features.triline_geometry import (
    _safe_pt, _lateral_tilt_deg, render_triline_frame
)

VIDEO_DIR = Path("/mnt/c/Users/jason/Zening/Swingcue/教学视频")
OUT_DIR   = PROJ / "output/gate1_triline"
OUT_WIN   = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/gate1_triline")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_WIN.mkdir(parents=True, exist_ok=True)

KP_THR  = 0.30
MIN_SHW = 15.0
NaN     = float("nan")

CANDIDATES = [
    ("clip_099", "70. 让挥杆变得更简单#高尔夫#高尔夫教学#高尔夫挥杆.mp4",            5.7),
    ("clip_126", "97. 高尔夫推杆分享⛳＃#高尔夫教学#高尔夫挥杆#江山高尔夫学院#广州高尔夫好去处#玩转高尔夫#高尔夫练习场#江山高尔夫俱乐部#魅力高尔夫.mp4", 4.1),
    ("clip_128", "99. 🏌🏻️‍♀️高尔夫学习#高尔夫教学#高尔夫挥杆#玩转高尔夫#广州高尔夫好去处#高尔夫练习场#江山高尔夫学院＃#江山高尔夫俱乐部.mp4", 3.4),
    ("clip_124", "93. 你看懂了吗？#高尔夫挥杆.mp4",                                    2.2),
]

# ─── pipeline ────────────────────────────────────────────────────────────────

JOINT_NAMES = ["nose","left_eye","right_eye","left_ear","right_ear",
               "left_shoulder","right_shoulder","left_elbow","right_elbow",
               "left_wrist","right_wrist","left_hip","right_hip",
               "left_knee","right_knee","left_ankle","right_ankle"]
KP_CONF_THR = 0.30

_body = None
def get_body():
    global _body
    if _body is None:
        pipeline = PosePipeline()
        _body = pipeline._get_body()
    return _body

def infer_frame(body, bgr):
    """Call rtmlib Body on a BGR panel, return persons list with keypoint dicts."""
    kps_arr, sc_arr = body(bgr)
    persons = []
    if kps_arr is not None and len(kps_arr) > 0:
        kps = kps_arr[0]; sc = sc_arr[0]
        kp_dict = {}
        for i, name in enumerate(JOINT_NAMES):
            score = float(sc[i]) if i < len(sc) else 0.0
            kp_dict[name] = {
                "x": float(kps[i][0]), "y": float(kps[i][1]), "score": score
            }
        persons.append({"keypoints": kp_dict})
    return {"persons": persons}

# ─── full extraction ──────────────────────────────────────────────────────────

def extract_full(video_path: Path, x0: int, x1: int, body) -> list:
    """Extract all frames from x0:x1 crop. Returns list of frame dicts."""
    cap = cv2.VideoCapture(str(video_path))
    nf  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for fi in range(nf):
        ok, bgr = cap.read()
        if not ok:
            frames.append({"persons": []})
            continue
        panel = bgr[:, x0:x1]
        fd = infer_frame(body, panel)
        fd["frame_idx"] = fi
        frames.append(fd)
        if fi % 30 == 0:
            print(f"    fr{fi}/{nf}", end="\r", flush=True)
    cap.release()
    print(f"    done {nf} frames           ")
    return frames

# ─── top detection ────────────────────────────────────────────────────────────

def find_top_idx(frames: list, front_pct: float = 0.70) -> int | None:
    """Full-frame wrist-Y minimum in front_pct portion."""
    n = len(frames)
    end = max(1, int(n * front_pct))
    best_i, best_wy = None, float("inf")
    for i in range(end):
        p = frames[i].get("persons", [])
        if not p: continue
        kps = p[0].get("keypoints", {})
        lw = _safe_pt(kps, "left_wrist");  rw = _safe_pt(kps, "right_wrist")
        wy = min(pt[1] for pt in [lw, rw] if pt) if (lw or rw) else None
        if wy is not None and wy < best_wy:
            best_wy, best_i = wy, i
    return best_i

def tilt_window_median(frames: list, top_idx: int, window: int = 5):
    """Median tilt across top±window frames."""
    n = len(frames)
    tilts = []
    for i in range(max(0, top_idx - window), min(n, top_idx + window + 1)):
        p = frames[i].get("persons", [])
        if not p: continue
        kps = p[0].get("keypoints", {})
        ls = _safe_pt(kps, "left_shoulder"); rs = _safe_pt(kps, "right_shoulder")
        lh = _safe_pt(kps, "left_hip");      rh = _safe_pt(kps, "right_hip")
        if not (ls and rs and lh and rh): continue
        sh_w = math.hypot(rs[0]-ls[0], rs[1]-ls[1])
        if sh_w < MIN_SHW: continue
        sx = (ls[0]+rs[0])/2; sy = (ls[1]+rs[1])/2
        hx = (lh[0]+rh[0])/2; hy = (lh[1]+rh[1])/2
        tilts.append(_lateral_tilt_deg((hx, hy), (sx, sy)))
    return statistics.median(tilts) if tilts else None, tilts

# ─── side-by-side render ─────────────────────────────────────────────────────

def render_side_by_side(
    vp: Path,
    frames_l: list, top_l: int,
    frames_r: list, top_r: int,
    tilt_l: float, tilt_r: float,
    pid: str,
    half_w: int,
) -> np.ndarray:
    """Read top frames from video, render triline, stitch side by side."""
    cap = cv2.VideoCapture(str(vp))
    nf  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frame(fi):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, bgr = cap.read()
        return bgr if ok else None

    bgr_l = read_frame(top_l)
    bgr_r = read_frame(top_r)
    cap.release()

    def annotate(bgr, x0, frames, top_idx, tilt, side_label):
        panel = bgr[:, x0:x0+half_w] if bgr is not None else np.zeros((300, half_w, 3), np.uint8)
        p  = frames[top_idx].get("persons", []) if top_idx is not None else []
        kps = p[0].get("keypoints", {}) if p else {}
        feat = {"shoulder_lateral_tilt": tilt, "pelvis_center_x_norm": NaN}
        label = f"{pid} {side_label} | tilt@top={tilt:+.1f}°"
        return render_triline_frame(panel, kps, feat, label=label)

    ann_l = annotate(bgr_l, 0,      frames_l, top_l, tilt_l, "LEFT")
    ann_r = annotate(bgr_r, half_w, frames_r, top_r, tilt_r, "RIGHT")

    # Resize to same height
    h = max(ann_l.shape[0], ann_r.shape[0])
    def pad_h(img, target_h):
        dh = target_h - img.shape[0]
        return cv2.copyMakeBorder(img, 0, dh, 0, 0, cv2.BORDER_CONSTANT, value=(30,30,30)) if dh > 0 else img
    ann_l = pad_h(ann_l, h)
    ann_r = pad_h(ann_r, h)

    # Divider
    div = np.full((h, 4, 3), (80, 80, 80), np.uint8)
    canvas = np.hstack([ann_l, div, ann_r])

    # Header banner
    diff = abs(tilt_l - tilt_r)
    banner_h = 36
    banner = np.zeros((banner_h, canvas.shape[1], 3), np.uint8)
    txt = f"{pid}  |  LEFT tilt={tilt_l:+.1f}  RIGHT tilt={tilt_r:+.1f}  diff={diff:.1f}  |  [人工审查: 同人? face-on? top对? 讲侧倾?]"
    cv2.putText(banner, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([banner, canvas])


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    body = get_body()
    rows = []

    for pid, src, prior_diff in CANDIDATES:
        print(f"\n{'='*60}")
        print(f"  {pid}  prior_diff={prior_diff}°")
        print(f"  {src[:60]}")
        vp = VIDEO_DIR / src

        cap = cv2.VideoCapture(str(vp))
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        nf  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        half = w // 2

        print(f"  Extracting LEFT  ({nf} frames)...")
        frames_l = extract_full(vp, 0, half, body)
        print(f"  Extracting RIGHT ({nf} frames)...")
        frames_r = extract_full(vp, half, w, body)

        top_l = find_top_idx(frames_l, front_pct=0.70)
        top_r = find_top_idx(frames_r, front_pct=0.70)

        tilt_l, tw_l = tilt_window_median(frames_l, top_l) if top_l is not None else (None, [])
        tilt_r, tw_r = tilt_window_median(frames_r, top_r) if top_r is not None else (None, [])

        diff = abs(tilt_l - tilt_r) if (tilt_l is not None and tilt_r is not None) else None

        tl_s = f"{tilt_l:+.2f}" if tilt_l is not None else "N/A"
        tr_s = f"{tilt_r:+.2f}" if tilt_r is not None else "N/A"
        df_s = f"{diff:.2f}"    if diff    is not None else "N/A"
        print(f"  LEFT:  top={top_l}  tilt_median={tl_s}  window={[round(t,1) for t in tw_l]}")
        print(f"  RIGHT: top={top_r}  tilt_median={tr_s}  window={[round(t,1) for t in tw_r]}")
        print(f"  diff={df_s}")

        # Render side-by-side
        canvas = render_side_by_side(
            vp, frames_l, top_l, frames_r, top_r,
            tilt_l or 0, tilt_r or 0, pid, half
        )
        out_path = OUT_DIR / f"candidate_{pid}_sidebyside.jpg"
        cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        import shutil
        shutil.copy(out_path, OUT_WIN / out_path.name)
        print(f"  -> {out_path.name}")

        rows.append({
            "pid": pid, "src": src, "prior_diff": prior_diff,
            "tilt_l": tilt_l, "tilt_r": tilt_r, "diff": diff,
            "top_l": top_l, "top_r": top_r,
            "window_l": [round(t,1) for t in tw_l],
            "window_r": [round(t,1) for t in tw_r],
        })

    # JSON output
    out_json = OUT_DIR / "candidate_screen_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    import shutil
    shutil.copy(out_json, OUT_WIN / out_json.name)

    # Report
    print("\n\n=== SUMMARY ===")
    print(f"{'pid':10s}  {'tilt_L':>8s}  {'tilt_R':>8s}  {'diff':>7s}")
    for r in rows:
        tl = f"{r['tilt_l']:+.2f}" if r['tilt_l'] is not None else "N/A"
        tr = f"{r['tilt_r']:+.2f}" if r['tilt_r'] is not None else "N/A"
        df = f"{r['diff']:.2f}"    if r['diff']    is not None else "N/A"
        print(f"  {r['pid']:10s}  {tl:>8s}  {tr:>8s}  {df:>7s}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
