#!/usr/bin/env python3
"""
diag_D2_render.py — D2 叠加渲染

fo-eet-1: 每5帧一张 (228帧 → 46张)
fo-eet-2: 每12帧一张 (123帧 → ~10张对照)

每张: 原始画面 + 全部17关键点 + 骨架连线 + 左/右腕高亮 + bbox(若kp_cache无bbox则用肩宽估算) + 帧号 + 腕坐标+分数
"""
import sys, json, math, cv2, shutil
from pathlib import Path
import numpy as np

PROJ    = Path("/home/jason/projects/swingcue-postest")
OUT_DIR = PROJ / "output/diag_D2_renders"
OUT_WIN = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/diag_D2")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_WIN.mkdir(parents=True, exist_ok=True)

KP_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle",
]

BONES = [
    ("nose","left_eye"),("nose","right_eye"),
    ("left_eye","left_ear"),("right_eye","right_ear"),
    ("left_shoulder","right_shoulder"),
    ("left_shoulder","left_elbow"),("left_elbow","left_wrist"),
    ("right_shoulder","right_elbow"),("right_elbow","right_wrist"),
    ("left_shoulder","left_hip"),("right_shoulder","right_hip"),
    ("left_hip","right_hip"),
    ("left_hip","left_knee"),("left_knee","left_ankle"),
    ("right_hip","right_knee"),("right_knee","right_ankle"),
]

KP_COLOR = (0, 220, 0)       # green
WRIST_L_COLOR = (0, 60, 255) # red  = left_wrist
WRIST_R_COLOR = (255, 60, 0) # blue = right_wrist
BONE_COLOR  = (200, 200, 0)  # yellow
ZONE_COLOR  = (0, 200, 200)  # cyan (zone boundary indicator)


def render_frame(bgr: np.ndarray, kps: dict, frame_idx: int,
                 n_total: int, zone_lo: int, zone_hi: int,
                 global_top: int) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]

    # Zone bar at top
    in_zone = zone_lo <= frame_idx <= zone_hi
    bar_col = (0, 180, 0) if in_zone else (60, 60, 60)
    cv2.rectangle(out, (0, 0), (w, 12), bar_col, -1)
    cv2.putText(out, f"fr{frame_idx}/{n_total-1} {'[IN ZONE]' if in_zone else '[OUT OF ZONE]'}  top@fr{global_top}",
                (4, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1, cv2.LINE_AA)

    # Top frame indicator
    if frame_idx == global_top:
        cv2.rectangle(out, (0, 0), (w-1, h-1), (0, 0, 255), 4)

    # Bones
    for a, b in BONES:
        pa = kps.get(a, {}); pb = kps.get(b, {})
        if pa.get("score", 0) >= 0.30 and pb.get("score", 0) >= 0.30:
            p1 = (int(pa["x"]), int(pa["y"]))
            p2 = (int(pb["x"]), int(pb["y"]))
            cv2.line(out, p1, p2, BONE_COLOR, 1, cv2.LINE_AA)

    # Keypoints
    for name in KP_NAMES:
        kp = kps.get(name, {})
        if not kp or kp.get("score", 0) < 0.05: continue
        x, y, sc = int(kp["x"]), int(kp["y"]), kp["score"]
        if name == "left_wrist":
            col, r = WRIST_L_COLOR, 7
        elif name == "right_wrist":
            col, r = WRIST_R_COLOR, 7
        else:
            col, r = KP_COLOR, 3
        alpha = min(1.0, sc)
        cv2.circle(out, (x, y), r, col, -1, cv2.LINE_AA)

    # Wrist info overlay
    lw = kps.get("left_wrist",  {}); rw = kps.get("right_wrist", {})
    lwy = lw.get("y"); rwy = rw.get("y")
    wy_min = min(v for v in [lwy, rwy] if v) if (lwy or rwy) else None
    lw_s = f"LW({lw.get('x',0):.0f},{lw.get('y',0):.0f}) sc={lw.get('score',0):.2f}" if lw else "LW:?"
    rw_s = f"RW({rw.get('x',0):.0f},{rw.get('y',0):.0f}) sc={rw.get('score',0):.2f}" if rw else "RW:?"
    wy_s = f"wrist_y_min={wy_min:.1f}" if wy_min else "wrist_y:?"
    cv2.putText(out, lw_s,  (4, h-46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, WRIST_L_COLOR, 1, cv2.LINE_AA)
    cv2.putText(out, rw_s,  (4, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, WRIST_R_COLOR, 1, cv2.LINE_AA)
    cv2.putText(out, wy_s,  (4, h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1, cv2.LINE_AA)

    return out


def render_clip(clip_id: str, batch: str, step: int, label: str):
    vid_path = PROJ / "input" / f"{clip_id}.mp4"
    kp_json  = json.load(open(PROJ / f"engine/kp_cache/{batch}/{clip_id}.json"))
    frames   = kp_json["frames"]
    n        = len(frames)
    zone_lo  = int(n * 0.15)
    zone_hi  = int(n * 0.65)

    # Global wrist_y minimum (true top)
    best_fi, best_wy = -1, 9999.0
    for fr in frames:
        fi = fr["frame_idx"]
        p  = fr.get("persons", [])
        if not p: continue
        kps = p[0]["keypoints"]
        lw  = kps.get("left_wrist",{}); rw = kps.get("right_wrist",{})
        ly  = lw.get("y"); ry = rw.get("y")
        wy  = min(v for v in [ly,ry] if v is not None) if (ly or ry) else None
        if wy and wy < best_wy:
            best_wy, best_fi = wy, fi

    cap = cv2.VideoCapture(str(vid_path))
    total_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_sub  = OUT_DIR / label
    out_win2 = OUT_WIN / label
    out_sub.mkdir(parents=True, exist_ok=True)
    out_win2.mkdir(parents=True, exist_ok=True)

    rendered = 0
    for fi in range(0, n, step):
        if fi >= n: break
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, bgr = cap.read()
        if not ok: continue

        fr  = frames[fi]
        p   = fr.get("persons", [])
        kps = p[0]["keypoints"] if p else {}

        ann = render_frame(bgr, kps, fi, n, zone_lo, zone_hi, best_fi)
        fname = f"{label}_fr{fi:04d}.jpg"
        cv2.imwrite(str(out_sub / fname), ann, [cv2.IMWRITE_JPEG_QUALITY, 88])
        shutil.copy(out_sub / fname, out_win2 / fname)
        rendered += 1
        if fi % 30 == 0:
            print(f"  {label} fr{fi}", end="\r", flush=True)

    cap.release()
    print(f"  {label}: {rendered} frames rendered  global_top=fr{best_fi}({best_wy:.0f}px)  zone=[{zone_lo},{zone_hi}]  top_in_zone={'YES' if zone_lo<=best_fi<=zone_hi else 'NO'}")


def main():
    print("D2 Rendering fo-eet-1 (every 5 frames)...")
    render_clip("fo-eet-1", "batch3", step=5,  label="fo-eet-1")

    print("D2 Rendering fo-eet-2 (every 12 frames, ~10 samples)...")
    render_clip("fo-eet-2", "batch3", step=12, label="fo-eet-2")

    print(f"\nDone → {OUT_WIN}")
    print("Legend: RED frame border = global top. Cyan header = in zone. Green header = in zone. Dark header = out of zone.")
    print("Wrist: RED dot = left_wrist, BLUE dot = right_wrist")


if __name__ == "__main__":
    main()
