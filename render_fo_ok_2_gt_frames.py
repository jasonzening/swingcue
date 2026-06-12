#!/usr/bin/env python3
"""
render_fo_ok_2_gt_frames.py
渲染 fo-ok-2 的 address 帧和 fr75 帧带线叠加图
输出到 Desktop/rtmpose_results/preview/batch2/peak_frames/fo-ok-2/
线条规范: FAULT_VISUAL_STANDARDS v0.2 face-on 节
  - 头部纵向参考线（address 时头部高度）: 洋红 horizontal
  - 头部横向参考线（address 时头部 x）: 橙色 vertical
  - 脊柱轴线 shoulder_mid → hip_mid: 青色
  - 前臂链（肘→腕，鸡翅检测用）: 绿色 + 肘角标注
"""
import sys, json, math
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, "/home/jason/projects/swingcue-postest")
from engine.a_measurement.pose_pipeline import PosePipeline
from engine.b_phase.swing_phase import SwingPhaseEngine

PROJ    = Path("/home/jason/projects/swingcue-postest")
INPUT   = PROJ / "input"
KP_PATH = PROJ / "engine/kp_cache/batch2/fo-ok-2.json"
VIDEO   = INPUT / "fo-ok-2.mp4"
DEST    = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/batch2/peak_frames/fo-ok-2")
DEST.mkdir(parents=True, exist_ok=True)

FONT = cv2.FONT_HERSHEY_DUPLEX
LW   = 3

# Colors (BGR)
C_HEAD_H  = (200, 0, 200)    # 洋红 — 头部纵向参考线
C_HEAD_V  = (0, 140, 255)    # 橙色 — 头部横向参考线
C_SPINE   = (255, 220, 0)    # 青黄 — 脊柱轴线
C_FOREARM = (0, 200, 60)     # 绿色 — 前臂链
C_WRIST   = (80, 255, 80)    # 浅绿 — 手腕点
C_WHITE   = (255, 255, 255)
C_BLACK   = (0, 0, 0)
C_GRAY    = (160, 160, 160)


def get_frame(vpath, idx):
    cap = cv2.VideoCapture(str(vpath))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, f = cap.read()
    cap.release()
    return f if ret else np.zeros((1280, 720, 3), np.uint8)


def kp_pt(m, name, thr=0.25):
    pt = m.keypoints.get(name)
    sc = m.confidences.get(name, 0.0)
    if pt is not None and sc >= thr:
        return (float(pt[0]), float(pt[1]))
    return None


def draw_label(img, text, pos, color=C_WHITE, scale=0.55, thickness=1):
    x, y = int(pos[0]), int(pos[1])
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    cv2.rectangle(img, (x-2, y-th-4), (x+tw+4, y+4), C_BLACK, -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def angle_deg(a, b, c):
    """Angle at vertex b (a-b-c), degrees."""
    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag = math.sqrt(ba[0]**2+ba[1]**2) * math.sqrt(bc[0]**2+bc[1]**2)
    if mag == 0:
        return 0.0
    return math.degrees(math.acos(max(-1, min(1, dot/mag))))


def render_frame(img, m, addr_m, fr_idx, phase_label):
    h, w = img.shape[:2]
    out = img.copy()

    # --- anchor keypoints from current frame ---
    nose    = kp_pt(m, "nose")
    l_eye   = kp_pt(m, "left_eye")
    r_eye   = kp_pt(m, "right_eye")
    l_sh    = kp_pt(m, "left_shoulder")
    r_sh    = kp_pt(m, "right_shoulder")
    l_hip   = kp_pt(m, "left_hip")
    r_hip   = kp_pt(m, "right_hip")
    l_elbow = kp_pt(m, "left_elbow")
    r_elbow = kp_pt(m, "right_elbow")
    l_wrist = kp_pt(m, "left_wrist")
    r_wrist = kp_pt(m, "right_wrist")

    # head center (prefer nose; fallback eye mid)
    if nose:
        head_pt = nose
    elif l_eye and r_eye:
        head_pt = ((l_eye[0]+r_eye[0])/2, (l_eye[1]+r_eye[1])/2)
    else:
        head_pt = None

    # mid points
    sh_mid  = ((l_sh[0]+r_sh[0])/2, (l_sh[1]+r_sh[1])/2) if l_sh and r_sh else None
    hip_mid = ((l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2) if l_hip and r_hip else None

    # --- address reference values ---
    addr_nose  = kp_pt(addr_m, "nose")
    addr_l_eye = kp_pt(addr_m, "left_eye")
    addr_r_eye = kp_pt(addr_m, "right_eye")
    if addr_nose:
        addr_head_pt = addr_nose
    elif addr_l_eye and addr_r_eye:
        addr_head_pt = ((addr_l_eye[0]+addr_r_eye[0])/2, (addr_l_eye[1]+addr_r_eye[1])/2)
    else:
        addr_head_pt = None

    addr_l_sh  = kp_pt(addr_m, "left_shoulder")
    addr_r_sh  = kp_pt(addr_m, "right_shoulder")
    addr_l_hip = kp_pt(addr_m, "left_hip")
    addr_r_hip = kp_pt(addr_m, "right_hip")
    addr_sh_mid  = ((addr_l_sh[0]+addr_r_sh[0])/2, (addr_l_sh[1]+addr_r_sh[1])/2) if addr_l_sh and addr_r_sh else None
    addr_hip_mid = ((addr_l_hip[0]+addr_r_hip[0])/2, (addr_l_hip[1]+addr_r_hip[1])/2) if addr_l_hip and addr_r_hip else None

    # 1. 头部纵向参考线（洋红水平线 at address head Y）
    if addr_head_pt:
        ref_y = int(addr_head_pt[1])
        cv2.line(out, (0, ref_y), (w, ref_y), C_HEAD_H, LW, cv2.LINE_AA)
        draw_label(out, f"addr head Y={ref_y}", (4, ref_y - 8), C_HEAD_H)

    # 2. 头部横向参考线（橙色垂直线 at address head X）
    if addr_head_pt:
        ref_x = int(addr_head_pt[0])
        cv2.line(out, (ref_x, 0), (ref_x, h), C_HEAD_V, LW, cv2.LINE_AA)
        draw_label(out, f"addr head X={ref_x}", (ref_x + 4, 30), C_HEAD_V)

    # 3. 当前帧头部位置（圆点）
    if head_pt:
        cv2.circle(out, (int(head_pt[0]), int(head_pt[1])), 7, C_WHITE, -1, cv2.LINE_AA)
        if addr_head_pt:
            dy_pct = (head_pt[1] - addr_head_pt[1]) / max(1,
                     abs(addr_hip_mid[1] - addr_sh_mid[1]) if addr_hip_mid and addr_sh_mid else 200) * 100
            dx_pct = (head_pt[0] - addr_head_pt[0]) / max(1,
                     abs(addr_hip_mid[1] - addr_sh_mid[1]) if addr_hip_mid and addr_sh_mid else 200) * 100
            sign_v = "+" if dy_pct < 0 else ""  # negative dy = head UP (y decreases up)
            draw_label(out, f"head dy={dy_pct:.1f}% dx={dx_pct:.1f}%",
                       (int(head_pt[0])+10, int(head_pt[1])-10), C_WHITE)

    # 4. 脊柱轴线（青黄线，address 叠加当前）
    if addr_sh_mid and addr_hip_mid:
        cv2.line(out,
                 (int(addr_sh_mid[0]), int(addr_sh_mid[1])),
                 (int(addr_hip_mid[0]), int(addr_hip_mid[1])),
                 C_SPINE, 1, cv2.LINE_AA)
        draw_label(out, "addr spine", (int(addr_sh_mid[0])+4, int(addr_sh_mid[1])-4), C_GRAY, 0.45)
    if sh_mid and hip_mid:
        cv2.line(out,
                 (int(sh_mid[0]), int(sh_mid[1])),
                 (int(hip_mid[0]), int(hip_mid[1])),
                 C_SPINE, LW, cv2.LINE_AA)
        draw_label(out, "spine", (int(sh_mid[0])+4, int(sh_mid[1])-4), C_SPINE, 0.45)

    # 5. 前臂链（两侧，绿色）+ 肘角标注
    for side, elbow, wrist, shoulder in [
        ("L", l_elbow, l_wrist, l_sh),
        ("R", r_elbow, r_wrist, r_sh),
    ]:
        if elbow and wrist:
            cv2.line(out, (int(elbow[0]), int(elbow[1])),
                     (int(wrist[0]), int(wrist[1])), C_FOREARM, LW, cv2.LINE_AA)
            cv2.circle(out, (int(wrist[0]), int(wrist[1])), 5, C_WRIST, -1)
            if shoulder:
                ang = angle_deg(shoulder, elbow, wrist)
                draw_label(out, f"{side}elbow {ang:.0f}°",
                           (int(elbow[0])+8, int(elbow[1])), C_FOREARM)
                cv2.line(out, (int(shoulder[0]), int(shoulder[1])),
                         (int(elbow[0]), int(elbow[1])), C_FOREARM, 1, cv2.LINE_AA)

    # 6. Key skeleton joints (dots)
    for pt, c in [(l_sh, C_GRAY), (r_sh, C_GRAY), (l_hip, C_GRAY), (r_hip, C_GRAY)]:
        if pt:
            cv2.circle(out, (int(pt[0]), int(pt[1])), 5, c, -1, cv2.LINE_AA)

    # 7. Frame info banner
    q = m.measurement_quality
    banner = f"fo-ok-2  fr{fr_idx:04d}  phase={phase_label}  quality={q}"
    (tw, th), _ = cv2.getTextSize(banner, FONT, 0.65, 1)
    cv2.rectangle(out, (0, h-th-16), (tw+16, h), C_BLACK, -1)
    cv2.putText(out, banner, (8, h-8), FONT, 0.65, C_WHITE, 1, cv2.LINE_AA)

    return out


def main():
    # Load kp cache
    pipeline = PosePipeline(device="cpu")
    with open(KP_PATH) as f:
        kp_json = json.load(f)
    measurements, fps = pipeline.run_from_json(kp_json)
    n = len(measurements)
    print(f"fo-ok-2: {n} frames @ {fps:.1f} fps")

    # B layer
    engine = SwingPhaseEngine()
    annotations, anchors = engine.run(measurements, fps, angle="face-on")
    phase_labels = [a.phase for a in annotations]

    addr_fr  = anchors.address
    impact_fr = anchors.impact
    print(f"anchors: addr=fr{addr_fr}  top=fr{anchors.top}  impact=fr{impact_fr}  finish=fr{anchors.finish}")

    addr_m = measurements[addr_fr]

    # Frames to render: address + fr75
    targets = [
        (addr_fr,  phase_labels[addr_fr],  "address"),
        (75,       phase_labels[min(75, n-1)], "fr075"),
    ]

    saved_paths = []
    for fr_idx, phase_label, label in targets:
        raw = get_frame(VIDEO, fr_idx)
        m   = measurements[min(fr_idx, n-1)]
        rendered = render_frame(raw, m, addr_m, fr_idx, phase_label)
        out_path = DEST / f"fo-ok-2_{label}_fr{fr_idx:04d}.jpg"
        cv2.imwrite(str(out_path), rendered, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_paths.append(out_path)
        print(f"  Saved: {out_path}")

    # Also render impact frame for reference
    fr_idx = impact_fr
    raw = get_frame(VIDEO, fr_idx)
    m   = measurements[min(fr_idx, n-1)]
    rendered = render_frame(raw, m, addr_m, fr_idx, phase_labels[fr_idx])
    out_path = DEST / f"fo-ok-2_impact_fr{fr_idx:04d}.jpg"
    cv2.imwrite(str(out_path), rendered, [cv2.IMWRITE_JPEG_QUALITY, 95])
    saved_paths.append(out_path)
    print(f"  Saved: {out_path}")

    print("\n=== Windows 路径 ===")
    for p in saved_paths:
        win_path = str(p).replace("/mnt/c/", "C:\\").replace("/", "\\")
        print(f"  {win_path}")

    return saved_paths


if __name__ == "__main__":
    main()
