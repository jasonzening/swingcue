"""
render_compare_v3.py — PR-8b.1 visual verification with head_crown.

Same as v1 (../render_compare.py) but:
  - Pulls WHAM 2D from the NEW infer_video smoke JSON
    (docs/PR-8b/smoke_b32e0f21-2656-473c-aa87-e1eaf6e1221f.json)
    instead of re-projecting from the cached PR-7a.5 3D json.
  - Draws head_crown as a magenta dot (larger), connected to the
    existing H36M head dot via a thin line so the face-to-crown
    vector is visible.

Outputs to docs/PR-8b_VISUAL_VERIFY/v3_calibrated/:
  f044_side_by_side.png  (1440 x 1280)
  f045_side_by_side.png
  f060_side_by_side.png
  f044_torso_overlay.png
  overlay_full.mp4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
VIDEO = ROOT / "python/benchmark/test_videos/b32e0f21-2656-473c-aa87-e1eaf6e1221f.mp4"
SMOKE_JSON = ROOT / "docs/PR-8b/smoke_b32e0f21-2656-473c-aa87-e1eaf6e1221f.json"
OUT_DIR = ROOT / "docs/PR-8b_VISUAL_VERIFY/v3_calibrated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEY_FRAMES = [44, 45, 60]

MEDIAPIPE_TO_COCO_IDX: dict[str, int] = {
    "nose":          0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow":    13, "right_elbow":    14,
    "left_wrist":    15, "right_wrist":    16,
    "left_hip":      23, "right_hip":      24,
    "left_knee":     25, "right_knee":     26,
    "left_ankle":    27, "right_ankle":    28,
}

# BGR colors.
MP_COLOR_SECONDARY   = (255, 180, 30)
MP_COLOR_PRIMARY     = (0, 165, 255)
WHAM_COLOR_SECONDARY = (80, 220, 80)
WHAM_COLOR_PRIMARY   = (0, 255, 100)
# v2: head_crown gets magenta — distinct from both MediaPipe orange
# and WHAM green so the new landmark is unambiguous.
HEAD_CROWN_COLOR     = (255, 0, 255)   # magenta
HEAD_FADED_COLOR     = (50, 140, 50)   # darker green for the H36M head
                                       # (de-emphasized when crown is shown)

PRIMARY_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_hip",      "right_hip"),
    ("neck",          "pelvis"),
]
SECONDARY_EDGES = [
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
]


def add_derived(joints: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    out = dict(joints)
    if "left_shoulder" in out and "right_shoulder" in out:
        ls, rs = out["left_shoulder"], out["right_shoulder"]
        out["neck"] = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    if "left_hip" in out and "right_hip" in out:
        lh, rh = out["left_hip"], out["right_hip"]
        out["pelvis"] = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
    return out


def ipt(p): return (int(round(p[0])), int(round(p[1])))


def draw_skeleton(
    canvas, joints, sec_color, pri_color, thickness=3, dot_radius=7,
    head_crown=None, head_faded=False,
):
    # Secondary edges first.
    for a, b in SECONDARY_EDGES:
        if a in joints and b in joints:
            cv2.line(canvas, ipt(joints[a]), ipt(joints[b]), sec_color, thickness)
    # Primary edges on top.
    for a, b in PRIMARY_EDGES:
        if a in joints and b in joints:
            cv2.line(canvas, ipt(joints[a]), ipt(joints[b]), pri_color, thickness + 2)
    # Joint dots — head gets faded color when crown is also drawn.
    for name, p in joints.items():
        color = HEAD_FADED_COLOR if (name == "head" and head_faded) else sec_color
        cv2.circle(canvas, ipt(p), dot_radius, color, -1)
        cv2.circle(canvas, ipt(p), dot_radius + 1, (0, 0, 0), 1)
    # v2: head_crown on top with thin connector line to head.
    if head_crown is not None and "head" in joints:
        # Thin connector line — head → crown (face-to-crown vector).
        cv2.line(canvas, ipt(joints["head"]), ipt(head_crown), HEAD_CROWN_COLOR, 1)
        # Larger magenta dot for the crown itself.
        cv2.circle(canvas, ipt(head_crown), dot_radius + 3, HEAD_CROWN_COLOR, -1)
        cv2.circle(canvas, ipt(head_crown), dot_radius + 4, (0, 0, 0), 1)


def render_side_by_side(fi, frame, mp_joints, wham_joints, wham_crown):
    left = frame.copy()
    right = frame.copy()
    draw_skeleton(left, add_derived(mp_joints), MP_COLOR_SECONDARY, MP_COLOR_PRIMARY)
    draw_skeleton(right, wham_joints, WHAM_COLOR_SECONDARY, WHAM_COLOR_PRIMARY,
                  head_crown=wham_crown, head_faded=True)

    for canvas, label, color in (
        (left,  f"MediaPipe  surface  f={fi}", MP_COLOR_PRIMARY),
        (right, f"WHAM  bone-center + crown  f={fi}", WHAM_COLOR_PRIMARY),
    ):
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(canvas, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    # Legend on right canvas — mark the new crown landmark.
    cv2.rectangle(right, (right.shape[1] - 260, 60), (right.shape[1] - 10, 110), (0, 0, 0), -1)
    cv2.circle(right, (right.shape[1] - 240, 75), 8, HEAD_CROWN_COLOR, -1)
    cv2.putText(right, "head_crown (vtx 411)", (right.shape[1] - 220, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.circle(right, (right.shape[1] - 240, 100), 7, HEAD_FADED_COLOR, -1)
    cv2.putText(right, "H36M head (faded)", (right.shape[1] - 220, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return np.hstack([left, right])


def render_torso_overlay(fi, frame, mp_joints, wham_joints, wham_crown):
    canvas = frame.copy()
    draw_skeleton(canvas, add_derived(mp_joints), MP_COLOR_SECONDARY, MP_COLOR_PRIMARY, dot_radius=6)
    draw_skeleton(canvas, wham_joints, WHAM_COLOR_SECONDARY, WHAM_COLOR_PRIMARY, dot_radius=6,
                  head_crown=wham_crown, head_faded=True)

    pts = list(mp_joints.values()) + list(wham_joints.values())
    if wham_crown: pts.append(wham_crown)
    if not pts: return canvas
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    y_min, y_max = min(ys), max(ys)
    y_mid = (y_min + y_max) / 2
    pad_top, pad_bot = 80, 80
    y_top = max(0, int(y_min - pad_top))
    y_bot = min(canvas.shape[0], int(y_mid + (y_max - y_mid) * 0.55 + pad_bot))
    x_min = max(0, int(min(xs) - 50))
    x_max = min(canvas.shape[1], int(max(xs) + 50))
    cropped = canvas[y_top:y_bot, x_min:x_max]
    target_w = 1280
    scale = target_w / cropped.shape[1]
    cropped = cv2.resize(cropped, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(cropped, (0, 0), (cropped.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(cropped, f"OVERLAY f={fi} — MP (orange) + WHAM (green) + head_crown (magenta) — torso",
                (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return cropped


def render_full_overlay_frame(fi, frame, mp_joints, wham_joints, wham_crown):
    canvas = frame.copy()
    draw_skeleton(canvas, add_derived(mp_joints), MP_COLOR_SECONDARY, MP_COLOR_PRIMARY)
    draw_skeleton(canvas, wham_joints, WHAM_COLOR_SECONDARY, WHAM_COLOR_PRIMARY,
                  head_crown=wham_crown, head_faded=True)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(canvas, f"f={fi:03d}  MP orange | WHAM green | crown magenta",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return canvas


def run_mediapipe(video_path: Path):
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                    smooth_landmarks=True, enable_segmentation=False,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(str(video_path))
    out = {}
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if result.pose_landmarks:
            kps = {}
            for name, idx in MEDIAPIPE_TO_COCO_IDX.items():
                lm = result.pose_landmarks.landmark[idx]
                if lm.visibility > 0.1:
                    kps[name] = (lm.x * w, lm.y * h)
            if kps:
                out[fi] = kps
        fi += 1
    cap.release()
    pose.close()
    return out


def load_wham_2d(smoke_json: Path):
    """Returns (joints_by_fi, crown_by_fi).
    joints_by_fi: dict[fi -> dict[name -> (x, y)]] for the 16 H36M joints.
    crown_by_fi:  dict[fi -> (x, y) | None] for head_crown.
    """
    d = json.loads(smoke_json.read_text())
    joints_by_fi: dict[int, dict[str, tuple[float, float]]] = {}
    crown_by_fi: dict[int, tuple[float, float] | None] = {}
    for fr in d["frames"]:
        fi = fr["frame_idx"]
        kp = fr["keypoints_2d_projected"]
        if kp is None:
            continue
        joints = {}
        for name, p in kp.items():
            if name == "head_crown":
                continue
            if p is not None:
                joints[name] = (p["x"], p["y"])
        joints_by_fi[fi] = joints
        crown = kp.get("head_crown")
        crown_by_fi[fi] = (crown["x"], crown["y"]) if crown is not None else None
    return joints_by_fi, crown_by_fi


def main() -> int:
    print(f"[render_compare_v3] video      = {VIDEO}")
    print(f"[render_compare_v3] smoke_json = {SMOKE_JSON}")
    print(f"[render_compare_v3] out_dir    = {OUT_DIR}")

    # WHAM from new smoke JSON.
    wham_joints_by_fi, wham_crown_by_fi = load_wham_2d(SMOKE_JSON)
    print(f"[render_compare_v3] WHAM frames: {len(wham_joints_by_fi)} "
          f"(crown valid: {sum(1 for c in wham_crown_by_fi.values() if c is not None)})")

    # MediaPipe live.
    print(f"[render_compare_v3] running MediaPipe ...")
    mp_2d_by_idx = run_mediapipe(VIDEO)
    print(f"[render_compare_v3] MediaPipe frames: {len(mp_2d_by_idx)}")

    # Coord log
    log_lines = [
        f"# render_compare_v3 — head_crown coord dump for KEY_FRAMES={KEY_FRAMES}",
        f"# WHAM source: {SMOKE_JSON.name}",
        "",
    ]
    for fi in KEY_FRAMES:
        mp_kps = mp_2d_by_idx.get(fi, {})
        wham_kps = wham_joints_by_fi.get(fi, {})
        crown = wham_crown_by_fi.get(fi)
        log_lines.append(f"=== frame {fi} ===")
        log_lines.append(f"  WHAM head        : {wham_kps.get('head')}")
        log_lines.append(f"  WHAM neck        : {wham_kps.get('neck')}")
        log_lines.append(f"  WHAM head_crown  : {crown}")
        if crown and "head" in wham_kps and "neck" in wham_kps:
            h_xy = wham_kps['head']
            n_xy = wham_kps['neck']
            log_lines.append(f"  delta (crown - head) : ({crown[0] - h_xy[0]:+.1f}, {crown[1] - h_xy[1]:+.1f})  "
                             f"<- crown should be above head: dy negative")
            log_lines.append(f"  delta (crown - neck) : ({crown[0] - n_xy[0]:+.1f}, {crown[1] - n_xy[1]:+.1f})")
        log_lines.append("")
    (OUT_DIR / "render_log.txt").write_text("\n".join(log_lines), encoding="utf-8")

    # Side-by-side renders + torso overlay
    cap = cv2.VideoCapture(str(VIDEO))
    for fi in KEY_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok: continue
        mp_kps = mp_2d_by_idx.get(fi, {})
        wham_kps = wham_joints_by_fi.get(fi, {})
        crown = wham_crown_by_fi.get(fi)
        sbs = render_side_by_side(fi, frame, mp_kps, wham_kps, crown)
        sbs_path = OUT_DIR / f"f{fi:03d}_side_by_side.png"
        cv2.imwrite(str(sbs_path), sbs)
        print(f"[render_compare_v3] wrote {sbs_path.name}  {sbs.shape[1]}x{sbs.shape[0]}")
    cap.release()

    # Torso closeup f=44
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 44)
    ok, frame = cap.read()
    cap.release()
    if ok:
        torso = render_torso_overlay(44, frame, mp_2d_by_idx.get(44, {}),
                                       wham_joints_by_fi.get(44, {}),
                                       wham_crown_by_fi.get(44))
        torso_path = OUT_DIR / "f044_torso_overlay.png"
        cv2.imwrite(str(torso_path), torso)
        print(f"[render_compare_v3] wrote {torso_path.name}  {torso.shape[1]}x{torso.shape[0]}")

    # Full overlay MP4
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mp4_path = OUT_DIR / "overlay_full.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, (w, h))
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        overlay = render_full_overlay_frame(fi, frame, mp_2d_by_idx.get(fi, {}),
                                              wham_joints_by_fi.get(fi, {}),
                                              wham_crown_by_fi.get(fi))
        writer.write(overlay)
        fi += 1
    cap.release()
    writer.release()
    print(f"[render_compare_v3] wrote overlay_full.mp4 ({fi} frames, {mp4_path.stat().st_size//1024} KB)")
    print(f"[render_compare_v3] DONE — {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
