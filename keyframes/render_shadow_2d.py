#!/usr/bin/env python3
"""
render_shadow_2d.py — Plan A: 2D geometric retargeting

"Ideal shadow" = address frame posture (frame 10) treated as illustrative
standard, bone-length scaled to user's own limbs, anchor-aligned to user's
foot position at impact frame.

Label: "示意标准姿势 (2D retargeting)"
"""

import cv2, numpy as np, json, math
from pathlib import Path

OUT = Path("/home/jason/projects/swingcue-postest/keyframes/preview")
OUT.mkdir(parents=True, exist_ok=True)
DTL_VIDEO = "/home/jason/projects/swingcue-postest/input/test-dwontheline.mp4"
KP_JSON   = "output/rtmpose/test-dwontheline_keypoints.json"

SHADOW_COLOR = (60, 180, 60)   # green BGR
SHADOW_ALPHA = 0.55            # shadow opacity
CONTOUR_W    = 6
FILL_ALPHA   = 0.55

JOINTS = ["nose","left_eye","right_eye","left_ear","right_ear",
          "left_shoulder","right_shoulder","left_elbow","right_elbow",
          "left_wrist","right_wrist","left_hip","right_hip",
          "left_knee","right_knee","left_ankle","right_ankle"]

BONES = [
    ("left_shoulder","right_shoulder"),
    ("left_shoulder","left_elbow"),("left_elbow","left_wrist"),
    ("right_shoulder","right_elbow"),("right_elbow","right_wrist"),
    ("left_shoulder","left_hip"),("right_shoulder","right_hip"),
    ("left_hip","right_hip"),
    ("left_hip","left_knee"),("left_knee","left_ankle"),
    ("right_hip","right_knee"),("right_knee","right_ankle"),
    ("left_shoulder","nose"),
]

def load_kps(data, frame_idx):
    fd = data["frames"][frame_idx]
    if not fd["persons"]: return None
    kps = fd["persons"][0]["keypoints"]
    return {k: np.array([v["x"], v["y"]]) for k, v in kps.items()}

def bone_len(kps, a, b):
    if a not in kps or b not in kps: return 1.0
    return float(np.linalg.norm(kps[a] - kps[b]))

def load_frame(idx):
    cap = cv2.VideoCapture(DTL_VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, f = cap.read(); cap.release()
    return f

def retarget_2d(src_kps, tgt_kps):
    """
    Scale src_kps (address) so each bone matches the user's
    bone lengths in tgt_kps (impact), then align ankles.
    Uses hip-center as global scale anchor.
    """
    # Scale factor: ratio of user's torso height (sh_mid→hip_mid)
    def torso(k):
        sh = (k["left_shoulder"] + k["right_shoulder"]) / 2
        hp = (k["left_hip"]      + k["right_hip"])      / 2
        return float(np.linalg.norm(sh - hp))

    scale = torso(tgt_kps) / (torso(src_kps) or 1)
    src_center = (src_kps["left_hip"] + src_kps["right_hip"]) / 2
    tgt_center = (tgt_kps["left_hip"] + tgt_kps["right_hip"]) / 2

    scaled = {}
    for k, v in src_kps.items():
        scaled[k] = (v - src_center) * scale + tgt_center

    # Align by mid-ankle
    src_ankle = (scaled["left_ankle"] + scaled["right_ankle"]) / 2
    tgt_ankle = (tgt_kps["left_ankle"] + tgt_kps["right_ankle"]) / 2
    offset = tgt_ankle - src_ankle
    return {k: v + offset for k, v in scaled.items()}

def draw_shadow(canvas, kps, color, contour_w, fill_alpha):
    """Draw filled stick-figure shadow."""
    # Collect all points for convex hull → filled silhouette
    pts_list = []
    for k, v in kps.items():
        pts_list.append([int(v[0]), int(v[1])])

    # Convex hull fill (approximate body silhouette)
    hull_pts = np.array(pts_list, dtype=np.int32)
    hull = cv2.convexHull(hull_pts)

    fill_layer = canvas.copy()
    cv2.fillPoly(fill_layer, [hull], color)
    cv2.addWeighted(fill_layer, fill_alpha, canvas, 1 - fill_alpha, 0, canvas)

    # Skeleton lines
    for a, b in BONES:
        if a in kps and b in kps:
            p1 = tuple(kps[a].astype(int))
            p2 = tuple(kps[b].astype(int))
            cv2.line(canvas, p1, p2, color, contour_w, cv2.LINE_AA)

    # Joint dots
    for k, v in kps.items():
        cv2.circle(canvas, tuple(v.astype(int)), 5, color, -1, cv2.LINE_AA)

def main():
    import os; os.chdir("/home/jason/projects/swingcue-postest")

    with open(KP_JSON) as f: data = json.load(f)

    kps_addr   = load_kps(data, 10)   # address = illustrative "standard"
    kps_impact = load_kps(data, 47)   # impact = current user frame

    # Retarget: scale address posture to user's bone lengths, align to impact feet
    shadow_kps = retarget_2d(kps_addr, kps_impact)

    # Render
    frame = load_frame(47)
    overlay = frame.copy()
    draw_shadow(overlay, shadow_kps, SHADOW_COLOR, CONTOUR_W, FILL_ALPHA)

    result = frame.copy()
    cv2.addWeighted(overlay, SHADOW_ALPHA, frame, 1 - SHADOW_ALPHA, 0, result)

    # Label
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(result, "2D retargeting  [standard pose: illustrative]",
                (14, 38), font, 0.65, (220,255,220), 2, cv2.LINE_AA)
    cv2.putText(result, "2D retargeting  [standard pose: illustrative]",
                (14, 38), font, 0.65, (40,180,60),   1, cv2.LINE_AA)

    out = OUT / "shadow_2D_retarget_DTL_impact_fr47.png"
    cv2.imwrite(str(out), result)
    print(f"Saved: {out}")

    import shutil
    desk = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview")
    desk.mkdir(parents=True, exist_ok=True)
    shutil.copy(out, desk / out.name)
    print(f"Desktop: {desk / out.name}")

if __name__ == "__main__":
    main()
