#!/usr/bin/env python3
"""
render_shadow_3d.py — Plan B: 3D retargeting + projection

1. Load user's RTMW3D joints at impact (frame 47) — actual 3D positions
2. Load "standard" pose from address frame (frame 10) RTMW3D as template
3. In 3D: scale template bones to user's bone lengths (retarget)
4. Project back to 2D using the same camera (approximate pinhole from known 2D↔3D pairs)
5. Render as semi-transparent shadow

Label: "示意标准姿势 (3D retargeting)"
"""

import cv2, numpy as np, json, math
from pathlib import Path
from scipy.optimize import least_squares

OUT = Path("/home/jason/projects/swingcue-postest/keyframes/preview")
OUT.mkdir(parents=True, exist_ok=True)
DTL_VIDEO = "/home/jason/projects/swingcue-postest/input/test-dwontheline.mp4"
JSON_3D   = "output/rtmw3d/test-dwontheline_keypoints3d.json"

SHADOW_COLOR = (60, 180, 60)
SHADOW_ALPHA = 0.55
CONTOUR_W    = 6
FILL_ALPHA   = 0.55

JOINTS_17 = ["nose","left_eye","right_eye","left_ear","right_ear",
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

def load_frame_3d(data, frame_idx):
    fd = data["frames"][frame_idx]
    if not fd["persons"]: return None, None
    p = fd["persons"][0]
    kps3d = {k: np.array([v["x"], v["y"], v["z"]]) for k, v in p["body17_3d"].items()}
    kps2d = {k: np.array([v["x"], v["y"]])          for k, v in p["body17_2d"].items()}
    return kps3d, kps2d

def load_frame_img(idx):
    cap = cv2.VideoCapture(DTL_VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, f = cap.read(); cap.release()
    return f

def estimate_camera(kps3d, kps2d):
    """
    Fit a simple affine projection: [u,v] ≈ A @ [x,y,z,1]
    using known 3D→2D correspondences from RTMW3D.
    Returns 2x4 projection matrix A.
    """
    pts3 = np.array([[v[0],v[1],v[2],1] for v in kps3d.values()])  # Nx4
    pts2 = np.array([[v[0],v[1]]         for v in kps2d.values()])  # Nx2

    # Solve via least squares: pts3 @ A.T = pts2 → A.T = lstsq(pts3, pts2)
    A_T, _, _, _ = np.linalg.lstsq(pts3, pts2, rcond=None)
    A = A_T.T   # 2x4
    return A

def project(pts3d_dict, A):
    """Project dict of 3D points to 2D using affine matrix A."""
    proj = {}
    for k, v in pts3d_dict.items():
        h = np.array([v[0], v[1], v[2], 1.0])
        uv = A @ h
        proj[k] = np.array([float(uv[0]), float(uv[1])])
    return proj

def retarget_3d(src_3d, tgt_3d):
    """
    Scale src_3d (address) bone lengths to match tgt_3d (impact user),
    then translate to align hip centers.
    Returns retargeted 3D positions.
    """
    def torso_h(k):
        sh = (k["left_shoulder"] + k["right_shoulder"]) / 2
        hp = (k["left_hip"]      + k["right_hip"])      / 2
        return float(np.linalg.norm(sh - hp))

    scale = torso_h(tgt_3d) / (torso_h(src_3d) or 1)
    src_hip = (src_3d["left_hip"] + src_3d["right_hip"]) / 2
    tgt_hip = (tgt_3d["left_hip"] + tgt_3d["right_hip"]) / 2

    scaled = {k: (v - src_hip) * scale + tgt_hip for k, v in src_3d.items()}

    # Align ankles vertically (ground plane)
    src_ank = (scaled["left_ankle"] + scaled["right_ankle"]) / 2
    tgt_ank = (tgt_3d["left_ankle"] + tgt_3d["right_ankle"]) / 2
    offset  = tgt_ank - src_ank
    return {k: v + offset for k, v in scaled.items()}

def draw_shadow(canvas, kps2d, color, contour_w, fill_alpha):
    pts_list = np.array([[int(v[0]), int(v[1])] for v in kps2d.values()], dtype=np.int32)
    hull = cv2.convexHull(pts_list)

    fill = canvas.copy()
    cv2.fillPoly(fill, [hull], color)
    cv2.addWeighted(fill, fill_alpha, canvas, 1 - fill_alpha, 0, canvas)

    for a, b in BONES:
        if a in kps2d and b in kps2d:
            cv2.line(canvas, tuple(kps2d[a].astype(int)),
                     tuple(kps2d[b].astype(int)), color, contour_w, cv2.LINE_AA)
    for v in kps2d.values():
        cv2.circle(canvas, tuple(v.astype(int)), 5, color, -1, cv2.LINE_AA)

def main():
    import os; os.chdir("/home/jason/projects/swingcue-postest")

    with open(JSON_3D) as f: data = json.load(f)

    # Address (fr10) = illustrative standard pose
    src_3d, _       = load_frame_3d(data, 10)
    # Impact (fr47)  = actual user pose
    tgt_3d, tgt_2d  = load_frame_3d(data, 47)

    # 1. Estimate camera from impact frame (3D→2D correspondences)
    A = estimate_camera(tgt_3d, tgt_2d)
    print("Camera fitted. Reprojection check:")
    for k in list(tgt_3d.keys())[:3]:
        proj = (A @ np.array([*tgt_3d[k], 1]))
        print(f"  {k}: 3D→proj={proj.round(1)}  actual2D={tgt_2d[k].round(1)}")

    # 2. Retarget address 3D pose to user's body proportions
    shadow_3d = retarget_3d(src_3d, tgt_3d)

    # 3. Project shadow 3D → 2D using same camera
    shadow_2d = project(shadow_3d, A)

    # 4. Render
    frame   = load_frame_img(47)
    overlay = frame.copy()
    draw_shadow(overlay, shadow_2d, SHADOW_COLOR, CONTOUR_W, FILL_ALPHA)

    result = frame.copy()
    cv2.addWeighted(overlay, SHADOW_ALPHA, frame, 1 - SHADOW_ALPHA, 0, result)

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(result, "3D retargeting  [standard pose: illustrative]",
                (14, 38), font, 0.65, (220,255,220), 2, cv2.LINE_AA)
    cv2.putText(result, "3D retargeting  [standard pose: illustrative]",
                (14, 38), font, 0.65, (40,180,60),   1, cv2.LINE_AA)

    out = OUT / "shadow_3D_retarget_DTL_impact_fr47.png"
    cv2.imwrite(str(out), result)
    print(f"Saved: {out}")

    import shutil
    desk = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview")
    desk.mkdir(parents=True, exist_ok=True)
    shutil.copy(out, desk / out.name)
    print(f"Desktop: {desk / out.name}")

if __name__ == "__main__":
    main()
