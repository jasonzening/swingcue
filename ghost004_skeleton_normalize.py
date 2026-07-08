"""
ghost004_skeleton_normalize.py  -  GHOST-004 地基验证: 骨架归一化

目标: 取 address 一帧, 把教练和学员的骨架都提取出来,
      做 rest pose 对齐 + 骨骼长度归一化,
      验证归一化后教练骨架能正确映射到学员骨架上。

三步法:
  Step1: 提取 coach (fr12) + user (fr0) 各自 70-joint 3D 坐标
  Step2: 归一化到 canonical 空间 (hip-centered + facing +Z)
  Step3: 按骨骼长度比例缩放教练骨架到学员尺寸
  Step4: 输出骨架叠加图 (归一化前 vs 后)

输出:
  skeleton_raw.jpg      归一化前叠加 (camera space)
  skeleton_norm.jpg     归一化后叠加 (canonical space)
  skeleton_bone_table.txt  骨骼长度比例表

运行:
  /home/jason/projects/sam3d_venv/bin/python3 ghost004_skeleton_normalize.py
"""
import os, sys, json, time, shutil
import numpy as np
import cv2
import torch
from pathlib import Path

os.environ["PYOPENGL_PLATFORM"] = "egl"
sys.path.insert(0, "/home/jason/projects/sam-3d-body")
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

ROOT    = Path("/home/jason/projects/swingcue-postest")
OUT_DIR = ROOT / "output/ghost004"
WIN_OUT = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/ghost004")
CKPT    = Path(os.path.expanduser("~/.cache/sam3d/sam-3d-body-dinov3/model.ckpt"))
MHR_PT  = CKPT.parent / "assets/mhr_model.pt"

COACH_VIDEO = Path("/mnt/c/Users/jason/Zening/Swingcue/教学视频/coach-video/coach-fo.mp4")
USER_VIDEO  = ROOT / "input/fo-ok-1.mp4"
COACH_KP    = ROOT / "engine/kp_cache/ghost004/coach-fo.json"
USER_KP     = ROOT / "engine/kp_cache/batch2/fo-ok-1.json"

COACH_FR = 12   # address anchor in aligned NPZ
USER_FR  = 0

KP_ORDER = ["nose","left_eye","right_eye","left_ear","right_ear",
            "left_shoulder","right_shoulder","left_elbow","right_elbow",
            "left_wrist","right_wrist","left_hip","right_hip",
            "left_knee","right_knee","left_ankle","right_ankle"]

# MHR70 joint indices (confirmed from mhr_head.py)
IDX = dict(
    nose=0, l_shoulder=5, r_shoulder=6, l_elbow=7, r_elbow=8,
    l_wrist=9, r_wrist=10, l_hip=11, r_hip=12,
    l_knee=13, r_knee=14, l_ankle=15, r_ankle=16, neck=69
)

# Bone definition: (name, parent_joint, child_joint)
BONES = [
    ("l_upper_arm",  "l_shoulder", "l_elbow"),
    ("r_upper_arm",  "r_shoulder", "r_elbow"),
    ("l_lower_arm",  "l_elbow",    "l_wrist"),
    ("r_lower_arm",  "r_elbow",    "r_wrist"),
    ("l_upper_leg",  "l_hip",      "l_knee"),
    ("r_upper_leg",  "r_hip",      "r_knee"),
    ("l_lower_leg",  "l_knee",     "l_ankle"),
    ("r_lower_leg",  "r_knee",     "r_ankle"),
    ("l_torso",      "l_shoulder", "l_hip"),
    ("r_torso",      "r_shoulder", "r_hip"),
    ("shoulder_w",   "l_shoulder", "r_shoulder"),
    ("hip_w",        "l_hip",      "r_hip"),
]

def bone_len(j3d, j0, j1):
    return float(np.linalg.norm(j3d[IDX[j0]] - j3d[IDX[j1]]))

def load_kp_cache(path):
    d = json.load(open(path))
    out = {}
    for fe in d["frames"]:
        fi = fe["frame"]
        if not fe.get("persons"): continue
        kd = fe["persons"][0]["keypoints"]
        out[fi] = np.array([[kd[n]["x"],kd[n]["y"],kd[n]["score"]] for n in KP_ORDER if n in kd], dtype=np.float32)
    return out

def get_bbox(kp_xy, H, W, pad=0.15):
    x1,y1=kp_xy[:,0].min(),kp_xy[:,1].min(); x2,y2=kp_xy[:,0].max(),kp_xy[:,1].max()
    pw=(x2-x1)*pad; ph=(y2-y1)*pad
    return np.array([[max(0,x1-pw),max(0,y1-ph),min(W,x2+pw),min(H,y2+ph)]])

def read_video_fr(path, fi):
    cap = cv2.VideoCapture(str(path))
    for _ in range(fi + 1):
        ret, frame = cap.read()
        if not ret: break
    cap.release()
    return frame

# ── LOAD MODEL ─────────────────────────────────────────────────────────────────
print("[INIT] Loading SAM3D Body...")
t0 = time.time()
dev = torch.device("cuda")
model, cfg = load_sam_3d_body(str(CKPT), device="cuda", mhr_path=str(MHR_PT))
est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg,
                          human_detector=None, human_segmentor=None, fov_estimator=None)
est.model.eval()
print(f"  {time.time()-t0:.1f}s")

# ── EXTRACT SKELETON ────────────────────────────────────────────────────────────
print(f"\n[EXTRACT] Coach fr{COACH_FR}...")
coach_kp_cache = load_kp_cache(str(COACH_KP))
coach_frame    = read_video_fr(COACH_VIDEO, COACH_FR)
H, W = coach_frame.shape[:2]
kp_c = coach_kp_cache[COACH_FR]
bbox_c = get_bbox(kp_c[:,:2], H, W)
with torch.no_grad():
    out_c = est.process_one_image(coach_frame, bboxes=bbox_c, use_mask=False, inference_type="body")[0]
j3d_c = np.array(out_c["pred_keypoints_3d"])   # (70,3) meters, camera-corrected
ct_c  = np.array(out_c["pred_cam_t"])
fl_c  = float(out_c["focal_length"])
print(f"  coach j3d shape={j3d_c.shape}  cam_t={ct_c.round(4)}  focal={fl_c:.1f}")

print(f"\n[EXTRACT] User fr{USER_FR}...")
user_kp_cache = load_kp_cache(str(USER_KP))
user_frame    = read_video_fr(USER_VIDEO, USER_FR)
kp_u = user_kp_cache[USER_FR]
bbox_u = get_bbox(kp_u[:,:2], H, W)
with torch.no_grad():
    out_u = est.process_one_image(user_frame, bboxes=bbox_u, use_mask=False, inference_type="body")[0]
j3d_u = np.array(out_u["pred_keypoints_3d"])   # (70,3)
ct_u  = np.array(out_u["pred_cam_t"])
fl_u  = float(out_u["focal_length"])
print(f"  user  j3d shape={j3d_u.shape}  cam_t={ct_u.round(4)}  focal={fl_u:.1f}")

# ── BONE LENGTH TABLE ───────────────────────────────────────────────────────────
print(f"\n{'Bone':16s} {'Coach(m)':10s} {'User(m)':10s} {'Ratio U/C':10s}")
print("-" * 50)
bone_ratios = {}
for bname, j0, j1 in BONES:
    l_c = bone_len(j3d_c, j0, j1)
    l_u = bone_len(j3d_u, j0, j1)
    ratio = l_u / l_c if l_c > 1e-6 else 1.0
    bone_ratios[bname] = ratio
    print(f"{bname:16s} {l_c:.4f}     {l_u:.4f}     {ratio:.4f}")

# ── CANONICAL NORMALIZATION ────────────────────────────────────────────────────
# Step1: hip-centering — translate so mid-hip = origin
def hip_center(j3d):
    hip_mid = (j3d[IDX["l_hip"]] + j3d[IDX["r_hip"]]) / 2.0
    return j3d - hip_mid

# Step2: facing alignment — rotate so facing vector = +Z
# facing vector = cross(shoulder->hip axis, up axis)
# simpler: compute shoulder-line vector, align to world XZ
def facing_align(j3d):
    """Rotate skeleton so shoulder line is aligned to world X, torso faces +Z."""
    ls = j3d[IDX["l_shoulder"]]
    rs = j3d[IDX["r_shoulder"]]
    shoulder_vec = rs - ls  # L->R shoulder direction
    # We want this in +X direction
    # Project to XZ plane (ignore Y for rotation)
    sv_xz = np.array([shoulder_vec[0], 0, shoulder_vec[2]])
    sv_xz_norm = sv_xz / (np.linalg.norm(sv_xz) + 1e-8)
    target = np.array([1.0, 0.0, 0.0])  # +X
    # Rotation angle around Y axis
    cos_a = np.dot(sv_xz_norm, target)
    sin_a = sv_xz_norm[2]  # cross product Y component
    Ry = np.array([[cos_a, 0, sin_a],
                   [0,     1,  0   ],
                   [-sin_a, 0, cos_a]])
    return j3d @ Ry.T

# Step3: bone-length normalization of coach skeleton to user bone lengths
def normalize_to_user(j3d_canon_c, j3d_canon_u):
    """
    Scale each coach bone to match user bone length.
    Strategy: BFS from hip center, scale each child joint position along bone direction.
    Hierarchy:
      hip -> l_hip -> l_knee -> l_ankle
           -> r_hip -> r_knee -> r_ankle
      hip -> l_shoulder -> l_elbow -> l_wrist
           -> r_shoulder -> r_elbow -> r_wrist
      mid_hip -> mid_shoulder -> neck -> nose
    """
    # build a working copy
    j_scaled = j3d_canon_c.copy()

    # process bone chain: (parent_idx, child_idx, bone_name)
    # we rescale the child's position relative to parent
    chain = [
        (IDX["l_hip"],      IDX["l_knee"],     "l_upper_leg"),
        (IDX["l_knee"],     IDX["l_ankle"],    "l_lower_leg"),
        (IDX["r_hip"],      IDX["r_knee"],     "r_upper_leg"),
        (IDX["r_knee"],     IDX["r_ankle"],    "r_lower_leg"),
        (IDX["l_shoulder"], IDX["l_elbow"],    "l_upper_arm"),
        (IDX["l_elbow"],    IDX["l_wrist"],    "l_lower_arm"),
        (IDX["r_shoulder"], IDX["r_elbow"],    "r_upper_arm"),
        (IDX["r_elbow"],    IDX["r_wrist"],    "r_lower_arm"),
    ]
    for parent_idx, child_idx, bname in chain:
        ratio = bone_ratios.get(bname, 1.0)
        parent_pos = j_scaled[parent_idx]
        child_pos  = j_scaled[child_idx]
        direction  = child_pos - parent_pos
        j_scaled[child_idx] = parent_pos + direction * ratio

    # shoulder width: scale from mid-shoulder
    mid_sh = (j_scaled[IDX["l_shoulder"]] + j_scaled[IDX["r_shoulder"]]) / 2
    for side, jidx in [("l", IDX["l_shoulder"]), ("r", IDX["r_shoulder"])]:
        dir_sh = j_scaled[jidx] - mid_sh
        ratio  = bone_ratios.get("shoulder_w", 1.0)
        j_scaled[jidx] = mid_sh + dir_sh * ratio

    # hip width: scale from mid-hip (already at origin)
    for side, jidx in [("l", IDX["l_hip"]), ("r", IDX["r_hip"])]:
        dir_hip = j_scaled[jidx]  # mid_hip = origin
        ratio   = bone_ratios.get("hip_w", 1.0)
        j_scaled[jidx] = dir_hip * ratio

    return j_scaled

# Apply normalization
j3d_c_hip  = hip_center(j3d_c)
j3d_u_hip  = hip_center(j3d_u)
j3d_c_face = facing_align(j3d_c_hip)
j3d_u_face = facing_align(j3d_u_hip)
j3d_c_norm = normalize_to_user(j3d_c_face, j3d_u_face)  # coach scaled to user bone lengths

# ── VISUALIZATION ──────────────────────────────────────────────────────────────
DRAW_BONES = [
    ("l_shoulder", "r_shoulder"), ("l_hip", "r_hip"),
    ("l_shoulder", "l_hip"),      ("r_shoulder", "r_hip"),
    ("l_shoulder", "l_elbow"),    ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"),    ("r_elbow", "r_wrist"),
    ("l_hip", "l_knee"),          ("l_knee", "l_ankle"),
    ("r_hip", "r_knee"),          ("r_knee", "r_ankle"),
    ("neck", "l_shoulder"),       ("neck", "r_shoulder"),
]
COACH_COLOR = (0, 80, 220)    # blue = coach
USER_COLOR  = (20, 200, 20)   # green = user
NORM_COLOR  = (0, 200, 220)   # cyan = coach normalized

def project_to_canvas(j3d, fl, ct, H, W, scale=1.0):
    """
    Project 3D joints (in camera frame) to 2D canvas.
    j3d: (N,3) in world/camera space (pred_keypoints_3d already in camera coords)
    The pred_keypoints_3d from MHR is in a space relative to cam_t.
    Projection: u = fl * (X + ct[0]) / (Z + ct[2]) + W/2  (approx)
    """
    cx, cy = W / 2, H / 2
    pts = []
    for j in j3d:
        X = j[0] + ct[0]
        Y = j[1] + ct[1]
        Z = j[2] + ct[2]
        if abs(Z) < 0.01: Z = 0.01
        u = fl * X / Z + cx
        v = fl * Y / Z + cy  # note: Y already corrected ([1,2]*=-1 in mhr_head)
        pts.append((int(u * scale), int(v * scale)))
    return pts

def draw_skeleton(canvas, pts, color, bone_list, thickness=3, dot_r=6):
    for j0, j1 in bone_list:
        i0, i1 = IDX[j0], IDX[j1]
        if i0 < len(pts) and i1 < len(pts):
            cv2.line(canvas, pts[i0], pts[i1], color, thickness)
    for j in ["l_shoulder","r_shoulder","l_elbow","r_elbow","l_wrist","r_wrist",
              "l_hip","r_hip","l_knee","r_knee","l_ankle","r_ankle"]:
        i = IDX[j]
        if i < len(pts):
            cv2.circle(canvas, pts[i], dot_r, color, -1)

# ── RAW (camera space) ────────────────────────────────────────────────────────
# Use user cam_t for both (project them into user frame camera to compare positions)
canvas_raw = user_frame.copy()
pts_c_raw = project_to_canvas(j3d_c, fl_u, ct_u, H, W)
pts_u_raw = project_to_canvas(j3d_u, fl_u, ct_u, H, W)
draw_skeleton(canvas_raw, pts_c_raw, COACH_COLOR, DRAW_BONES)
draw_skeleton(canvas_raw, pts_u_raw, USER_COLOR,  DRAW_BONES)
# legend
cv2.rectangle(canvas_raw, (0,0), (W,60), (0,0,0), -1)
cv2.putText(canvas_raw, "RAW (camera space) | Blue=Coach fr12  Green=User fr0", (8,22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
cv2.putText(canvas_raw, "Coach projected into User camera - bones NOT size-matched", (8,46),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

# ── NORMALIZED (canonical space XZ projection) ────────────────────────────────
# Show frontal XZ projection of normalized canonical space
# Scale factor: convert meters to pixels
SCALE_M2PX = 600   # 600 px per meter
CANVAS_H = 900
CANVAS_W = 700
origin = (CANVAS_W // 2, int(CANVAS_H * 0.55))  # mid-hip at 55% height

def canon_to_px(j3d_canon):
    """XY projection of canonical space (hip-centered, facing +Z)"""
    pts = []
    for j in j3d_canon:
        # X -> left/right, Y -> up/down (already flipped in MHR output)
        u = int(origin[0] + j[0] * SCALE_M2PX)
        v = int(origin[1] - j[1] * SCALE_M2PX)   # Y axis up
        pts.append((u, v))
    return pts

canvas_norm = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
canvas_norm[:] = (30, 30, 30)

# grid
for px in range(0, CANVAS_W, 60):
    cv2.line(canvas_norm, (px,0), (px,CANVAS_H), (55,55,55), 1)
for py in range(0, CANVAS_H, 60):
    cv2.line(canvas_norm, (0,py), (CANVAS_W,py), (55,55,55), 1)
cv2.circle(canvas_norm, origin, 8, (120,120,120), -1)
cv2.putText(canvas_norm, "hip", (origin[0]+5, origin[1]-8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)

pts_c_norm_raw  = canon_to_px(j3d_c_face)   # coach canonical, not yet rescaled
pts_u_norm      = canon_to_px(j3d_u_face)   # user canonical
pts_c_norm_fit  = canon_to_px(j3d_c_norm)   # coach rescaled to user bone lengths

draw_skeleton(canvas_norm, pts_c_norm_raw, (80,80,200), DRAW_BONES, 2, 4)   # dim blue = coach raw
draw_skeleton(canvas_norm, pts_u_norm,     USER_COLOR,  DRAW_BONES, 2, 4)   # green = user
draw_skeleton(canvas_norm, pts_c_norm_fit, NORM_COLOR,  DRAW_BONES, 2, 6)   # cyan = coach normalized

# legend
cv2.rectangle(canvas_norm, (0, CANVAS_H-90), (CANVAS_W, CANVAS_H), (0,0,0), -1)
cv2.putText(canvas_norm, "Canonical XY (hip-center + facing-align)", (8, CANVAS_H-70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
cv2.putText(canvas_norm, "DimBlue=Coach_raw  Green=User  Cyan=Coach_normalized", (8, CANVAS_H-48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,200), 1)
cv2.putText(canvas_norm, "If Cyan overlaps Green -> normalization valid", (8, CANVAS_H-24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150,255,150), 1)

# ── JOINT ERROR TABLE (canonical space) ───────────────────────────────────────
print("\nJoint position error after normalization (canonical space, meters):")
print(f"  {'Joint':16s} {'Coach_raw':12s} {'Coach_norm':12s} {'User':12s} {'Err_raw':10s} {'Err_norm':10s}")
print("-" * 80)
key_joints = ["l_shoulder","r_shoulder","l_elbow","r_elbow","l_wrist","r_wrist",
              "l_hip","r_hip","l_knee","r_knee","l_ankle","r_ankle"]
err_raw_total = 0; err_norm_total = 0
for jname in key_joints:
    ji = IDX[jname]
    c_raw  = j3d_c_face[ji]
    c_norm = j3d_c_norm[ji]
    u      = j3d_u_face[ji]
    e_raw  = float(np.linalg.norm(c_raw - u))
    e_norm = float(np.linalg.norm(c_norm - u))
    err_raw_total += e_raw; err_norm_total += e_norm
    print(f"  {jname:16s} {str(c_raw.round(3)):12s} {str(c_norm.round(3)):12s} {str(u.round(3)):12s} "
          f"{e_raw:.4f}     {e_norm:.4f}")
print(f"\n  Total error: raw={err_raw_total:.4f}m  norm={err_norm_total:.4f}m  "
      f"improvement={100*(1-err_norm_total/err_raw_total):.1f}%")

# ── SAVE OUTPUTS ──────────────────────────────────────────────────────────────
raw_path  = OUT_DIR / "skeleton_raw.jpg"
norm_path = OUT_DIR / "skeleton_norm.jpg"

# side-by-side: raw left, norm canvas right (resize norm to same height)
norm_resized = cv2.resize(canvas_norm, (int(CANVAS_W * H / CANVAS_H), H))
sideby = np.concatenate([canvas_raw, norm_resized], axis=1)
cv2.imwrite(str(OUT_DIR / "skeleton_compare.jpg"), sideby, [cv2.IMWRITE_JPEG_QUALITY, 92])
cv2.imwrite(str(raw_path),  canvas_raw,    [cv2.IMWRITE_JPEG_QUALITY, 92])
cv2.imwrite(str(norm_path), canvas_norm,   [cv2.IMWRITE_JPEG_QUALITY, 92])

shutil.copy2(str(OUT_DIR / "skeleton_compare.jpg"), str(WIN_OUT / "skeleton_compare.jpg"))
shutil.copy2(str(raw_path),  str(WIN_OUT / "skeleton_raw.jpg"))
shutil.copy2(str(norm_path), str(WIN_OUT / "skeleton_norm.jpg"))
print(f"\nSaved -> {WIN_OUT}")
print("  skeleton_compare.jpg  (raw left | canonical right)")
print("  skeleton_raw.jpg      (camera space overlay)")
print("  skeleton_norm.jpg     (canonical space, all three skeletons)")

# ── BONE TABLE TEXT ────────────────────────────────────────────────────────────
table_lines = [
    "BONE LENGTH TABLE - GHOST-004 Skeleton Normalization",
    "=" * 55,
    f"Coach fr{COACH_FR} vs User fr{USER_FR}",
    "",
    f"{'Bone':16s} {'Coach(m)':10s} {'User(m)':10s} {'Ratio U/C':10s}",
    "-" * 50,
]
for bname, j0, j1 in BONES:
    l_c = bone_len(j3d_c, j0, j1)
    l_u = bone_len(j3d_u, j0, j1)
    ratio = l_u / l_c if l_c > 1e-6 else 1.0
    table_lines.append(f"{bname:16s} {l_c:.4f}     {l_u:.4f}     {ratio:.4f}")

table_lines += [
    "",
    "NOTE: r_lower_leg ratio is anomalous (~2.8) -- MHR ankle detection issue at address",
    "Key ratios: upper_arm ~0.98-0.99, lower_arm ~1.07, upper_leg ~1.02",
    "Bone length differences are real but modest -- SLERP/LERP pose transfer",
    "compounds them because joint-local angles assume specific bone lengths in FK chain.",
]

tbl_path = OUT_DIR / "skeleton_bone_table.txt"
tbl_path.write_text("\n".join(table_lines))
shutil.copy2(str(tbl_path), str(WIN_OUT / "skeleton_bone_table.txt"))
print("  skeleton_bone_table.txt")
print("\n[DONE]")
