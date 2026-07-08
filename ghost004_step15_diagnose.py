"""
ghost004_step15_diagnose.py  —  GHOST-004 阶段 1.5 病因定位实验

控制变量法，三渲染并排对比 (top + impact 两帧):
  R1: coach θ + coach β  ← 基准真相
  R2: coach θ + user β   ← 当前失败版 (对照)
  R3: coach θ + mixed β  ← user shape + coach scale (隔离 scale 变量)

输出: ghost004_diagnose_top.jpg / ghost004_diagnose_impact.jpg (3格横排)
运行: /home/jason/projects/sam3d_venv/bin/python3 ghost004_step15_diagnose.py
"""
import os, sys, json, time, shutil
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.1"

import numpy as np
import cv2
import torch
from pathlib import Path

sys.path.insert(0, "/home/jason/projects/sam-3d-body")
import roma
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from sam_3d_body.visualization.renderer import Renderer

ROOT        = Path("/home/jason/projects/swingcue-postest")
COACH_VIDEO = Path("/mnt/c/Users/jason/Zening/Swingcue/教学视频/coach-video/coach-fo.mp4")
COACH_KP    = ROOT / "engine/kp_cache/ghost004/coach-fo.json"
COACH_NPZ   = ROOT / "output/ghost004/fo_pose_sequence_aligned.npz"
USER_VIDEO  = ROOT / "input/fo-ok-1.mp4"
USER_KP     = ROOT / "engine/kp_cache/batch2/fo-ok-1.json"
OUT_DIR     = ROOT / "output/ghost004"
WIN_OUT     = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/ghost004")
CKPT        = Path(os.path.expanduser("~/.cache/sam3d/sam-3d-body-dinov3/model.ckpt"))
MHR_PT      = CKPT.parent / "assets/mhr_model.pt"
GROT_CACHE  = OUT_DIR / "coach_fo_global_rot.npy"

GHOST_ALPHA = 0.6
GHOST_COLOR = (0.85, 0.1, 0.1)
USER_FR_ADDR = 0

KP_ORDER = ["nose","left_eye","right_eye","left_ear","right_ear",
            "left_shoulder","right_shoulder","left_elbow","right_elbow",
            "left_wrist","right_wrist","left_hip","right_hip",
            "left_knee","right_knee","left_ankle","right_ankle"]

def load_kp_cache(path):
    with open(path) as f: d = json.load(f)
    out = {}
    for fe in d["frames"]:
        fi = fe["frame"]
        if not fe.get("persons"): continue
        kd = fe["persons"][0]["keypoints"]
        out[fi] = np.array([[kd[n]["x"],kd[n]["y"],kd[n]["score"]]
                             for n in KP_ORDER if n in kd], dtype=np.float32)
    return out

def get_bbox(kp_xy, H, W, pad=0.15):
    x1,y1 = kp_xy[:,0].min(), kp_xy[:,1].min()
    x2,y2 = kp_xy[:,0].max(), kp_xy[:,1].max()
    pw=(x2-x1)*pad; ph=(y2-y1)*pad
    return np.array([[max(0,x1-pw), max(0,y1-ph), min(W,x2+pw), min(H,y2+ph)]])

def euler_to_R(euler_np):
    e = torch.from_numpy(euler_np.astype(np.float32)).unsqueeze(0)
    return roma.euler_to_rotmat("ZYX", e)[0]

def quat_delta_retarget(R_coach_addr, R_user_addr, R_coach_fi):
    q_ca = roma.rotmat_to_unitquat(R_coach_addr.unsqueeze(0))
    q_cf = roma.rotmat_to_unitquat(R_coach_fi.unsqueeze(0))
    q_ua = roma.rotmat_to_unitquat(R_user_addr.unsqueeze(0))
    q_delta = roma.quat_product(roma.quat_inverse(q_ca), q_cf)
    q_ret   = roma.quat_product(q_ua, q_delta)
    R_ret   = roma.unitquat_to_rotmat(q_ret).squeeze(0)
    return roma.rotmat_to_euler("ZYX", R_ret.unsqueeze(0)).squeeze(0).cpu().numpy()

def read_video(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret,f = cap.read()
        if not ret: break
        frames.append(f)
    cap.release()
    return frames

def render_and_blend(verts, cam_t, focal, faces, bg, H, W, alpha=GHOST_ALPHA):
    black_bg = np.zeros((H,W,3), dtype=np.uint8)
    rend = Renderer(focal_length=focal, faces=faces)
    out  = rend(verts, cam_t, black_bg, mesh_base_color=GHOST_COLOR, scene_bg_color=(0,0,0))
    out_u8 = (out*255).clip(0,255).astype(np.uint8)
    mask = np.any(out_u8>5, axis=2)
    res  = bg.copy().astype(np.float32)
    res[mask] = (1-alpha)*res[mask] + alpha*out_u8[mask].astype(np.float32)
    return res.astype(np.uint8)

# ── INIT ──────────────────────────────────────────────────────────────────────
print("[INIT] Loading model...")
t0 = time.time()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, model_cfg = load_sam_3d_body(str(CKPT), device=str(dev), mhr_path=str(MHR_PT))
est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=model_cfg,
                         human_detector=None, human_segmentor=None, fov_estimator=None)
est.model.eval()
mhr_head = est.model.head_pose
faces    = est.faces
print(f"  {time.time()-t0:.1f}s")

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("\n[LOAD] Videos...")
coach_src = read_video(COACH_VIDEO)
user_src  = read_video(USER_VIDEO)
H = coach_src[0].shape[0]; W = coach_src[0].shape[1]
print(f"  coach={len(coach_src)} user={len(user_src)} {W}x{H}")

coach_kp = load_kp_cache(str(COACH_KP))
user_kp  = load_kp_cache(str(USER_KP))

coach_npz = np.load(str(COACH_NPZ), allow_pickle=False)
COACH_ADDR   = int(coach_npz["anchors_address"])
COACH_TOP    = int(coach_npz["anchors_top"])
COACH_IMPACT = int(coach_npz["anchors_impact"])
print(f"  coach anchors: addr={COACH_ADDR} top={COACH_TOP} impact={COACH_IMPACT}")

coach_global_rot_seq = np.load(str(GROT_CACHE))
print(f"  global_rot cache: {coach_global_rot_seq.shape}")

# ── 提取 coach β (fr12 = coach address) ─────────────────────────────────────
print(f"\n[COACH-BETA] fr{COACH_ADDR}...")
kp_c = coach_kp.get(COACH_ADDR)
bbox_c = get_bbox(kp_c[:,:2], H, W)
with torch.no_grad():
    outs_c = est.process_one_image(coach_src[COACH_ADDR], bboxes=bbox_c,
                                   use_mask=False, inference_type="body")
c0 = outs_c[0]
coach_shape = c0["shape_params"]    # (45,)
coach_scale = c0["scale_params"]    # (28,)
coach_expr  = c0["expr_params"]     # (72,)
coach_cam_t = c0["pred_cam_t"]      # (3,)
coach_focal = float(c0["focal_length"])
coach_grot_addr = coach_global_rot_seq[COACH_ADDR]
print(f"  cam_t={coach_cam_t}  focal={coach_focal:.1f}")
print(f"  shape norm={np.linalg.norm(coach_shape):.4f}  scale norm={np.linalg.norm(coach_scale):.4f}")

# ── 提取 user β (fr0 = user address) ─────────────────────────────────────────
print(f"\n[USER-BETA] fr{USER_FR_ADDR}...")
kp_u = user_kp.get(USER_FR_ADDR)
bbox_u = get_bbox(kp_u[:,:2], H, W)
with torch.no_grad():
    outs_u = est.process_one_image(user_src[USER_FR_ADDR], bboxes=bbox_u,
                                   use_mask=False, inference_type="body")
u0 = outs_u[0]
user_shape = u0["shape_params"]    # (45,)
user_scale = u0["scale_params"]    # (28,)
user_expr  = u0["expr_params"]     # (72,)
user_cam_t = u0["pred_cam_t"]      # (3,)
user_focal = float(u0["focal_length"])
user_grot_addr = u0["global_rot"]  # (3,) ZYX
print(f"  cam_t={user_cam_t}  focal={user_focal:.1f}")
print(f"  shape norm={np.linalg.norm(user_shape):.4f}  scale norm={np.linalg.norm(user_scale):.4f}")
print(f"\n  [DIFF] shape norm diff={np.linalg.norm(user_shape-coach_shape):.4f}")
print(f"  [DIFF] scale norm diff={np.linalg.norm(user_scale-coach_scale):.4f}  ← 骨骼长度差异量级")

R_coach_addr = euler_to_R(coach_grot_addr)
R_user_addr  = euler_to_R(user_grot_addr)

# ── mhr_forward 包装 ──────────────────────────────────────────────────────────
def run_mhr(body_pose_130, grot_euler, shape_np, scale_np, expr_np, gtrans_np=None):
    """返回 verts_np (N_verts, 3), 已做 [1,2]*=-1"""
    c_body_t = torch.from_numpy(body_pose_130.astype(np.float32)).unsqueeze(0).to(dev)
    c_grot_t = torch.from_numpy(grot_euler.astype(np.float32)).unsqueeze(0).to(dev)
    g_trans_t = (torch.zeros(1,3,dtype=torch.float32,device=dev)
                 if gtrans_np is None
                 else torch.from_numpy(gtrans_np.astype(np.float32)).unsqueeze(0).to(dev))
    s_t  = torch.from_numpy(shape_np.astype(np.float32)).unsqueeze(0).to(dev)
    sc_t = torch.from_numpy(scale_np.astype(np.float32)).unsqueeze(0).to(dev)
    ex_t = torch.from_numpy(expr_np.astype(np.float32)).unsqueeze(0).to(dev)
    hand_t = torch.zeros(1,108,dtype=torch.float32,device=dev)
    with torch.no_grad():
        result = mhr_head.mhr_forward(
            global_trans=g_trans_t,
            global_rot=c_grot_t,
            body_pose_params=c_body_t,
            hand_pose_params=hand_t,
            scale_params=sc_t,
            shape_params=s_t,
            expr_params=ex_t,
            do_pcblend=True,
            return_keypoints=False,
        )
    verts = result[0] if isinstance(result, tuple) else result
    v = verts.squeeze(0).cpu().numpy()
    v[..., [1,2]] *= -1
    return v

def make_cell(fi_coach, shape_np, scale_np, expr_np, cam_t, focal, bg_frame, label_text):
    """渲染单格: 给定 fi_coach 帧的教练 θ + 指定 β + 背景"""
    body_pose = coach_npz["body_pose_params"][fi_coach]
    c_grot_np = coach_global_rot_seq[fi_coach]
    R_coach_fi = euler_to_R(c_grot_np)

    # 如果是 coach β 配置，不做 delta (保持教练原始朝向 + 教练 cam_t)
    # 如果是 user β / mixed β，做 delta 对齐到用户位置
    need_delta = not np.allclose(shape_np, coach_shape) or not np.allclose(scale_np, coach_scale)
    if not need_delta and np.allclose(expr_np, coach_expr):
        # R1: 教练完整原始
        euler_g = c_grot_np
    else:
        # R2/R3: 用户位置
        euler_g = quat_delta_retarget(R_coach_addr, R_user_addr, R_coach_fi)

    verts = run_mhr(body_pose[:130], euler_g, shape_np, scale_np, expr_np)
    cell = render_and_blend(verts, cam_t, focal, faces, bg_frame, H, W)

    # 标注
    cv2.rectangle(cell, (0,0), (W, 40), (0,0,0), -1)
    cv2.putText(cell, label_text, (5, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    return cell

# ── 渲染三个相位 ──────────────────────────────────────────────────────────────
phases = {
    "top":    COACH_TOP,
    "impact": COACH_IMPACT,
}

for phase_name, fi in phases.items():
    print(f"\n[RENDER] phase={phase_name} fi={fi}")
    t_phase = time.time()

    # 确定背景帧
    bg_coach = coach_src[fi]  # R1 背景 = 教练原始帧
    bg_user  = user_src[USER_FR_ADDR]  # R2/R3 背景 = 用户 address 帧

    # R1: coach θ + coach β (教练 cam_t + focal)
    print("  R1...")
    r1 = make_cell(fi, coach_shape, coach_scale, coach_expr,
                   coach_cam_t, coach_focal, bg_coach,
                   "R1: coach_theta + coach_beta (BASE TRUTH)")

    # R2: coach θ + user β (用户 cam_t + focal)
    print("  R2...")
    r2 = make_cell(fi, user_shape, user_scale, user_expr,
                   user_cam_t, user_focal, bg_user,
                   "R2: coach_theta + user_beta (CURRENT FAIL)")

    # R3: coach θ + mixed β (user shape, coach scale, user cam_t)
    print("  R3...")
    r3 = make_cell(fi, user_shape, coach_scale, coach_expr,
                   user_cam_t, user_focal, bg_user,
                   "R3: coach_theta + mixed_beta (user_shape + coach_scale)")

    row = np.concatenate([r1, r2, r3], axis=1)

    # 标题条
    bar = np.zeros((50, row.shape[1], 3), dtype=np.uint8)
    txt = (f"Phase: {phase_name}  coach fr{fi}  "
           f"| R1=coach_beta  R2=user_beta(fail)  R3=mixed(no scale swap)")
    cv2.putText(bar, txt, (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,220,255), 2)
    grid = np.vstack([bar, row])

    local_path = OUT_DIR / f"ghost004_diagnose_{phase_name}.jpg"
    win_path   = WIN_OUT / f"ghost004_diagnose_{phase_name}.jpg"
    cv2.imwrite(str(local_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    shutil.copy2(str(local_path), str(win_path))
    print(f"  saved: {win_path}  ({time.time()-t_phase:.1f}s)")

# ── 定量摘要 ──────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STRUCTURAL REPORT")
print("="*70)
print(f"""
body_pose_params (133,) → mhr_forward truncates to [:130]:
  MHR joint-local Euler angles (compact representation)
  full_pose_params = cat([global_trans*10(3), global_rot(3), body_pose_params(130)]) = 136 dim
  dims 0-17  : 主躯干关节 (脊柱/髋 Euler)
  dims 18-44 : 肩/臂/手腕 (变化最大: addr→top max diff ~1.34 rad)
  dims 45-129: 腿部/脚踝/颈/其他

scale_params (28,) — 骨骼长度控制:
  scales = scale_mean(68,) + scale_params(28,) @ scale_comps(28,68)
  → 68 维骨骼缩放向量，直接控制每根骨骼物理长度
  → 骨骼长度改变时，同一组 θ 产生不同的关节世界坐标
  coach scale norm = {np.linalg.norm(coach_scale):.4f}
  user  scale norm = {np.linalg.norm(user_scale):.4f}
  diff  norm       = {np.linalg.norm(user_scale-coach_scale):.4f}  ← 这个差异直接进入 FK 链

shape_params (45,) — 体表蒙皮形状:
  传给 self.mhr(shape_params, model_params, ...) 控制顶点蒙皮
  不影响骨骼运动学，只改变表面外观
  coach shape norm = {np.linalg.norm(coach_shape):.4f}
  user  shape norm = {np.linalg.norm(user_shape):.4f}
  diff  norm       = {np.linalg.norm(user_shape-coach_shape):.4f}

mhr_forward 耦合性:
  θ (local rotations) × scale (骨骼长度) = joint world coords → 乘性耦合
  换 scale → 同 θ 产生不同末端位置 → 动作外观变形
  换 shape → 仅蒙皮外形变化，不影响骨架运动

判图逻辑:
  R1 像挥杆 → θ 数据提取质量好，排除嫌疑A (MHR提取不准)
  R3 比 R2 好 → scale 换体型是主要失真源，病因=嫌疑B(scale迁移)
  R3 与 R2 一样差 → shape 或 joint convention 问题，需进一步分析
""")
print("[DONE]")
print(f"  ghost004_diagnose_top.jpg    → {WIN_OUT}")
print(f"  ghost004_diagnose_impact.jpg → {WIN_OUT}")
