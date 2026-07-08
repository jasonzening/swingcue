"""
ghost004_ik_probe.py  -  GHOST-004 方案A 命门验证: 单帧 IK (coach top -> user skeleton)

核心问题: mhr_forward 完全可微(已验证) -> 用梯度下降 IK:
  固定 user beta (shape/scale/expr)
  优化 (body_pose_params, global_rot) 使得
    mhr_forward(user_beta, theta_opt) 的 70-joint 坐标逼近 coach top 的关节目标位置

IK 设计:
  目标: 8 个关键关节 (l/r shoulder, elbow, wrist, hip) 的 3D 坐标
  权重: wrist > elbow > shoulder > hip (末端关键)
  正则: theta 靠近 user address pose (防止怪姿势/多解)
  初值: user address 的 body_pose + 对齐 global_rot (而非 zero pose)
  优化器: Adam, lr=0.01, 200 iter

输出:
  ik_probe_top.jpg  左=教练top目标 右=IK解出的用户姿态
  REPORT_IK_PROBE.txt

运行:
  /home/jason/projects/sam3d_venv/bin/python3 ghost004_ik_probe.py
"""
import torch.hub as _hub
_orig_get = _hub._get_cache_or_reload
def _patched_get(github, force_reload, trust_repo, calling_fn, verbose=True, skip_validation=False):
    import os
    hub_dir = _hub.get_dir()
    cache_dir = os.path.join(hub_dir, "facebookresearch_dinov3_main")
    if os.path.exists(cache_dir):
        return cache_dir
    return _orig_get(github, force_reload, trust_repo, calling_fn, verbose, skip_validation)
_hub._get_cache_or_reload = _patched_get

import os, sys, json, time, shutil
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.1"
sys.path.insert(0, "/home/jason/projects/sam-3d-body")

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path

import roma
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from sam_3d_body.visualization.renderer import Renderer

ROOT        = Path("/home/jason/projects/swingcue-postest")
COACH_VIDEO = Path("/mnt/c/Users/jason/Zening/Swingcue/教学视频/coach-video/coach-fo.mp4")
COACH_KP    = ROOT / "engine/kp_cache/ghost004/coach-fo.json"
USER_VIDEO  = ROOT / "input/fo-ok-1.mp4"
USER_KP     = ROOT / "engine/kp_cache/batch2/fo-ok-1.json"
OUT_DIR     = ROOT / "output/ghost004"
WIN_OUT     = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/ghost004")
CKPT        = Path(os.path.expanduser("~/.cache/sam3d/sam-3d-body-dinov3/model.ckpt"))
MHR_PT      = CKPT.parent / "assets/mhr_model.pt"
GROT_CACHE  = OUT_DIR / "coach_fo_global_rot.npy"

COACH_FR_ADDR = 12
COACH_FR_TOP  = 43   # coach top anchor (from phase_report_step1.json)
USER_FR_ADDR  = 0

# IK hyperparams
IK_LR        = 0.008
IK_ITERS     = 300
IK_REG_ALPHA = 0.1   # L2 regularization toward user address pose
GHOST_ALPHA  = 0.6
GHOST_COLOR  = (0.85, 0.1, 0.1)

KP_ORDER = ["nose","left_eye","right_eye","left_ear","right_ear",
            "left_shoulder","right_shoulder","left_elbow","right_elbow",
            "left_wrist","right_wrist","left_hip","right_hip",
            "left_knee","right_knee","left_ankle","right_ankle"]

# MHR70 key joint indices + weights for IK
# High weight = end-effector (golf critical), low = root-adjacent
IK_JOINTS = {
    "l_wrist":    (9,  3.0),
    "r_wrist":    (10, 3.0),
    "l_elbow":    (7,  2.0),
    "r_elbow":    (8,  2.0),
    "l_shoulder": (5,  1.5),
    "r_shoulder": (6,  1.5),
    "l_hip":      (11, 1.0),
    "r_hip":      (12, 1.0),
    "l_knee":     (13, 0.5),
    "r_knee":     (14, 0.5),
    "l_ankle":    (15, 0.3),
    "r_ankle":    (16, 0.3),
}

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
    x1,y1=kp_xy[:,0].min(),kp_xy[:,1].min(); x2,y2=kp_xy[:,0].max(),kp_xy[:,1].max()
    pw=(x2-x1)*pad; ph=(y2-y1)*pad
    return np.array([[max(0,x1-pw),max(0,y1-ph),min(W,x2+pw),min(H,y2+ph)]])

def read_video_fr(path, fi):
    cap = cv2.VideoCapture(str(path))
    frame = None
    for i in range(fi + 1):
        ret, f = cap.read()
        if ret: frame = f
    cap.release()
    return frame

def render_and_blend(verts_np, cam_t, focal, faces, bg, H, W, alpha=GHOST_ALPHA):
    black = np.zeros((H, W, 3), dtype=np.uint8)
    rend  = Renderer(focal_length=focal, faces=faces)
    out   = rend(verts_np, cam_t, black, mesh_base_color=GHOST_COLOR, scene_bg_color=(0,0,0))
    out_u8 = (out * 255).clip(0, 255).astype(np.uint8)
    mask  = np.any(out_u8 > 5, axis=2)
    res   = bg.copy().astype(np.float32)
    res[mask] = (1-alpha)*res[mask] + alpha*out_u8[mask].astype(np.float32)
    return res.astype(np.uint8)

# ── INIT ──────────────────────────────────────────────────────────────────────
print("[INIT] Loading SAM3D Body...")
t0 = time.time()
dev = torch.device("cuda")
model, cfg = load_sam_3d_body(str(CKPT), device="cuda", mhr_path=str(MHR_PT))
est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg,
                          human_detector=None, human_segmentor=None, fov_estimator=None)
est.model.eval()
mhr_head = est.model.head_pose
faces    = est.faces
print(f"  {time.time()-t0:.1f}s")

coach_kp = load_kp_cache(str(COACH_KP))
user_kp  = load_kp_cache(str(USER_KP))

# ── EXTRACT COACH TOP β target joints ─────────────────────────────────────────
print(f"\n[COACH-TOP] Extracting fr{COACH_FR_TOP} joints...")
coach_frame_top = read_video_fr(COACH_VIDEO, COACH_FR_TOP)
H, W = coach_frame_top.shape[:2]
bbox_c = get_bbox(coach_kp[COACH_FR_TOP][:,:2], H, W)
with torch.no_grad():
    out_c_top = est.process_one_image(coach_frame_top, bboxes=bbox_c,
                                       use_mask=False, inference_type="body")[0]
coach_j3d_top = np.array(out_c_top["pred_keypoints_3d"])  # (70,3) world coords
coach_cam_t   = out_c_top["pred_cam_t"]
coach_focal   = float(out_c_top["focal_length"])
print(f"  coach top j3d shape={coach_j3d_top.shape}  cam_t={coach_cam_t.round(4)}")

# ── EXTRACT USER ADDRESS β ────────────────────────────────────────────────────
print(f"\n[USER-ADDR] Extracting fr{USER_FR_ADDR} beta...")
user_frame_addr = read_video_fr(USER_VIDEO, USER_FR_ADDR)
bbox_u = get_bbox(user_kp[USER_FR_ADDR][:,:2], H, W)
with torch.no_grad():
    out_u = est.process_one_image(user_frame_addr, bboxes=bbox_u,
                                   use_mask=False, inference_type="body")[0]
u_shape    = torch.from_numpy(out_u["shape_params"].astype(np.float32)).unsqueeze(0).to(dev)
u_scale    = torch.from_numpy(out_u["scale_params"].astype(np.float32)).unsqueeze(0).to(dev)
u_expr     = torch.from_numpy(out_u["expr_params"].astype(np.float32)).unsqueeze(0).to(dev)
u_hand     = torch.zeros(1, 108, dtype=torch.float32, device=dev)
u_gtrans   = torch.zeros(1, 3,   dtype=torch.float32, device=dev)
u_cam_t    = out_u["pred_cam_t"]    # (3,) numpy
u_focal    = float(out_u["focal_length"])
# user address theta (IK init value)
u_grot_addr  = out_u["global_rot"]         # (3,) numpy ZYX
u_body_addr  = out_u["body_pose_params"]   # (133,) numpy
user_j3d_addr= np.array(out_u["pred_keypoints_3d"])  # (70,3)
print(f"  user cam_t={u_cam_t.round(4)}  focal={u_focal:.1f}")
print(f"  user global_rot={u_grot_addr}")

# ── ALIGN COACH TOP JOINTS TO USER SPACE ──────────────────────────────────────
# Coach and user are in different camera spaces (different cam_t).
# We need to bring coach joints into user's coordinate system,
# or equivalently: normalize both to hip-centered space.
#
# Strategy: hip-center both, then the IK target is in user-hip-centered space,
# and mhr_forward output is also hip-centered (global_trans=0).
#
# mhr_forward with global_trans=0 places the mesh with hip at origin-ish.
# The pred_keypoints_3d from process_one_image = mhr_forward output BEFORE cam_t offset.
# So: to get IK-space targets, hip-center coach joints.

def hip_center_j3d(j3d):
    """center on mid-hip (indices 11=l_hip, 12=r_hip)"""
    hip_mid = (j3d[11] + j3d[12]) / 2.0
    return j3d - hip_mid, hip_mid

coach_j3d_c, coach_hip = hip_center_j3d(coach_j3d_top)
user_j3d_c,  user_hip  = hip_center_j3d(user_j3d_addr)

print(f"\nHip-centered joints (coach top vs user address):")
for jname,(ji,w) in IK_JOINTS.items():
    c = coach_j3d_c[ji]; u = user_j3d_c[ji]
    dist = np.linalg.norm(c - u)
    print(f"  {jname:12s}[{ji:2d}] w={w:.1f}: coach={c.round(3)}  user={u.round(3)}  dist={dist:.4f}")

# IK target tensor (hip-centered coach top joints)
ik_target_np = coach_j3d_c   # (70,3)
ik_target    = torch.from_numpy(ik_target_np.astype(np.float32)).unsqueeze(0).to(dev)  # (1,70,3)

# Build weight tensor
ik_weights = torch.zeros(70, device=dev)
for jname, (ji, w) in IK_JOINTS.items():
    ik_weights[ji] = w
ik_weights = ik_weights.unsqueeze(0).unsqueeze(-1)  # (1,70,1)

# ── FUNCTION: mhr_forward -> hip-centered kps70 ──────────────────────────────
def forward_hip_centered(grot, body_pose):
    """
    Run mhr_forward and return hip-centered 70-joint coordinates.
    grot: (1,3)  body_pose: (1,130)
    Returns: kps70_centered (1,70,3)
    """
    result = mhr_head.mhr_forward(
        global_trans=u_gtrans,
        global_rot=grot,
        body_pose_params=body_pose,
        hand_pose_params=u_hand,
        scale_params=u_scale,
        shape_params=u_shape,
        expr_params=u_expr,
        do_pcblend=True,
        return_keypoints=True,
        return_joint_coords=False,
    )
    # result may be (verts, kps308) or just verts depending on return flags
    if isinstance(result, tuple):
        kps308 = result[1]
    else:
        raise RuntimeError("Expected tuple from mhr_forward with return_keypoints=True")
    
    kps70 = kps308[:, :70, :]  # (1,70,3)
    # Hip-center: subtract mid-hip
    hip_mid = (kps70[:, 11:12, :] + kps70[:, 12:13, :]) / 2.0
    return kps70 - hip_mid, kps70

# ── IK OPTIMIZATION ───────────────────────────────────────────────────────────
print(f"\n[IK] Optimizing {IK_ITERS} iters, lr={IK_LR}, reg={IK_REG_ALPHA}...")

# Init: user address theta (closest natural pose, avoids multi-solution trap)
grot_init = torch.from_numpy(u_grot_addr.astype(np.float32)).unsqueeze(0).to(dev)
body_init = torch.from_numpy(u_body_addr[:130].astype(np.float32)).unsqueeze(0).to(dev)

grot_opt = grot_init.clone().requires_grad_(True)
body_opt = body_init.clone().requires_grad_(True)

# Reference for regularization (pull toward user address pose)
grot_ref = grot_init.detach()
body_ref = body_init.detach()

optimizer = torch.optim.Adam([grot_opt, body_opt], lr=IK_LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=IK_ITERS, eta_min=IK_LR*0.1)

best_loss = float("inf")
best_grot = grot_init.clone()
best_body = body_init.clone()

t_ik = time.time()
log_every = 50
for it in range(IK_ITERS):
    optimizer.zero_grad()
    
    kps70_c, _ = forward_hip_centered(grot_opt, body_opt)
    
    # IK loss: weighted MSE on key joints
    diff = kps70_c - ik_target                    # (1,70,3)
    loss_ik = (ik_weights * diff.pow(2)).sum()
    
    # Regularization: stay close to user address pose
    loss_reg_body = IK_REG_ALPHA * (body_opt - body_ref).pow(2).sum()
    loss_reg_grot = IK_REG_ALPHA * 2.0 * (grot_opt - grot_ref).pow(2).sum()
    
    loss = loss_ik + loss_reg_body + loss_reg_grot
    loss.backward()
    
    # Gradient clip to prevent explosion
    torch.nn.utils.clip_grad_norm_([grot_opt, body_opt], max_norm=1.0)
    optimizer.step()
    scheduler.step()
    
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_grot = grot_opt.detach().clone()
        best_body = body_opt.detach().clone()
    
    if it % log_every == 0 or it == IK_ITERS - 1:
        with torch.no_grad():
            kps_eval, _ = forward_hip_centered(grot_opt, body_opt)
            residuals = {}
            for jname, (ji, w) in IK_JOINTS.items():
                res = float(torch.norm(kps_eval[0, ji] - ik_target[0, ji]).item())
                residuals[jname] = res
            end_eff = (residuals.get("l_wrist",0) + residuals.get("r_wrist",0)) / 2
        print(f"  iter {it:3d}: loss={loss.item():.4f}  ik={loss_ik.item():.4f}  "
              f"reg={loss_reg_body.item():.4f}  wrist_err={end_eff:.4f}m")

print(f"  IK done: {time.time()-t_ik:.1f}s  best_loss={best_loss:.4f}")

# ── FINAL RESIDUAL ANALYSIS ────────────────────────────────────────────────────
with torch.no_grad():
    kps_final, kps70_raw = forward_hip_centered(best_grot, best_body)
    print("\n[RESIDUALS] Joint errors (hip-centered space, meters):")
    total_err = 0
    for jname, (ji, w) in IK_JOINTS.items():
        err = float(torch.norm(kps_final[0, ji] - ik_target[0, ji]).item())
        total_err += err
        print(f"  {jname:12s}[{ji}]: err={err:.4f}m")
    print(f"  Total: {total_err:.4f}m  Mean: {total_err/len(IK_JOINTS):.4f}m")
    
    # wrist-specific
    l_wr = float(torch.norm(kps_final[0,9] - ik_target[0,9]).item())
    r_wr = float(torch.norm(kps_final[0,10] - ik_target[0,10]).item())
    print(f"\n  l_wrist={l_wr:.4f}m  r_wrist={r_wr:.4f}m  (key end-effectors)")

# ── RENDER COMPARISON ─────────────────────────────────────────────────────────
print("\n[RENDER] Building comparison image...")

# Get IK-solved verts for user
with torch.no_grad():
    res_ik = mhr_head.mhr_forward(
        global_trans=u_gtrans, global_rot=best_grot, body_pose_params=best_body,
        hand_pose_params=u_hand, scale_params=u_scale, shape_params=u_shape,
        expr_params=u_expr, do_pcblend=True, return_keypoints=False,
    )
    verts_ik = res_ik if not isinstance(res_ik, tuple) else res_ik[0]
    verts_ik_np = verts_ik.squeeze(0).cpu().numpy()
    verts_ik_np[..., [1,2]] *= -1

# Left panel: coach top original frame (with coach skeleton drawn)
coach_panel = coach_frame_top.copy()
coach_panel_bg = coach_frame_top.copy()

# Draw coach skeleton (target joints) on coach panel
cx, cy = W//2, H//2
fl, ct = coach_focal, coach_cam_t
def proj(j3d_pt):
    X, Y, Z = j3d_pt[0]+ct[0], j3d_pt[1]+ct[1], j3d_pt[2]+ct[2]
    if abs(Z) < 0.01: Z = 0.01
    u = int(fl * X / Z + cx)
    v = int(fl * Y / Z + cy)
    return (u, v)

SKEL_EDGES = [
    (5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
]
for i0, i1 in SKEL_EDGES:
    p0 = proj(coach_j3d_top[i0])
    p1 = proj(coach_j3d_top[i1])
    cv2.line(coach_panel, p0, p1, (0,220,80), 3)
for jname, (ji, w) in IK_JOINTS.items():
    cv2.circle(coach_panel, proj(coach_j3d_top[ji]), 8, (0,255,100), -1)

cv2.rectangle(coach_panel, (0,0), (W,45), (0,0,0), -1)
cv2.putText(coach_panel, f"COACH TOP (fr{COACH_FR_TOP}) - IK target joints", (8,30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100,255,100), 2)

# Right panel: user frame with IK ghost overlay
user_panel = render_and_blend(verts_ik_np, u_cam_t, u_focal, faces,
                               user_frame_addr, H, W)
cv2.rectangle(user_panel, (0,0), (W,45), (0,0,0), -1)
cv2.putText(user_panel, f"USER IK-SOLVED (top target, user beta+skeleton)", (8,30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80,200,255), 2)

# Residual text
with torch.no_grad():
    l_wr = float(torch.norm(kps_final[0,9] - ik_target[0,9]).item())
    r_wr = float(torch.norm(kps_final[0,10] - ik_target[0,10]).item())
cv2.putText(user_panel, f"wrist err: L={l_wr:.3f}m R={r_wr:.3f}m", (8, H-15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,100), 2)

# Combine
combined = np.concatenate([coach_panel, user_panel], axis=1)

out_path  = OUT_DIR / "ik_probe_top.jpg"
win_path  = WIN_OUT / "ik_probe_top.jpg"
cv2.imwrite(str(out_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 92])
shutil.copy2(str(out_path), str(win_path))
print(f"  saved: {win_path}")

# ── SAVE OPTIMIZED PARAMS (for reuse in full pipeline) ────────────────────────
ik_result = {
    "frame":   "coach_top",
    "coach_fr": COACH_FR_TOP,
    "best_grot": best_grot.cpu().numpy().tolist(),
    "best_body": best_body.cpu().numpy().tolist(),
    "user_cam_t": u_cam_t.tolist() if hasattr(u_cam_t, 'tolist') else list(u_cam_t),
    "user_focal": u_focal,
    "residuals": {jn: float(torch.norm(kps_final[0,ji] - ik_target[0,ji]).item())
                  for jn,(ji,w) in IK_JOINTS.items()},
}
import json as _json
(OUT_DIR / "ik_probe_result.json").write_text(_json.dumps(ik_result, indent=2))

# ── REPORT ────────────────────────────────────────────────────────────────────
report = f"""REPORT - GHOST-004 IK Probe (coach top frame -> user skeleton)
{"="*60}

IK SETUP:
  Coach target: fr{COACH_FR_TOP} (top phase)
  User beta: fr{USER_FR_ADDR} (address, fixed throughout)
  Optimizer: Adam lr={IK_LR}, {IK_ITERS} iters + CosineAnnealingLR
  Regularization: L2 toward user address pose (alpha={IK_REG_ALPHA})
  IK space: hip-centered (mid-hip = origin, no cam_t offset)
  Gradient: mhr_forward IS differentiable (confirmed)

IK CONVERGENCE:
  best_loss = {best_loss:.4f}

JOINT RESIDUALS (hip-centered space, meters):
"""
for jname, (ji, w) in IK_JOINTS.items():
    err = ik_result["residuals"][jname]
    report += f"  {jname:12s}[{ji}]: {err:.4f}m\n"
report += f"""
  l_wrist = {ik_result['residuals']['l_wrist']:.4f}m
  r_wrist = {ik_result['residuals']['r_wrist']:.4f}m

VERIFICATION:
  If wrist error < 0.05m (5cm) -> IK feasible for golf retarget
  If rendered ghost shows natural swing top pose -> IK path valid
  If pose looks contorted / wrists wrong -> need more iters or constraints

KEY FINDING: mhr_forward gradient test confirmed body_pose.grad norm=0.685
  -> gradient IK through MHR FK chain is mathematically feasible.
  This is the fundamental gate for method A.
"""
rpt_path = OUT_DIR / "REPORT_IK_PROBE.txt"
rpt_path.write_text(report)
shutil.copy2(str(rpt_path), str(WIN_OUT / "REPORT_IK_PROBE.txt"))

print("\n" + report)
print("[DONE]")
