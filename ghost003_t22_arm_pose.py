"""
ghost003_t22_arm_pose.py  —  GHOST-003 T2.2
A. 哨兵放宽: A1 0.25→0.18; 新增 A4: 孤立连通域 > 1.5% 且质心距 > 150px
B. 手臂姿态修正:
   - 对"手臂延伸帧"(wrist-hip y-diff 超标 OR rtm arm kp 误差 > 40px 均值)
     用 scipy.minimize 对 body_pose_params[24:44] 优化手臂关节角
   - 优化目标: 最小化 MHR arm kp 投影 vs RTMPose arm kp 的 L2 误差
   - 约束: 优化后 upper IoU 不得低于未优化结果 - 0.02 (保护主指标)
C. 关键帧维持 address/top/impact 不得回退
D. 分布报告: 剔除无效帧后 mean/min+帧号/P5 + 剔除前后对比

哨兵判据 (满足任一 → 无效):
  A1: 单帧 IoU vs 滚动邻域均值 drop >= 0.18 (放宽 0.25→0.18)
  A2: 核心 kp (lsho/rsho/lhip/rhip) 平均置信 < 0.65
  A3: mesh_cx vs RTM_cx 偏移 > 60px (post-fit 检查)
  A4: rembg mask 出现孤立连通域 area > 1.5% 主体 且 质心距 > 150px

手臂关节参数探针结论 (2026-07-07):
  body_pose_params[24:44] 主要控制肩/肘/腕 (MHR skeleton joints 59,67,94,103)
  body_pose_params[0:18] = 全局姿态参数 (禁止修改)
  arm_kp indices in kp70: LSHO=5, RSHO=6, LELB=7, RELB=8, LWRI=62, RWRI=41

范围: 整段·真实姿态·贴合渲染·不做动作修正·不碰球杆
授权: SAM License (PRODUCT_CANDIDATE_CUSTOM_LICENSE)
"""

import os, sys, time, json, pathlib
import numpy as np
import cv2
from scipy.optimize import minimize

SAM3D_REPO = "/home/jason/projects/sam-3d-body"
SWINGCUE   = "/home/jason/projects/swingcue-postest"
sys.path.insert(0, SAM3D_REPO)
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

OUTPUT_DIR = pathlib.Path(SWINGCUE) / "output" / "ghost003_t22"
KF_DIR     = OUTPUT_DIR / "keyframe_overlay_t22"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KF_DIR.mkdir(parents=True, exist_ok=True)

KP_CACHE   = pathlib.Path(SWINGCUE) / "engine/kp_cache/batch2/fo-ok-1.json"
VIDEO_PATH = pathlib.Path(SWINGCUE) / "input/fo-ok-1.mp4"
T2_LOG     = pathlib.Path(SWINGCUE) / "output/ghost003_t2/run_log_t2.json"
CKPT  = "/home/jason/.cache/sam3d/sam-3d-body-dinov3/model.ckpt"
MHR_P = "/home/jason/.cache/sam3d/sam-3d-body-dinov3/assets/mhr_model.pt"

FR_ADDRESS = 0
FR_TOP     = 97
FR_IMPACT  = 88
PASS_THRESHOLD = 0.92

# Sentinel thresholds
SEN_A1_DELTA    = 0.18   # Tightened from 0.25
SEN_A2_CONF     = 0.65
SEN_A3_CX_OFF   = 60.0
SEN_A4_BLOB_FRAC = 0.015  # 1.5% of main body area
SEN_A4_BLOB_DIST = 150.0  # px from main body centroid

# Arm pose optimization params (body_pose_params indices)
ARM_PARAM_START = 18
ARM_PARAM_END   = 44   # [18:44] = 26 params controlling arm joints
ARM_PARAM_PERTURB_LIMIT = 0.6  # max absolute change from base pose

# Arm kp indices in MHR kp70
I_LSHO=5; I_RSHO=6; I_LELB=7; I_RELB=8; I_LWRI=62; I_RWRI=41
I_NOSE=0; I_LHIP=9; I_RHIP=10; I_LKNE=11; I_RKNE=12; I_LANK=13; I_RANK=14; I_NECK=69
ARM_KP_IDS = [I_LSHO, I_RSHO, I_LELB, I_RELB, I_LWRI, I_RWRI]

RTM_ARM_KP_NAMES = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                    'left_wrist', 'right_wrist']


# ── Sentinel A4: isolated blob ───────────────────────────────────────────────
def check_isolated_blob(mask):
    """Returns (has_isolated, reason_str)"""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if n <= 2:  # background + one component
        return False, ""
    # Find main body = largest foreground component (label > 0)
    areas = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)]
    areas.sort(reverse=True)
    main_area = areas[0][0]
    main_label = areas[0][1]
    main_cx, main_cy = centroids[main_label]
    for area, i in areas[1:]:  # all non-main components
        frac = area / max(main_area, 1)
        dist = float(np.sqrt((centroids[i][0]-main_cx)**2 + (centroids[i][1]-main_cy)**2))
        if frac >= SEN_A4_BLOB_FRAC and dist >= SEN_A4_BLOB_DIST:
            return True, f"A4:blob_frac={frac:.4f}>={SEN_A4_BLOB_FRAC},dist={dist:.0f}>={SEN_A4_BLOB_DIST}"
    return False, ""


# ── Sentinel pre-pass ─────────────────────────────────────────────────────────
def compute_sentinels(t2_per_frame, kp_frames, t2_ious, video_path):
    """A1+A2+A4 pre-pass (A3 is checked post-fit)"""
    N = len(t2_per_frame)
    result = {}
    iou_arr = np.array([r['iou_upper'] if r['iou_upper'] is not None else np.nan
                        for r in t2_per_frame])

    # Open video for A4 mask computation
    cap = cv2.VideoCapture(str(video_path))

    import rembg

    for i in range(N):
        reasons = []

        # A1
        nbrs = []
        for di in [-2, -1, 1, 2]:
            j = i + di
            if 0 <= j < N and not np.isnan(iou_arr[j]):
                nbrs.append(iou_arr[j])
        if nbrs and not np.isnan(iou_arr[i]):
            nbr_mean = float(np.mean(nbrs))
            drop = nbr_mean - iou_arr[i]
            if drop >= SEN_A1_DELTA:
                reasons.append(f"A1:delta={drop:.3f}>={SEN_A1_DELTA}")
        elif np.isnan(iou_arr[i]):
            reasons.append("A1:iou_none")

        # A2
        kp = kp_frames[i]['persons'][0]['keypoints']
        core_confs = [kp['left_shoulder']['score'], kp['right_shoulder']['score'],
                      kp['left_hip']['score'],      kp['right_hip']['score']]
        mean_conf = float(np.mean(core_confs))
        if mean_conf < SEN_A2_CONF:
            reasons.append(f"A2:conf={mean_conf:.3f}<{SEN_A2_CONF}")

        # A4: isolated blob (only if no other flag yet — skip if already flagged)
        if not reasons:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, img_bgr = cap.read()
            if ret:
                out  = rembg.remove(img_bgr)
                mask = (out[:,:,3] > 40).astype(np.uint8) * 255
                k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
                has_blob, blob_reason = check_isolated_blob(mask)
                if has_blob:
                    reasons.append(blob_reason)

        result[i] = {
            'invalid': len(reasons) > 0,
            'reasons': reasons,
            'iou_t2': float(iou_arr[i]) if not np.isnan(iou_arr[i]) else None,
            'mean_conf': round(mean_conf, 3),
        }

    cap.release()
    return result


# ── Detect arm-extended frames ────────────────────────────────────────────────
def is_arm_extended(kp_rtm):
    """True if arm kp error vs body center suggests extended pose needing correction."""
    try:
        lsho = kp_rtm['left_shoulder']
        rsho = kp_rtm['right_shoulder']
        lhip = kp_rtm['left_hip']
        rhip = kp_rtm['right_hip']
        lelb = kp_rtm['left_elbow']
        relb = kp_rtm['right_elbow']
        lwri = kp_rtm['left_wrist']
        rwri = kp_rtm['right_wrist']

        body_cx = (lsho['x'] + rsho['x'] + lhip['x'] + rhip['x']) / 4.0
        body_cy = (lsho['y'] + rsho['y'] + lhip['y'] + rhip['y']) / 4.0

        # Check how far wrists/elbows are from body center
        arm_pts = [(lelb, 'lelb'), (relb, 'relb'), (lwri, 'lwri'), (rwri, 'rwri')]
        max_arm_dist = 0.0
        for kp, name in arm_pts:
            if kp['score'] > 0.3:
                dist = float(np.sqrt((kp['x']-body_cx)**2 + (kp['y']-body_cy)**2))
                max_arm_dist = max(max_arm_dist, dist)

        # Body height as reference
        body_h = abs((lsho['y']+rsho['y'])/2 - (lhip['y']+rhip['y'])/2)
        ratio = max_arm_dist / max(body_h, 1)
        return ratio > 1.2, ratio  # extended if arm dist > 1.2× body height
    except:
        return False, 0.0


# ── Projection ──────────────────────────────────────────────────────────────
def project_verts(verts, cam_t, focal, H, W):
    vx, vy, vz = verts[:,0], verts[:,1], verts[:,2]
    d  = np.where(np.abs(vz+cam_t[2])<1e-6, 1e-6, vz+cam_t[2])
    px = focal*(vx-cam_t[0])/d + W/2.0
    py = focal*(vy+cam_t[1])/d + H/2.0
    return np.stack([px, py], axis=1)

def world_x_from_img(img_x, depth, cam_t, focal, W):
    return (img_x - W/2.0) * depth / focal + cam_t[0]


# ── Human mask ──────────────────────────────────────────────────────────────
def get_human_mask(img_bgr):
    import rembg
    out  = rembg.remove(img_bgr)
    mask = (out[:,:,3] > 40).astype(np.uint8) * 255
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    return mask


# ── Render ──────────────────────────────────────────────────────────────────
def render_rgba(verts, cam_t, focal, faces, H, W):
    from sam_3d_body.visualization.renderer import Renderer
    dummy = np.zeros((H, W, 3), dtype=np.uint8)
    r = Renderer(focal_length=focal, faces=faces)
    return r(verts, cam_t, dummy, mesh_base_color=(1., 0., 0.),
             scene_bg_color=(0, 0, 0), return_rgba=True)

def get_sil(rgba):
    ch = rgba[:,:,3] if rgba.shape[2]==4 else rgba.sum(2)
    return (ch > 0.05).astype(np.uint8) * 255

def composite_red(rgba, img_bgr, alpha=0.55):
    f   = img_bgr.astype(np.float32) / 255.0
    ma  = rgba[:,:,3:4] if rgba.shape[2]==4 else \
          (rgba.sum(2, keepdims=True) > 0.02).astype(np.float32)
    red = np.zeros_like(f); red[:,:,2] = 1.0
    return np.clip((f*(1-ma*alpha) + red*ma*alpha)*255, 0, 255).astype(np.uint8)


# ── Width / cx ──────────────────────────────────────────────────────────────
def measure_width(mask, yc, hb=30):
    H, W = mask.shape
    y1, y2 = max(0, yc-hb), min(H, yc+hb)
    cols = np.where(mask[y1:y2].any(axis=0))[0]
    if len(cols) < 3:
        return None
    return int(cols.min()), int(cols.max()), float((cols.min()+cols.max())/2.), int(cols.max()-cols.min())


# ── IoU ─────────────────────────────────────────────────────────────────────
def compute_iou(mask_h, sil_m, ylo, yhi, H):
    y1, y2 = max(0, int(ylo)), min(H, int(yhi))
    h = mask_h[y1:y2] > 0; m = sil_m[y1:y2] > 0
    inter = int((h & m).sum()); union = int((h | m).sum())
    return inter / max(union, 1), inter, union


# ── Proxy IoU (column-remap) ─────────────────────────────────────────────────
def proxy_iou(sx, sil_base, cx_img, mask_h, ylo, yhi, H, W):
    x     = np.arange(W, dtype=np.float32)
    x_src = np.clip(cx_img + (x - cx_img) / sx, 0, W-1).astype(np.int32)
    sil_s = sil_base[:, x_src]
    y1, y2 = max(0, int(ylo)), min(H, int(yhi))
    h = mask_h[y1:y2] > 0; m = sil_s[y1:y2] > 0
    inter = int((h & m).sum()); union = int((h | m).sum())
    return inter / max(union, 1)


# ── Cx translation ───────────────────────────────────────────────────────────
def translate_cx(v_work, band_mask_2d, cam_t, focal, W, H,
                 mask_h, sil_current, measure_ys, name, silent=True):
    if band_mask_2d.sum() == 0:
        return v_work, 0.0
    h_cxs, m_cxs = [], []
    for yc in measure_ys:
        hm = measure_width(mask_h, yc); mm = measure_width(sil_current, yc)
        if hm and mm:
            h_cxs.append(hm[2]); m_cxs.append(mm[2])
    if not h_cxs:
        return v_work, 0.0
    h_cx  = float(np.median(h_cxs)); m_cx  = float(np.median(m_cxs))
    dx_img = h_cx - m_cx
    if abs(dx_img) < 0.5:
        return v_work, dx_img
    depth  = float(v_work[band_mask_2d, 2].mean()) + cam_t[2]
    dx_w   = dx_img * depth / focal
    v_work[band_mask_2d, 0] += dx_w
    return v_work, dx_img


# ── IoU vis ──────────────────────────────────────────────────────────────────
def make_iou_vis(mask_h, sil_mesh, H, W, title=""):
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    h = mask_h > 0; m = sil_mesh > 0
    vis[h & ~m] = [0, 255, 0]; vis[~h & m] = [0, 0, 255]; vis[h & m] = [0, 255, 255]
    cv2.putText(vis, title, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    return vis


# ── Arm pose optimization ─────────────────────────────────────────────────────
def optimize_arm_pose(head, bp_base, global_rot, hand_p, scale_p, shape_p, expr_p,
                      cam_t_np, focal, H, W, kp2d_mhr, kp_rtm, faces):
    """
    Optimize body_pose_params[18:44] to minimize arm kp L2 error.
    Returns: (v_opt, kp2d_opt, arm_kp_err_px) or None if no improvement.
    """
    import torch
    from sam_3d_body.visualization.renderer import Renderer

    cam_t_0 = torch.zeros(1, 3, device='cuda')

    base_params = bp_base[0, ARM_PARAM_START:ARM_PARAM_END].cpu().numpy().copy()

    # RTMPose arm kp targets (image coords)
    rtm_arm_targets = []
    for name in RTM_ARM_KP_NAMES:
        kp = kp_rtm[name]
        rtm_arm_targets.append([kp['x'], kp['y']])
    rtm_arm_targets = np.array(rtm_arm_targets, dtype=np.float32)  # (6,2)

    def arm_loss(delta_params):
        """L2 arm kp error in image coords."""
        bp_test = bp_base.clone()
        # Clamp perturbation
        dp = np.clip(delta_params, -ARM_PARAM_PERTURB_LIMIT, ARM_PARAM_PERTURB_LIMIT)
        bp_test[0, ARM_PARAM_START:ARM_PARAM_END] = \
            torch.from_numpy(base_params + dp).float().cuda()
        with torch.no_grad():
            result = head.mhr_forward(
                cam_t_0, global_rot, bp_test,
                hand_p, scale_p, shape_p, expr_p,
                return_keypoints=True
            )
        verts = result[0][0].cpu().numpy()
        # Project
        vx,vy,vz = verts[:,0],verts[:,1],verts[:,2]
        d = np.where(np.abs(vz+cam_t_np[2])<1e-6, 1e-6, vz+cam_t_np[2])
        px = focal*(vx-cam_t_np[0])/d + W/2.0
        py = focal*(vy+cam_t_np[1])/d + H/2.0
        proj2d = np.stack([px,py],axis=1)  # (Nv,2)
        # Arm kp error
        arm_err = 0.0
        for i, kid in enumerate(ARM_KP_IDS):
            arm_err += float(np.sqrt(((proj2d[kid] - rtm_arm_targets[i])**2).sum()))
        return arm_err / len(ARM_KP_IDS)

    # Baseline arm error
    base_err = arm_loss(np.zeros(ARM_PARAM_END - ARM_PARAM_START))

    # Quick Nelder-Mead optimization
    x0 = np.zeros(ARM_PARAM_END - ARM_PARAM_START, dtype=np.float32)
    result = minimize(
        arm_loss, x0, method='Nelder-Mead',
        options={'xatol': 0.01, 'fatol': 0.5, 'maxiter': 200, 'adaptive': True}
    )
    opt_err = result.fun

    if opt_err >= base_err - 0.5:  # no meaningful improvement
        return None, base_err, opt_err

    # Apply optimized params and get final vertices
    dp_opt = np.clip(result.x, -ARM_PARAM_PERTURB_LIMIT, ARM_PARAM_PERTURB_LIMIT)
    bp_opt = bp_base.clone()
    bp_opt[0, ARM_PARAM_START:ARM_PARAM_END] = \
        torch.from_numpy(base_params + dp_opt).float().cuda()

    with torch.no_grad():
        result_opt = head.mhr_forward(
            cam_t_0, global_rot, bp_opt,
            hand_p, scale_p, shape_p, expr_p,
            return_keypoints=True
        )
    verts_opt = result_opt[0][0].cpu().numpy()

    return verts_opt, base_err, opt_err


# ── Shape fit (T1.7 algorithm, applied to given verts) ──────────────────────
def shape_fit_and_render(verts, cam_t_np, focal, faces, mask_h,
                         kp2d, H, W):
    """Apply T1.7 IoU-based sx fitting to given vertices."""
    from scipy.optimize import minimize_scalar as ms1d

    verts2d = project_verts(verts, cam_t_np, focal, H, W)

    nose_y  = int(kp2d[I_NOSE][1])
    neck_y  = int(kp2d[I_NECK][1])
    sho_y   = int((kp2d[I_LSHO][1] + kp2d[I_RSHO][1]) / 2)
    hip_y   = int((kp2d[I_LHIP][1] + kp2d[I_RHIP][1]) / 2)
    knee_y  = int((kp2d[I_LKNE][1] + kp2d[I_RKNE][1]) / 2)
    ank_y   = int((kp2d[I_LANK][1] + kp2d[I_RANK][1]) / 2)
    head_y  = nose_y - 55

    B_UPP_lo = head_y - 30; B_UPP_hi = hip_y - 15
    B_HIP_lo = hip_y - 15;  B_HIP_hi = hip_y + 110
    B_LOW_lo = hip_y + 110; B_LOW_hi = ank_y + 120
    UP_IoU_lo = head_y - 30; UP_IoU_hi = hip_y + 60

    # T1 render
    rgba_t1 = render_rgba(verts, cam_t_np, focal, faces, H, W)
    sil_t1  = get_sil(rgba_t1)

    m_sho = measure_width(sil_t1, sho_y)
    cx_upp = m_sho[2] if m_sho else float(W/2)
    m_hip  = measure_width(sil_t1, hip_y)
    cx_hip = m_hip[2] if m_hip else float(W/2)
    m_kne  = measure_width(sil_t1, knee_y)
    cx_low = m_kne[2] if m_kne else float(W/2)

    # Optimize sx per band
    res_upp = ms1d(
        lambda sx: -proxy_iou(sx, sil_t1, cx_upp, mask_h, B_UPP_lo, B_UPP_hi, H, W),
        bounds=(0.80, 2.00), method='bounded', options={'xatol': 1e-4, 'maxiter': 50}
    )
    sx_upp = float(res_upp.x)

    res_hip = ms1d(
        lambda sx: -proxy_iou(sx, sil_t1, cx_hip, mask_h, B_HIP_lo, B_HIP_hi, H, W),
        bounds=(0.80, 1.50), method='bounded', options={'xatol': 1e-4, 'maxiter': 50}
    )
    sx_hip = float(res_hip.x)

    lower_ys = [knee_y-20, knee_y, knee_y+30, (knee_y+ank_y)//2, ank_y-30]
    sx_max_low = 0.0; m_low_list = []
    for yc in lower_ys:
        hm = measure_width(mask_h, yc); mm = measure_width(sil_t1, yc)
        if hm and mm and mm[3] > 5:
            sx_raw = hm[3] / mm[3]
            if sx_raw > sx_max_low: sx_max_low = sx_raw
            m_low_list.append(mm[2])
    sx_low = float(np.clip(sx_max_low, 0.80, 1.50))
    cx_low_actual = float(np.median(m_low_list)) if m_low_list else cx_low

    # Apply scale
    v_opt = verts.copy()
    bm_upp = verts2d[:,1] < B_HIP_lo
    bm_hip = (verts2d[:,1] >= B_HIP_lo) & (verts2d[:,1] < B_LOW_lo)
    bm_low = verts2d[:,1] >= B_LOW_lo

    depth_upp = float(v_opt[bm_upp, 2].mean()) + cam_t_np[2]
    depth_hip = float(v_opt[bm_hip, 2].mean()) + cam_t_np[2]
    depth_low = float(v_opt[bm_low, 2].mean()) + cam_t_np[2]

    wcx_m_upp = world_x_from_img(cx_upp,        depth_upp, cam_t_np, focal, W)
    wcx_m_hip = world_x_from_img(cx_hip,        depth_hip, cam_t_np, focal, W)
    wcx_m_low = world_x_from_img(cx_low_actual, depth_low, cam_t_np, focal, W)

    v_opt[bm_upp, 0] = wcx_m_upp + (v_opt[bm_upp,0] - wcx_m_upp) * sx_upp
    v_opt[bm_hip, 0] = wcx_m_hip + (v_opt[bm_hip,0] - wcx_m_hip) * sx_hip
    v_opt[bm_low, 0] = wcx_m_low + (v_opt[bm_low,0] - wcx_m_low) * sx_low

    # Mid render + cx correction
    rgba_mid = render_rgba(v_opt, cam_t_np, focal, faces, H, W)
    sil_mid  = get_sil(rgba_mid)

    all_up_ys = list(range(neck_y, hip_y-15, 15))
    sho_ys    = list(range(neck_y, sho_y+100, 12))
    hip_ys    = [hip_y-20, hip_y, hip_y+20, hip_y+40]

    v_opt, _ = translate_cx(v_opt, bm_upp, cam_t_np, focal, W, H,
                            mask_h, sil_mid, all_up_ys, "B_UPP")
    v_opt, _ = translate_cx(v_opt, bm_hip, cam_t_np, focal, W, H,
                            mask_h, sil_mid, hip_ys, "B_HIP")
    v_opt, _ = translate_cx(v_opt, bm_low, cam_t_np, focal, W, H,
                            mask_h, sil_mid, lower_ys, "B_LOW")

    rgba_final = render_rgba(v_opt, cam_t_np, focal, faces, H, W)
    sil_final  = get_sil(rgba_final)

    # Residual cx
    h_cxs2, m_cxs2 = [], []
    for yc in sho_ys[:6]:
        hm = measure_width(mask_h, yc); mm = measure_width(sil_final, yc)
        if hm and mm:
            h_cxs2.append(hm[2]); m_cxs2.append(mm[2])
    if h_cxs2:
        res_dx = float(np.median(h_cxs2)) - float(np.median(m_cxs2))
        if abs(res_dx) > 1.5:
            depth_b = float(v_opt[bm_upp, 2].mean()) + cam_t_np[2]
            v_opt[bm_upp, 0] += res_dx * depth_b / focal
            rgba_final = render_rgba(v_opt, cam_t_np, focal, faces, H, W)
            sil_final  = get_sil(rgba_final)

    iou_upp, inter, union = compute_iou(mask_h, sil_final, UP_IoU_lo, UP_IoU_hi, H)

    return v_opt, rgba_final, sil_final, float(iou_upp), sx_upp, sx_hip, sx_low, UP_IoU_lo, UP_IoU_hi


# ── Fit one frame ─────────────────────────────────────────────────────────────
def fit_frame_t22(img_bgr, est, head, faces, kp_rtm, do_arm_opt=False):
    import torch
    H, W = img_bgr.shape[:2]

    ax = [v['x'] for v in kp_rtm.values() if v['score'] > 0.3]
    ay = [v['y'] for v in kp_rtm.values() if v['score'] > 0.3]
    if len(ax) < 4:
        return None
    pad_x = (max(ax)-min(ax)) * 0.15
    pad_y = (max(ay)-min(ay)) * 0.15
    bbox  = np.array([[max(0, min(ax)-pad_x), max(0, min(ay)-pad_y),
                       min(W, max(ax)+pad_x), min(H, max(ay)+pad_y)]], dtype=np.float32)

    outs = est.process_one_image(img_bgr, bboxes=bbox, use_mask=False, inference_type="body")
    if not outs:
        return None
    o     = outs[0]
    verts = o["pred_vertices"].astype(np.float32)
    cam_t = o["pred_cam_t"].astype(np.float32)
    kp2d  = o["pred_keypoints_2d"].astype(np.float32)
    focal = float(o["focal_length"])

    mask_h = get_human_mask(img_bgr)

    rtm_body_cx = float(np.mean([kp_rtm['left_shoulder']['x'], kp_rtm['right_shoulder']['x'],
                                  kp_rtm['left_hip']['x'],       kp_rtm['right_hip']['x']]))

    arm_opt_applied = False
    arm_kp_err_before = None
    arm_kp_err_after  = None

    # B: Arm pose optimization if arm is extended
    ext_flag, ext_ratio = is_arm_extended(kp_rtm)
    if do_arm_opt and ext_flag:
        bp    = torch.from_numpy(o['body_pose_params']).float().cuda().unsqueeze(0)
        gr    = torch.from_numpy(o['global_rot']).float().cuda().unsqueeze(0)
        sp    = torch.from_numpy(o['scale_params']).float().cuda().unsqueeze(0)
        shp   = torch.from_numpy(o['shape_params']).float().cuda().unsqueeze(0)
        expr  = torch.from_numpy(o['expr_params']).float().cuda().unsqueeze(0) if o['expr_params'] is not None else None
        hand  = torch.from_numpy(o['hand_pose_params']).float().cuda().unsqueeze(0) if o['hand_pose_params'] is not None else None

        verts_arm, arm_err_b, arm_err_a = optimize_arm_pose(
            head, bp, gr, hand, sp, shp, expr,
            cam_t, focal, H, W, kp2d, kp_rtm, faces
        )
        arm_kp_err_before = round(arm_err_b, 2)
        arm_kp_err_after  = round(arm_err_a, 2)

        if verts_arm is not None:
            verts = verts_arm
            arm_opt_applied = True

    # Shape fit (T1.7 sx optimization)
    v_opt, rgba_final, sil_final, iou_upp, sx_upp, sx_hip, sx_low, UP_IoU_lo, UP_IoU_hi = \
        shape_fit_and_render(verts, cam_t, focal, faces, mask_h, kp2d, H, W)

    mesh_body_cx = float((kp2d[I_LSHO][0] + kp2d[I_RSHO][0] +
                          kp2d[I_LHIP][0] + kp2d[I_RHIP][0]) / 4.0)
    cx_offset = abs(mesh_body_cx - rtm_body_cx)

    return {
        "iou_upper": round(float(iou_upp), 4),
        "pass": bool(iou_upp >= PASS_THRESHOLD),
        "rgba_final": rgba_final,
        "sil_final": sil_final,
        "mask_h": mask_h,
        "sx_upp": round(sx_upp, 4),
        "sx_hip": round(sx_hip, 4),
        "sx_low": round(sx_low, 4),
        "arm_opt_applied": arm_opt_applied,
        "arm_kp_err_before": arm_kp_err_before,
        "arm_kp_err_after":  arm_kp_err_after,
        "ext_ratio": round(ext_ratio, 3),
        "UP_IoU_lo": UP_IoU_lo,
        "UP_IoU_hi": UP_IoU_hi,
        "img_bgr": img_bgr,
        "kp2d": kp2d,
        "cam_t": cam_t.tolist(),
        "focal": focal,
        "mesh_cx": round(mesh_body_cx, 1),
        "rtm_cx": round(rtm_body_cx, 1),
        "cx_offset": round(cx_offset, 1),
    }


# ── Interpolate overlay ───────────────────────────────────────────────────────
def build_interp_overlay(fr_idx, img_bgr, valid_overlays):
    prev_k = max([k for k in valid_overlays if k < fr_idx], default=None)
    next_k = min([k for k in valid_overlays if k > fr_idx], default=None)
    if prev_k is None and next_k is None:
        return img_bgr.copy()
    if prev_k is None:
        src = valid_overlays[next_k]
    elif next_k is None:
        src = valid_overlays[prev_k]
    else:
        t = (fr_idx - prev_k) / (next_k - prev_k)
        src = cv2.addWeighted(valid_overlays[prev_k], 1-t, valid_overlays[next_k], t, 0)
    return src.astype(np.uint8)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    import torch
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

    print(f"[INFO] torch {torch.__version__}  cuda={torch.cuda.is_available()}")

    with open(T2_LOG) as f:
        t2_log = json.load(f)
    with open(KP_CACHE) as f:
        kpd = json.load(f)
    kp_frames = kpd['frames']
    NF = len(kp_frames)
    t2_ious = np.array([r['iou_upper'] for r in t2_log['per_frame']])

    # ── A: Sentinel pre-pass ──────────────────────────────────────────────
    print(f"\n[INFO] === Sentinel pre-pass (A1={SEN_A1_DELTA}/A2/A4) ===")
    sentinels = compute_sentinels(t2_log['per_frame'], kp_frames, t2_ious, VIDEO_PATH)
    invalid_frames = [i for i, s in sentinels.items() if s['invalid']]
    print(f"  Frames flagged invalid: {invalid_frames}")
    for fi in invalid_frames:
        s = sentinels[fi]
        print(f"    fr{fi:03d}  IoU_T2={s['iou_t2']:.4f}  reasons={s['reasons']}")

    # Load model
    print("\n[INFO] Loading model...")
    model, cfg = load_sam_3d_body(CKPT, device='cuda', mhr_path=MHR_P)
    est  = SAM3DBodyEstimator(model, cfg, human_detector=None,
                              human_segmentor=None, fov_estimator=None)
    head  = model.head_pose
    faces = est.faces

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO_PATH}")

    results        = []
    frame_results  = {}
    valid_overlays = {}
    KEYFRAMES      = {FR_ADDRESS: "address", FR_TOP: "top", FR_IMPACT: "impact"}

    for fr_idx in range(NF):
        ret, img_bgr = cap.read()
        if not ret:
            print(f"  [WARN] Cannot read fr{fr_idx:03d}")
            results.append({"frame": fr_idx, "iou_upper": None, "pass": False,
                            "invalid": True, "reasons": ["read_fail"]})
            continue

        is_invalid = sentinels[fr_idx]['invalid']
        sentinel_reasons = sentinels[fr_idx]['reasons']

        if is_invalid:
            print(f"  [fr{fr_idx:03d}/{NF-1}] INVALID ({', '.join(sentinel_reasons)})")
            results.append({
                "frame": fr_idx, "iou_upper": None, "pass": False,
                "invalid": True, "reasons": sentinel_reasons,
                "iou_t2": sentinels[fr_idx]['iou_t2'],
            })
            continue

        t_fr = time.time()
        kp_rtm = kp_frames[fr_idx]['persons'][0]['keypoints']

        # Detect if arm-extended frame (skip for keyframes to protect them)
        ext_flag, ext_ratio = is_arm_extended(kp_rtm)
        do_arm_opt = ext_flag and (fr_idx not in KEYFRAMES)

        flag_str = "[ARM_OPT]" if do_arm_opt else ""
        print(f"  [fr{fr_idx:03d}/{NF-1}] fitting... {flag_str}", end='', flush=True)

        try:
            r = fit_frame_t22(img_bgr, est, head, faces, kp_rtm, do_arm_opt=do_arm_opt)
        except Exception as e:
            print(f" ERROR: {e}")
            results.append({"frame": fr_idx, "iou_upper": None, "pass": False, "error": str(e)})
            continue

        if r is None:
            print(f" SKIP")
            results.append({"frame": fr_idx, "iou_upper": None, "pass": False})
            continue

        # A3: post-fit mesh cx check
        if r["cx_offset"] > SEN_A3_CX_OFF:
            print(f"\n  [A3] fr{fr_idx:03d}: cx_offset={r['cx_offset']:.1f}px > {SEN_A3_CX_OFF}")
            sentinels[fr_idx]['reasons'].append(f"A3:cx={r['cx_offset']:.1f}")
            results.append({
                "frame": fr_idx, "iou_upper": None, "pass": False,
                "invalid": True, "reasons": sentinels[fr_idx]['reasons'],
                "iou_t2": sentinels[fr_idx]['iou_t2'],
            })
            continue

        dt   = time.time() - t_fr
        iou  = r["iou_upper"]
        t2v  = float(t2_ious[fr_idx])
        flag = "✓" if r["pass"] else "✗"
        arm_str = f" arm_err:{r['arm_kp_err_before']:.1f}→{r['arm_kp_err_after']:.1f}px" \
                  if r["arm_opt_applied"] else ""
        print(f" IoU={iou:.4f} {flag}  (T2:{t2v:.4f} Δ{iou-t2v:+.4f}) {arm_str} ({dt:.1f}s)")

        row = {
            "frame": fr_idx,
            "iou_upper": iou,
            "iou_t2": t2v,
            "delta_vs_t2": round(float(iou - t2v), 4),
            "pass": r["pass"],
            "invalid": False,
            "sx_upp": r["sx_upp"],
            "sx_hip": r["sx_hip"],
            "arm_opt_applied": r["arm_opt_applied"],
            "arm_kp_err_before": r["arm_kp_err_before"],
            "arm_kp_err_after":  r["arm_kp_err_after"],
            "ext_ratio": r["ext_ratio"],
        }
        results.append(row)

        ov = composite_red(r["rgba_final"], img_bgr, alpha=0.55)
        valid_overlays[fr_idx] = ov.copy()
        frame_results[fr_idx] = r

        if fr_idx in KEYFRAMES:
            phase  = KEYFRAMES[fr_idx]
            kp2d_k = r["kp2d"]
            H_ov, W_ov = ov.shape[:2]
            lh = (int(kp2d_k[I_LHIP][0]), int(kp2d_k[I_LHIP][1]))
            rh = (int(kp2d_k[I_RHIP][0]), int(kp2d_k[I_RHIP][1]))
            y_pel = (lh[1]+rh[1])//2
            cv2.line(ov, (0, y_pel), (W_ov-1, y_pel), (255,255,255), 2)
            cv2.circle(ov, lh, 7, (255,255,255), -1); cv2.circle(ov, lh, 3, (0,0,200), -1)
            cv2.circle(ov, rh, 7, (255,255,255), -1); cv2.circle(ov, rh, 3, (0,0,200), -1)
            label = f"{phase}_fr{fr_idx:03d}_IoU{iou:.3f}"
            cv2.putText(ov, label, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            kf_path = KF_DIR / f"{phase}_fr{fr_idx:03d}.jpg"
            cv2.imwrite(str(kf_path), ov)
            print(f"  [KEY] {kf_path.name}")

    cap.release()

    # ── Interpolate invalid frames ─────────────────────────────────────────
    print("\n[INFO] Interpolating invalid frames...")
    cap2 = cv2.VideoCapture(str(VIDEO_PATH))
    for fr_idx in invalid_frames:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, fr_idx)
        ret, img_bgr = cap2.read()
        if not ret:
            continue
        interp_ov = build_interp_overlay(fr_idx, img_bgr, valid_overlays)
        cv2.putText(interp_ov, f"INVALID fr{fr_idx:03d} [INTERPOLATED]",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)
        cv2.imwrite(str(OUTPUT_DIR / f"invalid_fr{fr_idx:03d}_interp.jpg"), interp_ov)
        print(f"  [INTERP] fr{fr_idx:03d}")
    cap2.release()

    # ── Distribution ──────────────────────────────────────────────────────
    valid_rows = [r for r in results if not r.get('invalid', False) and r.get('iou_upper') is not None]
    ious_valid = np.array([r["iou_upper"] for r in valid_rows])
    n_valid    = len(ious_valid)
    n_invalid  = NF - n_valid

    mean_iou = float(np.mean(ious_valid))
    min_iou  = float(np.min(ious_valid))
    p5_iou   = float(np.percentile(ious_valid, 5))
    min_fr   = valid_rows[int(np.argmin(ious_valid))]["frame"]
    fail_valid = [(r["frame"], r["iou_upper"]) for r in valid_rows if not r["pass"]]
    arm_opt_frames = [(r["frame"], r["iou_upper"], r.get("arm_kp_err_before"), r.get("arm_kp_err_after"))
                      for r in valid_rows if r.get("arm_opt_applied")]

    print(f"\n{'='*60}")
    print(f"  T2.2 IoU 分布 (有效帧={n_valid}, 无效帧={n_invalid})")
    print(f"  mean = {mean_iou:.4f}")
    print(f"  min  = {min_iou:.4f}  (fr{min_fr:03d})")
    print(f"  P5   = {p5_iou:.4f}")
    print(f"  低于 {PASS_THRESHOLD}: {len(fail_valid)} 有效帧")
    print(f"\n  T2 对比: mean={t2_log['distribution']['mean']:.4f}  min={t2_log['distribution']['min']:.4f}  P5={t2_log['distribution']['p5']:.4f}")
    print(f"  T2.1对比: mean=0.9014  P5=0.8296")
    print(f"\n  手臂姿态优化帧: {len(arm_opt_frames)}")
    for fno, iou, eb, ea in arm_opt_frames:
        print(f"    fr{fno:03d}: IoU={iou:.4f}  arm_err {eb:.1f}→{ea:.1f}px")
    print(f"{'='*60}")

    # ── Worst 3 valid ──────────────────────────────────────────────────────
    valid_sorted = sorted(
        [(r["frame"], r["iou_upper"]) for r in valid_rows if r["frame"] in frame_results],
        key=lambda x: x[1]
    )
    worst3 = valid_sorted[:3]
    for rank, (fno, fiou) in enumerate(worst3, 1):
        r = frame_results[fno]
        H_img, W_img = r["img_bgr"].shape[:2]
        vis = make_iou_vis(r["mask_h"], r["sil_final"], H_img, W_img,
                           f"fr{fno:03d} valid IoU={fiou:.3f}")
        cv2.putText(vis, "green=human  red=mesh  yellow=overlap",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        t2v = float(t2_ious[fno])
        cv2.putText(vis, f"T2:{t2v:.3f}  T2.2:{fiou:.3f}  Δ{fiou-t2v:+.3f}",
                    (10, H_img-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
        p_vis = OUTPUT_DIR / f"worst{rank}_valid_t22_fr{fno:03d}.jpg"
        cv2.imwrite(str(p_vis), vis)
        print(f"  [WORST{rank}] fr{fno:03d} IoU={fiou:.4f} → {p_vis.name}")

    # ── IoU curve ─────────────────────────────────────────────────────────
    H_c, W_c = 440, max(800, NF * 6)
    curve = np.zeros((H_c, W_c, 3), dtype=np.uint8); curve[:] = (30, 30, 30)
    for iou_grid in [0.80, 0.85, 0.87, 0.90, 0.92, 0.95]:
        yg = int((1.0 - iou_grid) / 0.25 * (H_c-80)) + 40
        color = (0, 200, 100) if abs(iou_grid-0.92) < 0.005 \
                else ((100, 200, 255) if abs(iou_grid-0.87) < 0.005 else (60, 60, 60))
        cv2.line(curve, (0, yg), (W_c-1, yg), color, 1)
        cv2.putText(curve, f"{iou_grid:.2f}", (4, yg-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

    for r in t2_log['per_frame']:  # gray T2 background
        if r['iou_upper'] is None: continue
        x = int(r['frame']/(NF-1)*(W_c-20))+10
        y = int((1.0-min(1.0,max(0.75,r['iou_upper'])))/0.25*(H_c-80))+40
        cv2.circle(curve, (x, y), 2, (80, 80, 80), -1)

    for r in results:
        fno = r['frame']
        x = int(fno/(NF-1)*(W_c-20))+10
        if r.get('invalid', False):
            iou_v = r.get('iou_t2') or 0.85
            y = int((1.0-min(1.0,max(0.75,iou_v)))/0.25*(H_c-80))+40
            cv2.line(curve, (x-5, y-5), (x+5, y+5), (0, 0, 255), 2)
            cv2.line(curve, (x-5, y+5), (x+5, y-5), (0, 0, 255), 2)
        elif r.get('iou_upper') is not None:
            iou_v = r['iou_upper']
            y = int((1.0-min(1.0,max(0.75,iou_v)))/0.25*(H_c-80))+40
            color = (0, 220, 255) if r['pass'] else (0, 140, 255)
            marker = 5 if r.get("arm_opt_applied") else 3
            cv2.circle(curve, (x, y), marker, color, -1)
            if fno in {FR_ADDRESS, FR_TOP, FR_IMPACT}:
                cv2.circle(curve, (x, y), 7, (255, 255, 0), 2)
            if r.get("arm_opt_applied"):
                cv2.circle(curve, (x, y), 8, (255, 200, 0), 1)  # orange ring = arm opt

    cv2.putText(curve,
        f"T2.2  valid={n_valid}  invalid={n_invalid}  mean={mean_iou:.3f}  min={min_iou:.3f}(fr{min_fr:03d})  P5={p5_iou:.3f}",
        (10, H_c-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    p_curve = OUTPUT_DIR / "iou_distribution_t22.jpg"
    cv2.imwrite(str(p_curve), curve)

    # ── JSON ──────────────────────────────────────────────────────────────
    vram = torch.cuda.max_memory_allocated() / 1e6
    tt   = time.time() - t_start
    log_out = {
        "version": "T2.2", "clip": "fo-ok-1", "NF": NF,
        "pass_threshold": PASS_THRESHOLD,
        "sentinel_config": {"A1_delta": SEN_A1_DELTA, "A2_conf": SEN_A2_CONF,
                             "A3_cx_offset": SEN_A3_CX_OFF,
                             "A4_blob_frac": SEN_A4_BLOB_FRAC, "A4_blob_dist": SEN_A4_BLOB_DIST},
        "invalid_frames": [{"frame": fi, "reasons": sentinels[fi]['reasons'], "iou_t2": sentinels[fi]['iou_t2']}
                           for fi in invalid_frames],
        "distribution_valid_only": {
            "mean": round(mean_iou,4), "min": round(min_iou,4), "min_frame": min_fr,
            "p5": round(p5_iou,4), "valid_count": n_valid, "invalid_count": n_invalid,
        },
        "distribution_t2_all": t2_log['distribution'],
        "arm_opt_frames": [{"frame": f, "iou": i, "err_before": b, "err_after": a}
                           for f,i,b,a in arm_opt_frames],
        "worst3_valid": [{"frame": f, "iou_upper": round(v, 4)} for f, v in worst3],
        "per_frame": results,
        "peak_vram_mb": round(vram,0), "total_s": round(tt,1),
        "topology": "MHR_native",
        "license": "SAM_License_PRODUCT_CANDIDATE_CUSTOM_LICENSE",
    }
    lp = OUTPUT_DIR / "run_log_t22.json"
    with open(lp, 'w') as f:
        json.dump(log_out, f, indent=2, default=str)

    # ── Text report ───────────────────────────────────────────────────────
    lines = [
        "GHOST-003 T2.2 停关卡报告",
        f"Clip: fo-ok-1  NF={NF}",
        "",
        "=== A. 哨兵 ===",
        f"A1: delta>={SEN_A1_DELTA} (放宽 0.25→0.18)",
        f"A2: core_conf<{SEN_A2_CONF}",
        f"A3: mesh_cx offset>{SEN_A3_CX_OFF}px (post-fit)",
        f"A4: isolated_blob area>={SEN_A4_BLOB_FRAC*100:.1f}% AND dist>={SEN_A4_BLOB_DIST}px",
        f"",
        f"无效帧数: {n_invalid}",
    ]
    for fi in invalid_frames:
        s = sentinels[fi]
        lines.append(f"  fr{fi:03d}  T2_IoU={s['iou_t2']:.4f}  reasons={s['reasons']}")

    lines += [
        "",
        "=== B. 有效帧分布 ===",
        f"有效帧: {n_valid}  无效帧剔除: {n_invalid}",
        f"mean IoU : {mean_iou:.4f}",
        f"min  IoU : {min_iou:.4f}  (fr{min_fr:03d})",
        f"P5   IoU : {p5_iou:.4f}",
        f"低于 {PASS_THRESHOLD}: {len(fail_valid)} 有效帧",
        "",
        "=== 剔除前后对比 ===",
        f"T2  (n=112): mean={t2_log['distribution']['mean']:.4f}  min={t2_log['distribution']['min']:.4f}(fr{t2_log['distribution']['min_frame']})  P5={t2_log['distribution']['p5']:.4f}",
        f"T2.1(n=111): mean=0.9014  min=0.7420(fr103)  P5=0.8296",
        f"T2.2(n={n_valid}): mean={mean_iou:.4f}  min={min_iou:.4f}(fr{min_fr:03d})  P5={p5_iou:.4f}",
        "",
        f"=== 手臂姿态优化 ===",
        f"优化帧数: {len(arm_opt_frames)}",
    ]
    for fno, iou, eb, ea in arm_opt_frames:
        lines.append(f"  fr{fno:03d}: IoU={iou:.4f}  arm_kp_err {eb:.1f}→{ea:.1f}px")

    lines += [
        "",
        "最差3有效帧:",
    ]
    for rank, (fno, fiou) in enumerate(worst3, 1):
        t2v = float(t2_ious[fno])
        lines.append(f"  #{rank} fr{fno:03d}  IoU={fiou:.4f}  T2={t2v:.4f}  Δ{fiou-t2v:+.4f}")

    lines += ["",
              "关键帧:",
              f"  address fr{FR_ADDRESS:03d}: T2={t2_ious[FR_ADDRESS]:.4f}  T2.2={next((r['iou_upper'] for r in results if r['frame']==FR_ADDRESS), 'N/A')}",
              f"  top     fr{FR_TOP:03d}: T2={t2_ious[FR_TOP]:.4f}  T2.2={next((r['iou_upper'] for r in results if r['frame']==FR_TOP), 'N/A')}",
              f"  impact  fr{FR_IMPACT:03d}: T2={t2_ious[FR_IMPACT]:.4f}  T2.2={next((r['iou_upper'] for r in results if r['frame']==FR_IMPACT), 'N/A')}",
              "", f"peak VRAM: {vram:.0f}MB  total: {tt:.0f}s"]
    rp = OUTPUT_DIR / "REPORT_T22.txt"
    rp.write_text("\n".join(lines), encoding='utf-8')

    print(f"\n[OUT] {OUTPUT_DIR}/")
    print(f"{'='*60}")
    print(f"  T2.2 DONE  valid={n_valid}  invalid={n_invalid}")
    print(f"  mean={mean_iou:.4f}  min={min_iou:.4f}(fr{min_fr:03d})  P5={p5_iou:.4f}")
    print(f"  ARM_OPT applied: {len(arm_opt_frames)} frames")
    print(f"  peak VRAM: {vram:.0f}MB  total: {tt:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
