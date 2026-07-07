"""
ghost003_t21_sentinel_tighten.py  —  GHOST-003 T2.1
A. 崩帧哨兵: 3条判据联合，标记无效帧，不参与 IoU 统计
B. 动态帧紧化: 2D 联合优化 (sx + cy_shift) + 宽化搜索界 [0.80, 2.00]

哨兵判据 (满足任一 → 无效):
  A1: 单帧 IoU vs 滚动邻域均值 drop >= 0.25
  A2: 核心关键点 (lsho/rsho/lhip/rhip) 平均置信 < 0.65
  A3: mesh 躯干 cx vs RTM 躯干 cx 偏移 > 60px

无效帧: 用前后最近有效帧插值 overlay, 不参与 IoU 统计

B 优化升级 (对有效帧):
  - scipy.minimize (Nelder-Mead 2D): 对 B_UPP / B_HIP 各优化 [sx, cy_shift]
  - sx 搜索界: [0.80, 2.00] (宽化, 原 [0.90, 1.70])
  - cy_shift: [-50px, +50px] 允许纵向微调
  - 目标: 最大化 proxy IoU (column-remap + row-shift)

停关卡产出:
  - REPORT_T21.txt           有效帧分布 + 无效帧列表 + 剔除前后对比
  - run_log_t21.json         逐帧数据
  - iou_distribution_t21.jpg 无效帧用红叉标出
  - worst3_valid_frXXX.jpg   最差3有效帧 silhouette_compare
  - keyframe_overlay_t21/    address/top/impact 各一张

范围: 整段·真实姿态·不改姿态·不做动作修正·不碰球杆
授权: SAM License (PRODUCT_CANDIDATE_CUSTOM_LICENSE)
"""

import os, sys, time, json, pathlib
import numpy as np
import cv2
from scipy.optimize import minimize_scalar, minimize

SAM3D_REPO = "/home/jason/projects/sam-3d-body"
SWINGCUE   = "/home/jason/projects/swingcue-postest"
sys.path.insert(0, SAM3D_REPO)
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

OUTPUT_DIR = pathlib.Path(SWINGCUE) / "output" / "ghost003_t21"
KF_DIR     = OUTPUT_DIR / "keyframe_overlay_t21"
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
SEN_A1_DELTA    = 0.25   # IoU drop vs rolling neighbor mean
SEN_A2_CONF     = 0.65   # mean core kp confidence
SEN_A3_CX_OFF   = 60.0   # mesh cx vs RTM body cx offset (px)

# MHR70 indices
I_NOSE=0; I_LSHO=5; I_RSHO=6; I_LELB=7; I_RELB=8
I_LHIP=9; I_RHIP=10; I_LKNE=11; I_RKNE=12; I_LANK=13; I_RANK=14
I_RWRI=41; I_LWRI=62; I_NECK=69


# ── Sentinel pre-pass (uses T2 log + kp_cache, no model needed) ────────────
def compute_sentinels(t2_per_frame, kp_frames, t2_ious):
    """
    Returns dict: {frame_idx: {'invalid': bool, 'reasons': [str], ...}}
    Uses T2 IoU data + kp_cache confidence.
    A3 (mesh cx offset) is computed live per-frame during main loop.
    """
    N = len(t2_per_frame)
    result = {}

    # Build rolling neighbor mean (window ±2 frames, skip None)
    iou_arr = np.array([r['iou_upper'] if r['iou_upper'] is not None else np.nan
                        for r in t2_per_frame])

    for i in range(N):
        reasons = []

        # A1: IoU delta
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

        # A2: core kp confidence
        kp = kp_frames[i]['persons'][0]['keypoints']
        core_confs = [kp['left_shoulder']['score'], kp['right_shoulder']['score'],
                      kp['left_hip']['score'],      kp['right_hip']['score']]
        mean_conf = float(np.mean(core_confs))
        if mean_conf < SEN_A2_CONF:
            reasons.append(f"A2:conf={mean_conf:.3f}<{SEN_A2_CONF}")

        result[i] = {
            'invalid': len(reasons) > 0,
            'reasons': reasons,
            'iou_t2': float(iou_arr[i]) if not np.isnan(iou_arr[i]) else None,
            'mean_conf': round(mean_conf, 3),
        }

    return result


# ── Projection ──────────────────────────────────────────────────────────────
def project_verts(verts, cam_t, focal, H, W):
    vx, vy, vz = verts[:,0], verts[:,1], verts[:,2]
    d  = np.where(np.abs(vz+cam_t[2])<1e-6, 1e-6, vz+cam_t[2])
    px = focal*(vx-cam_t[0])/d + W/2.0
    py = focal*(vy+cam_t[1])/d + H/2.0
    return np.stack([px, py], axis=1)

def world_x_from_img(img_x, depth, cam_t, focal, W):
    return (img_x - W/2.0) * depth / focal + cam_t[0]

def world_y_from_img(img_y, depth, cam_t, focal, H):
    return (img_y - H/2.0) * depth / focal - cam_t[1]


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


# ── Width / cx measurement ──────────────────────────────────────────────────
def measure_width(mask, yc, hb=30):
    H, W = mask.shape
    y1, y2 = max(0, yc-hb), min(H, yc+hb)
    cols = np.where(mask[y1:y2].any(axis=0))[0]
    if len(cols) < 3:
        return None
    return int(cols.min()), int(cols.max()), float((cols.min()+cols.max())/2.), int(cols.max()-cols.min())


# ── IoU ──────────────────────────────────────────────────────────────────────
def compute_iou(mask_h, sil_m, ylo, yhi, H):
    y1, y2 = max(0, int(ylo)), min(H, int(yhi))
    h = mask_h[y1:y2] > 0
    m = sil_m[y1:y2]  > 0
    inter = int((h & m).sum())
    union = int((h | m).sum())
    return inter / max(union, 1), inter, union


# ── 2D proxy IoU: sx (x-scale) + dy (y-shift in pixels) ─────────────────────
def proxy_iou_2d(sx, dy, sil_base, cx_img, mask_h, ylo, yhi, H, W):
    """Column-remap by sx around cx, then row-shift by dy."""
    x     = np.arange(W, dtype=np.float32)
    x_src = np.clip(cx_img + (x - cx_img) / sx, 0, W-1).astype(np.int32)
    sil_s = sil_base[:, x_src]   # x-scaled

    # row-shift: sil_s[y] ← sil_s_shifted[y] = sil_s[y - dy]
    dy_int = int(round(dy))
    if dy_int != 0:
        sil_shifted = np.zeros_like(sil_s)
        if dy_int > 0:   # shift down: new[y] = old[y-dy]
            sil_shifted[dy_int:, :] = sil_s[:H-dy_int, :]
        else:            # shift up
            sil_shifted[:H+dy_int, :] = sil_s[-dy_int:, :]
    else:
        sil_shifted = sil_s

    y1, y2 = max(0, int(ylo)), min(H, int(yhi))
    h = mask_h[y1:y2] > 0
    m = sil_shifted[y1:y2] > 0
    inter = int((h & m).sum())
    union = int((h | m).sum())
    return inter / max(union, 1)


# ── cx translation ───────────────────────────────────────────────────────────
def translate_cx(v_work, band_mask_2d, cam_t, focal, W, H,
                 mask_h, sil_current, measure_ys, name, silent=True):
    if band_mask_2d.sum() == 0:
        return v_work, 0.0
    h_cxs, m_cxs = [], []
    for yc in measure_ys:
        hm = measure_width(mask_h, yc)
        mm = measure_width(sil_current, yc)
        if hm and mm:
            h_cxs.append(hm[2]); m_cxs.append(mm[2])
    if not h_cxs:
        return v_work, 0.0
    h_cx  = float(np.median(h_cxs))
    m_cx  = float(np.median(m_cxs))
    dx_img = h_cx - m_cx
    if abs(dx_img) < 0.5:
        return v_work, dx_img
    depth  = float(v_work[band_mask_2d, 2].mean()) + cam_t[2]
    dx_w   = dx_img * depth / focal
    if not silent:
        print(f"  [{name:20s}] cx {m_cx:.1f}→{h_cx:.1f} dx={dx_img:+.1f}px")
    v_work[band_mask_2d, 0] += dx_w
    return v_work, dx_img


# ── IoU visualisation ────────────────────────────────────────────────────────
def make_iou_vis(mask_h, sil_mesh, H, W, title=""):
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    h = mask_h > 0; m = sil_mesh > 0
    vis[h & ~m] = [0, 255, 0]
    vis[~h & m] = [0, 0, 255]
    vis[h & m]  = [0, 255, 255]
    cv2.putText(vis, title, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    return vis


# ── Fit one frame (T2.1: 2D joint optimization) ──────────────────────────────
def fit_frame_t21(img_bgr, est, faces, kp_rtm):
    H, W = img_bgr.shape[:2]

    # Build bbox
    ax = [v['x'] for v in kp_rtm.values() if v['score'] > 0.3]
    ay = [v['y'] for v in kp_rtm.values() if v['score'] > 0.3]
    if len(ax) < 4:
        return None
    pad_x = (max(ax)-min(ax)) * 0.15
    pad_y = (max(ay)-min(ay)) * 0.15
    bbox  = np.array([[max(0, min(ax)-pad_x), max(0, min(ay)-pad_y),
                       min(W, max(ax)+pad_x), min(H, max(ay)+pad_y)]], dtype=np.float32)

    outs  = est.process_one_image(img_bgr, bboxes=bbox, use_mask=False, inference_type="body")
    if not outs:
        return None
    o     = outs[0]
    verts = o["pred_vertices"].astype(np.float32)
    cam_t = o["pred_cam_t"].astype(np.float32)
    kp2d  = o["pred_keypoints_2d"].astype(np.float32)
    focal = float(o["focal_length"])

    # Human mask
    mask_h = get_human_mask(img_bgr)

    # T1 render
    rgba_t1    = render_rgba(verts, cam_t, focal, faces, H, W)
    sil_t1     = get_sil(rgba_t1)
    verts2d_t1 = project_verts(verts, cam_t, focal, H, W)

    # y-centers
    nose_y  = int(kp2d[I_NOSE][1])
    neck_y  = int(kp2d[I_NECK][1])
    sho_y   = int((kp2d[I_LSHO][1] + kp2d[I_RSHO][1]) / 2)
    hip_y   = int((kp2d[I_LHIP][1] + kp2d[I_RHIP][1]) / 2)
    knee_y  = int((kp2d[I_LKNE][1] + kp2d[I_RKNE][1]) / 2)
    ank_y   = int((kp2d[I_LANK][1] + kp2d[I_RANK][1]) / 2)
    head_y  = nose_y - 55

    # RTM body cx (sentinel A3 reference)
    rtm_body_cx = float(np.mean([kp_rtm['left_shoulder']['x'], kp_rtm['right_shoulder']['x'],
                                  kp_rtm['left_hip']['x'],       kp_rtm['right_hip']['x']]))

    # Band boundaries
    B_UPP_lo = head_y - 30;  B_UPP_hi = hip_y - 15
    B_HIP_lo = hip_y - 15;   B_HIP_hi = hip_y + 110
    B_LOW_lo = hip_y + 110;  B_LOW_hi = ank_y + 120
    UP_IoU_lo = head_y - 30; UP_IoU_hi = hip_y + 60

    # cx from T1 sil
    m_sho = measure_width(sil_t1, sho_y)
    cx_upp = m_sho[2] if m_sho else float(W/2)
    m_hip  = measure_width(sil_t1, hip_y)
    cx_hip = m_hip[2] if m_hip else float(W/2)
    m_kne  = measure_width(sil_t1, knee_y)
    cx_low = m_kne[2] if m_kne else float(W/2)

    # Mesh body cx (for sentinel A3)
    mesh_body_cx = float((kp2d[I_LSHO][0] + kp2d[I_RSHO][0] +
                          kp2d[I_LHIP][0] + kp2d[I_RHIP][0]) / 4.0)
    cx_offset = abs(mesh_body_cx - rtm_body_cx)

    # ── B: 2D joint optimization for B_UPP and B_HIP ──
    # Optimize [sx, dy] jointly via Nelder-Mead proxy
    # dy: vertical shift of the silhouette in proxy (in pixels)
    SX_LO, SX_HI = 0.80, 2.00
    DY_LIM = 50.0

    def neg_proxy_2d_upp(params):
        sx, dy = params
        if sx < SX_LO or sx > SX_HI or abs(dy) > DY_LIM:
            return 0.0   # out of bounds penalty handled by clamping
        return -proxy_iou_2d(sx, dy, sil_t1, cx_upp, mask_h, B_UPP_lo, B_UPP_hi, H, W)

    def neg_proxy_2d_hip(params):
        sx, dy = params
        if sx < SX_LO or sx > SX_HI or abs(dy) > DY_LIM:
            return 0.0
        return -proxy_iou_2d(sx, dy, sil_t1, cx_hip, mask_h, B_HIP_lo, B_HIP_hi, H, W)

    # Start from T2 1D result as warm start
    # First get 1D result (bounded scalar, same as T2)
    res_upp_1d = minimize_scalar(
        lambda sx: -proxy_iou_2d(sx, 0, sil_t1, cx_upp, mask_h, B_UPP_lo, B_UPP_hi, H, W),
        bounds=(SX_LO, SX_HI), method='bounded', options={'xatol': 1e-4, 'maxiter': 50}
    )
    sx_upp_1d = float(res_upp_1d.x)

    res_hip_1d = minimize_scalar(
        lambda sx: -proxy_iou_2d(sx, 0, sil_t1, cx_hip, mask_h, B_HIP_lo, B_HIP_hi, H, W),
        bounds=(SX_LO, SX_HI), method='bounded', options={'xatol': 1e-4, 'maxiter': 50}
    )
    sx_hip_1d = float(res_hip_1d.x)

    # 2D Nelder-Mead starting from 1D result
    res_upp_2d = minimize(neg_proxy_2d_upp, x0=[sx_upp_1d, 0.0], method='Nelder-Mead',
                          options={'xatol': 0.005, 'fatol': 0.001, 'maxiter': 100, 'adaptive': True})
    sx_upp, dy_upp = float(res_upp_2d.x[0]), float(res_upp_2d.x[1])
    sx_upp = float(np.clip(sx_upp, SX_LO, SX_HI))
    dy_upp = float(np.clip(dy_upp, -DY_LIM, DY_LIM))

    res_hip_2d = minimize(neg_proxy_2d_hip, x0=[sx_hip_1d, 0.0], method='Nelder-Mead',
                          options={'xatol': 0.005, 'fatol': 0.001, 'maxiter': 100, 'adaptive': True})
    sx_hip, dy_hip = float(res_hip_2d.x[0]), float(res_hip_2d.x[1])
    sx_hip = float(np.clip(sx_hip, SX_LO, SX_HI))
    dy_hip = float(np.clip(dy_hip, -DY_LIM, DY_LIM))

    # B_LOW: containment (达标即止, unchanged)
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

    # ── Apply scale + y-shift ──
    v_opt = verts.copy()
    depth_upp = float(v_opt[verts2d_t1[:,1] < B_HIP_lo, 2].mean()) + cam_t[2]
    depth_hip = float(v_opt[(verts2d_t1[:,1]>=B_HIP_lo)&(verts2d_t1[:,1]<B_LOW_lo), 2].mean()) + cam_t[2]
    depth_low = float(v_opt[verts2d_t1[:,1] >= B_LOW_lo, 2].mean()) + cam_t[2]

    wcx_m_upp = world_x_from_img(cx_upp,        depth_upp, cam_t, focal, W)
    wcx_m_hip = world_x_from_img(cx_hip,        depth_hip, cam_t, focal, W)
    wcx_m_low = world_x_from_img(cx_low_actual, depth_low, cam_t, focal, W)

    bm_upp = verts2d_t1[:,1] < B_HIP_lo
    bm_hip = (verts2d_t1[:,1] >= B_HIP_lo) & (verts2d_t1[:,1] < B_LOW_lo)
    bm_low = verts2d_t1[:,1] >= B_LOW_lo

    # x-scale
    v_opt[bm_upp, 0] = wcx_m_upp + (v_opt[bm_upp,0] - wcx_m_upp) * sx_upp
    v_opt[bm_hip, 0] = wcx_m_hip + (v_opt[bm_hip,0] - wcx_m_hip) * sx_hip
    v_opt[bm_low, 0] = wcx_m_low + (v_opt[bm_low,0] - wcx_m_low) * sx_low

    # y-shift (world coords): dy_img → dy_world
    if abs(dy_upp) >= 1.0:
        dy_w_upp = world_y_from_img(H/2 + dy_upp, depth_upp, cam_t, focal, H) - \
                   world_y_from_img(H/2,           depth_upp, cam_t, focal, H)
        v_opt[bm_upp, 1] += dy_w_upp

    if abs(dy_hip) >= 1.0:
        dy_w_hip = world_y_from_img(H/2 + dy_hip, depth_hip, cam_t, focal, H) - \
                   world_y_from_img(H/2,           depth_hip, cam_t, focal, H)
        v_opt[bm_hip, 1] += dy_w_hip

    # cx correction (mid-render)
    rgba_mid = render_rgba(v_opt, cam_t, focal, faces, H, W)
    sil_mid  = get_sil(rgba_mid)

    all_up_ys_full = list(range(neck_y, hip_y-15, 15))
    sho_ys   = list(range(neck_y, sho_y+100, 12))
    hip_ys   = [hip_y-20, hip_y, hip_y+20, hip_y+40]

    v_opt, _ = translate_cx(v_opt, bm_upp, cam_t, focal, W, H,
                            mask_h, sil_mid, all_up_ys_full, "B_UPP")
    v_opt, _ = translate_cx(v_opt, bm_hip, cam_t, focal, W, H,
                            mask_h, sil_mid, hip_ys, "B_HIP")
    v_opt, _ = translate_cx(v_opt, bm_low, cam_t, focal, W, H,
                            mask_h, sil_mid, lower_ys, "B_LOW")

    # Final render
    rgba_final = render_rgba(v_opt, cam_t, focal, faces, H, W)
    sil_final  = get_sil(rgba_final)

    # Residual shoulder cx
    h_cxs2, m_cxs2 = [], []
    for yc in sho_ys[:6]:
        hm = measure_width(mask_h, yc); mm = measure_width(sil_final, yc)
        if hm and mm:
            h_cxs2.append(hm[2]); m_cxs2.append(mm[2])
    if h_cxs2:
        res_dx = float(np.median(h_cxs2)) - float(np.median(m_cxs2))
        if abs(res_dx) > 1.5:
            depth_b = float(v_opt[bm_upp, 2].mean()) + cam_t[2]
            v_opt[bm_upp, 0] += res_dx * depth_b / focal
            rgba_final = render_rgba(v_opt, cam_t, focal, faces, H, W)
            sil_final  = get_sil(rgba_final)

    # IoU
    iou_upp, inter, union = compute_iou(mask_h, sil_final, UP_IoU_lo, UP_IoU_hi, H)

    return {
        "iou_upper": round(float(iou_upp), 4),
        "pass": bool(iou_upp >= PASS_THRESHOLD),
        "rgba_final": rgba_final,
        "sil_final": sil_final,
        "mask_h": mask_h,
        "sx_upp": round(sx_upp, 4),
        "sy_upp_dy": round(dy_upp, 2),
        "sx_hip": round(sx_hip, 4),
        "sy_hip_dy": round(dy_hip, 2),
        "sx_low": round(sx_low, 4),
        "UP_IoU_lo": UP_IoU_lo,
        "UP_IoU_hi": UP_IoU_hi,
        "head_y": head_y,
        "hip_y": hip_y,
        "img_bgr": img_bgr,
        "kp2d": kp2d,
        "cam_t": cam_t.tolist(),
        "focal": focal,
        "mesh_cx": round(mesh_body_cx, 1),
        "rtm_cx": round(rtm_body_cx, 1),
        "cx_offset": round(cx_offset, 1),
    }


# ── Interpolate overlay for invalid frames ───────────────────────────────────
def build_interp_overlay(fr_idx, img_bgr, valid_overlays):
    """Blend nearest prev/next valid overlay onto img_bgr."""
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

    # Load T2 log + kp cache
    with open(T2_LOG) as f:
        t2_log = json.load(f)
    with open(KP_CACHE) as f:
        kpd = json.load(f)
    kp_frames = kpd['frames']
    NF = len(kp_frames)
    t2_ious = np.array([r['iou_upper'] for r in t2_log['per_frame']])

    # ── A. Sentinel pre-pass ──────────────────────────────────────────────
    print("\n[INFO] === Sentinel pre-pass (A1+A2) ===")
    sentinels = compute_sentinels(t2_log['per_frame'], kp_frames, t2_ious)
    invalid_frames = [i for i, s in sentinels.items() if s['invalid']]
    print(f"  Frames flagged invalid: {invalid_frames}")
    for fi in invalid_frames:
        s = sentinels[fi]
        print(f"    fr{fi:03d}  IoU_T2={s['iou_t2']:.4f}  reasons={s['reasons']}")

    # Load model
    print("\n[INFO] Loading model...")
    model, cfg = load_sam_3d_body(CKPT, device='cuda', mhr_path=MHR_P)
    est = SAM3DBodyEstimator(model, cfg, human_detector=None,
                             human_segmentor=None, fov_estimator=None)
    faces = est.faces

    # Open video
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO_PATH}")

    results = []
    frame_results = {}
    valid_overlays = {}   # fr_idx -> overlay img (for interpolation)
    KEYFRAMES = {FR_ADDRESS: "address", FR_TOP: "top", FR_IMPACT: "impact"}

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
            # Skip fitting, record as invalid
            print(f"  [fr{fr_idx:03d}/{NF-1}] INVALID ({', '.join(sentinel_reasons)}) — skip fit, interpolate overlay")
            results.append({
                "frame": fr_idx,
                "iou_upper": None,
                "pass": False,
                "invalid": True,
                "reasons": sentinel_reasons,
                "iou_t2": sentinels[fr_idx]['iou_t2'],
            })
            continue

        t_fr = time.time()
        kp_rtm = kp_frames[fr_idx]['persons'][0]['keypoints']
        print(f"  [fr{fr_idx:03d}/{NF-1}] fitting...", end='', flush=True)

        try:
            r = fit_frame_t21(img_bgr, est, faces, kp_rtm)
        except Exception as e:
            print(f" ERROR: {e}")
            results.append({"frame": fr_idx, "iou_upper": None, "pass": False, "error": str(e)})
            continue

        if r is None:
            print(f" SKIP (no output)")
            results.append({"frame": fr_idx, "iou_upper": None, "pass": False})
            continue

        # Sentinel A3: mesh cx offset check
        a3_flag = r["cx_offset"] > SEN_A3_CX_OFF
        if a3_flag:
            print(f"\n  [A3 WARN] fr{fr_idx:03d}: mesh_cx={r['mesh_cx']:.1f} rtm_cx={r['rtm_cx']:.1f} offset={r['cx_offset']:.1f}px > {SEN_A3_CX_OFF}")
            sentinels[fr_idx]['reasons'].append(f"A3:cx_offset={r['cx_offset']:.1f}")
            # Re-flag as invalid (A3 detected post-fit)
            results.append({
                "frame": fr_idx, "iou_upper": None, "pass": False,
                "invalid": True, "reasons": sentinels[fr_idx]['reasons'],
                "iou_t2": sentinels[fr_idx]['iou_t2'],
            })
            continue

        dt = time.time() - t_fr
        iou = r["iou_upper"]
        t2_iou = t2_ious[fr_idx]
        delta = iou - t2_iou
        flag = "✓" if r["pass"] else "✗"
        print(f" IoU={iou:.4f} {flag}  (T2:{t2_iou:.4f} Δ{delta:+.4f})  dy_upp={r['sy_upp_dy']:+.1f}px  ({dt:.1f}s)")

        row = {
            "frame": fr_idx,
            "iou_upper": iou,
            "iou_t2": float(t2_iou),
            "delta_vs_t2": round(float(iou - t2_iou), 4),
            "pass": r["pass"],
            "invalid": False,
            "sx_upp": r["sx_upp"],
            "dy_upp": r["sy_upp_dy"],
            "sx_hip": r["sx_hip"],
            "dy_hip": r["sy_hip_dy"],
            "sx_low": r["sx_low"],
            "cx_offset": r["cx_offset"],
        }
        results.append(row)

        # Store overlay for interpolation
        ov = composite_red(r["rgba_final"], img_bgr, alpha=0.55)
        valid_overlays[fr_idx] = ov.copy()
        frame_results[fr_idx] = r

        # Save keyframes
        if fr_idx in KEYFRAMES:
            phase = KEYFRAMES[fr_idx]
            kp2d = r["kp2d"]
            lh = (int(kp2d[I_LHIP][0]), int(kp2d[I_LHIP][1]))
            rh = (int(kp2d[I_RHIP][0]), int(kp2d[I_RHIP][1]))
            y_pel = (lh[1]+rh[1])//2
            H_img, W_img = img_bgr.shape[:2]
            cv2.line(ov, (0, y_pel), (W_img-1, y_pel), (255,255,255), 2)
            cv2.circle(ov, lh, 7, (255,255,255), -1); cv2.circle(ov, lh, 3, (0,0,200), -1)
            cv2.circle(ov, rh, 7, (255,255,255), -1); cv2.circle(ov, rh, 3, (0,0,200), -1)
            label = f"{phase}_fr{fr_idx:03d}_IoU{iou:.3f}"
            cv2.putText(ov, label, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            kf_path = KF_DIR / f"{phase}_fr{fr_idx:03d}.jpg"
            cv2.imwrite(str(kf_path), ov)
            print(f"  [KEY] saved {kf_path.name}")

    cap.release()

    # ── Generate interpolated overlays for invalid frames ─────────────────
    print("\n[INFO] Building interpolated overlays for invalid frames...")
    cap2 = cv2.VideoCapture(str(VIDEO_PATH))
    for fr_idx in invalid_frames:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, fr_idx)
        ret, img_bgr = cap2.read()
        if not ret:
            continue
        interp_ov = build_interp_overlay(fr_idx, img_bgr, valid_overlays)
        cv2.putText(interp_ov, f"INVALID fr{fr_idx:03d} [INTERPOLATED]", (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)
        kf_path = OUTPUT_DIR / f"invalid_fr{fr_idx:03d}_interp.jpg"
        cv2.imwrite(str(kf_path), interp_ov)
        print(f"  [INTERP] fr{fr_idx:03d} → {kf_path.name}")
    cap2.release()

    # ── Compute valid-only distribution ───────────────────────────────────
    valid_rows = [r for r in results if not r.get('invalid', False) and r.get('iou_upper') is not None]
    ious_valid = np.array([r["iou_upper"] for r in valid_rows])
    n_valid = len(ious_valid)
    n_invalid = NF - n_valid

    mean_iou = float(np.mean(ious_valid))
    min_iou  = float(np.min(ious_valid))
    p5_iou   = float(np.percentile(ious_valid, 5))
    min_fr   = valid_rows[np.argmin(ious_valid)]["frame"]
    fail_valid = [(r["frame"], r["iou_upper"]) for r in valid_rows if not r["pass"]]

    print(f"\n{'='*58}")
    print(f"  T2.1 IoU 分布 (有效帧={n_valid}, 无效帧={n_invalid})")
    print(f"  mean = {mean_iou:.4f}")
    print(f"  min  = {min_iou:.4f}  (fr{min_fr:03d})")
    print(f"  P5   = {p5_iou:.4f}")
    print(f"  低于 {PASS_THRESHOLD}: {len(fail_valid)} 有效帧")
    print(f"\n  对比 T2: mean={t2_log['distribution']['mean']:.4f}  min={t2_log['distribution']['min']:.4f}(fr{t2_log['distribution']['min_frame']:03d})  P5={t2_log['distribution']['p5']:.4f}  (含无效帧)")
    print(f"{'='*58}")

    # ── Worst 3 valid frames: silhouette compare ───────────────────────────
    valid_sorted = sorted(
        [(r["frame"], r["iou_upper"]) for r in valid_rows if r["frame"] in frame_results],
        key=lambda x: x[1]
    )
    worst3 = valid_sorted[:3]
    print(f"\n[INFO] Worst 3 valid frames: {[f for f, _ in worst3]}")
    for rank, (fno, fiou) in enumerate(worst3, 1):
        r = frame_results[fno]
        H_img, W_img = r["img_bgr"].shape[:2]
        vis = make_iou_vis(r["mask_h"], r["sil_final"], H_img, W_img,
                           f"fr{fno:03d} valid IoU={fiou:.3f}")
        cv2.putText(vis, "green=human  red=mesh  yellow=overlap",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        t2_v = sentinels[fno]['iou_t2'] if sentinels[fno]['iou_t2'] else 0.0
        cv2.putText(vis, f"T2:{t2_v:.3f}  T2.1:{fiou:.3f}  Δ{fiou-t2_v:+.3f}",
                    (10, H_img-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
        p_vis = OUTPUT_DIR / f"worst{rank}_valid_fr{fno:03d}.jpg"
        cv2.imwrite(str(p_vis), vis)
        print(f"  [WORST{rank}] fr{fno:03d} IoU={fiou:.4f} → {p_vis.name}")

    # ── IoU distribution curve ─────────────────────────────────────────────
    H_c, W_c = 440, max(800, NF * 6)
    curve = np.zeros((H_c, W_c, 3), dtype=np.uint8)
    curve[:] = (30, 30, 30)

    for iou_grid in [0.80, 0.85, 0.87, 0.90, 0.92, 0.95, 1.0]:
        yg = int((1.0 - iou_grid) / 0.25 * (H_c - 80)) + 40
        color = (0, 200, 100) if abs(iou_grid - PASS_THRESHOLD) < 0.005 \
                else ((100, 200, 255) if abs(iou_grid - 0.87) < 0.005 else (60, 60, 60))
        cv2.line(curve, (0, yg), (W_c-1, yg), color, 1 if iou_grid not in [0.87, 0.92] else 2)
        cv2.putText(curve, f"{iou_grid:.2f}", (4, yg-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

    # T2 dots (gray, background)
    for r in t2_log['per_frame']:
        if r['iou_upper'] is None: continue
        fno = r['frame']; iou_v = r['iou_upper']
        x = int(fno / max(NF-1,1) * (W_c-20)) + 10
        y = int((1.0 - min(1.0, max(0.75, iou_v))) / 0.25 * (H_c-80)) + 40
        cv2.circle(curve, (x, y), 2, (80, 80, 80), -1)

    # T2.1 valid dots
    for r in results:
        fno = r['frame']
        if r.get('invalid', False):
            # Red X for invalid
            x = int(fno / max(NF-1,1) * (W_c-20)) + 10
            iou_v = r.get('iou_t2') or 0.85
            y = int((1.0 - min(1.0, max(0.75, iou_v))) / 0.25 * (H_c-80)) + 40
            cv2.line(curve, (x-5, y-5), (x+5, y+5), (0, 0, 255), 2)
            cv2.line(curve, (x-5, y+5), (x+5, y-5), (0, 0, 255), 2)
        elif r.get('iou_upper') is not None:
            iou_v = r['iou_upper']
            x = int(fno / max(NF-1,1) * (W_c-20)) + 10
            y = int((1.0 - min(1.0, max(0.75, iou_v))) / 0.25 * (H_c-80)) + 40
            color = (0, 220, 255) if r['pass'] else (0, 140, 255)
            cv2.circle(curve, (x, y), 3, color, -1)
            if fno in {FR_ADDRESS, FR_TOP, FR_IMPACT}:
                cv2.circle(curve, (x, y), 7, (255, 255, 0), 2)

    # Legend
    legend_x = W_c - 220
    cv2.circle(curve, (legend_x, 15), 4, (0, 220, 255), -1);  cv2.putText(curve, "T2.1 pass", (legend_x+10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
    cv2.circle(curve, (legend_x, 30), 4, (0, 140, 255), -1);  cv2.putText(curve, "T2.1 <0.92", (legend_x+10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
    cv2.circle(curve, (legend_x, 45), 3, (80, 80, 80), -1);   cv2.putText(curve, "T2 (gray)", (legend_x+10, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
    cv2.line(curve, (legend_x-4, 56), (legend_x+4, 64), (0,0,255), 2); cv2.line(curve, (legend_x-4, 64), (legend_x+4, 56), (0,0,255), 2)
    cv2.putText(curve, "invalid(X)", (legend_x+10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

    summary = f"T2.1  valid={n_valid}  invalid={n_invalid}  mean={mean_iou:.3f}  min={min_iou:.3f}(fr{min_fr:03d})  P5={p5_iou:.3f}"
    cv2.putText(curve, summary, (10, H_c-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    p_curve = OUTPUT_DIR / "iou_distribution_t21.jpg"
    cv2.imwrite(str(p_curve), curve)
    print(f"\n[OUT] {p_curve}")

    # ── JSON log ──────────────────────────────────────────────────────────
    vram = torch.cuda.max_memory_allocated() / 1e6
    tt   = time.time() - t_start
    log_out = {
        "version": "T2.1", "clip": "fo-ok-1", "NF": NF,
        "pass_threshold": PASS_THRESHOLD,
        "sentinel_config": {
            "A1_delta": SEN_A1_DELTA,
            "A2_conf":  SEN_A2_CONF,
            "A3_cx_offset": SEN_A3_CX_OFF,
        },
        "invalid_frames": [
            {"frame": fi, "reasons": sentinels[fi]['reasons'], "iou_t2": sentinels[fi]['iou_t2']}
            for fi in invalid_frames
        ],
        "distribution_valid_only": {
            "mean": round(mean_iou, 4),
            "min":  round(min_iou, 4), "min_frame": min_fr,
            "p5":   round(p5_iou, 4),
            "valid_count": n_valid,
            "invalid_count": n_invalid,
        },
        "distribution_t2_all": t2_log['distribution'],
        "fail_valid_frames": [{"frame": f, "iou_upper": round(v, 4)} for f, v in fail_valid],
        "worst3_valid": [{"frame": f, "iou_upper": round(v, 4)} for f, v in worst3],
        "per_frame": results,
        "peak_vram_mb": round(vram, 0),
        "total_s": round(tt, 1),
        "topology": "MHR_native", "license": "SAM_License_PRODUCT_CANDIDATE_CUSTOM_LICENSE",
    }
    lp = OUTPUT_DIR / "run_log_t21.json"
    with open(lp, 'w') as f:
        json.dump(log_out, f, indent=2, default=str)
    print(f"[OUT] {lp}")

    # ── Text report ───────────────────────────────────────────────────────
    report_lines = [
        "GHOST-003 T2.1 停关卡报告",
        f"Clip: fo-ok-1  NF={NF}",
        "",
        "=== A. 哨兵判据 ===",
        f"A1: 单帧 IoU 比相邻帧均值低 >= {SEN_A1_DELTA}",
        f"A2: 核心关键点 (lsho/rsho/lhip/rhip) 平均置信 < {SEN_A2_CONF}",
        f"A3: mesh_cx vs RTM_cx 偏移 > {SEN_A3_CX_OFF}px",
        "",
        f"无效帧数: {n_invalid}",
    ]
    for fi in invalid_frames:
        s = sentinels[fi]
        report_lines.append(f"  fr{fi:03d}  T2_IoU={s['iou_t2']:.4f}  reasons={s['reasons']}")

    report_lines += [
        "",
        "=== B. 有效帧 IoU 分布 (剔除无效帧后) ===",
        f"有效帧数: {n_valid}  (无效帧已剔除: {n_invalid})",
        f"mean IoU : {mean_iou:.4f}",
        f"min  IoU : {min_iou:.4f}  (fr{min_fr:03d}) ← 最差有效帧",
        f"P5   IoU : {p5_iou:.4f}",
        f"低于 {PASS_THRESHOLD} 的有效帧: {len(fail_valid)}",
        "",
        "=== 剔除前后对比 ===",
        f"T2 (含无效帧, n=112): mean={t2_log['distribution']['mean']:.4f}  min={t2_log['distribution']['min']:.4f}(fr{t2_log['distribution']['min_frame']:03d})  P5={t2_log['distribution']['p5']:.4f}",
        f"T2.1 (有效帧, n={n_valid}): mean={mean_iou:.4f}  min={min_iou:.4f}(fr{min_fr:03d})  P5={p5_iou:.4f}",
        f"改善: mean Δ{mean_iou - t2_log['distribution']['mean']:+.4f}  P5 Δ{p5_iou - t2_log['distribution']['p5']:+.4f}",
        "",
        "最差3有效帧:",
    ]
    for rank, (fno, fiou) in enumerate(worst3, 1):
        t2_v = t2_ious[fno]
        report_lines.append(f"  #{rank} fr{fno:03d}  IoU={fiou:.4f}  (T2:{t2_v:.4f}  Δ{fiou-t2_v:+.4f})")

    report_lines += [
        "",
        "关键帧:",
        f"  address fr{FR_ADDRESS:03d}: T2={t2_ious[FR_ADDRESS]:.4f}  T2.1={next((r['iou_upper'] for r in results if r['frame']==FR_ADDRESS), 'N/A')}",
        f"  top     fr{FR_TOP:03d}: T2={t2_ious[FR_TOP]:.4f}  T2.1={next((r['iou_upper'] for r in results if r['frame']==FR_TOP), 'N/A')}",
        f"  impact  fr{FR_IMPACT:03d}: T2={t2_ious[FR_IMPACT]:.4f}  T2.1={next((r['iou_upper'] for r in results if r['frame']==FR_IMPACT), 'N/A')}",
        "",
        f"peak VRAM: {vram:.0f}MB  total: {tt:.0f}s",
    ]
    rp = OUTPUT_DIR / "REPORT_T21.txt"
    rp.write_text("\n".join(report_lines), encoding='utf-8')
    print(f"[OUT] {rp}")

    print(f"\n{'='*58}")
    print(f"  T2.1 DONE  valid={n_valid}  invalid={n_invalid}")
    print(f"  mean={mean_iou:.4f}  min={min_iou:.4f}(fr{min_fr:03d})  P5={p5_iou:.4f}")
    print(f"  peak VRAM: {vram:.0f}MB  total: {tt:.1f}s")
    print(f"{'='*58}")


if __name__ == "__main__":
    main()
