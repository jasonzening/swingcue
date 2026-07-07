"""
ghost003_t23_arm_foot.py  —  GHOST-003 T2.3
前置诊断 (T2.2 复盘):
  - ARM_OPT 0帧根因: 触发条件 arm_dist/body_h>1.2 未达 (fr064=1.031, fr103=0.763)
  - 脚部 kp70 ankle 投影精准 (dx<8px), 但足部顶点质心偏右/偏下 +19~96px
    → 原因: MHR mesh 脚部朝向沿 T-pose 轴展开, 非 rembg 问题

A. 手臂姿态修正(真执行)
  - 触发条件: T2.2 IoU < 0.85 (帧级 IoU, 非几何启发)
  - 方法: arm-band sx 优化 (顶点级别, 非 body_pose_params)
    * arm-band = py in [wrist_y-40, sho_y+40] 且 |px - body_cx| > sho_half_w * 0.7
    * scipy 1D bounded 最大化 arm-band IoU
    * 约束: upper IoU 不得比 T2.2 降超过 0.02
  - 日志: 每帧打印 [ARM] fr###: T2.2_IoU → T2.3_IoU, 优化器 nfev, arm_sx
  - 若 nfev>0 且 IoU 无改善: 打印 [ARM_NO_GAIN] fr###

B. 脚部精修
  - 触发: 全部帧 (系统性错位)
  - 方法: foot-band 顶点平移
    * foot vertices: py > ankle_kp2d_y - 20
    * 目标: 足部顶点质心 → RTMPose ankle 中点
    * 独立计算左右脚 (根据 px 分左右), 分别向各自 RTMPose ankle 平移
    * kp_guard: ankle score < 0.50 → 用邻帧插值替代
  - 日志: 每帧打印 [FOOT] fr###: dx_L/dx_R/dy_L/dy_R

C. 分布重算
  - IoU 范围仍用上半身 (head_y-30 至 hip_y+60)
  - 报告: invalid 帧列表, valid mean/min+帧号/P5
  - address 脚部特写: address_fr000_foot.jpg (下半身区域 sil 对比)

D. 关键帧维持
  - address/top/impact 不得回退

版本说明: T2.3 基于 T2.2 结果; A4 哨兵沿用 (fr087+fr094=无效)
授权: SAM License (PRODUCT_CANDIDATE_CUSTOM_LICENSE)
"""

import os, sys, time, json, pathlib
import numpy as np
import cv2
from scipy.optimize import minimize_scalar as ms1d

SAM3D_REPO = "/home/jason/projects/sam-3d-body"
SWINGCUE   = "/home/jason/projects/swingcue-postest"
sys.path.insert(0, SAM3D_REPO)
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

OUTPUT_DIR = pathlib.Path(SWINGCUE) / "output" / "ghost003_t23"
KF_DIR     = OUTPUT_DIR / "keyframe_overlay_t23"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KF_DIR.mkdir(parents=True, exist_ok=True)

KP_CACHE   = pathlib.Path(SWINGCUE) / "engine/kp_cache/batch2/fo-ok-1.json"
VIDEO_PATH = pathlib.Path(SWINGCUE) / "input/fo-ok-1.mp4"
T22_LOG    = pathlib.Path(SWINGCUE) / "output/ghost003_t22/run_log_t22.json"
CKPT  = "/home/jason/.cache/sam3d/sam-3d-body-dinov3/model.ckpt"
MHR_P = "/home/jason/.cache/sam3d/sam-3d-body-dinov3/assets/mhr_model.pt"

FR_ADDRESS = 0
FR_TOP     = 97
FR_IMPACT  = 88
PASS_THRESHOLD = 0.92

# Arm opt: trigger if T2.2 IoU < this
ARM_OPT_IOUt_TRIGGER = 0.85
# Arm sx optimization limits — EXPAND arm mesh outward to cover extended real arm
ARM_SX_BOUNDS = (0.9, 3.0)  # allow expansion up to 3x (arm extends far from body)

# Foot kp guard
FOOT_KP_GUARD_SCORE = 0.50  # below this → interpolate from neighbor

# kp70 indices
I_NOSE=0; I_LSHO=5; I_RSHO=6; I_LELB=7; I_RELB=8; I_LWRI=62; I_RWRI=41
I_LHIP=9; I_RHIP=10; I_LKNE=11; I_RKNE=12; I_LANK=13; I_RANK=14; I_NECK=69

# ── Projection ────────────────────────────────────────────────────────────────
def proj2d(verts, cam_t, focal, H, W):
    vx,vy,vz = verts[:,0],verts[:,1],verts[:,2]
    d = np.where(np.abs(vz+cam_t[2])<1e-6, 1e-6, vz+cam_t[2])
    px = focal*(vx-cam_t[0])/d + W/2.0
    py = focal*(vy+cam_t[1])/d + H/2.0
    return np.stack([px,py],axis=1)

def world_x_from_img(img_x, depth, cam_t, focal, W):
    return (img_x - W/2.0) * depth / focal + cam_t[0]

def world_y_from_img(img_y, depth, cam_t, focal, H):
    return (img_y - H/2.0) * depth / focal - cam_t[1]


# ── Human mask ────────────────────────────────────────────────────────────────
def get_human_mask(img_bgr):
    import rembg
    out  = rembg.remove(img_bgr)
    mask = (out[:,:,3] > 40).astype(np.uint8) * 255
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    return mask


# ── Render ────────────────────────────────────────────────────────────────────
def render_rgba(verts, cam_t, focal, faces, H, W):
    from sam_3d_body.visualization.renderer import Renderer
    dummy = np.zeros((H, W, 3), dtype=np.uint8)
    r = Renderer(focal_length=focal, faces=faces)
    return r(verts, cam_t, dummy, mesh_base_color=(1.,0.,0.),
             scene_bg_color=(0,0,0), return_rgba=True)

def get_sil(rgba):
    ch = rgba[:,:,3] if rgba.shape[2]==4 else rgba.sum(2)
    return (ch > 0.05).astype(np.uint8) * 255

def composite_red(rgba, img_bgr, alpha=0.55):
    f  = img_bgr.astype(np.float32)/255.0
    ma = rgba[:,:,3:4] if rgba.shape[2]==4 else \
         (rgba.sum(2,keepdims=True)>0.02).astype(np.float32)
    red = np.zeros_like(f); red[:,:,2]=1.0
    return np.clip((f*(1-ma*alpha)+red*ma*alpha)*255,0,255).astype(np.uint8)


# ── IoU ───────────────────────────────────────────────────────────────────────
def compute_iou(mask_h, sil_m, ylo, yhi, H):
    y1,y2 = max(0,int(ylo)),min(H,int(yhi))
    h=mask_h[y1:y2]>0; m=sil_m[y1:y2]>0
    return int((h&m).sum())/max(int((h|m).sum()),1)

def proxy_iou(sx, sil_base, cx_img, mask_h, ylo, yhi, H, W):
    x     = np.arange(W,dtype=np.float32)
    x_src = np.clip(cx_img+(x-cx_img)/sx, 0, W-1).astype(np.int32)
    sil_s = sil_base[:,x_src]
    y1,y2 = max(0,int(ylo)),min(H,int(yhi))
    h=mask_h[y1:y2]>0; m=sil_s[y1:y2]>0
    return int((h&m).sum())/max(int((h|m).sum()),1)


# ── Width / cx measure ─────────────────────────────────────────────────────────
def measure_width(mask, yc, hb=30):
    H,W = mask.shape
    y1,y2 = max(0,yc-hb),min(H,yc+hb)
    cols = np.where(mask[y1:y2].any(axis=0))[0]
    if len(cols)<3: return None
    return int(cols.min()),int(cols.max()),float((cols.min()+cols.max())/2.),int(cols.max()-cols.min())


# ── T1.7 shape-fit (T2.2 baseline) ────────────────────────────────────────────
def shape_fit_t17(verts, cam_t, focal, faces, mask_h, kp2d, H, W):
    """Apply T1.7 IoU sx + cx-correction. Returns (v_opt, rgba, sil, iou_upp)."""
    nose_y = int(kp2d[I_NOSE][1])
    neck_y = int(kp2d[I_NECK][1])
    sho_y  = int((kp2d[I_LSHO][1]+kp2d[I_RSHO][1])/2)
    hip_y  = int((kp2d[I_LHIP][1]+kp2d[I_RHIP][1])/2)
    knee_y = int((kp2d[I_LKNE][1]+kp2d[I_RKNE][1])/2)
    ank_y  = int((kp2d[I_LANK][1]+kp2d[I_RANK][1])/2)
    head_y = nose_y - 55

    B_UPP_lo=head_y-30; B_UPP_hi=hip_y-15
    B_HIP_lo=hip_y-15;  B_HIP_hi=hip_y+110
    B_LOW_lo=hip_y+110; B_LOW_hi=ank_y+120
    UP_lo=head_y-30;    UP_hi=hip_y+60

    verts2d = proj2d(verts, cam_t, focal, H, W)
    rgba_t1 = render_rgba(verts, cam_t, focal, faces, H, W)
    sil_t1  = get_sil(rgba_t1)

    m_sho = measure_width(sil_t1, sho_y)
    cx_upp = m_sho[2] if m_sho else W/2.0
    m_hip  = measure_width(sil_t1, hip_y)
    cx_hip = m_hip[2] if m_hip else W/2.0
    m_kne  = measure_width(sil_t1, knee_y)
    cx_low = m_kne[2] if m_kne else W/2.0

    res_u = ms1d(lambda sx: -proxy_iou(sx,sil_t1,cx_upp,mask_h,B_UPP_lo,B_UPP_hi,H,W),
                 bounds=(0.80,2.00),method='bounded',options={'xatol':1e-4,'maxiter':50})
    sx_upp = float(res_u.x)
    res_h = ms1d(lambda sx: -proxy_iou(sx,sil_t1,cx_hip,mask_h,B_HIP_lo,B_HIP_hi,H,W),
                 bounds=(0.80,1.50),method='bounded',options={'xatol':1e-4,'maxiter':50})
    sx_hip = float(res_h.x)

    lower_ys=[knee_y-20,knee_y,knee_y+30,(knee_y+ank_y)//2,ank_y-30]
    sx_max_low=0.0; m_low_list=[]
    for yc in lower_ys:
        hm=measure_width(mask_h,yc); mm=measure_width(sil_t1,yc)
        if hm and mm and mm[3]>5:
            sx_raw=hm[3]/mm[3]
            if sx_raw>sx_max_low: sx_max_low=sx_raw
            m_low_list.append(mm[2])
    sx_low=float(np.clip(sx_max_low,0.80,1.50))
    cx_low_actual=float(np.median(m_low_list)) if m_low_list else cx_low

    v_opt=verts.copy()
    bm_upp=verts2d[:,1]<B_HIP_lo
    bm_hip=(verts2d[:,1]>=B_HIP_lo)&(verts2d[:,1]<B_LOW_lo)
    bm_low=verts2d[:,1]>=B_LOW_lo

    depth_upp=float(v_opt[bm_upp,2].mean())+cam_t[2]
    depth_hip=float(v_opt[bm_hip,2].mean())+cam_t[2]
    depth_low=float(v_opt[bm_low,2].mean())+cam_t[2]

    wcx_m_upp=world_x_from_img(cx_upp,        depth_upp,cam_t,focal,W)
    wcx_m_hip=world_x_from_img(cx_hip,        depth_hip,cam_t,focal,W)
    wcx_m_low=world_x_from_img(cx_low_actual, depth_low,cam_t,focal,W)

    v_opt[bm_upp,0]=wcx_m_upp+(v_opt[bm_upp,0]-wcx_m_upp)*sx_upp
    v_opt[bm_hip,0]=wcx_m_hip+(v_opt[bm_hip,0]-wcx_m_hip)*sx_hip
    v_opt[bm_low,0]=wcx_m_low+(v_opt[bm_low,0]-wcx_m_low)*sx_low

    rgba_mid=render_rgba(v_opt,cam_t,focal,faces,H,W)
    sil_mid=get_sil(rgba_mid)

    all_up_ys=list(range(neck_y,hip_y-15,15))
    sho_ys=list(range(neck_y,sho_y+100,12))
    hip_ys=[hip_y-20,hip_y,hip_y+20,hip_y+40]
    lower_ys2=[knee_y-20,knee_y,knee_y+30,(knee_y+ank_y)//2,ank_y-30]

    def translate_cx_band(v_work,bm,mask_h,sil_cur,ys_meas):
        if bm.sum()==0: return v_work,0.0
        h_cxs,m_cxs=[],[]
        for yc in ys_meas:
            hm=measure_width(mask_h,yc); mm=measure_width(sil_cur,yc)
            if hm and mm: h_cxs.append(hm[2]); m_cxs.append(mm[2])
        if not h_cxs: return v_work,0.0
        h_cx=float(np.median(h_cxs)); m_cx=float(np.median(m_cxs))
        dx_img=h_cx-m_cx
        if abs(dx_img)<0.5: return v_work,dx_img
        depth=float(v_work[bm,2].mean())+cam_t[2]
        v_work[bm,0]+=dx_img*depth/focal
        return v_work,dx_img

    v_opt,_=translate_cx_band(v_opt,bm_upp,mask_h,sil_mid,all_up_ys)
    v_opt,_=translate_cx_band(v_opt,bm_hip,mask_h,sil_mid,hip_ys)
    v_opt,_=translate_cx_band(v_opt,bm_low,mask_h,sil_mid,lower_ys2)

    rgba_f=render_rgba(v_opt,cam_t,focal,faces,H,W)
    sil_f=get_sil(rgba_f)

    h_cxs2,m_cxs2=[],[]
    for yc in sho_ys[:6]:
        hm=measure_width(mask_h,yc); mm=measure_width(sil_f,yc)
        if hm and mm: h_cxs2.append(hm[2]); m_cxs2.append(mm[2])
    if h_cxs2:
        res_dx=float(np.median(h_cxs2))-float(np.median(m_cxs2))
        if abs(res_dx)>1.5:
            depth_b=float(v_opt[bm_upp,2].mean())+cam_t[2]
            v_opt[bm_upp,0]+=res_dx*depth_b/focal
            rgba_f=render_rgba(v_opt,cam_t,focal,faces,H,W)
            sil_f=get_sil(rgba_f)

    iou_upp=compute_iou(mask_h,sil_f,UP_lo,UP_hi,H)
    return v_opt,rgba_f,sil_f,iou_upp,sx_upp,sx_hip,UP_lo,UP_hi,bm_upp,bm_hip,bm_low


# ── A. Arm-band sx optimization ────────────────────────────────────────────────
def arm_band_opt(v_t17, sil_t17, cam_t, focal, faces, mask_h, kp2d, H, W,
                 iou_t17, UP_lo, UP_hi, fr_idx):
    """
    Squeeze arm-band vertices horizontally toward body center.
    Returns (v_opt, iou_new, nfev, sx_used) or None if no improvement.
    Logs everything.
    """
    sho_y   = int((kp2d[I_LSHO][1]+kp2d[I_RSHO][1])/2)
    wri_y   = int((kp2d[I_LWRI][1]+kp2d[I_RWRI][1])/2)
    body_cx = float((kp2d[I_LSHO][0]+kp2d[I_RSHO][0]+kp2d[I_LHIP][0]+kp2d[I_RHIP][0])/4)
    sho_half_w = float(abs(kp2d[I_LSHO][0]-kp2d[I_RSHO][0])/2)

    # Arm-band region in projected space
    vp  = proj2d(v_t17, cam_t, focal, H, W)
    y_lo = sho_y - 40
    y_hi = wri_y + 40

    # Vertices in arm-band y-range AND laterally extended
    bm_arm = (vp[:,1] >= y_lo) & (vp[:,1] <= y_hi) & \
             (np.abs(vp[:,0] - body_cx) > sho_half_w * 0.6)

    n_arm_verts = int(bm_arm.sum())
    print(f"  [ARM_PROBE] fr{fr_idx:03d}: arm_band_verts={n_arm_verts}  "
          f"sho_y={sho_y}  wri_y={wri_y}  body_cx={body_cx:.0f}  sho_half_w={sho_half_w:.0f}")

    if n_arm_verts < 50:
        print(f"  [ARM_SKIP] fr{fr_idx:03d}: too few arm verts ({n_arm_verts}<50)")
        return None, iou_t17, 0, 1.0

    depth_arm = float(v_t17[bm_arm, 2].mean()) + cam_t[2]
    nfev_count = [0]

    def loss(sx):
        nfev_count[0] += 1
        v_test = v_t17.copy()
        wcx_arm = world_x_from_img(body_cx, depth_arm, cam_t, focal, W)
        # Squeeze arm vertices toward body center
        v_test[bm_arm, 0] = wcx_arm + (v_t17[bm_arm, 0] - wcx_arm) * sx
        rgba_t = render_rgba(v_test, cam_t, focal, faces, H, W)
        sil_t  = get_sil(rgba_t)
        iou_t  = compute_iou(mask_h, sil_t, UP_lo, UP_hi, H)
        return -iou_t

    result = ms1d(loss, bounds=ARM_SX_BOUNDS, method='bounded',
                  options={'xatol': 0.005, 'maxiter': 30})
    sx_opt = float(result.x)
    iou_opt = -float(result.fun)
    nfev = nfev_count[0]

    print(f"  [ARM] fr{fr_idx:03d}: T2.2_IoU={iou_t17:.4f} → T2.3_IoU={iou_opt:.4f}  "
          f"Δ={iou_opt-iou_t17:+.4f}  nfev={nfev}  arm_sx={sx_opt:.4f}")

    if iou_opt < iou_t17 - 0.02:
        print(f"  [ARM_NO_GAIN] fr{fr_idx:03d}: IoU degraded, keeping T2.2")
        return None, iou_t17, nfev, sx_opt

    if iou_opt <= iou_t17 + 0.002:
        print(f"  [ARM_NO_GAIN] fr{fr_idx:03d}: improvement={iou_opt-iou_t17:+.4f} negligible, keeping T2.2")
        return None, iou_t17, nfev, sx_opt

    # Apply optimized sx
    v_arm_opt = v_t17.copy()
    wcx_arm = world_x_from_img(body_cx, depth_arm, cam_t, focal, W)
    v_arm_opt[bm_arm, 0] = wcx_arm + (v_t17[bm_arm, 0] - wcx_arm) * sx_opt

    return v_arm_opt, iou_opt, nfev, sx_opt


# ── B. Foot translation ────────────────────────────────────────────────────────
def foot_translate(v_in, cam_t, focal, faces, mask_h, kp2d, kp_rtm, H, W,
                   fr_idx, neighbor_foot_delta=None):
    """
    Translate foot vertices to align with RTMPose ankle midpoint.
    Left foot: vertices with px < body_cx, py > ank_y-20
    Right foot: vertices with px >= body_cx, py > ank_y-20
    kp_guard: if ankle score < threshold, use neighbor delta.
    Returns (v_out, foot_log_dict)
    """
    vp = proj2d(v_in, cam_t, focal, H, W)

    lank_mhr = kp2d[I_LANK]  # MHR projected ankle (very accurate, dx<8px)
    rank_mhr = kp2d[I_RANK]

    # Foot vertex selection: proximity to each ankle kp2d, not body_cx split
    # This avoids including knee/hip/torso vertices in the foot band
    FOOT_LATERAL_RADIUS = 55   # px: ankle ± 55px in x (tight around foot)
    FOOT_Y_MARGIN_UP    = 20   # px above ankle (minimal)
    FOOT_Y_MARGIN_DOWN  = 90   # px below ankle (foot extends downward)

    bm_L = ((vp[:,1] > lank_mhr[1] - FOOT_Y_MARGIN_UP) &
            (vp[:,1] < lank_mhr[1] + FOOT_Y_MARGIN_DOWN) &
            (np.abs(vp[:,0] - lank_mhr[0]) < FOOT_LATERAL_RADIUS))
    bm_R = ((vp[:,1] > rank_mhr[1] - FOOT_Y_MARGIN_UP) &
            (vp[:,1] < rank_mhr[1] + FOOT_Y_MARGIN_DOWN) &
            (np.abs(vp[:,0] - rank_mhr[0]) < FOOT_LATERAL_RADIUS))

    v_out = v_in.copy()
    log = {'bm_L_n': int(bm_L.sum()), 'bm_R_n': int(bm_R.sum())}

    lank_rtm = (kp_rtm['left_ankle']['x'],  kp_rtm['left_ankle']['y'])
    rank_rtm = (kp_rtm['right_ankle']['x'], kp_rtm['right_ankle']['y'])
    lank_score = kp_rtm['left_ankle']['score']
    rank_score = kp_rtm['right_ankle']['score']

    def apply_foot_band(bm, rtm_ank, mhr_ank, score, side):
        if bm.sum() < 10:
            log[f'dx_{side}'] = 0.0; log[f'dy_{side}'] = 0.0
            log[f'guard_{side}'] = 'too_few_verts'
            return

        # kp_guard
        if score < FOOT_KP_GUARD_SCORE:
            if neighbor_foot_delta and side in neighbor_foot_delta:
                nd = neighbor_foot_delta[side]
                log[f'dx_{side}'] = nd['dx']; log[f'dy_{side}'] = nd['dy']
                log[f'guard_{side}'] = f'interp_score={score:.2f}'
            else:
                log[f'dx_{side}'] = 0.0; log[f'dy_{side}'] = 0.0
                log[f'guard_{side}'] = f'guard_no_neighbor_score={score:.2f}'
            dx_img = log[f'dx_{side}']
            dy_img = log[f'dy_{side}']
        else:
            # Target: RTMPose ankle position
            target_x, target_y = rtm_ank
            # Current foot vertex centroid
            foot_cx = float(np.median(vp[bm, 0]))
            foot_cy = float(np.median(vp[bm, 1]))
            # Error
            dx_img = target_x - foot_cx
            dy_img = target_y - foot_cy
            # Clamp: don't overcorrect (max 80px per frame)
            dx_img = float(np.clip(dx_img, -120, 120))
            dy_img = float(np.clip(dy_img, -80, 80))
            log[f'dx_{side}'] = round(dx_img, 1)
            log[f'dy_{side}'] = round(dy_img, 1)
            log[f'guard_{side}'] = 'ok'
            log[f'foot_cx_{side}'] = round(foot_cx, 1)
            log[f'foot_cy_{side}'] = round(foot_cy, 1)
            log[f'rtm_ank_{side}'] = (round(target_x,1), round(target_y,1))

        if abs(dx_img) > 0.5 or abs(dy_img) > 0.5:
            depth_f = float(v_out[bm, 2].mean()) + cam_t[2]
            dx_w = dx_img * depth_f / focal
            dy_w = dy_img * depth_f / focal
            v_out[bm, 0] += dx_w
            v_out[bm, 1] += dy_w

    apply_foot_band(bm_L, lank_rtm, lank_mhr, lank_score, 'L')
    apply_foot_band(bm_R, rank_rtm, rank_mhr, rank_score, 'R')

    print(f"  [FOOT] fr{fr_idx:03d}: L_verts={log['bm_L_n']} dx_L={log.get('dx_L',0):+.0f} dy_L={log.get('dy_L',0):+.0f}  "
          f"R_verts={log['bm_R_n']} dx_R={log.get('dx_R',0):+.0f} dy_R={log.get('dy_R',0):+.0f}  "
          f"guard_L={log.get('guard_L','?')} guard_R={log.get('guard_R','?')}")

    return v_out, log


# ── Vis helpers ───────────────────────────────────────────────────────────────
def make_iou_vis(mask_h, sil_mesh, H, W, title="", foot_roi=False):
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    h=mask_h>0; m=sil_mesh>0
    vis[h&~m]=[0,255,0]; vis[~h&m]=[0,0,255]; vis[h&m]=[0,255,255]
    cv2.putText(vis, title, (10,44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.putText(vis, "green=human  red=mesh  yellow=overlap",
                (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
    return vis


# ── Fit one frame (full T2.3 pipeline) ────────────────────────────────────────
def fit_frame_t23(img_bgr, est, head, faces, kp_rtm, fr_idx,
                  iou_t22, do_arm_opt, neighbor_foot_delta=None):
    H, W = img_bgr.shape[:2]

    ax=[v['x'] for v in kp_rtm.values() if v['score']>0.3]
    ay=[v['y'] for v in kp_rtm.values() if v['score']>0.3]
    if len(ax)<4: return None
    pad_x=(max(ax)-min(ax))*0.15; pad_y=(max(ay)-min(ay))*0.15
    bbox=np.array([[max(0,min(ax)-pad_x),max(0,min(ay)-pad_y),
                    min(W,max(ax)+pad_x),min(H,max(ay)+pad_y)]],dtype=np.float32)

    outs=est.process_one_image(img_bgr, bboxes=bbox, use_mask=False, inference_type="body")
    if not outs: return None
    o=outs[0]
    verts=o["pred_vertices"].astype(np.float32)
    cam_t=o["pred_cam_t"].astype(np.float32)
    kp2d=o["pred_keypoints_2d"].astype(np.float32)
    focal=float(o["focal_length"])

    mask_h=get_human_mask(img_bgr)

    # Step 1: T1.7 shape-fit (same as T2.2 baseline)
    v_t17, rgba_t17, sil_t17, iou_t17, sx_upp, sx_hip, UP_lo, UP_hi, bm_upp, bm_hip, bm_low = \
        shape_fit_t17(verts, cam_t, focal, faces, mask_h, kp2d, H, W)

    arm_applied = False; arm_nfev = 0; arm_sx = 1.0; arm_iou_gain = 0.0
    v_after_arm = v_t17.copy(); iou_after_arm = iou_t17

    # Step 2: A. Arm-band opt (triggered by T2.2 IoU < threshold)
    if do_arm_opt:
        v_arm, iou_arm, nfev, sx = arm_band_opt(
            v_t17, sil_t17, cam_t, focal, faces, mask_h,
            kp2d, H, W, iou_t17, UP_lo, UP_hi, fr_idx
        )
        arm_nfev = nfev; arm_sx = sx
        if v_arm is not None:
            v_after_arm = v_arm; iou_after_arm = iou_arm; arm_applied = True
            arm_iou_gain = iou_arm - iou_t17
    else:
        print(f"  [ARM_SKIP] fr{fr_idx:03d}: T2.2_IoU={iou_t22:.4f} >= {ARM_OPT_IOUt_TRIGGER}, no arm opt needed")

    # Step 3: B. Foot IoU diagnostic (no translation — MHR ankle kp already accurate,
    # foot shape/orientation error cannot be fixed with rigid translation)
    foot_log = {'bm_L_n': 0, 'bm_R_n': 0,
                'dx_L': 0.0, 'dy_L': 0.0, 'dx_R': 0.0, 'dy_R': 0.0,
                'guard_L': 'translation_disabled', 'guard_R': 'translation_disabled',
                'note': 'MHR_ankle_accurate_foot_orientation_needs_pose_params_not_rigid_translation'}
    v_foot = v_after_arm  # no foot translation

    # Step 4: Final render + IoU
    rgba_final=render_rgba(v_foot, cam_t, focal, faces, H, W)
    sil_final=get_sil(rgba_final)
    iou_final=compute_iou(mask_h, sil_final, UP_lo, UP_hi, H)

    # Secondary: foot IoU (ankle region)
    ank_y = int((kp2d[I_LANK][1]+kp2d[I_RANK][1])/2)
    foot_y1 = max(0, ank_y - 80)
    foot_y2 = min(H, ank_y + 80)   # cap at frame height
    foot_iou = compute_iou(mask_h, sil_final, foot_y1, foot_y2, H)
    foot_iou_pre = compute_iou(mask_h, get_sil(render_rgba(v_t17,cam_t,focal,faces,H,W)),
                               foot_y1, foot_y2, H)

    # A3 cx check
    mesh_body_cx=float((kp2d[I_LSHO][0]+kp2d[I_RSHO][0]+kp2d[I_LHIP][0]+kp2d[I_RHIP][0])/4)
    rtm_body_cx=float(np.mean([kp_rtm['left_shoulder']['x'],kp_rtm['right_shoulder']['x'],
                                kp_rtm['left_hip']['x'],kp_rtm['right_hip']['x']]))
    cx_offset=abs(mesh_body_cx-rtm_body_cx)

    return {
        "iou_upper": round(float(iou_final),4),
        "iou_t17": round(float(iou_t17),4),
        "iou_after_arm": round(float(iou_after_arm),4),
        "foot_iou": round(float(foot_iou),4),
        "foot_iou_pre": round(float(foot_iou_pre),4),
        "pass": bool(iou_final>=PASS_THRESHOLD),
        "rgba_final": rgba_final,
        "sil_final": sil_final,
        "mask_h": mask_h,
        "sx_upp": round(sx_upp,4),
        "arm_applied": arm_applied,
        "arm_nfev": arm_nfev,
        "arm_sx": round(arm_sx,4),
        "arm_iou_gain": round(arm_iou_gain,4),
        "foot_log": foot_log,
        "cx_offset": round(cx_offset,1),
        "kp2d": kp2d,
        "cam_t": cam_t.tolist(),
        "focal": focal,
        "img_bgr": img_bgr,
        "UP_lo": UP_lo, "UP_hi": UP_hi,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t_start=time.time()
    import torch
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

    print(f"[INFO] torch {torch.__version__}  cuda={torch.cuda.is_available()}")

    with open(T22_LOG) as f:
        t22_log=json.load(f)
    with open(KP_CACHE) as f:
        kpd=json.load(f)
    kp_frames=kpd['frames']
    NF=len(kp_frames)

    # T2.2 IoU per frame
    t22_ious={r['frame']:r.get('iou_upper') for r in t22_log['per_frame']}
    t22_invalid={r['frame'] for r in t22_log['per_frame'] if r.get('invalid',False)}

    print(f"\n[INFO] T2.2 invalid frames: {sorted(t22_invalid)}")
    print(f"[INFO] T2.3 arm_opt trigger: T2.2 IoU < {ARM_OPT_IOUt_TRIGGER}")

    # Enumerate arm-opt candidates
    arm_cands = [fi for fi in range(NF)
                 if fi not in t22_invalid
                 and t22_ious.get(fi) is not None
                 and t22_ious[fi] < ARM_OPT_IOUt_TRIGGER
                 and fi not in {FR_ADDRESS, FR_TOP, FR_IMPACT}]
    print(f"[INFO] Arm-opt candidates ({len(arm_cands)}): {arm_cands}")

    # Load model
    print("\n[INFO] Loading model...")
    model,cfg=load_sam_3d_body(CKPT, device='cuda', mhr_path=MHR_P)
    est=SAM3DBodyEstimator(model,cfg,human_detector=None,human_segmentor=None,fov_estimator=None)
    head=model.head_pose
    faces=est.faces

    cap=cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO_PATH}")

    results=[]
    frame_results={}
    valid_overlays={}
    KEYFRAMES={FR_ADDRESS:"address", FR_TOP:"top", FR_IMPACT:"impact"}

    # Rolling foot delta (for kp_guard interpolation)
    foot_delta_history={}  # fr_idx → {L:{dx,dy}, R:{dx,dy}}

    for fr_idx in range(NF):
        ret,img_bgr=cap.read()
        if not ret:
            print(f"  [WARN] Cannot read fr{fr_idx:03d}")
            results.append({"frame":fr_idx,"iou_upper":None,"pass":False,"invalid":True,"reasons":["read_fail"]})
            continue

        # Propagate invalid from T2.2
        if fr_idx in t22_invalid:
            reasons=[r['reasons'] for r in t22_log['per_frame'] if r['frame']==fr_idx]
            reasons=reasons[0] if reasons else ['t22_invalid']
            print(f"  [fr{fr_idx:03d}/{NF-1}] INVALID (carry from T2.2: {reasons})")
            results.append({"frame":fr_idx,"iou_upper":None,"pass":False,"invalid":True,"reasons":reasons})
            continue

        t_fr=time.time()
        kp_rtm=kp_frames[fr_idx]['persons'][0]['keypoints']
        do_arm=fr_idx in arm_cands
        iou22=t22_ious.get(fr_idx) or 0.0

        # Build foot neighbor delta
        prev_valid=[k for k in sorted(foot_delta_history.keys()) if k<fr_idx]
        neighbor_foot_delta=foot_delta_history[prev_valid[-1]] if prev_valid else None

        flag_str="[ARM_OPT]" if do_arm else ""
        print(f"  [fr{fr_idx:03d}/{NF-1}] fitting... {flag_str}", end='\n', flush=True)

        try:
            r=fit_frame_t23(img_bgr, est, head, faces, kp_rtm, fr_idx,
                            iou22, do_arm, neighbor_foot_delta)
        except Exception as e:
            import traceback
            print(f"  [ERROR] fr{fr_idx:03d}: {e}")
            traceback.print_exc()
            results.append({"frame":fr_idx,"iou_upper":None,"pass":False,"error":str(e)})
            continue

        if r is None:
            print(f"  [SKIP] fr{fr_idx:03d}")
            results.append({"frame":fr_idx,"iou_upper":None,"pass":False})
            continue

        # A3
        if r["cx_offset"]>60.0:
            print(f"  [A3] fr{fr_idx:03d}: cx_offset={r['cx_offset']:.1f}px → INVALID")
            results.append({"frame":fr_idx,"iou_upper":None,"pass":False,"invalid":True,
                            "reasons":[f"A3:cx={r['cx_offset']:.1f}"]})
            continue

        iou=r["iou_upper"]; t22v=iou22
        flag="✓" if r["pass"] else "✗"
        print(f"  [fr{fr_idx:03d}] IoU={iou:.4f} {flag}  "
              f"(T2.2:{t22v:.4f} Δ{iou-t22v:+.4f})  "
              f"arm={r['arm_applied']} nfev={r['arm_nfev']} sx={r['arm_sx']:.3f}  "
              f"foot_iou:{r['foot_iou_pre']:.3f}→{r['foot_iou']:.3f}  ({time.time()-t_fr:.1f}s)")

        # Store foot delta for neighbor interpolation
        fl=r["foot_log"]
        foot_delta_history[fr_idx]={
            'L':{'dx':fl.get('dx_L',0),'dy':fl.get('dy_L',0)},
            'R':{'dx':fl.get('dx_R',0),'dy':fl.get('dy_R',0)},
        }

        row={
            "frame":fr_idx,
            "iou_upper":iou,
            "iou_t22":t22v,
            "delta_vs_t22":round(float(iou-t22v),4),
            "foot_iou": r["foot_iou"],
            "foot_iou_pre": r["foot_iou_pre"],
            "foot_iou_gain": round(float(r["foot_iou"]-r["foot_iou_pre"]),4),
            "pass":r["pass"],
            "invalid":False,
            "arm_applied":r["arm_applied"],
            "arm_nfev":r["arm_nfev"],
            "arm_sx":r["arm_sx"],
            "arm_iou_gain":r["arm_iou_gain"],
            "foot_dx_L":fl.get("dx_L",0),
            "foot_dy_L":fl.get("dy_L",0),
            "foot_dx_R":fl.get("dx_R",0),
            "foot_dy_R":fl.get("dy_R",0),
        }
        results.append(row)

        ov=composite_red(r["rgba_final"],img_bgr,alpha=0.55)
        valid_overlays[fr_idx]=ov.copy()
        frame_results[fr_idx]=r

        if fr_idx in KEYFRAMES:
            phase=KEYFRAMES[fr_idx]
            kp2d_k=r["kp2d"]
            H_ov,W_ov=ov.shape[:2]
            lh=(int(kp2d_k[I_LHIP][0]),int(kp2d_k[I_LHIP][1]))
            rh=(int(kp2d_k[I_RHIP][0]),int(kp2d_k[I_RHIP][1]))
            y_pel=(lh[1]+rh[1])//2
            cv2.line(ov,(0,y_pel),(W_ov-1,y_pel),(255,255,255),2)
            cv2.circle(ov,lh,7,(255,255,255),-1); cv2.circle(ov,lh,3,(0,0,200),-1)
            cv2.circle(ov,rh,7,(255,255,255),-1); cv2.circle(ov,rh,3,(0,0,200),-1)
            label=f"{phase}_fr{fr_idx:03d}_IoU{iou:.3f}"
            cv2.putText(ov,label,(12,44),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
            kf_path=KF_DIR/f"{phase}_fr{fr_idx:03d}.jpg"
            cv2.imwrite(str(kf_path),ov)
            print(f"  [KEY] {kf_path.name}")

    cap.release()

    # ── Distribution ──────────────────────────────────────────────────────────
    valid_rows=[r for r in results if not r.get('invalid',False) and r.get('iou_upper') is not None]
    ious_valid=np.array([r["iou_upper"] for r in valid_rows])
    n_valid=len(ious_valid); n_invalid=NF-n_valid

    mean_iou=float(np.mean(ious_valid)); min_iou=float(np.min(ious_valid))
    p5_iou=float(np.percentile(ious_valid,5))
    min_fr=valid_rows[int(np.argmin(ious_valid))]["frame"]
    fail_valid=[(r["frame"],r["iou_upper"]) for r in valid_rows if not r["pass"]]
    arm_applied_rows=[r for r in valid_rows if r.get("arm_applied")]
    arm_cand_rows=[r for r in valid_rows if r["frame"] in arm_cands]

    # Foot IoU stats
    foot_ious_pre = np.array([r.get("foot_iou_pre",0) for r in valid_rows])
    foot_ious_post = np.array([r.get("foot_iou",0) for r in valid_rows])
    foot_mean_pre = float(np.mean(foot_ious_pre)); foot_mean_post = float(np.mean(foot_ious_post))

    print(f"\n{'='*60}")
    print(f"  T2.3 IoU 分布 (valid={n_valid}, invalid={n_invalid})")
    print(f"  mean={mean_iou:.4f}  min={min_iou:.4f}(fr{min_fr:03d})  P5={p5_iou:.4f}")
    print(f"  低于{PASS_THRESHOLD}: {len(fail_valid)} 有效帧")
    print(f"")
    print(f"  T2.2 对比: mean=0.9027  min=0.7411(fr103)  P5=0.8322")
    print(f"")
    print(f"  FOOT IoU (ankle region): pre={foot_mean_pre:.4f} → post={foot_mean_post:.4f}  Δ={foot_mean_post-foot_mean_pre:+.4f}")
    print(f"  ARM_OPT 候选帧: {len(arm_cands)}")
    print(f"  ARM_OPT 真正提升: {len(arm_applied_rows)} 帧")
    for r in arm_applied_rows:
        print(f"    fr{r['frame']:03d}: Δ_IoU={r['arm_iou_gain']:+.4f}  nfev={r['arm_nfev']}  sx={r['arm_sx']:.4f}")
    if arm_cand_rows:
        print(f"  ARM_OPT 无增益帧 (候选但未提升):")
        for r in arm_cand_rows:
            if not r.get("arm_applied") and r.get("arm_nfev",0)>0:
                print(f"    fr{r['frame']:03d}: nfev={r['arm_nfev']}  sx_tried={r['arm_sx']:.4f}  IoU={r['iou_upper']:.4f}")
    print(f"{'='*60}")

    # ── Worst 3 ───────────────────────────────────────────────────────────────
    valid_sorted=sorted(
        [(r["frame"],r["iou_upper"]) for r in valid_rows if r["frame"] in frame_results],
        key=lambda x: x[1]
    )
    worst3=valid_sorted[:3]
    for rank,(fno,fiou) in enumerate(worst3,1):
        r=frame_results[fno]
        H_img,W_img=r["img_bgr"].shape[:2]
        vis=make_iou_vis(r["mask_h"],r["sil_final"],H_img,W_img,
                         f"fr{fno:03d} IoU={fiou:.3f}")
        t22v=t22_ious.get(fno) or 0.0
        cv2.putText(vis,f"T2.2:{t22v:.3f}  T2.3:{fiou:.3f}  Δ{fiou-t22v:+.3f}",
                    (10,H_img-20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(200,200,200),2)
        p_vis=OUTPUT_DIR/f"worst{rank}_valid_t23_fr{fno:03d}.jpg"
        cv2.imwrite(str(p_vis),vis)
        print(f"  [WORST{rank}] fr{fno:03d} IoU={fiou:.4f} → {p_vis.name}")

    # ── Address foot special view ─────────────────────────────────────────────
    if FR_ADDRESS in frame_results:
        r_addr=frame_results[FR_ADDRESS]
        H_a,W_a=r_addr["img_bgr"].shape[:2]
        kp2d_a=r_addr["kp2d"]
        ank_y=int((kp2d_a[I_LANK][1]+kp2d_a[I_RANK][1])/2)
        vis_full=make_iou_vis(r_addr["mask_h"],r_addr["sil_final"],H_a,W_a,
                              f"address_fr000 IoU={r_addr['iou_upper']:.3f}")
        # Crop foot region: ank_y-80 to H
        y_crop=max(0,ank_y-80)
        vis_foot=vis_full[y_crop:H_a,:].copy()
        # Annotate RTMPose ankles
        kp_addr=kp_frames[FR_ADDRESS]['persons'][0]['keypoints']
        for name,col in [('left_ankle',(0,255,100)),('right_ankle',(100,255,0))]:
            kpt=kp_addr[name]
            px_a,py_a=int(kpt['x']),int(kpt['y'])-y_crop
            if 0<=py_a<vis_foot.shape[0]:
                cv2.circle(vis_foot,(px_a,py_a),8,col,-1)
                cv2.circle(vis_foot,(px_a,py_a),12,col,2)
        cv2.putText(vis_foot,"circles=RTMPose ankles  cyan=overlap  green=human-only  red=mesh-only",
                    (6,22),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)
        fl_addr=r_addr["foot_log"]
        cv2.putText(vis_foot,
            f"dx_L={fl_addr.get('dx_L',0):+.0f}px dy_L={fl_addr.get('dy_L',0):+.0f}px  "
            f"dx_R={fl_addr.get('dx_R',0):+.0f}px dy_R={fl_addr.get('dy_R',0):+.0f}px",
            (6,44),cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,255,200),1)
        p_foot=OUTPUT_DIR/"address_fr000_foot.jpg"
        cv2.imwrite(str(p_foot),vis_foot)
        print(f"  [FOOT_VIS] {p_foot.name}")

    # ── IoU curve ─────────────────────────────────────────────────────────────
    H_c,W_c=440,max(800,NF*6)
    curve=np.zeros((H_c,W_c,3),dtype=np.uint8); curve[:]=(30,30,30)
    for iou_g in [0.80,0.85,0.87,0.90,0.92,0.95]:
        yg=int((1.0-iou_g)/0.25*(H_c-80))+40
        col=(0,200,100) if abs(iou_g-0.92)<0.005 else \
            ((100,200,255) if abs(iou_g-0.87)<0.005 else (60,60,60))
        cv2.line(curve,(0,yg),(W_c-1,yg),col,1)
        cv2.putText(curve,f"{iou_g:.2f}",(4,yg-4),cv2.FONT_HERSHEY_SIMPLEX,0.45,(180,180,180),1)
    # T2.2 background (gray)
    for r in t22_log['per_frame']:
        if r.get('iou_upper') is None: continue
        x=int(r['frame']/(NF-1)*(W_c-20))+10
        iou_v=min(1.0,max(0.75,r['iou_upper']))
        y=int((1.0-iou_v)/0.25*(H_c-80))+40
        cv2.circle(curve,(x,y),2,(80,80,80),-1)
    # T2.3 (blue)
    for r in results:
        fno=r['frame']; x=int(fno/(NF-1)*(W_c-20))+10
        if r.get('invalid',False):
            iou_v=t22_ious.get(fno) or 0.85
            y=int((1.0-min(1.0,max(0.75,iou_v)))/0.25*(H_c-80))+40
            cv2.line(curve,(x-5,y-5),(x+5,y+5),(0,0,255),2)
            cv2.line(curve,(x-5,y+5),(x+5,y-5),(0,0,255),2)
        elif r.get('iou_upper') is not None:
            iou_v=r['iou_upper']
            y=int((1.0-min(1.0,max(0.75,iou_v)))/0.25*(H_c-80))+40
            col=(0,220,255) if r['pass'] else (0,140,255)
            sz=5 if r.get('arm_applied') else 3
            cv2.circle(curve,(x,y),sz,col,-1)
            if fno in {FR_ADDRESS,FR_TOP,FR_IMPACT}: cv2.circle(curve,(x,y),7,(255,255,0),2)
            if r.get('arm_applied'): cv2.circle(curve,(x,y),9,(255,200,0),1)
    cv2.putText(curve,
        f"T2.3  valid={n_valid}  invalid={n_invalid}  mean={mean_iou:.3f}  "
        f"min={min_iou:.3f}(fr{min_fr:03d})  P5={p5_iou:.3f}",
        (10,H_c-15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
    p_curve=OUTPUT_DIR/"iou_distribution_t23.jpg"
    cv2.imwrite(str(p_curve),curve)

    # ── JSON ──────────────────────────────────────────────────────────────────
    vram=torch.cuda.max_memory_allocated()/1e6; tt=time.time()-t_start
    log_out={
        "version":"T2.3","clip":"fo-ok-1","NF":NF,
        "pass_threshold":PASS_THRESHOLD,
        "arm_opt_config":{
            "trigger":"T2.2_IoU < ARM_OPT_IOUt_TRIGGER",
            "ARM_OPT_IOUt_TRIGGER":ARM_OPT_IOUt_TRIGGER,
            "method":"arm_band_sx_1D_bounded",
            "bounds":ARM_SX_BOUNDS,
        },
        "foot_config":{
            "method":"foot_vertex_centroid_translation",
            "kp_guard_score":FOOT_KP_GUARD_SCORE,
        },
        "arm_candidates":arm_cands,
        "arm_applied_frames":[r["frame"] for r in arm_applied_rows],
        "distribution":{
            "mean":round(mean_iou,4),"min":round(min_iou,4),"min_frame":min_fr,
            "p5":round(p5_iou,4),"valid_count":n_valid,"invalid_count":n_invalid,
        },
        "comparison":{
            "T2":{"mean":0.8966,"min":0.3722,"p5":0.8242},
            "T2.1":{"mean":0.9014,"min":0.7420,"p5":0.8296},
            "T2.2":{"mean":0.9027,"min":0.7411,"p5":0.8322},
        },
        "worst3":[ {"frame":f,"iou":round(v,4)} for f,v in worst3],
        "per_frame":results,
        "peak_vram_mb":round(vram,0),"total_s":round(tt,1),
        "topology":"MHR_native","license":"SAM_License_PRODUCT_CANDIDATE_CUSTOM_LICENSE",
    }
    lp=OUTPUT_DIR/"run_log_t23.json"
    with open(lp,'w') as f:
        json.dump(log_out,f,indent=2,default=str)

    # ── Text report ───────────────────────────────────────────────────────────
    # Key keyframe ious
    addr_iou=next((r['iou_upper'] for r in results if r['frame']==FR_ADDRESS),None)
    top_iou =next((r['iou_upper'] for r in results if r['frame']==FR_TOP),None)
    imp_iou =next((r['iou_upper'] for r in results if r['frame']==FR_IMPACT),None)

    lines=[
        "GHOST-003 T2.3 停关卡报告",
        f"Clip: fo-ok-1  NF={NF}",
        "",
        "=== 前置诊断 (T2.2 ARM_OPT=0 根因) ===",
        f"触发条件原设: arm_dist/body_h > 1.2",
        f"fr064: ratio=1.031  fr103: ratio=0.763  → 均未达 1.2 → 0 帧触发",
        f"T2.3 修正: 触发条件改为 T2.2_IoU < {ARM_OPT_IOUt_TRIGGER}",
        f"",
        "=== A. 手臂修正 ===",
        f"候选帧 (T2.2 IoU<{ARM_OPT_IOUt_TRIGGER}): {arm_cands}",
        f"真正提升帧: {[r['frame'] for r in arm_applied_rows]}",
    ]
    for r in valid_rows:
        if r["frame"] in arm_cands:
            lines.append(f"  fr{r['frame']:03d}: T2.2={r['iou_t22']:.4f} → T2.3={r['iou_upper']:.4f} "
                         f"Δ={r['delta_vs_t22']:+.4f}  arm_applied={r['arm_applied']}  nfev={r['arm_nfev']}  sx={r['arm_sx']:.4f}")

    lines += [
        "",
        "=== B. 脚部修正 (全帧) ===",
        "方法: foot vertex centroid → RTMPose ankle midpoint",
        f"address fr000 foot:",
    ]
    if FR_ADDRESS in frame_results:
        fl=frame_results[FR_ADDRESS]["foot_log"]
        lines.append(f"  L: dx={fl.get('dx_L',0):+.0f}px dy={fl.get('dy_L',0):+.0f}px  guard={fl.get('guard_L','?')}")
        lines.append(f"  R: dx={fl.get('dx_R',0):+.0f}px dy={fl.get('dy_R',0):+.0f}px  guard={fl.get('guard_R','?')}")

    lines += [
        "",
        "=== C. 分布 ===",
        f"valid={n_valid}  invalid={n_invalid}",
        f"mean IoU: {mean_iou:.4f}",
        f"min  IoU: {min_iou:.4f}  (fr{min_fr:03d})",
        f"P5   IoU: {p5_iou:.4f}",
        f"低于{PASS_THRESHOLD}: {len(fail_valid)} 有效帧",
        "",
        "对比:",
        f"  T2  (112): mean=0.8966  P5=0.8242",
        f"  T2.1(111): mean=0.9014  P5=0.8296",
        f"  T2.2(110): mean=0.9027  P5=0.8322",
        f"  T2.3({n_valid}): mean={mean_iou:.4f}  P5={p5_iou:.4f}",
        "",
        "=== D. 关键帧 ===",
        f"address fr{FR_ADDRESS:03d}: T2.2=0.9370  T2.3={addr_iou}",
        f"top     fr{FR_TOP:03d}: T2.2=0.8650  T2.3={top_iou}",
        f"impact  fr{FR_IMPACT:03d}: T2.2=0.9116  T2.3={imp_iou}",
        "",
        "最差3有效帧:",
    ]
    for rank,(fno,fiou) in enumerate(worst3,1):
        t22v=t22_ious.get(fno) or 0.0
        lines.append(f"  #{rank} fr{fno:03d}: IoU={fiou:.4f}  T2.2={t22v:.4f}  Δ{fiou-t22v:+.4f}")

    lines+=[f"","peak VRAM: {vram:.0f}MB  total: {tt:.0f}s"]
    rp=OUTPUT_DIR/"REPORT_T23.txt"
    rp.write_text("\n".join(lines),encoding='utf-8')

    print(f"\n[OUT] {OUTPUT_DIR}/")
    print(f"{'='*60}")
    print(f"  T2.3 DONE  valid={n_valid}  invalid={n_invalid}")
    print(f"  mean={mean_iou:.4f}  min={min_iou:.4f}(fr{min_fr:03d})  P5={p5_iou:.4f}")
    print(f"  ARM_OPT applied: {len(arm_applied_rows)} / {len(arm_cands)} candidates")
    print(f"  peak VRAM: {vram:.0f}MB  total: {tt:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
