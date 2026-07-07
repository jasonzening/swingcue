"""
ghost003_t17_iou_fit.py  —  GHOST-003 T1.7 IoU-based shape fitting
策略: 废弃单向 edge miss; 以「最大化上半身 IoU」为优化目标
  - IoU = |human ∩ mesh| / |human ∪ mesh| in upper body y-region
  - scipy minimize_scalar (bounded) 对每 band 独立优化 sx
  - 快速代理: column-remap proxy (无需 pyrender), O(microseconds/eval)
  - 最终 pyrender 渲染输出真实 IoU
  - 下肢: 达标即止, 沿用 T1.6 策略 (sx=max_raw, containment=1.0)

可视化 (silhouette_compare): 左=T1.6 / 右=T1.7
  绿=仅真人  红=仅 mesh  黄=重合(intersection)  黄越多越好

报告: 只报上半身 IoU 一个数 + 右侧黄色比例

范围: 单帧·真实姿态·address 相位·不改姿态·不做整段·不碰球杆
授权: SAM License (PRODUCT_CANDIDATE_CUSTOM_LICENSE)
"""

import os, sys, time, json, pathlib
import numpy as np
import cv2
from scipy.optimize import minimize_scalar

SAM3D_REPO = "/home/jason/projects/sam-3d-body"
SWINGCUE   = "/home/jason/projects/swingcue-postest"
sys.path.insert(0, SAM3D_REPO)
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

OUTPUT_DIR = pathlib.Path(SWINGCUE) / "output" / "ghost003"
KP_CACHE   = pathlib.Path(SWINGCUE) / "engine/kp_cache/batch2/fo-ok-1.json"
ADDR_JPG   = OUTPUT_DIR / "address_frame.jpg"
CKPT  = "/home/jason/.cache/sam3d/sam-3d-body-dinov3/model.ckpt"
MHR_P = "/home/jason/.cache/sam3d/sam-3d-body-dinov3/assets/mhr_model.pt"

I_NOSE=0; I_LSHO=5; I_RSHO=6; I_LELB=7; I_RELB=8
I_LHIP=9; I_RHIP=10; I_LKNE=11; I_RKNE=12; I_LANK=13; I_RANK=14
I_RWRI=41; I_LWRI=62; I_NECK=69

RTM_GT = {
    'nose': (294.5,517.4), 'left_shoulder': (366.7,548.4), 'right_shoulder': (256.3,565.9),
    'left_elbow': (347.1,638.2), 'right_elbow': (279.0,647.5),
    'left_wrist': (323.4,706.3), 'right_wrist': (299.6,721.8),
    'left_hip': (357.4,679.5), 'right_hip': (292.4,684.6),
    'left_knee': (371.9,782.7), 'right_knee': (267.6,787.9),
    'left_ankle': (391.5,910.7), 'right_ankle': (262.5,912.8),
}
MHR_NAMED = {
    'nose':I_NOSE,'neck':I_NECK,'left_shoulder':I_LSHO,'right_shoulder':I_RSHO,
    'left_elbow':I_LELB,'right_elbow':I_RELB,'left_wrist':I_LWRI,'right_wrist':I_RWRI,
    'left_hip':I_LHIP,'right_hip':I_RHIP,'left_knee':I_LKNE,'right_knee':I_RKNE,
    'left_ankle':I_LANK,'right_ankle':I_RANK,
}


# ── Projection ───────────────────────────────────────────────────────────────
def project_verts(verts, cam_t, focal, H, W):
    vx,vy,vz = verts[:,0], verts[:,1], verts[:,2]
    d  = np.where(np.abs(vz+cam_t[2])<1e-6, 1e-6, vz+cam_t[2])
    px = focal*(vx-cam_t[0])/d + W/2.0
    py = focal*(vy+cam_t[1])/d + H/2.0
    return np.stack([px,py],axis=1)

def world_x_from_img(img_x, depth, cam_t, focal, W):
    return (img_x-W/2.0)*depth/focal + cam_t[0]


# ── Human mask ───────────────────────────────────────────────────────────────
def get_human_mask(img_bgr):
    import rembg
    out = rembg.remove(img_bgr)
    mask = (out[:,:,3] > 40).astype(np.uint8)*255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    return mask


# ── Render ───────────────────────────────────────────────────────────────────
def render_rgba(verts, cam_t, focal, faces, H, W):
    from sam_3d_body.visualization.renderer import Renderer
    dummy = np.zeros((H,W,3), dtype=np.uint8)
    r = Renderer(focal_length=focal, faces=faces)
    return r(verts, cam_t, dummy, mesh_base_color=(1.,0.,0.),
             scene_bg_color=(0,0,0), return_rgba=True)

def get_sil(rgba):
    ch = rgba[:,:,3] if rgba.shape[2]==4 else rgba.sum(2)
    return (ch>0.05).astype(np.uint8)*255

def composite_red(rgba, img_bgr, alpha=0.55):
    f  = img_bgr.astype(np.float32)/255.0
    ma = rgba[:,:,3:4] if rgba.shape[2]==4 else (rgba.sum(2,keepdims=True)>0.02).astype(np.float32)
    red = np.zeros_like(f); red[:,:,2] = 1.0
    return np.clip((f*(1-ma*alpha)+red*ma*alpha)*255, 0, 255).astype(np.uint8)


# ── Width measurement ─────────────────────────────────────────────────────────
def measure_width(mask, yc, hb=30):
    H,W = mask.shape
    y1,y2 = max(0,yc-hb), min(H,yc+hb)
    cols = np.where(mask[y1:y2].any(axis=0))[0]
    if len(cols)<3: return None
    return int(cols.min()), int(cols.max()), float((cols.min()+cols.max())/2.), int(cols.max()-cols.min())


# ── IoU computation ───────────────────────────────────────────────────────────
def compute_iou(mask_h, sil_m, ylo, yhi, H):
    y1,y2 = max(0,int(ylo)), min(H,int(yhi))
    h = mask_h[y1:y2] > 0
    m = sil_m[y1:y2] > 0
    inter = int((h & m).sum())
    union = int((h | m).sum())
    iou   = inter / max(union, 1)
    return iou, inter, union


# ── Fast IoU proxy via column-remap ──────────────────────────────────────────
# When we scale the mesh x by sx around cx:
#   each column x' (output) came from x = cx + (x' - cx) / sx (inverse map)
# This is a valid approximation for frontal body silhouette (roughly flat in camera)
def proxy_iou(sx, sil_base, cx_img, mask_h, ylo, yhi, H, W):
    """Fast IoU proxy: column-remap sil_base by sx around cx_img, no render needed."""
    x = np.arange(W, dtype=np.float32)
    x_src = np.clip(cx_img + (x - cx_img) / sx, 0, W-1).astype(np.int32)
    sil_s = sil_base[:, x_src]
    y1,y2 = max(0,int(ylo)), min(H,int(yhi))
    h = mask_h[y1:y2] > 0
    m = sil_s[y1:y2]  > 0
    inter = int((h & m).sum())
    union = int((h | m).sum())
    return inter / max(union, 1)


# ── Translate band: sil cx → human cx ────────────────────────────────────────
def translate_cx(v_work, verts2d_t1, cam_t, focal, W, H,
                 ylo, yhi, mask_h, sil_current, measure_ys, name):
    py = verts2d_t1[:,1]
    bm = (py>=ylo)&(py<yhi)
    if bm.sum()==0: return v_work, 0.0
    h_cxs, m_cxs = [], []
    for yc in measure_ys:
        hm = measure_width(mask_h, yc)
        mm = measure_width(sil_current, yc)
        if hm and mm: h_cxs.append(hm[2]); m_cxs.append(mm[2])
    if not h_cxs: return v_work, 0.0
    h_cx = float(np.median(h_cxs)); m_cx = float(np.median(m_cxs))
    dx_img = h_cx - m_cx
    if abs(dx_img) < 0.5:
        print(f"  [{name:20s}] cx ok, residual {dx_img:+.1f}px"); return v_work, dx_img
    depth = float(v_work[bm,2].mean()) + cam_t[2]
    dx_w  = dx_img * depth / focal
    print(f"  [{name:20s}] sil_cx {m_cx:.1f}→{h_cx:.1f}  dx={dx_img:+.1f}px  dx_w={dx_w:+.5f}")
    v_work[bm, 0] += dx_w
    return v_work, dx_img


# ── IoU visualization: green/red/yellow ──────────────────────────────────────
def make_iou_vis(mask_h, sil_mesh, H, W, title=""):
    """BGR: green=human only, red=mesh only, yellow=overlap."""
    vis = np.zeros((H,W,3), dtype=np.uint8)
    h = mask_h > 0; m = sil_mesh > 0
    vis[h & ~m] = [0, 255, 0]    # green
    vis[~h & m] = [0, 0, 255]    # red
    vis[h & m]  = [0, 255, 255]  # yellow (BGR)
    cv2.putText(vis, title, (10,44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    return vis


# ── Edge miss (for reference only, not threshold) ────────────────────────────
def edge_miss_pct(sil_p, sil_h, yc, band, H):
    y1,y2 = max(0,yc-band), min(H,yc+band)
    hb=(sil_h[y1:y2]>0); pb=(sil_p[y1:y2]>0)
    nm=int(((hb)&(~pb)).sum()); nh=int(hb.sum())
    return round(nm/max(nh,1)*100, 1)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    import torch
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    print(f"[INFO] torch {torch.__version__}  cuda={torch.cuda.is_available()}")

    model, cfg = load_sam_3d_body(CKPT, device='cuda', mhr_path=MHR_P)

    with open(KP_CACHE) as f: kpd = json.load(f)
    fr0 = kpd['frames'][0]['persons'][0]['keypoints']
    ax = [v['x'] for v in fr0.values() if v['score']>0.3]
    ay = [v['y'] for v in fr0.values() if v['score']>0.3]
    pad_x=(max(ax)-min(ax))*0.15; pad_y=(max(ay)-min(ay))*0.15
    bbox = np.array([[max(0,min(ax)-pad_x),max(0,min(ay)-pad_y),
                      min(720,max(ax)+pad_x),min(1280,max(ay)+pad_y)]],dtype=np.float32)

    img_bgr = cv2.imread(str(ADDR_JPG))
    H, W    = img_bgr.shape[:2]

    est = SAM3DBodyEstimator(model, cfg, human_detector=None,
                             human_segmentor=None, fov_estimator=None)
    print("[INFO] MHR inference...")
    t_inf = time.time()
    outs  = est.process_one_image(img_bgr, bboxes=bbox, use_mask=False, inference_type="body")
    dt_inf = time.time()-t_inf

    o     = outs[0]
    verts = o["pred_vertices"].astype(np.float32)
    cam_t = o["pred_cam_t"].astype(np.float32)
    kp2d  = o["pred_keypoints_2d"].astype(np.float32)
    focal = float(o["focal_length"])
    faces = est.faces
    vram  = torch.cuda.max_memory_allocated()/1e6
    print(f"[INFO] inf {dt_inf:.1f}s  VRAM {vram:.0f}MB  focal={focal:.1f}")

    print("[INFO] rembg mask...")
    mask_h = get_human_mask(img_bgr)
    cv2.imwrite(str(OUTPUT_DIR/"human_mask.jpg"), mask_h)

    print("[INFO] T1 render...")
    rgba_t1 = render_rgba(verts, cam_t, focal, faces, H, W)
    sil_t1  = get_sil(rgba_t1)
    cv2.imwrite(str(OUTPUT_DIR/"sil_t1.jpg"), sil_t1)
    verts2d_t1 = project_verts(verts, cam_t, focal, H, W)

    # y-centers
    nose_y = int(kp2d[I_NOSE][1]); neck_y = int(kp2d[I_NECK][1])
    sho_y  = int((kp2d[I_LSHO][1]+kp2d[I_RSHO][1])/2)
    hip_y  = int((kp2d[I_LHIP][1]+kp2d[I_RHIP][1])/2)
    knee_y = int((kp2d[I_LKNE][1]+kp2d[I_RKNE][1])/2)
    ank_y  = int((kp2d[I_LANK][1]+kp2d[I_RANK][1])/2)
    head_y = int(nose_y-55)
    print(f"[INFO] y: head={head_y} sho={sho_y} hip={hip_y} knee={knee_y} ank={ank_y}")

    # Band boundaries (same as T1.6)
    B_UPP_lo, B_UPP_hi = head_y-30, hip_y-15     # upper body
    B_HIP_lo, B_HIP_hi = hip_y-15,  hip_y+110    # hip
    B_LOW_lo, B_LOW_hi = hip_y+110, ank_y+120    # lower — 达标即止

    # Upper body IoU region (full head-to-hip range)
    UP_IoU_lo = head_y - 30
    UP_IoU_hi = hip_y  + 60

    sho_ys   = list(range(neck_y, sho_y+100, 12))
    hip_ys   = [hip_y-20, hip_y, hip_y+20, hip_y+40]
    lower_ys = [knee_y-20, knee_y, knee_y+30, (knee_y+ank_y)//2, ank_y-30]
    all_up_ys = sho_ys + list(range(sho_y+100, hip_y-15, 20))

    # ── STEP 1: IoU-based optimization for upper body bands ──
    print("\n[INFO] === Step 1: IoU-based optimization ===")

    # Get cx from T1 sil at shoulder (representative)
    m_sho = measure_width(sil_t1, sho_y)
    cx_upp = m_sho[2] if m_sho else float(W/2)
    m_hip  = measure_width(sil_t1, hip_y)
    cx_hip = m_hip[2] if m_hip else float(W/2)
    m_low  = measure_width(sil_t1, knee_y)
    cx_low = m_low[2] if m_low else float(W/2)

    print(f"  T1 sil cx: upper={cx_upp:.1f}  hip={cx_hip:.1f}  lower={cx_low:.1f}")

    # Optimize B_UPP sx
    res_upp = minimize_scalar(
        lambda sx: -proxy_iou(sx, sil_t1, cx_upp, mask_h, B_UPP_lo, B_UPP_hi, H, W),
        bounds=(0.90, 1.70), method='bounded',
        options={'xatol': 1e-4, 'maxiter': 50}
    )
    sx_upp = float(res_upp.x)
    iou_upp_proxy = -float(res_upp.fun)
    print(f"  B_UPP: opt sx={sx_upp:.4f}  proxy_IoU={iou_upp_proxy:.4f}")

    # Optimize B_HIP sx
    res_hip = minimize_scalar(
        lambda sx: -proxy_iou(sx, sil_t1, cx_hip, mask_h, B_HIP_lo, B_HIP_hi, H, W),
        bounds=(0.90, 1.50), method='bounded',
        options={'xatol': 1e-4, 'maxiter': 50}
    )
    sx_hip = float(res_hip.x)
    iou_hip_proxy = -float(res_hip.fun)
    print(f"  B_HIP: opt sx={sx_hip:.4f}  proxy_IoU={iou_hip_proxy:.4f}")

    # Lower: keep T1.6 approach (max sx, containment=1.0 = 达标即止)
    sx_max_low = 0.0; m_low_list = []
    for yc in lower_ys:
        hm = measure_width(mask_h, yc); mm = measure_width(sil_t1, yc)
        if hm and mm and mm[3]>5:
            sx_raw = hm[3]/mm[3]
            if sx_raw > sx_max_low: sx_max_low = sx_raw
            m_low_list.append(mm[2])
    sx_low = float(np.clip(sx_max_low, 0.90, 1.40))
    cx_low_actual = float(np.median(m_low_list)) if m_low_list else cx_low
    print(f"  B_LOW: sx={sx_low:.4f} (max_raw, containment=1.0 — 达标即止)")

    # ── STEP 2: Apply scale transforms ──
    print("\n[INFO] Step 2: Apply scale...")
    v_opt = verts.copy()
    depth_upp = float(v_opt[verts2d_t1[:,1] < B_HIP_lo, 2].mean()) + cam_t[2]
    depth_hip = float(v_opt[(verts2d_t1[:,1]>=B_HIP_lo)&(verts2d_t1[:,1]<B_LOW_lo),2].mean()) + cam_t[2]
    depth_low = float(v_opt[verts2d_t1[:,1]>=B_LOW_lo, 2].mean()) + cam_t[2]

    wcx_m_upp = world_x_from_img(cx_upp,           depth_upp, cam_t, focal, W)
    wcx_m_hip = world_x_from_img(cx_hip,           depth_hip, cam_t, focal, W)
    wcx_m_low = world_x_from_img(cx_low_actual,    depth_low, cam_t, focal, W)

    bm_upp = verts2d_t1[:,1] < B_HIP_lo
    bm_hip = (verts2d_t1[:,1]>=B_HIP_lo)&(verts2d_t1[:,1]<B_LOW_lo)
    bm_low = verts2d_t1[:,1] >= B_LOW_lo

    v_opt[bm_upp, 0] = wcx_m_upp + (v_opt[bm_upp,0] - wcx_m_upp) * sx_upp
    v_opt[bm_hip, 0] = wcx_m_hip + (v_opt[bm_hip,0] - wcx_m_hip) * sx_hip
    v_opt[bm_low, 0] = wcx_m_low + (v_opt[bm_low,0] - wcx_m_low) * sx_low

    print(f"  scaled: UPP={bm_upp.sum()} HIP={bm_hip.sum()} LOW={bm_low.sum()} verts")

    # ── STEP 3: Mid render for cx correction ──
    print("\n[INFO] Step 3: Mid render for cx correction...")
    rgba_mid = render_rgba(v_opt, cam_t, focal, faces, H, W)
    sil_mid  = get_sil(rgba_mid)

    all_up_ys_full = list(range(neck_y, hip_y-15, 15))
    v_opt, dx_upp = translate_cx(v_opt, verts2d_t1, cam_t, focal, W, H,
                                 B_UPP_lo, B_UPP_hi, mask_h, sil_mid,
                                 all_up_ys_full, "B_UPP_translate")
    v_opt, dx_hip = translate_cx(v_opt, verts2d_t1, cam_t, focal, W, H,
                                 B_HIP_lo, B_HIP_hi, mask_h, sil_mid,
                                 hip_ys, "B_HIP_translate")
    v_opt, dx_low = translate_cx(v_opt, verts2d_t1, cam_t, focal, W, H,
                                 B_LOW_lo, B_LOW_hi, mask_h, sil_mid,
                                 lower_ys, "B_LOW_translate")

    # ── STEP 4: Final render ──
    print("\n[INFO] Step 4: Final render T1.7...")
    rgba_t17 = render_rgba(v_opt, cam_t, focal, faces, H, W)
    sil_t17  = get_sil(rgba_t17)
    cv2.imwrite(str(OUTPUT_DIR/"sil_t17.jpg"), sil_t17)

    # ── STEP 5: Residual cx check ──
    h_cxs2, m_cxs2 = [], []
    for yc in sho_ys[:6]:
        hm = measure_width(mask_h, yc); mm = measure_width(sil_t17, yc)
        if hm and mm: h_cxs2.append(hm[2]); m_cxs2.append(mm[2])
    if h_cxs2:
        res_dx = float(np.median(h_cxs2)) - float(np.median(m_cxs2))
        print(f"  shoulder cx residual: {res_dx:+.1f}px")
        if abs(res_dx) > 1.5:
            depth_b = float(v_opt[bm_upp,2].mean()) + cam_t[2]
            v_opt[bm_upp, 0] += res_dx * depth_b / focal
            rgba_t17 = render_rgba(v_opt, cam_t, focal, faces, H, W)
            sil_t17  = get_sil(rgba_t17)
            cv2.imwrite(str(OUTPUT_DIR/"sil_t17.jpg"), sil_t17)
            print(f"  applied residual {res_dx:+.1f}px, re-rendered")

    # ── STEP 6: Compute actual IoU ──
    print("\n[INFO] Step 6: IoU computation...")

    iou_upp_actual, inter_upp, union_upp = compute_iou(mask_h, sil_t17, UP_IoU_lo, UP_IoU_hi, H)
    iou_sho_actual, _,_ = compute_iou(mask_h, sil_t17, sho_y-60, sho_y+60, H)
    iou_hip_actual, _,_ = compute_iou(mask_h, sil_t17, hip_y-60, hip_y+60, H)
    iou_low_actual, _,_ = compute_iou(mask_h, sil_t17, knee_y-60, ank_y+60, H)

    # T1.6 IoU for comparison
    sil_t16 = cv2.imread(str(OUTPUT_DIR/"sil_t16.jpg"), cv2.IMREAD_GRAYSCALE)
    iou_t16 = 0.0
    if sil_t16 is not None:
        iou_t16, _, _ = compute_iou(mask_h, sil_t16, UP_IoU_lo, UP_IoU_hi, H)

    print(f"\n=== IoU REPORT ===")
    print(f"  [上半身 head→hip+60, y={UP_IoU_lo}~{UP_IoU_hi}]")
    print(f"  T1.6 IoU:  {iou_t16:.4f}")
    print(f"  T1.7 IoU:  {iou_upp_actual:.4f}  ← 主指标")
    print(f"  改善:      {iou_upp_actual-iou_t16:+.4f}")
    print(f"  inter={inter_upp}px  union={union_upp}px")
    print(f"\n  [局部 IoU]")
    print(f"  shoulder zone: {iou_sho_actual:.4f}")
    print(f"  hip zone:      {iou_hip_actual:.4f}")
    print(f"  lower (达标即止): {iou_low_actual:.4f}")

    # edge miss for reference
    print(f"\n  [Edge miss 参考 — 非通过条件]")
    for yc,label in [(sho_y,'shoulder'),(hip_y,'hip')]:
        miss = edge_miss_pct(sil_t17, mask_h, yc, 60, H)
        print(f"  {label}: {miss}%")

    # ── STEP 7: Composites + visualization ──
    print("\n[INFO] Step 7: Composites...")
    ov_t17 = composite_red(rgba_t17, img_bgr, alpha=0.55)
    lh_pt  = (int(kp2d[I_LHIP][0]),int(kp2d[I_LHIP][1]))
    rh_pt  = (int(kp2d[I_RHIP][0]),int(kp2d[I_RHIP][1]))
    y_pel  = (lh_pt[1]+rh_pt[1])//2
    cv2.line(ov_t17, (0,y_pel),(W-1,y_pel),(255,255,255),2)
    for pt in [lh_pt,rh_pt]:
        cv2.circle(ov_t17,pt,7,(255,255,255),-1); cv2.circle(ov_t17,pt,3,(0,0,200),-1)

    p_ov = OUTPUT_DIR/"mhr_overlay_t17.jpg"
    cv2.imwrite(str(p_ov), ov_t17)
    cv2.imwrite(str(OUTPUT_DIR/"mhr_overlay.jpg"), ov_t17)

    # ── Silhouette compare: left=T1.6 right=T1.7 (yellow=overlap) ──
    fn = cv2.FONT_HERSHEY_SIMPLEX
    vis_t16 = make_iou_vis(mask_h, sil_t16, H, W,
                           f"T1.6  upper-IoU={iou_t16:.3f}") if sil_t16 is not None else \
              np.zeros((H,W,3),dtype=np.uint8)
    vis_t17 = make_iou_vis(mask_h, sil_t17, H, W,
                           f"T1.7  upper-IoU={iou_upp_actual:.3f}")
    sil_cmp = np.concatenate([vis_t16, vis_t17], axis=1)
    # legend
    cv2.putText(sil_cmp, "green=human  red=mesh  yellow=overlap", (10,80), fn,0.8,(200,200,200),2)
    p_sil = OUTPUT_DIR/"silhouette_compare.jpg"
    cv2.imwrite(str(p_sil), sil_cmp)

    # 3-panel side_by_side
    rgba_t1_img = composite_red(rgba_t1, img_bgr, alpha=0.55)
    sbs = np.concatenate([img_bgr, rgba_t1_img, ov_t17], axis=1)
    for txt,x in [("Original",0),("T1 (baseline)",W),(f"T1.7 (IoU={iou_upp_actual:.3f})",2*W)]:
        cv2.putText(sbs, txt, (x+12,44), fn, 1.1,(255,255,255),2)
    p_sbs = OUTPUT_DIR/"side_by_side_t17.jpg"
    cv2.imwrite(str(p_sbs), sbs)
    cv2.imwrite(str(OUTPUT_DIR/"side_by_side.jpg"), np.concatenate([img_bgr,ov_t17],axis=1))

    # ── Joint fit (姿态 unchanged) ──
    print("\n=== 逐点关节误差 ===")
    jf = {}
    for jn,ji in MHR_NAMED.items():
        mx,my = float(kp2d[ji][0]),float(kp2d[ji][1])
        if jn in RTM_GT:
            rx,ry=RTM_GT[jn]; d=float(np.sqrt((mx-rx)**2+(my-ry)**2))
            g='好' if d<=10 else ('可' if d<=25 else '差')
        else:
            rx=ry=d=None; g='--'
        jf[jn]={'mhr_xy':[mx,my],'rtm_xy':[rx,ry],'dist_px':d,'grade':g}
        print(f"  {jn:22s}: d={f'{d:.1f}px' if d else 'N/A':>8}  {g}")

    # ── Log ──
    tt    = time.time()-t0
    vram2 = torch.cuda.max_memory_allocated()/1e6
    log   = {
        "version":"T1.7","phase":"address","frame":0,"clip":"fo-ok-1",
        "canvas_hw":[H,W],"inference_s":round(dt_inf,2),
        "total_s":round(tt,2),"peak_vram_mb":round(vram2,0),
        "focal_length":focal,"cam_t":cam_t.tolist(),
        "optimization":{"method":"scipy.minimize_scalar.bounded","proxy":"column_remap"},
        "opt_sx":{"B_UPP":round(sx_upp,4),"B_HIP":round(sx_hip,4),"B_LOW":round(sx_low,4)},
        "proxy_iou":{"B_UPP":round(iou_upp_proxy,4),"B_HIP":round(iou_hip_proxy,4)},
        "iou_t16_upper":round(iou_t16,4),
        "iou_t17_upper":round(iou_upp_actual,4),
        "iou_shoulder":round(iou_sho_actual,4),
        "iou_hip":round(iou_hip_actual,4),
        "iou_lower":round(iou_low_actual,4),
        "joint_fit":jf,
        "topology":"MHR_native","smpl_dependency":None,"yolo_used":False,
        "license":"SAM_License_PRODUCT_CANDIDATE_CUSTOM_LICENSE",
        "tiered_standard":{"lower_body_policy":"达标即止 — 不再优化"},
    }
    lp = OUTPUT_DIR/"run_log_t17.json"
    with open(lp,'w') as f: json.dump(log,f,indent=2,default=str)

    print(f"\n[OUT] {p_ov}")
    print(f"[OUT] {p_sbs}")
    print(f"[OUT] {p_sil}")
    print(f"[OUT] {lp}")
    print(f"\n{'='*50}")
    print(f"  上半身 IoU: T1.6={iou_t16:.4f}  T1.7={iou_upp_actual:.4f}  改善={iou_upp_actual-iou_t16:+.4f}")
    print(f"  Jason 裁决放行线: __________")
    print(f"{'='*50}")
    print(f"[DONE] Total {tt:.1f}s  VRAM {vram2:.0f}MB")


if __name__ == "__main__":
    main()
