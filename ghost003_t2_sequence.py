"""
ghost003_t2_sequence.py  —  GHOST-003 T2 整段序列贴合
策略: 对 fo-ok-1 NF=112 逐帧跑 IoU-based shape fitting（沿用 T1.7 算法）
  - 每帧目标: 最大化上半身 IoU（head→hip+60 区域）
  - 放行线: IoU ≥ 0.92（Jason 裁决 2026-07-07）
  - 下肢: 达标即止，不牺牲上半身 IoU
  - cx 纠偏: 沿用 T1.7 逻辑（B_UPP/B_HIP/B_LOW 各自 cx 对齐）

停关卡产出:
  - run_log_t2.json           逐帧 IoU 数据
  - iou_distribution.jpg      衰减曲线（按帧号）
  - worst3_compare_frXXX.jpg  最差3帧 silhouette_compare
  - keyframe_overlay/address_fr000.jpg / top_fr097.jpg / impact_frXXX.jpg
  - REPORT_T2.txt             mean / min+帧号 / P5 分位 + 低于0.92帧列表

范围: 整段·真实姿态·不改姿态·不做动作修正·不碰球杆
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

OUTPUT_DIR  = pathlib.Path(SWINGCUE) / "output" / "ghost003_t2"
KF_DIR      = OUTPUT_DIR / "keyframe_overlay"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KF_DIR.mkdir(parents=True, exist_ok=True)

KP_CACHE    = pathlib.Path(SWINGCUE) / "engine/kp_cache/batch2/fo-ok-1.json"
VIDEO_PATH  = pathlib.Path(SWINGCUE) / "input/fo-ok-1.mp4"
CKPT  = "/home/jason/.cache/sam3d/sam-3d-body-dinov3/model.ckpt"
MHR_P = "/home/jason/.cache/sam3d/sam-3d-body-dinov3/assets/mhr_model.pt"

# Ground-truth anchor frames
FR_ADDRESS = 0
FR_TOP     = 97   # Jason GT

# Impact heuristic: wrist y returns near address height after top
# fr88: left_wrist_y=702 ≈ fr0 wrist_y=706 → impact estimate
FR_IMPACT  = 88

PASS_THRESHOLD = 0.92

# MHR70 indices
I_NOSE=0; I_LSHO=5; I_RSHO=6; I_LELB=7; I_RELB=8
I_LHIP=9; I_RHIP=10; I_LKNE=11; I_RKNE=12; I_LANK=13; I_RANK=14
I_RWRI=41; I_LWRI=62; I_NECK=69


# ── Projection ─────────────────────────────────────────────────────────────
def project_verts(verts, cam_t, focal, H, W):
    vx, vy, vz = verts[:,0], verts[:,1], verts[:,2]
    d  = np.where(np.abs(vz+cam_t[2])<1e-6, 1e-6, vz+cam_t[2])
    px = focal*(vx-cam_t[0])/d + W/2.0
    py = focal*(vy+cam_t[1])/d + H/2.0
    return np.stack([px, py], axis=1)

def world_x_from_img(img_x, depth, cam_t, focal, W):
    return (img_x - W/2.0) * depth / focal + cam_t[0]


# ── Human mask ─────────────────────────────────────────────────────────────
def get_human_mask(img_bgr):
    import rembg
    out  = rembg.remove(img_bgr)
    mask = (out[:,:,3] > 40).astype(np.uint8) * 255
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    return mask


# ── Render ─────────────────────────────────────────────────────────────────
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


# ── Width measurement ───────────────────────────────────────────────────────
def measure_width(mask, yc, hb=30):
    H, W = mask.shape
    y1, y2 = max(0, yc-hb), min(H, yc+hb)
    cols = np.where(mask[y1:y2].any(axis=0))[0]
    if len(cols) < 3:
        return None
    return int(cols.min()), int(cols.max()), float((cols.min()+cols.max())/2.), int(cols.max()-cols.min())


# ── IoU computation ─────────────────────────────────────────────────────────
def compute_iou(mask_h, sil_m, ylo, yhi, H):
    y1, y2 = max(0, int(ylo)), min(H, int(yhi))
    h = mask_h[y1:y2] > 0
    m = sil_m[y1:y2]  > 0
    inter = int((h & m).sum())
    union = int((h | m).sum())
    return inter / max(union, 1), inter, union


# ── Proxy IoU (column-remap, no render) ─────────────────────────────────────
def proxy_iou(sx, sil_base, cx_img, mask_h, ylo, yhi, H, W):
    x     = np.arange(W, dtype=np.float32)
    x_src = np.clip(cx_img + (x - cx_img) / sx, 0, W-1).astype(np.int32)
    sil_s = sil_base[:, x_src]
    y1, y2 = max(0, int(ylo)), min(H, int(yhi))
    h = mask_h[y1:y2] > 0
    m = sil_s[y1:y2]  > 0
    inter = int((h & m).sum())
    union = int((h | m).sum())
    return inter / max(union, 1)


# ── Translate band cx ───────────────────────────────────────────────────────
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
    h_cx = float(np.median(h_cxs))
    m_cx = float(np.median(m_cxs))
    dx_img = h_cx - m_cx
    if abs(dx_img) < 0.5:
        return v_work, dx_img
    depth  = float(v_work[band_mask_2d, 2].mean()) + cam_t[2]
    dx_w   = dx_img * depth / focal
    if not silent:
        print(f"  [{name:20s}] cx {m_cx:.1f}→{h_cx:.1f} dx={dx_img:+.1f}px")
    v_work[band_mask_2d, 0] += dx_w
    return v_work, dx_img


# ── IoU visualisation (green/red/yellow) ────────────────────────────────────
def make_iou_vis(mask_h, sil_mesh, H, W, title=""):
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    h = mask_h > 0; m = sil_mesh > 0
    vis[h & ~m] = [0, 255, 0]
    vis[~h & m] = [0, 0, 255]
    vis[h & m]  = [0, 255, 255]
    cv2.putText(vis, title, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return vis


# ── Fit one frame (T1.7 algorithm) ──────────────────────────────────────────
def fit_frame(img_bgr, est, faces, kp_rtm):
    H, W = img_bgr.shape[:2]

    # Build bbox from RTMPose kp
    ax = [v['x'] for v in kp_rtm.values() if v['score'] > 0.3]
    ay = [v['y'] for v in kp_rtm.values() if v['score'] > 0.3]
    if len(ax) < 4:
        return None
    pad_x = (max(ax)-min(ax)) * 0.15
    pad_y = (max(ay)-min(ay)) * 0.15
    bbox  = np.array([[max(0, min(ax)-pad_x), max(0, min(ay)-pad_y),
                       min(W, max(ax)+pad_x), min(H, max(ay)+pad_y)]],
                     dtype=np.float32)

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

    # Band boundaries
    B_UPP_lo = head_y - 30;  B_UPP_hi = hip_y - 15
    B_HIP_lo = hip_y - 15;   B_HIP_hi = hip_y + 110
    B_LOW_lo = hip_y + 110;  B_LOW_hi = ank_y + 120
    UP_IoU_lo = head_y - 30; UP_IoU_hi = hip_y + 60

    # cx from T1 sil
    m_sho = measure_width(sil_t1, sho_y)
    cx_upp = m_sho[2] if m_sho else float(W/2)
    m_hip = measure_width(sil_t1, hip_y)
    cx_hip = m_hip[2] if m_hip else float(W/2)
    m_kne = measure_width(sil_t1, knee_y)
    cx_low = m_kne[2] if m_kne else float(W/2)

    # Optimize B_UPP sx
    res_upp = minimize_scalar(
        lambda sx: -proxy_iou(sx, sil_t1, cx_upp, mask_h, B_UPP_lo, B_UPP_hi, H, W),
        bounds=(0.90, 1.70), method='bounded', options={'xatol': 1e-4, 'maxiter': 50}
    )
    sx_upp = float(res_upp.x)

    # Optimize B_HIP sx
    res_hip = minimize_scalar(
        lambda sx: -proxy_iou(sx, sil_t1, cx_hip, mask_h, B_HIP_lo, B_HIP_hi, H, W),
        bounds=(0.90, 1.50), method='bounded', options={'xatol': 1e-4, 'maxiter': 50}
    )
    sx_hip = float(res_hip.x)

    # B_LOW: max containment (达标即止)
    sx_max_low = 0.0; m_low_list = []
    lower_ys = [knee_y-20, knee_y, knee_y+30, (knee_y+ank_y)//2, ank_y-30]
    for yc in lower_ys:
        hm = measure_width(mask_h, yc); mm = measure_width(sil_t1, yc)
        if hm and mm and mm[3] > 5:
            sx_raw = hm[3] / mm[3]
            if sx_raw > sx_max_low: sx_max_low = sx_raw
            m_low_list.append(mm[2])
    sx_low = float(np.clip(sx_max_low, 0.90, 1.40))
    cx_low_actual = float(np.median(m_low_list)) if m_low_list else cx_low

    # Apply scale
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

    v_opt[bm_upp, 0] = wcx_m_upp + (v_opt[bm_upp,0] - wcx_m_upp) * sx_upp
    v_opt[bm_hip, 0] = wcx_m_hip + (v_opt[bm_hip,0] - wcx_m_hip) * sx_hip
    v_opt[bm_low, 0] = wcx_m_low + (v_opt[bm_low,0] - wcx_m_low) * sx_low

    # Mid render + cx correction
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

    # Residual shoulder cx check
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

    # Compute actual IoU
    iou_upp, inter, union = compute_iou(mask_h, sil_final, UP_IoU_lo, UP_IoU_hi, H)

    return {
        "iou_upper": round(float(iou_upp), 4),
        "pass": bool(iou_upp >= PASS_THRESHOLD),
        "rgba_final": rgba_final,
        "sil_final": sil_final,
        "mask_h": mask_h,
        "sx_upp": round(sx_upp, 4),
        "sx_hip": round(sx_hip, 4),
        "sx_low": round(sx_low, 4),
        "UP_IoU_lo": UP_IoU_lo,
        "UP_IoU_hi": UP_IoU_hi,
        "head_y": head_y,
        "hip_y": hip_y,
        "img_bgr": img_bgr,
        "kp2d": kp2d,
        "cam_t": cam_t.tolist(),
        "focal": focal,
    }


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    import torch
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

    print(f"[INFO] torch {torch.__version__}  cuda={torch.cuda.is_available()}")

    # Load model once
    print("[INFO] Loading model...")
    model, cfg = load_sam_3d_body(CKPT, device='cuda', mhr_path=MHR_P)
    est = SAM3DBodyEstimator(model, cfg, human_detector=None,
                             human_segmentor=None, fov_estimator=None)
    faces = est.faces

    # Load kp cache
    with open(KP_CACHE) as f:
        kpd = json.load(f)
    frames = kpd['frames']
    NF = len(frames)
    print(f"[INFO] Clip fo-ok-1  NF={NF}")

    # Open video
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO_PATH}")

    results = []     # list of per-frame dicts
    frame_results = {}  # fr_idx -> fit result (kept for keyframe/worst visualisation)
    KEYFRAMES = {FR_ADDRESS: "address", FR_TOP: "top", FR_IMPACT: "impact"}

    for fr_idx in range(NF):
        ret, img_bgr = cap.read()
        if not ret:
            print(f"  [WARN] Cannot read fr{fr_idx:03d}")
            results.append({"frame": fr_idx, "iou_upper": None, "pass": False, "skipped": True})
            continue

        t_fr = time.time()
        kp_rtm = frames[fr_idx]['persons'][0]['keypoints']
        print(f"  [fr{fr_idx:03d}/{NF-1}] fitting...", end='', flush=True)

        try:
            r = fit_frame(img_bgr, est, faces, kp_rtm)
        except Exception as e:
            print(f" ERROR: {e}")
            results.append({"frame": fr_idx, "iou_upper": None, "pass": False, "error": str(e)})
            continue

        if r is None:
            print(f" SKIP (no output)")
            results.append({"frame": fr_idx, "iou_upper": None, "pass": False, "skipped": True})
            continue

        dt = time.time() - t_fr
        iou = r["iou_upper"]
        flag = "✓" if r["pass"] else "✗ <0.92"
        print(f" IoU={iou:.4f} {flag}  ({dt:.1f}s)")

        row = {
            "frame": fr_idx,
            "iou_upper": iou,
            "pass": r["pass"],
            "sx_upp": r["sx_upp"],
            "sx_hip": r["sx_hip"],
            "sx_low": r["sx_low"],
        }
        results.append(row)

        # Save keyframes
        if fr_idx in KEYFRAMES:
            phase = KEYFRAMES[fr_idx]
            ov = composite_red(r["rgba_final"], img_bgr, alpha=0.55)
            # pelvis line
            kp2d = r["kp2d"]
            lh = (int(kp2d[I_LHIP][0]), int(kp2d[I_LHIP][1]))
            rh = (int(kp2d[I_RHIP][0]), int(kp2d[I_RHIP][1]))
            y_pel = (lh[1]+rh[1])//2
            H_img, W_img = img_bgr.shape[:2]
            cv2.line(ov, (0, y_pel), (W_img-1, y_pel), (255,255,255), 2)
            cv2.circle(ov, lh, 7, (255,255,255), -1); cv2.circle(ov, lh, 3, (0,0,200), -1)
            cv2.circle(ov, rh, 7, (255,255,255), -1); cv2.circle(ov, rh, 3, (0,0,200), -1)
            fn_label = f"{phase}_fr{fr_idx:03d}_IoU{iou:.3f}"
            cv2.putText(ov, fn_label, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            kf_path = KF_DIR / f"{phase}_fr{fr_idx:03d}.jpg"
            cv2.imwrite(str(kf_path), ov)
            print(f"  [KEY] saved {kf_path.name}")

        # Keep result for worst-frame visualisation
        frame_results[fr_idx] = r

    cap.release()

    # ── Compute distribution ──────────────────────────────────────────────
    ious = [r["iou_upper"] for r in results if r["iou_upper"] is not None]
    iou_arr = np.array(ious)
    mean_iou = float(np.mean(iou_arr))
    min_iou  = float(np.min(iou_arr))
    p5_iou   = float(np.percentile(iou_arr, 5))
    min_fr   = results[[r["iou_upper"] for r in results].index(min_iou)]["frame"]
    fail_frames = [(r["frame"], r["iou_upper"]) for r in results
                   if r["iou_upper"] is not None and not r["pass"]]

    print(f"\n{'='*55}")
    print(f"  T2 IoU 分布 (NF={NF}  有效帧={len(ious)})")
    print(f"  mean = {mean_iou:.4f}")
    print(f"  min  = {min_iou:.4f}  (fr{min_fr:03d})")
    print(f"  P5   = {p5_iou:.4f}")
    print(f"  低于放行线 (< {PASS_THRESHOLD}) 帧数: {len(fail_frames)}")
    for fno, v in fail_frames:
        print(f"    fr{fno:03d}  IoU={v:.4f}")
    print(f"{'='*55}")

    # ── Worst 3 silhouette compare ────────────────────────────────────────
    valid_sorted = sorted(
        [(r["frame"], r["iou_upper"]) for r in results if r["iou_upper"] is not None
         and r["frame"] in frame_results],
        key=lambda x: x[1]
    )
    worst3 = valid_sorted[:3]
    print(f"\n[INFO] Worst 3 frames: {[f for f,_ in worst3]}")
    for rank, (fno, fiou) in enumerate(worst3, 1):
        r = frame_results[fno]
        H_img, W_img = r["img_bgr"].shape[:2]
        vis = make_iou_vis(r["mask_h"], r["sil_final"], H_img, W_img,
                           f"fr{fno:03d} IoU={fiou:.3f}")
        cv2.putText(vis, "green=human  red=mesh  yellow=overlap",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        p_vis = OUTPUT_DIR / f"worst{rank}_sil_fr{fno:03d}.jpg"
        cv2.imwrite(str(p_vis), vis)
        print(f"  [WORST{rank}] fr{fno:03d} IoU={fiou:.4f} → {p_vis.name}")

    # ── IoU distribution curve ────────────────────────────────────────────
    H_c, W_c = 400, max(800, NF * 6)
    curve = np.zeros((H_c, W_c, 3), dtype=np.uint8)
    curve[:] = (30, 30, 30)

    # Grid lines
    for iou_grid in [0.80, 0.85, 0.90, 0.92, 0.95, 1.0]:
        yg = int((1.0 - iou_grid) / 0.25 * (H_c - 60)) + 30
        color = (0, 200, 100) if abs(iou_grid - PASS_THRESHOLD) < 0.005 else (60, 60, 60)
        cv2.line(curve, (0, yg), (W_c-1, yg), color, 1)
        cv2.putText(curve, f"{iou_grid:.2f}", (4, yg-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

    # Threshold label
    y_thresh = int((1.0 - PASS_THRESHOLD) / 0.25 * (H_c - 60)) + 30
    cv2.putText(curve, f"pass >= {PASS_THRESHOLD}", (W_c-160, y_thresh-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,100), 1)

    # Plot per-frame IoU
    for r in results:
        if r["iou_upper"] is None: continue
        fno = r["frame"]
        iou_v = r["iou_upper"]
        x = int(fno / max(NF-1, 1) * (W_c - 20)) + 10
        y = int((1.0 - min(1.0, max(0.75, iou_v))) / 0.25 * (H_c - 60)) + 30
        color = (0, 200, 255) if r["pass"] else (0, 80, 255)
        cv2.circle(curve, (x, y), 3, color, -1)
        # Mark keyframes
        if fno in KEYFRAMES:
            cv2.circle(curve, (x, y), 7, (255, 255, 0), 2)
            cv2.putText(curve, KEYFRAMES[fno], (x-10, y-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)

    cv2.putText(curve, f"fo-ok-1 T2  mean={mean_iou:.3f} min={min_iou:.3f}(fr{min_fr:03d}) P5={p5_iou:.3f}",
                (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    p_curve = OUTPUT_DIR / "iou_distribution.jpg"
    cv2.imwrite(str(p_curve), curve)
    print(f"\n[OUT] {p_curve}")

    # ── JSON log ─────────────────────────────────────────────────────────
    vram = torch.cuda.max_memory_allocated() / 1e6
    log = {
        "version": "T2", "clip": "fo-ok-1", "NF": NF,
        "pass_threshold": PASS_THRESHOLD,
        "distribution": {
            "mean": round(mean_iou, 4),
            "min":  round(min_iou, 4), "min_frame": min_fr,
            "p5":   round(p5_iou, 4),
            "valid_count": len(ious),
        },
        "fail_frames": [{"frame": f, "iou_upper": round(v, 4)} for f, v in fail_frames],
        "worst3_frames": [{"frame": f, "iou_upper": round(v, 4)} for f, v in worst3],
        "per_frame": results,
        "peak_vram_mb": round(vram, 0),
        "total_s": round(time.time() - t_start, 1),
        "topology": "MHR_native", "license": "SAM_License_PRODUCT_CANDIDATE_CUSTOM_LICENSE",
    }
    lp = OUTPUT_DIR / "run_log_t2.json"
    with open(lp, 'w') as f:
        json.dump(log, f, indent=2, default=str)
    print(f"[OUT] {lp}")

    # ── Text report ───────────────────────────────────────────────────────
    report_lines = [
        "GHOST-003 T2 整段序列 IoU 分布报告",
        f"Clip: fo-ok-1  NF={NF}  有效帧={len(ious)}",
        f"放行线: 上半身 IoU >= {PASS_THRESHOLD}  (Jason 裁决 2026-07-07)",
        "",
        f"mean IoU : {mean_iou:.4f}",
        f"min  IoU : {min_iou:.4f}  (fr{min_fr:03d}) ← 最差帧",
        f"P5  IoU  : {p5_iou:.4f}",
        f"低于放行线: {len(fail_frames)} 帧",
    ]
    if fail_frames:
        report_lines.append("")
        report_lines.append("低于 0.92 帧列表:")
        for fno, v in fail_frames:
            report_lines.append(f"  fr{fno:03d}  IoU={v:.4f}")
    report_lines += [
        "",
        f"最差3帧: {[f'fr{f}(IoU={v:.3f})' for f,v in worst3]}",
        "",
        f"关键帧 overlay 已保存:",
        f"  address fr{FR_ADDRESS:03d}  →  keyframe_overlay/address_fr{FR_ADDRESS:03d}.jpg",
        f"  top     fr{FR_TOP:03d}  →  keyframe_overlay/top_fr{FR_TOP:03d}.jpg",
        f"  impact  fr{FR_IMPACT:03d}  →  keyframe_overlay/impact_fr{FR_IMPACT:03d}.jpg",
        "",
        f"peak VRAM: {vram:.0f}MB  total: {time.time()-t_start:.0f}s",
    ]
    rp = OUTPUT_DIR / "REPORT_T2.txt"
    rp.write_text("\n".join(report_lines), encoding='utf-8')
    print(f"[OUT] {rp}")

    print(f"\n{'='*55}")
    print(f"  T2 DONE  mean={mean_iou:.4f}  min={min_iou:.4f}(fr{min_fr:03d})  P5={p5_iou:.4f}")
    print(f"  低于放行线: {len(fail_frames)}/{NF} 帧")
    print(f"  peak VRAM: {vram:.0f}MB  total: {time.time()-t_start:.1f}s")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
