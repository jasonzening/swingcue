"""
ghost003_t16_shoulder_fit.py  —  GHOST-003 T1.6 v4 上半身精调层
分级铁律 (GHOST 线永久铁律):
  上半身(肩/躯干/髋): edge miss ≤3%
  下肢(膝/踝/脚):     达标即止, 不再优化

修复历史:
  v1: z-scaling → cx偏移 18px
  v2: 用 vertex mean 作 mesh cx → 差 66px, 反向偏移 miss↑58%
  v3: sil-based cx + no z-scaling, 但 B0 band污染导致 shoulder cx 实际渲染后偏移 +16px
  v4: 迭代纠偏: 初始transform → render(fast) → 量actual sil cx vs human cx →
      apply 纯平移修正到 shoulder+torso+hip bands → final render

注意: pyrender z-buffering 导致 3D vertex 平移后 rendered silhouette cx
      与预期 projected cx 不同, 因此必须迭代量测而非纯解析

范围: 单帧·真实姿态·address相位·不改姿态·不做整段·不碰球杆
授权: SAM License (PRODUCT_CANDIDATE_CUSTOM_LICENSE)
"""

import os, sys, time, json, pathlib
import numpy as np
import cv2

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
    'nose':           (294.5, 517.4),
    'left_shoulder':  (366.7, 548.4), 'right_shoulder': (256.3, 565.9),
    'left_elbow':     (347.1, 638.2), 'right_elbow':    (279.0, 647.5),
    'left_wrist':     (323.4, 706.3), 'right_wrist':    (299.6, 721.8),
    'left_hip':       (357.4, 679.5), 'right_hip':      (292.4, 684.6),
    'left_knee':      (371.9, 782.7), 'right_knee':     (267.6, 787.9),
    'left_ankle':     (391.5, 910.7), 'right_ankle':    (262.5, 912.8),
}
MHR_NAMED = {
    'nose': I_NOSE, 'neck': I_NECK,
    'left_shoulder': I_LSHO, 'right_shoulder': I_RSHO,
    'left_elbow': I_LELB,    'right_elbow':    I_RELB,
    'left_wrist': I_LWRI,    'right_wrist':    I_RWRI,
    'left_hip':   I_LHIP,    'right_hip':      I_RHIP,
    'left_knee':  I_LKNE,    'right_knee':     I_RKNE,
    'left_ankle': I_LANK,    'right_ankle':    I_RANK,
}


# ── Projection ───────────────────────────────────────────────────────────────
def project_verts(verts, cam_t, focal, H, W):
    vx, vy, vz = verts[:,0], verts[:,1], verts[:,2]
    d  = np.where(np.abs(vz + cam_t[2]) < 1e-6, 1e-6, vz + cam_t[2])
    px = focal * (vx - cam_t[0]) / d + W / 2.0
    py = focal * (vy + cam_t[1]) / d + H / 2.0
    return np.stack([px, py], axis=1)


def world_x_from_img(img_x, depth, cam_t, focal, W):
    return (img_x - W/2.0) * depth / focal + cam_t[0]


# ── Human mask ───────────────────────────────────────────────────────────────
def get_human_mask(img_bgr):
    import rembg
    out   = rembg.remove(img_bgr)
    alpha = out[:, :, 3]
    mask  = (alpha > 40).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    return mask


# ── Render ───────────────────────────────────────────────────────────────────
def render_rgba(verts, cam_t, focal, faces, H, W):
    from sam_3d_body.visualization.renderer import Renderer
    dummy = np.zeros((H, W, 3), dtype=np.uint8)
    r = Renderer(focal_length=focal, faces=faces)
    return r(verts, cam_t, dummy,
             mesh_base_color=(1.0, 0.0, 0.0),
             scene_bg_color=(0, 0, 0), return_rgba=True)


def get_sil(rgba):
    ch = rgba[:, :, 3] if rgba.shape[2] == 4 else rgba.sum(2)
    return (ch > 0.05).astype(np.uint8) * 255


# ── Pure-red composite ────────────────────────────────────────────────────────
def composite_red(rgba, img_bgr, alpha=0.55):
    f  = img_bgr.astype(np.float32) / 255.0
    ma = rgba[:, :, 3:4] if rgba.shape[2] == 4 else \
         (rgba.sum(2, keepdims=True) > 0.02).astype(np.float32)
    red = np.zeros_like(f); red[:, :, 2] = 1.0
    blended = f * (1.0 - ma * alpha) + red * ma * alpha
    return np.clip(blended * 255.0, 0, 255).astype(np.uint8)


# ── Width measurement ─────────────────────────────────────────────────────────
def measure_width(mask, yc, hb=30):
    H, W = mask.shape
    y1, y2 = max(0, yc-hb), min(H, yc+hb)
    cols = np.where(mask[y1:y2].any(axis=0))[0]
    if len(cols) < 3:
        return None
    return (int(cols.min()), int(cols.max()),
            float((cols.min()+cols.max())/2.0),
            int(cols.max()-cols.min()))


# ── Edge miss ────────────────────────────────────────────────────────────────
def edge_scores(sil_p, sil_h, regions_y, band=60):
    from scipy.ndimage import distance_transform_edt
    dm  = distance_transform_edt(sil_p == 0)
    H   = sil_p.shape[0]
    out = {}
    for rn, yc in regions_y.items():
        y1 = max(0, yc-band); y2 = min(H, yc+band)
        hb = sil_h[y1:y2]; pb = sil_p[y1:y2]; db = dm[y1:y2]
        miss = (hb > 0) & (pb == 0)
        nh   = int((hb > 0).sum()); nm = int(miss.sum())
        md   = float(db[miss].mean()) if nm > 0 else 0.0
        out[rn] = {'miss_px': nm,
                   'miss_ratio': round(nm/max(nh,1)*100, 1),
                   'mean_dist_px': round(md, 1),
                   'total_human_px': nh}
    return out


# ── Scale band (from sil_t1 measurements, no z-scale) ────────────────────────
def scale_band(v_work, verts2d_t1, cam_t, focal, W,
               ylo, yhi, mask_h, sil_ref, measure_ys, containment, name):
    """Scale x for band based on sil_ref measurements. NO translation here."""
    py  = verts2d_t1[:, 1]
    bm  = (py >= ylo) & (py < yhi)
    cnt = bm.sum()
    if cnt == 0:
        print(f"  [{name}] 0 verts — skip"); return v_work

    depth = float(v_work[bm, 2].mean()) + cam_t[2]
    sx_list, m_cxs = [], []
    for yc in measure_ys:
        hm = measure_width(mask_h, yc)
        mm = measure_width(sil_ref, yc)
        if hm is None or mm is None or mm[3] < 5: continue
        sx_list.append(hm[3] / mm[3])
        m_cxs.append(mm[2])

    if not sx_list:
        print(f"  [{name}] no data — skip"); return v_work

    sx     = float(np.clip(max(sx_list) * containment, 0.90, 1.80))
    wcx_m  = world_x_from_img(float(np.median(m_cxs)), depth, cam_t, focal, W)
    print(f"  [{name:20s}] verts={cnt:5d}  sx={sx:.3f}  scale_center_imgx={float(np.median(m_cxs)):.1f}")

    v_work[bm, 0] = wcx_m + (v_work[bm, 0] - wcx_m) * sx   # scale around mesh cx, NO z
    return v_work


# ── Translate band to align rendered sil cx → human cx ───────────────────────
def translate_band_to_human_cx(v_work, verts2d_t1, cam_t, focal, W, H,
                               ylo, yhi, mask_h, sil_current, measure_ys, name):
    """
    After scale, rendered sil cx may differ from human cx.
    Translate band verts so sil cx matches human cx.
    Returns v_work and correction_dx (image pixels).
    """
    py  = verts2d_t1[:, 1]
    bm  = (py >= ylo) & (py < yhi)
    cnt = bm.sum()
    if cnt == 0: return v_work, 0.0

    h_cxs, m_cxs = [], []
    for yc in measure_ys:
        hm = measure_width(mask_h, yc)
        mm = measure_width(sil_current, yc)
        if hm and mm:
            h_cxs.append(hm[2]); m_cxs.append(mm[2])

    if not h_cxs: return v_work, 0.0

    h_cx_img = float(np.median(h_cxs))
    m_cx_img = float(np.median(m_cxs))
    dx_img   = h_cx_img - m_cx_img   # positive = shift right needed

    if abs(dx_img) < 1.0:
        print(f"  [{name:20s}] cx_diff={dx_img:+.1f}px — no correction needed"); return v_work, dx_img

    depth  = float(v_work[bm, 2].mean()) + cam_t[2]
    dx_w   = dx_img * depth / focal   # world x offset
    print(f"  [{name:20s}] verts={cnt:5d}  sil_cx: {m_cx_img:.1f}→{h_cx_img:.1f}  "
          f"dx_img={dx_img:+.1f}px  dx_world={dx_w:+.5f}")

    v_work[bm, 0] += dx_w
    return v_work, dx_img


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    import torch
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    print(f"[INFO] torch {torch.__version__}  cuda={torch.cuda.is_available()}")

    model, cfg = load_sam_3d_body(CKPT, device='cuda', mhr_path=MHR_P)

    with open(KP_CACHE) as f:
        kpd = json.load(f)
    fr0 = kpd['frames'][0]['persons'][0]['keypoints']
    ax  = [v['x'] for v in fr0.values() if v['score'] > 0.3]
    ay  = [v['y'] for v in fr0.values() if v['score'] > 0.3]
    pad_x = (max(ax)-min(ax))*0.15; pad_y = (max(ay)-min(ay))*0.15
    bbox  = np.array([[max(0,min(ax)-pad_x), max(0,min(ay)-pad_y),
                       min(720,max(ax)+pad_x), min(1280,max(ay)+pad_y)]], dtype=np.float32)

    img_bgr = cv2.imread(str(ADDR_JPG))
    H, W    = img_bgr.shape[:2]    # 1280 × 720

    est = SAM3DBodyEstimator(model, cfg, human_detector=None,
                             human_segmentor=None, fov_estimator=None)
    print("[INFO] MHR inference...")
    t_inf = time.time()
    outs  = est.process_one_image(img_bgr, bboxes=bbox, use_mask=False, inference_type="body")
    dt_inf = time.time() - t_inf

    o      = outs[0]
    verts  = o["pred_vertices"].astype(np.float32)
    cam_t  = o["pred_cam_t"].astype(np.float32)
    kp2d   = o["pred_keypoints_2d"].astype(np.float32)
    focal  = float(o["focal_length"])
    faces  = est.faces
    vram   = torch.cuda.max_memory_allocated() / 1e6
    print(f"[INFO] inf {dt_inf:.1f}s  VRAM {vram:.0f}MB  focal={focal:.1f}  cam_t={cam_t}")

    print("[INFO] rembg silhouette...")
    mask_h = get_human_mask(img_bgr)
    cv2.imwrite(str(OUTPUT_DIR/"human_mask.jpg"), mask_h)
    print(f"[INFO] human_mask px: {(mask_h>0).sum()}")

    print("[INFO] T1 render (baseline)...")
    rgba_t1 = render_rgba(verts, cam_t, focal, faces, H, W)
    sil_t1  = get_sil(rgba_t1)
    cv2.imwrite(str(OUTPUT_DIR/"sil_t1.jpg"), sil_t1)

    verts2d_t1 = project_verts(verts, cam_t, focal, H, W)

    # y-centers from kp2d
    nose_y = int(kp2d[I_NOSE][1])
    neck_y = int(kp2d[I_NECK][1])
    sho_y  = int((kp2d[I_LSHO][1] + kp2d[I_RSHO][1]) / 2)
    hip_y  = int((kp2d[I_LHIP][1] + kp2d[I_RHIP][1]) / 2)
    knee_y = int((kp2d[I_LKNE][1] + kp2d[I_RKNE][1]) / 2)
    ank_y  = int((kp2d[I_LANK][1] + kp2d[I_RANK][1]) / 2)
    head_y = int(nose_y - 55)
    print(f"[INFO] y-centers: head={head_y} sho={sho_y} hip={hip_y} knee={knee_y} ank={ank_y}")

    # Band boundaries — NOTE: NO B0 (head_neck) to avoid shoulder contamination
    # B1: shoulder zone      [head_y-30, hip_y-15]  ← single upper body band
    # B2: hip zone           [hip_y-15,  hip_y+110]
    # B3: lower (达标即止)   [hip_y+110, ank_y+120]
    B_UPP_lo, B_UPP_hi = head_y - 30,  hip_y - 15    # ≈ [413, 676)
    B_HIP_lo, B_HIP_hi = hip_y - 15,   hip_y + 110   # ≈ [676, 801)
    B_LOW_lo, B_LOW_hi = hip_y + 110,  ank_y + 120   # ≈ [801, 1033)

    sho_ys    = list(range(neck_y, sho_y + 100, 15))   # dense shoulder measurements
    torso_ys  = list(range(sho_y + 100, hip_y - 15, 15))
    all_up_ys = sho_ys + torso_ys
    hip_ys    = [hip_y - 20, hip_y, hip_y + 20, hip_y + 40]
    low_ys    = [knee_y - 20, knee_y, knee_y + 30, (knee_y+ank_y)//2, ank_y-30]

    # ── STEP 1: Scale bands based on T1 silhouette measurements ──
    v_opt = verts.copy()
    print("\n[INFO] === Step 1: Scale (no translate) ===")

    v_opt = scale_band(v_opt, verts2d_t1, cam_t, focal, W,
                       B_UPP_lo, B_UPP_hi, mask_h, sil_t1,
                       measure_ys=all_up_ys, containment=1.10, name="B_UPP_scale")
    v_opt = scale_band(v_opt, verts2d_t1, cam_t, focal, W,
                       B_HIP_lo, B_HIP_hi, mask_h, sil_t1,
                       measure_ys=hip_ys, containment=1.05, name="B_HIP_scale")
    v_opt = scale_band(v_opt, verts2d_t1, cam_t, focal, W,
                       B_LOW_lo, B_LOW_hi, mask_h, sil_t1,
                       measure_ys=low_ys, containment=1.00, name="B_LOW_scale")

    # ── STEP 2: Render after scale, measure actual sil cx ──
    print("\n[INFO] Step 2 render (post-scale)...")
    rgba_s2 = render_rgba(v_opt, cam_t, focal, faces, H, W)
    sil_s2  = get_sil(rgba_s2)

    print("\n[INFO] === Step 2: Translate to align sil cx → human cx ===")
    v_opt, dx_up = translate_band_to_human_cx(
        v_opt, verts2d_t1, cam_t, focal, W, H,
        B_UPP_lo, B_UPP_hi, mask_h, sil_s2,
        measure_ys=sho_ys, name="B_UPP_translate")
    v_opt, dx_hip = translate_band_to_human_cx(
        v_opt, verts2d_t1, cam_t, focal, W, H,
        B_HIP_lo, B_HIP_hi, mask_h, sil_s2,
        measure_ys=hip_ys, name="B_HIP_translate")
    # lower: also translate to avoid discontinuity at hip boundary
    v_opt, dx_low = translate_band_to_human_cx(
        v_opt, verts2d_t1, cam_t, focal, W, H,
        B_LOW_lo, B_LOW_hi, mask_h, sil_s2,
        measure_ys=low_ys, name="B_LOW_translate")

    # ── STEP 3: Final render + verify ──
    print("\n[INFO] Step 3: Final render T1.6...")
    rgba_t16 = render_rgba(v_opt, cam_t, focal, faces, H, W)
    sil_t16  = get_sil(rgba_t16)
    cv2.imwrite(str(OUTPUT_DIR/"sil_t16.jpg"), sil_t16)

    # ── STEP 4: Check if shoulder/torso cx still needs correction (2nd pass) ──
    print("\n[INFO] Step 4: Check cx after final render...")
    h_cxs_final, m_cxs_final = [], []
    for yc in sho_ys[:6]:   # check shoulder range
        hm = measure_width(mask_h, yc)
        mm = measure_width(sil_t16, yc)
        if hm and mm:
            h_cxs_final.append(hm[2]); m_cxs_final.append(mm[2])
    if h_cxs_final:
        final_dx = float(np.median(h_cxs_final)) - float(np.median(m_cxs_final))
        print(f"  shoulder cx residual: {final_dx:+.1f}px")
        if abs(final_dx) > 2:
            print(f"  applying residual correction...")
            depth_b = float(v_opt[verts2d_t1[:,1] < B_HIP_lo, 2].mean()) + cam_t[2]
            dx_w2   = final_dx * depth_b / focal
            bm_up   = verts2d_t1[:,1] < B_HIP_lo
            v_opt[bm_up, 0] += dx_w2
            rgba_t16 = render_rgba(v_opt, cam_t, focal, faces, H, W)
            sil_t16  = get_sil(rgba_t16)
            cv2.imwrite(str(OUTPUT_DIR/"sil_t16.jpg"), sil_t16)
            print(f"  re-rendered after residual correction dx={final_dx:+.1f}px")

    # ── Width check ──
    print("\n[INFO] Width check T1 vs T1.6:")
    for yc, label in [(sho_y,'shoulder'), ((sho_y+hip_y)//2,'torso'),
                      (hip_y,'hip'), (knee_y,'knee'), (ank_y,'ankle')]:
        hm   = measure_width(mask_h, yc)
        mt1  = measure_width(sil_t1, yc)
        mt16 = measure_width(sil_t16, yc)
        hs  = f"{hm[3]}px(cx={hm[2]:.0f})"    if hm  else "N/A"
        s1  = f"{mt1[3]}px(cx={mt1[2]:.0f})"  if mt1 else "N/A"
        s16 = f"{mt16[3]}px(cx={mt16[2]:.0f})" if mt16 else "N/A"
        print(f"  {label:10s} @ y={yc}: human={hs}  T1={s1}  T1.6={s16}")

    # ── Edge miss ──
    regions_upper = {'shoulder': sho_y, 'torso': (sho_y+hip_y)//2, 'hip': hip_y}
    regions_lower = {'knee': knee_y, 'ankle': ank_y}
    regions_all   = {**regions_upper, **regions_lower}
    sc_t1  = edge_scores(sil_t1,  mask_h, regions_all)
    sc_t16 = edge_scores(sil_t16, mask_h, regions_all)

    print("\n=== EDGE MISS 分级报告 ===")
    print(f"  {'区域':14s} {'T1 miss':>16} {'T1.6 miss':>16} {'改进':>8} {'达标':>6}")
    print(f"  {'─'*66}")
    print("  [上半身 — 严格 ≤3%]")
    for r in ['shoulder', 'torso', 'hip']:
        t1_ = sc_t1[r]; t16_ = sc_t16[r]
        imp = t1_['miss_px'] - t16_['miss_px']
        ok  = "✓" if t16_['miss_ratio'] <= 3.0 else "✗"
        print(f"  {r:14s} {t1_['miss_px']:>8}px({t1_['miss_ratio']:>4.1f}%) "
              f"{t16_['miss_px']:>8}px({t16_['miss_ratio']:>4.1f}%) "
              f"{imp:>+8}px  {ok}")
    print("  [下肢 — 达标即止]")
    for r in ['knee', 'ankle']:
        t1_ = sc_t1[r]; t16_ = sc_t16[r]
        imp = t1_['miss_px'] - t16_['miss_px']
        print(f"  {r:14s} {t1_['miss_px']:>8}px({t1_['miss_ratio']:>4.1f}%) "
              f"{t16_['miss_px']:>8}px({t16_['miss_ratio']:>4.1f}%) "
              f"{imp:>+8}px  [达标即止]")

    upper_pass = all(sc_t16[r]['miss_ratio'] <= 3.0 for r in ['shoulder','torso','hip'])
    print(f"\n  上半身三区 ≤3%: {'[通过] 放行 T2' if upper_pass else '[未达标]'}")

    # ── Composites ──
    print("\n[INFO] Composites...")
    ov_t16 = composite_red(rgba_t16, img_bgr, alpha=0.55)
    ov_t1  = composite_red(rgba_t1,  img_bgr, alpha=0.55)
    lh_pt  = (int(kp2d[I_LHIP][0]), int(kp2d[I_LHIP][1]))
    rh_pt  = (int(kp2d[I_RHIP][0]), int(kp2d[I_RHIP][1]))
    y_pel  = (lh_pt[1] + rh_pt[1]) // 2
    for ov in [ov_t16, ov_t1]:
        cv2.line(ov, (0,y_pel),(W-1,y_pel),(255,255,255),2)
        for pt in [lh_pt, rh_pt]:
            cv2.circle(ov,pt,7,(255,255,255),-1); cv2.circle(ov,pt,3,(0,0,200),-1)

    p_ov  = OUTPUT_DIR/"mhr_overlay_t16.jpg"
    p_sbs = OUTPUT_DIR/"side_by_side_t16.jpg"
    p_sil = OUTPUT_DIR/"silhouette_compare.jpg"

    cv2.imwrite(str(p_ov), ov_t16)
    cv2.imwrite(str(OUTPUT_DIR/"mhr_overlay.jpg"), ov_t16)
    sbs = np.concatenate([img_bgr, ov_t1, ov_t16], axis=1)
    fn  = cv2.FONT_HERSHEY_SIMPLEX
    for txt,x in [("Original",0),("T1 (baseline MHR)",W),("T1.6 (shoulder-fit)",2*W)]:
        cv2.putText(sbs, txt, (x+12,44), fn, 1.1,(255,255,255),2)
    cv2.imwrite(str(p_sbs), sbs)
    cv2.imwrite(str(OUTPUT_DIR/"side_by_side.jpg"), np.concatenate([img_bgr, ov_t16],axis=1))

    def blend_sils(hm, mm):
        out = np.zeros((H,W,3),dtype=np.uint8)
        out[:,:,1]=hm; out[:,:,2]=mm; return out
    sil_cmp = np.concatenate([blend_sils(mask_h,sil_t1), blend_sils(mask_h,sil_t16)],axis=1)
    cv2.putText(sil_cmp,"T1: G=human R=mesh",(10,40),fn,1,(255,255,255),2)
    cv2.putText(sil_cmp,"T1.6: G=human R=mesh",(W+10,40),fn,1,(255,255,255),2)
    cv2.imwrite(str(p_sil), sil_cmp)

    # ── Joint fit ──
    print("\n=== 逐点关节误差 (姿态 unchanged) ===")
    jf = {}
    for jn, ji in MHR_NAMED.items():
        mx, my = float(kp2d[ji][0]), float(kp2d[ji][1])
        if jn in RTM_GT:
            rx,ry=RTM_GT[jn]; d=float(np.sqrt((mx-rx)**2+(my-ry)**2))
            g='好' if d<=10 else ('可' if d<=25 else '差')
        else:
            rx=ry=d=None; g='--'
        jf[jn]={'mhr_xy':[mx,my],'rtm_xy':[rx,ry],'dist_px':d,'grade':g}
        print(f"  {jn:22s}: d={f'{d:.1f}px' if d else 'N/A':>8}  {g}")

    # ── Log ──
    tt    = time.time() - t0
    vram2 = torch.cuda.max_memory_allocated() / 1e6
    log   = {
        "version":"T1.6_v4","phase":"address","frame":0,"clip":"fo-ok-1",
        "canvas_hw":[H,W],"inference_s":round(dt_inf,2),"total_s":round(tt,2),
        "peak_vram_mb":round(vram2,0),"focal_length":focal,"cam_t":cam_t.tolist(),
        "bands":{"B_UPP":[B_UPP_lo,B_UPP_hi],"B_HIP":[B_HIP_lo,B_HIP_hi],
                 "B_LOW":[B_LOW_lo,B_LOW_hi]},
        "method":"scale_then_translate_iterative","z_scaling":"disabled",
        "edge_t1":sc_t1,"edge_t16":sc_t16,"upper_body_pass":upper_pass,
        "joint_fit":jf,"topology":"MHR_native","smpl_dependency":None,"yolo_used":False,
        "license":"SAM_License_PRODUCT_CANDIDATE_CUSTOM_LICENSE",
        "tiered_standard":{"upper_body_threshold_pct":3.0,
                           "lower_body_policy":"达标即止 — 不再优化"},
    }
    lp = OUTPUT_DIR/"run_log_t16.json"
    with open(lp,'w') as f: json.dump(log,f,indent=2,default=str)

    print(f"\n[OUT] {p_ov}")
    print(f"[OUT] {p_sbs}")
    print(f"[OUT] {p_sil}")
    print(f"[OUT] {lp}")
    print(f"[DONE] Total {tt:.1f}s  VRAM {vram2:.0f}MB  upper_pass={upper_pass}")


if __name__ == "__main__":
    main()
