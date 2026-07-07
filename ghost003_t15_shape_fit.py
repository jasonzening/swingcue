"""
ghost003_t15_shape_fit.py  —  GHOST-003 T1.5 MHR 体型拟合优化层
授权: SAM License (PRODUCT_CANDIDATE_CUSTOM_LICENSE)
策略(v3 — 正确投影分带):
  1. rembg 取真人 silhouette mask
  2. 向量化投影所有 18439 顶点 → 2D 图像坐标, 按 image_y 分带
  3. 每带: 测量 human_width / mesh_width → sx (上限 1.5)
  4. 在 3D 空间 scale x, 以 world_cx (来自 mesh image_cx) 为中心
  5. 渲染: 红色半透明 (1,0,0) alpha=0.55 + 白色骨盆基准线
  6. 边缘吻合评估: 肩/髋/膝/踝 区域 miss_px + mean_dist
范围: 单帧 · 真实姿态 · 不改姿态 · 不做整段 · 不碰球杆
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

MHR_IDX = {
    'nose': 0,
    'left_shoulder': 5,  'right_shoulder': 6,
    'left_elbow': 7,     'right_elbow': 8,
    'left_hip': 9,       'right_hip': 10,
    'left_knee': 11,     'right_knee': 12,
    'left_ankle': 13,    'right_ankle': 14,
    'right_wrist': 41,   'left_wrist': 62,
    'neck': 69,
}
RTM_GT = {
    'nose':           (294.5, 517.4),
    'left_shoulder':  (366.7, 548.4), 'right_shoulder': (256.3, 565.9),
    'left_elbow':     (347.1, 638.2), 'right_elbow':    (279.0, 647.5),
    'left_wrist':     (323.4, 706.3), 'right_wrist':    (299.6, 721.8),
    'left_hip':       (357.4, 679.5), 'right_hip':      (292.4, 684.6),
    'left_knee':      (371.9, 782.7), 'right_knee':     (267.6, 787.9),
    'left_ankle':     (391.5, 910.7), 'right_ankle':    (262.5, 912.8),
}


# ── Projection ──────────────────────────────────────────────────────────────
# pyrender Renderer.__call__ applies:
#   1. camera_translation = cam_t with cam_t[0] *= -1
#   2. mesh rotated 180° around x  => (vx, vy, vz) → (vx, -vy, -vz)
#   3. camera at camera_translation, identity rotation, looks -Z (OpenGL)
#
# world-space after mesh rotation: vr = (vx, -vy, -vz)
# camera-space: vc = vr - camera_pos = (vx - (-cx), -vy - cy, -vz - cz)
#             = (vx + ct0, -vy - ct1, -vz - ct2)   where ct0=cam_t[0]*(-1)..
# NOTE: cam_t[0] *= -1 → ct0 = -cam_t[0_orig]
# projection (look -Z → depth = -vc.z):
#   depth = vz + cam_t[2]
#   image_x = focal * (vx - cam_t[0]) / depth + W/2
#   image_y = focal * (vy + cam_t[1]) / depth + H/2   (plus sign from double-neg)

def project_verts(verts, cam_t, focal, H, W):
    """
    Vectorized projection of all vertices to image coords.
    Returns (V,2) float array [image_x, image_y].
    """
    vx = verts[:, 0]
    vy = verts[:, 1]
    vz = verts[:, 2]
    depth = vz + cam_t[2]
    depth = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    px = focal * (vx - cam_t[0]) / depth + W / 2.0
    py = focal * (vy + cam_t[1]) / depth + H / 2.0
    return np.stack([px, py], axis=1)


# ── Human mask ──────────────────────────────────────────────────────────────
def get_human_mask(img_bgr):
    import rembg
    out = rembg.remove(img_bgr)
    alpha = out[:, :, 3]
    mask = (alpha > 40).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    return mask


# ── Render to RGBA ───────────────────────────────────────────────────────────
def render_rgba(verts, cam_t, focal, faces, H, W):
    from sam_3d_body.visualization.renderer import Renderer
    dummy = np.zeros((H, W, 3), dtype=np.uint8)
    rend  = Renderer(focal_length=focal, faces=faces)
    rgba  = rend(verts, cam_t, dummy.copy(),
                 mesh_base_color=(1.0, 0.0, 0.0),
                 scene_bg_color=(0, 0, 0), return_rgba=True)
    return rgba   # (H,W,4) float 0-1


def get_silhouette(rgba):
    if rgba.shape[2] == 4:
        return (rgba[:, :, 3] > 0.05).astype(np.uint8) * 255
    return (rgba.sum(2) > 0.05).astype(np.uint8) * 255


# ── Width measurement ────────────────────────────────────────────────────────
def measure_width(mask, y_center, half_band=35):
    """Returns (xmin, xmax, cx, width) or None."""
    H, W = mask.shape
    y1, y2 = max(0, y_center-half_band), min(H, y_center+half_band)
    band = mask[y1:y2]
    cols = np.where(band.any(axis=0))[0]
    if len(cols) < 3:
        return None
    return int(cols.min()), int(cols.max()), int((cols.min()+cols.max())//2), int(cols.max()-cols.min())


# ── Edge miss scoring ────────────────────────────────────────────────────────
def edge_scores(sil_pred, sil_human, regions_y, band=60):
    from scipy.ndimage import distance_transform_edt
    dist_map = distance_transform_edt(sil_pred == 0)
    H = sil_pred.shape[0]
    out = {}
    for rname, yc in regions_y.items():
        y1 = max(0, yc-band); y2 = min(H, yc+band)
        hb = sil_human[y1:y2]; pb = sil_pred[y1:y2]; db = dist_map[y1:y2]
        miss = (hb>0)&(pb==0)
        nh = (hb>0).sum()
        nm = miss.sum()
        md = float(db[miss].mean()) if nm>0 else 0.0
        out[rname] = {'miss_px': int(nm),
                      'miss_ratio': round(nm/max(nh,1)*100, 1),
                      'mean_dist_px': round(md, 1)}
    return out


# ── Scale vertices ───────────────────────────────────────────────────────────
def scale_verts_by_band(verts, verts2d, bands, H):
    """
    bands: list of dicts:
      {name, iy_lo, iy_hi, sx, world_cx}
    Apply scale x around world_cx for verts whose projected y ∈ [iy_lo, iy_hi].
    """
    v = verts.copy()
    py = verts2d[:, 1]   # projected y for each vertex

    for b in bands:
        iy_lo, iy_hi = b['iy_lo'], b['iy_hi']
        sx        = b['sx']
        world_cx  = b['world_cx']
        bmask = (py >= iy_lo) & (py < iy_hi)
        if bmask.sum() == 0:
            continue
        v[bmask, 0] = world_cx + (v[bmask, 0] - world_cx) * sx
        # z proportional to maintain depth/thickness
        v[bmask, 2] = v[bmask, 2] * np.sqrt(sx)  # gentler z change
    return v


def world_cx_from_img(img_cx, verts, verts2d, depth_mean, cam_t, focal, W):
    """
    Given image cx of mesh at this band, compute corresponding 3D cx.
    image_x = focal * (vx - cam_t[0]) / depth + W/2
    => vx = (image_x - W/2) * depth / focal + cam_t[0]
    """
    return (img_cx - W / 2.0) * depth_mean / focal + cam_t[0]


# ── Composite overlay ────────────────────────────────────────────────────────
def composite_red(rgba, img_bgr, alpha=0.55):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if rgba.shape[2] == 4:
        mesh_a = rgba[:, :, 3:4]
        mesh_c = rgba[:, :, :3]
    else:
        mesh_c = rgba
        mesh_a = (rgba.sum(2, keepdims=True) > 0.02).astype(np.float32)
    blended = img_rgb * (1 - mesh_a * alpha) + mesh_c * mesh_a * alpha
    return cv2.cvtColor(
        np.clip(blended * 255, 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    import torch
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    print(f"[INFO] torch {torch.__version__}  cuda={torch.cuda.is_available()}")

    model, cfg = load_sam_3d_body(CKPT, device='cuda', mhr_path=MHR_P)

    with open(KP_CACHE) as f:
        kpd = json.load(f)
    fr0 = kpd['frames'][0]['persons'][0]['keypoints']
    ax = [v['x'] for v in fr0.values() if v['score']>0.3]
    ay = [v['y'] for v in fr0.values() if v['score']>0.3]
    pad_x = (max(ax)-min(ax))*0.15; pad_y = (max(ay)-min(ay))*0.15
    bbox = np.array([[max(0,min(ax)-pad_x), max(0,min(ay)-pad_y),
                      min(720,max(ax)+pad_x), min(1280,max(ay)+pad_y)]], dtype=np.float32)

    img_bgr = cv2.imread(str(ADDR_JPG))
    H, W = img_bgr.shape[:2]    # 1280, 720

    est = SAM3DBodyEstimator(model, cfg,
                             human_detector=None,
                             human_segmentor=None,
                             fov_estimator=None)
    print("[INFO] MHR inference...")
    t_inf = time.time()
    outs = est.process_one_image(img_bgr, bboxes=bbox, use_mask=False, inference_type="body")
    dt_inf = time.time() - t_inf

    o      = outs[0]
    verts  = o["pred_vertices"].astype(np.float32)
    cam_t  = o["pred_cam_t"].astype(np.float32)
    kp2d   = o["pred_keypoints_2d"].astype(np.float32)
    focal  = float(o["focal_length"])
    faces  = est.faces
    vram   = torch.cuda.max_memory_allocated() / 1e6
    print(f"[INFO] inf {dt_inf:.1f}s  VRAM {vram:.0f}MB  focal={focal:.1f}  cam_t={cam_t}")

    # debug: vertex y range in 3D
    print(f"[INFO] vert y range: {verts[:,1].min():.3f} .. {verts[:,1].max():.3f}")
    print(f"[INFO] vert z range: {verts[:,2].min():.3f} .. {verts[:,2].max():.3f}")

    # ── rembg human mask ──
    print("[INFO] rembg silhouette...")
    mask_h = get_human_mask(img_bgr)
    cv2.imwrite(str(OUTPUT_DIR/"human_mask.jpg"), mask_h)
    print(f"[INFO] human mask px: {(mask_h>0).sum()}")

    # ── project all vertices ──
    verts2d = project_verts(verts, cam_t, focal, H, W)
    print(f"[INFO] verts2d x: {verts2d[:,0].min():.1f}..{verts2d[:,0].max():.1f}")
    print(f"[INFO] verts2d y: {verts2d[:,1].min():.1f}..{verts2d[:,1].max():.1f}")

    # ── T1 silhouette ──
    print("[INFO] T1 render...")
    rgba_t1 = render_rgba(verts, cam_t, focal, faces, H, W)
    sil_t1  = get_silhouette(rgba_t1)
    cv2.imwrite(str(OUTPUT_DIR/"sil_t1.jpg"), sil_t1)

    # ── Define y-bands from kp2d (image coords) ──
    # These are the y-centers for human landmark positions
    # We create 5 bands covering shoulder, torso, hip, thigh, leg
    kp = kp2d  # shorthand

    # key y positions in image
    sho_y  = int((kp[5][1] + kp[6][1])   / 2)   # shoulders
    hip_y  = int((kp[9][1] + kp[10][1])  / 2)   # hips
    knee_y = int((kp[11][1] + kp[12][1]) / 2)   # knees
    ank_y  = int((kp[13][1] + kp[14][1]) / 2)   # ankles
    head_y = int(kp[0][1] - 40)                  # above nose

    print(f"[INFO] y-centers: head={head_y} sho={sho_y} hip={hip_y} knee={knee_y} ank={ank_y}")

    # 5 bands (by image_y of projected vertices)
    img_bands = [
        {'name':'head_sho', 'iy_lo': head_y,       'iy_hi': sho_y+40},
        {'name':'torso',    'iy_lo': sho_y+40,      'iy_hi': hip_y-15},
        {'name':'hip',      'iy_lo': hip_y-40,      'iy_hi': hip_y+60},
        {'name':'thigh',    'iy_lo': hip_y+60,      'iy_hi': knee_y+40},
        {'name':'leg',      'iy_lo': knee_y+40,     'iy_hi': ank_y+50},
    ]
    # note: bands overlap is fine; each vertex is assigned to the band whose center it's closest to

    # For non-overlapping assignment, use exclusive bands
    excl_bands = [
        {'name':'head_sho', 'iy_lo': head_y,        'iy_hi': int((sho_y+40+hip_y-40)/2)},
        {'name':'hip_down', 'iy_lo': int((sho_y+40+hip_y-40)/2), 'iy_hi': ank_y+80},
    ]
    # Simpler: 3 non-overlapping bands for scaling
    mid1 = int((sho_y + hip_y) / 2)
    mid2 = int((hip_y + ank_y) / 2)
    scale_bands = [
        {'name':'upper', 'iy_lo': head_y - 30, 'iy_hi': mid1,    'meas_y': sho_y},
        {'name':'mid',   'iy_lo': mid1,         'iy_hi': mid2,    'meas_y': hip_y},
        {'name':'lower', 'iy_lo': mid2,         'iy_hi': ank_y+80,'meas_y': knee_y},
    ]

    # ── Measure widths and compute sx ──
    print("\n[INFO] Width measurements:")
    depth_mean = float(verts[:, 2].mean()) + cam_t[2]

    for b in scale_bands:
        yc = b['meas_y']
        h_m = measure_width(mask_h, yc, half_band=35)
        m_m = measure_width(sil_t1, yc, half_band=35)

        if h_m is None or m_m is None:
            print(f"  {b['name']:10s} @ y={yc}: NO DATA → sx=1.0")
            b['sx'] = 1.0
            b['world_cx'] = float(verts[:, 0].mean())
            continue

        h_w, m_w = h_m[3], m_m[3]
        h_cx, m_cx = h_m[2], m_m[2]

        sx = h_w / max(m_w, 1)
        sx = min(max(sx, 0.90), 1.50)   # cap: 0.90..1.50

        # world cx from mesh image center
        b['world_cx'] = world_cx_from_img(m_cx, verts, verts2d, depth_mean, cam_t, focal, W)
        b['sx'] = sx

        print(f"  {b['name']:10s} @ y={yc}: human={h_w}px(@{h_cx})  mesh={m_w}px(@{m_cx})"
              f"  sx={sx:.3f}  world_cx={b['world_cx']:.4f}")

    # ── Apply scaling ──
    verts_opt = scale_verts_by_band(verts, verts2d, scale_bands, H)
    print(f"[INFO] Scaled {verts_opt.shape[0]} vertices")

    # ── Verify: measure new widths ──
    print("[INFO] T1.5 render (verify)...")
    rgba_opt = render_rgba(verts_opt, cam_t, focal, faces, H, W)
    sil_opt  = get_silhouette(rgba_opt)
    cv2.imwrite(str(OUTPUT_DIR/"sil_t15.jpg"), sil_opt)

    print("\n[INFO] Post-scale width verification:")
    for b in scale_bands:
        yc = b['meas_y']
        h_m = measure_width(mask_h, yc, half_band=35)
        m_m = measure_width(sil_opt, yc, half_band=35)
        h_str = f"{h_m[3]}px(@{h_m[2]})" if h_m else "N/A"
        m_str = f"{m_m[3]}px(@{m_m[2]})" if m_m else "N/A"
        print(f"  {b['name']:10s} @ y={yc}: human={h_str}  mesh={m_str}  target_sx={b['sx']:.3f}")

    # ── Edge scores ──
    regions_y = {'shoulder':sho_y, 'hip':hip_y, 'knee':knee_y, 'ankle':ank_y}
    sc_t1  = edge_scores(sil_t1,  mask_h, regions_y)
    sc_t15 = edge_scores(sil_opt, mask_h, regions_y)

    print("\n[INFO] Edge miss comparison:")
    print(f"  {'region':12s} {'T1 miss':>12} {'T1 dist':>10} {'T1.5 miss':>12} {'T1.5 dist':>10} {'改进':>8}")
    for r in regions_y:
        t1  = sc_t1[r];  t15 = sc_t15[r]
        imp = t1['miss_px'] - t15['miss_px']
        print(f"  {r:12s} {t1['miss_px']:>8}px({t1['miss_ratio']:>4.1f}%) "
              f"{t1['mean_dist_px']:>8.1f}px "
              f"{t15['miss_px']:>8}px({t15['miss_ratio']:>4.1f}%) "
              f"{t15['mean_dist_px']:>8.1f}px "
              f"{imp:>+8}px")

    # ── Final render ──
    print("\n[INFO] Final render + composite...")
    rgba_final = rgba_opt
    overlay    = composite_red(rgba_final, img_bgr, alpha=0.55)

    # pelvis baseline
    lh = kp2d[MHR_IDX['left_hip']]; rh = kp2d[MHR_IDX['right_hip']]
    y_mid = int((lh[1]+rh[1])/2)
    cv2.line(overlay, (0,y_mid), (W-1,y_mid), (255,255,255), 2)
    for pt in [lh, rh]:
        cv2.circle(overlay, (int(pt[0]),int(pt[1])), 6, (255,255,255), -1)
        cv2.circle(overlay, (int(pt[0]),int(pt[1])), 3, (0,0,200), -1)

    # T1 for comparison (also red)
    t1_overlay = composite_red(rgba_t1, img_bgr, alpha=0.55)
    cv2.line(t1_overlay, (0,y_mid), (W-1,y_mid), (255,255,255), 2)

    # 3-panel
    sbs3 = np.concatenate([img_bgr, t1_overlay, overlay], axis=1)
    labels = [("Original",0), ("T1 (original MHR)",W), ("T1.5 (shape-fit)",2*W)]
    for txt, x in labels:
        cv2.putText(sbs3, txt, (x+10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2)

    # Silhouette comparison panel
    sil_rgb_h  = cv2.cvtColor(mask_h, cv2.COLOR_GRAY2BGR)
    sil_rgb_t1 = cv2.cvtColor(sil_t1, cv2.COLOR_GRAY2BGR)
    sil_rgb_t15= cv2.cvtColor(sil_opt, cv2.COLOR_GRAY2BGR)
    # overlay human (green) and mesh (red) silhouettes
    def blend_sils(human, mesh):
        out = np.zeros((H,W,3), dtype=np.uint8)
        out[:,:,1] = human   # green = human
        out[:,:,2] = mesh    # red = mesh (opencv BGR)
        return out
    sil_compare = np.concatenate([
        blend_sils(mask_h, sil_t1),
        blend_sils(mask_h, sil_opt),
    ], axis=1)
    cv2.putText(sil_compare, "T1: green=human, red=mesh", (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(sil_compare, "T1.5: green=human, red=mesh", (W+10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Save
    p_ov    = OUTPUT_DIR / "mhr_overlay_t15.jpg"
    p_sbs3  = OUTPUT_DIR / "side_by_side_t15.jpg"
    p_silc  = OUTPUT_DIR / "silhouette_compare.jpg"

    cv2.imwrite(str(p_ov),   overlay)
    cv2.imwrite(str(p_sbs3), sbs3)
    cv2.imwrite(str(p_silc), sil_compare)

    # overwrite main deliverable names
    cv2.imwrite(str(OUTPUT_DIR/"mhr_overlay.jpg"), overlay)
    cv2.imwrite(str(OUTPUT_DIR/"side_by_side.jpg"),
                np.concatenate([img_bgr, overlay], axis=1))

    # ── Joint fit (unchanged) ──
    print("\n=== 逐点关节误差 (joints unchanged by shape opt) ===")
    jf = {}
    for jn, ji in MHR_IDX.items():
        mx,my = float(kp2d[ji][0]), float(kp2d[ji][1])
        if jn in RTM_GT:
            rx,ry = RTM_GT[jn]
            d = float(np.sqrt((mx-rx)**2+(my-ry)**2))
            g = '好' if d<=10 else ('可' if d<=25 else '差')
        else:
            rx=ry=d=None; g='--'
        jf[jn] = {'mhr_xy':[mx,my],'rtm_xy':[rx,ry],'dist_px':d,'grade':g}
        print(f"  {jn:20s}: d={f'{d:.1f}px' if d else 'N/A':>8}  {g}")

    # ── Log ──
    tt = time.time() - t0
    log = {
        "phase":"address","frame":0,"clip":"fo-ok-1",
        "canvas_hw":[H,W],"inference_s":round(dt_inf,2),
        "total_s":round(tt,2),"peak_vram_mb":round(vram,0),
        "focal_length":focal,"cam_t":cam_t.tolist(),
        "scale_bands":[{k:v for k,v in b.items() if k!='iy_lo' and k!='iy_hi'}
                       for b in scale_bands],
        "joint_fit":jf,
        "edge_t1":sc_t1,"edge_t15":sc_t15,
        "topology":"MHR_native","smpl_dependency":None,"yolo_used":False,
        "license":"SAM_License_PRODUCT_CANDIDATE_CUSTOM_LICENSE",
    }
    lp = OUTPUT_DIR/"run_log_t15.json"
    with open(lp,'w') as f:
        json.dump(log,f,indent=2,default=str)

    print(f"\n[OUT] {p_ov}")
    print(f"[OUT] {p_sbs3}")
    print(f"[OUT] {p_silc}")
    print(f"[OUT] {lp}")
    print(f"[DONE] Total {tt:.1f}s  VRAM {vram:.0f}MB")


if __name__ == "__main__":
    main()
