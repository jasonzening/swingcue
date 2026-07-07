"""
ghost003_mhr_overlay.py  —  GHOST-003 T1 MHR 单帧贴合探针
授权: SAM License (PRODUCT_CANDIDATE_CUSTOM_LICENSE)
     MHR native topology; 无 SMPL/SMPL-X pkl 依赖; 无 Ultralytics YOLO (AGPL)
范围锁定: 单帧 · 真实姿态 · 贴合渲染  (不做动作修正/整段/球杆)
相位: address (B层8相位体系)
输入: fo-ok-1 frame 0 (address帧)  720x1280 画布
GHOST-003 验收标准: 头/颈/肩/肘/腕/髋/膝/踝 逐点契合
"""

import os
import sys
import time
import json
import pathlib
import numpy as np
import cv2

# ---------- 路径设置 ----------
SAM3D_REPO = "/home/jason/projects/sam-3d-body"
SWINGCUE   = "/home/jason/projects/swingcue-postest"
sys.path.insert(0, SAM3D_REPO)
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

OUTPUT_DIR = pathlib.Path(SWINGCUE) / "output" / "ghost003"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KP_CACHE = pathlib.Path(SWINGCUE) / "engine/kp_cache/batch2/fo-ok-1.json"

# fo-ok-1 address 帧源图
# 提取 frame 0 的真实视频帧
VIDEO_FILE = None  # 若有视频可直接解帧; 否则用缓存帧
ADDRESS_JPG = OUTPUT_DIR / "address_frame.jpg"

HF_REPO = "facebook/sam-3d-body-dinov3"

# ---------- helpers ----------

def extract_address_frame():
    """从视频提取 address 帧 (fr0)，若已存在则跳过"""
    if ADDRESS_JPG.exists():
        print(f"[INFO] address_frame.jpg already exists: {ADDRESS_JPG}")
        return cv2.imread(str(ADDRESS_JPG))

    # 尝试从视频文件提取
    video_candidates = [
        pathlib.Path(SWINGCUE) / "data/batch2/fo-ok-1.mp4",
        pathlib.Path(SWINGCUE) / "data/batch2/fo-ok-1.mov",
        pathlib.Path(SWINGCUE) / "data/fo-ok-1.mp4",
    ]
    for vp in video_candidates:
        if vp.exists():
            cap = cv2.VideoCapture(str(vp))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            cap.release()
            if ret:
                cv2.imwrite(str(ADDRESS_JPG), frame)
                print(f"[INFO] Extracted fr0 from {vp} -> {ADDRESS_JPG}")
                return frame

    print("[WARN] No video found; checking for pre-saved address frame from ghost002...")
    ghost002_addr = pathlib.Path(SWINGCUE) / "output/ghost002/address_frame.jpg"
    if ghost002_addr.exists():
        img = cv2.imread(str(ghost002_addr))
        cv2.imwrite(str(ADDRESS_JPG), img)
        print(f"[INFO] Copied address_frame from ghost002: {ADDRESS_JPG}")
        return img

    raise FileNotFoundError(
        "Cannot find address frame. Provide fo-ok-1 video at data/batch2/ or pre-saved address_frame.jpg"
    )


def get_rtmpose_bbox():
    """从 RTMPose kp_cache 提取 address 帧 bbox (15% padding, x1y1x2y2)"""
    with open(KP_CACHE) as f:
        data = json.load(f)

    fr0 = data['frames'][0]
    kp = fr0['persons'][0]['keypoints']

    all_x = [v['x'] for v in kp.values() if v['score'] > 0.3]
    all_y = [v['y'] for v in kp.values() if v['score'] > 0.3]
    x1, y1 = min(all_x), min(all_y)
    x2, y2 = max(all_x), max(all_y)
    pad_x = (x2 - x1) * 0.15
    pad_y = (y2 - y1) * 0.15
    bbox = np.array([[
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(720, x2 + pad_x),
        min(1280, y2 + pad_y),
    ]], dtype=np.float32)
    print(f"[INFO] RTMPose bbox [x1,y1,x2,y2]: {bbox[0].tolist()}")
    return bbox


def render_red_translucent(vertices, cam_t, faces, img_bgr, alpha=0.55):
    """
    红色半透明 MHR mesh 叠加在 img_bgr 上。
    使用 SAM3D 自带 Renderer, 再做 alpha 合成。
    返回 (H,W,3) uint8 BGR
    """
    from sam_3d_body.visualization.renderer import Renderer

    RED_NORM = (1.0, 0.0, 0.0)  # mesh_base_color as normalized RGB
    rend = Renderer(focal_length=cam_t[2] * 0.9 if cam_t[2] > 100 else 5000.0, faces=faces)

    # focal_length from model output (pred_cam_t[2] * scale factor).
    # We pass focal_length separately after inference; placeholder here.
    # Actual call happens in main after we have focal_length.

    overlay = rend(
        vertices,
        cam_t,
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        mesh_base_color=RED_NORM,
        scene_bg_color=(0, 0, 0),
        return_rgba=True,
    )
    # overlay may be (H,W,3) float 0-1 or (H,W,4)
    if overlay.ndim == 3 and overlay.shape[2] == 4:
        mask = overlay[:, :, 3:4]
        rgb  = overlay[:, :, :3]
    else:
        # derive mask from non-black pixels
        rgb  = overlay
        mask = (overlay.sum(axis=2, keepdims=True) > 0.05).astype(np.float32)

    base = img_bgr.astype(np.float32) / 255.0
    base_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    blended = base_rgb * (1.0 - mask * alpha) + rgb * mask * alpha
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)


def draw_pelvis_baseline(img_bgr, vertices):
    """
    白色骨盆基准线: MHR 原生拓扑中左右髋顶点均值 → 水平线
    MHR 70关节：joint indices 1=left_hip, 2=right_hip (0-based, 参见 mhr70.py)
    """
    # We draw from projected 2D positions of left/right hip vertices
    # Use pred_keypoints_2d from model output instead (more accurate)
    return img_bgr  # placeholder; actual draw in main with 2d keypoints


def draw_pelvis_baseline_2d(img_bgr, kp2d_left_hip, kp2d_right_hip, h, w):
    """
    kp2d_*: (x, y) in image coords
    """
    x1 = int(np.clip(kp2d_left_hip[0],  0, w-1))
    y1 = int(np.clip(kp2d_left_hip[1],  0, h-1))
    x2 = int(np.clip(kp2d_right_hip[0], 0, w-1))
    y2 = int(np.clip(kp2d_right_hip[1], 0, h-1))
    # horizontal baseline through mid-hip
    y_mid = (y1 + y2) // 2
    out = img_bgr.copy()
    cv2.line(out, (0, y_mid), (w-1, y_mid), (255, 255, 255), 2)
    # small dot at each hip
    cv2.circle(out, (x1, y1), 5, (255, 255, 255), -1)
    cv2.circle(out, (x2, y2), 5, (255, 255, 255), -1)
    return out


# ---------- main ----------

def main():
    t0 = time.time()

    import torch
    print(f"[INFO] torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    device = torch.device("cuda")

    # 1. Address frame
    img_bgr = extract_address_frame()
    h, w = img_bgr.shape[:2]
    print(f"[INFO] image shape: {h}x{w} (HxW)")

    # 2. RTMPose bbox
    bbox = get_rtmpose_bbox()  # shape (1,4)

    # 3. Load SAM 3D Body from local cache (已下载到 ~/.cache/sam3d/)
    CKPT  = "/home/jason/.cache/sam3d/sam-3d-body-dinov3/model.ckpt"
    MHR_P = "/home/jason/.cache/sam3d/sam-3d-body-dinov3/assets/mhr_model.pt"
    print(f"[INFO] Loading SAM 3D Body from local cache: {CKPT}")
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    model, model_cfg = load_sam_3d_body(CKPT, device=str(device), mhr_path=MHR_P)

    # 4. Run inference (body only; no YOLO; no SAM2 segmentor)
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )

    print("[INFO] Running MHR inference (address frame, body mode)...")
    t_infer_start = time.time()
    outputs = estimator.process_one_image(
        img_bgr,          # BGR numpy array
        bboxes=bbox,      # RTMPose-derived bbox, no YOLO
        use_mask=False,
        inference_type="body",  # body decoder only; no hand
    )
    t_infer = time.time() - t_infer_start
    print(f"[INFO] Inference done in {t_infer:.1f}s  outputs={len(outputs)}")

    if len(outputs) == 0:
        print("[ERROR] No person detected. Check bbox.")
        return

    out0 = outputs[0]
    vertices   = out0["pred_vertices"]       # (V, 3)
    cam_t      = out0["pred_cam_t"]          # (3,)
    kp2d       = out0["pred_keypoints_2d"]   # (J, 2)  MHR 70 joints, image coords
    focal_len  = float(out0["focal_length"])

    print(f"[INFO] vertices={vertices.shape}  cam_t={cam_t}  focal={focal_len:.1f}")
    print(f"[INFO] kp2d shape={kp2d.shape}")

    # peak VRAM
    vram_mb = torch.cuda.max_memory_allocated() / 1e6
    print(f"[INFO] Peak VRAM: {vram_mb:.0f} MB")

    # MHR70 keypoint indices (sam_3d_body/metadata/mhr70.py, 0-based):
    # 0=nose, 1=left-eye, 2=right-eye, 3=left-ear, 4=right-ear,
    # 5=left-shoulder, 6=right-shoulder, 7=left-elbow, 8=right-elbow,
    # 9=left-hip, 10=right-hip, 11=left-knee, 12=right-knee,
    # 13=left-ankle, 14=right-ankle, 41=right-wrist, 62=left-wrist, 69=neck
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

    print("\n[INFO] MHR 2D keypoint projection (image coords):")
    fit_table = {}
    for name, idx in MHR_IDX.items():
        if idx < kp2d.shape[0]:
            x, y = kp2d[idx]
            fit_table[name] = (float(x), float(y))
            print(f"  {name:20s}: ({x:.1f}, {y:.1f})")

    # 5. Red translucent overlay
    from sam_3d_body.visualization.renderer import Renderer
    faces = estimator.faces  # MHR native faces

    RED = (1.0, 0.0, 0.0)
    rend = Renderer(focal_length=focal_len, faces=faces)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    overlay_raw = rend(
        vertices,
        cam_t,
        img_rgb.copy(),
        mesh_base_color=RED,
        scene_bg_color=(0, 0, 0),
        return_rgba=True,
    )

    # alpha composite
    ALPHA = 0.55
    if overlay_raw.ndim == 3 and overlay_raw.shape[2] == 4:
        mask = overlay_raw[:, :, 3:4]
        rgb_layer = overlay_raw[:, :, :3]
    else:
        rgb_layer = overlay_raw
        mask = (overlay_raw.sum(axis=2, keepdims=True) > 0.02).astype(np.float32)

    base_f = img_rgb.astype(np.float32) / 255.0
    blended = base_f * (1.0 - mask * ALPHA) + rgb_layer * mask * ALPHA
    blended_bgr = cv2.cvtColor(
        np.clip(blended * 255, 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR
    )

    # 6. White pelvis baseline
    if 'left_hip' in fit_table and 'right_hip' in fit_table:
        blended_bgr = draw_pelvis_baseline_2d(
            blended_bgr,
            fit_table['left_hip'],
            fit_table['right_hip'],
            h, w
        )

    # 7. Save outputs
    overlay_path = OUTPUT_DIR / "mhr_overlay.jpg"
    cv2.imwrite(str(overlay_path), blended_bgr)
    print(f"[OUT] mhr_overlay.jpg -> {overlay_path}")

    # side_by_side
    sbs = np.concatenate([img_bgr, blended_bgr], axis=1)
    sbs_path = OUTPUT_DIR / "side_by_side.jpg"
    cv2.imwrite(str(sbs_path), sbs)
    print(f"[OUT] side_by_side.jpg -> {sbs_path}")

    # 8. Fit log JSON
    t_total = time.time() - t0
    log = {
        "phase": "address",
        "frame": 0,
        "clip": "fo-ok-1",
        "canvas_hw": [h, w],
        "bbox_rtmpose": bbox[0].tolist(),
        "focal_length": focal_len,
        "cam_t": cam_t.tolist(),
        "inference_s": round(t_infer, 2),
        "total_s": round(t_total, 2),
        "peak_vram_mb": round(vram_mb, 0),
        "mhr_kp2d": {k: list(v) for k, v in fit_table.items()},
        "topology": "MHR_native",
        "smpl_dependency": None,
        "yolo_used": False,
        "license": "SAM_License_PRODUCT_CANDIDATE_CUSTOM_LICENSE",
    }
    log_path = OUTPUT_DIR / "run_log.json"
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"[OUT] run_log.json -> {log_path}")
    print(f"\n[DONE] Total: {t_total:.1f}s  VRAM: {vram_mb:.0f}MB")


if __name__ == "__main__":
    main()
