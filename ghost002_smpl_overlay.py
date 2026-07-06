"""
ghost002_smpl_overlay.py  — GHOST-002 T1 SMPL 单帧贴合验证

范围严格锁定：单帧 address 帧 + 真实姿态 + 贴合渲染
不做动作修正 / 不做整段视频 / 不做时序 / 不碰球杆

流程：
1. 从 kp_cache 读取 RTMPose address 帧关节点（kp_guard 已验证）
2. 从关节点计算 bounding box（绕过 detectron2）
3. HMR2.0 推理得 SMPL 参数
4. pyrender RGBA 渲染 SMPL mesh（红色半透明）
5. 叠加到 address 帧 + 加白色垂直基准线（过骨盆中心）
6. 产出 overlay.jpg + side_by_side.jpg

相位命名: B层 8 相位体系
  address / takeaway / backswing / top / transition / downswing / impact / follow_through
"""
import os, sys, json, pathlib, time
import numpy as np
import cv2

os.environ['PYOPENGL_PLATFORM'] = 'egl'
sys.path.insert(0, '/home/jason/projects/4d-humans')

ROOT   = pathlib.Path('/home/jason/projects/swingcue-postest')
OUT    = ROOT / 'output/ghost002'
OUT.mkdir(parents=True, exist_ok=True)

VIDEO  = pathlib.Path('/mnt/c/Users/jason/Zening/Swingcue/Video/fo-ok-1.mp4')
KP_CACHE = ROOT / 'engine/kp_cache/batch2/fo-ok-1.json'
ADDR_FR  = 0   # address frame (B-layer)

# ── 1. Extract address frame image ────────────────────────────────────────────
print("Step 1: Extracting address frame...")
cap = cv2.VideoCapture(str(VIDEO))
cap.set(cv2.CAP_PROP_POS_FRAMES, ADDR_FR)
ok, addr_img = cap.read()
cap.release()
assert ok, f"Cannot read fr{ADDR_FR} from {VIDEO}"
H_img, W_img = addr_img.shape[:2]
print(f"  address frame: {W_img}x{H_img}")

# save address frame for reference
cv2.imwrite(str(OUT / 'address_frame.jpg'), addr_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

# ── 2. Load RTMPose keypoints for address frame ────────────────────────────────
print("Step 2: Loading RTMPose keypoints...")
kp_data = json.loads(KP_CACHE.read_text())
frames = kp_data['frames'] if 'frames' in kp_data else kp_data

# get address frame keypoints
addr_kps = None
for fr in frames:
    if fr['frame_idx'] == ADDR_FR:
        addr_kps = np.array(fr['keypoints'])  # (17, 3) or (17, 2)
        addr_scores = np.array(fr.get('scores', fr['keypoints'])[:, 2] if addr_kps.shape[1] == 3 else np.ones(17))
        break

if addr_kps is None:
    # fallback: first frame
    fr = frames[0]
    addr_kps = np.array(fr['keypoints'])
    print(f"  [warn] frame {ADDR_FR} not found, using frame 0")

# RTMPose COCO17 keypoints: xy in pixel coords
if addr_kps.shape[1] == 3:
    kp_xy = addr_kps[:, :2]   # (17, 2)
    kp_sc = addr_kps[:, 2]
else:
    kp_xy = addr_kps[:, :2]
    kp_sc = np.ones(17)

print(f"  keypoints shape: {addr_kps.shape}, score range: [{kp_sc.min():.2f}, {kp_sc.max():.2f}]")

# ── 3. Compute bounding box from keypoints (replaces detectron2) ───────────────
print("Step 3: Computing bounding box from RTMPose keypoints...")
valid_mask = kp_sc > 0.3
valid_kps = kp_xy[valid_mask]
x_min, y_min = valid_kps.min(axis=0)
x_max, y_max = valid_kps.max(axis=0)

# expand box by 20% margin
w_box = x_max - x_min
h_box = y_max - y_min
margin_x = w_box * 0.15
margin_y = h_box * 0.15
x_min = max(0, x_min - margin_x)
y_min = max(0, y_min - margin_y)
x_max = min(W_img, x_max + margin_x)
y_max = min(H_img, y_max + margin_y)

box = np.array([[x_min, y_min, x_max, y_max]])
print(f"  box: [{x_min:.0f}, {y_min:.0f}, {x_max:.0f}, {y_max:.0f}]  ({x_max-x_min:.0f}x{y_max-y_min:.0f})")

# ── 4. Run HMR2.0 ─────────────────────────────────────────────────────────────
print("Step 4: Running HMR2.0...")
import torch
from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from hmr2.utils.renderer import Renderer, cam_crop_to_full

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  device: {device}")

t0 = time.time()
model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
model = model.to(device)
model.eval()
print(f"  model loaded in {time.time()-t0:.1f}s")

# feed address frame (BGR→RGB for the dataset)
img_rgb = cv2.cvtColor(addr_img, cv2.COLOR_BGR2RGB)
dataset  = ViTDetDataset(model_cfg, img_rgb, box)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

all_verts  = []
all_cam_t  = []
pred_output = None

t1 = time.time()
for batch in dataloader:
    batch = recursive_to(batch, device)
    with torch.no_grad():
        out = model(batch)

    pred_cam     = out['pred_cam']
    box_center   = batch['box_center'].float()
    box_size     = batch['box_size'].float()
    img_size     = batch['img_size'].float()
    scaled_fl    = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
    cam_t_full   = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_fl).detach().cpu().numpy()

    verts  = out['pred_vertices'][0].detach().cpu().numpy()   # (6890, 3)
    cam_t  = cam_t_full[0]
    all_verts.append(verts)
    all_cam_t.append(cam_t)
    pred_output = out

    print(f"  HMR2 done in {time.time()-t1:.1f}s  |  verts: {verts.shape}")
    break  # single person

# ── 5. Render SMPL mesh (red semi-transparent) ────────────────────────────────
print("Step 5: Rendering SMPL overlay...")
renderer = Renderer(model_cfg, faces=model.smpl.faces)

img_size_hw = torch.tensor([[H_img, W_img]], dtype=torch.float32)

# render_rgba_multiple returns RGBA [0,1] float32 (H, W, 4)
RED_COLOR = (0.85, 0.12, 0.08)   # vivid red
misc_args = dict(
    mesh_base_color=RED_COLOR,
    scene_bg_color=(0.0, 0.0, 0.0),
    focal_length=scaled_fl,
)
cam_view = renderer.render_rgba_multiple(
    all_verts,
    cam_t=all_cam_t,
    render_res=img_size_hw[0],
    **misc_args
)
print(f"  cam_view shape: {cam_view.shape}  alpha range: [{cam_view[:,:,3].min():.2f}, {cam_view[:,:,3].max():.2f}]")

# ── 6. Alpha-composite overlay + white vertical baseline ──────────────────────
print("Step 6: Compositing...")
# base image as float [0,1]
base = addr_img.astype(np.float32)[:, :, ::-1] / 255.0   # BGR→RGB float

# resize cam_view to match base if needed
if cam_view.shape[:2] != base.shape[:2]:
    cam_view = cv2.resize(cam_view, (W_img, H_img), interpolation=cv2.INTER_LINEAR)

alpha       = cam_view[:, :, 3:4]
mesh_rgb    = cam_view[:, :, :3]

# semi-transparent: blend with alpha * 0.75 for see-through effect
ALPHA_SCALE = 0.75
overlay = base * (1 - alpha * ALPHA_SCALE) + mesh_rgb * (alpha * ALPHA_SCALE)
overlay = np.clip(overlay, 0, 1)

# white vertical baseline: through pelvis (mid-hip x-coordinate)
# COCO17: left_hip=11, right_hip=12
lhip_x = kp_xy[11, 0]
rhip_x = kp_xy[12, 0]
pelvis_x = int((lhip_x + rhip_x) / 2)
# draw vertical white line
LINE_THICKNESS = 2
overlay[:, max(0, pelvis_x-LINE_THICKNESS):pelvis_x+LINE_THICKNESS, :] = 1.0

# convert back to BGR uint8
overlay_bgr = (overlay[:, :, ::-1] * 255).clip(0, 255).astype(np.uint8)

# ── 7. Save overlay and side-by-side ──────────────────────────────────────────
overlay_path = OUT / 'smpl_overlay.jpg'
cv2.imwrite(str(overlay_path), overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 93])
print(f"  overlay → {overlay_path}")

# side-by-side: original LEFT | overlay RIGHT
def add_label(img_bgr, text):
    bar = np.zeros((44, img_bgr.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (220, 220, 220), 2)
    return np.vstack([bar, img_bgr])

sbs = np.hstack([
    add_label(addr_img,    'REAL  address'),
    add_label(overlay_bgr, 'SMPL  overlay (red semi-transparent)'),
])
sbs_path = OUT / 'side_by_side.jpg'
cv2.imwrite(str(sbs_path), sbs, [cv2.IMWRITE_JPEG_QUALITY, 93])
print(f"  side_by_side → {sbs_path}")

# ── 8. Runtime log ────────────────────────────────────────────────────────────
total_time = time.time() - t0
peak_vram  = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

log = {
    'frame':        'address (fr0, B-layer)',
    'input_res':    f'{W_img}x{H_img}',
    'box_px':       [float(x_min), float(y_min), float(x_max), float(y_max)],
    'box_source':   'RTMPose kp_guard-verified keypoints (no detectron2)',
    'total_time_s': round(total_time, 1),
    'peak_vram_gb': round(peak_vram, 2),
    'alpha_scale':  ALPHA_SCALE,
    'mesh_color_rgb': RED_COLOR,
    'pelvis_x_px':  pelvis_x,
    'fit_quality':  'PENDING_HUMAN_REVIEW',
    'deviation_notes': {
        'head':   'PENDING_HUMAN_REVIEW',
        'shoulder': 'PENDING_HUMAN_REVIEW',
        'hip':    'PENDING_HUMAN_REVIEW',
        'knee':   'PENDING_HUMAN_REVIEW',
        'ankle':  'PENDING_HUMAN_REVIEW',
        'shape_match': 'PENDING_HUMAN_REVIEW',
    }
}
(OUT / 'run_log.json').write_text(json.dumps(log, indent=2, ensure_ascii=False))
print(f"  run_log → {OUT / 'run_log.json'}")

# copy to Windows
import shutil
WIN_OUT = pathlib.Path('/mnt/c/Users/jason/Desktop/rtmpose_results/preview/ghost002')
WIN_OUT.mkdir(parents=True, exist_ok=True)
for f in OUT.glob('*.jpg'):
    shutil.copy2(f, WIN_OUT / f.name)
for f in OUT.glob('*.json'):
    shutil.copy2(f, WIN_OUT / f.name)
print(f"  copied to Windows: {WIN_OUT}")

print(f"\n[DONE] total={total_time:.1f}s  peak_vram={peak_vram:.2f}GB")
print(f"Outputs:")
print(f"  {overlay_path}")
print(f"  {sbs_path}")
