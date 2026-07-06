"""
ghost001_run.py  — GHOST-001 T1 MimicMotion 推理
严格对齐官方 inference.py API

前置：
  1. SVD 已下载（gated，需 HF token）
  2. MimicMotion_1-1.pth: /home/jason/projects/mimicmotion/models/MimicMotion_1-1.pth
  3. DWPose: /home/jason/projects/mimicmotion/models/DWPose/

用法：
  cd /home/jason/projects/swingcue-postest
  source .venv/bin/activate
  python ghost001_run.py [--svd-path /local/path/to/svd]

输出：
  output/ghost001/generated.mp4         生成片段
  output/ghost001/compare_P1.jpg        P1 并排拼图 (address)
  output/ghost001/compare_P3.jpg        P3 并排拼图 (impact)
  output/ghost001/compare_P4.jpg        P4 并排拼图 (finish)
  output/ghost001/side_by_side.mp4      原片 vs 生成同步对比视频
  output/ghost001/run_log.json          显存峰值 / 耗时 / 伪影占位记录
"""
import os, sys, math, time, json, argparse
import pathlib
import numpy as np
import cv2

ROOT       = pathlib.Path('/home/jason/projects/swingcue-postest')
MIMIC_ROOT = pathlib.Path('/home/jason/projects/mimicmotion')
OUT        = ROOT / 'output/ghost001'
OUT.mkdir(parents=True, exist_ok=True)

# ── mimicmotion on sys.path ────────────────────────────────────────────────────
sys.path.insert(0, str(MIMIC_ROOT))

# ── CLI args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--svd-path',
                    default='stabilityai/stable-video-diffusion-img2vid-xt-1-1',
                    help='Local dir or HF repo-id for SVD-XT-1.1')
parser.add_argument('--resolution',  type=int, default=576)
parser.add_argument('--num-frames',  type=int, default=72)
parser.add_argument('--tile-size',   type=int, default=16,
                    help='Tile size for pipeline (lower = less VRAM)')
parser.add_argument('--tile-overlap',type=int, default=6)
parser.add_argument('--num-steps',   type=int, default=25)
parser.add_argument('--guidance',    type=float, default=2.0)
parser.add_argument('--fps',         type=int, default=15)
parser.add_argument('--seed',        type=int, default=42)
parser.add_argument('--sample-stride', type=int, default=1,
                    help='Pose sampling stride for drive video')
parser.add_argument('--cpu-offload', action='store_true', default=True,
                    help='Enable model CPU offload (needed for 8GB VRAM)')
args = parser.parse_args()

# ── phase map (B-layer 8-phase system, from ghost001_prep.py) ────────────────
# B层正式相位: address/takeaway/backswing/top/transition/downswing/impact/follow_through
PHASE_DS_IDX = {
    'address':        0,
    'top':            65,
    'impact':         69,
    'follow_through': 71,
}
COMPARE_PHASES = ['address', 'impact', 'follow_through']  # three compare frames

# ── imports ────────────────────────────────────────────────────────────────────
import torch
from omegaconf import OmegaConf
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import pil_to_tensor, resize, center_crop, to_pil_image
from mimicmotion.utils.geglu_patch import patch_geglu_inplace
patch_geglu_inplace()
from mimicmotion.utils.loader import create_pipeline
from mimicmotion.utils.utils import save_to_mp4
from mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose
from constants import ASPECT_RATIO

torch.set_default_dtype(torch.float16)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU:  {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ── load pipeline ──────────────────────────────────────────────────────────────
infer_config = OmegaConf.create({
    'base_model_path': args.svd_path,
    'ckpt_path': str(MIMIC_ROOT / 'models/MimicMotion_1-1.pth'),
})

print("\nLoading pipeline...")
t0 = time.time()
pipeline = create_pipeline(infer_config, device)

if args.cpu_offload:
    # Offload image_encoder -> unet -> vae sequentially
    # model_cpu_offload_seq is already set in MimicMotionPipeline
    pipeline.enable_model_cpu_offload()
    print("CPU offload: ENABLED (image_encoder->unet->vae)")
else:
    pipeline.to(device)
    print("CPU offload: DISABLED (full GPU mode)")

print(f"Pipeline ready in {time.time()-t0:.1f}s")
torch.cuda.reset_peak_memory_stats()

# ── preprocess: exactly as in inference.py ────────────────────────────────────
def preprocess(video_path, image_path, resolution=576, sample_stride=1):
    image_pixels = pil_loader(str(image_path))
    image_pixels = pil_to_tensor(image_pixels)  # (C, H, W)
    h, w = image_pixels.shape[-2:]
    # compute target size
    if h > w:
        w_target, h_target = resolution, int(resolution / ASPECT_RATIO // 64) * 64
    else:
        w_target, h_target = int(resolution / ASPECT_RATIO // 64) * 64, resolution
    h_w_ratio = float(h) / float(w)
    if h_w_ratio < h_target / w_target:
        h_resize, w_resize = h_target, math.ceil(h_target / h_w_ratio)
    else:
        h_resize, w_resize = math.ceil(w_target * h_w_ratio), w_target
    image_pixels = resize(image_pixels, [h_resize, w_resize], antialias=None)
    image_pixels = center_crop(image_pixels, [h_target, w_target])
    image_pixels = image_pixels.permute((1, 2, 0)).numpy()
    # get poses
    image_pose = get_image_pose(image_pixels)
    video_pose = get_video_pose(str(video_path), image_pixels, sample_stride=sample_stride)
    pose_pixels  = np.concatenate([np.expand_dims(image_pose, 0), video_pose])
    image_pixels = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))
    return (torch.from_numpy(pose_pixels.copy()) / 127.5 - 1,
            torch.from_numpy(image_pixels) / 127.5 - 1)

ref_image_path   = OUT / 'address_ref.jpg'
drive_video_path = OUT / 'drive_segment.mp4'

print(f"\nPreprocessing (DWPose)...")
t1 = time.time()
pose_pixels, image_pixels = preprocess(
    drive_video_path, ref_image_path,
    resolution=args.resolution,
    sample_stride=args.sample_stride,
)
print(f"DWPose done in {time.time()-t1:.1f}s")
print(f"  pose_pixels:  {tuple(pose_pixels.shape)}")
print(f"  image_pixels: {tuple(image_pixels.shape)}")

# ── run pipeline: exactly as in inference.py ──────────────────────────────────
print(f"\nRunning MimicMotion (steps={args.num_steps}, res={args.resolution}, "
      f"tile={args.tile_size}, frames={pose_pixels.size(0)})...")
t2 = time.time()

image_pil_list = [to_pil_image(img.to(torch.uint8)) for img in (image_pixels + 1.0) * 127.5]
generator = torch.Generator(device=device)
generator.manual_seed(args.seed)

frames_tensor = pipeline(
    image_pil_list,
    image_pose=pose_pixels,
    num_frames=pose_pixels.size(0),
    tile_size=args.tile_size,
    tile_overlap=args.tile_overlap,
    height=pose_pixels.shape[-2],
    width=pose_pixels.shape[-1],
    fps=7,
    noise_aug_strength=0,
    num_inference_steps=args.num_steps,
    generator=generator,
    min_guidance_scale=args.guidance,
    max_guidance_scale=args.guidance,
    decode_chunk_size=8,
    output_type='pt',
    device=device,
).frames.cpu()

infer_time = time.time() - t2
peak_vram  = torch.cuda.max_memory_allocated() / 1e9
print(f"\nInference done: {infer_time:.1f}s  |  peak VRAM: {peak_vram:.2f} GB")

# video_frames: (B, T, C, H, W) uint8 — skip first frame (ref image)
video_frames = (frames_tensor * 255.0).to(torch.uint8)
_video_frames = video_frames[0, 1:]  # drop ref frame; shape (T-1, C, H, W)
T = _video_frames.shape[0]
H_out = _video_frames.shape[2]
W_out = _video_frames.shape[3]
print(f"Generated {T} frames  tensor shape (T,C,H,W) = {tuple(_video_frames.shape)}")
print(f"Output resolution: W={W_out} x H={H_out}  ({'portrait' if H_out > W_out else 'landscape'})")

# ── STEP 1: 落盘帧序列 PNG（防编码失败重跑推理）─────────────────────────────
# 任何后续编码/拼图失败都不需要重跑推理，直接从 frames/ 恢复
FRAMES_DIR = OUT / 'frames'
FRAMES_DIR.mkdir(exist_ok=True)
print(f"\nSaving {T} frames to {FRAMES_DIR} ...")
gen_bgr = []
for i in range(T):
    fr_np = _video_frames[i].permute(1, 2, 0).numpy()   # H W C, RGB uint8
    fr_bgr = cv2.cvtColor(fr_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(FRAMES_DIR / f'frame_{i:04d}.png'), fr_bgr)
    gen_bgr.append(fr_bgr)
print(f"Frames saved → {FRAMES_DIR}")

# ── STEP 2: 编码 generated.mp4 ────────────────────────────────────────────────
gen_path = OUT / 'generated.mp4'
fourcc_gen = cv2.VideoWriter_fourcc(*'mp4v')
vw_gen = cv2.VideoWriter(str(gen_path), fourcc_gen, float(args.fps), (W_out, H_out))
for fr_bgr in gen_bgr:
    vw_gen.write(fr_bgr)
vw_gen.release()
print(f"Generated → {gen_path}")

# ── read original drive frames ─────────────────────────────────────────────────
cap = cv2.VideoCapture(str(drive_video_path))
orig_frames = []
while True:
    ok, fr = cap.read()
    if not ok: break
    orig_frames.append(fr)
cap.release()

def resize_to(fr, target_h, target_w):
    return cv2.resize(fr, (target_w, target_h))

def add_label(img, text):
    bar = np.zeros((44, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (220, 220, 220), 2)
    return np.vstack([bar, img])

# ── three compare images ───────────────────────────────────────────────────────
# Generated T frames = drive_segment frames - 1 (ref frame skipped)
# ds_idx maps to generated index: gen_idx = ds_idx - 1 (or clamped to 0)
for ph_name in COMPARE_PHASES:
    ds_orig = PHASE_DS_IDX[ph_name]
    gen_idx = max(0, ds_orig - 1)   # skip ref frame offset
    if ds_orig >= len(orig_frames) or gen_idx >= len(gen_bgr):
        print(f"[warn] {ph_name}: ds={ds_orig} gen={gen_idx} out of range (orig={len(orig_frames)}, gen={len(gen_bgr)})")
        continue
    orig_fr = resize_to(orig_frames[ds_orig], H_out, W_out)
    gen_fr  = gen_bgr[gen_idx]
    compare = np.hstack([
        add_label(orig_fr, f'REAL  {ph_name}'),
        add_label(gen_fr,  f'GHOST {ph_name}'),
    ])
    tag = ph_name.split('_')[0]  # P1 / P3 / P4
    save_path = OUT / f'compare_{tag}.jpg'
    cv2.imwrite(str(save_path), compare, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Compare → {save_path}")

# ── side-by-side sync video ────────────────────────────────────────────────────
sbs_path = OUT / 'side_by_side.mp4'
n_sync   = min(len(orig_frames) - 1, len(gen_bgr))   # -1 for ref frame
fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
sbs_vw   = cv2.VideoWriter(str(sbs_path), fourcc, float(args.fps), (W_out * 2, H_out))
for i in range(n_sync):
    orig_fr = resize_to(orig_frames[i + 1], H_out, W_out)   # +1 skip ref
    gen_fr  = gen_bgr[i]
    sbs_vw.write(np.hstack([orig_fr, gen_fr]))
sbs_vw.release()
print(f"Side-by-side → {sbs_path}")

# ── copy to Windows desktop ────────────────────────────────────────────────────
WIN_OUT = pathlib.Path('/mnt/c/Users/jason/Desktop/rtmpose_results/preview/ghost001')
WIN_OUT.mkdir(parents=True, exist_ok=True)
import shutil
for f in OUT.glob('*.jpg'): shutil.copy2(f, WIN_OUT / f.name)
for f in OUT.glob('*.mp4'): shutil.copy2(f, WIN_OUT / f.name)
print(f"Copied to Windows: {WIN_OUT}")

# ── run log ────────────────────────────────────────────────────────────────────
log = {
    'infer_time_s':  round(infer_time, 1),
    'peak_vram_gb':  round(peak_vram, 2),
    'num_frames_generated': T,
    'resolution':    args.resolution,
    'num_steps':     args.num_steps,
    'tile_size':     args.tile_size,
    'cpu_offload':   args.cpu_offload,
    'phase_ds_idx':  PHASE_DS_IDX,
    'artifact_record': {
        'club':  {'P1': 'PENDING_HUMAN_REVIEW', 'P3': 'PENDING_HUMAN_REVIEW', 'P4': 'PENDING_HUMAN_REVIEW'},
        'hands': {'P1': 'PENDING_HUMAN_REVIEW', 'P3': 'PENDING_HUMAN_REVIEW', 'P4': 'PENDING_HUMAN_REVIEW'},
        'face':  {'P1': 'PENDING_HUMAN_REVIEW', 'P3': 'PENDING_HUMAN_REVIEW', 'P4': 'PENDING_HUMAN_REVIEW'},
    },
    'note': '伪影记录待 Jason 目视后填入; 球杆表现仅记录不判定'
}
(OUT / 'run_log.json').write_text(json.dumps(log, indent=2, ensure_ascii=False))
print(f"\nRun log → {OUT/'run_log.json'}")
print(f"\n[DONE] peak_vram={peak_vram:.2f}GB  time={infer_time:.0f}s")
