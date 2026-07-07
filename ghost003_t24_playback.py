"""
ghost003_t24_playback.py  —  GHOST-003 T2.4
闪动帧处理 + 全段重新合成 playback_t24.mp4

策略:
  fr000–083: 正常帧逐帧推理 + T2.3 sx 拟合
  fr084–095: FLICKER SEGMENT — 用 fr083/fr096 两锚帧线性插值 mesh
  fr096–111: 正常帧逐帧推理 + T2.3 sx 拟合
  fr087/094 (T2.2 invalid): 已在 FLICKER SEGMENT 内，自动被插值覆盖

锚帧插值公式:
  weight(fi) = (fi - 83) / (96 - 83)  # 0→1 across segment
  verts(fi)  = (1-w)*verts_083 + w*verts_096
  cam_t(fi)  = (1-w)*cam_t_083 + w*cam_t_096
  focal(fi)  = (1-w)*focal_083 + w*focal_096
"""
import os, sys, json, time
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.1"

import numpy as np
import cv2
from pathlib import Path
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/jason/projects/sam-3d-body")
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from sam_3d_body.visualization.renderer import Renderer

# ── 路径 ──────────────────────────────────────────────────────────────────────
VIDEO_IN   = Path("/home/jason/projects/swingcue-postest/input/fo-ok-1.mp4")
KP_CACHE   = Path("/home/jason/projects/swingcue-postest/engine/kp_cache/batch2/fo-ok-1.json")
RUN_LOG    = Path("/home/jason/projects/swingcue-postest/output/ghost003_t23/run_log_t23.json")
OUTPUT_DIR = Path("/home/jason/projects/swingcue-postest/output/ghost003_t24")
WIN_OUT    = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/ghost003_t24")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WIN_OUT.mkdir(parents=True, exist_ok=True)

CKPT   = Path(os.path.expanduser("~/.cache/sam3d/sam-3d-body-dinov3/model.ckpt"))
MHR_PT = CKPT.parent / "assets/mhr_model.pt"

NF  = 112; H = 1280; W = 720
FPS = 30.0

# Flicker segment (closed interval, both endpoints are anchor frames rendered normally)
FLICKER_LO  = 84   # first flicker frame
FLICKER_HI  = 95   # last  flicker frame
ANCHOR_PRE  = 83   # last stable frame before segment
ANCHOR_POST = 96   # first stable frame after segment

# Ghost render params
GHOST_ALPHA = 0.55
GHOST_COLOR = (0.85, 0.1, 0.1)   # RGB

# ── KP 解析 ──────────────────────────────────────────────────────────────────
KP_ORDER = ["nose","left_eye","right_eye","left_ear","right_ear",
            "left_shoulder","right_shoulder","left_elbow","right_elbow",
            "left_wrist","right_wrist","left_hip","right_hip",
            "left_knee","right_knee","left_ankle","right_ankle"]

def load_kp_cache(path):
    with open(path) as f: d = json.load(f)
    out = {}
    for fe in d["frames"]:
        fi = fe["frame"]
        if not fe.get("persons"): continue
        kd = fe["persons"][0]["keypoints"]
        out[fi] = np.array([[kd[n]["x"], kd[n]["y"], kd[n]["score"]]
                             for n in KP_ORDER if n in kd], dtype=np.float32)
    return out

def load_t23_params(path):
    with open(path) as f: d = json.load(f)
    return {r["frame"]: r for r in d["per_frame"]}

def get_bbox(kp2d, H, W, pad=0.15):
    ax=kp2d[:,0]; ay=kp2d[:,1]
    px=(max(ax)-min(ax))*pad; py=(max(ay)-min(ay))*pad
    return np.array([[max(0,min(ax)-px),max(0,min(ay)-py),
                      min(W,max(ax)+px),min(H,max(ay)+py)]],dtype=np.float32)

def proj2d(verts, cam_t, focal, H, W):
    vx=verts[:,0]; vy=verts[:,1]; vz=verts[:,2]
    d=vz+cam_t[2]
    px=focal*(vx-cam_t[0])/d+W/2
    py=focal*(vy+cam_t[1])/d+H/2
    return np.stack([px,py],axis=1)

# ── sx 拟合 (T1.7 logic, replicated from T2.3) ───────────────────────────────
I_NOSE=0; I_LSHO=5; I_RSHO=6; I_LHIP=11; I_RHIP=12
I_LKNE=13; I_RKNE=14; I_LANK=15; I_RANK=16; I_NECK=69

def apply_t23_sx(verts, cam_t, focal, kp2d, p, H, W):
    """Apply T2.3 stored sx params to verts. Returns adjusted verts."""
    v = verts.copy()
    vp = proj2d(v, cam_t, focal, H, W)

    nose_y = kp2d[I_NOSE][1]
    hip_y  = (kp2d[I_LHIP][1]+kp2d[I_RHIP][1])/2
    ank_y  = (kp2d[I_LANK][1]+kp2d[I_RANK][1])/2
    sho_y  = (kp2d[I_LSHO][1]+kp2d[I_RSHO][1])/2

    UP_lo  = max(0,   int(nose_y-30))
    UP_hi  = min(H,   int(hip_y+60))
    HIP_lo = max(0,   int(hip_y-40))
    HIP_hi = min(H,   int(ank_y-40))
    LOW_lo = HIP_hi;  LOW_hi = H

    def scale_band(v_in, vp_in, y_lo, y_hi, sx, dx_px=0.0):
        if abs(sx-1.0) < 1e-4 and abs(dx_px) < 0.5:
            return v_in
        bm = (vp_in[:,1]>=y_lo) & (vp_in[:,1]<=y_hi)
        if not np.any(bm):
            return v_in
        v_out = v_in.copy()
        cx3d  = float(np.median(v_out[bm,0]))
        v_out[bm,0] = cx3d + sx*(v_out[bm,0]-cx3d)
        if abs(dx_px)>0.5:
            depth = float(np.mean(np.abs(v_out[bm,2]+cam_t[2])))
            v_out[bm,0] += dx_px * depth / focal
        return v_out

    v = scale_band(v, vp, UP_lo,  UP_hi,  p.get("sx_upper",1.0), p.get("dx_upper",0.0))
    vp= proj2d(v, cam_t, focal, H, W)
    v = scale_band(v, vp, HIP_lo, HIP_hi, p.get("sx_hip",1.0),   p.get("dx_hip",0.0))
    vp= proj2d(v, cam_t, focal, H, W)
    v = scale_band(v, vp, LOW_lo, LOW_hi, p.get("sx_lower",1.0))

    # Arm band
    arm_sx = p.get("arm_sx",1.0)
    if p.get("arm_applied") and abs(arm_sx-1.0)>1e-4:
        vp = proj2d(v, cam_t, focal, H, W)
        v = scale_band(v, vp, max(0,int(sho_y-20)), UP_hi, arm_sx)

    return v

# ── 渲染 ─────────────────────────────────────────────────────────────────────
def render_and_blend(verts, cam_t, focal, faces, bg_frame, alpha=GHOST_ALPHA):
    """Render ghost mesh and blend onto bg_frame (BGR). Returns BGR uint8."""
    black_bg = np.zeros((H, W, 3), dtype=np.uint8)
    rend = Renderer(focal_length=focal, faces=faces)
    out  = rend(verts, cam_t, black_bg,
                mesh_base_color=GHOST_COLOR, scene_bg_color=(0,0,0))
    # out: H×W×3 float [0,1] BGR
    out_u8 = (out*255).clip(0,255).astype(np.uint8)
    mask   = np.any(out_u8>5, axis=2)
    result = bg_frame.copy().astype(np.float32)
    result[mask] = (1-alpha)*result[mask] + alpha*out_u8[mask].astype(np.float32)
    return result.astype(np.uint8)

# ── 加载 ─────────────────────────────────────────────────────────────────────
print("[INIT] Loading model...")
t0=time.time()
device="cuda" if torch.cuda.is_available() else "cpu"
model, model_cfg = load_sam_3d_body(str(CKPT), device=device, mhr_path=str(MHR_PT))
estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=model_cfg,
                                human_detector=None, human_segmentor=None,
                                fov_estimator=None)
estimator.model.eval()
faces = estimator.faces
print(f"  {time.time()-t0:.1f}s  faces={len(faces)}")

print("[INIT] Reading inputs...")
cap=cv2.VideoCapture(str(VIDEO_IN))
actual_fps=cap.get(cv2.CAP_PROP_FPS)
src=[]
while True:
    ret,f=cap.read()
    if not ret: break
    src.append(f)
cap.release()
print(f"  {len(src)} frames @ {actual_fps:.2f}fps")

kp_frames  = load_kp_cache(KP_CACHE)
t23_params = load_t23_params(RUN_LOG)

# ── 单帧完整推理 + sx 拟合 ────────────────────────────────────────────────────
def infer_frame(fi):
    """Full MHR inference + T2.3 sx. Returns (verts_adjusted, cam_t, focal, kp2d_mhr)."""
    kp = kp_frames.get(fi)
    if kp is None:
        return None
    bbox = get_bbox(kp[:,:2], H, W)
    with torch.no_grad():
        outs = estimator.process_one_image(src[fi], bboxes=bbox,
                                           use_mask=False, inference_type="body")
    if not outs:
        return None
    o     = outs[0]
    verts = o["pred_vertices"].astype(np.float32)
    cam_t = o["pred_cam_t"].astype(np.float32)
    focal = float(o["focal_length"])
    kp2d  = o["pred_keypoints_2d"].astype(np.float32)

    p = t23_params.get(fi, {})
    verts_adj = apply_t23_sx(verts, cam_t, focal, kp2d, p, H, W)
    return verts_adj, cam_t, focal, kp2d

# ── Step 1: 推理两个锚帧 ──────────────────────────────────────────────────────
print(f"\n[ANCHOR] Inferring anchor frames fr{ANCHOR_PRE:03d} and fr{ANCHOR_POST:03d}...")
anc_pre  = infer_frame(ANCHOR_PRE)
anc_post = infer_frame(ANCHOR_POST)
assert anc_pre  is not None, f"anchor fr{ANCHOR_PRE} failed"
assert anc_post is not None, f"anchor fr{ANCHOR_POST} failed"
verts_pre,  cam_t_pre,  focal_pre,  kp2d_pre  = anc_pre
verts_post, cam_t_post, focal_post, kp2d_post = anc_post
print(f"  fr{ANCHOR_PRE:03d}: cam_t={cam_t_pre.tolist()}  focal={focal_pre:.1f}")
print(f"  fr{ANCHOR_POST:03d}: cam_t={cam_t_post.tolist()}  focal={focal_post:.1f}")

# ── Step 2: 全段渲染 ──────────────────────────────────────────────────────────
print(f"\n[RENDER] Processing all {NF} frames...")
out_path = OUTPUT_DIR / "playback_t24_raw.mp4"
fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
writer   = cv2.VideoWriter(str(out_path), fourcc, actual_fps, (W, H))

# Track mesh_cx for comparison plot
mesh_cx_t23 = []   # from probe (approx, using raw verts_pre interp for flicker)
mesh_cx_t24 = []
frame_ids   = []

t_start = time.time()
for fi in range(NF):
    is_flicker = (FLICKER_LO <= fi <= FLICKER_HI)

    if is_flicker:
        # Cross-segment linear interpolation
        w = (fi - ANCHOR_PRE) / (ANCHOR_POST - ANCHOR_PRE)   # 0 < w < 1
        verts_interp = (1-w)*verts_pre  + w*verts_post
        cam_t_interp = (1-w)*cam_t_pre  + w*cam_t_post
        focal_interp = (1-w)*focal_pre  + w*focal_post

        frame_out = render_and_blend(verts_interp, cam_t_interp, focal_interp,
                                     faces, src[fi])
        # cx from interpolated proj
        vp_i = proj2d(verts_interp, cam_t_interp, focal_interp, H, W)
        mc   = float(np.median(vp_i[:,0]))
    else:
        result = infer_frame(fi)
        if result is None:
            frame_out = src[fi].copy()
            mc = W/2
        else:
            verts_adj, cam_t, focal, kp2d = result
            frame_out = render_and_blend(verts_adj, cam_t, focal, faces, src[fi])
            vp = proj2d(verts_adj, cam_t, focal, H, W)
            mc = float(np.median(vp[:,0]))

    writer.write(frame_out)
    mesh_cx_t24.append(mc)
    frame_ids.append(fi)

    if fi % 10 == 0:
        elapsed = time.time()-t_start
        eta     = elapsed/(fi+1)*(NF-fi-1) if fi>0 else 0
        mode    = "INTERP" if is_flicker else "infer"
        print(f"  fr{fi:03d} [{mode}]  mesh_cx={mc:.1f}  elapsed={elapsed:.1f}s  ETA={eta:.0f}s")

writer.release()
print(f"\n  raw video written: {time.time()-t_start:.1f}s")

# ── Step 3: re-encode ─────────────────────────────────────────────────────────
enc_path = OUTPUT_DIR / "playback_t24.mp4"
os.system(f"ffmpeg -y -i {out_path} -vcodec libx264 -crf 20 -pix_fmt yuv420p {enc_path} 2>/dev/null")
if enc_path.exists() and enc_path.stat().st_size > 0:
    out_path.unlink()
    print(f"  re-encoded → {enc_path}")
else:
    enc_path = out_path
    print(f"  ffmpeg failed, using raw mp4")

# ── Step 4: comparison plot ───────────────────────────────────────────────────
# Load T2.3 probe mesh_cx for reference
probe_json = Path("/home/jason/projects/swingcue-postest/output/ghost003_t24/probe_t24.json")
mesh_cx_t23_plot = {}
if probe_json.exists():
    with open(probe_json) as f: pd = json.load(f)
    for r in pd["per_frame"]:
        if not r.get("invalid") and "mesh_cx" in r:
            mesh_cx_t23_plot[r["frame"]] = r["mesh_cx"]

# Human hip cx from kp_cache
human_cx_plot = {}
for fi, kp in kp_frames.items():
    human_cx_plot[fi] = float((kp[11,0]+kp[12,0])/2)

frames_arr  = np.array(frame_ids)
cx24_arr    = np.array(mesh_cx_t24)
cx23_arr    = np.array([mesh_cx_t23_plot.get(fi, np.nan) for fi in frame_ids])
hcx_arr     = np.array([human_cx_plot.get(fi, np.nan) for fi in frame_ids])

fig, axes = plt.subplots(2,1,figsize=(14,8))
fig.suptitle("T2.4 Flicker Fix — mesh_cx before vs after (orange=flicker segment)", fontsize=12)

ax=axes[0]
ax.plot(frames_arr, cx23_arr, 'r-',  lw=1.2, alpha=0.7, label="T2.3 raw (flicker)")
ax.plot(frames_arr, cx24_arr, 'b-',  lw=1.5, label="T2.4 fixed (interp)")
ax.plot(frames_arr, hcx_arr,  'g--', lw=1.0, alpha=0.8, label="human_hip_cx (RTMPose)")
ax.axvspan(FLICKER_LO, FLICKER_HI, alpha=0.12, color='orange', label="flicker segment")
ax.axvline(ANCHOR_PRE,  color='gray', ls=':', lw=1)
ax.axvline(ANCHOR_POST, color='gray', ls=':', lw=1)
ax.set_ylabel("mesh_cx (px)"); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
ax.set_title("mesh_cx: T2.3 vs T2.4 vs human")

# frame-to-frame jump: only T2.4
ax=axes[1]
d_cx24 = np.abs(np.diff(cx24_arr))
d_cx23 = np.abs(np.diff(cx23_arr))
d_hum  = np.abs(np.diff(hcx_arr))
jump_f = frames_arr[1:]
ax.plot(jump_f, d_cx23, 'r-', lw=1, alpha=0.7, label="Δmesh_cx T2.3")
ax.plot(jump_f, d_cx24, 'b-', lw=1.5, label="Δmesh_cx T2.4")
ax.plot(jump_f, d_hum,  'g--',lw=1, alpha=0.7, label="Δhuman_cx")
ax.axvspan(FLICKER_LO, FLICKER_HI, alpha=0.12, color='orange')
ax.axhline(15.0, color='orange', ls=':', lw=1, label="thr=15px")
ax.set_ylabel("Δpx"); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
ax.set_title("Frame-to-frame mesh_cx jump: before vs after fix")
ax.set_xlabel("frame")

plt.tight_layout()
plot_p = OUTPUT_DIR / "cx_comparison_t24.jpg"
plt.savefig(str(plot_p), dpi=120, bbox_inches='tight')
plt.close()
print(f"  plot → {plot_p}")

# ── Step 5: report + copy to Windows ─────────────────────────────────────────
flicker_frames = list(range(FLICKER_LO, FLICKER_HI+1))

report_lines = [
    "GHOST-003 T2.4 停关卡报告",
    f"Clip: fo-ok-1  NF={NF}",
    "",
    "=== 闪动帧处理 ===",
    f"探针标记候选帧: fr084/085/086/088/089/090/091/092/093/095",
    f"+ T2.2 原有无效帧: fr087/094 (已在段内)",
    f"最终闪动段: fr{FLICKER_LO:03d}–fr{FLICKER_HI:03d} ({FLICKER_HI-FLICKER_LO+1} 帧)",
    "",
    "=== 插值方式 ===",
    f"锚帧 A: fr{ANCHOR_PRE:03d} (段前最后稳定帧, T2.3 sx 拟合)",
    f"锚帧 B: fr{ANCHOR_POST:03d} (段后第一稳定帧, T2.3 sx 拟合)",
    f"插值: verts(fi) = (1-w)*verts_083 + w*verts_096",
    f"       cam_t(fi) = (1-w)*cam_t_083 + w*cam_t_096",
    f"       weight w  = (fi-83)/(96-83)",
    f"目的: mesh 在高速段平滑过渡, 不抖动, 不跳变",
    "",
    "=== 段内帧列表 ===",
]
for fi in flicker_frames:
    if fi in [87, 94]:
        report_lines.append(f"  fr{fi:03d}  T2.2 invalid (A1/A4 哨兵) + 插值覆盖")
    else:
        w = (fi - ANCHOR_PRE) / (ANCHOR_POST - ANCHOR_PRE)
        report_lines.append(f"  fr{fi:03d}  flicker  w={w:.3f}  (探针时序跳变超阈值)")

report_lines += [
    "",
    "=== 阶段分布 ===",
    "  downswing  (fr084–088): 5 帧",
    "  follow_through (fr089–095): 7 帧",
    "",
    "=== 关键帧保护 ===",
    f"address fr000: 正常渲染, 不受影响",
    f"top     fr097: 正常渲染, 不受影响",
    f"impact  fr088: 段内 (闪动段), 已用插值覆盖",
    "  → impact 帧 IoU 不再独立统计 (插值帧不参与 IoU)",
    "",
    "=== 验收 ===",
    "playback_t24.mp4 → 看 downswing→impact 段 (fr084–095)",
    "红人应平滑跟随, 无跳变, 无重影",
    "主体其余部分严丝合缝不受影响",
]
rp = OUTPUT_DIR / "REPORT_T24.txt"
rp.write_text("\n".join(report_lines), encoding="utf-8")

import shutil
shutil.copy2(str(enc_path),  str(WIN_OUT / "playback_t24.mp4"))
shutil.copy2(str(plot_p),    str(WIN_OUT / "cx_comparison_t24.jpg"))
shutil.copy2(str(rp),        str(WIN_OUT / "REPORT_T24.txt"))
print(f"\n[OUT] Windows:")
print(f"  {WIN_OUT}/playback_t24.mp4")
print(f"  {WIN_OUT}/cx_comparison_t24.jpg")
print(f"  {WIN_OUT}/REPORT_T24.txt")

tt = time.time()-t_start
print(f"\n{'='*60}")
print(f"T2.4 DONE  NF={NF}  flicker_segment=fr{FLICKER_LO}-{FLICKER_HI}({FLICKER_HI-FLICKER_LO+1}帧)")
print(f"total: {tt:.1f}s")
print(f"{'='*60}")
