"""
ghost003_t25_playback.py  —  GHOST-003 T2.5
修正 T2.4 问题: 全段保留真实姿态(verts), 只平滑位置抖动(cam_t x/y, focal)

策略:
  1. 全段逐帧 MHR 推理, 取原始 verts + cam_t + focal
  2. 对 cam_t[:,0] (x) / cam_t[:,1] (y) / focal 做 Savitzky-Golay 时序平滑
     窗口=9帧, poly=3 (保留曲率, 压制突变)
  3. 每帧用原始 verts (真实姿态) + 平滑后的 cam_t/focal 渲染
  4. fr087/fr094 (T2.2 哨兵判为真崩帧): verts 也做前后帧平均插值
  5. T2.3 sx 拟合在平滑 cam_t 下重新执行 (用 t23_params 缓存 sx 做近似)

输出:
  playback_t25.mp4
  cam_t_smooth_t25.jpg  (平滑前后 cam_t x/y/focal 曲线对比)
  REPORT_T25.txt
"""
import os, sys, json, time
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.1"

import numpy as np
import cv2
from pathlib import Path
import torch
from scipy.signal import savgol_filter
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
OUTPUT_DIR = Path("/home/jason/projects/swingcue-postest/output/ghost003_t25")
WIN_OUT    = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/ghost003_t25")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WIN_OUT.mkdir(parents=True, exist_ok=True)

CKPT   = Path(os.path.expanduser("~/.cache/sam3d/sam-3d-body-dinov3/model.ckpt"))
MHR_PT = CKPT.parent / "assets/mhr_model.pt"

NF  = 112; H = 1280; W = 720

# T2.2 哨兵判为真崩帧: verts 也做插值
HARD_INVALID = {87, 94}

# Savitzky-Golay 平滑参数
SG_WINDOW = 9    # 奇数, 覆盖约 ±4 帧
SG_POLY   = 3    # 多项式阶数

# Ghost render params
GHOST_ALPHA = 0.55
GHOST_COLOR = (0.85, 0.1, 0.1)

# ── 关节索引 ─────────────────────────────────────────────────────────────────
I_NOSE=0; I_LSHO=5; I_RSHO=6; I_LHIP=11; I_RHIP=12
I_LKNE=13; I_RKNE=14; I_LANK=15; I_RANK=16

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

# ── T2.3 sx 拟合 (轻量版) ────────────────────────────────────────────────────
def apply_t23_sx(verts, cam_t, focal, kp2d, p, H, W):
    v = verts.copy()
    vp= proj2d(v, cam_t, focal, H, W)

    nose_y = kp2d[I_NOSE][1]
    hip_y  = (kp2d[I_LHIP][1]+kp2d[I_RHIP][1])/2
    ank_y  = (kp2d[I_LANK][1]+kp2d[I_RANK][1])/2
    sho_y  = (kp2d[I_LSHO][1]+kp2d[I_RSHO][1])/2

    UP_lo  = max(0, int(nose_y-30));  UP_hi  = min(H, int(hip_y+60))
    HIP_lo = max(0, int(hip_y-40));   HIP_hi = min(H, int(ank_y-40))
    LOW_lo = HIP_hi;                  LOW_hi = H

    def scale_band(v_in, vp_in, y_lo, y_hi, sx, dx_px=0.0):
        if abs(sx-1.0)<1e-4 and abs(dx_px)<0.5: return v_in
        bm = (vp_in[:,1]>=y_lo)&(vp_in[:,1]<=y_hi)
        if not np.any(bm): return v_in
        vo = v_in.copy()
        cx3d = float(np.median(vo[bm,0]))
        vo[bm,0] = cx3d + sx*(vo[bm,0]-cx3d)
        if abs(dx_px)>0.5:
            depth = float(np.mean(np.abs(vo[bm,2]+cam_t[2])))
            vo[bm,0] += dx_px*depth/focal
        return vo

    v = scale_band(v, vp, UP_lo,  UP_hi,  p.get("sx_upper",1.0), p.get("dx_upper",0.0))
    vp= proj2d(v, cam_t, focal, H, W)
    v = scale_band(v, vp, HIP_lo, HIP_hi, p.get("sx_hip",1.0),   p.get("dx_hip",0.0))
    vp= proj2d(v, cam_t, focal, H, W)
    v = scale_band(v, vp, LOW_lo, LOW_hi, p.get("sx_lower",1.0))
    arm_sx = p.get("arm_sx",1.0)
    if p.get("arm_applied") and abs(arm_sx-1.0)>1e-4:
        vp = proj2d(v, cam_t, focal, H, W)
        v = scale_band(v, vp, max(0,int(sho_y-20)), UP_hi, arm_sx)
    return v

# ── 渲染 ─────────────────────────────────────────────────────────────────────
def render_and_blend(verts, cam_t, focal, faces, bg_frame):
    black_bg = np.zeros((H,W,3), dtype=np.uint8)
    rend     = Renderer(focal_length=focal, faces=faces)
    out      = rend(verts, cam_t, black_bg,
                    mesh_base_color=GHOST_COLOR, scene_bg_color=(0,0,0))
    out_u8   = (out*255).clip(0,255).astype(np.uint8)
    mask     = np.any(out_u8>5, axis=2)
    result   = bg_frame.copy().astype(np.float32)
    result[mask] = (1-GHOST_ALPHA)*result[mask] + GHOST_ALPHA*out_u8[mask].astype(np.float32)
    return result.astype(np.uint8)

# ── 加载 ─────────────────────────────────────────────────────────────────────
print("[INIT] Loading model...")
t0=time.time()
device="cuda" if torch.cuda.is_available() else "cpu"
model, model_cfg = load_sam_3d_body(str(CKPT), device=device, mhr_path=str(MHR_PT))
estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=model_cfg,
                                human_detector=None, human_segmentor=None, fov_estimator=None)
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

# ── PASS 1: 全段推理, 收集 verts / cam_t / focal / kp2d ──────────────────────
print(f"\n[PASS1] Full inference all {NF} frames...")
raw = {}   # fi → {verts, cam_t, focal, kp2d, ok}
t_p1 = time.time()

for fi in range(NF):
    kp = kp_frames.get(fi)
    if kp is None:
        raw[fi] = {"ok": False}
        continue
    bbox = get_bbox(kp[:,:2], H, W)
    with torch.no_grad():
        outs = estimator.process_one_image(src[fi], bboxes=bbox,
                                           use_mask=False, inference_type="body")
    if not outs:
        raw[fi] = {"ok": False}
        continue
    o = outs[0]
    raw[fi] = {
        "ok":    True,
        "verts": o["pred_vertices"].astype(np.float32),
        "cam_t": o["pred_cam_t"].astype(np.float32),
        "focal": float(o["focal_length"]),
        "kp2d":  o["pred_keypoints_2d"].astype(np.float32),
    }
    if fi % 10 == 0:
        print(f"  fr{fi:03d}  cam_t={raw[fi]['cam_t'].tolist()}  focal={raw[fi]['focal']:.1f}")

print(f"  pass1 done in {time.time()-t_p1:.1f}s")
ok_frames = [fi for fi in range(NF) if raw[fi].get("ok")]
print(f"  ok frames: {len(ok_frames)}/{NF}")

# ── PASS 2: 对 cam_t x/y 和 focal 做时序平滑 ─────────────────────────────────
print(f"\n[PASS2] Smoothing cam_t and focal with Savitzky-Golay (window={SG_WINDOW}, poly={SG_POLY})...")

# Build dense arrays over ok_frames only (interpolate over hard_invalid slots)
fi_arr  = np.array(ok_frames, dtype=int)
ctx_raw = np.array([raw[fi]["cam_t"][0] for fi in ok_frames])  # cam_t x
cty_raw = np.array([raw[fi]["cam_t"][1] for fi in ok_frames])  # cam_t y
ctz_raw = np.array([raw[fi]["cam_t"][2] for fi in ok_frames])  # cam_t z (don't smooth)
foc_raw = np.array([raw[fi]["focal"]    for fi in ok_frames])

# Apply SG filter (pads at edges automatically with mode='mirror')
ctx_sm  = savgol_filter(ctx_raw, SG_WINDOW, SG_POLY, mode='mirror')
cty_sm  = savgol_filter(cty_raw, SG_WINDOW, SG_POLY, mode='mirror')
foc_sm  = savgol_filter(foc_raw, SG_WINDOW, SG_POLY, mode='mirror')
# DO NOT smooth cam_t z (depth) — changing depth changes apparent size which is bad

# Store smoothed values back into raw dict
for idx, fi in enumerate(ok_frames):
    raw[fi]["cam_t_smooth"] = np.array([ctx_sm[idx], cty_sm[idx], ctz_raw[idx]], dtype=np.float32)
    raw[fi]["focal_smooth"] = float(foc_sm[idx])

# Print delta stats for flicker segment
seg = [fi for fi in range(84, 96) if fi in raw and raw[fi].get("ok")]
print(f"\n  cam_t_x before/after smoothing (flicker segment fr84-95):")
print(f"  {'fr':>4}  {'ctx_raw':>9}  {'ctx_sm':>9}  {'Δ':>8}  {'cty_raw':>9}  {'cty_sm':>9}  {'Δ':>8}")
for fi in seg:
    idx = list(ok_frames).index(fi)
    print(f"  fr{fi:03d}  {ctx_raw[idx]:9.3f}  {ctx_sm[idx]:9.3f}  {ctx_sm[idx]-ctx_raw[idx]:+8.4f}"
          f"  {cty_raw[idx]:9.3f}  {cty_sm[idx]:9.3f}  {cty_sm[idx]-cty_raw[idx]:+8.4f}")

# ── PASS 3: 处理 HARD_INVALID verts (前后平均插值) ───────────────────────────
print(f"\n[PASS3] Hard-invalid verts interpolation for fr{sorted(HARD_INVALID)}...")
for fi in sorted(HARD_INVALID):
    prev_fi = next((f for f in range(fi-1, -1, -1) if raw.get(f,{}).get("ok")), None)
    next_fi = next((f for f in range(fi+1, NF)     if raw.get(f,{}).get("ok")), None)
    if prev_fi is not None and next_fi is not None:
        w = (fi - prev_fi) / (next_fi - prev_fi)
        raw[fi]["verts"] = (1-w)*raw[prev_fi]["verts"] + w*raw[next_fi]["verts"]
        raw[fi]["kp2d"]  = (1-w)*raw[prev_fi]["kp2d"]  + w*raw[next_fi]["kp2d"]
        # cam_t/focal already smoothed or use neighbor avg
        raw[fi]["cam_t_smooth"] = (1-w)*raw[prev_fi].get("cam_t_smooth", raw[prev_fi]["cam_t"]) + \
                                   w*raw[next_fi].get("cam_t_smooth", raw[next_fi]["cam_t"])
        raw[fi]["focal_smooth"] = (1-w)*raw[prev_fi].get("focal_smooth", raw[prev_fi]["focal"]) + \
                                   w*raw[next_fi].get("focal_smooth", raw[next_fi]["focal"])
        raw[fi]["ok"] = True
        print(f"  fr{fi:03d}: verts interpolated from fr{prev_fi}(w={1-w:.2f}) + fr{next_fi}(w={w:.2f})")
    elif prev_fi is not None:
        raw[fi] = {**raw[prev_fi], "ok": True}
        print(f"  fr{fi:03d}: copied from fr{prev_fi}")

# ── PASS 4: 渲染全段 ──────────────────────────────────────────────────────────
print(f"\n[PASS4] Rendering {NF} frames with smoothed cam_t...")
out_path = OUTPUT_DIR / "playback_t25_raw.mp4"
fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
writer   = cv2.VideoWriter(str(out_path), fourcc, actual_fps, (W, H))

mesh_cx_raw = []   # using raw cam_t
mesh_cx_sm  = []   # using smoothed cam_t
frame_ids   = []

t_p4 = time.time()
for fi in range(NF):
    r = raw.get(fi, {})
    if not r.get("ok"):
        writer.write(src[fi].copy())
        mesh_cx_raw.append(np.nan); mesh_cx_sm.append(np.nan)
        frame_ids.append(fi)
        continue

    verts_orig = r["verts"]
    cam_t_sm   = r.get("cam_t_smooth", r["cam_t"])
    focal_sm   = r.get("focal_smooth",  r["focal"])
    kp2d       = r["kp2d"]
    p          = t23_params.get(fi, {})

    # Apply T2.3 sx with smoothed cam_t (shape fitting stays the same)
    verts_adj = apply_t23_sx(verts_orig, cam_t_sm, focal_sm, kp2d, p, H, W)

    frame_out = render_and_blend(verts_adj, cam_t_sm, focal_sm, faces, src[fi])
    writer.write(frame_out)

    # cx tracking
    vp_sm  = proj2d(verts_adj, cam_t_sm,  focal_sm,  H, W)
    vp_raw = proj2d(verts_orig, r["cam_t"], r["focal"], H, W)
    mesh_cx_sm.append(float(np.median(vp_sm[:,0])))
    mesh_cx_raw.append(float(np.median(vp_raw[:,0])))
    frame_ids.append(fi)

    if fi % 10 == 0:
        elapsed=time.time()-t_p4; eta=elapsed/(fi+1)*(NF-fi-1) if fi>0 else 0
        tag = "HARD_INV" if fi in HARD_INVALID else ("smooth" if 84<=fi<=95 else "normal")
        print(f"  fr{fi:03d} [{tag}]  cx_raw={mesh_cx_raw[-1]:.1f}  cx_sm={mesh_cx_sm[-1]:.1f}  ETA={eta:.0f}s")

writer.release()
print(f"  render done in {time.time()-t_p4:.1f}s")

# ── re-encode ──────────────────────────────────────────────────────────────────
enc_path = OUTPUT_DIR / "playback_t25.mp4"
os.system(f"ffmpeg -y -i {out_path} -vcodec libx264 -crf 20 -pix_fmt yuv420p {enc_path} 2>/dev/null")
if enc_path.exists() and enc_path.stat().st_size > 0:
    out_path.unlink()
    print(f"  re-encoded → {enc_path}")
else:
    enc_path = out_path

# ── 曲线对比图 ────────────────────────────────────────────────────────────────
frames_arr  = np.array(frame_ids)
cx_raw_arr  = np.array(mesh_cx_raw)
cx_sm_arr   = np.array(mesh_cx_sm)
ctx_full = np.full(NF, np.nan); cty_full = np.full(NF, np.nan); foc_full = np.full(NF, np.nan)
ctx_sm_full= np.full(NF, np.nan); cty_sm_full=np.full(NF,np.nan); foc_sm_full=np.full(NF,np.nan)
for idx, fi in enumerate(ok_frames):
    ctx_full[fi]=ctx_raw[idx]; cty_full[fi]=cty_raw[idx]; foc_full[fi]=foc_raw[idx]
    ctx_sm_full[fi]=ctx_sm[idx]; cty_sm_full[fi]=cty_sm[idx]; foc_sm_full[fi]=foc_sm[idx]

fig, axes = plt.subplots(4,1,figsize=(14,14))
fig.suptitle("T2.5 — cam_t smoothing: before (red) vs after (blue)\norange band = flicker segment fr84-95", fontsize=12)

for ax, raw_v, sm_v, ylabel, title in zip(
    axes,
    [ctx_full, cty_full, foc_full, cx_raw_arr],
    [ctx_sm_full, cty_sm_full, foc_sm_full, cx_sm_arr],
    ["cam_t x", "cam_t y", "focal (px)", "mesh_cx (px)"],
    ["cam_t[0] x (raw vs SG-smoothed)",
     "cam_t[1] y (raw vs SG-smoothed)",
     "focal length (raw vs SG-smoothed)",
     "mesh centroid cx in image (raw verts+raw cam vs smoothed cam)"]
):
    ax.plot(frames_arr if ylabel=="mesh_cx (px)" else np.arange(NF),
            raw_v, 'r-', lw=1.2, alpha=0.8, label="raw")
    ax.plot(frames_arr if ylabel=="mesh_cx (px)" else np.arange(NF),
            sm_v,  'b-', lw=1.5, label="SG-smoothed")
    ax.axvspan(84, 95, alpha=0.12, color='orange', label="flicker seg")
    ax.set_ylabel(ylabel, fontsize=8); ax.legend(fontsize=8); ax.set_title(title, fontsize=9)
    ax.grid(True,alpha=0.3)
axes[-1].set_xlabel("frame")
plt.tight_layout()
plot_p = OUTPUT_DIR / "cam_t_smooth_t25.jpg"
plt.savefig(str(plot_p), dpi=120, bbox_inches='tight')
plt.close()

# ── 报告 ──────────────────────────────────────────────────────────────────────
# SG delta stats for flicker segment
delta_ctx = [abs(ctx_sm_full[fi]-ctx_full[fi]) for fi in range(84,96) if not np.isnan(ctx_full[fi])]
delta_cty = [abs(cty_sm_full[fi]-cty_full[fi]) for fi in range(84,96) if not np.isnan(cty_full[fi])]
# frame-to-frame jump reduction
def fj(arr):
    v=[arr[i] for i in range(len(arr)) if not np.isnan(arr[i])]
    return np.mean(np.abs(np.diff(v))) if len(v)>1 else 0

report_lines=[
    "GHOST-003 T2.5 停关卡报告",
    f"Clip: fo-ok-1  NF={NF}",
    "",
    "=== 修正方向 ===",
    "T2.4 问题: verts+cam_t 全插 → 姿态失真 (downswing 收窄)",
    "T2.5 修正: verts 不动(保留真实姿态), 只平滑 cam_t x/y + focal",
    "",
    "=== 平滑参数 ===",
    f"方法: Savitzky-Golay  window={SG_WINDOW}帧  poly={SG_POLY}",
    f"平滑对象: cam_t[0](x), cam_t[1](y), focal  —  cam_t[2](z/depth) 不动",
    "",
    "=== cam_t 平滑效果 (闪动段 fr084-095) ===",
    f"cam_t_x 修正量: mean={np.mean(delta_ctx):.4f}  max={np.max(delta_ctx):.4f}",
    f"cam_t_y 修正量: mean={np.mean(delta_cty):.4f}  max={np.max(delta_cty):.4f}",
    f"mesh_cx 帧间跳变: 平滑前={fj(cx_raw_arr):.2f}px/帧  平滑后={fj(cx_sm_arr):.2f}px/帧",
    "",
    "=== 帧处理汇总 ===",
    f"fr000-083: 正常推理 + T2.3 sx + SG 平滑 cam_t",
    f"fr084-095: 正常推理(保留真实姿态) + SG 平滑 cam_t  ← 核心修正",
    f"fr087:     T2.2 hard-invalid → verts 邻帧插值, cam_t SG 平滑",
    f"fr094:     T2.2 hard-invalid → verts 邻帧插值, cam_t SG 平滑",
    f"fr096-111: 正常推理 + T2.3 sx + SG 平滑 cam_t",
    "",
    "=== 验收要点 ===",
    "playback_t25.mp4 — 重点看 fr084-095 (downswing→impact):",
    "  1. 红人姿态应与真人一样舒展有力 (下盘叉开, 身体转动)",
    "  2. 位置不再左右上下跳变",
    "  3. 主体其余帧严丝合缝不受影响",
]
rp = OUTPUT_DIR/"REPORT_T25.txt"
rp.write_text("\n".join(report_lines), encoding="utf-8")

import shutil
shutil.copy2(str(enc_path),  str(WIN_OUT/"playback_t25.mp4"))
shutil.copy2(str(plot_p),    str(WIN_OUT/"cam_t_smooth_t25.jpg"))
shutil.copy2(str(rp),        str(WIN_OUT/"REPORT_T25.txt"))

tt=time.time()-t0
print(f"\n[OUT] Windows:")
print(f"  {WIN_OUT}/playback_t25.mp4")
print(f"  {WIN_OUT}/cam_t_smooth_t25.jpg")
print(f"\n{'='*60}")
print(f"T2.5 DONE  NF={NF}  total={tt:.1f}s")
print(f"cam_t_x jump: {fj(ctx_full):.3f} raw → {fj(ctx_sm_full):.3f} smoothed")
print(f"cam_t_y jump: {fj(cty_full):.3f} raw → {fj(cty_sm_full):.3f} smoothed")
print(f"{'='*60}")
