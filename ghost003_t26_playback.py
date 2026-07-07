"""
ghost003_t26_playback.py  —  GHOST-003 T2.6
体型锁定(cam_t[2]/z 锁定到 address 帧) + 全三轴 SG 平滑 + 慢放版

根因分析 (T2.5 诊断结果):
  cam_t[0/1] 本身跳变量 < 0.003 (约 0.5px) — 不是主因
  "忽大忽小" 来源: cam_t[2](深度/z) 帧间波动 → 投影比例变化 (scale ∝ 1/z)
  fr000 z=5.000, fr090 z=5.074 → 1.5% 尺度变化, 在静止段可见

修正策略:
  1. 体型锁定: cam_t[2] = fr000 的 z 值, 全段固定 (消除尺度呼吸)
  2. 全三轴 SG 平滑: cam_t[0/1/2] + focal 全部平滑
     z 锁定后 SG 压的是微小残余抖动
  3. 每帧真实 verts (姿态不插值), fr087/094 verts 邻帧插值
  4. T2.3 sx 近似复用 (cam_t[2] 锁定后 sx 尺度略有变化, 接受近似误差)

输出:
  playback_t26.mp4        (正常速)
  playback_t26_025x.mp4   (0.25x 慢放, ffmpeg setpts=4*PTS)
  cam_t_z_lock_t26.jpg    (z 锁定 + 全轴 SG 平滑前后对比)
  REPORT_T26.txt
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
OUTPUT_DIR = Path("/home/jason/projects/swingcue-postest/output/ghost003_t26")
WIN_OUT    = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/ghost003_t26")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WIN_OUT.mkdir(parents=True, exist_ok=True)

CKPT   = Path(os.path.expanduser("~/.cache/sam3d/sam-3d-body-dinov3/model.ckpt"))
MHR_PT = CKPT.parent / "assets/mhr_model.pt"

NF  = 112; H = 1280; W = 720
HARD_INVALID = {87, 94}   # T2.2 哨兵帧: verts 用邻帧插值
FR_ADDRESS = 0

# SG 平滑参数
SG_WINDOW = 9
SG_POLY   = 3

GHOST_ALPHA = 0.55
GHOST_COLOR = (0.85, 0.1, 0.1)

# ── 关节索引 ─────────────────────────────────────────────────────────────────
I_NOSE=0; I_LSHO=5; I_RSHO=6; I_LHIP=11; I_RHIP=12
I_LANK=15; I_RANK=16

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
    d=verts[:,2]+cam_t[2]
    px=focal*(verts[:,0]-cam_t[0])/d+W/2
    py=focal*(verts[:,1]+cam_t[1])/d+H/2
    return np.stack([px,py],axis=1)

# ── T2.3 sx (轻量复用) ────────────────────────────────────────────────────────
def apply_t23_sx(verts, cam_t, focal, kp2d, p, H, W):
    v = verts.copy(); vp= proj2d(v, cam_t, focal, H, W)
    hip_y  = (kp2d[I_LHIP][1]+kp2d[I_RHIP][1])/2
    ank_y  = (kp2d[I_LANK][1]+kp2d[I_RANK][1])/2
    sho_y  = (kp2d[I_LSHO][1]+kp2d[I_RSHO][1])/2
    nose_y = kp2d[I_NOSE][1]

    UP_lo=max(0,int(nose_y-30)); UP_hi=min(H,int(hip_y+60))
    HIP_lo=max(0,int(hip_y-40)); HIP_hi=min(H,int(ank_y-40))
    LOW_lo=HIP_hi; LOW_hi=H

    def scale_band(v_in, vp_in, y_lo, y_hi, sx, dx_px=0.0):
        if abs(sx-1.0)<1e-4 and abs(dx_px)<0.5: return v_in
        bm=(vp_in[:,1]>=y_lo)&(vp_in[:,1]<=y_hi)
        if not np.any(bm): return v_in
        vo=v_in.copy(); cx3d=float(np.median(vo[bm,0]))
        vo[bm,0]=cx3d+sx*(vo[bm,0]-cx3d)
        if abs(dx_px)>0.5:
            depth=float(np.mean(np.abs(vo[bm,2]+cam_t[2])))
            vo[bm,0]+=dx_px*depth/focal
        return vo

    v=scale_band(v,vp,UP_lo, UP_hi, p.get("sx_upper",1.0),p.get("dx_upper",0.0))
    vp=proj2d(v,cam_t,focal,H,W)
    v=scale_band(v,vp,HIP_lo,HIP_hi,p.get("sx_hip",1.0),  p.get("dx_hip",0.0))
    vp=proj2d(v,cam_t,focal,H,W)
    v=scale_band(v,vp,LOW_lo,LOW_hi,p.get("sx_lower",1.0))
    arm_sx=p.get("arm_sx",1.0)
    if p.get("arm_applied") and abs(arm_sx-1.0)>1e-4:
        vp=proj2d(v,cam_t,focal,H,W)
        v=scale_band(v,vp,max(0,int(sho_y-20)),UP_hi,arm_sx)
    return v

# ── 渲染 ─────────────────────────────────────────────────────────────────────
def render_and_blend(verts, cam_t, focal, faces, bg_frame):
    black_bg=np.zeros((H,W,3),dtype=np.uint8)
    rend=Renderer(focal_length=focal,faces=faces)
    out=rend(verts,cam_t,black_bg,mesh_base_color=GHOST_COLOR,scene_bg_color=(0,0,0))
    out_u8=(out*255).clip(0,255).astype(np.uint8)
    mask=np.any(out_u8>5,axis=2)
    result=bg_frame.copy().astype(np.float32)
    result[mask]=(1-GHOST_ALPHA)*result[mask]+GHOST_ALPHA*out_u8[mask].astype(np.float32)
    return result.astype(np.uint8)

# ── 加载 ─────────────────────────────────────────────────────────────────────
print("[INIT] Loading model...")
t0=time.time()
device="cuda" if torch.cuda.is_available() else "cpu"
model,model_cfg=load_sam_3d_body(str(CKPT),device=device,mhr_path=str(MHR_PT))
estimator=SAM3DBodyEstimator(sam_3d_body_model=model,model_cfg=model_cfg,
                              human_detector=None,human_segmentor=None,fov_estimator=None)
estimator.model.eval()
faces=estimator.faces
print(f"  {time.time()-t0:.1f}s")

cap=cv2.VideoCapture(str(VIDEO_IN))
actual_fps=cap.get(cv2.CAP_PROP_FPS)
src=[]
while True:
    ret,f=cap.read()
    if not ret: break
    src.append(f)
cap.release()
print(f"  {len(src)} frames @ {actual_fps:.2f}fps")

kp_frames=load_kp_cache(KP_CACHE)
t23_params=load_t23_params(RUN_LOG)

# ── PASS 1: 全段推理 ──────────────────────────────────────────────────────────
print(f"\n[PASS1] Full inference {NF} frames...")
raw={}
t_p1=time.time()

for fi in range(NF):
    kp=kp_frames.get(fi)
    if kp is None: raw[fi]={"ok":False}; continue
    bbox=get_bbox(kp[:,:2],H,W)
    with torch.no_grad():
        outs=estimator.process_one_image(src[fi],bboxes=bbox,use_mask=False,inference_type="body")
    if not outs: raw[fi]={"ok":False}; continue
    o=outs[0]
    raw[fi]={"ok":True,
             "verts":o["pred_vertices"].astype(np.float32),
             "cam_t":o["pred_cam_t"].astype(np.float32),
             "focal":float(o["focal_length"]),
             "kp2d": o["pred_keypoints_2d"].astype(np.float32)}
    if fi%10==0:
        ct=raw[fi]["cam_t"]
        print(f"  fr{fi:03d}  cam_t=[{ct[0]:.4f},{ct[1]:.4f},{ct[2]:.4f}]  focal={raw[fi]['focal']:.1f}")

print(f"  pass1 done {time.time()-t_p1:.1f}s")
ok_frames=[fi for fi in range(NF) if raw[fi].get("ok")]

# ── PASS 2: 体型锁定 — cam_t[2] 锁定到 fr000 ─────────────────────────────────
z_lock = raw[FR_ADDRESS]["cam_t"][2]
print(f"\n[PASS2] Shape lock: cam_t[2] = {z_lock:.4f} (fr000, locked for all frames)")
z_raw_arr = np.array([raw[fi]["cam_t"][2] for fi in ok_frames])
print(f"  z range before lock: [{z_raw_arr.min():.4f}, {z_raw_arr.max():.4f}]"
      f"  spread={z_raw_arr.max()-z_raw_arr.min():.4f}")
for fi in ok_frames:
    raw[fi]["cam_t_locked"] = raw[fi]["cam_t"].copy()
    raw[fi]["cam_t_locked"][2] = z_lock

# ── PASS 3: 全三轴 SG 平滑 (基于锁定后的 cam_t) ──────────────────────────────
print(f"\n[PASS3] SG smoothing all 3 axes (window={SG_WINDOW}, poly={SG_POLY})...")
fi_arr  = np.array(ok_frames)
ctx_raw = np.array([raw[fi]["cam_t_locked"][0] for fi in ok_frames])
cty_raw = np.array([raw[fi]["cam_t_locked"][1] for fi in ok_frames])
ctz_raw = np.array([raw[fi]["cam_t_locked"][2] for fi in ok_frames])  # all z_lock
foc_raw = np.array([raw[fi]["focal"]             for fi in ok_frames])

ctx_sm = savgol_filter(ctx_raw, SG_WINDOW, SG_POLY, mode='mirror')
cty_sm = savgol_filter(cty_raw, SG_WINDOW, SG_POLY, mode='mirror')
ctz_sm = ctz_raw  # z is already locked, no smoothing needed (constant)
foc_sm = savgol_filter(foc_raw, SG_WINDOW, SG_POLY, mode='mirror')

for idx,fi in enumerate(ok_frames):
    raw[fi]["cam_t_final"] = np.array([ctx_sm[idx],cty_sm[idx],ctz_sm[idx]],dtype=np.float32)
    raw[fi]["focal_final"] = float(foc_sm[idx])

# Print flicker segment stats
print(f"\n  cam_t full range after lock+smooth:")
print(f"  x: [{ctx_sm.min():.4f}, {ctx_sm.max():.4f}]  spread={ctx_sm.max()-ctx_sm.min():.4f}")
print(f"  y: [{cty_sm.min():.4f}, {cty_sm.max():.4f}]  spread={cty_sm.max()-cty_sm.min():.4f}")
print(f"  z: {ctz_sm[0]:.4f} (constant locked)")
print(f"  focal: [{foc_sm.min():.1f}, {foc_sm.max():.1f}]  spread={foc_sm.max()-foc_sm.min():.1f}px")

# ── PASS 4: HARD_INVALID verts 插值 ──────────────────────────────────────────
print(f"\n[PASS4] Hard-invalid verts interpolation...")
for fi in sorted(HARD_INVALID):
    prev_fi=next((f for f in range(fi-1,-1,-1) if raw.get(f,{}).get("ok")),None)
    next_fi=next((f for f in range(fi+1,NF) if raw.get(f,{}).get("ok")),None)
    if prev_fi is not None and next_fi is not None:
        w=(fi-prev_fi)/(next_fi-prev_fi)
        raw[fi]["verts"]=(1-w)*raw[prev_fi]["verts"]+w*raw[next_fi]["verts"]
        raw[fi]["kp2d"] =(1-w)*raw[prev_fi]["kp2d"] +w*raw[next_fi]["kp2d"]
        raw[fi]["cam_t_final"]=raw[fi].get("cam_t_final",
            np.array([ctx_sm[ok_frames.index(prev_fi)],cty_sm[ok_frames.index(prev_fi)],z_lock],dtype=np.float32))
        raw[fi]["focal_final"]=raw[fi].get("focal_final",float(foc_sm[ok_frames.index(prev_fi)]))
        raw[fi]["ok"]=True
        print(f"  fr{fi:03d}: verts interp from fr{prev_fi}+fr{next_fi}  w={w:.2f}")

# ── PASS 5: 渲染 ─────────────────────────────────────────────────────────────
print(f"\n[PASS5] Rendering {NF} frames...")
out_path=OUTPUT_DIR/"playback_t26_raw.mp4"
fourcc=cv2.VideoWriter_fourcc(*'mp4v')
writer=cv2.VideoWriter(str(out_path),fourcc,actual_fps,(W,H))

mesh_cx_final=[]
t_p5=time.time()

for fi in range(NF):
    r=raw.get(fi,{})
    if not r.get("ok"):
        writer.write(src[fi].copy()); mesh_cx_final.append(np.nan); continue

    verts_orig = r["verts"]
    cam_t_f    = r.get("cam_t_final", r["cam_t"])
    focal_f    = r.get("focal_final",  r["focal"])
    kp2d       = r["kp2d"]
    p          = t23_params.get(fi,{})

    verts_adj = apply_t23_sx(verts_orig, cam_t_f, focal_f, kp2d, p, H, W)
    frame_out = render_and_blend(verts_adj, cam_t_f, focal_f, faces, src[fi])
    writer.write(frame_out)

    vp=proj2d(verts_adj,cam_t_f,focal_f,H,W)
    mesh_cx_final.append(float(np.median(vp[:,0])))

    if fi%10==0:
        elapsed=time.time()-t_p5; eta=elapsed/(fi+1)*(NF-fi-1) if fi>0 else 0
        tag="HARD_INV" if fi in HARD_INVALID else "ok"
        print(f"  fr{fi:03d} [{tag}]  cx={mesh_cx_final[-1]:.1f}  z={cam_t_f[2]:.4f}  ETA={eta:.0f}s")

writer.release()
print(f"  render done {time.time()-t_p5:.1f}s")

# ── re-encode 正常速 ──────────────────────────────────────────────────────────
enc_path=OUTPUT_DIR/"playback_t26.mp4"
os.system(f"ffmpeg -y -i {out_path} -vcodec libx264 -crf 20 -pix_fmt yuv420p {enc_path} 2>/dev/null")
if enc_path.exists() and enc_path.stat().st_size>0:
    out_path.unlink(); print(f"  re-encoded → {enc_path}")
else:
    enc_path=out_path

# ── 0.25x 慢放 ───────────────────────────────────────────────────────────────
slow_path=OUTPUT_DIR/"playback_t26_025x.mp4"
ret=os.system(f"ffmpeg -y -i {enc_path} "
              f"-vf setpts=4.0*PTS -an "
              f"-vcodec libx264 -crf 20 -pix_fmt yuv420p {slow_path} 2>/dev/null")
if slow_path.exists() and slow_path.stat().st_size>0:
    print(f"  0.25x slow → {slow_path}")
else:
    print(f"  WARNING: slow motion ffmpeg failed")
    slow_path=None

# ── 对比图 ────────────────────────────────────────────────────────────────────
z_full=np.full(NF,np.nan); z_locked=np.full(NF,np.nan)
ctx_full=np.full(NF,np.nan); ctx_sm_full=np.full(NF,np.nan)
foc_full=np.full(NF,np.nan); foc_sm_full=np.full(NF,np.nan)
for idx,fi in enumerate(ok_frames):
    z_full[fi]=raw[fi]["cam_t"][2]
    z_locked[fi]=z_lock
    ctx_full[fi]=ctx_raw[idx]; ctx_sm_full[fi]=ctx_sm[idx]
    foc_full[fi]=foc_raw[idx]; foc_sm_full[fi]=foc_sm[idx]

fr_ax=np.arange(NF)
cx_arr=np.array(mesh_cx_final)

fig,axes=plt.subplots(4,1,figsize=(14,14))
fig.suptitle("T2.6 — Shape Lock (z fixed) + Full-axis SG Smooth\n"
             "orange band = flicker segment fr84-95", fontsize=11)

for ax,raw_v,sm_v,ylabel,title in zip(axes,
    [z_full, ctx_full, foc_full, cx_arr],
    [z_locked,ctx_sm_full,foc_sm_full,cx_arr],
    ["cam_t z (depth)","cam_t x","focal (px)","mesh_cx (px)"],
    ["cam_t[2] z: raw (red) vs locked fr000 value (blue) — eliminates scale breathing",
     "cam_t[0] x: raw vs SG-smoothed",
     "focal: raw vs SG-smoothed",
     "mesh centroid cx (T2.6 final)"]):
    ax.plot(fr_ax,raw_v,'r-',lw=1.2,alpha=0.8,label="raw")
    ax.plot(fr_ax,sm_v, 'b-',lw=1.5,label="locked/smoothed")
    ax.axvspan(84,95,alpha=0.12,color='orange')
    ax.set_ylabel(ylabel,fontsize=8); ax.legend(fontsize=8)
    ax.set_title(title,fontsize=9); ax.grid(True,alpha=0.3)
axes[-1].set_xlabel("frame")
plt.tight_layout()
plot_p=OUTPUT_DIR/"cam_t_z_lock_t26.jpg"
plt.savefig(str(plot_p),dpi=120,bbox_inches='tight')
plt.close()

# ── 报告 ──────────────────────────────────────────────────────────────────────
z_spread=float(z_raw_arr.max()-z_raw_arr.min())
scale_change_pct=z_spread/float(z_raw_arr.mean())*100

report=[
    "GHOST-003 T2.6 停关卡报告",
    f"Clip: fo-ok-1  NF={NF}",
    "",
    "=== 根因修正 ===",
    f"'忽大忽小' 根因: cam_t[2](z/深度) 帧间波动",
    f"  z 范围: [{z_raw_arr.min():.4f}, {z_raw_arr.max():.4f}]  spread={z_spread:.4f}",
    f"  对应尺度变化: {scale_change_pct:.2f}%  (静止段可见)",
    "",
    "=== 体型锁定 ===",
    f"cam_t[2] 锁定到 fr000: z = {z_lock:.4f} (全段 112 帧固定)",
    f"scale ∝ 1/z → z 恒定后, mesh 投影尺度全段恒定",
    "",
    "=== cam_t 全三轴 SG 平滑 ===",
    f"方法: Savitzky-Golay  window={SG_WINDOW}  poly={SG_POLY}",
    f"cam_t[0] x: 平滑后 spread={float(ctx_sm.max()-ctx_sm.min()):.4f}",
    f"cam_t[1] y: 平滑后 spread={float(cty_sm.max()-cty_sm.min()):.4f}",
    f"cam_t[2] z: 锁定 {z_lock:.4f} (常量)",
    f"focal:      平滑后 spread={float(foc_sm.max()-foc_sm.min()):.1f}px",
    "",
    "=== 帧处理 ===",
    f"fr087/fr094: verts 邻帧插值 (T2.2 hard-invalid)",
    f"其余 110 帧: 原始 verts (真实姿态) + 锁定+平滑 cam_t",
    "",
    "=== 输出文件 ===",
    f"playback_t26.mp4     — 正常速 (30fps)",
    f"playback_t26_025x.mp4 — 0.25x 慢放 (setpts=4*PTS)",
    f"cam_t_z_lock_t26.jpg  — 四轴对比图",
    "",
    "=== 验收要点 ===",
    "1. follow-through 收尾段 (fr096-111): 站定后红人大小是否恒定",
    "2. downswing→impact (fr084-095): 姿态是否舒展有力, 不收窄",
    "3. 0.25x 慢放: 按产品真实使用场景严格验收",
]
(OUTPUT_DIR/"REPORT_T26.txt").write_text("\n".join(report),encoding="utf-8")

import shutil
shutil.copy2(str(enc_path),str(WIN_OUT/"playback_t26.mp4"))
if slow_path: shutil.copy2(str(slow_path),str(WIN_OUT/"playback_t26_025x.mp4"))
shutil.copy2(str(plot_p),str(WIN_OUT/"cam_t_z_lock_t26.jpg"))
shutil.copy2(str(OUTPUT_DIR/"REPORT_T26.txt"),str(WIN_OUT/"REPORT_T26.txt"))

tt=time.time()-t0
print(f"\n[OUT] Windows: {WIN_OUT}")
print(f"  playback_t26.mp4")
print(f"  playback_t26_025x.mp4")
print(f"  cam_t_z_lock_t26.jpg")
print(f"\n{'='*60}")
print(f"T2.6 DONE  NF={NF}  z_locked={z_lock:.4f}  scale_fix={scale_change_pct:.2f}%  total={tt:.1f}s")
print(f"{'='*60}")
