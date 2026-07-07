"""
ghost003_t24_probe.py  —  GHOST-003 T2.4 前置探针
逐帧轻量推理：记录 mesh_cx / mesh_cy / mesh_area 及 RTMPose human_cx / human_cy
计算帧间时序跳变，输出候选闪动帧列表 + 时序折线图供 Jason 确认
不做 sx 拟合，不做渲染叠加，纯数据探针，快速完成
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

# ── 路径 ──────────────────────────────────────────────────────────────────────
VIDEO_IN   = Path("/home/jason/projects/swingcue-postest/input/fo-ok-1.mp4")
KP_CACHE   = Path("/home/jason/projects/swingcue-postest/engine/kp_cache/batch2/fo-ok-1.json")
OUTPUT_DIR = Path("/home/jason/projects/swingcue-postest/output/ghost003_t24")
WIN_OUT    = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/ghost003_t24")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WIN_OUT.mkdir(parents=True, exist_ok=True)

CKPT  = Path(os.path.expanduser("~/.cache/sam3d/sam-3d-body-dinov3/model.ckpt"))
MHR_PT= CKPT.parent / "assets/mhr_model.pt"

NF = 112; H = 1280; W = 720
INVALID_SET = {87, 94}   # T2.2 哨兵已剔除帧

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

# ── 加载模型 ──────────────────────────────────────────────────────────────────
print("[INIT] Loading model...")
t0=time.time()
device="cuda" if torch.cuda.is_available() else "cpu"
model, model_cfg = load_sam_3d_body(str(CKPT), device=device, mhr_path=str(MHR_PT))
estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=model_cfg,
                                human_detector=None, human_segmentor=None, fov_estimator=None)
estimator.model.eval()
print(f"  loaded {time.time()-t0:.1f}s")

# ── 读视频帧 ─────────────────────────────────────────────────────────────────
print("[INIT] Reading video...")
cap=cv2.VideoCapture(str(VIDEO_IN))
FPS=cap.get(cv2.CAP_PROP_FPS)
src=[]
while True:
    ret,f=cap.read()
    if not ret: break
    src.append(f)
cap.release()
print(f"  {len(src)} frames @ {FPS:.2f}fps")

kp_frames = load_kp_cache(KP_CACHE)

# ── 逐帧推理，记录 mesh 轨迹 ──────────────────────────────────────────────────
print("\n[PROBE] Running per-frame inference...")
records = []   # list of dicts per valid frame
t_start=time.time()

for fi in range(NF):
    if fi in INVALID_SET:
        records.append({"frame":fi,"invalid":True})
        continue
    kp = kp_frames.get(fi)
    if kp is None:
        records.append({"frame":fi,"invalid":True})
        continue

    bbox = get_bbox(kp[:,:2], H, W)
    with torch.no_grad():
        outs = estimator.process_one_image(src[fi], bboxes=bbox,
                                           use_mask=False, inference_type="body")
    if not outs:
        records.append({"frame":fi,"invalid":True,"reason":"no_output"})
        continue

    o = outs[0]
    verts = o["pred_vertices"].astype(np.float32)
    cam_t = o["pred_cam_t"].astype(np.float32)
    focal = float(o["focal_length"])
    kp2d_mhr = o["pred_keypoints_2d"].astype(np.float32)  # (70,2)

    # Project all vertices
    vp = proj2d(verts, cam_t, focal, H, W)

    # Clip to image bounds for valid px
    in_frame = (vp[:,0]>=0)&(vp[:,0]<W)&(vp[:,1]>=0)&(vp[:,1]<H)
    vp_valid = vp[in_frame]

    mesh_cx  = float(np.median(vp_valid[:,0])) if len(vp_valid)>0 else W/2
    mesh_cy  = float(np.median(vp_valid[:,1])) if len(vp_valid)>0 else H/2
    mesh_area= float(len(vp_valid))   # proxy: # vertices in frame

    # Human centroid from RTMPose: mid of hip + ankle
    I_LHIP=11; I_RHIP=12; I_LANK=15; I_RANK=16
    I_LSHO=5;  I_RSHO=6
    human_cx = float((kp[I_LHIP,0]+kp[I_RHIP,0])/2)
    human_cy = float((kp[I_LHIP,1]+kp[I_RHIP,1])/2)
    sho_cx   = float((kp[I_LSHO,0]+kp[I_RSHO,0])/2)
    ank_cx   = float((kp[I_LANK,0]+kp[I_RANK,0])/2)
    ank_cy   = float((kp[I_LANK,1]+kp[I_RANK,1])/2)

    records.append({
        "frame":fi, "invalid":False,
        "mesh_cx":mesh_cx, "mesh_cy":mesh_cy, "mesh_area":mesh_area,
        "human_cx":human_cx, "human_cy":human_cy,
        "sho_cx":sho_cx, "ank_cx":ank_cx, "ank_cy":ank_cy,
        "cam_t":cam_t.tolist(), "focal":focal,
    })
    if fi % 10 == 0:
        elapsed=time.time()-t_start
        eta=elapsed/(fi+1)*(NF-fi-1) if fi>0 else 0
        print(f"  fr{fi:03d}  mesh_cx={mesh_cx:.1f} cy={mesh_cy:.1f} area={mesh_area:.0f}  ETA={eta:.0f}s")

print(f"  probe done in {time.time()-t_start:.1f}s")

# ── 计算帧间跳变 ──────────────────────────────────────────────────────────────
valid_recs = [r for r in records if not r.get("invalid")]

# Build per-frame arrays
frames_v   = np.array([r["frame"]     for r in valid_recs])
mesh_cx_v  = np.array([r["mesh_cx"]   for r in valid_recs])
mesh_cy_v  = np.array([r["mesh_cy"]   for r in valid_recs])
mesh_area_v= np.array([r["mesh_area"] for r in valid_recs])
hum_cx_v   = np.array([r["human_cx"]  for r in valid_recs])
hum_cy_v   = np.array([r["human_cy"]  for r in valid_recs])

# Frame-to-frame deltas (absolute pixel shift between consecutive valid frames)
d_mesh_cx   = np.abs(np.diff(mesh_cx_v))
d_mesh_cy   = np.abs(np.diff(mesh_cy_v))
d_hum_cx    = np.abs(np.diff(hum_cx_v))
d_hum_cy    = np.abs(np.diff(hum_cy_v))
d_mesh_area = np.abs(np.diff(mesh_area_v)) / (mesh_area_v[:-1]+1) * 100  # % change

# Ratio: mesh jump / human jump (large ratio = mesh moved more than human → suspicious)
eps = 1.0
ratio_cx = d_mesh_cx / (d_hum_cx + eps)
ratio_cy = d_mesh_cy / (d_hum_cy + eps)

# Detect candidates: mesh_cx jumped > THR_ABS px AND ratio > THR_RATIO
THR_ABS_CX  = 15.0   # px absolute mesh_cx jump
THR_ABS_CY  = 25.0   # px absolute mesh_cy jump
THR_RATIO   = 3.0    # mesh jump is 3× human jump
THR_AREA    = 15.0   # % area change

# A frame pair (i → i+1): flag the LATER frame as flicker candidate
flicker_cands = []
for i in range(len(frames_v)-1):
    fr_next = int(frames_v[i+1])
    cx_flag = (d_mesh_cx[i] > THR_ABS_CX) and (ratio_cx[i] > THR_RATIO)
    cy_flag = (d_mesh_cy[i] > THR_ABS_CY) and (ratio_cy[i] > THR_RATIO)
    area_flag = d_mesh_area[i] > THR_AREA
    if cx_flag or cy_flag or area_flag:
        reasons = []
        if cx_flag:   reasons.append(f"cx_jump={d_mesh_cx[i]:.1f}px ratio={ratio_cx[i]:.1f}x")
        if cy_flag:   reasons.append(f"cy_jump={d_mesh_cy[i]:.1f}px ratio={ratio_cy[i]:.1f}x")
        if area_flag: reasons.append(f"area_chg={d_mesh_area[i]:.1f}%")
        flicker_cands.append({"frame":fr_next, "reasons":reasons,
                               "d_mesh_cx":float(d_mesh_cx[i]),
                               "d_mesh_cy":float(d_mesh_cy[i]),
                               "ratio_cx":float(ratio_cx[i]),
                               "ratio_cy":float(ratio_cy[i]),
                               "area_chg":float(d_mesh_area[i])})

# ── 打印结果 ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"T2.4 PROBE — 闪动帧候选")
print(f"阈值: cx_jump>{THR_ABS_CX}px && ratio>{THR_RATIO}x | cy_jump>{THR_ABS_CY}px | area_chg>{THR_AREA}%")
print(f"候选帧数: {len(flicker_cands)}")
print()
for fc in flicker_cands:
    print(f"  fr{fc['frame']:03d}  " + "  |  ".join(fc["reasons"]))
print()
# Summary by phase
phase_map = {}
for fi in range(NF):
    if   fi<=53:  phase_map[fi]="address"
    elif fi<=76:  phase_map[fi]="backswing"
    elif fi<=88:  phase_map[fi]="downswing"
    else:         phase_map[fi]="follow_through"
from collections import Counter
phase_cnt = Counter(phase_map[fc["frame"]] for fc in flicker_cands if fc["frame"] in phase_map)
print(f"  阶段分布: {dict(phase_cnt)}")
print(f"{'='*60}\n")

# ── 绘图 ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4,1,figsize=(14,14))
fig.suptitle("T2.4 Probe — Temporal Mesh vs Human Trajectory", fontsize=13)

# 1. mesh_cx vs human_cx
ax=axes[0]
ax.plot(frames_v, mesh_cx_v, 'r-', lw=1.2, label="mesh_cx")
ax.plot(frames_v, hum_cx_v,  'b-', lw=1.2, label="human_hip_cx (RTMPose)")
for fc in flicker_cands:
    ax.axvline(fc["frame"], color='orange', alpha=0.5, lw=1)
ax.set_ylabel("cx (px)"); ax.legend(fontsize=8); ax.set_title("mesh_cx vs human_cx")
ax.grid(True,alpha=0.3)

# 2. mesh_cy vs human_cy
ax=axes[1]
ax.plot(frames_v, mesh_cy_v, 'r-', lw=1.2, label="mesh_cy")
ax.plot(frames_v, hum_cy_v,  'b-', lw=1.2, label="human_hip_cy")
for fc in flicker_cands:
    ax.axvline(fc["frame"], color='orange', alpha=0.5, lw=1)
ax.set_ylabel("cy (px)"); ax.legend(fontsize=8); ax.set_title("mesh_cy vs human_cy")
ax.grid(True,alpha=0.3)

# 3. mesh_area
ax=axes[2]
ax.plot(frames_v, mesh_area_v, 'g-', lw=1.2, label="mesh_area (vertex count in frame)")
for fc in flicker_cands:
    ax.axvline(fc["frame"], color='orange', alpha=0.5, lw=1)
ax.set_ylabel("verts in frame"); ax.legend(fontsize=8); ax.set_title("mesh_area (proxy silhouette)")
ax.grid(True,alpha=0.3)

# 4. frame-to-frame jumps
jump_frames = frames_v[1:]
ax=axes[3]
ax.plot(jump_frames, d_mesh_cx, 'r-', lw=1, label="Δmesh_cx")
ax.plot(jump_frames, d_hum_cx,  'b-', lw=1, label="Δhuman_cx")
ax.plot(jump_frames, d_mesh_cy, 'r--', lw=0.8, alpha=0.7, label="Δmesh_cy")
ax.axhline(THR_ABS_CX, color='orange', ls=':', lw=1, label=f"thr_cx={THR_ABS_CX}px")
for fc in flicker_cands:
    ax.axvline(fc["frame"], color='orange', alpha=0.5, lw=1)
ax.set_ylabel("Δpx"); ax.legend(fontsize=8); ax.set_title("Frame-to-frame jumps (orange=flicker candidate)")
ax.set_xlabel("frame index")
ax.grid(True,alpha=0.3)

plt.tight_layout()
plot_path = OUTPUT_DIR / "probe_temporal_t24.jpg"
plt.savefig(str(plot_path), dpi=120, bbox_inches='tight')
plt.close()
print(f"[OUT] {plot_path}")

# ── 保存结果 JSON ─────────────────────────────────────────────────────────────
result = {
    "clip": "fo-ok-1", "NF": NF,
    "thresholds": {"cx_abs":THR_ABS_CX,"cy_abs":THR_ABS_CY,
                   "ratio":THR_RATIO,"area_pct":THR_AREA},
    "flicker_candidates": flicker_cands,
    "phase_distribution": dict(phase_cnt),
    "per_frame": records,
}
jp = OUTPUT_DIR / "probe_t24.json"
with open(jp,'w') as f: json.dump(result,f,indent=2)

import shutil
shutil.copy2(str(plot_path), str(WIN_OUT / "probe_temporal_t24.jpg"))
print(f"[OUT] {WIN_OUT}/probe_temporal_t24.jpg")

# Extract thumbnail frames for each flicker candidate
print("\n[FRAMES] Extracting flicker candidate thumbnails...")
fc_dir = OUTPUT_DIR / "flicker_cands"
fc_dir.mkdir(exist_ok=True)
fc_win = WIN_OUT / "flicker_cands"
fc_win.mkdir(exist_ok=True)

for fc in flicker_cands:
    fi = fc["frame"]
    if fi < len(src):
        frame = src[fi].copy()
        # Draw frame number
        cv2.putText(frame, f"fr{fi:03d}", (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,200,255), 3)
        reason_str = " | ".join(fc["reasons"])[:60]
        cv2.putText(frame, reason_str, (10,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
        out_p = fc_dir / f"cand_fr{fi:03d}.jpg"
        cv2.imwrite(str(out_p), frame)
        shutil.copy2(str(out_p), str(fc_win / f"cand_fr{fi:03d}.jpg"))

print(f"  {len(flicker_cands)} thumbnails → {fc_win}")
print(f"\nT2.4 PROBE DONE — {len(flicker_cands)} flicker candidates")
print(f"  阶段分布: {dict(phase_cnt)}")
