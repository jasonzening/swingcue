"""
ghost004_step2_retarget_v2.py  —  GHOST-004 Step 2 修复版 (root 对齐 + delta 迁移)

根因修复:
  原版错误: coach NPZ 里 pred_global_rots[:,0] 全程为 identity → 转 Euler 恒为 [0,0,0]
            → ghost root 朝向固定在 rest-pose，body_pose 在本地坐标转，两套分裂

  正确做法:
    1. 重跑 MHR on coach-fo 每帧，取真实 global_rot (3,) Euler ZYX
    2. 取用户 fr0 global_rot (3,) 作为用户 address 基准朝向
    3. delta 法:
         R_coach_addr = euler_to_rotmat(coach_global_rot[COACH_ADDR])
         R_coach_fi   = euler_to_rotmat(coach_global_rot[fi])
         R_delta      = R_coach_addr.T @ R_coach_fi      # 教练相对 address 的旋转变化量
         R_retarget   = R_user_addr @ R_delta            # 叠加到用户 address 朝向上
    4. body_pose_params (133,) 直接从 coach 搬到用户 (同一 MHR 模型，local joint convention 一致)
    5. 用户 β: shape_params (45,) + scale_params (28,) 从用户 fr0 提取

  单帧探针模式 (PROBE_ONLY=True):
    只渲染 address + top 两帧 + 用户 address 参考 → 三帧并排
    address retarget 应和用户 address 重合
    top retarget 应做出教练 top 姿态但站在用户位置
    两帧通过后改 PROBE_ONLY=False 跑全序列

运行:
  /home/jason/projects/sam3d_venv/bin/python3 ghost004_step2_retarget_v2.py
"""
import os, sys, json, time
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.1"

import numpy as np
import cv2
import torch
from pathlib import Path
from scipy.signal import savgol_filter

sys.path.insert(0, "/home/jason/projects/sam-3d-body")
import roma
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from sam_3d_body.visualization.renderer import Renderer

# ──────────────────────────────────────────────────────────────────────────────
PROBE_ONLY = True   # True = 只渲出 address+top 两帧对比图; False = 全序列视频
# ──────────────────────────────────────────────────────────────────────────────

ROOT        = Path("/home/jason/projects/swingcue-postest")
COACH_VIDEO = Path("/mnt/c/Users/jason/Zening/Swingcue/教学视频/coach-video/coach-fo.mp4")
COACH_KP    = ROOT / "engine/kp_cache/ghost004/coach-fo.json"
COACH_NPZ   = ROOT / "output/ghost004/fo_pose_sequence_aligned.npz"
USER_VIDEO  = ROOT / "input/fo-ok-1.mp4"
USER_KP     = ROOT / "engine/kp_cache/batch2/fo-ok-1.json"
OUT_DIR     = ROOT / "output/ghost004"
WIN_OUT     = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/ghost004")
CKPT        = Path(os.path.expanduser("~/.cache/sam3d/sam-3d-body-dinov3/model.ckpt"))
MHR_PT      = CKPT.parent / "assets/mhr_model.pt"

OUT_DIR.mkdir(parents=True, exist_ok=True)
WIN_OUT.mkdir(parents=True, exist_ok=True)

GHOST_ALPHA = 0.55
GHOST_COLOR = (0.85, 0.1, 0.1)
SG_WINDOW   = 9
SG_POLY     = 3
USER_FR_ADDR = 0

KP_ORDER = ["nose","left_eye","right_eye","left_ear","right_ear",
            "left_shoulder","right_shoulder","left_elbow","right_elbow",
            "left_wrist","right_wrist","left_hip","right_hip",
            "left_knee","right_knee","left_ankle","right_ankle"]

# ── helpers ───────────────────────────────────────────────────────────────────
def load_kp_cache(path):
    with open(path) as f: d = json.load(f)
    out = {}
    for fe in d["frames"]:
        fi = fe["frame"]
        if not fe.get("persons"): continue
        kd = fe["persons"][0]["keypoints"]
        out[fi] = np.array([[kd[n]["x"],kd[n]["y"],kd[n]["score"]] for n in KP_ORDER if n in kd], dtype=np.float32)
    return out

def get_bbox(kp2d, H, W, pad=0.15):
    ax,ay = kp2d[:,0],kp2d[:,1]
    px=(max(ax)-min(ax))*pad; py=(max(ay)-min(ay))*pad
    return np.array([[max(0,min(ax)-px),max(0,min(ay)-py),min(W,max(ax)+px),min(H,max(ay)+py)]], dtype=np.float32)

def euler_to_R(euler_zyx_np):
    """numpy (3,) ZYX euler → torch (3,3) rotmat"""
    t = torch.from_numpy(euler_zyx_np.astype(np.float32)).unsqueeze(0)
    return roma.euler_to_rotmat("ZYX", t).squeeze(0)

def R_to_euler(R_3x3_torch):
    """torch (3,3) → numpy (3,) euler ZYX"""
    return roma.rotmat_to_euler("ZYX", R_3x3_torch.unsqueeze(0)).squeeze(0).numpy()

def render_and_blend(verts, cam_t, focal, faces, bg, H, W, alpha=GHOST_ALPHA):
    black_bg = np.zeros((H,W,3), dtype=np.uint8)
    rend = Renderer(focal_length=focal, faces=faces)
    out  = rend(verts, cam_t, black_bg, mesh_base_color=GHOST_COLOR, scene_bg_color=(0,0,0))
    out_u8 = (out*255).clip(0,255).astype(np.uint8)
    mask = np.any(out_u8>5, axis=2)
    res  = bg.copy().astype(np.float32)
    res[mask] = (1-alpha)*res[mask] + alpha*out_u8[mask].astype(np.float32)
    return res.astype(np.uint8)

def label(img, text, y=35, color=(255,255,80)):
    cv2.putText(img, text, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return img

# ── INIT ──────────────────────────────────────────────────────────────────────
print("[INIT] Loading model...")
t0 = time.time()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, model_cfg = load_sam_3d_body(str(CKPT), device=str(dev), mhr_path=str(MHR_PT))
est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=model_cfg,
                         human_detector=None, human_segmentor=None, fov_estimator=None)
est.model.eval()
mhr_head = est.model.head_pose
faces    = est.faces
print(f"  {time.time()-t0:.1f}s")

# ── LOAD VIDEOS ───────────────────────────────────────────────────────────────
def read_video(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret,f = cap.read()
        if not ret: break
        frames.append(f)
    cap.release()
    return frames, fps

print("\n[LOAD] Videos...")
coach_src, coach_fps = read_video(COACH_VIDEO)
user_src,  user_fps  = read_video(USER_VIDEO)
H = user_src[0].shape[0]; W = user_src[0].shape[1]
print(f"  coach: NF={len(coach_src)}  user: NF={len(user_src)}  {W}x{H}")

coach_kp = load_kp_cache(str(COACH_KP))
user_kp  = load_kp_cache(str(USER_KP))

# ── LOAD COACH NPZ (body_pose_params) ─────────────────────────────────────────
coach_npz    = np.load(str(COACH_NPZ), allow_pickle=False)
COACH_NF     = coach_npz["body_pose_params"].shape[0]
COACH_ADDR   = int(coach_npz["anchors_address"])
COACH_TOP    = int(coach_npz["anchors_top"])
COACH_IMPACT = int(coach_npz["anchors_impact"])
COACH_FINISH = int(coach_npz["anchors_finish"])
print(f"\n[COACH NPZ] NF={COACH_NF}  addr={COACH_ADDR}  top={COACH_TOP}  impact={COACH_IMPACT}")

# ── PASS A: 用户 β + address global_rot ───────────────────────────────────────
print("\n[USER-BETA] Extracting user β from fr0...")
kp_u = user_kp[USER_FR_ADDR]
bbox_u = get_bbox(kp_u[:,:2], H, W)
with torch.no_grad():
    outs_u = est.process_one_image(user_src[USER_FR_ADDR], bboxes=bbox_u, use_mask=False, inference_type="body")
u0 = outs_u[0]

user_shape       = u0["shape_params"]           # (45,)
user_scale       = u0["scale_params"]           # (28,)
user_expr        = u0["expr_params"]            # (72,)
user_cam_t       = u0["pred_cam_t"]             # (3,)
user_focal       = float(u0["focal_length"])
user_global_rot  = u0["global_rot"]             # (3,) Euler ZYX  ← 用户 address 基准朝向
user_verts_addr  = u0["pred_vertices"]          # reference mesh

R_user_addr = euler_to_R(user_global_rot)       # (3,3) torch
print(f"  user global_rot (addr): {user_global_rot}")
print(f"  user cam_t: {user_cam_t}  focal: {user_focal:.1f}")

u_shape_t  = torch.from_numpy(user_shape).unsqueeze(0).to(dev)   # (1,45)
u_scale_t  = torch.from_numpy(user_scale).unsqueeze(0).to(dev)   # (1,28)
u_expr_t   = torch.from_numpy(user_expr).unsqueeze(0).to(dev)    # (1,72) — use user expr (neutral)
u_gtrans_t = torch.zeros(1,3,dtype=torch.float32,device=dev)
u_hand_t   = torch.zeros(1,108,dtype=torch.float32,device=dev)   # neutral hands v1

# ── PASS B: coach global_rot 序列 (重新 MHR 推理取真实 global_rot) ─────────────
cache_path = OUT_DIR / "coach_fo_global_rot.npy"
if cache_path.exists():
    print(f"\n[COACH-GROT] Loading cache: {cache_path.name}")
    coach_global_rot_seq = np.load(str(cache_path))  # (NF_coach, 3)
else:
    print(f"\n[COACH-GROT] Re-running MHR on coach-fo ({len(coach_src)} frames)...")
    coach_global_rot_seq = np.zeros((len(coach_src), 3), dtype=np.float32)
    t_cg = time.time()
    for fi in range(len(coach_src)):
        kp = coach_kp.get(fi)
        if kp is None: continue
        bbox = get_bbox(kp[:,:2], H, W)
        with torch.no_grad():
            outs = est.process_one_image(coach_src[fi], bboxes=bbox, use_mask=False, inference_type="body")
        if outs:
            coach_global_rot_seq[fi] = outs[0]["global_rot"]
        if fi % 20 == 0:
            print(f"  fr{fi:03d}  global_rot={coach_global_rot_seq[fi]}")
    np.save(str(cache_path), coach_global_rot_seq)
    print(f"  done {time.time()-t_cg:.1f}s  saved: {cache_path.name}")

# Verify: coach address frame global_rot
print(f"\nCoach address (fr{COACH_ADDR}) global_rot: {coach_global_rot_seq[COACH_ADDR]}")
print(f"Coach top     (fr{COACH_TOP}) global_rot:     {coach_global_rot_seq[COACH_TOP]}")
R_coach_addr = euler_to_R(coach_global_rot_seq[COACH_ADDR])   # (3,3) torch

# ── DELTA RETARGET FUNCTION ───────────────────────────────────────────────────
def retarget_frame(fi_coach, label_text=""):
    """
    Retarget coach frame fi_coach using delta root alignment.
    Returns (verts_np, cam_t, focal) or None on failure.
    """
    ok_flag = bool(coach_npz["frame_ok"][fi_coach]) if fi_coach < COACH_NF else True

    # Coach theta for this frame
    if fi_coach < COACH_NF:
        c_body_pose = coach_npz["body_pose_params"][fi_coach]   # (133,) local Euler
    else:
        c_body_pose = coach_npz["body_pose_params"][COACH_NF-1]

    c_grot_np = coach_global_rot_seq[fi_coach]           # (3,) global Euler ZYX (true)
    R_coach_fi = euler_to_R(c_grot_np)                   # (3,3)

    # Delta: rotation change from coach's address to fi (in coach's local frame)
    R_delta = R_coach_addr.T @ R_coach_fi                # (3,3)  coach address.inv() × coach_fi
    # Apply delta to user's address orientation
    R_retarget = R_user_addr @ R_delta                   # (3,3)  user_addr × delta
    euler_retarget = torch.from_numpy(R_to_euler(R_retarget).astype(np.float32))

    # Tensors
    c_body_t   = torch.from_numpy(c_body_pose[:130].astype(np.float32)).unsqueeze(0).to(dev)
    c_grot_t   = euler_retarget.unsqueeze(0).to(dev)

    with torch.no_grad():
        result = mhr_head.mhr_forward(
            global_trans=u_gtrans_t,
            global_rot=c_grot_t,
            body_pose_params=c_body_t,
            hand_pose_params=u_hand_t,
            scale_params=u_scale_t,
            shape_params=u_shape_t,
            expr_params=u_expr_t,
            do_pcblend=True,
            return_keypoints=False,
        )
    verts = result[0] if isinstance(result, tuple) else result
    verts_np = verts.squeeze(0).cpu().numpy()
    verts_np[..., [1,2]] *= -1   # SAM3D camera convention

    raw_euler = R_to_euler(R_coach_fi)
    delta_euler = R_to_euler(R_delta)
    print(f"  fr{fi_coach:03d} {label_text}")
    print(f"    coach_grot_raw={c_grot_np}  delta={delta_euler}  retarget={euler_retarget.numpy()}")
    return verts_np, user_cam_t.copy(), user_focal

# ── MODE: PROBE (address + top 单帧验证) ────────────────────────────────────────
if PROBE_ONLY:
    print(f"\n{'='*60}")
    print("PROBE MODE: address + top single-frame retarget")
    print(f"{'='*60}")

    panels = []
    for fi_coach, ph_name in [(COACH_ADDR, "address"), (COACH_TOP, "top")]:
        result = retarget_frame(fi_coach, ph_name)
        verts_rt, cam_t_rt, focal_rt = result

        # Retarget render on user address bg
        fi_user = min(fi_coach, len(user_src)-1)
        bg = user_src[USER_FR_ADDR].copy()
        rt_img = render_and_blend(verts_rt, cam_t_rt, focal_rt, faces, bg, H, W)
        label(rt_img, f"retarget: user-body + coach-{ph_name}")

        # User original render (reference)
        ref = user_src[USER_FR_ADDR].copy()
        ref_img = render_and_blend(user_verts_addr, user_cam_t, user_focal, faces, ref, H, W)
        label(ref_img, f"user-body-only (reference)")

        # Side by side
        sc = 0.48
        lh,lw = int(H*sc),int(W*sc)
        row = np.hstack([cv2.resize(rt_img,(lw,lh)), cv2.resize(ref_img,(lw,lh))])
        cv2.putText(row, f"{ph_name.upper()}  coach fr{fi_coach:03d}",
                    (10, lh-15), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 3, cv2.LINE_AA)
        panels.append(row)

        print(f"  {ph_name} done  verts range x=[{verts_rt[:,0].min():.3f},{verts_rt[:,0].max():.3f}]")

    grid = np.vstack(panels)
    out_path = OUT_DIR / "retarget_v2_probe.jpg"
    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 93])
    import shutil
    shutil.copy2(str(out_path), str(WIN_OUT / out_path.name))
    print(f"\nProbe saved: {out_path}")
    print(f"Windows:     retarget_v2_probe.jpg")
    print("\n--- PROBE CHECKLIST ---")
    print("  1. address row: retarget ghost (left) should OVERLAP user reference (right)")
    print("  2. top row:     retarget ghost should show coach TOP posture at user's position")
    print("  If both OK: set PROBE_ONLY=False and re-run for full sequence")

# ── MODE: FULL SEQUENCE ────────────────────────────────────────────────────────
else:
    print(f"\n{'='*60}")
    print(f"FULL SEQUENCE MODE: retarget {COACH_NF} frames")
    print(f"{'='*60}")

    # Build retarget results
    rt_verts = []
    rt_cam_t = []
    rt_focal  = []
    rt_ok     = []

    for fi in range(COACH_NF):
        ok = bool(coach_npz["frame_ok"][fi])
        if ok:
            v, ct, fo = retarget_frame(fi)
            rt_verts.append(v); rt_cam_t.append(ct); rt_focal.append(fo); rt_ok.append(True)
        else:
            rt_verts.append(None); rt_cam_t.append(user_cam_t.copy())
            rt_focal.append(user_focal); rt_ok.append(False)
        if fi % 10 == 0:
            print(f"  fr{fi:03d}/{COACH_NF}")

    # Interpolate failed frames
    for fi in range(COACH_NF):
        if rt_ok[fi]: continue
        pv = next((f for f in range(fi-1,-1,-1) if rt_ok[f]), None)
        nv = next((f for f in range(fi+1,COACH_NF) if rt_ok[f]), None)
        if pv is not None and nv is not None:
            w=(fi-pv)/(nv-pv); rt_verts[fi]=(1-w)*rt_verts[pv]+w*rt_verts[nv]; rt_ok[fi]=True
        elif pv is not None:
            rt_verts[fi]=rt_verts[pv].copy(); rt_ok[fi]=True
        elif nv is not None:
            rt_verts[fi]=rt_verts[nv].copy(); rt_ok[fi]=True

    # SG smooth cam_t (z already locked)
    ok_idx = [fi for fi in range(COACH_NF) if rt_ok[fi]]
    sw = min(SG_WINDOW, len(ok_idx) if len(ok_idx)%2==1 else len(ok_idx)-1)
    ctx_sm = savgol_filter([rt_cam_t[fi][0] for fi in ok_idx], sw, SG_POLY, mode="mirror")
    cty_sm = savgol_filter([rt_cam_t[fi][1] for fi in ok_idx], sw, SG_POLY, mode="mirror")
    foc_sm = savgol_filter([rt_focal[fi] for fi in ok_idx], sw, SG_POLY, mode="mirror")
    for idx,fi in enumerate(ok_idx):
        rt_cam_t[fi] = np.array([ctx_sm[idx],cty_sm[idx],float(user_cam_t[2])], dtype=np.float32)
        rt_focal[fi] = float(foc_sm[idx])

    # Render
    def phase_label(fi):
        if fi <= COACH_ADDR: return "address"
        elif fi <= COACH_TOP:
            sp=COACH_TOP-COACH_ADDR; r=fi-COACH_ADDR
            return "takeaway" if r<sp*0.3 else "backswing" if r<sp*0.8 else "top"
        elif COACH_IMPACT-1<=fi<=COACH_IMPACT+2: return "impact"
        elif fi<=COACH_IMPACT: return "downswing"
        else: return "follow_through"

    raw_mp4 = OUT_DIR / "retarget_v2_raw.mp4"
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(str(raw_mp4), fourcc, user_fps, (W,H))
    print(f"\n[RENDER] {COACH_NF} frames...")
    t_ren = time.time()
    for fi in range(COACH_NF):
        uf = min(fi, len(user_src)-1)
        bg = user_src[uf].copy()
        if rt_ok[fi]:
            bg = render_and_blend(rt_verts[fi], rt_cam_t[fi], rt_focal[fi], faces, bg, H, W)
        label(bg, f"user-body+coach-motion  {phase_label(fi)}  fr{fi:03d}/{COACH_NF-1}")
        writer.write(bg)
        if fi%20==0: print(f"  fr{fi:03d}/{COACH_NF}")
    writer.release()
    print(f"  render {time.time()-t_ren:.1f}s")

    final_mp4  = OUT_DIR / "retarget_fo_playback_v2.mp4"
    slowmo_mp4 = OUT_DIR / "retarget_fo_playback_v2_025x.mp4"
    os.system(f'ffmpeg -y -i "{raw_mp4}" -c:v libx264 -crf 18 -pix_fmt yuv420p "{final_mp4}" -loglevel error')
    os.system(f'ffmpeg -y -i "{raw_mp4}" -vf "setpts=4*PTS" -c:v libx264 -crf 18 -pix_fmt yuv420p "{slowmo_mp4}" -loglevel error')
    raw_mp4.unlink(missing_ok=True)

    # Keyframe stills
    kf_panels = []
    for ph_name, fi_c in [("address",COACH_ADDR),("top",COACH_TOP),
                           ("impact",COACH_IMPACT),("follow",min(COACH_IMPACT+10,COACH_NF-1))]:
        uf = min(fi_c, len(user_src)-1)
        bg = user_src[uf].copy()
        if rt_ok[fi_c]:
            bg = render_and_blend(rt_verts[fi_c], rt_cam_t[fi_c], rt_focal[fi_c], faces, bg, H, W)
        label(bg, f"{ph_name}  fr{fi_c:03d}")
        ref = user_src[USER_FR_ADDR].copy()
        ref = render_and_blend(user_verts_addr, user_cam_t, user_focal, faces, ref, H, W)
        label(ref, "user-addr ref")
        sc=0.47
        lh,lw=int(H*sc),int(W*sc)
        kf_panels.append(np.hstack([cv2.resize(bg,(lw,lh)),cv2.resize(ref,(lw,lh))]))

    kf_grid = np.vstack(kf_panels)
    kf_path = OUT_DIR / "retarget_v2_keyframes.jpg"
    cv2.imwrite(str(kf_path), kf_grid, [cv2.IMWRITE_JPEG_QUALITY, 92])

    import shutil
    for f in [final_mp4, slowmo_mp4, kf_path]:
        shutil.copy2(str(f), str(WIN_OUT / Path(f).name))
        print(f"  -> Windows: {Path(f).name}")

    print(f"\nFull sequence complete.  ok={sum(rt_ok)}/{COACH_NF}")
    print("Windows: retarget_fo_playback_v2.mp4 / _025x.mp4 / _keyframes.jpg")
