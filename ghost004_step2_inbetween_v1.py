"""
ghost004_step2_inbetween_v1.py  —  GHOST-004 阶段二: Pose Inbetweening

框架: 教练挥杆提供离散关键姿态锚点, ghost 按学员节奏 SLERP 插值过渡
  - 教练: 提供 θ 目标 (address/takeaway/top/impact/follow 5个锚点)
  - 学员: 提供时间轴 (每个 phase 的帧数/速度)
  - 插值: global_rot 用四元数 SLERP, body_pose 用 LERP (v1)
  - 体型: 全程 user β 锁定
  - root:  delta 法对齐 (沿用 v2 成功方案)

输出:
  retarget_inbetween_v1.mp4      正常速
  retarget_inbetween_v1_025x.mp4 0.25x 慢放
  inbetween_v1_keyframes.jpg     8-phase 定格对比图

运行:
  /home/jason/projects/sam3d_venv/bin/python3 ghost004_step2_inbetween_v1.py
"""
import os, sys, json, time, shutil
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

# ── PATHS ─────────────────────────────────────────────────────────────────────
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
GROT_CACHE  = OUT_DIR / "coach_fo_global_rot.npy"

OUT_DIR.mkdir(parents=True, exist_ok=True)
WIN_OUT.mkdir(parents=True, exist_ok=True)

GHOST_ALPHA = 0.6
GHOST_COLOR = (0.85, 0.1, 0.1)
USER_FR_ADDR = 0

KP_ORDER = ["nose","left_eye","right_eye","left_ear","right_ear",
            "left_shoulder","right_shoulder","left_elbow","right_elbow",
            "left_wrist","right_wrist","left_hip","right_hip",
            "left_knee","right_knee","left_ankle","right_ankle"]

# ── HELPERS ────────────────────────────────────────────────────────────────────
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

def get_bbox(kp_xy, H, W, pad=0.15):
    x1,y1 = kp_xy[:,0].min(), kp_xy[:,1].min()
    x2,y2 = kp_xy[:,0].max(), kp_xy[:,1].max()
    pw=(x2-x1)*pad; ph=(y2-y1)*pad
    return np.array([[max(0,x1-pw), max(0,y1-ph), min(W,x2+pw), min(H,y2+ph)]])

def euler_to_R(euler_np):
    e = torch.from_numpy(euler_np.astype(np.float32)).unsqueeze(0)
    return roma.euler_to_rotmat("ZYX", e)[0]

def R_to_quat(R):  # (3,3) → (4,) xyzw
    return roma.rotmat_to_unitquat(R.unsqueeze(0)).squeeze(0)

def quat_to_euler(q):  # (4,) → (3,) ZYX
    R = roma.unitquat_to_rotmat(q.unsqueeze(0)).squeeze(0)
    return roma.rotmat_to_euler("ZYX", R.unsqueeze(0)).squeeze(0).cpu().numpy()

def quat_slerp(q0, q1, t):
    """Quaternion SLERP: q0,q1 are (4,) tensors, t in [0,1]"""
    # ensure same hemisphere
    dot = (q0 * q1).sum()
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = dot.clamp(-1, 1)
    if dot > 0.9995:
        return torch.nn.functional.normalize(q0 + t * (q1 - q0), dim=0)
    theta_0 = torch.acos(dot)
    theta   = theta_0 * t
    sin0    = torch.sin(theta_0)
    sin_t   = torch.sin(theta)
    sin_rem = torch.sin(theta_0 - theta)
    return (sin_rem / sin0) * q0 + (sin_t / sin0) * q1

def read_video(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
    cap.release()
    return frames, fps

def render_and_blend(verts, cam_t, focal, faces, bg, H, W, alpha=GHOST_ALPHA):
    black = np.zeros((H, W, 3), dtype=np.uint8)
    rend  = Renderer(focal_length=focal, faces=faces)
    out   = rend(verts, cam_t, black, mesh_base_color=GHOST_COLOR, scene_bg_color=(0,0,0))
    out_u8 = (out * 255).clip(0, 255).astype(np.uint8)
    mask  = np.any(out_u8 > 5, axis=2)
    res   = bg.copy().astype(np.float32)
    res[mask] = (1-alpha)*res[mask] + alpha*out_u8[mask].astype(np.float32)
    return res.astype(np.uint8)

# ── LOAD MODEL ─────────────────────────────────────────────────────────────────
print("[INIT] Loading SAM3D Body model...")
t0 = time.time()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, model_cfg = load_sam_3d_body(str(CKPT), device=str(dev), mhr_path=str(MHR_PT))
est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=model_cfg,
                         human_detector=None, human_segmentor=None, fov_estimator=None)
est.model.eval()
mhr_head = est.model.head_pose
faces    = est.faces
print(f"  {time.time()-t0:.1f}s")

# ── LOAD DATA ──────────────────────────────────────────────────────────────────
print("\n[LOAD] Videos + KP cache...")
coach_src, coach_fps = read_video(COACH_VIDEO)
user_src,  user_fps  = read_video(USER_VIDEO)
H = user_src[0].shape[0]; W = user_src[0].shape[1]
USER_NF = len(user_src)
print(f"  coach={len(coach_src)} user={USER_NF} {W}x{H} fps={user_fps:.2f}")

coach_kp = load_kp_cache(str(COACH_KP))
user_kp  = load_kp_cache(str(USER_KP))

coach_npz = np.load(str(COACH_NPZ), allow_pickle=False)
COACH_ADDR   = int(coach_npz["anchors_address"])
COACH_TOP    = int(coach_npz["anchors_top"])
COACH_IMPACT = int(coach_npz["anchors_impact"])
COACH_FINISH = int(coach_npz["anchors_finish"])

# coach phase_map 里的 takeaway start
# 从 phase_report_step1.json 读
phase_report = json.load(open(str(OUT_DIR / "phase_report_step1.json")))
fo_phases    = phase_report["fo"]["phase_map"]
COACH_TAKEAWAY = fo_phases["takeaway"]["start"]  # 13

print(f"  coach anchors: addr={COACH_ADDR} takeaway={COACH_TAKEAWAY} top={COACH_TOP} impact={COACH_IMPACT} finish={COACH_FINISH}")

coach_global_rot_seq = np.load(str(GROT_CACHE))  # (NF_coach_raw, 3)
print(f"  global_rot cache: {coach_global_rot_seq.shape}")

# ── EXTRACT USER β (fr0) ───────────────────────────────────────────────────────
print(f"\n[USER-BETA] fr{USER_FR_ADDR}...")
kp_u   = user_kp.get(USER_FR_ADDR)
bbox_u = get_bbox(kp_u[:,:2], H, W)
with torch.no_grad():
    outs_u = est.process_one_image(user_src[USER_FR_ADDR], bboxes=bbox_u,
                                   use_mask=False, inference_type="body")
u0 = outs_u[0]
user_shape = u0["shape_params"]   # (45,)
user_scale = u0["scale_params"]   # (28,)
user_expr  = u0["expr_params"]    # (72,)
user_cam_t = u0["pred_cam_t"]     # (3,)
user_focal = float(u0["focal_length"])
user_grot_addr_np = u0["global_rot"]  # (3,) ZYX Euler
R_user_addr = euler_to_R(user_grot_addr_np)
q_user_addr = R_to_quat(R_user_addr)
print(f"  cam_t={user_cam_t}  focal={user_focal:.1f}")
print(f"  global_rot={user_grot_addr_np}")

u_shape_t = torch.from_numpy(user_shape.astype(np.float32)).unsqueeze(0).to(dev)
u_scale_t = torch.from_numpy(user_scale.astype(np.float32)).unsqueeze(0).to(dev)
u_expr_t  = torch.from_numpy(user_expr.astype(np.float32)).unsqueeze(0).to(dev)
u_hand_t  = torch.zeros(1, 108, dtype=torch.float32, device=dev)
u_gtrans_t= torch.zeros(1, 3,   dtype=torch.float32, device=dev)

# ── DETECT USER PHASE ANCHORS ──────────────────────────────────────────────────
print("\n[USER-PHASE] Detecting user phase anchors from wrist trajectory...")

# 提取右手腕 y 坐标序列 (RTMPose index 10 = right_wrist)
RWRIST_IDX = 10  # right_wrist in KP_ORDER
wrist_y = np.full(USER_NF, np.nan)
for fi in range(USER_NF):
    kp = user_kp.get(fi)
    if kp is not None and len(kp) > RWRIST_IDX:
        wrist_y[fi] = kp[RWRIST_IDX, 1]

# 平滑 wrist_y 减少噪声
wrist_y_valid = wrist_y.copy()
nans = np.isnan(wrist_y_valid)
if nans.any():
    x = np.arange(USER_NF)
    wrist_y_valid[nans] = np.interp(x[nans], x[~nans], wrist_y_valid[~nans])
wrist_y_sm = savgol_filter(wrist_y_valid, 9, 3)

# address 静止段: wrist_y 基本不变, 取最后静止帧
addr_level = float(np.median(wrist_y_sm[:20]))  # address 基准水平
MOVE_THRESH = 15.0  # px
takeaway_start = 0
for fi in range(10, USER_NF):
    if abs(wrist_y_sm[fi] - addr_level) > MOVE_THRESH:
        takeaway_start = fi
        break
# 向前找最后一个静止帧
addr_end = max(0, takeaway_start - 1)

# top (上杆最高点) = wrist_y 局部最小值 (wrist 最高 -> y 最小)
# 约束搜索窗口: takeaway 后最多 55 帧内找第一个局部最小
# (避免搜到 follow-through 的更低点)
search_top_end = min(takeaway_start + 55, USER_NF - 1)
search_top = wrist_y_sm[takeaway_start: search_top_end]
# 找第一个局部最小 (左右各 3 帧都比它大)
top_rel = None
for i in range(3, len(search_top) - 3):
    if all(search_top[i] <= search_top[i-k] for k in range(1,4)) and \
       all(search_top[i] <= search_top[i+k] for k in range(1,4)):
        top_rel = i
        break
if top_rel is None:
    top_rel = int(np.argmin(search_top))  # fallback
top_frame = takeaway_start + top_rel

# impact = top 之后, wrist_y 最大值 (wrist 最低 = 击球), 在 top 后搜索
search_impact = wrist_y_sm[top_frame: top_frame + 35]
impact_rel = int(np.argmax(search_impact))
impact_frame = top_frame + impact_rel

# follow_through peak = impact 之后 wrist_y 再次最小
search_follow = wrist_y_sm[impact_frame: impact_frame + 35]
follow_rel = int(np.argmin(search_follow))
follow_frame = impact_frame + follow_rel

# 最后帧
end_frame = USER_NF - 1

print(f"  address_level={addr_level:.1f}px  move_thresh={MOVE_THRESH}px")
print(f"  user anchor frames:")
print(f"    address_end  = fr{addr_end:03d}  (static end)")
print(f"    takeaway     = fr{takeaway_start:03d}")
print(f"    top          = fr{top_frame:03d}  (wrist_y={wrist_y_sm[top_frame]:.1f})")
print(f"    impact       = fr{impact_frame:03d}  (wrist_y={wrist_y_sm[impact_frame]:.1f})")
print(f"    follow_peak  = fr{follow_frame:03d}  (wrist_y={wrist_y_sm[follow_frame]:.1f})")
print(f"    end          = fr{end_frame:03d}")

# ── BUILD ANCHOR MAPPING ────────────────────────────────────────────────────────
# 5 coach θ 锚点 → 5 user 时间轴帧
# 锚点: (user_frame, coach_frame, label)
ANCHORS = [
    (0,              COACH_ADDR,     "address"),
    (takeaway_start, COACH_TAKEAWAY, "takeaway"),
    (top_frame,      COACH_TOP,      "top"),
    (impact_frame,   COACH_IMPACT,   "impact"),
    (follow_frame,   COACH_FINISH,   "follow"),
    (end_frame,      COACH_FINISH,   "end"),   # 结尾延续 finish 姿态
]

print("\n[ANCHORS] User→Coach mapping:")
for uf, cf, label in ANCHORS:
    print(f"  {label:12s}: user fr{uf:03d} → coach fr{cf:03d}")

# ── COMPUTE COACH ANCHOR DATA ──────────────────────────────────────────────────
print("\n[COACH-ANCHORS] Extracting coach θ + global_rot at anchor frames...")
# body_pose_params 直接从 NPZ
# global_rot 从 cache (NF_raw=97, 对应 aligned NF=75 的 fr0~fr74 = raw coach fr12~fr74+12=fr86?)
# 注意: aligned NPZ 裁取的是 coach 原始帧 fr12~fr74 (75帧), grot_cache shape=(97,3)
# aligned fr_i → coach_raw fr = fr_i + COACH_ADDR_ORIGINAL
# COACH_ADDR in aligned = fr12, 对应 raw cache index 12
# 所以 aligned[k] → grot_cache[k + raw_addr_offset]
# 但是 NPZ coach 里的 anchors 是 aligned 里的帧号 (addr=12, top=43, impact=52, finish=74)
# 而 grot_cache 是 raw coach video 的帧号 (原始 97 帧)
# 从 ghost004_step2_retarget_v2.py 的做法看: coach_global_rot_seq[COACH_ADDR] = grot_cache[12]
# 这说明 aligned NPZ 里的帧号 = coach_raw 帧号 (aligned 裁的是 coach raw fr0..74, 不是 fr12..74+12)
# 确认: aligned NF=75, coach_raw NF=97, 锚点 addr=12 说明 raw coach fr12 = aligned fr12

# 教练锚点 (coach_fi → 直接 indexing grot_cache 和 NPZ)
def get_coach_anchor(coach_fi):
    """coach_fi 是 aligned NPZ 里的帧号 (同 grot_cache 的 raw 帧号)"""
    # body_pose from aligned NPZ
    nf_aligned = coach_npz["body_pose_params"].shape[0]
    fi_safe = min(int(coach_fi), nf_aligned - 1)
    body_pose = coach_npz["body_pose_params"][fi_safe]  # (133,)
    # global_rot from grot_cache
    nf_cache = coach_global_rot_seq.shape[0]
    fi_cache  = min(int(coach_fi), nf_cache - 1)
    grot = coach_global_rot_seq[fi_cache]  # (3,) ZYX Euler
    return body_pose, grot

# 预计算每个锚点的 θ 和 quaternion
anchor_data = []
R_coach_addr_R = euler_to_R(coach_global_rot_seq[COACH_ADDR])
q_coach_addr   = R_to_quat(R_coach_addr_R)

for uf, cf, label in ANCHORS:
    body_pose, grot = get_coach_anchor(cf)
    R_coach_fi = euler_to_R(grot)
    # 四元数 delta → 用户 address 基准的 retargeted rotation
    q_cf = R_to_quat(R_coach_fi)
    q_ca = R_to_quat(R_coach_addr_R)
    q_delta = roma.quat_product(roma.quat_inverse(q_ca.unsqueeze(0)),
                                q_cf.unsqueeze(0)).squeeze(0)
    q_retarget = roma.quat_product(q_user_addr.unsqueeze(0),
                                   q_delta.unsqueeze(0)).squeeze(0)
    anchor_data.append({
        "user_frame":  uf,
        "coach_frame": cf,
        "label":       label,
        "body_pose":   body_pose,    # (133,) Euler angles
        "q_retarget":  q_retarget,   # (4,) quat for global_rot
        "grot_euler":  grot,         # coach original (for logging)
    })
    print(f"  {label:12s}: coach fr{cf:03d} grot={grot}  q_ret={q_retarget.cpu().numpy().round(4)}")

# ── INBETWEEN FUNCTION ─────────────────────────────────────────────────────────
def get_pose_at_user_frame(fi):
    """
    给定学员帧 fi, 返回 (body_pose_lerp, q_retarget_slerp)
    在相邻两个锚点间做 LERP/SLERP
    """
    # 找所在区间 [A, B]
    A = anchor_data[0]
    B = anchor_data[-1]
    for i in range(len(anchor_data) - 1):
        if anchor_data[i]["user_frame"] <= fi <= anchor_data[i+1]["user_frame"]:
            A = anchor_data[i]
            B = anchor_data[i+1]
            break

    span = B["user_frame"] - A["user_frame"]
    if span <= 0:
        t = 0.0
    else:
        t = float(fi - A["user_frame"]) / float(span)
    t = np.clip(t, 0.0, 1.0)
    t_f = float(t)

    # body_pose: LERP (Euler angles)
    body_lerp = (1 - t_f) * A["body_pose"] + t_f * B["body_pose"]

    # global_rot: quaternion SLERP
    q_slerp = quat_slerp(A["q_retarget"], B["q_retarget"], torch.tensor(t_f, dtype=torch.float32))

    return body_lerp, q_slerp

# ── MHR FORWARD ───────────────────────────────────────────────────────────────
def run_mhr_from_quat(body_pose_133, q_rot):
    """渲染 ghost: 给定 body_pose (133,) + q_rot (4,) 返回 verts_np"""
    euler_g = quat_to_euler(q_rot)
    c_body_t = torch.from_numpy(body_pose_133[:130].astype(np.float32)).unsqueeze(0).to(dev)
    c_grot_t = torch.from_numpy(euler_g.astype(np.float32)).unsqueeze(0).to(dev)
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
    v = verts.squeeze(0).cpu().numpy()
    v[..., [1, 2]] *= -1
    return v

# ── RENDER FULL SEQUENCE ───────────────────────────────────────────────────────
print(f"\n[RENDER] Full sequence: {USER_NF} frames...")
t_render = time.time()

raw_mp4 = OUT_DIR / "retarget_inbetween_v1_raw.mp4"
fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
writer  = cv2.VideoWriter(str(raw_mp4), fourcc, user_fps, (W, H))

keyframe_imgs = {}    # 相位 → rendered image
KF_PHASES = {
    "address":     0,
    "takeaway":    takeaway_start,
    "top":         top_frame,
    "impact":      impact_frame,
    "follow":      follow_frame,
}

for fi in range(USER_NF):
    body_lerp, q_slerp = get_pose_at_user_frame(fi)
    verts = run_mhr_from_quat(body_lerp, q_slerp)
    out   = render_and_blend(verts, user_cam_t, user_focal, faces,
                              user_src[fi], H, W)
    writer.write(out)

    # 保存关键帧
    for phase_name, phase_fi in KF_PHASES.items():
        if fi == phase_fi:
            keyframe_imgs[phase_name] = out.copy()

    if fi % 20 == 0:
        print(f"  fr{fi:03d}/{USER_NF}  {time.time()-t_render:.1f}s")

writer.release()
print(f"  render done: {time.time()-t_render:.1f}s  raw={raw_mp4}")

# ── RE-ENCODE (正常速 + 0.25x 慢放) ──────────────────────────────────────────
def encode_video(src, dst, speed=1.0):
    if speed == 1.0:
        vf = "copy"
    else:
        # 0.25x 慢放: setpts=4*PTS
        factor = 1.0 / speed
        vf = f"setpts={factor:.4f}*PTS"
    cmd = (f"ffmpeg -y -i {src} -vf \"{vf}\" -c:v libx264 -crf 18 "
           f"-preset fast -pix_fmt yuv420p {dst} -loglevel error")
    ret = os.system(cmd)
    return ret

normal_path = OUT_DIR / "retarget_inbetween_v1.mp4"
slow_path   = OUT_DIR / "retarget_inbetween_v1_025x.mp4"
encode_video(raw_mp4, normal_path, 1.0)
encode_video(raw_mp4, slow_path,   0.25)
print(f"  normal: {normal_path}")
print(f"  025x:   {slow_path}")

shutil.copy2(str(normal_path), str(WIN_OUT / "retarget_inbetween_v1.mp4"))
shutil.copy2(str(slow_path),   str(WIN_OUT / "retarget_inbetween_v1_025x.mp4"))
print(f"  → Windows: {WIN_OUT}")

# ── KEYFRAME GRID ─────────────────────────────────────────────────────────────
print("\n[KEYFRAMES] Building 8-phase comparison grid...")

# 每个相位: 左=学员原图 右=ghost overlay (已渲染)
phase_order = ["address", "takeaway", "top", "impact", "follow"]
cells = []
for phase in phase_order:
    uf = KF_PHASES[phase]
    # 教练对应帧
    cf = next(a["coach_frame"] for a in anchor_data if a["label"] == phase)

    user_bg = user_src[uf].copy()
    ghost_img = keyframe_imgs.get(phase, user_bg)

    # 标注 user 帧
    user_lab = user_bg.copy()
    cv2.rectangle(user_lab, (0,0), (W,38), (0,0,0), -1)
    cv2.putText(user_lab, f"{phase} | user fr{uf}", (5,26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,240,200), 2)

    # 标注 ghost 帧
    ghost_lab = ghost_img.copy()
    cv2.rectangle(ghost_lab, (0,0), (W,38), (0,0,0), -1)
    cv2.putText(ghost_lab, f"{phase} | ghost (coach fr{cf}→user fr{uf})", (5,26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,80), 2)

    pair = np.concatenate([user_lab, ghost_lab], axis=1)
    cells.append(pair)

grid = np.vstack(cells)
kf_path = OUT_DIR / "inbetween_v1_keyframes.jpg"
cv2.imwrite(str(kf_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
shutil.copy2(str(kf_path), str(WIN_OUT / "inbetween_v1_keyframes.jpg"))
print(f"  saved: {WIN_OUT}/inbetween_v1_keyframes.jpg")

# ── REPORT ────────────────────────────────────────────────────────────────────
report_lines = [
    "REPORT - GHOST-004 Step 2 Inbetweening v1",
    "=" * 60,
    f"user NF={USER_NF}  fps={user_fps:.2f}",
    "",
    "USER PHASE ANCHORS (auto-detected from wrist trajectory):",
    f"  address_end  : fr{addr_end:03d}  (wrist_y={wrist_y_sm[addr_end]:.1f})",
    f"  takeaway     : fr{takeaway_start:03d}  (wrist_y={wrist_y_sm[takeaway_start]:.1f})",
    f"  top          : fr{top_frame:03d}  (wrist_y={wrist_y_sm[top_frame]:.1f}  wrist最高点)",
    f"  impact       : fr{impact_frame:03d}  (wrist_y={wrist_y_sm[impact_frame]:.1f}  击球位置)",
    f"  follow_peak  : fr{follow_frame:03d}  (wrist_y={wrist_y_sm[follow_frame]:.1f}  收杆最高点)",
    f"  end          : fr{end_frame:03d}",
    "",
    "PHASE DURATIONS (user speed):",
]
for i in range(len(ANCHORS)-1):
    a = ANCHORS[i]; b = ANCHORS[i+1]
    dur = (b[0] - a[0]) / user_fps * 1000
    report_lines.append(f"  {a[2]:12s}->{b[2]:12s}: {b[0]-a[0]:3d} frames  {dur:.0f}ms")

report_lines += [
    "",
    "COACH ANCHOR MAPPING:",
]
for a in anchor_data:
    report_lines.append(f"  user fr{a['user_frame']:03d} <- coach fr{a['coach_frame']:03d} ({a['label']})")

report_lines += [
    "",
    "INTERPOLATION METHOD:",
    "  body_pose_params (133,): LERP (linear on Euler angles)",
    "  global_rot: quaternion SLERP (no gimbal lock)",
    "  beta: user beta fixed throughout (shape/scale/expr/cam_t/focal)",
    "",
    "VERIFICATION QUESTIONS:",
    "  1. Ghost swing or dance? Is ghost motion now a coherent swing?",
    "  2. User speed? Does each phase duration = user's timing?",
    "  3. Key pose hit? Does ghost top = coach top pose on user body?",
    "  4. Downswing arc: realistic arc or shortcut straight line? Need extra anchor?",
    "",
    "OUTPUTS:",
    f"  retarget_inbetween_v1.mp4      ({USER_NF} frames, normal speed)",
    f"  retarget_inbetween_v1_025x.mp4 (0.25x slow)",
    f"  inbetween_v1_keyframes.jpg     (5-phase pair: user | ghost)",
]

report_txt = "\n".join(report_lines)
print("\n" + report_txt)
rpt_path = OUT_DIR / "REPORT_INBETWEEN_v1.txt"
rpt_path.write_text(report_txt)
shutil.copy2(str(rpt_path), str(WIN_OUT / "REPORT_INBETWEEN_v1.txt"))

print("\n[DONE]")
