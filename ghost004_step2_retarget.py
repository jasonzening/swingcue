"""
ghost004_step2_retarget.py  —  GHOST-004 Step 2: Motion Retarget
教练 θ 序列 → 用户 β 体型 → 渲染

原理:
  - θ (motion) = coach 的 body_pose_params (133) + global_rot (3) + hand_pose_params (108)
                 来自 fo_pose_sequence_aligned.npz
  - β (shape)  = user 的 shape_params (45) + scale_params (28)
                 从 fo-ok-1 address 帧现场提取（process_one_image 单帧）
  - retarget:  model.head_pose.mhr_forward(
                   global_trans = user_address_global_trans (固定),
                   global_rot   = coach_global_rot[fi],
                   body_pose_params = coach_body_pose[fi],
                   hand_pose_params = coach_hand_pose[fi],
                   scale_params = user_scale,
                   shape_params = user_shape,
               ) → verts

  - root 位置策略: 锁定 cam_t 为 user address 帧值 + 全轴 SG 平滑 coach cam_t δ
                   (让幽灵落在用户画面合理位置)
  - 渲染: 直接复用 T2.6 稳定管线 (红色半透明, z-lock=user_address_z, SG平滑)

输入:
  output/ghost004/fo_pose_sequence_aligned.npz
  input/fo-ok-1.mp4                              (用户视频，背景)
  engine/kp_cache/batch2/fo-ok-1.json            (用户 RTMPose，用于 bbox 提取 β)

输出:
  output/ghost004/retarget_fo_playback.mp4        正常速
  output/ghost004/retarget_fo_playback_025x.mp4   0.25x 慢放
  output/ghost004/retarget_compare_keyframes.jpg  教练 vs 用户 β 对比4帧
  output/ghost004/REPORT_RETARGET.txt
  Windows: C:\\Users\\jason\\Desktop\\rtmpose_results\\preview\\ghost004\\

运行环境: /home/jason/projects/sam3d_venv
  /home/jason/projects/sam3d_venv/bin/python3 ghost004_step2_retarget.py
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
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from sam_3d_body.visualization.renderer import Renderer

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path("/home/jason/projects/swingcue-postest")
USER_VIDEO   = ROOT / "input/fo-ok-1.mp4"
USER_KP      = ROOT / "engine/kp_cache/batch2/fo-ok-1.json"
COACH_NPZ    = ROOT / "output/ghost004/fo_pose_sequence_aligned.npz"
OUT_DIR      = ROOT / "output/ghost004"
WIN_OUT      = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/ghost004")
CKPT         = Path(os.path.expanduser("~/.cache/sam3d/sam-3d-body-dinov3/model.ckpt"))
MHR_PT       = CKPT.parent / "assets/mhr_model.pt"

OUT_DIR.mkdir(parents=True, exist_ok=True)
WIN_OUT.mkdir(parents=True, exist_ok=True)

# rendering params (T2.6 baseline)
GHOST_ALPHA = 0.55
GHOST_COLOR = (0.85, 0.1, 0.1)
SG_WINDOW   = 9
SG_POLY     = 3
USER_FR_ADDRESS = 0  # user address frame

KP_ORDER = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]

# ── helpers ───────────────────────────────────────────────────────────────────
def load_kp_cache(path):
    with open(path) as f:
        d = json.load(f)
    out = {}
    for fe in d["frames"]:
        fi = fe["frame"]
        if not fe.get("persons"):
            continue
        kd = fe["persons"][0]["keypoints"]
        out[fi] = np.array(
            [[kd[n]["x"], kd[n]["y"], kd[n]["score"]] for n in KP_ORDER if n in kd],
            dtype=np.float32
        )
    return out


def get_bbox(kp2d, H, W, pad=0.15):
    ax = kp2d[:, 0]; ay = kp2d[:, 1]
    px = (max(ax) - min(ax)) * pad; py = (max(ay) - min(ay)) * pad
    return np.array([[
        max(0, min(ax) - px), max(0, min(ay) - py),
        min(W, max(ax) + px), min(H, max(ay) + py)
    ]], dtype=np.float32)


def proj2d(verts, cam_t, focal, H, W):
    d  = verts[:, 2] + cam_t[2]
    px = focal * (verts[:, 0] - cam_t[0]) / d + W / 2
    py = focal * (verts[:, 1] + cam_t[1]) / d + H / 2
    return np.stack([px, py], axis=1)


def render_and_blend(verts, cam_t, focal, faces, bg_frame, H, W):
    black_bg = np.zeros((H, W, 3), dtype=np.uint8)
    rend = Renderer(focal_length=focal, faces=faces)
    out  = rend(verts, cam_t, black_bg,
                mesh_base_color=GHOST_COLOR, scene_bg_color=(0, 0, 0))
    out_u8 = (out * 255).clip(0, 255).astype(np.uint8)
    mask   = np.any(out_u8 > 5, axis=2)
    result = bg_frame.copy().astype(np.float32)
    result[mask] = (1 - GHOST_ALPHA) * result[mask] + GHOST_ALPHA * out_u8[mask].astype(np.float32)
    return result.astype(np.uint8)


def draw_label(img, text, y=35, color=(255, 255, 80)):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, color, 2, cv2.LINE_AA)
    return img


# ── INIT: load model ──────────────────────────────────────────────────────────
print("[INIT] Loading SAM 3D Body model...")
t0 = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
model, model_cfg = load_sam_3d_body(str(CKPT), device=device, mhr_path=str(MHR_PT))
est = SAM3DBodyEstimator(
    sam_3d_body_model=model, model_cfg=model_cfg,
    human_detector=None, human_segmentor=None, fov_estimator=None
)
est.model.eval()
mhr_head = est.model.head_pose   # MHRHead — used for FK with swapped β
faces    = est.faces
print(f"  loaded in {time.time()-t0:.1f}s  device={device}")

# ── LOAD USER VIDEO + KP ──────────────────────────────────────────────────────
print("\n[LOAD] User video + keypoints...")
cap = cv2.VideoCapture(str(USER_VIDEO))
user_fps = cap.get(cv2.CAP_PROP_FPS)
user_src = []
while True:
    ret, f = cap.read()
    if not ret: break
    user_src.append(f)
cap.release()
USER_NF = len(user_src)
H = user_src[0].shape[0]; W = user_src[0].shape[1]
print(f"  user video: NF={USER_NF}  fps={user_fps:.2f}  {W}x{H}")

user_kp = load_kp_cache(str(USER_KP))

# ── EXTRACT USER β (address frame, single inference) ─────────────────────────
print("\n[BETA] Extracting user shape (β) from address frame fr0...")
kp0   = user_kp[USER_FR_ADDRESS]
bbox0 = get_bbox(kp0[:, :2], H, W)
with torch.no_grad():
    outs0 = est.process_one_image(
        user_src[USER_FR_ADDRESS], bboxes=bbox0, use_mask=False, inference_type="body"
    )
assert outs0, "MHR failed on user address frame!"
u0 = outs0[0]

user_shape       = u0["shape_params"]    # (45,) np.float32
user_scale       = u0["scale_params"]    # (28,) np.float32
user_cam_t       = u0["pred_cam_t"]      # (3,)
user_focal       = float(u0["focal_length"])
user_global_rot  = u0["global_rot"]      # (3,) Euler XYZ — user's address global orientation
user_verts_addr  = u0["pred_vertices"]   # (18439,3) — user address mesh (reference for comparison)

# global_trans: from mhr_head forward, it's zeros in inference (trans from cam_t)
user_global_trans = np.zeros(3, dtype=np.float32)
user_addr_z       = float(user_cam_t[2])

print(f"  user β: shape={user_shape.shape}  scale={user_scale.shape}")
print(f"  user cam_t={user_cam_t}  focal={user_focal:.1f}  z={user_addr_z:.4f}")

# ── LOAD COACH θ SEQUENCE ─────────────────────────────────────────────────────
print("\n[COACH] Loading coach θ sequence...")
coach = np.load(str(COACH_NPZ), allow_pickle=False)
COACH_NF      = coach["body_pose_params"].shape[0]
COACH_IMPACT  = int(coach["anchors_impact"])
COACH_TOP     = int(coach["anchors_top"])
COACH_ADDRESS = int(coach["anchors_address"])
COACH_FINISH  = int(coach["anchors_finish"])

print(f"  coach NF={COACH_NF}  address=fr{COACH_ADDRESS}  top=fr{COACH_TOP}  "
      f"impact=fr{COACH_IMPACT}  finish=fr{COACH_FINISH}")

# ── DETERMINE OUTPUT LENGTH ───────────────────────────────────────────────────
# Render on user video background: loop user video if shorter than coach sequence
# Map: coach fr0 → user fr0 (both at address), play in sync
OUT_NF = COACH_NF  # coach sets the pace

# Build user frame index (loop if needed)
user_frame_idx = [min(fi, USER_NF - 1) for fi in range(OUT_NF)]

# ── RETARGET: FK with user β + coach θ ───────────────────────────────────────
print(f"\n[RETARGET] Running FK retarget ({OUT_NF} frames)...")
t_rt = time.time()

# Convert user beta to torch tensors
dev = torch.device(device)
u_shape_t = torch.from_numpy(user_shape).unsqueeze(0).to(dev)    # (1,45)
u_scale_t = torch.from_numpy(user_scale).unsqueeze(0).to(dev)    # (1,28)
# user_global_trans stays zero (translation controlled by cam_t)
u_gtrans_t = torch.zeros(1, 3, dtype=torch.float32, device=dev)

retarget_verts = []    # list of np arrays (18439,3) or None
retarget_cam_t = []    # list of np arrays (3,)
retarget_focal = []    # list of floats
retarget_ok    = []

for fi in range(OUT_NF):
    ok_flag = bool(coach["frame_ok"][fi])
    if not ok_flag:
        retarget_verts.append(None)
        retarget_cam_t.append(user_cam_t.copy())
        retarget_focal.append(user_focal)
        retarget_ok.append(False)
        continue

    # Extract coach θ for this frame
    c_body_pose  = coach["body_pose_params"][fi]   # (133,)
    c_global_rot = coach["pred_global_rots"][fi, 0:1]  # first joint = global; shape (1,3,3)

    # Convert global rot matrix to euler (ZYX) for mhr_forward
    import roma
    gr_tensor = torch.from_numpy(c_global_rot).to(dev)   # (1,3,3)
    c_global_rot_euler = roma.rotmat_to_euler("ZYX", gr_tensor).squeeze(0)  # (3,)

    c_body_t  = torch.from_numpy(c_body_pose[:130]).unsqueeze(0).to(dev)  # (1,130)
    c_grot_t  = c_global_rot_euler.unsqueeze(0).to(dev)                    # (1,3)

    # Hand pose: use user address hand pose (neutral) for first version
    # (coach hand pose can be added later)
    u_hand_t  = torch.zeros(1, 108, dtype=torch.float32, device=dev)
    u_expr_t  = torch.zeros(1, 72,  dtype=torch.float32, device=dev)

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

    # result is just verts (18439,3) when return_keypoints=False
    if isinstance(result, tuple):
        verts = result[0]
    else:
        verts = result

    verts_np = verts.squeeze(0).cpu().numpy()      # (18439,3)
    # Apply SAM3D camera convention (y,z flip same as in mhr_head.forward)
    verts_np[..., [1, 2]] *= -1

    retarget_verts.append(verts_np)
    # cam_t: use user address cam_t (z-locked); x/y from coach cam_t delta
    # Strategy: user_cam_t is absolute position; coach has own cam_t trajectory
    # For now: pin to user address cam_t (simplest, wont drift out of frame)
    retarget_cam_t.append(user_cam_t.copy())
    retarget_focal.append(user_focal)
    retarget_ok.append(True)

    if fi % 10 == 0:
        print(f"  fr{fi:03d}/{OUT_NF}  verts_range=[{verts_np[:,0].min():.3f},{verts_np[:,0].max():.3f}]")

print(f"  retarget done {time.time()-t_rt:.1f}s  ok={sum(retarget_ok)}/{OUT_NF}")

# ── INTERPOLATE FAILED FRAMES ──────────────────────────────────────────────────
print("[INTERP] Interpolating failed frames...")
for fi in range(OUT_NF):
    if retarget_ok[fi]:
        continue
    prev = next((f for f in range(fi-1,-1,-1) if retarget_ok[f]), None)
    nxt  = next((f for f in range(fi+1,OUT_NF) if retarget_ok[f]), None)
    if prev is not None and nxt is not None:
        w = (fi-prev)/(nxt-prev)
        retarget_verts[fi] = (1-w)*retarget_verts[prev] + w*retarget_verts[nxt]
        retarget_ok[fi] = True
    elif prev is not None:
        retarget_verts[fi] = retarget_verts[prev].copy(); retarget_ok[fi] = True
    elif nxt is not None:
        retarget_verts[fi] = retarget_verts[nxt].copy(); retarget_ok[fi] = True

# ── SG SMOOTH cam_t x/y + focal (z already locked) ────────────────────────────
print("[SG] Smoothing cam_t x/y + focal...")
ok_idx = [fi for fi in range(OUT_NF) if retarget_ok[fi]]
# cam_t is constant user_cam_t here, so SG has no effect — keeping for pipeline consistency
# (future: if we vary cam_t by coach trajectory, SG becomes meaningful)
ctx_arr = np.array([retarget_cam_t[fi][0] for fi in ok_idx])
cty_arr = np.array([retarget_cam_t[fi][1] for fi in ok_idx])
foc_arr = np.array([retarget_focal[fi] for fi in ok_idx])
sw = min(SG_WINDOW, len(ok_idx) if len(ok_idx)%2==1 else len(ok_idx)-1)
ctx_sm = savgol_filter(ctx_arr, sw, SG_POLY, mode="mirror")
cty_sm = savgol_filter(cty_arr, sw, SG_POLY, mode="mirror")
foc_sm = savgol_filter(foc_arr, sw, SG_POLY, mode="mirror")
for idx, fi in enumerate(ok_idx):
    retarget_cam_t[fi] = np.array([ctx_sm[idx], cty_sm[idx], user_addr_z], dtype=np.float32)
    retarget_focal[fi] = float(foc_sm[idx])

# ── RENDER OVERLAY VIDEO ──────────────────────────────────────────────────────
print(f"\n[RENDER] Rendering {OUT_NF} frames on user background...")

# Phase label map (based on coach anchors)
def frame_phase(fi):
    if fi <= COACH_ADDRESS: return "address"
    elif fi <= COACH_TOP:
        span = COACH_TOP - COACH_ADDRESS
        rel  = fi - COACH_ADDRESS
        return "takeaway" if rel < span*0.3 else "backswing" if rel < span*0.8 else "top"
    elif fi >= COACH_IMPACT - 1 and fi <= COACH_IMPACT + 2:
        return "impact"
    elif fi <= COACH_IMPACT:
        span = COACH_IMPACT - COACH_TOP
        rel  = fi - COACH_TOP
        return "transition" if rel < span*0.3 else "downswing"
    else:
        return "follow_through"

raw_mp4 = OUT_DIR / "retarget_fo_raw.mp4"
fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
writer  = cv2.VideoWriter(str(raw_mp4), fourcc, user_fps, (W, H))

t_ren = time.time()
for fi in range(OUT_NF):
    uf = user_frame_idx[fi]
    bg = user_src[uf].copy()

    if retarget_ok[fi]:
        bg = render_and_blend(
            retarget_verts[fi], retarget_cam_t[fi], retarget_focal[fi],
            faces, bg, H, W
        )
    ph = frame_phase(fi)
    draw_label(bg, f"coach-motion  user-body  {ph}  fr{fi:03d}/{OUT_NF-1}")
    writer.write(bg)
    if fi % 20 == 0:
        print(f"  rendered fr{fi:03d}/{OUT_NF}")

writer.release()
print(f"  render done {time.time()-t_ren:.1f}s")

# Re-encode
final_mp4  = OUT_DIR / "retarget_fo_playback.mp4"
slowmo_mp4 = OUT_DIR / "retarget_fo_playback_025x.mp4"
os.system(f'ffmpeg -y -i "{raw_mp4}" -c:v libx264 -crf 18 -pix_fmt yuv420p "{final_mp4}" -loglevel error')
os.system(f'ffmpeg -y -i "{raw_mp4}" -vf "setpts=4*PTS" -c:v libx264 -crf 18 -pix_fmt yuv420p "{slowmo_mp4}" -loglevel error')
raw_mp4.unlink(missing_ok=True)
print(f"  final: {final_mp4}")
print(f"  slowmo: {slowmo_mp4}")

# ── COMPARISON KEYFRAME STILLS: coach verts vs user β retarget ───────────────
print("\n[COMPARE] Rendering comparison keyframe stills...")

KEY_FRAMES = {
    "address": COACH_ADDRESS,
    "top":     COACH_TOP,
    "impact":  COACH_IMPACT,
    "follow":  min(COACH_IMPACT + 10, OUT_NF - 1),
}

panels = []
for ph_name, fr_idx in KEY_FRAMES.items():
    # Left: user bg + retarget mesh (user body, coach motion)
    uf = user_frame_idx[fr_idx]
    left = user_src[uf].copy()
    if retarget_ok[fr_idx] and retarget_verts[fr_idx] is not None:
        left = render_and_blend(
            retarget_verts[fr_idx], retarget_cam_t[fr_idx],
            retarget_focal[fr_idx], faces, left, H, W
        )
    draw_label(left, f"user-body+coach-motion  {ph_name}  fr{fr_idx:03d}")

    # Right: user bg + user address mesh (reference: what user actually looks like at rest)
    right = user_src[USER_FR_ADDRESS].copy()
    right = render_and_blend(
        user_verts_addr, user_cam_t, user_focal, faces, right, H, W
    )
    draw_label(right, f"user-body address (ref)")

    # Side by side
    scale = 0.45
    lh, lw = int(H*scale), int(W*scale)
    lp = cv2.resize(left,  (lw, lh))
    rp = cv2.resize(right, (lw, lh))
    row = np.hstack([lp, rp])
    # Add phase label
    cv2.putText(row, ph_name.upper(), (10, lh - 10), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (255, 255, 255), 3, cv2.LINE_AA)
    panels.append(row)

grid = np.vstack(panels)
compare_path = OUT_DIR / "retarget_compare_keyframes.jpg"
cv2.imwrite(str(compare_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"  saved: {compare_path}")

# ── REPORT ────────────────────────────────────────────────────────────────────
print("\n[REPORT] Writing REPORT_RETARGET.txt...")
report_lines = [
    "=" * 65,
    "REPORT_RETARGET.txt — GHOST-004 Step 2",
    "Motion Retarget: coach-fo theta + user fo-ok-1 beta",
    "Generated: 2026-07-07",
    "=" * 65,
    "",
    "Retarget design:",
    "  theta source : coach-fo.mp4  (NF={}  impact=fr{:03d})".format(COACH_NF, COACH_IMPACT),
    "  beta source  : fo-ok-1.mp4  address fr0  (single MHR inference)",
    "  shape_params : (45,) from user address frame",
    "  scale_params : (28,) from user address frame",
    "  global_rot   : from coach each frame (coach body orientation)",
    "  cam_t        : user address cam_t locked (z={:.4f})".format(user_addr_z),
    "  focal        : user address focal={:.1f}  (SG smoothed)".format(user_focal),
    "  hand_pose    : neutral (zeros) v1 — add coach hand pose in v2",
    "",
    "Retarget stats:",
    "  OUT_NF     : {}".format(OUT_NF),
    "  ok_frames  : {}/{}".format(sum(retarget_ok), OUT_NF),
    "",
    "Output:",
    "  retarget_fo_playback.mp4        normal speed",
    "  retarget_fo_playback_025x.mp4   0.25x slowmo",
    "  retarget_compare_keyframes.jpg  coach-motion+user-body vs user-body-at-address",
    "",
    "Jason verifies (Step 2 acceptance):",
    "  Q1. Ghost is user body shape + coach motion? (not coach body size)",
    "  Q2. Motion is smooth, no collapse/penetration in slowmo?",
    "  Q3. Address frame: ghost starts at same position as user?",
    "  Q4. Face-on chain proven -> proceed to DTL retarget?",
    "",
    "Pass -> DTL retarget + final two-camera ghost",
    "Adjust -> specify which frames / body part",
    "-" * 65,
]
report_txt = "\n".join(report_lines)
with open(OUT_DIR / "REPORT_RETARGET.txt", "w", encoding="utf-8") as f:
    f.write(report_txt)

# ── COPY TO WINDOWS ───────────────────────────────────────────────────────────
import shutil
for f in [final_mp4, slowmo_mp4, compare_path, OUT_DIR / "REPORT_RETARGET.txt"]:
    dst = WIN_OUT / Path(f).name
    shutil.copy2(str(f), str(dst))
    print(f"  -> Windows: {dst.name}")

print("\nGHOST-004 Step 2 retarget complete.")
print("Windows: C:\\Users\\jason\\Desktop\\rtmpose_results\\preview\\ghost004\\")
