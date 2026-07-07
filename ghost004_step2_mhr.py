"""
ghost004_step2_mhr.py  —  GHOST-004 Step 2/2
MHR inference + stability pipeline + overlay rendering + pose sequence save.

Runs in sam3d_venv:
  /home/jason/projects/sam3d_venv/bin/python3 ghost004_step2_mhr.py

Reads:
  engine/kp_cache/ghost004/coach-fo.json
  engine/kp_cache/ghost004/coach-dtl.json
  output/ghost004/phase_report_step1.json

Outputs (per video):
  output/ghost004/<key>_pose_sequence.npz   — θ sequence (body_pose_params, global_rots,
                                              cam_t, focal) per frame  ← reusable for retarget
  output/ghost004/<key>_overlay.mp4          — normal speed overlay
  output/ghost004/<key>_overlay_025x.mp4     — 0.25x slowmo
  output/ghost004/<key>_phase_keyframes.jpg  — 4-panel address/top/impact/finish stills
  output/ghost004/REPORT_GHOST004_probe.txt  — full quality report

Stability pipeline (same as GHOST-003 T2.6):
  - z-lock: cam_t[2] locked to address-frame value
  - All 3 cam_t axes + focal: SG smoothed (window=9, poly=3)
  - Sentinel: frame with MHR fail → verts interpolated from neighbors
  - Upper-body IoU logged per frame (diagnostic)

Delivery:
  Windows: C:\\Users\\jason\\Desktop\\rtmpose_results\\preview\\ghost004\\
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

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path("/home/jason/projects/swingcue-postest")
KP_DIR     = ROOT / "engine/kp_cache/ghost004"
OUT_DIR    = ROOT / "output/ghost004"
WIN_OUT    = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/ghost004")
STEP1_RPT  = OUT_DIR / "phase_report_step1.json"
CKPT       = Path(os.path.expanduser("~/.cache/sam3d/sam-3d-body-dinov3/model.ckpt"))
MHR_PT     = CKPT.parent / "assets/mhr_model.pt"

OUT_DIR.mkdir(parents=True, exist_ok=True)
WIN_OUT.mkdir(parents=True, exist_ok=True)

VIDEOS = {
    "fo":  Path("/mnt/c/Users/jason/Zening/Swingcue/教学视频/coach-video/coach-fo.mp4"),
    "dtl": Path("/mnt/c/Users/jason/Zening/Swingcue/教学视频/coach-video/coach-dtl.mp4"),
}

# rendering
GHOST_ALPHA = 0.55
GHOST_COLOR = (0.85, 0.1, 0.1)
SG_WINDOW   = 9
SG_POLY     = 3

# joint indices (MHR70)
I_NOSE=0; I_LSHO=5; I_RSHO=6; I_LHIP=11; I_RHIP=12
I_LANK=15; I_RANK=16

KP_ORDER = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]

PHASE_NAMES = ["address","takeaway","backswing","top",
               "transition","downswing","impact","follow_through"]

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
    px = (max(ax) - min(ax)) * pad
    py = (max(ay) - min(ay)) * pad
    return np.array([[
        max(0, min(ax) - px), max(0, min(ay) - py),
        min(W, max(ax) + px), min(H, max(ay) + py)
    ]], dtype=np.float32)


def proj2d(verts, cam_t, focal, H, W):
    d  = verts[:, 2] + cam_t[2]
    px = focal * (verts[:, 0] - cam_t[0]) / d + W / 2
    py = focal * (verts[:, 1] + cam_t[1]) / d + H / 2
    return np.stack([px, py], axis=1)


def compute_upper_iou(verts, cam_t, focal, kp2d, H, W):
    """Upper-body silhouette IoU (same metric as GHOST-003)."""
    if kp2d is None or len(kp2d) < 13:
        return 0.0
    nose_y = kp2d[I_NOSE][1]; hip_y = (kp2d[I_LHIP][1] + kp2d[I_RHIP][1]) / 2
    y_lo = max(0, int(nose_y - 30)); y_hi = min(H, int(hip_y + 60))
    if y_hi <= y_lo:
        return 0.0

    # human mask from kp2d (convex hull approx)
    pts = kp2d[(kp2d[:, 1] >= y_lo) & (kp2d[:, 1] <= y_hi)][:, :2].astype(np.int32)
    if len(pts) < 3:
        return 0.0
    human_mask = np.zeros((H, W), dtype=np.uint8)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(human_mask, hull, 1)

    # mesh mask from projected verts
    vp = proj2d(verts, cam_t, focal, H, W)
    mesh_mask = np.zeros((H, W), dtype=np.uint8)
    vmask = (vp[:, 1] >= y_lo) & (vp[:, 1] <= y_hi) & \
            (vp[:, 0] >= 0) & (vp[:, 0] < W) & \
            (vp[:, 1] >= 0) & (vp[:, 1] < H)
    vp_int = vp[vmask].astype(np.int32)
    for px_v, py_v in vp_int:
        mesh_mask[py_v, px_v] = 1
    # dilate mesh to fill silhouette
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mesh_mask = cv2.dilate(mesh_mask, ker)

    # restrict to upper band
    human_mask[:y_lo, :] = 0; human_mask[y_hi:, :] = 0
    mesh_mask[:y_lo, :]  = 0; mesh_mask[y_hi:, :]  = 0

    inter = float(np.sum(human_mask & mesh_mask))
    union = float(np.sum(human_mask | mesh_mask))
    return inter / union if union > 0 else 0.0


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


def draw_phase_label(img, phase_name, frame_idx):
    """Overlay phase label + frame number on top-left."""
    label = f"{phase_name}  fr{frame_idx:03d}"
    cv2.putText(img, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 80), 2, cv2.LINE_AA)
    return img


# ── load model once ───────────────────────────────────────────────────────────
print("[INIT] Loading SAM 3D Body model...")
t0 = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
model, model_cfg = load_sam_3d_body(str(CKPT), device=device, mhr_path=str(MHR_PT))
estimator = SAM3DBodyEstimator(
    sam_3d_body_model=model, model_cfg=model_cfg,
    human_detector=None, human_segmentor=None, fov_estimator=None
)
estimator.model.eval()
faces = estimator.faces
print(f"  model loaded in {time.time()-t0:.1f}s  device={device}")

# load step1 report for anchor frames
with open(STEP1_RPT) as f:
    step1 = json.load(f)

# ── per-video processing ──────────────────────────────────────────────────────
summary_rows = []

for key, video_path in VIDEOS.items():
    print(f"\n{'='*60}")
    print(f"GHOST-004  {key.upper()}:  {video_path.name}")
    print(f"{'='*60}")

    kp_cache = load_kp_cache(KP_DIR / f"coach-{key}.json")
    vid_rpt  = step1.get(key, {})
    anchors  = vid_rpt.get("anchors", {})

    # address frame for z-lock
    FR_ADDRESS = anchors.get("address", 0)
    anchor_top    = anchors.get("top",     -1)
    anchor_impact = anchors.get("impact",  -1)
    anchor_finish = anchors.get("finish",  -1)

    # read video frames
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    src = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        src.append(f)
    cap.release()
    NF = len(src)
    H  = src[0].shape[0]
    W  = src[0].shape[1]
    print(f"  NF={NF}  fps={fps:.2f}  {W}x{H}")

    # ── PASS 1: MHR inference ──────────────────────────────────────────────
    print(f"\n[PASS1] MHR inference {NF} frames...")
    raw = {}
    t_p1 = time.time()

    for fi in range(NF):
        kp = kp_cache.get(fi)
        if kp is None:
            raw[fi] = {"ok": False, "reason": "no_kp"}
            continue
        bbox = get_bbox(kp[:, :2], H, W)
        try:
            with torch.no_grad():
                outs = estimator.process_one_image(
                    src[fi], bboxes=bbox, use_mask=False, inference_type="body"
                )
        except Exception as e:
            raw[fi] = {"ok": False, "reason": f"mhr_exc:{e}"}
            continue

        if not outs:
            raw[fi] = {"ok": False, "reason": "no_detection"}
            continue

        o = outs[0]
        raw[fi] = {
            "ok":              True,
            "verts":           o["pred_vertices"].astype(np.float32),
            "cam_t":           o["pred_cam_t"].astype(np.float32),
            "focal":           float(o["focal_length"]),
            "kp2d_mhr":        o["pred_keypoints_2d"].astype(np.float32),
            "body_pose_params": o["body_pose_params"].astype(np.float32),
            "pred_global_rots": o["pred_global_rots"].astype(np.float32),
        }
        if fi % 10 == 0:
            ct = raw[fi]["cam_t"]
            print(f"  fr{fi:03d}  cam_t=[{ct[0]:.4f},{ct[1]:.4f},{ct[2]:.4f}]  "
                  f"focal={raw[fi]['focal']:.1f}")

    ok_frames  = [fi for fi in range(NF) if raw[fi].get("ok")]
    fail_frames = [fi for fi in range(NF) if not raw[fi].get("ok")]
    print(f"  pass1 done {time.time()-t_p1:.1f}s  ok={len(ok_frames)}  fail={len(fail_frames)}")
    if fail_frames:
        print(f"  fail frames: {fail_frames[:20]}")

    # ── PASS 2: z-lock (address frame) ────────────────────────────────────
    addr_z = raw[FR_ADDRESS]["cam_t"][2] if raw[FR_ADDRESS].get("ok") else 5.0
    z_raw  = np.array([raw[fi]["cam_t"][2] for fi in ok_frames])
    print(f"\n[PASS2] z-lock: fr{FR_ADDRESS} z={addr_z:.4f}  "
          f"raw spread={z_raw.max()-z_raw.min():.4f} ({100*(z_raw.max()-z_raw.min())/addr_z:.2f}%)")

    for fi in ok_frames:
        raw[fi]["cam_t_locked"]    = raw[fi]["cam_t"].copy()
        raw[fi]["cam_t_locked"][2] = addr_z

    # ── PASS 3: SG smooth (all 3 axes + focal) ────────────────────────────
    print(f"[PASS3] SG smooth (window={SG_WINDOW}, poly={SG_POLY})...")
    ctx_raw = np.array([raw[fi]["cam_t_locked"][0] for fi in ok_frames])
    cty_raw = np.array([raw[fi]["cam_t_locked"][1] for fi in ok_frames])
    ctz_raw = np.array([raw[fi]["cam_t_locked"][2] for fi in ok_frames])  # constant
    foc_raw = np.array([raw[fi]["focal"]             for fi in ok_frames])

    sw = min(SG_WINDOW, len(ok_frames) if len(ok_frames) % 2 == 1 else len(ok_frames)-1)
    ctx_sm = savgol_filter(ctx_raw, sw, SG_POLY, mode="mirror")
    cty_sm = savgol_filter(cty_raw, sw, SG_POLY, mode="mirror")
    foc_sm = savgol_filter(foc_raw, sw, SG_POLY, mode="mirror")

    for idx, fi in enumerate(ok_frames):
        raw[fi]["cam_t_final"] = np.array([ctx_sm[idx], cty_sm[idx], ctz_raw[idx]], dtype=np.float32)
        raw[fi]["focal_final"] = float(foc_sm[idx])

    # ── PASS 4: fail-frame verts interpolation ─────────────────────────────
    print(f"[PASS4] Interpolating {len(fail_frames)} fail frames...")
    for fi in sorted(fail_frames):
        prev_fi = next((f for f in range(fi-1, -1, -1) if raw.get(f, {}).get("ok")), None)
        next_fi = next((f for f in range(fi+1, NF)     if raw.get(f, {}).get("ok")), None)
        if prev_fi is not None and next_fi is not None:
            w = (fi - prev_fi) / (next_fi - prev_fi)
            raw[fi]["verts"]           = (1-w)*raw[prev_fi]["verts"] + w*raw[next_fi]["verts"]
            raw[fi]["kp2d_mhr"]        = (1-w)*raw[prev_fi]["kp2d_mhr"] + w*raw[next_fi]["kp2d_mhr"]
            raw[fi]["body_pose_params"]= (1-w)*raw[prev_fi]["body_pose_params"] + w*raw[next_fi]["body_pose_params"]
            raw[fi]["pred_global_rots"]= (1-w)*raw[prev_fi]["pred_global_rots"] + w*raw[next_fi]["pred_global_rots"]
            # find nearest smoothed cam_t
            pidx = ok_frames.index(prev_fi)
            raw[fi]["cam_t_final"] = np.array([ctx_sm[pidx], cty_sm[pidx], ctz_raw[pidx]], dtype=np.float32)
            raw[fi]["focal_final"] = float(foc_sm[pidx])
            raw[fi]["ok"] = True
            print(f"  fr{fi:03d}: interp fr{prev_fi}+fr{next_fi}  w={w:.2f}")
        elif prev_fi is not None:
            raw[fi].update({k: raw[prev_fi][k] for k in ["verts","kp2d_mhr","body_pose_params",
                            "pred_global_rots","cam_t_final","focal_final"]})
            raw[fi]["ok"] = True
        elif next_fi is not None:
            raw[fi].update({k: raw[next_fi][k] for k in ["verts","kp2d_mhr","body_pose_params",
                            "pred_global_rots","cam_t_final","focal_final"]})
            raw[fi]["ok"] = True

    # ── PASS 5: upper-body IoU per frame (diagnostic) ─────────────────────
    print(f"[PASS5] Computing upper-body IoU per frame (diagnostic)...")
    iou_arr = np.zeros(NF)
    kp_cache_raw = load_kp_cache(KP_DIR / f"coach-{key}.json")

    for fi in range(NF):
        if not raw[fi].get("ok"):
            continue
        kp = kp_cache_raw.get(fi)
        iou_arr[fi] = compute_upper_iou(
            raw[fi]["verts"], raw[fi]["cam_t_final"], raw[fi]["focal_final"],
            kp, H, W
        )

    valid_iou = iou_arr[iou_arr > 0]
    if len(valid_iou) == 0:
        print(f"  IoU  all-zero (check kp_cache format or video orientation)")
        print(f"  DEBUG sample fr0 kp: {kp_cache_raw.get(0, None)}")
        iou_summary = {"mean": 0.0, "min": 0.0, "p5": 0.0, "p10": 0.0}
    else:
        print(f"  IoU  mean={valid_iou.mean():.4f}  min={valid_iou.min():.4f}  "
              f"P5={np.percentile(valid_iou,5):.4f}  P10={np.percentile(valid_iou,10):.4f}")
        iou_summary = {
            "mean": float(valid_iou.mean()), "min": float(valid_iou.min()),
            "p5": float(np.percentile(valid_iou, 5)), "p10": float(np.percentile(valid_iou, 10))
        }
    low_iou_frames = [fi for fi in range(NF) if 0 < iou_arr[fi] < 0.75]
    print(f"  low IoU (<0.75): {len(low_iou_frames)} frames  {low_iou_frames[:10]}")

    np.save(OUT_DIR / f"iou_{key}.npy", iou_arr)

    # ── PASS 6: save pose sequence NPZ (θ for retarget) ───────────────────
    print(f"[PASS6] Saving pose sequence NPZ...")
    seq_body_pose   = np.stack([raw[fi].get("body_pose_params", np.zeros(133)) for fi in range(NF)])
    seq_global_rots = np.stack([raw[fi].get("pred_global_rots", np.zeros((70,3,3))) for fi in range(NF)])
    seq_cam_t       = np.stack([raw[fi].get("cam_t_final", np.zeros(3)) for fi in range(NF)])
    seq_focal       = np.array([raw[fi].get("focal_final", 1000.0) for fi in range(NF)])
    seq_ok          = np.array([raw[fi].get("ok", False) for fi in range(NF)])

    npz_path = OUT_DIR / f"{key}_pose_sequence.npz"
    np.savez(str(npz_path),
             body_pose_params=seq_body_pose,
             pred_global_rots=seq_global_rots,
             cam_t=seq_cam_t,
             focal=seq_focal,
             frame_ok=seq_ok,
             anchors_address=np.int32(FR_ADDRESS),
             anchors_top=np.int32(anchor_top),
             anchors_impact=np.int32(anchor_impact),
             anchors_finish=np.int32(anchor_finish),
             iou_per_frame=iou_arr,
             fps=np.float32(fps))
    print(f"  saved: {npz_path}")

    # ── PASS 7: render overlay video ──────────────────────────────────────
    print(f"\n[PASS7] Rendering overlay video ({NF} frames)...")
    raw_mp4 = OUT_DIR / f"coach_{key}_overlay_raw.mp4"
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(str(raw_mp4), fourcc, fps, (W, H))

    # Phase label per frame: build frame→phase map
    frame_phase = {}
    for fi in range(NF):
        if fi <= anchors.get("address", 0):
            frame_phase[fi] = "address"
        elif fi <= anchors.get("top", NF):
            # split takeaway/backswing/top roughly
            span = anchors.get("top", NF) - anchors.get("address", 0)
            rel  = fi - anchors.get("address", 0)
            frame_phase[fi] = "takeaway" if rel < span*0.3 else \
                              "backswing" if rel < span*0.8 else "top"
        elif fi <= anchors.get("impact", NF):
            span = anchors.get("impact", NF) - anchors.get("top", NF)
            rel  = fi - anchors.get("top", NF)
            frame_phase[fi] = "transition" if rel < span*0.3 else "downswing"
        elif fi <= fi + 3 and fi >= anchors.get("impact", -1) - 1:
            frame_phase[fi] = "impact"
        else:
            frame_phase[fi] = "follow_through"
    # refine impact window
    imp_fr = anchors.get("impact", -1)
    for fi in range(max(0, imp_fr-1), min(NF, imp_fr+3)):
        frame_phase[fi] = "impact"

    t_p7 = time.time()
    for fi in range(NF):
        bg = src[fi].copy()
        if raw[fi].get("ok"):
            bg = render_and_blend(
                raw[fi]["verts"], raw[fi]["cam_t_final"], raw[fi]["focal_final"],
                faces, bg, H, W
            )
        phase_label = frame_phase.get(fi, "")
        draw_phase_label(bg, phase_label, fi)
        writer.write(bg)
        if fi % 20 == 0:
            print(f"  rendered fr{fi:03d}/{NF}  IoU={iou_arr[fi]:.3f}")

    writer.release()
    print(f"  render done {time.time()-t_p7:.1f}s")

    # Re-encode: normal speed
    final_mp4   = OUT_DIR / f"coach_{key}_overlay.mp4"
    slowmo_mp4  = OUT_DIR / f"coach_{key}_overlay_025x.mp4"
    os.system(f'ffmpeg -y -i "{raw_mp4}" -c:v libx264 -crf 18 -pix_fmt yuv420p "{final_mp4}" -loglevel error')
    os.system(f'ffmpeg -y -i "{raw_mp4}" -vf "setpts=4*PTS" -c:v libx264 -crf 18 -pix_fmt yuv420p "{slowmo_mp4}" -loglevel error')
    raw_mp4.unlink(missing_ok=True)
    print(f"  final: {final_mp4}")
    print(f"  slowmo: {slowmo_mp4}")

    # ── PASS 8: 4-panel keyframe stills ───────────────────────────────────
    print(f"[PASS8] Rendering 4-panel keyframe stills...")
    keyframe_map = {
        "address":       FR_ADDRESS,
        "top":           anchor_top,
        "impact":        anchor_impact,
        "follow_through": anchor_finish,
    }
    panels = []
    for ph_name, fr_idx in keyframe_map.items():
        if fr_idx < 0 or fr_idx >= NF:
            canvas = np.zeros((H, W, 3), dtype=np.uint8)
            cv2.putText(canvas, f"{ph_name}: n/a", (10, H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,200,200), 2)
        else:
            canvas = src[fr_idx].copy()
            if raw[fr_idx].get("ok"):
                canvas = render_and_blend(
                    raw[fr_idx]["verts"], raw[fr_idx]["cam_t_final"],
                    raw[fr_idx]["focal_final"], faces, canvas, H, W
                )
            draw_phase_label(canvas, ph_name, fr_idx)
            # IoU badge
            iou_v = iou_arr[fr_idx]
            cv2.putText(canvas, f"IoU={iou_v:.3f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 255, 80), 2)
        panels.append(canvas)

    # 2x2 grid
    scale  = 0.5
    ph     = int(H * scale); pw = int(W * scale)
    panels_s = [cv2.resize(p, (pw, ph)) for p in panels]
    row1   = np.hstack([panels_s[0], panels_s[1]])
    row2   = np.hstack([panels_s[2], panels_s[3]])
    grid   = np.vstack([row1, row2])
    still_path = OUT_DIR / f"coach_{key}_phase_keyframes.jpg"
    cv2.imwrite(str(still_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  saved: {still_path}")

    # ── summary for this video ─────────────────────────────────────────────
    summary_rows.append({
        "key": key,
        "nf":  NF,
        "fps": fps,
        "ok_frames": len([fi for fi in range(NF) if raw[fi].get("ok")]),
        "fail_frames": len([fi for fi in range(NF) if not raw[fi].get("ok")]),
        "iou_mean": iou_summary["mean"],
        "iou_min":  iou_summary["min"],
        "iou_p5":   iou_summary["p5"],
        "low_iou_count": len(low_iou_frames),
        "z_spread_pct": float(100*(z_raw.max()-z_raw.min())/addr_z),
        "anchors": {
            "address": FR_ADDRESS, "top": anchor_top,
            "impact": anchor_impact, "finish": anchor_finish,
        },
    })

    # copy to Windows
    import shutil
    for f in [final_mp4, slowmo_mp4, still_path]:
        dst = WIN_OUT / f.name
        shutil.copy2(str(f), str(dst))
        print(f"  -> Windows: {dst.name}")

# ── PASS 9: generate REPORT_GHOST004_probe.txt ────────────────────────────────
print(f"\n[PASS9] Writing REPORT_GHOST004_probe.txt...")
report_txt = OUT_DIR / "REPORT_GHOST004_probe.txt"

with open(STEP1_RPT) as f:
    s1 = json.load(f)

lines = [
    "=" * 65,
    "REPORT_GHOST004_probe.txt",
    "GHOST-004 Step 1+2: Coach video pose extraction quality report",
    "=" * 65,
    "",
]

for row in summary_rows:
    key = row["key"]
    s1v = s1.get(key, {})
    a   = s1v.get("anchors", {})
    pm  = s1v.get("phase_map", {})

    lines += [
        f"{'─'*65}",
        f"VIDEO: coach-{key}.mp4  ({row['nf']} frames @ {row['fps']:.2f}fps)",
        f"{'─'*65}",
        "",
        "RTMPose extraction quality:",
        f"  mean conf : {s1v.get('conf_mean', 0):.3f}",
        f"  min conf  : {s1v.get('conf_min', 0):.3f}",
        f"  P5 conf   : {s1v.get('conf_p5', 0):.3f}",
        f"  bad frames: {len(s1v.get('bad_frames', []))}",
    ]
    for b in s1v.get("bad_frames", [])[:5]:
        lines.append(f"    fr{b['frame']:03d}  {b['reason']}")

    lines += [
        "",
        "MHR inference quality:",
        f"  ok frames : {row['ok_frames']} / {row['nf']}",
        f"  fail frames: {row['fail_frames']}",
        f"  upper IoU mean={row['iou_mean']:.4f}  min={row['iou_min']:.4f}  P5={row['iou_p5']:.4f}",
        f"  low IoU (<0.75) frames: {row['low_iou_count']}",
        f"  z spread before lock: {row['z_spread_pct']:.2f}%",
        "",
        "8-phase detection:",
        f"  address  : fr{a.get('address',-1):03d}",
        f"  top      : fr{a.get('top',-1):03d}  (conf={a.get('top_conf',0):.3f})",
        f"  impact   : fr{a.get('impact',-1):03d}  (conf={a.get('impact_conf',0):.3f})",
        f"  finish   : fr{a.get('finish',-1):03d}",
        f"  swing_cnt: {a.get('swing_count', -1)}",
        "",
        "Phase timeline:",
    ]
    for ph in ["address","takeaway","backswing","top","transition","downswing","impact","follow_through"]:
        if ph in pm:
            v = pm[ph]
            lines.append(f"  {ph:20s} fr{v['start']:03d}–fr{v['end']:03d}  ({v['count']} frames)")
        else:
            lines.append(f"  {ph:20s} —")
    lines.append("")

# cross-alignment
if len(summary_rows) == 2:
    fo  = next(r for r in summary_rows if r["key"]=="fo")
    dtl = next(r for r in summary_rows if r["key"]=="dtl")
    lines += [
        "─"*65,
        "CROSS-VIDEO PHASE ALIGNMENT (normalized 0=start, 1=end):",
        f"  {'phase':15s}  {'FO':>8s}  {'DTL':>8s}  {'delta':>8s}  status",
    ]
    for ph_key, ph_label in [("address","address"),("top","top"),("impact","impact"),("finish","finish")]:
        fo_n  = fo["anchors"][ph_key]  / fo["nf"]
        dtl_n = dtl["anchors"][ph_key] / dtl["nf"]
        delta = abs(fo_n - dtl_n)
        flag  = "⚠ MISALIGN" if delta > 0.10 else "✓ aligned"
        lines.append(f"  {ph_label:15s}  {fo_n:8.3f}  {dtl_n:8.3f}  {delta:8.3f}  {flag}")
    lines.append("")

lines += [
    "─"*65,
    "Outputs:",
    "  fo_pose_sequence.npz    — body_pose_params + global_rots + cam_t + focal",
    "  dtl_pose_sequence.npz   — same for DTL",
    "  coach_fo_overlay.mp4    — normal speed",
    "  coach_fo_overlay_025x.mp4",
    "  coach_dtl_overlay.mp4",
    "  coach_dtl_overlay_025x.mp4",
    "  coach_fo_phase_keyframes.jpg",
    "  coach_dtl_phase_keyframes.jpg",
    "",
    "Jason verifies:",
    "  1. Are these swings pro-standard enough as reference motions?",
    "  2. Does the red mesh track cleanly (no collapse/flicker)?",
    "  3. Are phase anchors (top/impact) in the right frames?",
    "  4. Do FO and DTL phase timelines align (same swing)?",
    "─"*65,
]

with open(report_txt, "w") as f:
    f.write("\n".join(lines))
import shutil
shutil.copy2(str(report_txt), str(WIN_OUT / report_txt.name))
print(f"  saved: {report_txt}")

print("\nGHOST-004 Step 2 complete.")
print(f"Windows output: C:\\Users\\jason\\Desktop\\rtmpose_results\\preview\\ghost004\\")
