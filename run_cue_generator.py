#!/usr/bin/env python3
"""
run_cue_generator.py — CUE-003 关卡C 端到端脚本

从 clip_016 verdict payload 生成 Reverse Pivot Cue Plan JSON + 工程预览图。

产出:
  output/cue_plans/<clip_id>_cue_plan.json    Cue Plan JSON
  output/cue_plans/<clip_id>_preview.jpg      工程草图（供 Jason 审 Plan 几何）
  Windows Desktop 同步

验收标准:
  1. clip_016_left  Confirmed +29.1°  → alpha_angle Plan，校验8/8通过
  2. clip_016_right None     -6.8°   → neutral Plan
  3. fo-eet-1-neg-setup  SILENT      → retake Plan
  4. fo-eet-1-neg-truncated SILENT   → retake Plan
"""
import sys, json, math, statistics
from pathlib import Path
import numpy as np
import cv2

PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ))

from cue_generator import build_alpha_plan, validate, render_preview

OUT_DIR  = PROJ / "output/cue_plans"
DESK_DIR = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/cue_plans")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DESK_DIR.mkdir(parents=True, exist_ok=True)

# ── clip_016 video (split-screen) ─────────────────────────────────────────────
CLIP016_VIDEO = Path("/mnt/c/Users/jason/Zening/Swingcue/教学视频") / \
    "115. 好的旋转方向是：身体按照顺序自然旋转。#高尔夫 #golf #golfswing #高尔夫课程 #高尔夫挥杆 #golflesson #golflife.mp4"

# ── kp_cache loader ───────────────────────────────────────────────────────────
def load_kp(clip_id: str) -> dict | None:
    for sub in ["negatives", "batch3", "batch2", "batch1", ""]:
        p = (PROJ/"engine/kp_cache"/sub/f"{clip_id}.json") if sub \
            else (PROJ/"engine/kp_cache"/f"{clip_id}.json")
        if p.exists():
            return json.load(open(p))
    return None


def grab_frame_bgr(video_path: Path, frame_idx: int,
                   x_crop: tuple = None) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, bgr = cap.read(); cap.release()
    if not ok:
        return None
    if x_crop:
        bgr = bgr[:, x_crop[0]:x_crop[1]]
    return bgr


# ── clip_016: geometry from RTMPose top-frame measurement ─────────────────────
#
# 汇报 (CUE-004 修正二轮 ①②):
#   嵌入帧: 原 est_fr=int(NF*0.55)=81（估算），修正为 top 帧 fr=93
#   top 帧确定方式: RTMPose 80帧采样，wrist_y 最小帧 = 上杆顶点
#   top_frame_fr = 93 (NF=148, FPS=23.8)
#
# 锚点来源: RTMPose 实测 fr93 左半 540×1920 原始尺寸
#   hip_mid  = (220.2, 1013.0)  [left_hip+right_hip 中点]
#   sho_mid  = (399.4, 691.4)   [left_shoulder+right_shoulder 中点]
#   tilt_actual = 29.14° (与 gate1 GT 29.1° 一致 ✓)
#   body_bbox = (170, 607, 548, 1558) 在 540×1920 坐标下
#
# Canvas resize: run_cue004 将帧 resize 到 CANVAS_W×CANVAS_H
#   → 锚点须乘以 (CANVAS_W/orig_w, CANVAS_H/orig_h)
#   原始尺寸 540×1920，canvas 720×1280
#   scale_x = 720/540 = 1.333, scale_y = 1280/1920 = 0.667
#   hip_mid_canvas  ≈ (293.6, 675.3)
#   sho_mid_canvas  ≈ (532.5, 460.9)

# Raw RTMPose measurements at orig 540×1920
_CLIP016_TOP_FRAME   = 93
_CLIP016_ORIG_W      = 540
_CLIP016_ORIG_H      = 1920
_CLIP016_HIP_MID_RAW = (220.2, 1013.0)
_CLIP016_SHO_MID_RAW = (399.4, 691.4)
_CLIP016_BODY_BBOX   = (170, 607, 548, 1558)  # x1,y1,x2,y2 in orig coords
_CLIP016_TILT_DEG    = 29.1  # GT from flywheel baseline (actual=29.14°)


def _scale_pt(pt, orig_w, orig_h, canvas_w, canvas_h):
    sx = canvas_w / orig_w
    sy = canvas_h / orig_h
    return (pt[0] * sx, pt[1] * sy)


def make_clip016_geometry(side: str, tilt_gt: float, canvas_w=720, canvas_h=1280) -> dict:
    """Return frame_bgr, hip_mid, shoulder_mid for clip_016 left/right.

    v0.6 修正: 嵌入帧=top帧(fr93), 锚点=RTMPose实测(scale到canvas尺寸).
    right side: top 帧及锚点另外测量; 暂用比例估算 (right=None路由, 不影响几何诊断).
    """
    if not CLIP016_VIDEO.exists():
        print(f"  [warn] clip_016 video not found, using placeholder")
        frame_bgr = np.full((canvas_h, canvas_w // 2, 3), 40, dtype=np.uint8)
        return dict(frame_bgr=frame_bgr,
                    hip_mid=(canvas_w // 4, int(canvas_h * 0.6)),
                    shoulder_mid=(canvas_w // 4, int(canvas_h * 0.38)),
                    body_bbox=None)

    cap = cv2.VideoCapture(str(CLIP016_VIDEO))
    w_full = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    half = w_full // 2

    if side == "left":
        fr_idx = _CLIP016_TOP_FRAME
        x_crop = (0, half)
        frame_bgr = grab_frame_bgr(CLIP016_VIDEO, fr_idx, x_crop)
        if frame_bgr is None:
            frame_bgr = np.full((canvas_h, canvas_w, 3), 40, dtype=np.uint8)
        # Scale RTMPose measurements to canvas
        hip_mid = _scale_pt(_CLIP016_HIP_MID_RAW, _CLIP016_ORIG_W, _CLIP016_ORIG_H,
                            canvas_w, canvas_h)
        sho_mid = _scale_pt(_CLIP016_SHO_MID_RAW, _CLIP016_ORIG_W, _CLIP016_ORIG_H,
                            canvas_w, canvas_h)
        # Scale body bbox
        sx = canvas_w / _CLIP016_ORIG_W; sy = canvas_h / _CLIP016_ORIG_H
        bbox = (_CLIP016_BODY_BBOX[0]*sx, _CLIP016_BODY_BBOX[1]*sy,
                _CLIP016_BODY_BBOX[2]*sx, _CLIP016_BODY_BBOX[3]*sy)
    else:
        # right side — None路由，不需要精确锚点；仍取 top 区域帧
        cap2 = cv2.VideoCapture(str(CLIP016_VIDEO))
        nf2  = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        cap2.release()
        fr_idx = int(nf2 * 0.45)
        x_crop = (half, w_full)
        frame_bgr = grab_frame_bgr(CLIP016_VIDEO, fr_idx, x_crop)
        if frame_bgr is None:
            frame_bgr = np.full((canvas_h, canvas_w, 3), 40, dtype=np.uint8)
        fh, fw = frame_bgr.shape[:2]
        hip_mid = (fw // 2, int(fh * 0.60))
        sho_mid = (fw // 2, int(fh * 0.38))
        bbox = None

    return dict(frame_bgr=frame_bgr, hip_mid=hip_mid, shoulder_mid=sho_mid,
                body_bbox=bbox)


# ── negative clips: placeholder geometry ─────────────────────────────────────

def make_neg_geometry(clip_id: str) -> dict:
    kp = load_kp(clip_id)
    if kp is None:
        frame_bgr = np.full((720, 400, 3), 40, dtype=np.uint8)
    else:
        n = len(kp["frames"])
        fr = kp["frames"][n // 3]
        kps = (fr.get("persons", [{}]) or [{}])[0].get("keypoints", {})
        # Try to get a real frame from video — skip, just use placeholder for negatives
        frame_bgr = np.full((720, 400, 3), 40, dtype=np.uint8)

    hip_mid      = (200, 432)
    shoulder_mid = (200, 274)
    return dict(frame_bgr=frame_bgr, hip_mid=hip_mid, shoulder_mid=shoulder_mid)


# ── process one clip ──────────────────────────────────────────────────────────

def process_clip(clip_id: str, confidence: str, tilt_deg: float,
                 geom: dict) -> tuple[dict, Path, Path]:
    """Build Plan, validate, render preview. Return (plan_dict, json_path, preview_path)."""

    plan = build_alpha_plan(
        clip_id=clip_id,
        confidence=confidence,
        tilt_deg=tilt_deg,
        hip_mid=geom["hip_mid"],
        shoulder_mid=geom["shoulder_mid"],
        band_lower_deg=-18.8,
        band_upper_deg=+5.0,
    )

    # Attach body_bbox to plan dict for validator rule⑩b
    plan_dict = plan.to_dict()
    if geom.get("body_bbox"):
        plan_dict["_body_bbox_px"] = list(geom["body_bbox"])

    # Validate
    vr = validate(plan_dict)
    plan.validator_result = vr.to_dict()
    plan_dict["validator_result"] = vr.to_dict()

    # Save JSON
    json_path = OUT_DIR / f"{clip_id}_cue_plan.json"
    json_path.write_text(plan.to_json(), encoding="utf-8")

    # Render preview
    preview_path = OUT_DIR / f"{clip_id}_preview.jpg"
    render_preview(plan, geom["frame_bgr"], preview_path)

    return plan_dict, json_path, preview_path


def main():
    import shutil

    clips = [
        # (clip_id,                  confidence,   tilt_deg,  geometry)
        ("clip_016_left",            "Confirmed",  +29.1,    make_clip016_geometry("left",  +29.1)),
        ("clip_016_right",           "None",       -6.8,     make_clip016_geometry("right", -6.8)),
        ("fo-eet-1-neg-setup",       "SILENT",     0.0,      make_neg_geometry("fo-eet-1-neg-setup")),
        ("fo-eet-1-neg-truncated",   "SILENT",     0.0,      make_neg_geometry("fo-eet-1-neg-truncated")),
    ]

    print(f"\n{'='*70}")
    print("  CUE-003 关卡C — Cue Plan 生成器端到端")
    print(f"{'='*70}\n")

    results = []
    for clip_id, conf, tilt, geom in clips:
        plan_dict, jpath, ppath = process_clip(clip_id, conf, tilt, geom)
        vr = plan_dict["validator_result"]
        vstatus = "PASS" if vr["passed"] else f"FAIL {vr['violations']}"
        stype = plan_dict["sentence_type_id"]
        print(f"  {clip_id:<30s}  conf={conf:<12s}  type={stype:<14s}  validator={vstatus}")
        if not vr["passed"]:
            for v in vr["violations"]:
                print(f"    ✗ {v}")
        # copy to desktop
        for p in [jpath, ppath]:
            shutil.copy2(p, DESK_DIR / p.name)
        results.append((clip_id, conf, stype, vr["passed"]))

    print(f"\n{'='*70}")
    all_pass = all(r[3] for r in results)
    print(f"  生成器校验总结: {'全部通过 ✓' if all_pass else '存在违规 ✗'}")
    print(f"\n  输出目录: {OUT_DIR}")
    print(f"  Windows:  C:\\Users\\jason\\Desktop\\rtmpose_results\\preview\\cue_plans\\")
    print()

    # Print summary of the Confirmed plan for review
    print(f"\n--- clip_016_left Cue Plan 摘要 (供 Jason 审) ---")
    cjson = (OUT_DIR / "clip_016_left_cue_plan.json").read_text(encoding="utf-8")
    plan_d = json.loads(cjson)
    print(f"  sentence_type_id:  {plan_d['sentence_type_id']}")
    print(f"  contrast_structure:{plan_d['contrast_structure']}")
    print(f"  caption:           {plan_d['caption_badge']['text']}")
    print(f"  elements ({len(plan_d['elements'])}):")
    for el in plan_d["elements"]:
        anim = "ANIMATED" if el["animation_track"] else "STATIC"
        print(f"    [{el['layer']}] {el['primitive']:4s}  {el['semantic_role']:<22s}  {anim}")
    print(f"  validator: {'PASS' if plan_d['validator_result']['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
