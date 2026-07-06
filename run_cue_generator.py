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


# ── clip_016: geometry from gate1 GT values ───────────────────────────────────

def make_clip016_geometry(side: str, tilt_gt: float) -> dict:
    """Return frame_bgr, hip_mid, shoulder_mid for clip_016 left/right."""
    if not CLIP016_VIDEO.exists():
        print(f"  [warn] clip_016 video not found, using placeholder")
        frame_bgr = np.full((720, 400, 3), 40, dtype=np.uint8)
        hip_mid      = (200, 432)
        shoulder_mid = (200, 274)
        return dict(frame_bgr=frame_bgr, hip_mid=hip_mid, shoulder_mid=shoulder_mid)

    cap = cv2.VideoCapture(str(CLIP016_VIDEO))
    w_full = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    nf     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    half   = w_full // 2
    x_crop = (0, half) if side == "left" else (half, w_full)
    est_fr = int(nf * 0.55)
    frame_bgr = grab_frame_bgr(CLIP016_VIDEO, est_fr, x_crop)
    if frame_bgr is None:
        frame_bgr = np.full((720, half, 3), 40, dtype=np.uint8)

    fh, fw = frame_bgr.shape[:2]
    hip_mid      = (fw // 2, int(fh * 0.60))
    dist         = abs(int(fh * 0.38) - hip_mid[1])
    tilt_rad     = math.radians(tilt_gt)
    sho_x        = int(hip_mid[0] + dist * math.sin(tilt_rad))
    sho_y        = int(hip_mid[1] - dist * math.cos(tilt_rad))
    shoulder_mid = (sho_x, sho_y)
    return dict(frame_bgr=frame_bgr, hip_mid=hip_mid, shoulder_mid=shoulder_mid)


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

    # Validate
    vr = validate(plan.to_dict())
    plan.validator_result = vr.to_dict()

    # Save JSON
    json_path = OUT_DIR / f"{clip_id}_cue_plan.json"
    json_path.write_text(plan.to_json(), encoding="utf-8")

    # Render preview
    preview_path = OUT_DIR / f"{clip_id}_preview.jpg"
    render_preview(plan, geom["frame_bgr"], preview_path)

    return plan.to_dict(), json_path, preview_path


def main():
    import shutil

    clips = [
        # (clip_id,                  confidence,   tilt_deg,  geometry_fn)
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
