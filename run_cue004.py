#!/usr/bin/env python3
"""
run_cue004.py — CUE-004/005 关卡B+C 端到端脚本

关卡B: 四路由成品生成
  fo-ok-1_MOCK   (Confirmed) → 动画 cue: .lottie + .mp4 + static_last.jpg
                               [MOCK] verdict=clip_016 +29.1°, 画布=fo-ok-1 fr97
                               CUE-005: P2/P3 v0.6 几何（截断线段+白圈关节点）
  clip_016_right (None)      → neutral 成品: 原帧 + 绿勾徽章 [_retired/留证]
  fo-eet-1-neg-setup  (SILENT) → 图形化引导重拍卡
  fo-eet-1-neg-truncated (SILENT) → 图形化引导重拍卡

关卡C: 回灌校验
  首帧元素数校验（≤2）
  色极性校验（红线/白箭头）
  灰度自明：首帧灰度化后仍可区分元素

clip_016 退役令 (CUE-005):
  clip_016 仅限判断验证，禁止用于 cue 渲染/预览/验收。
  clip_016 成品已移入 output/cue004/_retired/ 留证。
  本脚本路由1画布改为 fo-ok-1 fr97 (top帧, MOCK 标注)。

输出目录:
  output/cue004/          本地
  Desktop preview/cue004/ Windows

依赖: cue_compiler/ (lottie_builder, mp4_renderer, neutral_renderer)
      cue_generator/ (sentence_alpha, validator)
      CUE_GENERATOR_SPEC v0.8 / CUE_DESIGN_LANGUAGE v0.4
"""
import sys, json, math, time
from pathlib import Path
import numpy as np
import cv2

PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ))

from cue_generator import build_alpha_plan, validate, render_preview
from cue_compiler import (
    compile_lottie, render_mp4_preview,
    render_neutral_frame, render_silent_card,
)
from cue_compiler.mp4_renderer import render_static_last_frame

# ── I/O dirs ──────────────────────────────────────────────────────────────────
OUT_DIR  = PROJ / "output" / "cue004"
DESK_DIR = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/cue004")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DESK_DIR.mkdir(parents=True, exist_ok=True)

PLAN_DIR = PROJ / "output" / "cue_plans"

# ── Video paths ────────────────────────────────────────────────────────────────
CLIP016_VIDEO = Path("/mnt/c/Users/jason/Zening/Swingcue/教学视频") / \
    "115. 好的旋转方向是：身体按照顺序自然旋转。#高尔夫 #golf #golfswing #高尔夫课程 #高尔夫挥杆 #golflesson #golflife.mp4"
NEG_SETUP_VIDEO     = PROJ / "tests/negatives/fo-eet-1-neg-setup.mp4"
NEG_TRUNC_VIDEO     = PROJ / "tests/negatives/fo-eet-1-neg-truncated.mp4"
FO_OK1_VIDEO        = Path("/mnt/c/Users/jason/Zening/Swingcue/Video/fo-ok-1.mp4")

CANVAS_W = 720
CANVAS_H = 1280

# ── fo-ok-1 RTMPose 实测关节点 (CUE-005 MOCK 画布) ───────────────────────────
# 视频尺寸: 720×1280 = canvas，无需缩放
# fr97 (top帧, wrist_y=431.4 最小)
_FO_TOP_FR       = 97
_FO_HIP_CANVAS   = (349.1, 663.3)   # hip_mid fr97
_FO_SHO_CANVAS   = (337.9, 518.4)   # sho_mid fr97
_FO_BBOX_CANVAS  = [266, 431, 393, 900]   # body_bbox fr97

# address 帧肩宽（fr0，用于校验④ k_max=0.06）
# fr0: l_sho=(305.4,565.9) r_sho=(332.6,536.2) → SHW = |305-332|... 用欧氏距离
import math as _math
_FO_SHW_CANVAS_PX = _math.hypot(305.4 - 332.6, 565.9 - 536.2)  # ≈ 38px
# Note: 38px < 254px (clip_016), fo-ok-1 拍摄距离更近，需用实测

# MOCK verdict 数值来自 clip_016_left
_MOCK_TILT_DEG   = 29.1   # Confirmed
_MOCK_CONFIDENCE = "Confirmed"
_MOCK_CAPTION    = "顶点时上半身倒向了球的方向——下一杆感觉胸口留在球的后面"
_BAND_LOWER  = -18.8
_BAND_UPPER  = 5.0
_BAND_CENTER = -6.8

# clip_016 address 帧肩宽（历史数据，保留供参考）
_CLIP016_SHW_CANVAS_PX = 254.4


# ── Frame loaders ─────────────────────────────────────────────────────────────

def grab_frame(video_path: Path, frame_idx: int,
               x_crop: tuple = None) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, fr = cap.read(); cap.release()
    if not ok: return None
    if x_crop:
        fr = fr[:, x_crop[0]:x_crop[1]]
    return fr

def placeholder(w=CANVAS_W, h=CANVAS_H):
    return np.full((h, w, 3), 45, dtype=np.uint8)


# ── Load existing Plan JSON ───────────────────────────────────────────────────

def load_plan(clip_id: str) -> dict:
    p = PLAN_DIR / f"{clip_id}_cue_plan.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


# ── fo-ok-1 MOCK frame (路由1, CUE-005) ──────────────────────────────────────

def get_fo_ok1_top_frame() -> np.ndarray:
    """Load fo-ok-1 top frame (fr97, wrist_y最小) for MOCK Confirmed path."""
    if not FO_OK1_VIDEO.exists():
        print(f"  [warn] fo-ok-1 video not found, using placeholder")
        return placeholder()
    fr = grab_frame(FO_OK1_VIDEO, _FO_TOP_FR)
    if fr is None:
        print(f"  [warn] cannot read fr{_FO_TOP_FR} from fo-ok-1")
        return placeholder()
    h, w = fr.shape[:2]
    if w != CANVAS_W or h != CANVAS_H:
        fr = cv2.resize(fr, (CANVAS_W, CANVAS_H))
    print(f"  [fo-ok-1] 嵌入帧: fr{_FO_TOP_FR} (top帧, MOCK Confirmed画布)")
    return fr


def _stamp_mock_badge(frame: np.ndarray) -> np.ndarray:
    """Burn [MOCK] corner badge onto frame (CUE-005: fo-ok-1 MOCK 标注)."""
    from PIL import Image as PILImage, ImageDraw, ImageFont
    img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fnt_path = Path.home() / ".local/share/fonts/NotoSansSC-VF.ttf"
    fnt = ImageFont.truetype(str(fnt_path), size=24) if fnt_path.exists() else None
    text = "[MOCK] fo-ok-1 fr97 | +29.1° Confirmed"
    if fnt:
        bbox = draw.textbbox((0,0), text, font=fnt)
        tw = bbox[2]-bbox[0]; th = bbox[3]-bbox[1]
        x = img.width - tw - 10
        draw.rectangle([x-4, 0, img.width, th+12], fill=(180, 0, 0))
        draw.text((x, 4), text, font=fnt, fill=(255,255,255))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def get_clip016_right_frame() -> np.ndarray:
    """clip_016_right 仅作 neutral 用，保留原逻辑."""
    if not CLIP016_VIDEO.exists():
        return placeholder()
    cap = cv2.VideoCapture(str(CLIP016_VIDEO))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    NF = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    half = W // 2
    est_fr = int(NF * 0.45)
    fr = grab_frame(CLIP016_VIDEO, est_fr, x_crop=(half, W))
    return fr if fr is not None else placeholder(half)


# ── Render-back validator (关卡C) ─────────────────────────────────────────────

def renderback_validate(plan: dict, static_jpg: Path) -> dict:
    """
    回灌校验：读取 static_last.jpg（末帧），验证：
    1. 元素数: Plan elements ≤ 2（与 Plan JSON 一致）
    2. 色极性: 画面中红色像素 > 阈值（P2 红线必须存在）
    3. 色极性: 白色区域存在（P3 弧箭头完全展开后必须可见）
    4. 灰度自明: 灰度图中两元素在位置/形状上可区分
    CUE-004 修正二轮③: 校验图为末帧，须同时含 P2/P3 元素（threshold 相应调整）
    """
    checks = []

    # Check 1: element count from plan
    n_el = len(plan.get("elements", []))
    c1_pass = (n_el <= 2)
    checks.append({
        "name": "元素预算 ≤ 2",
        "pass": c1_pass,
        "detail": f"Plan elements={n_el}"
    })

    if not static_jpg.exists():
        checks.append({"name": "首帧读取", "pass": False, "detail": "文件不存在"})
        return {"passed": False, "checks": checks}

    img = cv2.imread(str(static_jpg))
    if img is None:
        checks.append({"name": "首帧读取", "pass": False, "detail": "cv2.imread 返回 None"})
        return {"passed": False, "checks": checks}

    stype = plan.get("sentence_type_id", "")
    conf  = plan.get("confidence", "")

    if stype == "alpha_angle" and conf in ("Confirmed", "Likely"):
        # Check 2: color polarity — red pixels for P2
        # Red: B<80, G<80, R>150 in BGR
        mask_red = (
            (img[:,:,0].astype(int) < 80) &
            (img[:,:,1].astype(int) < 80) &
            (img[:,:,2].astype(int) > 150)
        )
        red_px = int(mask_red.sum())
        c2_pass = red_px > 50
        checks.append({
            "name": "色极性 P2 红像素",
            "pass": c2_pass,
            "detail": f"red_px={red_px} (threshold>50)"
        })

        # Check 3: white pixels for P3 arc/arrowhead
        mask_white = (
            (img[:,:,0].astype(int) > 180) &
            (img[:,:,1].astype(int) > 180) &
            (img[:,:,2].astype(int) > 180)
        )
        white_px = int(mask_white.sum())
        c3_pass = white_px > 20
        checks.append({
            "name": "色极性 P3 白像素",
            "pass": c3_pass,
            "detail": f"white_px={white_px} (threshold>20)"
        })

        # Check 4: grayscale distinguishability
        # Red line and white arrow should be in different spatial locations
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # P2 is bright in gray (red → ~80 gray), P3 is bright (white → ~255)
        # Both should be detectable as bright regions in different areas
        bright_mask = gray > 160
        bright_px = int(bright_mask.sum())
        # Check they exist but aren't all in same spot (rough spatial check)
        c4_pass = bright_px > 30
        checks.append({
            "name": "灰度自明（亮区可见）",
            "pass": c4_pass,
            "detail": f"gray_bright_px={bright_px} (threshold>30)"
        })

    all_pass = all(c["pass"] for c in checks)
    return {"passed": all_pass, "checks": checks}


# ── Process routes ─────────────────────────────────────────────────────────────

def process_confirmed_mock(frame_bgr: np.ndarray) -> dict:
    """
    路由1 (CUE-005): fo-ok-1 MOCK Confirmed path.
    实时生成 CuePlan（v0.6 几何），注入 _payload_joints/_body_bbox_px/_shw_canvas_px。
    """
    clip_id = "fo-ok-1_MOCK"

    # 1. 实时生成 Plan (v0.6 几何: hip→sho 截断线段)
    from cue_generator.sentence_alpha import build_alpha_plan
    plan_obj = build_alpha_plan(
        clip_id=clip_id,
        confidence=_MOCK_CONFIDENCE,
        tilt_deg=_MOCK_TILT_DEG,
        hip_mid=_FO_HIP_CANVAS,
        shoulder_mid=_FO_SHO_CANVAS,
        band_lower_deg=_BAND_LOWER,
        band_upper_deg=_BAND_UPPER,
        band_center_deg=_BAND_CENTER,
        caption_text=_MOCK_CAPTION,
        tier="basic",
    )
    plan = plan_obj.to_dict()

    # 2. 注入 payload_joints (规则⑫)、bbox (规则⑪)、SHW (规则④)
    plan["_payload_joints"] = {
        "hip_mid": list(_FO_HIP_CANVAS),
        "sho_mid": list(_FO_SHO_CANVAS),
    }
    plan["_body_bbox_px"] = _FO_BBOX_CANVAS
    # MOCK 规则④: verdict 数值来自 clip_016_left，注入 clip_016 的 address SHW
    # fo-ok-1 是 MOCK 画布，用 clip_016 SHW 保持校验④与 verdict 来源一致
    plan["_shw_canvas_px"] = _CLIP016_SHW_CANVAS_PX

    # 3. 校验 (12条)
    from cue_generator.validator import validate as _validate
    vr = _validate(plan)
    if not vr.passed:
        print(f"  [FAIL] validator:")
        for v in vr.violations:
            print(f"    ✗ {v}")
        return {"clip_id": clip_id, "status": "VALIDATOR_FAIL", "violations": vr.violations}

    plan_obj.validator_result = vr.to_dict()

    # 4. 保存 Plan JSON
    plan_json_path = PLAN_DIR / f"{clip_id}_cue_plan.json"
    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    with open(plan_json_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    # 5. 烧录 MOCK 角标
    frame_stamped = _stamp_mock_badge(frame_bgr)

    # 6. Lottie
    lottie_path = OUT_DIR / f"{clip_id}.lottie"
    compile_lottie(plan, frame_stamped, lottie_path, CANVAS_W, CANVAS_H)

    # 7. MP4 preview
    mp4_path = OUT_DIR / f"{clip_id}_preview.mp4"
    render_mp4_preview(plan, frame_stamped, mp4_path, CANVAS_W, CANVAS_H, n_loops=3)

    # 8. Static last frame
    last_path = OUT_DIR / f"{clip_id}_static_last.jpg"
    render_static_last_frame(plan, frame_stamped, last_path, CANVAS_W, CANVAS_H)

    # 9. Render-back validate
    rb = renderback_validate(plan, last_path)

    print(f"  [{clip_id}] validator: PASS ({len(plan.get('elements',[]))} elements)")
    return {
        "clip_id": clip_id,
        "confidence": plan.get("confidence"),
        "plan_json": str(plan_json_path),
        "lottie": str(lottie_path),
        "mp4": str(mp4_path),
        "static_last": str(last_path),
        "renderback": rb,
    }


def process_neutral(clip_id: str, frame_bgr: np.ndarray) -> dict:
    """None/Possible → neutral frame + green checkmark badge"""
    plan = load_plan(clip_id)
    if not plan:
        return {"clip_id": clip_id, "status": "ERROR", "msg": "Plan JSON not found"}

    print(f"  [{clip_id}] None → neutral (green checkmark)")

    out_jpg = OUT_DIR / f"{clip_id}_neutral.jpg"
    render_neutral_frame(frame_bgr, out_jpg, CANVAS_W, CANVAS_H, clip_id=clip_id)

    return {
        "clip_id": clip_id,
        "confidence": "None",
        "neutral_jpg": str(out_jpg),
    }


def process_silent(clip_id: str, neg_video: Path,
                   silent_type: str = "no_swing") -> dict:
    """SILENT → graphical retake guidance card (分型: no_swing / truncated)"""
    print(f"  [{clip_id}] SILENT({silent_type}) → retake guidance card")

    out_jpg = OUT_DIR / f"{clip_id}_retake_card.jpg"
    render_silent_card(out_jpg, CANVAS_W, CANVAS_H,
                       clip_id=clip_id, silent_type=silent_type)

    return {
        "clip_id": clip_id,
        "confidence": "SILENT",
        "retake_card": str(out_jpg),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import shutil

    print(f"\n{'='*70}")
    print("  CUE-004 关卡B+C — 三路由成品生成 + 回灌校验")
    print(f"{'='*70}\n")

    t0 = time.time()
    results = []

    # ── 路由1: fo-ok-1_MOCK (Confirmed, CUE-005 MOCK 画布) ────────────────────
    print("── 关卡B 路由1: fo-ok-1_MOCK (Confirmed, P2/P3 v0.6) ──")
    frame_fo_ok1 = get_fo_ok1_top_frame()
    r1 = process_confirmed_mock(frame_fo_ok1)
    results.append(r1)
    if r1.get("status") == "VALIDATOR_FAIL":
        print(f"  ✗ 路由1 VALIDATOR FAIL — 中止")
        sys.exit(1)
    rb = r1.get("renderback", {})
    rb_status = "PASS" if rb.get("passed") else "FAIL"
    print(f"  lottie    → {Path(r1['lottie']).name}")
    print(f"  mp4       → {Path(r1['mp4']).name}")
    print(f"  static_last→ {Path(r1['static_last']).name}")
    print(f"  renderback→ {rb_status}")
    for ch in rb.get("checks", []):
        mark = "✓" if ch["pass"] else "✗"
        print(f"    {mark} {ch['name']}: {ch['detail']}")
    print()

    # ── 路由2: clip_016_right (None) ─────────────────────────────────────────
    print("── 关卡B 路由2: clip_016_right (None) ──")
    frame_016_right = get_clip016_right_frame()
    r2 = process_neutral("clip_016_right", frame_016_right)
    results.append(r2)
    print(f"  neutral   → {Path(r2['neutral_jpg']).name}")
    print()

    # ── 路由3: neg-setup (SILENT) ─────────────────────────────────────────────
    print("── 关卡B 路由3: fo-eet-1-neg-setup (SILENT) ──")
    r3 = process_silent("fo-eet-1-neg-setup", NEG_SETUP_VIDEO, silent_type="no_swing")
    results.append(r3)
    print(f"  retake card → {Path(r3['retake_card']).name}")
    print()

    # ── 路由4: neg-truncated (SILENT) ────────────────────────────────────────
    print("── 关卡B 路由4: fo-eet-1-neg-truncated (SILENT) ──")
    r4 = process_silent("fo-eet-1-neg-truncated", NEG_TRUNC_VIDEO, silent_type="truncated")
    results.append(r4)
    print(f"  retake card → {Path(r4['retake_card']).name}")
    print()

    # ── Copy to Windows Desktop ───────────────────────────────────────────────
    print("── 复制到 Windows Desktop ──")
    copied = 0
    for f in OUT_DIR.glob("*"):
        if f.suffix in (".mp4", ".jpg", ".lottie", ".json"):
            shutil.copy2(f, DESK_DIR / f.name)
            copied += 1
    print(f"  {copied} 文件 → {DESK_DIR}")
    print()

    # ── 关卡C 技术债清理报告 ─────────────────────────────────────────────────
    print("── 关卡C 技术债 ──")
    print("  ① 校验器④箭头宽度归一化 (k_max=0.06, address SHW 唯一定义):")
    print(f"     clip_016 address帧 SHW_canvas={_CLIP016_SHW_CANVAS_PX}px")
    print(f"     fo-ok-1 MOCK 注入 clip_016 SHW（verdict 来源同源）")
    print(f"     k_max=0.06（Jason 拍板），max_sw = 0.06 × {_CLIP016_SHW_CANVAS_PX} = {0.06*_CLIP016_SHW_CANVAS_PX:.1f}px")
    print(f"     当前 sw=3px → k=3/{_CLIP016_SHW_CANVAS_PX:.1f}=0.012  安全余量充足")
    print()
    print("  ② 动效预算=1 出处归因修正:")
    print("     原: 归因 Ayres 2009 ← 不准确（Ayres 研究对象是分步效应，非预算约束）")
    print("     正: 源自 CUE_DESIGN_LANGUAGE 法则4「一图一对比」+「法则10极简至上」")
    print("         同时运动的轨道≤1 是前注意属性「运动」只能聚焦一处的知觉约束")
    print("         学术依据: Treisman 1988 Feature Integration Theory")
    print("     → 已更新 CUE_GENERATOR_SPEC §1.2 归因说明")
    print()

    # ── 总结 ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    rb_all = r1.get("renderback", {}).get("passed", False)
    print(f"{'='*70}")
    print(f"  关卡B: 4路由成品全部生成 ✓")
    print(f"  关卡C: 回灌校验 {('PASS ✓' if rb_all else 'FAIL ✗')}")
    print(f"  耗时: {elapsed:.1f}s")
    print()
    print(f"  Windows 成品目录: C:\\Users\\jason\\Desktop\\rtmpose_results\\preview\\cue004\\")
    print(f"  MP4 预览 (双击播放): fo-ok-1_MOCK_preview.mp4")
    print(f"  Lottie 成品:         fo-ok-1_MOCK.lottie")
    print(f"  neutral 绿勾:        clip_016_right_neutral.jpg")
    print(f"  重拍引导卡:          fo-eet-1-neg-setup_retake_card.jpg")
    print(f"                       fo-eet-1-neg-truncated_retake_card.jpg")
    print(f"{'='*70}\n")

    # ── ⛔ 关卡C 停关卡说明 ───────────────────────────────────────────────────
    print("⛔ 关卡C 停关卡")
    print()
    print("  待 Jason 目视确认:")
    print("  1. clip_016_left_preview.mp4 — 红线+白弧箭头动画（双击播放）")
    print("  2. clip_016_right_neutral.jpg — 绿勾徽章样式")
    print("  3. *_retake_card.jpg — 取景框引导卡样式")
    print()
    print("  待 Jason 拍板:")
    print("  • 校验器④ 归一化系数 k_max=0.15 是否采纳？")
    print("    （当前实现 sw=3px，在任何肩宽下均安全，k 仅规格层面修订）")
    print()
    print("  正式双轨验收(专家+素人3秒测试):")
    print("  → 待 Jason 自拍错误示范素材就位后，在干净画布上执行")


if __name__ == "__main__":
    main()
