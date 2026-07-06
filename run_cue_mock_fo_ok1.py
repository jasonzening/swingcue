"""
run_cue_mock_fo_ok1.py
CUE-004 关卡B 验收样张 — MOCK

画布:   fo-ok-1 top 帧 (fr76, 720×1280)
驱动:   clip_016_left verdict 数值 (tilt=+29.1°, Confirmed)
锚点:   fo-ok-1 fr76 实测关节点 (hip_mid, shoulder_mid)
标注:   文件名 + 画面角标均含 [MOCK] 字样

用途:   干净画布上的双轨验收（专家 + 素人 3 秒测试）
        若 Jason 后续提供真实阳性素材，则以真素材样张替换 MOCK 执行正式验收。

依据:   CUE_DESIGN_LANGUAGE v0.4 法则10 极简至上
        α 句型 basic 层: P2 红色现状线(静止) + P3 动画弧箭头(1.8s+0.5s 循环)
"""
from __future__ import annotations
import sys, json, math
from pathlib import Path
import cv2
import numpy as np

# ── project root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from cue_generator.sentence_alpha import build_alpha_plan
from cue_generator.validator import validate
from cue_generator.preview import render_preview

# ── MOCK 参数（clip_016 verdict 数值）────────────────────────────────────────
CLIP_ID      = "fo-ok-1_MOCK_clip016_verdict"
CONFIDENCE   = "Confirmed"
TILT_DEG     = 29.1          # clip_016_left 实测值 (>+28° → Confirmed)
BAND_LOWER   = -18.8
BAND_UPPER   = 5.0
BAND_CENTER  = -6.8
CAPTION_TEXT = "顶点时上半身倒向了球的方向——下一杆感觉胸口留在球的后面"

# ── fo-ok-1 fr76 实测关节点（kp_cache/batch2/fo-ok-1.json, frame=76）────────
# left_hip=(362.3,687.9)  right_hip=(312.6,691.9)
# left_shoulder=(305.4,565.9)  right_shoulder=(332.6,536.2)
HIP_MID = (337.5, 689.9)
SHO_MID = (319.0, 551.1)

# ── 画布：fo-ok-1 fr76 ───────────────────────────────────────────────────────
VIDEO_PATH = Path("/mnt/c/Users/jason/Zening/Swingcue/Video/fo-ok-1.mp4")
TOP_FRAME  = 76

# ── 输出路径 ──────────────────────────────────────────────────────────────────
OUT_DIR     = ROOT / "output" / "cue_plans"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON    = OUT_DIR / "fo-ok-1_MOCK_cue_plan.json"
OUT_PREVIEW = OUT_DIR / "fo-ok-1_MOCK_preview.jpg"
WIN_DIR     = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/cue_plans")


def _load_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    return frame


def _stamp_mock(frame: np.ndarray) -> np.ndarray:
    """Burn a [MOCK] corner badge onto the frame (top-right, red bg)."""
    from PIL import Image as PILImage, ImageDraw, ImageFont
    from pathlib import Path as PL
    img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fnt_path = PL.home() / ".local/share/fonts/NotoSansSC-VF.ttf"
    fnt = ImageFont.truetype(str(fnt_path), size=28) if fnt_path.exists() else None
    text = "[MOCK] fo-ok-1 fr76 | verdict: clip_016_left +29.1° Confirmed"
    if fnt:
        bbox = draw.textbbox((0,0), text, font=fnt)
        tw = bbox[2]-bbox[0]
        x = img.width - tw - 10
        draw.rectangle([x-4, 0, img.width, bbox[3]-bbox[1]+8], fill=(180,0,0))
        draw.text((x, 4), text, font=fnt, fill=(255,255,255))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    print("=" * 70)
    print("  CUE-004 关卡B — MOCK 验收样张生成")
    print("=" * 70)
    print(f"  画布  : fo-ok-1 fr{TOP_FRAME}")
    print(f"  驱动  : clip_016_left verdict  tilt={TILT_DEG}°  conf={CONFIDENCE}")
    print(f"  hip_mid : {HIP_MID}")
    print(f"  sho_mid : {SHO_MID}")
    print()

    # 1. 生成 CuePlan
    plan = build_alpha_plan(
        clip_id      = CLIP_ID,
        confidence   = CONFIDENCE,
        tilt_deg     = TILT_DEG,
        hip_mid      = HIP_MID,
        shoulder_mid = SHO_MID,
        band_lower_deg = BAND_LOWER,
        band_upper_deg = BAND_UPPER,
        band_center_deg= BAND_CENTER,
        caption_text = CAPTION_TEXT,
        tier         = "basic",
    )

    # 2. 校验
    vr = validate(plan.to_dict())
    plan.validator_result = vr.to_dict()

    status = "PASS" if vr.passed else f"FAIL ({len(vr.violations)} violations)"
    print(f"  validator : {status}")
    if not vr.passed:
        for v in vr.violations:
            print(f"    ✗ {v}")
        sys.exit(1)

    # 3. 打印摘要
    print(f"  elements  : {len(plan.elements)}")
    for el in plan.elements:
        anim_str = f"ANIMATED {el.animation_track.duration_s}s" if el.animation_track else "STATIC"
        print(f"    [{el.layer}] {el.primitive}  {el.semantic_role}  {anim_str}")

    # 4. 保存 JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(plan.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"\n  JSON → {OUT_JSON}")

    # 5. 载入画布帧 + 烧录 MOCK 角标
    frame = _load_frame(VIDEO_PATH, TOP_FRAME)
    frame_stamped = _stamp_mock(frame)

    # 6. 渲染预览图
    render_preview(plan, frame_stamped, OUT_PREVIEW)
    print(f"  JPG  → {OUT_PREVIEW}")

    # 7. 复制到 Windows Desktop
    if WIN_DIR.exists() or True:
        WIN_DIR.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(OUT_JSON,    WIN_DIR / OUT_JSON.name)
        shutil.copy(OUT_PREVIEW, WIN_DIR / OUT_PREVIEW.name)
        print(f"  WIN  → {WIN_DIR / OUT_PREVIEW.name}")

    print()
    print("=" * 70)
    print("  MOCK 样张生成完成")
    print()
    print("  验收说明:")
    print("  - 画面右上角红色角标标注 [MOCK]")
    print("  - 预览图头部蓝条标注 [PLAN PREVIEW v0.4] + MOCK clip_id")
    print("  - 若 Jason 后续提供真实阳性素材，以真素材替换本 MOCK 执行正式验收")
    print("=" * 70)


if __name__ == "__main__":
    main()
