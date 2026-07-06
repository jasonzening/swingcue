"""
sam2_silhouette_retake_card_v2.py
CUE-005: 专业剪影重拍卡 v2

SAM2 提取 fo-ok-1 address 帧 (fr0) 人体掩码
→ 暗底 + 发光轮廓剪影重拍卡（参考 sample_020 美学）
→ 替换原来的火柴棍人
→ 保留取景框角与分型文案

输出:
  output/cue004/retake_silhouette_v2_no_swing.jpg
  output/cue004/retake_silhouette_v2_truncated.jpg
  + 拷贝到 Windows Desktop
"""
from __future__ import annotations
import sys, cv2, json, shutil
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── SAM2 ─────────────────────────────────────────────────────────────────────
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

CKPT  = ROOT / "models/sam2/sam2.1_hiera_small.pt"
CFG   = "configs/sam2.1/sam2.1_hiera_s.yaml"
# Fallback to tiny if small not found
if not CKPT.exists():
    CKPT = ROOT / "models/sam2/sam2.1_hiera_tiny.pt"
    CFG  = "configs/sam2.1/sam2.1_hiera_t.yaml"

VIDEO_FO = Path("/mnt/c/Users/jason/Zening/Swingcue/Video/fo-ok-1.mp4")
ADDR_FR  = 0          # address frame

# fo-ok-1 fr0 RTMPose关节点（kp_cache已验证）
# hip=(324.9,682.1) sho=(311.5,557.2) → body keypoints as SAM2 prompts
PROMPT_PTS = np.array([
    [311.5, 557.2],   # sho_mid
    [324.9, 682.1],   # hip_mid
    [300.0, 620.0],   # torso center estimate
    [310.0, 730.0],   # upper leg estimate
    [300.0, 520.0],   # neck estimate
], dtype=np.float32)
PROMPT_LABELS = np.ones(len(PROMPT_PTS), dtype=np.int32)  # all positive

OUT_DIR  = ROOT / "output/cue004"
DESK_DIR = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/cue004")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DESK_DIR.mkdir(parents=True, exist_ok=True)

CANVAS_W, CANVAS_H = 720, 1280


def load_frame(video_path: Path, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    assert ok, f"Cannot read frame {idx}"
    return fr


def get_sam2_mask(frame_rgb: np.ndarray) -> np.ndarray:
    """Run SAM2 with keypoint prompts, return binary mask (H,W) bool."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  SAM2 device: {device}, ckpt={CKPT.name}")

    sam2_model = build_sam2(CFG, str(CKPT), device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(frame_rgb)

    masks, scores, _ = predictor.predict(
        point_coords=PROMPT_PTS,
        point_labels=PROMPT_LABELS,
        multimask_output=True,
    )
    best = int(np.argmax(scores))
    print(f"  SAM2 best mask idx={best}, score={scores[best]:.3f}")
    return masks[best].astype(bool)


def render_silhouette_card(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    silent_type: str,   # "no_swing" | "truncated"
    out_path: Path,
):
    """
    Render dark-background silhouette retake card:
    - bg: dark navy (#0B0E18)
    - silhouette body: filled darkened desaturated person (30% opacity original)
    - glow outline: red (#FF2200) + soft outer glow
    - vertical reference line: white dashed center line
    - corner frame guides: white L-brackets
    - retake text badge
    """
    h, w = frame_bgr.shape[:2]

    # Scale if not canvas size
    if w != CANVAS_W or h != CANVAS_H:
        frame_bgr = cv2.resize(frame_bgr, (CANVAS_W, CANVAS_H))
        mask_r = cv2.resize(mask.astype(np.uint8), (CANVAS_W, CANVAS_H),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        mask_r = mask

    h, w = CANVAS_H, CANVAS_W

    # 1. Dark background
    bg_color = np.array([24, 14, 11], dtype=np.uint8)   # BGR: #0B0E18 navy
    canvas = np.full((h, w, 3), bg_color, dtype=np.uint8)

    # 2. Silhouette fill: darkened desaturated person (30% opacity)
    body_region = frame_bgr.copy()
    # desaturate + darken body pixels
    gray_body = cv2.cvtColor(body_region, cv2.COLOR_BGR2GRAY)
    dark_fill = np.stack([gray_body]*3, axis=2).astype(np.float32) * 0.25
    # blend into canvas where mask
    mask3 = np.stack([mask_r]*3, axis=2)
    canvas = np.where(mask3, np.clip(dark_fill, 0, 255).astype(np.uint8), canvas)

    # 3. Glow outline: dilate - erode = ring, apply glow
    kernel_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = mask_r.astype(np.uint8) * 255
    dilated  = cv2.dilate(mask_u8, kernel_outer)
    eroded   = cv2.erode(mask_u8, kernel_inner)
    outline  = (dilated.astype(int) - eroded.astype(int)).clip(0, 255).astype(np.uint8)

    # Glow: gaussian blur of outline → soft halo
    glow_color = np.array([0, 34, 204], dtype=np.uint8)   # BGR: #CC2200 red-orange
    glow_layer = np.zeros_like(canvas)
    glow_layer[outline > 0] = glow_color
    glow_blurred = cv2.GaussianBlur(glow_layer, (21, 21), 8)
    canvas = cv2.addWeighted(canvas, 1.0, glow_blurred, 0.85, 0)

    # Core outline
    core_color = (0, 34, 255)   # BGR: #FF2200
    canvas[outline > 64] = core_color

    # 4. Vertical reference line (white dashed, center)
    cx = w // 2
    dash_on, dash_off = 18, 10
    y = 40
    while y < h - 40:
        cv2.line(canvas, (cx, y), (cx, min(y + dash_on, h-40)),
                 (180, 180, 180), 1, cv2.LINE_AA)
        y += dash_on + dash_off

    # 5. Corner frame guides (L-brackets)
    bracket_len = 50
    bracket_w   = 3
    margin      = 30
    col         = (200, 200, 200)
    # TL
    cv2.line(canvas, (margin, margin), (margin+bracket_len, margin), col, bracket_w)
    cv2.line(canvas, (margin, margin), (margin, margin+bracket_len), col, bracket_w)
    # TR
    cv2.line(canvas, (w-margin, margin), (w-margin-bracket_len, margin), col, bracket_w)
    cv2.line(canvas, (w-margin, margin), (w-margin, margin+bracket_len), col, bracket_w)
    # BL
    cv2.line(canvas, (margin, h-margin), (margin+bracket_len, h-margin), col, bracket_w)
    cv2.line(canvas, (margin, h-margin), (margin, h-margin-bracket_len), col, bracket_w)
    # BR
    cv2.line(canvas, (w-margin, h-margin), (w-margin-bracket_len, h-margin), col, bracket_w)
    cv2.line(canvas, (w-margin, h-margin), (w-margin, h-margin-bracket_len), col, bracket_w)

    # 6. Text badge (Pillow CJK)
    from PIL import Image as PILImage, ImageDraw, ImageFont
    from pathlib import Path as PL
    fnt_path = PL.home() / ".local/share/fonts/NotoSansSC-VF.ttf"

    pil = PILImage.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    if silent_type == "no_swing":
        main_text  = "未检测到挥杆动作"
        sub_text   = "请面对镜头，完整挥杆一次"
        icon_char  = "⟳"
    else:  # truncated
        main_text  = "请拍摄完整挥杆"
        sub_text   = "勿提前停止录制"
        icon_char  = "↓"

    # Main text (large, centered, bottom 30%)
    fnt_big = ImageFont.truetype(str(fnt_path), 44) if fnt_path.exists() else None
    fnt_sub = ImageFont.truetype(str(fnt_path), 30) if fnt_path.exists() else None

    if fnt_big:
        # background pill for readability
        bb = draw.textbbox((0, 0), main_text, font=fnt_big)
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
        tx = (CANVAS_W - tw) // 2
        ty = int(CANVAS_H * 0.76)
        # pill bg
        pad = 16
        draw.rounded_rectangle([tx-pad, ty-pad//2, tx+tw+pad, ty+th+pad], radius=12,
                                fill=(20, 20, 50, 200))
        draw.text((tx, ty), main_text, font=fnt_big, fill=(255, 255, 255))

    if fnt_sub:
        bb2 = draw.textbbox((0, 0), sub_text, font=fnt_sub)
        tw2 = bb2[2]-bb2[0]; th2 = bb2[3]-bb2[1]
        tx2 = (CANVAS_W - tw2) // 2
        ty2 = int(CANVAS_H * 0.76) + (draw.textbbox((0,0), main_text, font=fnt_big)[3] if fnt_big else 50) + 20
        draw.text((tx2, ty2), sub_text, font=fnt_sub, fill=(180, 180, 220))

    # Small label top-left
    if fnt_sub:
        draw.text((margin+5, margin+5), "重新录制", font=fnt_sub, fill=(255, 60, 60))

    canvas = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  → {out_path}")


def main():
    print("=" * 60)
    print("  CUE-005 剪影重拍卡 v2 — SAM2 + 发光轮廓")
    print("=" * 60)

    # Load address frame
    print(f"  载入 fo-ok-1 fr{ADDR_FR} address 帧...")
    frame_bgr = load_frame(VIDEO_FO, ADDR_FR)
    orig_h, orig_w = frame_bgr.shape[:2]
    print(f"  原始尺寸: {orig_w}x{orig_h}")

    # Resize to canvas if needed
    if orig_w != CANVAS_W or orig_h != CANVAS_H:
        frame_bgr = cv2.resize(frame_bgr, (CANVAS_W, CANVAS_H))
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # SAM2 mask
    print("  运行 SAM2 人体分割...")
    mask = get_sam2_mask(frame_rgb)
    mask_px = int(mask.sum())
    print(f"  掩码像素: {mask_px} ({mask_px/mask.size*100:.1f}%)")

    # Render both variants
    for stype, stem in [("no_swing", "retake_silhouette_v2_no_swing"),
                         ("truncated", "retake_silhouette_v2_truncated")]:
        out = OUT_DIR / f"{stem}.jpg"
        print(f"\n  渲染 {stype}...")
        render_silhouette_card(frame_bgr.copy(), mask, stype, out)
        if DESK_DIR.exists():
            shutil.copy2(out, DESK_DIR / out.name)
            print(f"  WIN → {DESK_DIR / out.name}")

    print("\n  完成 ✓")


if __name__ == "__main__":
    main()
