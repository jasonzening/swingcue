"""
cue_compiler/neutral_renderer.py
关卡B 非诊断路由渲染：

neutral (None/Possible):
  原帧 + 右上角绿勾徽章（"✓ 此项正常"），无其他元素。

SILENT:
  图形化引导重拍卡：
    - 取景框轮廓示意（相机取景框线框）
    - 人形站位示意（简单骨架轮廓占位）
    - 一句话文案（中文）
    - 禁黑屏纯文字（CUE-001 SILENT 黑屏问号图此处废弃替换）

依据: CUE-004 Jason 裁决 2026-07-05
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
from PIL import Image as PILImage, ImageDraw, ImageFont


_NOTO_PATH = Path.home() / ".local/share/fonts/NotoSansSC-VF.ttf"

def _font(size: int) -> ImageFont.FreeTypeFont | None:
    if _NOTO_PATH.exists():
        return ImageFont.truetype(str(_NOTO_PATH), size=size)
    return None

def _cv_to_pil(bgr: np.ndarray) -> PILImage.Image:
    return PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def _pil_to_cv(img: PILImage.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ── NEUTRAL: original frame + green checkmark badge ───────────────────────────

def render_neutral_frame(
    frame_bgr: np.ndarray,
    out_jpg: Path,
    canvas_w: int = 720,
    canvas_h: int = 1280,
    clip_id: str = "",
) -> Path:
    """
    neutral 路由: 原帧 + 右上角绿勾徽章。
    文案: "✓ 此项正常"
    """
    out_jpg = Path(out_jpg)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)

    canvas = cv2.resize(frame_bgr, (canvas_w, canvas_h))
    img = _cv_to_pil(canvas)
    draw = ImageDraw.Draw(img)

    fnt_badge = _font(36)
    fnt_sub   = _font(22)

    # Green checkmark badge — top-right corner
    badge_text = "✓ 此项正常"
    if fnt_badge:
        bbox = draw.textbbox((0,0), badge_text, font=fnt_badge)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        pad = 10
        bx = canvas_w - tw - pad*2 - 8
        by = 12
        # badge background
        draw.rounded_rectangle(
            [bx-pad, by-pad//2, bx+tw+pad, by+th+pad//2],
            radius=10,
            fill=(20, 160, 60),
        )
        draw.text((bx, by), badge_text, font=fnt_badge, fill=(255,255,255))

    # Clip ID sub-label (small, semi-transparent grey)
    if fnt_sub and clip_id:
        draw.text((10, canvas_h - 30), clip_id, font=fnt_sub, fill=(180,180,180))

    out_cv = _pil_to_cv(img)
    cv2.imwrite(str(out_jpg), out_cv, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out_jpg


# ── SILENT: camera viewfinder card + retake guidance ─────────────────────────

def render_silent_card(
    out_jpg: Path,
    canvas_w: int = 720,
    canvas_h: int = 1280,
    clip_id: str = "",
    silent_type: str = "no_swing",   # "no_swing" | "truncated"
) -> Path:
    """
    SILENT 路由: 图形化引导重拍卡。CUE-004 修正二轮⑤: 按病因分型。
    silent_type:
      "no_swing"  — 检测到挥杆振幅不足 / 无完整挥杆（neg-setup 型）
      "truncated" — 检测到截断型，上杆顶点前视频已结束（neg-truncated 型）
    替换 CUE-001 黑屏问号图。
    """
    out_jpg = Path(out_jpg)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)

    # ── 病因分型文案 ────────────────────────────────────────────────────────────
    if silent_type == "truncated":
        title_text = "请拍摄完整挥杆"
        instructions = [
            "勿提前停止录制",
            "拍到起杆、上杆、击球全过程",
        ]
        vf_color = (80, 160, 255)   # 蓝色取景框（信息提示）
    else:   # no_swing (default)
        title_text = "请重新录制"
        instructions = [
            "正面站立，完整挥杆",
            "确保全身在取景框内",
        ]
        vf_color = (80, 180, 255)   # 原蓝色

    # Deep grey background
    bg = np.full((canvas_h, canvas_w, 3), (35, 35, 40), dtype=np.uint8)
    img = _cv_to_pil(bg)
    draw = ImageDraw.Draw(img)

    fnt_title = _font(38)
    fnt_body  = _font(30)
    fnt_sub   = _font(22)

    # ── Camera viewfinder frame (inner guide box) ─────────────────────────────
    vf_margin = 60
    vf_x0 = vf_margin
    vf_y0 = int(canvas_h * 0.12)
    vf_x1 = canvas_w - vf_margin
    vf_y1 = int(canvas_h * 0.82)
    vf_color = (80, 180, 255)   # blue-ish viewfinder
    corner_len = 36
    corner_w = 4

    # Draw viewfinder corner brackets
    corners = [
        # top-left
        [(vf_x0, vf_y0), (vf_x0+corner_len, vf_y0)],
        [(vf_x0, vf_y0), (vf_x0, vf_y0+corner_len)],
        # top-right
        [(vf_x1-corner_len, vf_y0), (vf_x1, vf_y0)],
        [(vf_x1, vf_y0), (vf_x1, vf_y0+corner_len)],
        # bottom-left
        [(vf_x0, vf_y1-corner_len), (vf_x0, vf_y1)],
        [(vf_x0, vf_y1), (vf_x0+corner_len, vf_y1)],
        # bottom-right
        [(vf_x1, vf_y1-corner_len), (vf_x1, vf_y1)],
        [(vf_x1-corner_len, vf_y1), (vf_x1, vf_y1)],
    ]
    for p0, p1 in corners:
        draw.line([p0, p1], fill=vf_color, width=corner_w)

    # ── Simplified human figure silhouette ────────────────────────────────────
    fig_cx  = canvas_w // 2
    fig_top = vf_y0 + 40

    # Figure proportional to viewfinder height
    vf_h = vf_y1 - vf_y0
    head_r  = int(vf_h * 0.07)
    head_cy = fig_top + head_r
    body_top   = head_cy + head_r + 8
    body_bot   = body_top + int(vf_h * 0.28)
    leg_split  = body_bot + 8
    leg_bot    = leg_split + int(vf_h * 0.22)
    arm_y      = body_top + int(vf_h * 0.06)
    arm_spread = int(vf_h * 0.12)

    fig_color  = (200, 200, 210)
    fig_w      = 4

    # Head
    draw.ellipse(
        [fig_cx - head_r, head_cy - head_r, fig_cx + head_r, head_cy + head_r],
        outline=fig_color, width=fig_w
    )
    # Torso
    draw.line([(fig_cx, body_top), (fig_cx, body_bot)], fill=fig_color, width=fig_w)
    # Arms
    draw.line([(fig_cx - arm_spread, arm_y), (fig_cx + arm_spread, arm_y)],
              fill=fig_color, width=fig_w)
    # Legs
    leg_spread = int(vf_h * 0.09)
    draw.line([(fig_cx, leg_split), (fig_cx - leg_spread, leg_bot)],
              fill=fig_color, width=fig_w)
    draw.line([(fig_cx, leg_split), (fig_cx + leg_spread, leg_bot)],
              fill=fig_color, width=fig_w)

    # ── Golf club hint (simple line from hands down) ──────────────────────────
    club_color = (120, 160, 120)
    draw.line([(fig_cx, arm_y), (fig_cx + int(arm_spread*1.3), leg_bot + 20)],
              fill=club_color, width=3)

    # ── Guide arrows pointing to figure edges → "stay in frame" ─────────────
    arrow_color = (80, 180, 255)
    # left edge arrow pointing right
    draw.line([(vf_x0+20, canvas_h//2), (vf_x0+55, canvas_h//2)],
              fill=arrow_color, width=3)
    draw.polygon([(vf_x0+55, canvas_h//2-8),
                  (vf_x0+70, canvas_h//2),
                  (vf_x0+55, canvas_h//2+8)], fill=arrow_color)
    # right edge arrow pointing left
    draw.line([(vf_x1-20, canvas_h//2), (vf_x1-55, canvas_h//2)],
              fill=arrow_color, width=3)
    draw.polygon([(vf_x1-55, canvas_h//2-8),
                  (vf_x1-70, canvas_h//2),
                  (vf_x1-55, canvas_h//2+8)], fill=arrow_color)

    # ── Title ─────────────────────────────────────────────────────────────────
    if fnt_title:
        tb = draw.textbbox((0,0), title_text, font=fnt_title)
        tw = tb[2]-tb[0]
        tx = (canvas_w - tw) // 2
        ty = vf_y1 + 22
        draw.text((tx, ty), title_text, font=fnt_title, fill=(255, 220, 80))

    # ── Instruction text ──────────────────────────────────────────────────────
    if fnt_body:
        iy = vf_y1 + 80
        for line in instructions:
            lb = draw.textbbox((0,0), line, font=fnt_body)
            lw = lb[2]-lb[0]
            lx = (canvas_w - lw) // 2
            # outline
            for dx,dy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                draw.text((lx+dx, iy+dy), line, font=fnt_body, fill=(20,20,20))
            draw.text((lx, iy), line, font=fnt_body, fill=(220,220,220))
            iy += lb[3]-lb[1] + 8

    # Clip ID
    if fnt_sub and clip_id:
        draw.text((10, canvas_h - 30), clip_id, font=fnt_sub, fill=(120,120,120))

    out_cv = _pil_to_cv(img)
    cv2.imwrite(str(out_jpg), out_cv, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out_jpg
