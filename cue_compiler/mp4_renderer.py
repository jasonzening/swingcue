"""
cue_compiler/mp4_renderer.py
Lottie 动画 → MP4 预览（供 Jason 在 Windows 直接双击查看）

不依赖 Lottie 播放器：直接用 OpenCV + PIL 逐帧绘制动画。
动画逻辑与 lottie_builder 保持一致：
  ANIM_DUR_FR = 54fr  (1.8s @ 30fps)  — arc 从 0% 扫到 100%
  PAUSE_DUR_FR = 15fr (0.5s)          — 停顿保持 100%
  总帧数 = 69fr，单次循环，MP4 写入 3 次循环（约 6.9s）供预览

MP4 编码: MPEG-4 (mp4v) via OpenCV，Jason Windows 双击可直接播放。
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import cv2
from PIL import Image as PILImage, ImageDraw, ImageFont


# ── font ──────────────────────────────────────────────────────────────────────
_NOTO_PATH = Path.home() / ".local/share/fonts/NotoSansSC-VF.ttf"

def _font(size: int) -> ImageFont.FreeTypeFont | None:
    if _NOTO_PATH.exists():
        return ImageFont.truetype(str(_NOTO_PATH), size=size)
    return None


# ── colour helpers ─────────────────────────────────────────────────────────────
def _hex_rgb(h: str) -> tuple[int,int,int]:
    h = h.lstrip("#")
    return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))

def _cv_bgr(h: str) -> tuple[int,int,int]:
    r,g,b = _hex_rgb(h)
    return (b,g,r)


# ── geometry helpers ───────────────────────────────────────────────────────────
def _arc_pts(cx,cy,r, a_from_deg, a_to_deg, n=64):
    start = math.radians(-90 + a_from_deg)
    end   = math.radians(-90 + a_to_deg)
    return [
        (int(cx + r * math.cos(start + (end-start)*i/n)),
         int(cy + r * math.sin(start + (end-start)*i/n)))
        for i in range(n+1)
    ]

def _arrowhead(cx,cy,r, a_to_deg, tang_dir, hl=18, hw=11):
    tip_rad = math.radians(-90 + a_to_deg)
    tx = cx + r * math.cos(tip_rad)
    ty = cy + r * math.sin(tip_rad)
    tang_x = -math.sin(tip_rad) * tang_dir
    tang_y =  math.cos(tip_rad) * tang_dir
    bx = tx - tang_x*hl; by = ty - tang_y*hl
    px = -tang_y; py = tang_x
    return np.array([
        [int(tx), int(ty)],
        [int(bx + px*hw/2), int(by + py*hw/2)],
        [int(bx - px*hw/2), int(by - py*hw/2)],
    ])


# ── P2: static self-luminous line ─────────────────────────────────────────────
def _draw_p2(canvas: np.ndarray, el: dict) -> None:
    anc = [int(x) for x in el["anchor"]["coords_px"]]
    tip = [int(x) for x in el["anchor"]["secondary_coords_px"]]
    col = el["color"]
    bgr = _cv_bgr(col["stroke_hex"])
    sw  = col["stroke_width_px"]

    # glow halo
    cv2.line(canvas, anc, tip, bgr, sw * 6, cv2.LINE_AA)
    # overlay with lower alpha to simulate glow opacity
    glow = np.zeros_like(canvas)
    cv2.line(glow, anc, tip, bgr, sw * 6, cv2.LINE_AA)
    cv2.addWeighted(canvas, 1.0, glow, 0.35, 0, canvas)

    # re-draw core bright line on top
    cv2.line(canvas, anc, tip, bgr, sw, cv2.LINE_AA)


# ── P3: animated arc arrow (progress 0.0→1.0) ────────────────────────────────
def _draw_p3(canvas: np.ndarray, el: dict, progress: float) -> None:
    if progress <= 0:
        return
    sp  = el["shape_params"]
    col = el["color"]
    bgr = _cv_bgr(col["stroke_hex"])
    sw  = col["stroke_width_px"]

    cx,cy  = el["anchor"]["coords_px"]
    r      = sp.get("radius_px", 160)
    a_from = sp.get("angle_from_deg", 29.1)
    a_to   = sp.get("angle_to_deg", -6.8)
    tang_dir = 1 if a_to < a_from else -1

    # partial arc up to progress
    a_cur = a_from + (a_to - a_from) * progress
    pts = _arc_pts(cx, cy, r, a_from, a_cur, n=max(4, int(64*progress)))
    for i in range(len(pts)-1):
        cv2.line(canvas, pts[i], pts[i+1], bgr, sw, cv2.LINE_AA)

    # arrowhead visible from 90% progress
    if progress >= 0.90:
        ah_alpha = min(1.0, (progress - 0.90) / 0.10)
        ah_pts = _arrowhead(cx, cy, r, a_cur, tang_dir)
        # blend arrowhead
        overlay = canvas.copy()
        cv2.fillConvexPoly(overlay, ah_pts, bgr, cv2.LINE_AA)
        cv2.addWeighted(overlay, ah_alpha, canvas, 1-ah_alpha, 0, canvas)


# ── caption badge ─────────────────────────────────────────────────────────────
def _draw_caption(img_bgr: np.ndarray, text: str) -> np.ndarray:
    if not text:
        return img_bgr
    h, w = img_bgr.shape[:2]
    img = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fnt = _font(34)
    if fnt is None:
        return img_bgr
    bbox = draw.textbbox((0,0), text, font=fnt)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    by = h - th - 20
    bx = max(8, (w - tw)//2)
    # background strip
    draw.rectangle([0, by-10, w, h], fill=(0,0,0,200))
    # outline
    for dx,dy in [(-1,-1),(-1,1),(1,-1),(1,1),(0,-2),(0,2),(-2,0),(2,0)]:
        draw.text((bx+dx, by+dy), text, font=fnt, fill=(20,20,20))
    draw.text((bx, by), text, font=fnt, fill=(255,255,255))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ── easing function (ease_in_out cubic) ───────────────────────────────────────
def _ease_inout(t: float) -> float:
    """Cubic ease in-out: t in [0,1] → [0,1]."""
    if t < 0.5:
        return 4 * t * t * t
    else:
        p = -2 * t + 2
        return 1 - (p * p * p) / 2


# ── Main MP4 renderer ─────────────────────────────────────────────────────────

def render_mp4_preview(
    plan: dict,
    frame_bgr: "np.ndarray",
    out_mp4: Path,
    canvas_w: int = 720,
    canvas_h: int = 1280,
    n_loops: int = 3,
) -> Path:
    """
    Render MP4 preview of the Cue animation.
    n_loops: number of animation cycles to write (default 3, ~6.9s preview).

    Animation frames mirror Lottie timeline:
      ANIM_DUR_FR=54  → arc sweeps 0→100% (eased)
      PAUSE_DUR_FR=15 → arc holds at 100%
      Total: 69 frames per cycle @ 30fps

    Returns out_mp4 Path.
    """
    ANIM_FR  = int(1.8 * 30)   # 54
    PAUSE_FR = int(0.5 * 30)   # 15
    LOOP_FR  = ANIM_FR + PAUSE_FR  # 69
    FPS_OUT  = 30

    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    # Resize background
    bg = cv2.resize(frame_bgr, (canvas_w, canvas_h))

    # Get plan elements
    stype = plan.get("sentence_type_id", "")
    conf  = plan.get("confidence", "")
    elements = plan.get("elements", [])
    caption  = plan.get("caption_badge", {}).get("text", "")

    # Locate P2 and P3
    p2 = next((e for e in elements if e["primitive"] == "P2"), None)
    p3 = next((e for e in elements if e["primitive"] == "P3"), None)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_mp4), fourcc, FPS_OUT, (canvas_w, canvas_h))

    total_frames = LOOP_FR * n_loops

    for f in range(total_frames):
        frame_in_cycle = f % LOOP_FR

        canvas = bg.copy()

        if stype == "alpha_angle" and conf in ("Confirmed", "Likely"):
            # P2 static
            if p2:
                _draw_p2(canvas, p2)

            # P3 animated
            if p3:
                if frame_in_cycle < ANIM_FR:
                    t = frame_in_cycle / ANIM_FR
                    progress = _ease_inout(t)
                else:
                    progress = 1.0
                _draw_p3(canvas, p3, progress)

        # Caption on every frame
        if caption and stype == "alpha_angle":
            canvas = _draw_caption(canvas, caption)

        vw.write(canvas)

    vw.release()
    return out_mp4


# ── Static first-frame export (for static_downgrade verification) ─────────────

def render_static_first_frame(
    plan: dict,
    frame_bgr: "np.ndarray",
    out_jpg: Path,
    canvas_w: int = 720,
    canvas_h: int = 1280,
) -> Path:
    """
    Export the first frame of the animation (progress=0 for P3).
    Used to verify static_downgrade_note: arc should be at start position.
    Includes P2 (static) + arrowhead stub at a_from position.
    """
    out_jpg = Path(out_jpg)
    bg = cv2.resize(frame_bgr, (canvas_w, canvas_h))
    canvas = bg.copy()

    elements = plan.get("elements", [])
    stype = plan.get("sentence_type_id", "")
    conf  = plan.get("confidence", "")

    if stype == "alpha_angle" and conf in ("Confirmed", "Likely"):
        p2 = next((e for e in elements if e["primitive"] == "P2"), None)
        p3 = next((e for e in elements if e["primitive"] == "P3"), None)
        if p2:
            _draw_p2(canvas, p2)
        # first frame: draw minimal arc stub (5% progress) to show position
        if p3:
            _draw_p3(canvas, p3, 0.05)

    caption = plan.get("caption_badge", {}).get("text", "")
    if caption and stype == "alpha_angle":
        canvas = _draw_caption(canvas, caption)

    # stamp "STATIC DOWNGRADE FRAME 0" label
    cv2.putText(canvas, "STATIC DOWNGRADE - fr0", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,200,0), 2, cv2.LINE_AA)

    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_jpg), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_jpg
