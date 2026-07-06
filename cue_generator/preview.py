"""
cue_generator/preview.py
PIL 工程草图渲染器 — 按 CuePlan 几何参数画标注图供 Jason 审 Plan。

这是工程审查图，不是最终产品渲染，不受法则8 美学约束。
每个元素标注语义角色标签（原语编号 + semantic_role）。
"""
from __future__ import annotations
import math
import numpy as np
import cv2
from pathlib import Path
from PIL import Image as PILImage, ImageDraw, ImageFont

from .plan_schema import CuePlan, CueElement

# ── font ──────────────────────────────────────────────────────────────────────
_NOTO_SC_PATH = Path.home() / ".local/share/fonts/NotoSansSC-VF.ttf"

def _font(size: int) -> ImageFont.FreeTypeFont | None:
    if _NOTO_SC_PATH.exists():
        return ImageFont.truetype(str(_NOTO_SC_PATH), size=size)
    return None


def _hex_to_rgb(h: str, alpha: float = 1.0) -> tuple:
    h = h.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    a = int(alpha * 255)
    return (r, g, b, a)


def _draw_wedge(draw: ImageDraw.ImageDraw, cx: float, cy: float,
                r: float, angle_from: float, angle_to: float,
                fill_rgba: tuple, stroke_rgba: tuple, stroke_w: int) -> None:
    """Draw wedge (fan) from hip_mid. Angles in degrees from vertical, + = right."""
    # Convert to image angles: vertical up = 270° in PIL pieslice
    # tilt_deg convention: + toward target = screen right
    # image: 0°=right, angles go clockwise
    # vertical up = -90° (PIL). tilt=0 = vertical = -90°. tilt=+θ = rotate right = -90+θ
    start_img = -90 + angle_from
    end_img   = -90 + angle_to
    if start_img > end_img:
        start_img, end_img = end_img, start_img
    box = [cx - r, cy - r, cx + r, cy + r]
    # fill
    draw.pieslice(box, start=start_img, end=end_img, fill=fill_rgba[:3] + (fill_rgba[3],))
    # stroke arc
    draw.arc(box, start=start_img, end=end_img, fill=stroke_rgba[:3], width=stroke_w)
    # radial edges
    for angle_deg in [angle_from, angle_to]:
        rad = math.radians(-90 + angle_deg)
        ex = cx + r * math.cos(rad)
        ey = cy + r * math.sin(rad)
        draw.line([(cx, cy), (ex, ey)], fill=stroke_rgba[:3], width=stroke_w)


def _angle_to_vec(deg: float, r: float) -> tuple[float, float]:
    """Convert tilt angle (degrees from vertical, + right) to (dx, dy) vector."""
    rad = math.radians(deg)
    return (r * math.sin(rad), -r * math.cos(rad))


def _draw_line_element(draw: ImageDraw.ImageDraw, el: CueElement) -> None:
    anchor = el.anchor.coords_px
    tip    = el.anchor.secondary_coords_px
    if tip is None:
        return
    col = el.color
    rgba = _hex_to_rgb(col.stroke_hex, col.stroke_alpha)
    shape = el.shape_params
    if shape.get("dash"):
        # dashed line: draw segments
        dash, gap = shape["dash"][0], shape["dash"][1]
        x0,y0 = anchor; x1,y1 = tip
        length = math.hypot(x1-x0, y1-y0)
        if length < 1: return
        ux, uy = (x1-x0)/length, (y1-y0)/length
        d = 0.0
        on = True
        while d < length:
            seg = dash if on else gap
            nd = min(d + seg, length)
            if on:
                draw.line([(x0+ux*d, y0+uy*d), (x0+ux*nd, y0+uy*nd)],
                          fill=rgba[:3], width=col.stroke_width_px)
            d = nd; on = not on
    else:
        draw.line([tuple(anchor), tuple(tip)], fill=rgba[:3], width=col.stroke_width_px)


def _draw_arc_arrow(draw: ImageDraw.ImageDraw, el: CueElement, frame_size: tuple) -> None:
    """Draw arc arrow + arrowhead from angle_from to angle_to."""
    sp = el.shape_params
    cx, cy = el.anchor.coords_px
    r      = sp.get("radius_px", 120)
    a_from = sp.get("angle_from_deg", 0.0)
    a_to   = sp.get("angle_to_deg", 0.0)
    col    = el.color
    stroke_rgb = _hex_to_rgb(col.stroke_hex, col.stroke_alpha)[:3]

    start_img = -90 + min(a_from, a_to)
    end_img   = -90 + max(a_from, a_to)
    box = [cx - r, cy - r, cx + r, cy + r]
    draw.arc(box, start=start_img, end=end_img, fill=stroke_rgb, width=col.stroke_width_px)

    # Arrowhead at a_to end
    ah = sp.get("arrowhead", {"length_px": 14, "width_px": 8})
    tip_rad = math.radians(-90 + a_to)
    tip_x = cx + r * math.cos(tip_rad)
    tip_y = cy + r * math.sin(tip_rad)
    # tangent direction at tip (perpendicular to radius, in sweep direction)
    tang_dir = 1 if a_to > a_from else -1
    tang_x = -math.sin(tip_rad) * tang_dir
    tang_y =  math.cos(tip_rad) * tang_dir
    hl = ah.get("length_px", 14); hw = ah.get("width_px", 8)
    base_x = tip_x - tang_x * hl
    base_y = tip_y - tang_y * hl
    perp_x = -tang_y; perp_y = tang_x
    pts = [
        (int(tip_x), int(tip_y)),
        (int(base_x + perp_x*hw/2), int(base_y + perp_y*hw/2)),
        (int(base_x - perp_x*hw/2), int(base_y - perp_y*hw/2)),
    ]
    draw.polygon(pts, fill=stroke_rgb)


def _label_tag(draw: ImageDraw.ImageDraw, x: float, y: float,
               text: str, fnt, outline: str = "#141414", fill: str = "#FFD700") -> None:
    """Draw a small annotation tag for engineering review."""
    if fnt is None:
        return
    bbox = draw.textbbox((0,0), text, font=fnt)
    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    pad = 3
    draw.rectangle([x-pad, y-pad, x+w+pad, y+h+pad], fill=fill, outline=outline)
    draw.text((x, y), text, font=fnt, fill=outline)


def render_preview(plan: CuePlan, frame_bgr: np.ndarray, out_path: Path) -> Path:
    """
    Render engineering preview image from CuePlan.
    Draws all elements with semantic-role annotation tags.
    v0.4: P2 red line + P3 arc arrow only (basic layer).
          P3 anchor = sho_extended (red line tip), not sho_mid.
    Returns the saved file path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert BGR to RGB PIL
    h, w = frame_bgr.shape[:2]
    img = PILImage.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = PILImage.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    fnt_tag  = _font(14)
    fnt_cap  = _font(26)

    # ── draw each element by layer order ─────────────────────────────────────
    for layer_name in ("bg", "mid", "fg"):
        for el in plan.elements:
            if el.layer != layer_name:
                continue

            col = el.color
            prim = el.primitive
            sp   = el.shape_params

            if prim == "P1" and sp.get("type") == "wedge":
                cx, cy = el.anchor.coords_px
                fill_rgba   = _hex_to_rgb(col.fill_hex, col.fill_alpha)
                stroke_rgba = _hex_to_rgb(col.stroke_hex, col.stroke_alpha)
                _draw_wedge(draw, cx, cy,
                            sp["radius_px"],
                            sp["angle_from_deg"], sp["angle_to_deg"],
                            fill_rgba, stroke_rgba, col.stroke_width_px)
                _label_tag(draw, cx+4, cy - sp["radius_px"] - 22,
                           f"P1 {el.semantic_role}", fnt_tag)

            elif prim in ("P2", "P7") and sp.get("type") == "line":
                _draw_line_element(draw, el)
                ax, ay = el.anchor.coords_px
                role_label = f"{prim} {el.semantic_role}"
                if el.animation_track is None:
                    role_label += " [STATIC]"
                _label_tag(draw, ax + 8, ay - 18, role_label, fnt_tag,
                           fill="#AADDFF" if prim == "P7" else "#FFAAAA")

            elif prim == "P3" and sp.get("type") == "arc_arrow":
                _draw_arc_arrow(draw, el, (w, h))
                ax, ay = el.anchor.coords_px
                anim = el.animation_track
                anim_label = ""
                if anim:
                    anim_label = f" [ANIM {anim.duration_s}s loop]"
                _label_tag(draw, ax + 8, ay + 8,
                           f"P3 {el.semantic_role}{anim_label}", fnt_tag,
                           fill="#FFFFAA")

    # Composite overlay
    img = PILImage.alpha_composite(img, overlay).convert("RGB")
    draw2 = ImageDraw.Draw(img)

    # ── caption badge ─────────────────────────────────────────────────────────
    badge = plan.caption_badge
    if badge.text and fnt_cap:
        bbox = draw2.textbbox((0,0), badge.text, font=fnt_cap)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        bx = max(8, (w - tw)//2)
        by = h - th - 16
        draw2.rectangle([0, by - 8, w, h], fill=(0,0,0,200))
        # outline
        for dx, dy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            draw2.text((bx+dx, by+dy), badge.text, font=fnt_cap, fill=(20,20,20))
        draw2.text((bx, by), badge.text, font=fnt_cap, fill=(255,255,255))

    # ── engineering header ────────────────────────────────────────────────────
    fnt_hdr = _font(16)
    if fnt_hdr:
        hdr = (f"[PLAN PREVIEW v0.4] {plan.clip_id}  conf={plan.confidence}  "
               f"fault={plan.fault_id}  type={plan.sentence_type_id}  "
               f"elements={len(plan.elements)}")
        draw2.rectangle([0, 0, w, 24], fill=(30,30,80))
        draw2.text((4, 4), hdr, font=fnt_hdr, fill=(220,220,255))
        # validator status
        vr = plan.validator_result
        vstatus = "✓ VALID" if vr.get("passed") else f"✗ {len(vr.get('violations',[]))} VIOLATIONS"
        vcol = (80,220,80) if vr.get("passed") else (255,80,80)
        draw2.text((w - 160, 4), vstatus, font=fnt_hdr, fill=vcol)

    # Save
    arr = np.array(img)
    bgr_out = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr_out, [cv2.IMWRITE_JPEG_QUALITY, 93])
    return out_path
