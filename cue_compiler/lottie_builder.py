"""
cue_compiler/lottie_builder.py
Cue Plan JSON → Lottie JSON (After Effects body format v5.5)

供应链纪律: 纯 Python dict 构建 Lottie JSON，零第三方 lottie 依赖。
参考规范: https://lottiefiles.github.io/lottie-docs/

Lottie 坐标系: 原点左上角，Y 向下（与 OpenCV 一致）。
动画时间单位: frames (fr)，本模块统一使用 30fps。

关卡A 实现范围:
  - P2 红色现状线: Shape Layer，静止 polyline + 外发光（glow via 描边+opacity 渐变）
  - P3 白色弧箭头: Shape Layer，trim path 0→100% 动画（1.8s）+ 0.5s 停顿 + 无限循环
  - 背景: 栅格图层（Image Layer），top 帧 JPEG base64 嵌入
  - caption badge: Text Layer 底部
  - .lottie 格式 = Lottie JSON 压入 zip，主文件名 animation.json
"""
from __future__ import annotations
import base64, json, math, zipfile
from io import BytesIO
from pathlib import Path
from typing import Any
import numpy as np
import cv2


# ── Constants ──────────────────────────────────────────────────────────────────
FPS   = 30
W_PX  = 720    # canvas width  (portrait phone)
H_PX  = 1280   # canvas height

ANIM_DUR_FR  = int(1.8 * FPS)   # 54 frames sweep
PAUSE_DUR_FR = int(0.5 * FPS)   # 15 frames pause
LOOP_DUR_FR  = ANIM_DUR_FR + PAUSE_DUR_FR  # 69 total


# ── Lottie JSON helpers ────────────────────────────────────────────────────────

def _val(v: Any) -> dict:
    """Static value keyframe."""
    return {"a": 0, "k": v}

def _kf(t: int, v: Any, easing: bool = True) -> dict:
    """Single keyframe at time t."""
    kf: dict = {"t": t, "s": v if isinstance(v, list) else [v]}
    if easing:
        kf["o"] = {"x": [0.167], "y": [0.167]}
        kf["i"] = {"x": [0.833], "y": [0.833]}
    return kf

def _animated(keyframes: list) -> dict:
    return {"a": 1, "k": keyframes}

def _rgb(hex_str: str) -> list[float]:
    h = hex_str.lstrip("#")
    return [int(h[i:i+2],16)/255 for i in (0,2,4)]

def _make_transform(
    pos=(0,0), anchor=(0,0), scale=(100,100), rotation=0.0, opacity=100
) -> dict:
    return {
        "p": _val(list(pos)),
        "a": _val(list(anchor)),
        "s": _val(list(scale)),
        "r": _val(rotation),
        "o": _val(opacity),
    }


# ── Arc geometry (polyline approximation for Lottie Shape path) ───────────────

def _arc_polyline(cx: float, cy: float, r: float,
                  a_from_deg: float, a_to_deg: float,
                  n_pts: int = 32) -> list[list[float]]:
    """
    Return list of [x,y] pts along arc.
    Angle convention: 0° = up (vertical), + = clockwise (target side in face-on).
    """
    # convert to standard math angles (0=right, CCW positive)
    # tilt 0°=vertical=image-up → standard -90°
    # tilt +θ → rotate CW → standard -90+θ
    start = math.radians(-90 + a_from_deg)
    end   = math.radians(-90 + a_to_deg)
    pts = []
    for i in range(n_pts + 1):
        t = i / n_pts
        angle = start + (end - start) * t
        pts.append([cx + r * math.cos(angle), cy + r * math.sin(angle)])
    return pts

def _arrowhead_pts(cx: float, cy: float, r: float,
                   a_to_deg: float, tang_dir: int,
                   hl: float = 18, hw: float = 11) -> list[list[float]]:
    """Return [tip, left_wing, right_wing] as [x,y] list."""
    tip_rad = math.radians(-90 + a_to_deg)
    tip_x = cx + r * math.cos(tip_rad)
    tip_y = cy + r * math.sin(tip_rad)
    tang_x = -math.sin(tip_rad) * tang_dir
    tang_y =  math.cos(tip_rad) * tang_dir
    base_x = tip_x - tang_x * hl
    base_y = tip_y - tang_y * hl
    perp_x = -tang_y; perp_y = tang_x
    return [
        [tip_x, tip_y],
        [base_x + perp_x*hw/2, base_y + perp_y*hw/2],
        [base_x - perp_x*hw/2, base_y - perp_y*hw/2],
    ]


def _pts_to_lottie_shape(pts: list[list[float]], closed: bool = False) -> dict:
    """Convert point list to Lottie bezier shape (linear — in/out tangents = 0)."""
    verts  = [[p[0], p[1]] for p in pts]
    in_t   = [[0.0, 0.0]] * len(pts)
    out_t  = [[0.0, 0.0]] * len(pts)
    return {
        "ty": "sh",
        "ks": _val({"c": closed, "v": verts, "i": in_t, "o": out_t}),
        "nm": "path",
    }


# ── Layer builders ─────────────────────────────────────────────────────────────

def _layer_base(name: str, ind: int, ty: int) -> dict:
    return {
        "ty": ty,
        "nm": name,
        "ind": ind,
        "ip": 0,
        "op": LOOP_DUR_FR,
        "st": 0,
        "sr": 1,
        "ks": _make_transform(),
    }


def _build_p2_layer(plan: dict, ind: int) -> dict:
    """P2 — red static current-state line with self-luminous style (core + glow)."""
    el = next(e for e in plan["elements"] if e["primitive"] == "P2")
    anc  = el["anchor"]["coords_px"]            # hip_mid
    tip  = el["anchor"]["secondary_coords_px"]  # sho_extended
    col  = el["color"]
    stroke_rgb = _rgb(col["stroke_hex"])
    sw   = col["stroke_width_px"]

    layer = _layer_base("P2_current_state", ind, 4)  # ty=4 shape
    layer["shapes"] = [
        # Core line
        {
            "ty": "gr", "nm": "core_line",
            "it": [
                _pts_to_lottie_shape([anc, tip], closed=False),
                {"ty": "st", "nm": "stroke",
                 "c": _val(stroke_rgb), "o": _val(100),
                 "w": _val(sw), "lc": 2, "lj": 2},
                {"ty": "tr", "p": _val([0,0]), "a": _val([0,0]),
                 "s": _val([100,100]), "r": _val(0), "o": _val(100)},
            ]
        },
        # Glow halo (wider stroke, lower opacity) — self-luminous effect
        {
            "ty": "gr", "nm": "glow_halo",
            "it": [
                _pts_to_lottie_shape([anc, tip], closed=False),
                {"ty": "st", "nm": "glow",
                 "c": _val(stroke_rgb), "o": _val(35),
                 "w": _val(sw * 6), "lc": 2, "lj": 2},
                {"ty": "tr", "p": _val([0,0]), "a": _val([0,0]),
                 "s": _val([100,100]), "r": _val(0), "o": _val(100)},
            ]
        },
    ]
    return layer


def _build_p3_layer(plan: dict, ind: int) -> dict:
    """
    P3 — white animated arc arrow.
    Animation: trim path 0%→100% over ANIM_DUR_FR frames, hold PAUSE_DUR_FR, loop.
    Lottie trim path (ty='tm') on the arc polyline drives the sweep-grow effect.
    """
    el = next(e for e in plan["elements"] if e["primitive"] == "P3")
    sp  = el["shape_params"]
    col = el["color"]
    stroke_rgb = _rgb(col["stroke_hex"])
    sw  = col["stroke_width_px"]

    # arc center = anchor (sho_extended in v0.4)
    anc      = el["anchor"]["coords_px"]
    cx, cy   = anc[0], anc[1]
    r        = sp.get("radius_px", 160)
    a_from   = sp.get("angle_from_deg", 29.1)
    a_to     = sp.get("angle_to_deg", -6.8)
    tang_dir = 1 if a_to < a_from else -1   # CW or CCW sweep

    arc_pts  = _arc_polyline(cx, cy, r, a_from, a_to)
    arrowpts = _arrowhead_pts(cx, cy, r, a_to, tang_dir)

    # Trim path animation: 0%→100% in ANIM_DUR_FR, hold at 100% for PAUSE_DUR_FR
    trim_end_kfs = [
        _kf(0,               [0.0]),
        _kf(ANIM_DUR_FR,     [100.0]),
        _kf(LOOP_DUR_FR - 1, [100.0]),  # hold through pause
    ]

    # Arrowhead opacity: invisible until arc nearly complete
    ah_appear_fr = max(0, ANIM_DUR_FR - 4)
    arrow_opacity_kfs = [
        _kf(0,             [0],   easing=False),
        _kf(ah_appear_fr,  [0],   easing=False),
        _kf(ANIM_DUR_FR,   [100], easing=False),
        _kf(LOOP_DUR_FR-1, [100], easing=False),
    ]

    layer = _layer_base("P3_direction_instruction", ind, 4)
    layer["shapes"] = [
        # Arc stroke with trim
        {
            "ty": "gr", "nm": "arc",
            "it": [
                _pts_to_lottie_shape(arc_pts, closed=False),
                {"ty": "st", "nm": "stroke",
                 "c": _val(stroke_rgb), "o": _val(100),
                 "w": _val(sw), "lc": 2, "lj": 2},
                # trim path — drives the sweep animation
                {"ty": "tm", "nm": "trim",
                 "s": _val(0.0),
                 "e": _animated(trim_end_kfs),
                 "o": _val(0.0), "m": 1},
                {"ty": "tr", "p": _val([0,0]), "a": _val([0,0]),
                 "s": _val([100,100]), "r": _val(0), "o": _val(100)},
            ]
        },
        # Arrowhead (polygon, opacity animated)
        {
            "ty": "gr", "nm": "arrowhead",
            "it": [
                _pts_to_lottie_shape(arrowpts, closed=True),
                {"ty": "fl", "nm": "fill",
                 "c": _val(stroke_rgb), "o": _val(100)},
                {"ty": "tr", "p": _val([0,0]), "a": _val([0,0]),
                 "s": _val([100,100]), "r": _val(0),
                 "o": _animated(arrow_opacity_kfs)},
            ]
        },
    ]
    return layer


def _build_caption_layer(plan: dict, ind: int, w: int, h: int) -> dict:
    """Text layer for caption badge at bottom."""
    badge = plan.get("caption_badge", {})
    text  = badge.get("text", "")
    if not text:
        return None

    layer = _layer_base("caption_badge", ind, 5)  # ty=5 text
    layer["t"] = {
        "d": _val({
            "k": [{
                "s": {
                    "f": "NotoSansSC",   # font family (must be embedded or web font)
                    "fc": [1,1,1,1],     # white fill
                    "s": 32,             # font size px
                    "j": 2,              # center justify
                    "t": text,
                    "sc": [0.08, 0.08, 0.08, 1],  # stroke color (dark outline)
                    "sw": 2,
                    "of": False,
                },
                "t": 0,
            }],
        }),
        "p": {},
        "m": {"a": _val(0), "g": _val(1)},
        "a": [],
    }
    # position: bottom center
    layer["ks"] = _make_transform(pos=[w//2, h - 48])
    return layer


def _build_image_layer(jpeg_bytes: bytes, ind: int,
                        w: int, h: int, asset_id: str = "bg_frame") -> tuple[dict, dict]:
    """
    Build Lottie Image asset + Image layer.
    Returns (asset_dict, layer_dict).
    """
    b64 = base64.b64encode(jpeg_bytes).decode()
    asset = {
        "id": asset_id,
        "w":  w,
        "h":  h,
        "u":  "",
        "p":  f"data:image/jpeg;base64,{b64}",
        "e":  1,  # embedded
    }
    layer = _layer_base("background_frame", ind, 2)  # ty=2 image
    layer["refId"] = asset_id
    layer["w"] = w
    layer["h"] = h
    layer["ks"] = _make_transform(pos=[w//2, h//2])
    return asset, layer


# ── Main compile entry ─────────────────────────────────────────────────────────

def compile_lottie(
    plan: dict,
    frame_bgr: "np.ndarray",
    out_lottie: Path,
    canvas_w: int = W_PX,
    canvas_h: int = H_PX,
) -> dict:
    """
    Compile CuePlan + background frame into Lottie JSON.
    Writes .lottie (dotLottie zip) to out_lottie.
    Returns the Lottie dict.

    Animation: 69 frames at 30fps = 2.3s total (1.8s sweep + 0.5s pause), loop.
    Lottie player native loop + pause capability (no lock on playback controls).

    三镣铐落实:
      duration_s=1.8  → ANIM_DUR_FR=54fr
      pause_s=0.5     → PAUSE_DUR_FR=15fr
      loop=true       → Lottie 'lp':1
      pauseable=true  → 不锁播放控制，Lottie player 原生 pause() 可用
    """
    stype = plan.get("sentence_type_id", "")
    conf  = plan.get("confidence", "")

    # Resize frame to canvas
    frame_resized = cv2.resize(frame_bgr, (canvas_w, canvas_h))
    ok, jpeg_buf = cv2.imencode(".jpg", frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
    jpeg_bytes = jpeg_buf.tobytes() if ok else b""

    layers = []
    assets = []
    ind = 1

    # ── Image layer (background) ─────────────────────────────────────────────
    if jpeg_bytes:
        asset, img_layer = _build_image_layer(jpeg_bytes, ind, canvas_w, canvas_h)
        assets.append(asset)
        layers.append(img_layer)
        ind += 1

    # ── Cue element layers ───────────────────────────────────────────────────
    if stype == "alpha_angle" and conf in ("Confirmed", "Likely"):
        elements = plan.get("elements", [])
        prims = {e["primitive"] for e in elements}

        if "P2" in prims:
            layers.append(_build_p2_layer(plan, ind)); ind += 1

        if "P3" in prims:
            layers.append(_build_p3_layer(plan, ind)); ind += 1

        cap_layer = _build_caption_layer(plan, ind, canvas_w, canvas_h)
        if cap_layer:
            layers.append(cap_layer); ind += 1

    # Lottie JSON top-level
    lottie = {
        "v":  "5.5.7",
        "fr": FPS,
        "ip": 0,
        "op": LOOP_DUR_FR,
        "w":  canvas_w,
        "h":  canvas_h,
        "nm": f"swingcue_{plan.get('clip_id','?')}",
        "ddd": 0,
        "assets": assets,
        "fonts": {"list": []},
        "layers": list(reversed(layers)),  # Lottie: lower index = front
        "markers": [],
        "meta": {
            "g":  "SwingCue CUE-004",
            "a":  "CUE compiler v1.0",
            "k":  "",
            "d":  "CUE_GENERATOR_SPEC v0.4 / CUE_DESIGN_LANGUAGE v0.4",
            "tc": "",
        },
    }

    # Write .lottie (dotLottie = Lottie JSON in zip)
    out_lottie = Path(out_lottie)
    out_lottie.parent.mkdir(parents=True, exist_ok=True)
    lottie_json_bytes = json.dumps(lottie, ensure_ascii=False).encode()
    with zipfile.ZipFile(out_lottie, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("animation.json", lottie_json_bytes)
        # dotLottie manifest (optional but good practice)
        manifest = {
            "generator": "SwingCue CUE-004",
            "version":   "1.0",
            "animations": [{"id": plan.get("clip_id","?"), "speed": 1, "loop": True}],
        }
        zf.writestr("manifest.json", json.dumps(manifest))

    return lottie
