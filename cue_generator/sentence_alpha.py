"""
cue_generator/sentence_alpha.py
角度类（α）句型 Cue Plan 构造器 — Reverse Pivot 首例实现。

v0.4 改版（Jason 裁决 2026-07-05，法则10 极简至上）:
  basic 层: 仅 P2 红色现状线（静止）+ P3 动画弧箭头，共2元素。
  P1 绿楔整体删除（信号过载 + 违反法则3）。
  P7 绿色正确形线降为 intermediate 层，basic 默认 enabled=False。

v0.5 几何修正（Jason 裁决 2026-07-05，CUE-004 修正①②）:
  P2 tip = hip_mid + tilt_deg 方向 × line_len
    → 线的角度恒等于 verdict tilt_deg，不取画布身体真实轴。
  P3 弧心 = hip_mid，半径 = P2 线长
    → 弧起点正好落在 P2 上端点，几何严格闭合。
  validator 规则⑩: shape_params.tilt_deg 与坐标反算角度容差 ±1°。

MOCK 验收边界（CUE-004 文档备案）:
  MOCK 模式仅用于 QA 渲染几何与动画流程，不承担语义 3 秒测试。
  正式语义验收须在同一球员真实阳性素材（自拍错误示范）就位后执行。

职责: 从 verdict payload 坐标生成 CuePlan dataclass。
无渲染代码，无测量逻辑，纯几何参数组装。
"""
from __future__ import annotations
import math
from .plan_schema import (
    CuePlan, CueElement, AnchorSpec, ColorSpec, AnimationTrack, CaptionBadge
)

# band center from flywheel baseline_v1
_BAND_CENTER_DEG = -6.8

# Line extension beyond shoulder_mid (px in canvas coords)
_LINE_EXTEND_PX = 200


def _tip_from_angle(
    hip: tuple[float, float],
    shoulder_mid: tuple[float, float],
    tilt_deg: float,
    extend_px: float = _LINE_EXTEND_PX,
) -> tuple[float, float]:
    """
    Compute P2 tip from verdict tilt_deg.

    Geometry (face-on image frame, Y-down):
      tilt_deg convention: 0° = vertical (up), + = toward target (screen-right)
      line direction: from hip upward at tilt_deg from vertical.

    Line length = |shoulder_mid - hip_mid| + extend_px  (preserves body proportion)
    Tip = hip + line_len × (sin(tilt_deg), -cos(tilt_deg))

    This guarantees tip coords are exactly consistent with tilt_deg (validator⑩).
    shoulder_mid is used ONLY to determine line length, NOT direction.
    """
    line_len = math.hypot(shoulder_mid[0] - hip[0], shoulder_mid[1] - hip[1]) + extend_px
    tilt_rad = math.radians(tilt_deg)
    tip = (
        hip[0] + line_len * math.sin(tilt_rad),
        hip[1] - line_len * math.cos(tilt_rad),
    )
    return tip, line_len


def build_alpha_plan(
    clip_id: str,
    confidence: str,
    tilt_deg: float,
    hip_mid: tuple[float, float],
    shoulder_mid: tuple[float, float],
    band_lower_deg: float,
    band_upper_deg: float,
    band_center_deg: float = _BAND_CENTER_DEG,
    caption_text: str = "顶点时上半身倒向了球的方向——下一杆感觉胸口留在球的后面",
    tier: str = "basic",          # "basic" | "intermediate" (须 Jason 专项授权)
) -> CuePlan:
    """
    Build α-sentence CuePlan for angle-class faults (Reverse Pivot).

    v0.5 几何:
      P2: hip_mid → tip_from_angle(tilt_deg)     角度 == tilt_deg (误差 <0.001°)
      P3: arc center=hip_mid, radius=P2线长       起点=P2上端点 (几何闭合)

    Coordinate convention (face-on, image frame):
      - tilt_deg > 0 → shoulder tilted toward target (screen-right = LEFT in face-on)
      - band_lower_deg < band_center_deg < band_upper_deg
    """
    fault_id = "reverse_pivot"

    # ── SILENT path ─────────────────────────────────────────────────────────────
    if confidence == "SILENT":
        return CuePlan(
            clip_id=clip_id, fault_id=fault_id, confidence=confidence,
            sentence_type_id="retake",
            contrast_structure="single_subject",
            elements=[],
            caption_badge=CaptionBadge(
                text="画面质量不足，请重新录制：正面站立，完整挥杆，确保全身入镜",
            ),
        )

    # ── Neutral path (Possible / None) ─────────────────────────────────────────
    if confidence in ("Possible", "None"):
        return CuePlan(
            clip_id=clip_id, fault_id=fault_id, confidence=confidence,
            sentence_type_id="neutral",
            contrast_structure="single_subject",
            elements=[],
            caption_badge=CaptionBadge(text="此项未发现问题"),
        )

    # ── Full cue path (Confirmed / Likely) ──────────────────────────────────────

    hip = (float(hip_mid[0]), float(hip_mid[1]))
    sho = (float(shoulder_mid[0]), float(shoulder_mid[1]))

    # P2 tip 从 tilt_deg 反算（v0.5 几何修正）
    p2_tip, line_len = _tip_from_angle(hip, sho, tilt_deg, _LINE_EXTEND_PX)

    # Deviation amount for P2 colour gradient
    dev = tilt_deg - band_upper_deg          # positive = how far outside band
    if dev < 0:     p2_stroke = "#FFA000"    # orange: inside band (Likely only)
    elif dev < 10:  p2_stroke = "#CC4400"    # deep orange
    else:           p2_stroke = "#CC0000"    # red: well outside

    elements: list[CueElement] = []

    # ── intermediate 层可选: P7 正确形线（basic 默认跳过）──────────────────────
    if tier == "intermediate":
        _WEDGE_RADIUS_PX = 180
        tilt_rad_center = math.radians(band_center_deg)
        p7_tip = (
            hip[0] + _WEDGE_RADIUS_PX * math.sin(tilt_rad_center),
            hip[1] - _WEDGE_RADIUS_PX * math.cos(tilt_rad_center),
        )
        elements.append(CueElement(
            primitive="P7",
            anchor=AnchorSpec(
                source="hip_mid", coords_px=list(hip),
                secondary_coords_px=list(p7_tip),
            ),
            semantic_role="correct_shape",
            color=ColorSpec(
                fill_hex="#00CC00", fill_alpha=0.0,
                stroke_hex="#00CC00", stroke_alpha=0.70,
                stroke_width_px=2,
            ),
            shape_params={
                "type": "line",
                "dash": [6, 4],
                "band_center_deg": band_center_deg,
            },
            animation_track=None,
            layer="bg",
        ))

    # ── mid layer: P2 — 现状线（红色，自发光，静止）────────────────────────────
    # tip 完全由 tilt_deg 决定 → 角度一致性校验⑩恒成立
    elements.append(CueElement(
        primitive="P2",
        anchor=AnchorSpec(
            source="hip_mid", coords_px=list(hip),
            secondary_coords_px=list(p2_tip),
        ),
        semantic_role="current_state",
        color=ColorSpec(
            fill_hex=p2_stroke, fill_alpha=0.0,
            stroke_hex=p2_stroke, stroke_alpha=1.0,
            stroke_width_px=3,
        ),
        shape_params={
            "type":          "line",
            "tilt_deg":      tilt_deg,       # 与 coords 严格一致（校验⑩）
            "line_len_px":   line_len,        # 供 P3 radius 引用
            "self_luminous": True,            # rendering hint: glow effect
        },
        animation_track=None,   # STATIC — 校验③安全
        layer="mid",
    ))

    # ── fg layer: P3 — 动画弧箭头 ───────────────────────────────────────────────
    # v0.5 几何修正:
    #   弧心    = hip_mid （P2 的基点）
    #   半径    = P2 线长 （使弧起点 = P2 上端点，几何闭合）
    #   angle_from = tilt_deg  → 弧起点落在 P2 上端点
    #   angle_to   = band_center_deg → 扫向正确带中心
    elements.append(CueElement(
        primitive="P3",
        anchor=AnchorSpec(source="hip_mid", coords_px=list(hip)),
        semantic_role="direction_instruction",
        color=ColorSpec(
            fill_hex="#FFFFFF", fill_alpha=1.0,
            stroke_hex="#FFFFFF", stroke_alpha=1.0,
            stroke_width_px=3,          # ≤6, 校验④
        ),
        shape_params={
            "type":            "arc_arrow",
            "angle_from_deg":  tilt_deg,
            "angle_to_deg":    band_center_deg,
            "radius_px":       line_len,        # = P2 线长，起点 = P2 端点
            "arrowhead": {"length_px": 14, "width_px": 8},
            "motion_semantics": "arc_sweep",    # P8 字典: 弧=旋转
        },
        animation_track=AnimationTrack(
            motion_type="arc_sweep",
            duration_s=1.8,      # 校验⑧: 1.5~2.0
            pause_s=0.5,         # 校验⑧
            loop=True,           # 校验⑧
            pauseable=True,      # 校验⑧
            easing="ease_in_out",
            steps=None,          # α 句型，非时序类
        ),
        layer="fg",
    ))

    caption = CaptionBadge(text=caption_text)

    plan = CuePlan(
        clip_id=clip_id,
        fault_id=fault_id,
        confidence=confidence,
        sentence_type_id="alpha_angle",
        contrast_structure="single_subject",
        elements=elements,
        caption_badge=caption,
        fault_view="single",
        static_downgrade_note=(
            "静态降级: 弧箭头定格在 band_center 端点，"
            "加虚线圆弧辅助表达扫向方向"
        ),
    )
    return plan
