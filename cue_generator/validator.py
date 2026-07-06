"""
cue_generator/validator.py
Cue Plan 校验器 — 8条规则，任一不过拒绝出 Plan。

规则依据: CUE_GENERATOR_SPEC_v0.2.md §5
反例样板: INDICATOR_SAMPLE_DECODE_REPORT_v0.3.md
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class ValidationResult:
    passed: bool
    violations: list[str]  # 不通过时每条违规描述

    def to_dict(self) -> dict:
        return {"passed": self.passed, "violations": self.violations}


# ── colour polarity helpers ────────────────────────────────────────────────────

_CORRECT_ROLES = {"correct_zone", "correct_shape", "direction_instruction"}
_ERROR_ROLES   = {"current_state"}
_CORRECT_COLORS_HEX = {"#00b400", "#00cc00", "#00ff00", "#55cc55", "#ffffff",
                        "#ffff00"}  # green + white/yellow = 正确/指令极性
_ERROR_COLORS_HEX   = {"#cc0000", "#ff0000", "#aa0000", "#ffa000", "#ff8000",
                        "#c80000"}  # red/orange = 错误/现状极性


def _hex_lower(h: str) -> str:
    return h.lower().lstrip("#")


def _is_correct_color(fill_hex: str | None, stroke_hex: str | None) -> bool:
    """True if element colour signals correct/target (green / white / yellow family)."""
    for h in [fill_hex, stroke_hex]:
        if h and ("#" + _hex_lower(h)) in _CORRECT_COLORS_HEX:
            return True
    return False


def _is_error_color(fill_hex: str | None, stroke_hex: str | None) -> bool:
    """True if element colour signals error/current-state (red / orange family)."""
    for h in [fill_hex, stroke_hex]:
        if h and ("#" + _hex_lower(h)) in _ERROR_COLORS_HEX:
            return True
    return False


# ── rule implementations ───────────────────────────────────────────────────────

def _rule1_color_polarity(plan: dict) -> str | None:
    """
    ① 色极性=指令极性：绿=正确/目标, 红=错误/现状, 白/亮=指令箭头
    反例: _188 (Image_20260705144254_188_1.jpg) — 原作者正确侧用红色
    """
    for el in plan.get("elements", []):
        role = el.get("semantic_role", "")
        color = el.get("color", {})
        fill  = color.get("fill_hex")
        stroke = color.get("stroke_hex")

        if role in _CORRECT_ROLES:
            if _is_error_color(fill, stroke):
                return (f"规则①违规: semantic_role='{role}' 使用红/橙色 "
                        f"fill={fill} stroke={stroke} (应为绿/白/黄; 参考反例 _188)")
        if role in _ERROR_ROLES:
            if _is_correct_color(fill, stroke) and not _is_error_color(fill, stroke):
                return (f"规则①违规: semantic_role='{role}' 使用绿色 "
                        f"fill={fill} (应为红/橙; 参考反例 _188)")
    return None


def _rule2_current_and_instruction_paired(plan: dict) -> str | None:
    """
    ② 现状+指令成对：有 current_state 必有 direction_instruction，反之亦然
    反例: _190 (Image_20260705144256_190_1.jpg) — 纯指令箭头无现状标记
    """
    roles = [el.get("semantic_role", "") for el in plan.get("elements", [])]
    has_current  = "current_state"        in roles
    has_instruct = "direction_instruction" in roles
    stype = plan.get("sentence_type_id", "")
    if stype in ("neutral", "retake"):
        return None  # 中性/重拍路径豁免
    if has_current and not has_instruct:
        return "规则②违规: 有 current_state 元素但无 direction_instruction (参考反例 _190)"
    if has_instruct and not has_current:
        return "规则②违规: 有 direction_instruction 元素但无 current_state (参考反例 _190)"
    return None


def _rule3_animation_budget(plan: dict) -> str | None:
    """
    ③ 动效预算=1：animation_track 非 null 的元素最多 1 个
    依据: Ayres 2009 — 单次动画运动元素 ≤1 避免注意分割
    """
    animated = [el for el in plan.get("elements", [])
                if el.get("animation_track") is not None]
    if len(animated) > 1:
        primitives = [el.get("primitive") for el in animated]
        return (f"规则③违规: {len(animated)} 个元素有 animation_track (最多1个); "
                f"涉及原语: {primitives}")
    return None


def _rule4_arrow_width_and_overlap(plan: dict) -> str | None:
    """
    ④ 箭头细窄不遮身体：stroke_width_px ≤ 6；
       箭头路径与人体 bbox 中心区重叠比 ≤ 0.25
    反例: _187 (Image_20260705144253_187_1.jpg) 大箭头遮躯干
          _190 (Image_20260705144256_190_1.jpg) 箭头遮挡超标
    """
    ARROW_PRIMITIVES = {"P3", "P8"}
    body_bbox = plan.get("_body_bbox_px")  # [x1,y1,x2,y2] 如有
    for el in plan.get("elements", []):
        if el.get("primitive") not in ARROW_PRIMITIVES:
            continue
        color = el.get("color", {})
        sw = color.get("stroke_width_px", 0)
        if sw > 6:
            return (f"规则④违规: {el.get('primitive')} stroke_width_px={sw} > 6 "
                    f"(参考反例 _187/_190)")
        # overlap check (if body_bbox provided)
        if body_bbox and el.get("shape_params", {}).get("overlap_ratio") is not None:
            ratio = el["shape_params"]["overlap_ratio"]
            if ratio > 0.25:
                return (f"规则④违规: {el.get('primitive')} 与人体中心bbox重叠比={ratio:.2f} > 0.25 "
                        f"(参考反例 _187/_190)")
    return None


def _rule5_grayscale_legible(plan: dict) -> str | None:
    """
    ⑤ 灰度自明：所有元素须通过位置/形状区分，不得仅靠颜色区分
    检测：同 layer 内两个元素若 shape_params.type 相同且
           anchor.coords_px 相近（距离 < 20px），则视为仅靠色彩区分
    反例: _188 — 色语义失误导致灰度下无法区分正误
    """
    elements = plan.get("elements", [])
    for i, a in enumerate(elements):
        for j, b in enumerate(elements):
            if j <= i:
                continue
            if a.get("layer") != b.get("layer"):
                continue
            if a.get("shape_params", {}).get("type") != b.get("shape_params", {}).get("type"):
                continue
            ca = a.get("anchor", {}).get("coords_px")
            cb = b.get("anchor", {}).get("coords_px")
            if ca and cb:
                dx = ca[0] - cb[0]; dy = ca[1] - cb[1]
                dist = (dx*dx + dy*dy) ** 0.5
                if dist < 20:
                    return (f"规则⑤违规: 元素 #{i} ({a.get('primitive')}) 与 #{j} "
                            f"({b.get('primitive')}) 同layer同形态且坐标相近(dist={dist:.1f}px), "
                            f"灰度下无法区分 (参考反例 _188)")
    return None


def _rule6_p11_not_alone(plan: dict) -> str | None:
    """
    ⑥ P11 区域注意圈不得单独成图：必须与至少一个其他原语共存
    反例: _117 (Image_20260602214110_117_1.jpg) — 仅圈无结论，认知负荷高
    """
    primitives = [el.get("primitive") for el in plan.get("elements", [])]
    stype = plan.get("sentence_type_id", "")
    if stype in ("neutral", "retake"):
        return None
    if "P11" in primitives:
        non_p11 = [p for p in primitives if p != "P11"]
        if len(non_p11) == 0:
            return "规则⑥违规: P11 单独出现，无其他原语共存 (参考反例 _117)"
    return None


def _rule7_text_badge_only(plan: dict) -> str | None:
    """
    ⑦ 文字仅命名/裁决徽章：caption_badge.text 不含角度数字、百分比、置信分值
    """
    import re
    badge = plan.get("caption_badge", {})
    text = badge.get("text", "") or ""
    # 禁止: 纯数字度数 / 百分比 / Confirmed/Likely/Possible/None 置信词
    # 允许: 「球的后面」「胸口」等自然语言
    forbidden_patterns = [
        (r'\d+\.?\d*°',       "角度数字(°)"),
        (r'\d+\.?\d*%',       "百分比(%)"),
        (r'\d+\.?\d*\s*deg',  "deg字样"),
        (r'\b(Confirmed|Likely|Possible|top_conf)\b', "置信分值术语"),
    ]
    for pattern, desc in forbidden_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return f"规则⑦违规: caption_badge.text 含 {desc}: 「{text[:40]}」"
    return None


def _rule8_animation_constraints(plan: dict) -> str | None:
    """
    ⑧ 动画三镣铐：有 animation_track 的元素须满足
       1.5 ≤ duration_s ≤ 2.0, pause_s = 0.5, loop=true, pauseable=true
       δ句型须有 steps（非 null）
    反例: _119 (Image_20260602214121_119_1.jpg) — 时序类静态呈现素人读法失败
    依据: Ayres 2009
    """
    stype = plan.get("sentence_type_id", "")
    is_delta = stype == "delta_sequence"

    for el in plan.get("elements", []):
        anim = el.get("animation_track")
        if anim is None:
            continue
        primitive = el.get("primitive", "?")

        dur = anim.get("duration_s")
        if dur is None or not (1.5 <= dur <= 2.0):
            return (f"规则⑧违规: {primitive} animation_track.duration_s={dur} "
                    f"不在 [1.5, 2.0] 范围 (Ayres 2009 认知窗口)")

        pause = anim.get("pause_s")
        if pause is None or abs(pause - 0.5) > 0.05:
            return (f"规则⑧违规: {primitive} animation_track.pause_s={pause} "
                    f"须为 0.5s")

        if not anim.get("loop", False):
            return f"规则⑧违规: {primitive} animation_track.loop 须为 true"

        if not anim.get("pauseable", False):
            return f"规则⑧违规: {primitive} animation_track.pauseable 须为 true"

        if is_delta and anim.get("steps") is None:
            return (f"规则⑧违规: delta_sequence 句型的 {primitive} "
                    f"animation_track.steps 须非 null (参考反例 _119)")
    return None


def _rule10_angle_coord_consistency(plan: dict) -> str | None:
    """
    ⑩a MOCK 几何一致性（CUE-004 修正①）:
       P2 shape_params.tilt_deg 须与 anchor coords 反算角度一致，容差 ±1°。
    """
    import math
    stype = plan.get("sentence_type_id", "")
    if stype in ("neutral", "retake"):
        return None
    for el in plan.get("elements", []):
        if el.get("primitive") != "P2":
            continue
        sp = el.get("shape_params", {})
        stated_tilt = sp.get("tilt_deg")
        if stated_tilt is None:
            continue
        anc = el.get("anchor", {})
        hip = anc.get("coords_px")
        tip = anc.get("secondary_coords_px")
        if hip is None or tip is None:
            continue
        dx = tip[0] - hip[0]
        dy = tip[1] - hip[1]
        if math.hypot(dx, dy) < 1:
            continue
        actual_tilt = math.degrees(math.atan2(dx, -dy))
        diff = abs(actual_tilt - stated_tilt)
        if diff > 1.0:
            return (
                f"规则⑩a违规: P2 shape_params.tilt_deg={stated_tilt:.2f}° "
                f"与坐标反算角度={actual_tilt:.2f}° 偏差={diff:.2f}° > ±1° 容差 "
                f"(CUE-004 修正①: JSON coords 与 tilt_deg 必须一致)"
            )
    return None


def _rule10b_anchor_in_body_bbox(plan: dict) -> str | None:
    """
    ⑩b 锚点在人体 bbox 内（CUE-004 修正二轮②）:
       P2/P3 anchor.coords_px 须在 _body_bbox_px [x1,y1,x2,y2] 范围内。
       _body_bbox_px 由调用方注入（run_cue004/run_cue_generator 从 RTMPose 结果计算）。
       若 _body_bbox_px 不存在则豁免（MOCK/placeholder 路径）。
    """
    stype = plan.get("sentence_type_id", "")
    if stype in ("neutral", "retake"):
        return None
    bbox = plan.get("_body_bbox_px")
    if bbox is None:
        return None   # 无 bbox 信息则豁免
    x1, y1, x2, y2 = bbox
    # 允许 10% 宽高的容差（锚点可落在关节点附近，略超体表）
    dx = (x2 - x1) * 0.10
    dy = (y2 - y1) * 0.10
    for el in plan.get("elements", []):
        prim = el.get("primitive", "")
        if prim not in ("P2", "P3"):
            continue
        anc = el.get("anchor", {})
        pt  = anc.get("coords_px")
        if pt is None:
            continue
        px, py = pt[0], pt[1]
        if not (x1 - dx <= px <= x2 + dx and y1 - dy <= py <= y2 + dy):
            return (
                f"规则⑩b违规: {prim} anchor.coords_px=({px:.1f},{py:.1f}) "
                f"超出人体 bbox [({x1:.0f},{y1:.0f})→({x2:.0f},{y2:.0f})] ±10% 容差 "
                f"(CUE-004 修正二轮②: 锚点须在人体 bbox 内)"
            )
    return None


def _rule9_element_budget(plan: dict) -> str | None:
    """
    ⑨ 元素预算（法则10 极简至上）: basic 层 Plan 中 P1–P12 原语元素数 > 2 直接拒绝。
    fault_view=full_body 时豁免（进阶视图）。
    依据: Jason 裁决 2026-07-05，CUE_DESIGN_LANGUAGE 法则10
    """
    stype = plan.get("sentence_type_id", "")
    if stype in ("neutral", "retake"):
        return None  # 中性/重拍路径豁免
    fault_view = plan.get("fault_view", "single")
    if fault_view == "full_body":
        return None  # 进阶视图豁免
    elements = plan.get("elements", [])
    # 只计 P1–P12 原语（排除 caption_badge，caption_badge 不在 elements 列表里）
    prim_elements = [el for el in elements if el.get("primitive", "").startswith("P")]
    if len(prim_elements) > 2:
        primitives = [el.get("primitive") for el in prim_elements]
        return (
            f"规则⑨违规: basic 层元素数={len(prim_elements)} > 2 (原语: {primitives}); "
            f"法则10极简至上 — 删除多余元素或切换 fault_view=full_body"
        )
    return None


# ── public API ─────────────────────────────────────────────────────────────────

RULES = [
    _rule1_color_polarity,
    _rule2_current_and_instruction_paired,
    _rule3_animation_budget,
    _rule4_arrow_width_and_overlap,
    _rule5_grayscale_legible,
    _rule6_p11_not_alone,
    _rule7_text_badge_only,
    _rule8_animation_constraints,
    _rule9_element_budget,
    _rule10_angle_coord_consistency,
    _rule10b_anchor_in_body_bbox,
]


def validate(plan: dict) -> ValidationResult:
    """
    Run all 11 rules against the Cue Plan dict.
    Returns ValidationResult(passed, violations).
    """
    violations = []
    for rule_fn in RULES:
        msg = rule_fn(plan)
        if msg:
            violations.append(msg)
    return ValidationResult(passed=(len(violations) == 0), violations=violations)
