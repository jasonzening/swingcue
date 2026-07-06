# CUE_PLAN_SCHEMA.md — Cue Plan JSON 规格

**版本**: v0.1
**日期**: 2026-07-05
**任务**: CUE-003 关卡A

---

## 1. 顶层结构

```jsonc
{
  // ── 元数据 ──────────────────────────────────────────────
  "schema_version":    "cue_plan_v0.1",
  "clip_id":           "clip_016_left",
  "fault_id":          "reverse_pivot",
  "confidence":        "Confirmed",     // Confirmed|Likely|Possible|None|SILENT
  "timestamp_utc":     "2026-07-05T...",

  // ── 整图字段 ─────────────────────────────────────────────
  "sentence_type_id":  "alpha_angle",   // alpha_angle|beta_fence|gamma_deform|delta_sequence|neutral|retake
  "contrast_structure":"single_subject",// single_subject|side_by_side|sequential
  "static_downgrade_note": null,        // string if static fallback differs from animated intent

  "caption_badge": {
    "text":     "顶点时上半身倒向了球的方向——下一杆感觉胸口留在球的后面",
    "position": "bottom_center",        // bottom_center|top_center|none
    "font":     "NotoSansSC",
    "size_px":  28,
    "color_hex":"#FFFFFF",
    "outline_hex":"#141414"
  },

  // ── 元素列表（每个原语一条七元组）─────────────────────────
  "elements": [ /* 见下方七元组定义 */ ],

  // ── 校验结果（由 validator.py 填入）─────────────────────
  "validator_result": {
    "passed":    true,
    "violations": []    // 不通过时列出违规条目 strings
  }
}
```

---

## 2. 元素七元组定义

每个元素（element）是对一个视觉原语的完整描述，包含七个维度：

```jsonc
{
  // ① 原语类型
  "primitive":      "P1",           // P1~P12，见 CUE_DESIGN_LANGUAGE.md §2

  // ② 锚点绑定（来自 verdict payload 坐标）
  "anchor": {
    "source":       "hip_mid",      // payload 字段名（hip_mid|shoulder_mid|keypoint_name|fixed_px）
    "coords_px":    [200, 432],     // 计算后像素坐标 [x, y]
    "secondary_coords_px": null     // 第二端点（如 shoulder_mid），线段类必填
  },

  // ③ 语义角色
  "semantic_role":  "correct_zone", // correct_zone|current_state|direction_instruction|
                                    // correct_shape|fence|joint_ring|region_attention|shape_word

  // ④ 颜色规格（颜色极性=指令极性，引擎强制）
  "color": {
    "fill_hex":     "#00B400",      // 填充色
    "fill_alpha":   0.25,           // 0.0~1.0（0=透明，1=不透明）
    "stroke_hex":   "#00B400",
    "stroke_alpha": 0.8,
    "stroke_width_px": 3            // 校验④：≤6px
  },

  // ⑤ 形态参数（几何描述，与原语类型对应）
  "shape_params": {
    // P1 正确区：楔形
    "type":         "wedge",
    "angle_from_deg": -18.8,        // band_lower
    "angle_to_deg":   5.0,          // band_upper
    "radius_px":    180,            // 楔形半径
    "direction":    "from_anchor_vertical"

    // P2 现状线：从 anchor 到 secondary_anchor 的实线
    // "type": "line"

    // P3+P8 弧箭头：从当前角度扫向目标角度
    // "type": "arc_arrow"
    // "angle_from_deg": 29.1   (tilt_deg)
    // "angle_to_deg":  -6.8    (band_center)
    // "radius_px": 160
    // "arrowhead": {"length_px":14, "width_px":8}
  },

  // ⑥ 动画轨道（null = 静止；只允许一个元素非 null，校验③）
  "animation_track": null,          // 或 AnimationTrack 对象（见下）

  // ⑦ 层级（决定绘制顺序）
  "layer":          "bg"            // bg|mid|fg（bg最底，fg最顶）
}
```

### AnimationTrack 结构

```jsonc
{
  "motion_type":    "arc_sweep",    // arc_sweep|linear_move|fade_in|discrete_steps
  "duration_s":     1.8,            // 扫动时长（校验⑧：1.5~2.0）
  "pause_s":        0.5,            // 扫动结束后停顿（校验⑧：0.5）
  "loop":           true,           // 校验⑧
  "pauseable":      true,           // 校验⑧：Lottie goToAndStop API
  "easing":         "ease_in_out",
  "steps":          null            // δ句型时序类：[{part,start_pct,end_pct,anchor,...},...]
}
```

---

## 3. 句型 α — 角度类模板（Reverse Pivot）

### 3.1 槽位→payload 字段绑定表

| 槽位 | 来自 payload | 说明 |
|------|-------------|------|
| `hip_mid` | `payload.hip_mid` | P1/P3 锚点 |
| `shoulder_mid` | `payload.shoulder_mid` | P2 端点 |
| `tilt_deg` | `payload.tilt_deg` | P2/P3 当前角度 |
| `band_lower_deg` | `payload.band_lower_deg` | P1 楔形起始角 |
| `band_upper_deg` | `payload.band_upper_deg` | P1 楔形结束角 |
| `band_center_deg` | -6.8（flywheel baseline） | P7 正确形线角度；P3 动画终止角 |
| `frame_bgr` | `payload.frame_bgr` | 背景帧 |
| `confidence` | `payload.confidence` | 置信门路由 |

### 3.2 完整元素列表（α 模板，Confirmed/Likely 路径）

```
Layer bg:
  1. P1 正确区 — wedge(band_lower→band_upper, r=180px, 锚:hip_mid)
                  fill=#00B400 α=0.20, stroke=#00B400 α=0.6, w=2px
  2. P7 正确形线 — line(hip_mid→band_center方向端点, r=180px)
                  stroke=#00CC00 α=0.7, w=2px, dash=[6,4]

Layer mid:
  3. P2 现状线 — line(hip_mid→shoulder_mid, 延长至边界)
                  stroke gradient: 带内→#FFA000(橙), 带外→#CC0000(深红), w=3px
                  STATIC（animation_track=null）

Layer fg:
  4. P3+P8 弧箭头 — arc_arrow(锚:shoulder_mid, r=160px,
                    from=tilt_deg, to=band_center_deg)
                  stroke=#FFFFFF w=3px, arrowhead 14×8px
                  ANIMATED: arc_sweep, dur=1.8s, pause=0.5s, loop, pauseable

  5. caption_badge — 底部黑底白字，NotoSansSC 28px
```

### 3.3 None/Possible 路径（中性帧）

```json
{
  "sentence_type_id": "neutral",
  "elements": [],
  "caption_badge": {"text": "此项未发现问题", ...}
}
```

### 3.4 SILENT 路径（重拍引导）

```json
{
  "sentence_type_id": "retake",
  "elements": [],
  "caption_badge": {"text": "画面质量不足，请重新录制：正面站立，完整挥杆，确保全身入镜", ...}
}
```

---

## 4. 对 Lottie 的预留字段对齐

`animation_track` 字段是 Lottie shape layer 动画参数的规格蓝图：

| Cue Plan 字段 | Lottie 对应 |
|--------------|-------------|
| `motion_type: arc_sweep` | Trim Paths + Stroke 动画 |
| `duration_s` | in/out point（帧数 = dur × fps） |
| `pause_s` | 结束 keyframe hold |
| `loop=true` | Layer loop / composition loop |
| `pauseable=true` | goToAndStop API 调用点 |
| `easing: ease_in_out` | Bezier easing handles |

---

*CUE_PLAN_SCHEMA.md v0.1 — CUE-003 关卡A / 2026-07-05*
