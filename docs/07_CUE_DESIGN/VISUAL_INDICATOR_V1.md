# VISUAL_INDICATOR_V1.md — Reverse Pivot 视觉指示器规格

**版本**: v1.0
**日期**: 2026-07-05
**状态**: 已实现，关卡 B 验收中
**所属任务**: CUE-001
**对应映射表条目**: CUE_DESIGN_LANGUAGE.md 表格第 1 行（Reverse Pivot / 反向脊柱角）

---

## 1. 指示器身份

| 字段 | 值 |
|------|---|
| 错误名称 | Reverse Pivot / Reverse Spine Angle（反向脊柱角） |
| 主机位 | face-on（正面） |
| 诊断阶段 | top 帧（P4，反杆顶点） |
| 句型 | α 角度类（楔 + 线 + 弧箭头） |
| 文案模板 | "顶点时上半身倒向了球的方向——下一杆感觉胸口留在球的后面" |

---

## 2. Payload 输入规格

渲染模块为纯消费者，只接受以下结构化 verdict payload，内部不做任何测量或判断：

```python
@dataclass
class ReversePivotPayload:
    fault_id:        str    # "reverse_pivot"
    confidence:      str    # "Confirmed" | "Likely" | "Possible" | "None" | "SILENT"
    tilt_deg:        float  # shoulder_lateral_tilt at top (度，正=target侧)
    top_frame_idx:   int    # 来自 B 层 anchor.top
    hip_mid:         tuple  # (x, y) 髋中点坐标，来自 top 帧
    shoulder_mid:    tuple  # (x, y) 肩中点坐标，来自 top 帧
    band_lower_deg:  float  # 正确带下界，-18.8°
    band_upper_deg:  float  # 正确带上界，+5.0°
    frame_bgr:       ndarray  # top 帧 BGR 图像（全帧，渲染器自行裁剪/缩放）
    skeleton_kps:    dict   # top 帧全部关节点，用于绘制背景骨架
```

---

## 3. 画面构成规格（top 帧定格，单帧静态图）

### 3.1 背景骨架（全帧）
- 所有骨架连线：低饱和灰色（BGR ≈ (80,80,80)），线宽 2px
- 骨架关节点：灰色小圆点，半径 3px
- cue 锚点关节（left/right hip, left/right shoulder）：不参与灰化，由 P1/P2 接管

### 3.2 P1 — 正确区楔形（Confirmed/Likely 时渲染）
- 锚点：髋中点 (hip_mid)
- 楔形：以髋中点为顶，向上展开，覆盖 band_lower_deg 到 band_upper_deg 的角度范围
- 颜色：(0, 180, 0) 绿色，alpha=0.35 半透明（BGR 叠加）
- 灰度自查：浅灰扇形区域，位置独立可读

### 3.3 P2 — 现状线（Confirmed/Likely 时渲染）
- 从 hip_mid 到 shoulder_mid 的实线
- 偏差幅度渐变（tilt_deg > band_upper_deg）：
  - 轻度（+5° < tilt ≤ +15°）：橙色 (0, 165, 255)
  - 中度（+15° < tilt ≤ +28°）：深橙红 (0, 80, 200)
  - 重度（tilt > +28°）：深红 (0, 0, 200)
- 线宽：4px；端部加小圆点 r=6
- 灰度自查：深色实线，位置形状独立可读

### 3.4 P3 — 弧形方向箭头（Confirmed/Likely 时渲染）
- 起点：shoulder_mid 当前位置
- 终点：shoulder_mid 沿弧线移动到 P1 楔形中心角方向的对应点
- 弧形：以 hip_mid 为圆心，hip→shoulder 距离为半径，沿角度方向画弧
- 颜色：白色 (255,255,255)，描边黑色 2px，线宽 3px，箭头头部 12px
- 灰度自查：亮白箭头，高对比，独立可读

### 3.5 文案区（底部单行）
- 置信 Confirmed/Likely：
  "顶点时上半身倒向了球的方向——下一杆感觉胸口留在球的后面"
- 置信 Possible/None：
  "此项未发现问题"（无 P1/P2/P3 纠错元素）
- SILENT：
  "画面质量不足，请重新录制：正面站立，完整挥杆，确保全身入镜"（无任何诊断元素）
- 字体：cv2.FONT_HERSHEY_SIMPLEX，scale=0.7，颜色白色，黑色描边 2px
- 位置：图底部 padding=20px

### 3.6 灰度自查版
- 每张彩色 cue 图同时生成灰度版（cv2.cvtColor BGRA→GRAY 后保存为 *_gray.jpg）
- 灰度版文件名：`{stem}_gray.jpg`

---

## 4. 置信门接线

```
Confirmed → complete_cue(P1+P2+P3+文案)
Likely    → complete_cue(P1+P2+P3+文案)
Possible  → neutral_frame(灰骨架+中性文案，无纠错元素)
None      → neutral_frame
SILENT    → retake_guide(纯文字引导图，无诊断元素)
```

---

## 5. 输出路径约定

```
output/cue_renders/reverse_pivot/
  {clip_id}_top_cue.jpg        # 彩色 cue 图
  {clip_id}_top_cue_gray.jpg   # 灰度自查版
```

Windows Desktop 同步路径：
```
C:\Users\jason\Desktop\rtmpose_results\preview\cue_renders\reverse_pivot\
```

---

## 6. 架构约束

- `cue_renderer/` 模块：纯消费者，零测量，零判断，零阈值比较
- 判断引擎侧 (`gate3_no_false_positive.py` 或专用诊断接口)：负责生成 ReversePivotPayload
- 两者之间以 payload dataclass 解耦，渲染器不 import 引擎测量代码

---

*VISUAL_INDICATOR_V1 v1.0 — CUE-001 关卡 B*
