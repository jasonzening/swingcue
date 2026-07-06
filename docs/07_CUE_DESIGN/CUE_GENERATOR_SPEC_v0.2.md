# CUE_GENERATOR_SPEC v0.2 — Cue 生成器规格

**版本**: v0.3
**日期**: 2026-07-05
**状态**: Jason 裁决修订稿（CUE-003 规格修订）
**任务**: CUE-003

---

## 0. 定位与分层

Cue 生成器（Generator）位于诊断引擎与渲染器之间，职责为：

```
verdict payload  →  [Generator]  →  Cue Plan JSON  →  [Renderer v2 / Lottie]  →  最终动画
```

Generator 的唯一输出物是 **Cue Plan JSON**：一份描述「画什么、怎么动、文案是什么」的声明式规格，不含像素操作代码。Renderer 负责把 Plan 翻译为实际像素 / Lottie 动画。

---

## 1. 学术与工程依据

### 1.1 Tversky 2002 — 箭头即动词

> Tversky, B., Zacks, J., Lee, P.U., & Heiser, J. (2002). Lines, blobs, crosses and arrows: Diagrammatic communication with schematic figures. In *Proceedings of the International Conference on Theory and Application of Diagrams* (pp. 221–230). Springer.

**要点**：箭头在视觉中作动词而非名词；读者将箭头解读为「将要发生的运动方向」，而非「已经发生的轨迹」。箭头形状携带运动语义（弧=旋转，直=平移）。

**引擎应用**：
- P3 方向箭头方向 = 用户下一杆应做的动作方向（不画「你做错了这个方向」）
- P8 箭头字典：直箭头=平移错误或指令，弧箭头=旋转指令（_120/_181/_122 三重互证）
- 校验规则②：现状必须配指令，纯指令箭头禁止单独出现

### 1.2 Ayres 2009 — 动画与分步效应

> Ayres, P., & Sweller, J. (2005). The split-attention effect in multimedia learning environments. In R. E. Mayer (Ed.), *The Cambridge Handbook of Multimedia Learning* (pp. 135–146). Cambridge University Press.
> Ayres, P. (2009). *Animation and Learning*. In S. Tobias & T. M. Duffy (Eds.), Constructivist Instruction. Routledge.

**要点**：
- 动画比静态图在运动概念学习上更有效，但「暂停与重播」控制权给学习者可降低认知负荷
- 时序信息（A先于B）在静态图中承载力不足，动画原生表达无歧义
- 工作记忆容量有限：单次动画中运动元素数量不应超过 1（避免注意分割）

**引擎应用**：
- 法则9 动态优先：指示器默认动画，静态为降级形态
- 动效预算=1：每 cue 仅一个运动元素（指令箭头），现状线与正确形线静止
- 校验规则⑧：动画三镣铐（扫动时长/循环暂停/时序分步）

### 1.4 Signaling Principle 元分析 — 局部高亮与背景压暗的边界

> de Koning, B.B., Tabbers, H.K., Rikers, R.M.J.P., & Paas, F. (2009). Towards a framework for attention cueing in instructional animations: Guidelines for research and design. *Educational Psychology Review*, 21(2), 113–140.
>
> van Gog, T., Paas, F., & van Merriënboer, J.J.G. (2008). Effects of studying sequences of process-oriented and product-oriented worked examples on troubleshooting transfer efficiency. *Learning and Instruction*, 18(3), 211–222.
>
> Richter, J., Scheiter, K., & Eitel, A. (2016). Signaling text-picture relations in multimedia learning: A comprehensive meta-analysis. *Educational Research Review*, 17, 19–36.

**要点**：

- **Signaling Principle（信号原则）**：在多媒体材料中添加视觉信号（箭头、轮廓辉光、区域高亮）指引学习者注意到核心元素，可显著提高学习迁移效果（Richter et al. 2016 元分析，d ≈ 0.40–0.60）。
- **局部辉光 vs. 全局背景压暗的差异**：de Koning 等人区分两类注意引导机制——
  - **局部高亮**（local cueing）：在错误涉事部位叠加边界辉光 / 淡色块，不破坏背景关系，学习者仍能看到身体整体；
  - **全局压暗**（global dimming）：背景整体压暗后，关系性错误（如髋与肩的相对倾斜）失去参照系，读者难以判断"相对于什么"出了问题——这正是高尔夫姿态纠错的主要语义需求。
- **认知过载风险**：背景压暗作为一个独立视觉层（掩码计算 + 合成），本身增加渲染复杂度；若掩码边界失效（抠图错误）则引入噪声，反而干扰注意。
- **结论（Jason 裁决 2026-07-05）**：局部辉光（P10b）作为默认聚焦手段；SAM2 全局背景压暗降级为 A/B 候选，见 §7。

**引擎应用**：
- P10b 部位轮廓辉光：错误涉事部位的局部 SAM2 掩码辉光，一图仅高亮一处
- 法则 7 聚焦手段：移除「SAM2 背景压暗」，改为「P10b 局部辉光」

---

### 1.3 Lottie — 双层架构

> LottieFiles. (2024). *Lottie Animation Format Specification*. https://lottiefiles.github.io/lottie-docs/

**要点**：Lottie 是 JSON 格式的矢量动画规格（After Effects → JSON），支持路径动画、描边动画、遮罩；Web/iOS/Android 均有一致渲染器；动画可在运行时暂停/跳帧/速度控制。

**双层架构设计**：
```
Cue Plan JSON  →  Generator  →  两份产物
    ├── PIL 静态预览图（工程草图，供 Jason 审 Plan 几何）
    └── Lottie JSON（最终动画，供产品渲染）
```

Lottie 层（v2）不在本任务范围内实现，但 Cue Plan JSON 字段设计须与 Lottie 对齐（animation_track 字段即 Lottie shape layer 的参数蓝图）。

---

## 2. 动画三镣铐（Animation Constraints）

所有指令箭头的 animation_track 必须满足：

| 镣铐 | 规格 | 依据 |
|------|------|------|
| 扫动时长 | 1.5s ≤ duration_s ≤ 2.0s；停顿 pause_s = 0.5s | Ayres 2009：认知负荷窗口；实测「看3秒知道做什么」与动画节奏对齐 |
| 循环+可暂停 | loop=true；pauseable=true（Lottie goToAndStop API） | 低频反馈原则（法则5）；用户控制权降低认知负荷 |
| 时序类离散分步 | 句型 δ 时序类：steps 字段非 null，各分步有独立 start_pct/end_pct；连续类 steps=null | _119 实证：时序语义静态不可见，动画须分步显示 |

---

## 3. Cue Plan JSON 顶层结构

见 `CUE_PLAN_SCHEMA.md` 完整定义。摘要：

```
CuePlan
├── 元数据（schema_version, clip_id, fault_id, confidence, timestamp）
├── 整图字段（sentence_type_id, contrast_structure, caption_badge, static_downgrade_note）
├── elements[]（每个原语的七元组）
└── validator_result（8条校验通过/违规列表）
```

---

## 4. 句型模板（已实现）

### 句型 α — 角度类（Reverse Pivot 首例）

**适用**：目标是「把某角度调入正确带」。

**元素组合**：
```
P1 正确区（绿色楔，band_lower→band_upper，锚:hip_mid）    [layer: bg]
P7 正确形线（中心线，band_center 角度，锚:hip_mid）        [layer: bg]
P2 现状线（自发光红线，tilt_deg 角度，锚:hip_mid→sho_mid）[layer: mid, STATIC]
P3+P8 弧箭头（从 tilt_deg 扫向 band_center，锚:sho_mid） [layer: fg, ANIMATED]
```

**文案徽章**：底部单行，外部焦点语言，≤ 30字。

**置信门**：
- Confirmed / Likely → 完整 Plan（全部元素 + 文案）
- Possible / None → NeutralPlan（无纠错元素）
- SILENT → RetakePlan（零诊断元素）

---

## 5. 校验器（Validator）

8条规则，任一不过拒绝出 Plan，输出违规条目列表。

| # | 规则 | 反例样板 |
|---|------|---------|
| ① | 色极性=指令极性：绿=正确/目标，红=错误/现状，白/亮=指令箭头 | _188（原作者正确侧用红） |
| ② | 现状+指令成对：有 current_state 必有 direction_instruction，反之亦然 | _190（纯指令无现状） |
| ③ | 动效预算=1：animation_track 非 null 的元素最多 1 个 | — |
| ④ | 箭头细窄不遮身体：stroke_width_px ≤ 6；箭头路径与人体 bbox 中心区重叠比 ≤ 0.25 | _187（大箭头遮躯干）、_190（箭头遮挡超标） |
| ⑤ | 灰度自明：所有元素须在灰度图中通过位置/形状仍可区分（不依赖颜色作唯一信息） | _188（色彩失误图） |
| ⑥ | P11 不得单独成图：P11 注意圈必须与至少一个其他原语共存 | _117（仅圈无结论） |
| ⑦ | 文字仅命名/裁决徽章：caption_badge.text 不含角度数字、百分比、置信分值 | — |
| ⑧ | 动画三镣铐：有 animation_track 的元素须满足 1.5≤dur≤2.0，pause=0.5，loop=true，pauseable=true；δ句型须有 steps | _119（素人读法失败，时序类不分步） |

详见 `cue_generator/validator.py` 实现。

---

## 6. 范围界定（本任务 CUE-003）

| 项目 | 本任务 | 下一任务 |
|------|--------|---------|
| Cue Plan JSON Schema 与 α 模板 | ✅ | — |
| 8条校验器 | ✅ | — |
| PIL 静态工程预览图 | ✅（供 Jason 审几何） | — |
| Lottie JSON 编译 | ❌ | CUE-004 |
| Renderer v2 | ❌ | CUE-004（等 Jason 审过 Plan） |
| 时序类 δ 句型 | ❌ | 专项①单独立项 |

---

---

## 7. A/B 候选附录 — SAM2 全局背景压暗（降级候选）

**定义**：使用 SAM2 实例分割模型对人体建立掩码，将掩码区域外的背景整体压暗（亮度 ×0.3–0.5），使主体自然突出。

**技术路径**：
```
SAM2 mask → 人体二值掩码 → 背景 = frame * dim_factor
              └→ 前景 = frame（原亮度或略提亮）
```

**降级理由（Jason 裁决 2026-07-05，依据 §1.4 Signaling Principle 元分析）**：

| # | 降级理由 | 说明 |
|---|---------|------|
| 1 | **信号过载风险** | 背景压暗本身是一个视觉事件，与箭头/形线叠加后可能超出单帧信号预算（法则 4 一图一对比）；Ayres 2009 工作记忆容量约束要求同时动的轨道 ≤ 1 |
| 2 | **关系性错误需背景参照** | 高尔夫姿态错误（髋与肩相对倾斜、早伸离臀线）本质是关系性错误；背景压暗切断了「身体与环境/重力」的视觉参照系，读者难以判断「相对于什么偏了」（de Koning 2009 局部 vs. 全局压暗差异） |
| 3 | **掩码失效风险** | SAM2 在运动帧（模糊、遮挡、非标准机位）下掩码边界易失效；失效掩码 = 碎片化背景压暗，引入噪声而非信号（注意干扰而非注意引导） |

**适用条件（若将来 A/B 验收时考虑启用）**：
- 静态 setup 帧（address 帧，无运动模糊）
- DTL 机位背景简洁（无大面积人体肤色背景元素）
- SAM2 掩码经人工目视确认边界清晰
- 同一 cue 图仅用压暗聚焦，不同时使用 P10b 辉光（不叠加两种聚焦机制）

**结论**：默认渲染手段为 P10a 淡色块 + P10b 局部轮廓辉光（见 CUE_DESIGN_LANGUAGE P10a/P10b）。SAM2 背景压暗留作 A/B 测试备选，使用前需 Jason 专项授权。

---

*CUE_GENERATOR_SPEC v0.3 — v0.2 Jason 拍板 2026-07-05 / v0.3 Jason 裁决修订 2026-07-05（SAM2 降级 + Signaling Principle 元分析入库）*
