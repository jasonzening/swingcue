# CUE_GENERATOR_SPEC v0.2 — Cue 生成器规格

**版本**: v0.5
**日期**: 2026-07-05
**状态**: Jason 裁决修订稿（CUE-004 关卡C 技术债清理）
**任务**: CUE-004

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
- 工作记忆容量有限：同时变化的信息维度不应超过可整合数量

**引擎应用**：
- 法则9 动态优先：指示器默认动画，静态为降级形态
- 校验规则⑧：动画三镣铐（扫动时长/循环暂停/时序分步）

> **⚠ 归因修正（CUE-004 关卡C，Jason 裁决 2026-07-05）**：
> 「动效预算=1（同时动的轨道只能一条）」的约束**不来自** Ayres 2009。
> Ayres 研究对象是「分步呈现效应」，未对单帧运动元素数量上限做出约定。
> **正确出处**：
> - **法则4「一图一对比」**（CUE_DESIGN_LANGUAGE）：每张 cue 图只承载一个纠错动作
> - **法则10「极简至上」**（CUE_DESIGN_LANGUAGE v0.4）：元素预算 ≤ 2
> - **前注意属性知觉约束**：Treisman（1988, Feature Integration Theory）证明「运动」是前注意属性，但多个同时运动元素会产生注意竞争，无法并行聚焦——此为动效预算=1 的真实学术依据。
> Ayres 2009 仍作为「动画优于静态」和「用户可暂停控制」的依据，保留引用。

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

### 句型 α — 角度类（Reverse Pivot 首例）— v0.4 改版

**适用**：目标是「把某角度调入正确带」。

**basic 层（默认，校验⑨ ≤ 2元素）**：
```
P2 现状线（红色，自发光，静止，锚:hip_mid→sho_mid）      [layer: mid, STATIC]
P3 弧箭头（白色，起点=红线端点，扫向band_center方向）     [layer: fg, ANIMATED]
```

**v0.4 变更（Jason 裁决 2026-07-05）**：
- P1 绿楔从 α 句型中**整体删除**（信号过载 + 违反法则3「禁用缺失式cue」）
- P7 绿色正确形线**降为 intermediate 层**，basic 默认 `enabled=false`，须 Jason 专项授权开启
- basic 层 Plan 仅 2 个元素（P2 + P3），触发校验⑨自动验证

**intermediate 层（非默认，需授权）**：
```
P2 现状线（同 basic）                                     [layer: mid, STATIC]
P7 正确形线（绿色 dashed，band_center 角度，锚:hip_mid）   [layer: bg, STATIC]
P3 弧箭头（同 basic）                                     [layer: fg, ANIMATED]
```

**文案徽章**：底部单行，外部焦点语言，≤ 30字。

**置信门**：
- Confirmed / Likely → 完整 Plan（basic 元素 + 文案）
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
| ④ | 箭头细窄不遮身体：stroke_width_px ≤ k_max × SHW_canvas；k_max=**0.06**（Jason 拍板 2026-07-05，address 帧肩宽为唯一定义，与运动量门 0.8×SHW 同源）；_shw_canvas_px 须由调用方注入，否则回退绝对阈值 ≤ 15px；箭头路径与人体 bbox 中心区重叠比 ≤ 0.25 | _187（大箭头遮躯干）、_190（箭头遮挡超标）；clip_016_left 实测 SHW_canvas=254.4px → max_sw=15.3px，当前 sw=3px 安全 |
| ⑤ | 灰度自明：所有元素须在灰度图中通过位置/形状仍可区分（不依赖颜色作唯一信息） | _188（色彩失误图） |
| ⑥ | P11 不得单独成图：P11 注意圈必须与至少一个其他原语共存 | _117（仅圈无结论） |
| ⑦ | 文字仅命名/裁决徽章：caption_badge.text 不含角度数字、百分比、置信分值 | — |
| ⑧ | 动画三镣铐：有 animation_track 的元素须满足 1.5≤dur≤2.0，pause=0.5，loop=true，pauseable=true；δ句型须有 steps | _119（素人读法失败，时序类不分步） |

| ⑨ | 元素预算：basic 层 Plan 中 P1–P12 原语元素数 > 2（caption_badge 除外）直接拒绝 | — （法则10 极简至上，Jason 裁决 2026-07-05） |

详见 `cue_generator/validator.py` 实现。

---

## 5.1 Schema 新增字段 — fault_view

`fault_view` 字段声明 cue 视图层级，控制元素预算上限与渲染模式。

| 值 | 说明 | 元素预算 | 校验⑨ | 本任务 |
|----|------|---------|--------|--------|
| `single` | 单一问题，默认路径 | ≤ 2 原语 | 启用 | ✅ 唯一实现 |
| `linked_pair` | 两个运动学因果耦合问题；须声明 `coupling_reason` 字段描述因果关系（如 `"臀部侧滑导致肩面代偿过平"`） | 每问题 ≤ 2 原语 | 启用 | ❌ 待立项 |
| `full_body` | 全身多问题叠加，进阶视图 | 无硬性上限 | 豁免 | ❌ 远期 |

**默认值**: `single`。本任务 CUE-003 只实现 `single`，其余两个路径代码层面暂不实现。

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

## 8. MOCK 模式验收边界（CUE-004 修正① 备案）

### 8.1 MOCK 的用途与限制

| 维度 | MOCK 模式 | 正式验收 |
|------|-----------|---------|
| **目的** | QA 渲染管线几何与动画流程 | 语义 3 秒测试（专家+素人） |
| **画布来源** | 任意已标定的标准帧（如 fo-ok-1 fr76） | 同一球员真实阳性素材（自拍错误示范） |
| **verdict 数值** | clip_016 实测值（tilt=29.1°）注入 | 该球员本次挥杆真实诊断结果 |
| **承担语义测试** | **否** — 画布人物与 tilt 数值不对应 | **是** — 数值即该帧真实偏差 |
| **角度一致性校验⑩** | 必须通过（JSON coords 与 tilt_deg 误差 <1°） | 必须通过 |

### 8.2 校验器规则⑩ — 角度一致性（CUE-004 修正①）

```
规则⑩: P2 shape_params.tilt_deg 须与 anchor coords 反算角度一致，容差 ±1°
反算公式: tilt_actual = atan2(tip_x - hip_x, -(tip_y - hip_y))   [deg]
目的: 防止 MOCK 填数与坐标脱钩，保证 Lottie 动画角度忠实表达诊断
实现: validator._rule10_angle_coord_consistency
```

### 8.3 P2/P3 几何规范（v0.6，CUE-005 专家测试失败修正）

**clip_016 退役令（CUE-005）**: clip_016 仅限判断验证，禁止用于 cue 渲染/预览/验收。
clip_016 成品已移入 output/cue004/_retired/ 留证。正式 MOCK 画布改用 fo-ok-1。

```
P2 = hip_mid → sho_mid 实测两点截断线段（禁延长）
  coords_px           = hip_mid  (RTMPose 实测)
  secondary_coords_px = sho_mid  (RTMPose 实测)
  line_len_px         = |sho_mid - hip_mid| (不含延长)
  joint_dots = True   → 两端渲染白圈关节点（外白圈+内红芯，_122 语法）
  joint_dot_r = 8     → 外圈半径 px

P3 arc:
  center  = hip_mid                       (P2 基点)
  radius  = 0.6 × line_len_px             (较短，视觉紧凑，不遮身体)
  angle_from = tilt_deg                   (弧起于 P2 侧倾方向)
  angle_to   = band_center_deg            (扫向正确带中心 -6.8°)
```

### 8.4 校验器规则⑪ — 锚点在人体 bbox 内（合并单③）

```
规则⑪: P2/P3 anchor.coords_px 须在 _body_bbox_px [x1,y1,x2,y2] 范围内，±10% 容差
_body_bbox_px 由调用方注入（来自 RTMPose 检测器输出，canvas 坐标）。
若 _body_bbox_px 未注入则豁免（MOCK/placeholder 路径）。
目的: 阻断锚点落在草地/背景的情形（v0.2 旧版比例估算缺陷的机器防线）
实现: validator._rule11_anchor_in_body_bbox
```

### 8.5 校验器规则⑫ — P2 端点与 payload 关节坐标重合（CUE-005 新增）

```
规则⑫: v0.6 几何（joint_dots=True）时，
  P2 coords_px ≈ payload hip_mid (±5px)
  P2 secondary_coords_px ≈ payload sho_mid (±5px)
payload 关节坐标通过 _payload_joints 字段注入（可选）；
若未注入则豁免（GT 旁路 clip 补标定后方可生产）。
目的: 防止端点从 tilt_deg 反算（坐标不得从结论反算，纪律§8）
实现: validator._rule12_p2_endpoints_match_payload
```

---

*CUE_GENERATOR_SPEC v0.8 — v0.2 Jason 拍板 2026-07-05 / v0.3 SAM2降级 + Signaling Principle / v0.4 α句型basic层改版 + 校验⑨ + fault_view / v0.5 CUE-004关卡C 动效预算归因修正(Ayres→法则4+Treisman 1988) + 校验④归一化标注(k=0.15待拍板) / v0.6 CUE-004修正①②③: MOCK语义定死 + P2/P3几何重定义 + 校验⑩ + 溢出防护 / v0.7 合并单: 校验④ k_max=0.06 定版(address SHW唯一定义) + 校验⑪新增(锚点bbox) + truncated文案修正 / v0.8 CUE-005: P2/P3几何v0.6(截断线段+关节点+0.6×radius) + 校验⑫(端点重合) + clip_016退役令 + MOCK画布改fo-ok-1*
