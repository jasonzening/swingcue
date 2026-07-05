# 正面三线几何测量层 + 参考基准飞轮 — 设计规格 v0.1

> 归属:Layer 3 高尔夫包(golf-pack)/ 特征层 + 决策层
> 状态:**设计草案,待创始人拍板后交实现**
> 定位:阶段1第一份蓝图级规格。定义正面(face-on)机位下的通用几何测量层,
> 及其"先验基准 + 同人配对累积"的飞轮机制。第一版完整定义三线测量,
> 但只对"髋旋转 vs 平移"出诊断与指示器。

---

## 0. 设计原则(来自战略与纪律)

1. **给初/中级用户,不追专业级精度**:能判断"髋转了还是只平移了"这种量级的
   明显错误即可,现有 2D 关节点精度绰绰有余。
2. **正确基准→偏离即错误**:不逐一硬编码错误,用专业正确样本建基准,偏离即报。
3. **不误报优于全覆盖**:把握不足则沉默(Confidence Ledger)。
4. **一次建通用地基**:三线几何一套关节点,支撑髋旋转/平移/侧倾/肩髋分离多个错误。
5. **同人配对是最高质量数据**:同一人对错配对消除个体差异,信号最纯,权重最高。

---

## 1. 三条基准线(几何测量层核心)

所有线用 RTMPose 关节点构建;所有关节点先过 kp_guard(置信度+有效性),
不足则该帧标 NaN。

| 线 | 定义 | 关节点 |
|---|---|---|
| **髋线 (pelvis line)** | 左右髋关节中心连线 | left_hip, right_hip |
| **肩线 (shoulder line)** | 左右肩关节中心连线 | left_shoulder, right_shoulder |
| **踝线/站位线 (ankle line)** | 左右踝连线(作"未旋转"的静止参照) | left_ankle, right_ankle |

**归一化(必做,GolfMate 论文警告的坑)**:
- 所有坐标先按**肩宽**(两肩点距离)归一化,消除体型/远近差异。
- 角度类特征本身无量纲;位移类特征以肩宽为单位表达(如"0.3 肩宽")。

---

## 2. 三线几何特征(每帧计算,全定义)

### 2.1 髋线特征(第一版重点)
- `pelvis_width_norm`:髋线在画面内的**投影宽度** / 肩宽。
  - 物理含义:髋线越正对镜头(未旋转)→ 宽度大;越旋转 → 投影宽度变小。
- `pelvis_center_x_norm`:髋线中点 x,相对 address 帧的位移 / 肩宽。
  - 物理含义:横向平移量。
- `pelvis_line_angle`:髋线相对水平线的画面内夹角(度)。

### 2.2 肩线特征
- `shoulder_width_norm` / `shoulder_center_x_norm` / `shoulder_line_angle`(同上定义)。
- `shoulder_lateral_tilt`:肩线相对水平的侧倾角(复用 spine_lateral_tilt v0.1 思路,
  用于 reverse pivot / 上半身侧倾,第一版只测量不出指示器)。

### 2.3 踝线特征(参照基准)
- `ankle_width_norm`:作为"站位未动"的参照;若踝线自身在动(重心大幅移动/抬脚),
  作为质量降权信号。

### 2.4 跨线特征
- `shoulder_pelvis_separation`:肩线角 − 髋线角(X-factor 的 2D 近似,
  第一版只测量不出指示器)。

---

## 3. 核心判据:髋旋转 vs 平移(第一诊断目标)

> 依据:业界(DeepSwing)将 hip rotation 与 pelvis slide 列为独立检查项,
> 证明二者可区分。区分的几何本质如下。

**关键区分信号(2D 画面内可测):**

| 动作 | pelvis_width_norm(髋线宽度) | pelvis_center_x_norm(髋中点位移) |
|---|---|---|
| **正确:髋旋转** | 明显变化(投影宽度随转动改变) | 位移相对小 |
| **错误:髋平移(slide)** | 基本不变(没转,只整体滑过去) | 位移大 |

**判据(在下杆窗口 transition→impact 计算):**
- 定义 `rotation_signal` = pelvis_width_norm 从 top 到 impact 的变化量。
- 定义 `slide_signal` = pelvis_center_x_norm 从 top 到 impact 的位移量(朝目标方向)。
- **旋转充分**:rotation_signal 大 且 slide_signal 相对小。
- **平移为主(错误)**:slide_signal 大 且 rotation_signal 小。
- 具体阈值**不硬编码**,由第 4 节的基准 + 偏离度量给出。

**注意(诚实标注局限)**:
- pelvis_width 的绝对值受机位远近影响,故用"从 top 到 impact 的**变化量**"而非绝对值,
  且已肩宽归一化,降低机位敏感度。
- 若 address 机位偏离标准正面(camera_view 置信度低),触发沉默(Confidence Ledger)。

---

## 4. 参考基准飞轮(先验 + 同人配对累积)

### 4.1 三层数据身份与权重

| 层 | 来源 | 权重 |
|---|---|---|
| **先验基准(第0样本)** | FAULT_VISUAL_STANDARDS(TPI/PGA/文献) | 作为地基,提供初始"正确范围"与方向 |
| **同人对错配对(最高质量证据)** | 教学视频专业选手左对右错配对 | **最高权重**(消除个体差异,信号最纯) |
| 散点样本(单独对/单独错) | 将来非配对的专业样本 | 较低权重(带个体差异噪声) |
| 用户自己的杆 | 创始人/未来用户 | 不进基准(仅待测) |

### 4.2 权重累积机制(第一版:均等权重)

- 基准 = 先验 + 已累积配对样本的加权平均。
- **均等权重起步**:n 个配对样本时,每个样本对"配对贡献部分"权重 = 1/n。
  - 1 个样本:该样本主导;2 个:各 50%;10 个:各 10%。
- 先验基准始终作为底线锚点(即使 0 个配对样本也能工作)。
- **质量加权**:第一版不做,列为后续升级(机位标准/置信度高的样本权重更高)。

### 4.3 同人配对的三重贡献(为何配对最珍贵)

每个配对进来,先肩宽归一化,再提取三线几何,然后:
1. **正确半** → 更新"正确基准池"(定义什么是对)。
2. **错误半** → 更新"错误参照"(定义偏离到什么程度算此错)。
3. **差向量(错误半 − 正确半)** → 记录"此错误在三线几何上表现为哪条线怎么变",
   直接指导指示器方向。

### 4.4 偏离度量(诊断)

- 用户杆的 rotation_signal / slide_signal → 与基准比 → 算偏离。
- 偏离方向 + 大小 → 映射到诊断(髋旋转不足 / 平移为主)。
- 确定度分级:Possible / Likely / Confirmed(Likely 需两独立信号)。

---

## 5. 指示器映射(产品精髓,第一版只做髋)

> 不给角度数字,给一眼看懂的视觉提示。参照 GolfMate"可视化差异点"。

- 检出"髋平移为主" → 指示器:在髋部画一个**旋转箭头**(vs 当前的平移方向),
  配简单文案:"你的髋是滑过去的 → 试着转过去"。
- 只在 Confidence Ledger 通过时显示;把握不足则沉默或提示重拍。
- (Cue Effectiveness Loop:记录用户看此 cue 后下一杆 rotation_signal 是否改善。)

---

## 6. 第一步验证方案

1. 在 51 号专业配对(下杆转体正误对比)上:
   - 提取正确半 + 错误半的三线几何。
   - 验证:错误半 slide_signal 大 / rotation_signal 小;正确半反之。
   - 若这一对能干净区分 → 判据成立,建立首个基准点。
2. 从 24 对中找出其余"髋相关"配对,逐对验证并累积进基准。
3. 用创始人自己的杆(待测样本)测"不误报":正常杆不应被误报为髋平移。

---

## 7. 模块归属与边界

- 三线几何测量(纯关节点几何)→ 可下沉 Layer 2 通用(网球/理疗也用线角度)。
- 髋旋转 vs 平移的高尔夫判据 + 指示器 → Layer 3 golf-pack。
- 第一版先建在 golf-pack,通用部分标注"future: 下沉 core"。

---

## 8. 第一版实现状态 (2026-07-05 TOPV3-001 更新) ✅

关卡1/2/3 全部通过，第一诊断闭环成立。top 检测已升级至 v3（B 层单一事实源）。

| 模块 | 文件 | 状态 |
|------|------|------|
| 三线几何测量 | engine/features/triline_geometry.py | ✅ v0.1 |
| top 检测 v3 | gate3_no_false_positive.py (run_b_layer) | ✅ B层单一事实源+conf≥0.50+amp≥0.8SHW |
| B 层 top_conf 修复 | engine/b_phase/swing_phase.py | ✅ fallback prominence bug 修复(DIAG-001) |
| 参考基准飞轮 | engine/reference_flywheel/baseline_v1.json | ✅ n=1, 带[-18.8,+5.0]° |
| 不误报验证 v3 | output/gate3_no_fp/gate3_results_v3.json | ✅ 5项全过，含阴性对照+DTL门控 |

**top 检测版本沿革与废弃原因:**

| 版本 | 规则 | 废弃原因 |
|------|------|---------|
| v1 | zone[15%,65%] 全程最高腕位法 | setup 噪声；最高腕位误认收杆(DIAG-001 fr211) |
| v2 | zone[15%,65%] + amp≥40px 绝对阈值 | zone 截断真实挥杆段(fo-eet-1 top=fr185 在 82% 处，v2 zone 上限 65% 截断)；40px 绝对值不归一化 |
| **v3(当前)** | **B 层 SwingPhaseEngine 单一事实源 + top_conf≥0.50(provisional) + swing_amp≥0.8×SHW** | — |

**top 检测 v3 沉默路径(Confidence Ledger):**
1. camera_view 非 face_on → SILENT (camera_gate) — DTL 机位在此拦截，标注 SKIPPED(camera_gate)
2. top_conf < 0.50 (provisional) → SILENT (phase_detection_low_confidence)
3. 挥杆段 wrist_y 振幅 < 0.8 × SHW(address 帧) → SILENT (amp_gate)
4. 窗口有效帧 < 3 → SILENT (window_too_small)
5. tilt 计算失败 → SILENT (tilt_failed)

**关卡3 v3 验收结果 (2026-07-05):**

| clip | 机位 | top帧 | top_conf | 振幅(SHW) | tilt@top | 诊断 |
|------|------|-------|----------|-----------|----------|------|
| fo-eet-1 | face-on | fr185 ✅(GT±2) | 0.652 | 2.776 | -0.32° | None |
| fo-eet-2 | face-on | fr57 | 0.739 | 2.479 | -4.09° | None |
| fo-eet-3 | face-on | fr46 | 0.682 | 2.691 | -5.15° | None |
| fo-ok-1 | face-on | fr76 | 0.699 | 2.545 | -7.57° | None |
| fo-ok-2 | face-on | fr65 | 0.740 | 2.463 | -8.88° | None |
| dtl-eet-2 | DTL | — | — | — | — | SKIPPED(camera_gate) |
| dtl-eet-3 | DTL | — | — | — | — | SKIPPED(camera_gate) |
| fo-eet-1-neg-setup | face-on | — | 0.197 | 0.095 | — | SILENT(conf_gate) |
| fo-eet-1-neg-truncated | face-on | — | 0.000 | 1.612 | — | SILENT(conf_gate) |
| clip_016/left_ERROR | face-on | — | — | — | +29.1° | **Confirmed** ✅ |
| clip_016/right_OK | face-on | — | — | — | -6.8° | None ✅ |

**阴性对照门控详表 (top_conf 阈值定版依据):**

| 阴性对照 | top_conf | amp(SHW) | 拦截门 | 说明 |
|---------|----------|----------|-------|------|
| fo-eet-1-neg-setup (fr0-149, 纯 setup) | 0.197 | 0.095 | conf_gate | 无挥杆，conf 远低于 0.50 |
| fo-eet-1-neg-truncated (fr0-179, top fr185 不在片内) | 0.000 | 1.612 | conf_gate | 有运动但无完整上杆顶点，conf=0.000 |

已知限制:
  - 基准 n=1，带宽保守 ±12°；累积到 n≥5 后收窄
  - top 检测依赖正面 wrist_y；DTL 机位由 camera_gate 拦截
  - 仅 shoulder_lateral_tilt 出诊断；髋旋转/平移为第二目标

---

## 9. 技术债登记 (TOPV3-001 追加 B，2026-07-05)

与 [15%,65%] zone、40px 绝对阈值同类的已知债务，本次不改 B 层锚点搜索逻辑：

| 债务项 | 位置 | 性质 | 后续方向 |
|--------|------|------|---------|
| **B 层 0.82 百分比截止窗口** | swing_phase.py `top_end_idx = int(n_eff * 0.82)` | 硬编码百分比，fo-eet-1 top 在 81% 处紧贴截止线(margin=+1fr)，带病侥幸 | 改为运动信号锚定（击球事件前），单独立项 |
| **B 层 prominence=30 绝对阈值** | swing_phase.py `find_peaks(-ys_region, prominence=30)` | 绝对像素值，不随体型/机位距离归一化 | 改为相对阈值（如 % of ys_range），随 0.82 窗口问题一并修复 |
| **top_conf 沉默阈值 0.50 为 provisional** | gate3_no_false_positive.py `CONF_THR=0.50` | 阴性对照数据: neg-setup=0.197, neg-truncated=0.000，均远低于 0.50；当前正常视频最低 0.652；安全余量 0.15 | 正式定版待更多阴性对照数据后由 Jason 拍板 |
| **多挥杆段 top 候选不实现** | — | 多个 top 候选时仅取第一个 first_swing_end 内的 top | 规格说明：n_eff 截断已处理多挥杆段边界；多 top 候选需专项设计，本版沉默处理 |

1. 三线定义与髋旋转/平移判据是否认可。
2. 均等权重起步 + FAULT_VISUAL_STANDARDS 作先验 — 已确认。
3. 同人配对最高权重 — 已确认。
4. 第一版只对髋出指示器,其余线只测量 — 已确认。
