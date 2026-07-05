# 阶段 0 · 地基收敛 — 进展日志

> 状态:**进行中**
> 目标:摸清家底 + 把已有单点收编成清晰模块 + 保命备份,不新增野心。
> 本文件随工作持续更新。

## 阶段目标

用高尔夫把地基打扎实,为"第一个完整闭环"(阶段1)做好准备。不做多行业、不预先抽象通用接口。

## 破局点进展

### 破局点 1 · 终结单机裸奔 + 代码资产隔离防线

| 事项 | 状态 | 日期 | 备注 |
|---|---|---|---|
| .gitignore 排除大文件 | ✅ 完成 | 2026-07-04 | 提交体积从 4.1G→1.1M |
| 清理 Zone.Identifier 垃圾 | ✅ 完成 | 2026-07-04 | |
| 从 git 历史抹除 148MB SAM2 权重 | ✅ 完成 | 2026-07-04 | filter-branch 重写 30 commits |
| 推送 GitHub 私有仓库(异地备份) | ✅ 完成 | 2026-07-04 | github.com/jasonzening/swingcue |
| models/ 单独备份到网盘/NAS | ⬜ 待办 | | 模型不进 git,需另存 |
| 两仓库物理隔离(core / golf-pack) | ⬜ 待办 | | 作为重构任务,从容做 |

### 破局点 2 · 盘活 137 段教学视频 = 建 Video Profiler

| 事项 | 状态 | 日期 | 备注 |
|---|---|---|---|
| Video Profiler 独立模块 | ✅ 完成 | 2026-07-04 | engine/profiler/ 6个子模块 |
| 机位判定几何交叉复核(非VLM精判) | ✅ 完成 | 2026-07-04 | sh_lat_ratio 3证据 Gate1 100%准确 |
| 137 段全库体检表 | ✅ 完成 | 2026-07-04 | output/video_profile_full.json (313张身份证) |
| Gate1 机位验证 (10clip) | ✅ 完成 | 2026-07-04 | 10/10=100% |
| Gate2 完整身份证 (11clip) | ✅ 完成 | 2026-07-04 | 11/11 全字段正确 |
| Gate3 全库体检 (137seg) | ✅ 完成 | 2026-07-04 | 137/137 exit 0，Jason人工验收通过 |

### 破局点 3 · 第一诊断目标的"收敛选举"

| 事项 | 状态 | 日期 | 备注 |
|---|---|---|---|
| 由体检表统计"正面+专业对错配对"最多的错误 | ✅ 完成 | 2026-07-04 | 24对 face_on+full_swing 双半屏配对已识别 |
| 24对配对详表产出 | ✅ 完成 | 2026-07-04 | output/pair_candidates_24.md |
| clip_129/130/131 机位几何复核 | ✅ 完成 | 2026-07-04 | dlt-6确认face_on；dtl-1/dtl-2几何偏DTL，待人工复核 |
| 预判:Sway/Slide→spine_tilt;Head→head_ref v2 | 📌 待验证 | | 22+配对目视确认后由数据选定 |

## 其余阶段0事项

| 事项 | 状态 | 备注 |
|---|---|---|
| 建立文档集(本档案库) | ✅ 完成 | 2026-07-04 |
| 代码按 Layer 0/1/2/3 重构 | ⬜ 待办 | 见资产搬迁映射表 |
| 3D-Ready Data Contract schema 骨架 | ⬜ 待办 | 字段预留,第一版可空 |
| 数据按三层身份归类归档 | ⬜ 待办 | |
| (建议)试用 GOATY 作 cue 设计参照 | ⬜ 待办 | |

## 阶段0 退出标准(何时算完成)

- [x] 异地备份建立,单机风险解除。
- [x] 文档集建立。
- [x] Video Profiler 建成,137 段体检表产出。(Gate1/2/3 全部通过，Jason 验收 2026-07-04)
- [ ] 第一诊断目标由数据选定。(22+ pair 目视确认后锁定)
- [ ] 代码三层边界清晰。

以上全部完成 → 进入阶段 1。

## 时间线记录

- **2026-07-04**:蓝图定稿 v2.2;异地备份完成(GitHub 私库);文档集建立。
- **2026-07-04**:破局点2完成 — Video Profiler v0.1 Gate1/2/3;137段全库体检表;313张身份证。
- **2026-07-04**:破局点3阶段完成 — 24对 face_on+full_swing 配对识别;几何复核 clip_129(确认)/clip_130/131(待复核);pair_candidates_24.md 产出。
- **2026-07-04**:关卡1+2 完成 — 三线几何测量层 v0.1 建立;参考基准飞轮 baseline_v1 写入;合格配对筛查终审。
- **2026-07-04**:关卡3 完成 — 不误报验证 PASS (v2 top检测: zone[15%,65%]+amp≥40px); 第一诊断闭环成立。
- **2026-07-04**:DIAG-001 — fo-eet-1 感知链路诊断。关卡3 v2 裁决降级为有条件通过(见下)。

## 更新 2026-07-04:第一诊断目标已选定
- 破局点3收敛选举完成。22对配对可测性筛查:主导差异全部为 spine_lateral_tilt。
- **第一诊断目标 = 上半身侧倾 / reverse pivot(spine_lateral_tilt @ top)**。
- 依据:强信号5对(clip_129/016/035/039/041,tilt差13-39°),信号最强最普遍,
  且 spine_lateral_tilt v0.1 已能测,离闭环最近。由数据定,非主观。
- 髋旋转vs平移降为第二目标;三线几何测量层照建(一次支撑tilt与髋)。
- 阶段0退出标准"第一诊断目标由数据选定" ✅

## 更新 2026-07-04:分屏教学视频合格率发现 ⚠️ 重要
- 全量人工筛查结果: 24对候选 → 仅 clip_016 通过同人+正面+真top+讲侧倾四重判据 = **合格率 4%**
- 剔除原因统计:
  - 非同人(不同球员对比): clip_035(120杆vs72杆) / clip_041(Beginner vs Pro)
  - 错误机位(非face-on): clip_039(背面) / clip_129(讲手腕/杆头, 非侧倾)
  - top帧错误(收杆非顶点): clip_128
  - 内容不符(推杆非全挥杆): clip_126
  - 信号弱+噪声: clip_099 / clip_124
- **结论**: 教练分屏视频质量参差，无法依赖作为基准飞轮主力素材。
- **飞轮主力路线修正**: 飞轮主力素材将来需靠**自采同人配对** (同一人在镜头前分别做对/错各一次);
  教练分屏视频仅作补充参考，不作主力数据来源。
- **当前状态**: baseline_v1 以 clip_016 单对起步，权重100%，飞轮设计允许。

## 第一诊断闭环 (reverse pivot, face-on) v1 成立 — 2026-07-04 ✅

**关卡1** — 三线几何测量层 v0.1:
  engine/features/triline_geometry.py, kp_guard≥0.30, 肩宽归一化, 窗口中位数top±5帧

**关卡2** — 参考基准飞轮 baseline_v1:
  先验基准(TPI §4) + clip_016配对(正确-6.8°/错误+29.1°)
  正确带: [-18.8°, +5.0°]  置信账本: None/Possible/Likely/Confirmed

**关卡3** — 不误报验证 PASS:
  top检测v2: zone[15%,65%] + zone内wrist振幅≥40px
  结果: 5/5正常杆=None/SILENT(无误报), 阳性对照=Confirmed
  fo-eet-1 正确沉默(amp=10.6px < 40px, 无真实上杆峰)

**API**:
  from engine.reference_flywheel import query_tilt
  r = query_tilt(tilt_deg=12.5)  # → {"confidence": "Likely", ...}

**已知限制**:
  - 基准 n=1, 带宽保守(±12°); n≥5后收窄
  - top检测依赖正面wrist_y最低点, 侧面(DTL)视频不适用
  - 仅对 shoulder_lateral_tilt 出诊断; 髋旋转/平移为下一目标

## 关卡3 v2 裁决降级 — DIAG-001 (2026-07-04)

**原裁决**: PASS (fo-eet-1 SILENT = 正确行为)
**降级为**: 有条件通过 — fo-eet-1 的 SILENT 是**假沉默** (false silence)

原因: fo-eet-1 经 Jason GT 确认为完整上杆击球动作，但 find_top_v2 的 zone[15%,65%]
将整个挥杆段 (fr165~fr228, 占 clip 后 35%) 截断在外，误判为"无真实上杆"。

诊断结论 (DIAG-001, 待 Jason 视觉裁决确认):
  - RTMPose 感知层正常 (腕分数 0.63~0.93, 0帧丢失)
  - 全程腕振幅 311px (是 fo-eet-2 的 1.12x)，运动确实存在
  - 分叉点: find_top_v2 区间假设, 非感知层失效
  - B-layer gate1 已在 fr185 检测到 top (top_conf=0.0 = 低置信信号)

---

## TOPV3-001 完成记录 — 2026-07-05 ✅

**任务**: top 检测 v3 — 回归 B 层单一事实源

**关卡3 裁决更新**: 从"有条件通过"升级为 **PASS (v3)**

### 各步完成状态

| 步骤 | 内容 | 状态 | commit |
|------|------|------|--------|
| 第0步 | GT 登记: fo-eet-1 top GT=fr185, fr211反面事实 | ✅ | 58d4c4c |
| 第1步 | B 层 top_conf fallback bug 修复 (DIAG-001) | ✅ | 061dde3 |
| 第1步追加A | 6段视频 top_conf 分布 + 82%截止线边距报告 | ✅ | — (报告在 DIAG-001 链路) |
| 第1步追加B | 技术债登记 (FACE_ON_TRILINE_GEOMETRY_SPEC §9) | ✅ | 本批 |
| 第2步 | gate3 v3: B层单一事实源, 删除独立top检测器 | ✅ | 本批 |
| 第2步 | 阴性对照素材裁剪 + kp_cache (150帧+180帧) | ✅ | 本批 |
| 第3步 | 全量重验 5项验收标准全 PASS | ✅ | 本批 |

### 关卡3 v3 验收: 5项全过

1. fo-eet-1 top=fr185 ✅ (GT±2), 诊断 None ✅
2. fo-eet-2/3, fo-ok-1/2: top 合理, 诊断全 None ✅
3. clip_016: left=Confirmed ✅, right=None ✅
4. 阴性对照: fo-eet-1-neg-setup → SILENT(conf_gate, conf=0.197); fo-eet-1-neg-truncated → SILENT(conf_gate, conf=0.000) ✅
5. DTL 门控: dtl-eet-2/3 全部 SKIPPED(camera_gate) ✅

### 沉默阈值状态
- top_conf 沉默阈值 = **0.50 (provisional)**
- 阴性对照余量: 正常视频最低 0.652, 阴性最高 0.197, 安全余量 0.452
- 正式定版: 待更多阴性对照数据后由 Jason 拍板

---

## CUE-002 完成记录 — 2026-07-05

**任务**: Cue Intelligence Engine 知识资产入档（纯文档/资产任务，无渲染代码）

**里程碑**: 解构器 v0（人工模式）完成 19 样板全量解构并经 Jason 全量裁决；Cue Intelligence Engine 正式立项

### 关卡A — 样板库入档
- 19 张样板从 Windows 原始目录拷贝至 `docs/07_CUE_DESIGN/reference_samples/`（文件名原封保留）
- 总大小 11MB，直接入 git（最大单张 1.3MB，全部 < GitHub 100MB 限制）
- 生成 `SAMPLES_INDEX.md`：19 行，每行文件名/缩略图/针对错误/一句话读法
- 读法来源：解构报告 v0.2（Jason 裁决段）+ VLM 辅助补全（_117/_125~128/_184~187/_189）
- VLM 补全段标注「须经 Jason 后续复审」（铁律 CUE-B）

### 关卡B — 解构报告入档确认
- `INDICATOR_SAMPLE_DECODE_REPORT_v0.2.md` 确认存在于 `docs/07_CUE_DESIGN/`，纳入 git
- 内容完整（§0 勘误/§1 修正条目/§2 IDL Schema/§3 法则/§4 对照结构/§5 评估器/§6 生成器题/§7 纪律）
- 文件不做任何改写（Jason 全量裁决终版基线）

### 关卡C — CUE_DESIGN_LANGUAGE.md v0.2
- 原语表: P6→P12（P7正确形线/P8运动语义箭头字典/P9关节角度弧/P10部位淡色块/P11区域注意圈/P12形状词）
- 法则: 6条→9条（法则7骨架可见性/法则8身体可见性/法则9动态优先+动效预算=1）
- 错误分类学: 新增第4类「时序类」（δ句型，动画原生，_119 静态承载力不足实证）
- 对照结构三形态明确：双人并排（仅教学）/ 单主体融合（产品主句型）/ 顺序对照（Next-Swing Loop 原生）
- 双轨验收制度：专家轨（Jason）+ 素人轨（3秒测试）同时通过
- 映射表新增第14行: 转换期启动顺序错误（δ时序类）
- 全部修订处标注「源:样板解构 v0.2 / Jason 裁决 2026-07-05」

### 关卡D — 纪律备案（GT_IRON_RULES.md 追加）
- 铁律 CUE-A: VLM 解构样板可用 / 用户动作诊断仍禁
- 铁律 CUE-B: 样板读法真值权归 Jason
- 铁律 CUE-C: 样板引用文件名强绑定、禁相对编号（附编号错位事故为案例）

### 关卡E — CUE-001 遗留工程债
- 安装 Noto Sans SC 字体至 `~/.local/share/fonts/`（源: `/mnt/c/Windows/Fonts/NotoSansSC-VF.ttf`）
- 验证 Pillow+NotoSansSC CJK 渲染: 9711 非零像素（Python 测试通过）
- 更新 `cue_renderer/reverse_pivot.py`: `_put_text_outlined` + `_draw_caption` 改用 Pillow/NotoSansSC 路径，cv2 仅作无字体时降级
- 实际渲染验证: `clip_016_left_top_cue.jpg` 底栏 2079 白色像素（CJK 字符正确写入）
- 9/9 渲染验收重跑 PASS
- SILENT 引导图与 v1 渲染代码冻结，不返工（下一版由生成器输出规格驱动）

---

## CUE-001 完成记录 — 2026-07-05

**任务**: Cue 设计语言 v0.1 落档 + 第一个视觉指示器(reverse pivot)实现

**里程碑**: Next-Swing Loop 的 cue 端首次成形

### 关卡A — CUE_DESIGN_LANGUAGE v0.1
- 六条设计法则(前注意承载/箭头即指令/禁缺失式/一图一对比/外部焦点/色盲冗余)
- 六原语词汇表(P1正确区/P2现状线/P3方向箭头/P4关节圈/P5栅栏线/P6幽灵v1不实现)
- 三句型(α角度类/β越线类/γ形变类)
- 错误×指示器映射表13行
- 验收方法论: 3秒测试，裁决人Jason
- 文档地位: 与FAULT_VISUAL_STANDARDS同级先验基准，设计先行纪律延伸到cue层
- commit: 9c0a82d

### 关卡B — VISUAL_INDICATOR_V1 实现
- `cue_renderer/` 纯消费者模块: payload.py + reverse_pivot.py
- `run_cue_renders.py` 验收运行脚本
- 渲染: P1绿色楔(正确带) + P2偏差线(深红/橙渐变) + P3弧箭头 + 底部文案
- 置信门: Confirmed/Likely→完整cue; Possible/None→中性帧; SILENT→重拍引导
- 灰度自查版随彩色版同时输出

### 验收: 9/9 全通过 PASS (等待3秒测试终审)

| clip | 置信 | 预期 | 结果 |
|------|------|------|------|
| fo-eet-1 | None | None | ✅ |
| fo-eet-2 | None | None | ✅ |
| fo-eet-3 | None | None | ✅ |
| fo-ok-1 | None | None | ✅ |
| fo-ok-2 | None | None | ✅ |
| fo-eet-1-neg-setup | SILENT | SILENT | ✅ |
| fo-eet-1-neg-truncated | SILENT | SILENT | ✅ |
| clip_016/left | Confirmed | Confirmed | ✅ |
| clip_016/right | None | None | ✅ |

**输出位置**: `C:\Users\jason\Desktop\rtmpose_results\preview\cue_renders\reverse_pivot\`
**等待**: Jason 3秒测试终审 (clip_016_left_top_cue.jpg)
