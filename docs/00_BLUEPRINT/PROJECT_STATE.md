# PROJECT_STATE.md — SwingCue 项目当前状态快照

> **维护纪律**: 每个 ⛔ 关卡关闭时必须更新本文件并随关卡 commit push。
> 状态未更新 = 关卡未关闭（见 GT_IRON_RULES.md § CUE-D）。
> 本文件是跨会话/跨人自动接手的唯一状态入口，任何会话开始前先读本文件。

**最后更新**: 2026-07-05 — CUE-002 追加§7 关卡关闭
**最新 commit**: `71d8194` — docs(CUE-002-§7): PROJECT_STATE.md新建 + 铁律CUE-D(关卡强绑定) + design_iterations/预留目录
**分支**: master（已 push）

---

## 1. 当前阶段

**阶段**: STAGE 0 — Foundation（进行中）
**阶段目标**: 建立 face-on 正面 2D 测量 + cue 渲染闭环，单一错误类型（Reverse Pivot）完整跑通
**阶段状态**: 破局点 1/2/3 均已完成；cue 渲染端初版完成；解构器 v0 完成

---

## 2. 当前进行中任务

| 任务 | 状态 | 最后动作 |
|------|------|---------|
| CUE-002 追加§7（PROJECT_STATE+design_iterations） | **完成** | 当前会话 |
| CUE-001 3秒测试终审 | **等待 Jason** | clip_016_left_top_cue.jpg 已输出至 Windows Desktop |
| top_conf 沉默阈值正式定版 | **暂缓** | provisional=0.50，待更多阴性对照数据 |

---

## 3. 已完成里程碑（STAGE 0）

| # | 里程碑 | commit | 日期 |
|---|--------|--------|------|
| 破局点1 | 137 clip 全库 profiler Gate3 验收（137/137, 46.9min） | `44d9d63` | 2026-07-04 |
| 破局点2 | 24对配对候选详表 | `4631ca8` | 2026-07-04 |
| 破局点3 | 22对正面2D可测性筛查确认 spine_lateral_tilt 为主维度 | `bb347f4` | 2026-07-04 |
| 关卡1 | 三线几何测量层 + clip_016 唯一合格配对确认 | `1595cf6` | 2026-07-04 |
| 关卡2 | 参考基准飞轮 baseline_v1（Jason 验收通过） | `9be924a` | 2026-07-04 |
| TOPV3-001 | top 检测 v3，B层单一事实源，fallback bug 修复 | `351d4db` | 2026-07-05 |
| 关卡3 | 9/9 不误报验证 PASS（gate3 v3） | `351d4db` | 2026-07-05 |
| CUE-001 A | CUE_DESIGN_LANGUAGE v0.1 六法则/六原语/三句型 | `9c0a82d` | 2026-07-05 |
| CUE-001 B | VISUAL_INDICATOR_V1 渲染器，9/9 PASS（3秒测试终审待定） | `c38df76` | 2026-07-05 |
| CUE-002 | 19样板入档 + 解构报告v0.2 + 设计语言v0.2 + 纪律备案 + CJK渲染修复 | `286244a` | 2026-07-05 |

---

## 4. 引擎能力现状

### 已实现（face-on）
- A层: RTMPose ONNX 姿态估计，kp_guard 置信过滤
- B层: SwingPhaseEngine 8相位检测，top_conf，fallback prominence 已修复
- C层: triline_geometry.py — shoulder_lateral_tilt @ top（窗口中位数 top±5帧）
- Profiler: 6模块，137/137 clip 全库分类
- Flywheel: baseline_v1（center=-6.8°, lower=-18.8°, upper=+5.0°）
- CueRenderer: reverse_pivot v1（P1绿楔+P2偏差线+P3弧箭头+文案+灰度版，CJK渲染已修复）
- 置信门: Confirmed/Likely→完整cue; Possible/None→中性帧; SILENT→重拍引导

### 能力缺口（已登记，待立项）
- Sway / Hip Slide / Early Extension / Chicken Wing / Hip Rotation 等（见 CUE_DESIGN_LANGUAGE 映射表）
- 转换期启动顺序错误（δ 时序类，需动画生成器）
- DTL 机位诊断（dtl-eet-1 on hold）
- B层 0.82 窗口技术债（§9，下一规格）

---

## 5. 数据资产

| 资产 | 路径 | 说明 |
|------|------|------|
| kp_cache | engine/kp_cache/ | fo-eet-1/2/3, fo-ok-1/2, negatives |
| 全库 profile | output/video_profile_full.json | 137 clips, 313 cards, 246KB |
| 配对详表 | output/pair_candidates_24.md | 24对→1对合格（clip_016） |
| 测量基准 | engine/reference_flywheel/baseline_v1.json | n=1, Jason 验收 |
| GT 真值 | docs/06_GT_LABELS/GT_LABELS.md | fo-eet-1 top=fr185，fr211反面事实 |
| 样板库 | docs/07_CUE_DESIGN/reference_samples/ | 19张，11MB，解构v0.2 |
| cue 渲染输出 | output/cue_renders/reverse_pivot/ | 9张彩色+9张灰度 |

---

## 6. 阻塞项

| 项目 | 阻塞原因 | 等待 |
|------|---------|------|
| CUE-001 3秒测试终审 | clip_016_left_top_cue.jpg 需目视验收 | Jason |
| top_conf 沉默阈值定版 | 需更多阴性对照数据 | Jason 拍板 |
| dtl-eet-1 Step 3+5 | 疑似拼接素材需确认 | Jason |
| NEEDS_HUMAN.md 4项旧项 | 长期搁置 | deferred |
| SAMPLES_INDEX.md VLM补全段复审 | 10张一句话读法须 Jason 目视裁决 | Jason |

---

## 7. 下一步候选（供 Jason 选方向）

- CUE-001 终审关闭（3秒测试）
- B层 0.82 窗口技术债规格立项
- 第二错误类型诊断（Sway/Hip Slide 任选一）
- Cue 生成器规格立项（时序类 δ 句型动画）
- 飞轮扩充：自采同人配对数据录制方案

---

*PROJECT_STATE.md — 由 CUE-002 追加§7 创建 / 2026-07-05*
*维护人: Hermes（每关卡自动更新）/ 裁决人: Jason*
