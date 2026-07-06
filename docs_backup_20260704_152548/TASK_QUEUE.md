# TASK_QUEUE.md — Hermes 任务队列(按序执行,一次一个)

> 规则见 HERMES_RUNBOOK.md。每个任务:目标、依赖、timebox、产出、验收、状态。完成勾选并填产出路径;超时标 BLOCKED 跳下一个。**不跳号、不自创任务、不做红线项。开工前先 git status(见 RUNBOOK §3.2)。**

格式:`[ ]` 未做 / `[~]` 进行中 / `[x]` 完成 / `[!]` BLOCKED

> 排序原则:**先把侧面跑出第一条端到端可视化产出**(截图+overlay+report),再补测试和数据层,**正面紧随其后**。正面侧面都要做(产品红线),但正面调参不挡住第一条产出。

---

## T1 — 固化 SwingPhaseDetector 侧面逻辑为可复用模块
- [ ] 状态:
- 依赖:无(侧面逻辑已实测通过)
- timebox:2h
- 做:把现有 detect_keyframes 脚本重构为模块 `swing_phase_detector.py`,接口严格按 SWING_PHASE_DETECTOR_SPEC §1
- 产出:`src/swing_phase_detector.py` + 在 test-dwontheline 上重跑确认输出不变(10/31/47/68)
- 验收:输出四帧与已验收结果一致;impact 锚点逻辑保留;顺序/高度/节奏校验生效

## T2 — spine_angle 指示器模块
- [ ] 状态:
- 依赖:T1
- timebox:2.5h
- 做:按 INDICATOR_ENGINEERING_SPEC §2 实现 `indicators/spine_angle.py`,**严格用 §2 固定坐标系公式**(y 向下、vertical=(0,1)、arccos);输出统一 Schema
- 产出:`src/indicators/spine_angle.py` + test-dwontheline 的 indicators.json
- 验收:符合统一 Schema;low confidence 时 status=null 不给红绿;生理极限测试(0-70°)生效;命名用 Posture Line 不叫"真实脊柱角"

## T3 — MVP-0 端到端 CLI 管线(第一条可视化产出,最重要)
- [ ] 状态:
- 依赖:T1, T2
- timebox:2.5h
- 做:按 MVP0_EXECUTION_PLAN §2 串 `analyze.py`:视频→RTMPose→PhaseDetector→spine_angle→四帧截图+脊柱线 overlay 视频+report.md
- 产出:`analyze.py` + test-dwontheline 完整 output/ 目录(复制桌面 rtmpose_results/,带 T3)
- 验收:MVP0_EXECUTION_PLAN §5 全部 7 条;overlay 脊柱线贴身不漂
- ⚠️ 这是人最想先看到的——完成即在 PROGRESS.log 记一笔

## T4 — spine_angle overlay 渲染验证
- [ ] 状态:
- 依赖:T3
- timebox:1h
- 做:检查 overlay 里黄实线(当前脊柱)+绿虚线幽灵(address 脊柱)渲染正确、逐帧同步
- 产出:overlay.mp4 复制桌面(带 T4)
- 验收:线贴身、随帧同步、delta 超阈值变色正确;low confidence 帧不画红绿

## T5 — SwingPhaseDetector + spine_angle 单元测试
- [ ] 状态:
- 依赖:T1, T2
- timebox:1.5h
- 做:用归档 keypoints json 写 pytest:四帧顺序、阈值边界、生理极限(超人类极限报错)、缺帧>20% 返回 failed
- 产出:`tests/` + 运行通过截图
- 验收:用例全过;故意喂坏数据正确触发 low/failed

## T6 — 数据层骨架(对齐 API/DB spec)
- [ ] 状态:
- 依赖:T3
- timebox:1.5h
- 做:按 API_AND_DATABASE_SPEC §2 建 SQLite 表(含 region、consent、coach_annotations 预留),写读写函数把 CLI 产出落库;**落库时自动带 region 和 data_training_consent 字段**(API spec §3)
- 产出:`src/db.py` + schema.sql + 一条 test-dwontheline 记录入库
- 验收:表结构与 spec 一致;keypoints 永久保留(受 consent 约束);low confidence 记录 status=null

## T7 — 正面(face-on)关键帧参数组(侧面跑通后做,产品需要双角度)
- [ ] 状态:
- 依赖:T1, T3(侧面端到端先通)
- timebox:2h
- 做:在 test-faceon 上调通四帧检测,记录正面参数(params_version="faceon_v1")与侧面差异;生成正面 verify 定格视频
- 产出:正面 verify.mp4(复制桌面,带 T7)+ 参数差异记录进 SWING_PHASE_DETECTOR_SPEC
- 验收:Hard pass(impact 在击球窗口、不跑到下杆中段/收杆);不确定输出 low 不硬给
- ⚠️ **正面透视变形比侧面大,impact 大概率没侧面准。** 若 impact 偏移明显:Hermes 可自动调参(如 Savitzky-Golay 窗口长度)重跑一轮;仍不稳则在 NEEDS_HUMAN.md 记录"正面 impact 偏移情况"等人核对,**不要硬调到超过 timebox**。出第一个正面 verify 视频即在 PROGRESS.log 记一笔等人看

## T8 — RECORDING_GUIDE + 测试集目录结构
- [ ] 状态:
- 依赖:无
- timebox:0.5h
- 做:input/ 下建 test_set/(faceon/ dtl/);若 docs 里 RECORDING_GUIDE.md 已存在则补充完善,否则按主文档 4.6 + 本队列附注创建
- 产出:目录结构 + RECORDING_GUIDE.md
- 验收:人按指南能补拍
- 写入 NEEDS_HUMAN.md:提醒人**补拍正侧各 5 段**以完成 5 段验收(实际补拍是人的任务)

---

## 执行结束条件
T1-T8 全部 [x] 或 [!];总结写入 PROGRESS.log 末尾;NEEDS_HUMAN.md 汇总待人事项(预期含:正面 impact 偏移核对、补拍测试集、后端选型确认、是否进入 MVP-1 App 套壳)。
