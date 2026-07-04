# HERMES_RUNBOOK.md — Hermes 24 小时连续运行守则

> 这份是让 Hermes 能"无人值守跑通宵"而不跑偏、不卡死、不烧钱的纪律文件。Hermes 每开始一个任务前先读本文件。核心教训来自 Dell 那晚(被一个难装的东西卡死整夜)和 TAR-ViTPose(卡 mmcv 编译)。

## 1. 总原则

- **只做 TASK_QUEUE.md 里的任务,按顺序,一次一个。** 不自己发明任务,不跳号。
- **每个任务有硬性超时(timebox)。超时立即停该任务、记录卡点、跳下一个。** 绝不在一个任务上无限重试。
- **产出可验收的东西 > 追求完美。** 先出能跑、能看的结果,标注已知问题,继续推进。
- **不可逆/危险操作前必须停下等人确认**(见 §4),其余自主进行。

## 2. 环境(每次干活前确认)

- 工作目录 `~/projects/swingcue-postest`(WSL 原生,**绝不碰 /mnt/c** 跑计算,只在最后复制产出到桌面给人看)
- 先 `source .venv/bin/activate`;确认 `torch.cuda.is_available()` 为 True(GPU=RTX 4060 Ti)
- 新建代码/产出放对应子目录;原始 keypoints json 是资产,**只读不删**

## 3. 任务执行循环(每个任务都走这个流程)

1. 读 TASK_QUEUE.md,取**第一个未完成**任务
2. **开工前先 `git status`**:记录当前 branch / commit 到 PROGRESS.log;**若工作区有未提交改动,写入日志,不覆盖、不清理、不 reset**(多电脑多 agent 开发,最怕覆盖已有成果——这比代码本身还重要)
3. 读该任务引用的 spec 文档(MVP0/API/PhaseDetector/Indicator)
4. 设定 timebox(任务里写明,默认 2 小时)
5. 干活;每 30 分钟在 `~/projects/swingcue-postest/PROGRESS.log` 追加一行状态(时间、任务号、进展/卡点)
6. 完成 → 跑该任务验收标准 → 通过则在 TASK_QUEUE.md 勾选 + 写产出路径 → 进下一个
7. 超时未完成 → 停,在 PROGRESS.log 和 TASK_QUEUE.md 记"BLOCKED:卡在哪、试了什么" → **跳下一个独立任务**
8. 需要人确认的(§4)→ 停,写 `NEEDS_HUMAN.md`,继续做其它不依赖它的任务

## 4. 必须停下等人(不可自主)

- 删除/覆盖任何已有数据或归档
- 任何花钱操作(买 API 额度、付费服务、开通收费云资源)
- 安装需要 sudo 且可能改系统状态的东西(记录建议,等人来装)
- 提交代码到远程仓库、对外发布、动 App Store / 开发者账号相关
- 任何 spec 没覆盖、需要产品判断的决策(不要自己拍板产品方向)
- 单个依赖编译/安装超过 timebox(像 TAR-ViTPose 的 mmcv)→ 跳过记录,不死磕

## 5. 防跑偏红线

- 不做 TASK_QUEUE 之外的"顺手优化"
- 不重构已验收通过的代码(除非任务明确要求)
- 不引入深度学习模型到坎1/坎2(纯规则+几何,主文档已定)
- 不碰 IAP / Marketplace / Coach Me / Swing Score / 杆头检测(超出当前阶段)
- 拿不准就停下问,不猜

## 6. 留痕(无人值守的眼睛)

- `PROGRESS.log`:每 30 分钟一行,通宵可追溯
- `TASK_QUEUE.md`:实时勾选/标 BLOCKED
- `NEEDS_HUMAN.md`:所有需要人决策的事项汇总,人回来一眼看完
- 每完成一个里程碑,把可验收产出(截图/overlay 视频/report)复制到桌面 `rtmpose_results/`,文件名带任务号
- 上下文(ctx)用量过半时,先把状态写全到上述文件,确保换接续也能继续

## 7. 人回来时的交接

Hermes 在每轮结束/卡住时,产出一句话总线状态:"已完成 T1-T3,T4 BLOCKED(原因),T5-T6 待依赖,NEEDS_HUMAN 有 2 项"。人只需读 TASK_QUEUE.md + NEEDS_HUMAN.md 即可全掌握。

## 8. GT 生产流程（Ground Truth 生产铁律）

GT = Ground Truth，是 C/D 层阈值校准的唯一合法依据。本节固定流水线与铁律。

### 8.1 五步流水线

```
① 画线渲染
   按 FAULT_VISUAL_STANDARDS.md 当前版本，对目标视频跑 render_gt_lines.py。
   输出：gt_lines/<vid_id>/ 各子窗口标注帧。
   规范：所有参考线在 address 帧定坐标，固定叠加到窗口内每帧。

② Hermes 出测量报告（只数字，禁标签）
   跑 gt_measurement_report.py，输出每段视频几何测量值与曲线图。
   严禁在报告中出现任何缺陷名称、疑似判断、阈值比较结论。
   输出：gt_measure/ 曲线图 + peak_frames/ 峰值帧标注图 + GT_MEASUREMENT_SUMMARY.md

③ 数字指认峰值帧
   Hermes 将测量峰值（帧号 + 数值）以纯数字形式呈报给人。
   人根据峰值帧号打开对应标注图，用眼睛确认峰值帧的动作形态。

④ 人看帧定性 + 意图确认（Jason 执行）
   人工确认：该帧数值是否对应真实动作？意图错误是什么？
   Hermes 不得代替人完成定性判断。

⑤ 写入 GT_LABELS.md
   由 Hermes 按人工确认结果如实录入 docs/GT_LABELS.md。
   标签格式：视频 × 缺陷类型 × 确认结论（阴性/阳性/待定）+ 测量数据引用。
```

### 8.2 GT 铁律

- **Hermes 永不直接产出缺陷标签**。测量报告只输出数字，定性由人完成。
- **GT 必须经人工确认**。任何未经 Jason 确认的标签不得写入 GT_LABELS.md。
- **禁止用自己的检测值当 GT**（禁止"自造 GT"）。检测值是估算，GT 是人工确认。
- **GT_LABELS.md 写入后不得被引擎代码直接反向调参**。阈值调整需另起任务经人批准。
- **窗口定义以 FAULT_VISUAL_STANDARDS 当前版本为准**，不得用固定帧数代替动态标准。
  例：Chicken Wing 窗口 = [impact, 第一个 wrist_mid_y < hip_mid_y 的帧]，不得固定为 impact+8。

### 8.3 文件路径约定

| 文件 | 路径 | 用途 |
|---|---|---|
| 画线渲染 | Desktop/rtmpose_results/preview/gt_lines/ | 供人肉眼验证参考线位置 |
| 测量曲线 | Desktop/rtmpose_results/preview/gt_measure/ | 供人判读峰值 |
| 峰值标注帧 | Desktop/rtmpose_results/preview/gt_measure/peak_frames/ | 供人定性确认 |
| 修正窗口帧 | Desktop/rtmpose_results/preview/gt_measure/peak_frames_v2/ | 动态窗口修正后的帧 |
| GT 标签档案 | docs/GT_LABELS.md | 唯一合法 GT 来源，版本化管理 |

