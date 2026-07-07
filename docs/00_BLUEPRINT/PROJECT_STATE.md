# PROJECT_STATE.md — SwingCue 项目当前状态快照

> **维护纪律**: 每个 ⛔ 关卡关闭时必须更新本文件并随关卡 commit push。
> 状态未更新 = 关卡未关闭（见 GT_IRON_RULES.md § CUE-D）。
> 本文件是跨会话/跨人自动接手的唯一状态入口，任何会话开始前先读本文件。

**最后更新**: 2026-07-07 — GHOST-003 T2.1 完成，⛔ 等 Jason 目视最差有效帧裁决
**最新 commit**: `cae9ddc` — feat(GHOST-003-T2.1): 崩帧哨兵+2D joint opt
**分支**: master（已 push）

---

## 1. 当前阶段

**阶段**: STAGE 0 — Foundation（进行中）
**阶段目标**: 建立 face-on 正面 2D 测量 + cue 渲染闭环，单一错误类型（Reverse Pivot）完整跑通
**阶段状态**: 破局点 1/2/3 均已完成；cue 渲染端初版完成；解构器 v0（人工模式）19 样板全量裁决完成，v0.3 自包含基线入档

---

## 2. 当前进行中任务

| 任务 | 状态 | 说明 |
|------|------|------|
| CUE-005 正式双轨验收 | **等待 Jason 目视** | fo-ok-1_MOCK_static_last.jpg + retake_silhouette_v2_*.jpg 等三张成品，Jason 3秒目视后裁决 |
| CUE-001 3秒测试终审 | **等待 Jason** | clip_016_left_top_cue.jpg 在 Windows Desktop（已冻结） |
| CUE-004/005 正式双轨验收 | **等待 Jason 自拍素材** | Jason 提供真实阳性素材后在真实画布执行专家+素人3秒测试 |
| 样板20入库 | **等待 Jason 提供图片** | Jason 提供红色发光人体轮廓参考图后，copy 入 reference_samples/sample_020_silhouette_fence_ref.jpg |
| 下一步方向选择 | **等待 Jason** | 见 §7 候选项 |

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
| CUE-001 A | CUE_DESIGN_LANGUAGE v0.1 六法则/六原语/三句型 | `9c0a82d` | 2026-07-05 |
| CUE-001 B | VISUAL_INDICATOR_V1 渲染器，9/9 PASS（3秒测试终审待定） | `c38df76` | 2026-07-05 |
| CUE-002 | 解构报告 v0.3 自包含基线入档，SAMPLES_INDEX v1.2，铁律 CUE-A/B/C/D/E | `92cddb1` | 2026-07-05 |
| CUE-003 | Cue 生成器 v0.1 — Plan Schema + 校验器 11 条 + α句型 + 法则10极简至上 | `8ad9821` | 2026-07-05 |
| CUE-004 | Lottie 编译器+渲染器 v2+三路由成品+SPEC v0.7; 合并单 6 项阻断修正（关闭） | `72bc96f` | 2026-07-05 |
| CUE-005 | 专家测试失败修正+专业级重拍卡: P2/P3几何v0.6+白圈关节点+header CJK修复+SAM2剪影v2+Gate-P节+12条校验器+clip_016退役令（关闭） | *(pending)* | 2026-07-05 |

---

## 4. 引擎能力现状

### 已实现（face-on）
- A层: RTMPose ONNX 姿态估计，kp_guard 置信过滤
- B层: SwingPhaseEngine 8相位检测，top_conf，fallback prominence 已修复
- C层: triline_geometry.py — shoulder_lateral_tilt @ top（窗口中位数 top±5帧）
- Profiler: 6模块，137/137 clip 全库分类
- Flywheel: baseline_v1（center=-6.8°, lower=-18.8°, upper=+5.0°）
- CueGenerator: v0.1 — Plan Schema + α句型 + 校验器 11 条（k_max=0.06 定版）
- CueCompiler: Lottie 编译器 + MP4 渲染器 v2（P2红线+P3弧箭头，3路由）
- 置信门: Confirmed/Likely→.lottie+.mp4+static_last; None→neutral绿勾; SILENT→重拍引导卡（分型）

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
| CUE-004 正式双轨验收 | 需真实阳性素材（Jason 自拍错误示范） | Jason 录制 |
| dtl-eet-1 Step 3+5 | 疑似拼接素材需确认 | Jason |
| NEEDS_HUMAN.md 4项旧项 | 长期搁置 | deferred |
| SAMPLES_INDEX.md VLM补全段复审 | 10张一句话读法须 Jason 目视裁决 | Jason |

---

## 7. 下一步候选与 GHOST 线立项

### GHOST 线（Ghost 克隆管线，2026-07-05 立项）

产品新增 ghost 形态：**抽象 3D 人体叠加** — 红色半透明 MHR 人形精确贴合叠加在真实视频之上，同步做出正确动作。
原则"身体为因，球杆为果"已入档 CUE_DESIGN_LANGUAGE §12。

**方向变更 2026-07-06**: MimicMotion/SVD 生成路线冻结。SMPL/HMR2.0 = research-only 方向探针，退役不进产品。

**产品底座锁定（Jason 裁决 2026-07-06）**:
- **主轨**: SAM 3D Body (MHR topology) — SAM License，PRODUCT_CANDIDATE_CUSTOM_LICENSE
- **备轨**: Anny native topology (Apache 2.0) — PRODUCT_CLEAN_PERMISSIVE，永不下 smplx NC 包
- **红线**: SMPL / SMPL-X + Multi-HMR-Anny checkpoint → 仅研究对照，绝不进产品
- **red line**: Ultralytics YOLO (AGPL) 不得混入检测链

#### GHOST 授权关卡（上线前必须关闭）

| 关卡 | 内容 | 状态 |
|------|------|------|
| T0 核验 | 原始 LICENSE 逐字固化，裁决 SAM=无NC | ✅ 完成 2026-07-06，commit c2f782d，见 GHOST_LICENSE_T0.md |
| 上线 legal review | SAM License §1.b.v 贸易管制 + §1.b.ii 出版致谢义务 | ⏳ 产品化前必须关闭 |
| HF gated 申请 | facebook/sam-3d-body-dinov3 需申请 access | ⏳ Jason 提供 HF token 后 |

#### GHOST 技术关卡

| 关卡 | 目标 | 状态 |
|------|------|------|
| GHOST-001 T1 | MimicMotion 可行性验证（fo-ok-1 自我重建，72fr） | ✅ 完成 2026-07-06，commit a010f95 |
| GHOST-002 T1 | SMPL/HMR2.0 单帧贴合方向探针（research-only） | ✅ 完成 2026-07-06，commit 54c6ac9，退役不进产品 |
| GHOST-003 T1 | SAM 3D Body (MHR) 单帧精确贴合探针 | ✅ 完成 2026-07-06，commit 8f2b2ad |
| GHOST-003 T1.5 | 体型拟合优化层（rembg+3带宽度对齐） | ✅ 完成 2026-07-06，commit 351a81a，shoulder 6.2%, hip 2.8% |
| GHOST-003 T1.6 | 上半身精调层（纯红渲染+5带迭代cx纠偏） | ✅ 完成 2026-07-07，commit 48706ec，shoulder 1.2% hip 1.0% |
|| GHOST-003 T1.7 | IoU-based shape fitting（上半身 IoU 最大化） | ✅ 完成 2026-07-07，commit 9e5c6d8，upper IoU 0.9385 ≥ 放行线 0.92 ✓ |
|| GHOST-003 T2 | MHR 整段挥杆序列贴合（address→follow_through） | ✅ 完成 2026-07-07，commit 3b61eb6，mean=0.897 min=0.372(fr087) P5=0.824 |
|| GHOST-003 T2.1 | 崩帧哨兵 + 动态帧 2D 紧化 | ⛔ 等 Jason 目视最差有效帧裁决 — mean=0.901 min=0.742(fr103) P5=0.830 |
| GHOST-004   | 修正动作驱动（基线正确姿态驱动用户 MHR） | ⏳ 待 T2 验收 |

**逐点契合铁律（GHOST 线 2026-07-06 立）**: address 帧红人须在头/颈/肩/肘/腕/髋/膝/踝逐点契合真人；address 不契合不得进入动作修正阶段。

**GHOST-003 分级契合放行判据（Jason 裁决 2026-07-07，永久有效）**:
- **主指标**: 上半身 IoU（head→hip+60 区域；green=human / red=mesh / yellow=overlap）
- **放行线**: 上半身 IoU ≥ 0.92。引擎每帧自算，低于 0.92 触发重优化或标记待查
- **下肢（膝/踝/脚）**: 达标即止，不作通过条件；引擎不得为压下肢 miss 牺牲上半身 IoU
- T1.7 实测 0.9385 ≥ 0.92 → address 契合正式通过，T2 放行

**GHOST 线分级契合铁律（2026-07-07 Jason 裁决，永久有效）**:
- **上半身（肩/躯干/髋）**: edge miss 目标 ≤3%。诊断发生区 + 动作归因区，不容妥协
- **下肢（膝/踝/脚）**: 合理贴合即可，不追求完美（支撑点、非诊断区，达标即止）
- 达此标准 = address 契合通过，放行 T2 整段序列

**GHOST-001 T1 实测**: 推理 11:56 + VAE CPU decode ~25min，总 ~37min；peak VRAM 10.07GB（CPU offload 生效）；72fr W576×H1024 竖版正确；帧序列 frames/ 落盘防重跑。MimicMotion 路线冻结于此。

**GHOST-003 T1.7 实测（2026-07-07）**:
- 算法: scipy.minimize_scalar.bounded + column-remap proxy IoU
- B_UPP opt_sx=1.1028 (proxy_IoU=0.9252) / B_HIP opt_sx=1.0380 (proxy_IoU=0.9093)
- B_LOW sx=1.1133 (达标即止, 含迭代cx纠偏)
- 上半身实际 IoU: T1.6=0.8729 → T1.7=0.9385 (改善 +0.0656)
- 局部: shoulder=0.9622 / hip=0.9702 / lower=0.6650 (达标即止)
- Edge miss 参考: shoulder 2.7% / hip 1.4%
- 总耗时 16.6s  VRAM 3610MB
- **Jason 裁决**: 上半身 IoU ≥ 0.92 为放行线；0.9385 ≥ 0.92 ✓ → T1.7 正式关闭，T2 放行

**GHOST-003 T2.1 实测（2026-07-07）**:
- 哨兵: A1(delta>=0.25) / A2(conf<0.65) / A3(cx偏移>60px)
- 无效帧: 1帧 — fr087 (A1: delta=0.537, T2_IoU=0.372) → 插值填充
- 2D joint opt: Nelder-Mead [sx,dy]; 结论: dy_upp≈0px全程 (y方向无需调整)
- 有效帧 (n=111): mean=0.9014 / min=0.742(fr103) / P5=0.830
- 对比 T2 (n=112含无效帧): mean Δ+0.005 / P5 Δ+0.005
- 最差3有效帧: fr103(0.742) / fr064(0.761) / fr094(0.783)
- 关键帧: address(0.938) / top(0.865) / impact(0.912)
- Windows交付: ghost003_t21/
- Clip: fo-ok-1  NF=112  有效帧=112  总耗时 163s  peak VRAM 3611MB
- 放行线: IoU ≥ 0.92
- mean IoU: 0.8966
- min  IoU: 0.3722 (fr087) ← 最差帧
- P5   IoU: 0.8242
- 低于放行线 (< 0.92) 帧: 56/112（fr054~fr111 动态段为主）
- 静态 address 段 (fr0~fr053): IoU ≈ 0.93~0.94 全通过
- 动态段主要失效区: fr062/064/072/087/094/103/104（arm swing/occlusion）
- 最差3帧: fr087 IoU=0.372 / fr103 IoU=0.734 / fr064 IoU=0.763
- 关键帧: address_fr000 / top_fr097 (IoU=0.861) / impact_fr088 (IoU=0.912)
- Windows 交付: C:\Users\jason\Desktop\rtmpose_results\preview\ghost003_t2\
- ⛔ 等 Jason 目视最差帧裁决 T2 是否通过

**最后更新**: 2026-07-07 — GHOST-003 T2 数据产出，⛔ 等 Jason 目视最差帧裁决

**相位命名规范 (铁律)**: 一律使用 B 层 8 相位体系：
`address / takeaway / backswing / top / transition / downswing / impact / follow_through`
禁止自造 P1/P2/P3/P4/finish 等替代命名。ghost001_prep.py/run.py 已修正 (commit 待补)。

### CUE 线候选（GHOST 线外）

- CUE-001 3秒测试终审（clip_016_left_top_cue.jpg 目视）
- CUE-004/005 正式双轨验收（Jason 自拍错误示范素材就位后）
- 第二错误类型诊断（Sway / Hip Slide）
- B层 0.82 窗口技术债立项
- 飞轮扩充（自采同人配对录制方案）（供 Jason 选方向）

- CUE-001 3秒测试终审（clip_016_left_top_cue.jpg 目视）
- CUE-004/005 正式双轨验收（Jason 自拍错误示范素材就位后）
- 第二错误类型诊断（Sway/Hip Slide 任选一）
- B层 0.82 窗口技术债规格立项
- 飞轮扩充：自采同人配对数据录制方案
- 时序类 δ 句型动画（CUE 生成器专项）

---

*PROJECT_STATE.md — GHOST-003 T1.6 ⛔ 等待 Jason 验收 2026-07-07 / 维护人: Hermes / 裁决人: Jason*
