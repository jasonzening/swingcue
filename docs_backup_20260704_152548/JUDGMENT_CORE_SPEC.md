# JUDGMENT_CORE_SPEC.md — 判断内核规格(D层规则 + E层根因图谱,第一闭环)

> 这是诊断引擎的"判断本体":定义什么算错、为什么错、怎么定主因。纯逻辑+高尔夫知识,**零视频依赖**——
> 全部规则可用合成数据单元测试,B 层(阶段)修好后直接接入。
> 阈值标 ⚙️ 为草拟初值,待 Jason 确认或真实数据校准。架构依据:FAULT_DIAGNOSIS_ENGINE.md。

---

## 1. D 层规则判官 — 第一闭环两条规则

### R1: loss_of_posture(起身/前倾角丢失)
- **特征**: `spine_delta[f] = spine_angle[f] − spine_angle_baseline`
  - spine_angle = 肩中点-髋中点连线与垂直方向夹角(固定坐标系:y向下,vertical=(0,1),arccos)
  - baseline = 该用户自己 address 阶段的中位数(个人基准,非外部标准)
- **角度门控**: 仅 DTL 侧面
- **阶段窗口**: downswing 起 → impact 末
- **触发**: spine_delta 向"变直"方向超过 ⚙️8°,且**连续 ≥3 帧**(防单帧噪声)
- **严重度**: 8-12° = mild;>12° = significant ⚙️
- **置信度**: 窗口内相关关节点(肩/髋)检测置信度的最小均值;低于 0.4 → 本规则不输出

### R2: hip_toward_ball(髋部向球前顶)
- **特征**: `hip_shift[f] = (hip_center_x[f] − hip_center_x_baseline) · ball_dir / torso_height`
  - baseline = address 阶段髋中点 x 的中位数;torso_height = address 时肩中点到髋中点距离(归一化,跨体型可比)
  - **ball_dir(球方向)定义**: address 时"髋中点 → 双腕中点"的水平方向符号(球在手的延长线下方,稳健且不依赖球检测)⚙️
- **角度门控**: 仅 DTL 侧面
- **阶段窗口**: transition 起 → impact 末(髋前顶发生在转换/下杆,比 R1 窗口早一段)
- **触发**: hip_shift > ⚙️5% torso_height,连续 ≥3 帧
- **严重度**: 5-9% = mild;>9% = significant ⚙️
- **置信度**: 同上,基于髋/腕点

### 哨兵(物理判官最小版): bone_length_sentinel
- 逐帧算躯干长(肩中点-髋中点),与 address 中位数比;偏差 >20% → 该帧标记"测量不可信"
- 不可信帧**从规则评估中剔除**(不参与"连续≥3帧"计数),并降低整体置信度
- 这是"conf=1.00 却错 22 帧"那类问题的防线:测量自身先体检,再谈判断

### 异常输出统一格式
```json
{ "fault_type": "loss_of_posture", "phase_window": "downswing-impact",
  "evidence": { "feature": "spine_delta", "peak_value": 11.3, "onset_frame": 152, "frames_sustained": 9 },
  "severity": "mild", "confidence": 0.78 }
```
> onset_frame(起始帧)必须输出——E 层归因要用"谁先发生"判断因果方向。

---

## 2. E 层根因图谱 — 最小版(一条因果链)

### 图谱节点(第一版,手工依据公认教学)
```
early_extension(根因): 髋部过早向球方向顶出
  ├─ 导致 → loss_of_posture(前倾角丢失/起身)
  ├─ 导致 → head_rise(头部上移)          [第二步加]
  └─ 导致 → arm_space_compressed(手臂空间被挤) [第二步加]
```

### 归因逻辑(第一闭环)
```
输入: D 层异常列表
1. R1(起身) 且 R2(髋前顶) 都触发:
   ├─ 因果时序校验: R2.onset_frame ≤ R1.onset_frame + ⚙️3帧容差
   │   (原因应先于或同时于结果;髋先顶,身才起)
   ├─ 时序成立 → 诊断 early_extension,确定度 Likely(2证据上限)
   │             loss_of_posture 标记为"结果",不单独报
   └─ 时序不成立(起身明显先于髋顶) → 各自独立报,不强行归因 ⚙️
2. 仅 R1 → 诊断 loss_of_posture(单纯起身),确定度按严重度 Possible/Likely
3. 仅 R2 → 确定度 Possible 的 early_extension 倾向;severity=mild 时不输出(低于报告线)⚙️
4. 都未触发 → "本次挥杆未检出明显问题"(正反馈,鼓励式)
```

### 确定度三档(铁律落地)
| 档 | 条件 | 措辞 |
|---|---|---|
| Possible | 1 证据 / 严重度低 / 置信度 0.4-0.6 | "可能有…的倾向" |
| Likely | 2 证据 + 时序成立 + 置信度 >0.6 | "很可能…" |
| Confirmed | ≥4 证据(第二步加厚后)| "确认…" 第一版不可达 |

---

## 3. F 层输出契约(一句话模板)

| 诊断 | 档 | 输出 |
|---|---|---|
| early_extension | Likely | "你很可能出现了 Early Extension:髋部过早向球方向顶,导致身体失去前倾角(起身)。优先修正:保持髋部在后方,不要向球顶。" |
| loss_of_posture | Possible/Likely | "下杆时身体有(些/明显)起身,前倾角没保持住。试着保持 address 时的前倾,头部高度不变。" |
| 无异常 | — | "这一杆姿势保持得不错,没有检出明显问题。" |
| 测量不可信占比高 | — | 失败文案(附录A.2),不给诊断 |

---

## 4. 合成数据单元测试(零视频依赖,现在就能跑)

构造假的特征序列(spine_delta[f], hip_shift[f], 帧置信度),验证判断逻辑:

| 用例 | 构造 | 期望输出 |
|---|---|---|
| T1 正常挥杆 | delta 全程 ±3°,hip ±2% | 无异常,输出正反馈 |
| T2 典型 EE | hip 自 transition +8%(onset 早),spine 随后 +12° | **Likely early_extension**,起身标结果 |
| T3 单纯起身 | spine +10°,hip 平稳 | loss_of_posture,无 EE 归因 |
| T4 时序倒置 | spine 先起(onset 早),hip 后顶 | 各自独立报,**不**归因 EE |
| T5 噪声帧 | 中间 2 帧骨长突变+delta 飙 40° | 哨兵剔除,不触发(不满足连续3帧) |
| T6 低置信 | 窗口内关节置信度 <0.4 | 规则不输出,整体降级 |
| T7 边界 | delta 恰 8.0° | 按 ≥ 触发(边界含),mild |

**验收**: 7 用例全过 = 判断内核逻辑成立。之后接入 B 层真实阶段+C 层真实特征,只换数据源不改逻辑。

---

## 5. 阈值定稿(2026-06-09 Jason 确认,按推荐值;仍标 ⚙️ 表示待真实数据校准可调)
- [x] R1 阈值 **8°**(起身),连续 ≥3 帧
- [x] R2 阈值 **5% 躯干高**(髋前顶),连续 ≥3 帧
- [x] 因果时序容差 **3 帧**(30fps≈0.1s)
- [x] 仅髋顶无起身:**mild 不报**(宁少报不误报)
- [x] 时序倒置(先起身后髋顶):**各自独立报,不强行归因 EE**(诚实优先)
> 校准计划:拍到"故意 early extension"测试视频后,用真实数据复核以上阈值,必要时微调。
