# SWING_PHASE_DETECTOR_SPEC.md — 关键帧检测模块技术规格

> 坎 1,最高优先级。侧面逻辑已实测通过(2026-06-03,test-dwontheline:address=10/top=31/impact=47/finish=68,impact 经 address 锚点修正后肉眼验收通过)。本规格固化已验证逻辑 + 定义剩余工作。

## 1. 接口

**输入:** `keypoints_json`(RTMPose 逐帧关节坐标+置信度)、`fps`、`video_width`、`video_height`、`angle`(可选,影响参数组)

**输出:**
```json
{
  "address_frame": 10, "top_frame": 31, "impact_frame": 47, "finish_frame": 68,
  "confidence": { "address": 0.88, "top": 0.81, "impact": 0.74, "finish": 0.79 },
  "warnings": [],
  "debug": { "impact_dist_to_addr_px": 47, "speed_peak_frame": 43, "params_version": "dtl_v1" }
}
```

## 2. 检测规则(已验证版,纯规则,不上深度学习)

预处理:取手腕(主导手或双腕中点)逐帧坐标 → 速度=位置一阶差分 → Savitzky-Golay/滑动平均平滑 → 异常值剔除(腕部击球瞬间会跳点,实测 max 69px)。

- **address:** 挥杆开始前,手腕速度低于静止阈值的最后一段的末帧;同时记录该帧手腕坐标为**锚点 (anchor_x, anchor_y)**
- **top:** 手腕最高点附近,垂直速度由上升转下降(过零反转)的帧
- **impact(关键,易错):** ⚠️ **不是速度峰值本身**——手腕速度峰值出现在下杆中段(能量尚未传给杆头)。正确规则:在速度峰值**之后**的帧里,找**距 address 锚点欧氏距离最近**的帧。物理依据:球不动,击球时手必回到起杆位置附近。实测:峰值在 43 帧,锚点法定位 47 帧(dist=47px),正确
- **finish:** impact 后,手腕速度降回静止阈值以下且身体进入稳定结束姿态的帧

## 3. 合理性校验与失败条件(违反即降置信/报错,绝不静默给错)

1. 顺序:`address < top < impact < finish` 不成立 → 报错,输出诊断,不给结果
2. impact 高度:impact 帧手腕高于 address 超过 **30% 躯干高度** → confidence=low + warning(已实现)
3. 节奏:`impact - top` 帧数应明显小于 `top - address`(下杆远快于上杆),违反 → warning + 降置信
4. 速度曲线多峰严重(多次挥杆/废动作) → 输出 "uncertain",建议用户一次只拍一杆
5. 关键点缺失超过 X%(建议 20%)的帧段覆盖关键区域 → 输出 "failed" + 失败文案码

## 4. 置信度计算

综合:①速度曲线特征清晰度(峰值尖锐噪声小=高,模糊多峰=低) ②impact 的 dist_to_addr(越小越高) ③该帧附近关节点检测置信度均值。输出 0-1,映射 high≥0.7 / medium 0.4-0.7 / low<0.4。

## 5. 剩余工作

- [ ] 正面(face-on)参数组:垂直方向特征类似,水平特征不同;调参后记录两角度参数差异(params_version: "faceon_v1")
- [ ] 重构为可复用模块 `SwingPhaseDetector`(类/函数,输入任意 json)
- [ ] 单元测试:用归档 keypoints json 跑,校验顺序/阈值/生理极限
- [ ] 验收:正面、侧面**各至少 5 段**测试视频通过(目前各 1 段,需补拍)

## 6. 验收标准(两级)

- **Hard pass(必须达到):** impact 落在击球附近窗口内,**不能明显跑到下杆中段或收杆**;四帧顺序正确;不确定时输出 low confidence 不硬给。早期最该防的是"明显错还 high confidence"。
- **Gold pass(理想目标):** impact 偏差 ≤2-3 帧(杆头在球附近)。
- 单个视频只差 4-5 帧但仍在击球窗口内 → 算 Hard pass 通过,不打回整个任务(单靠手腕锚点、无杆头检测时难稳定到 ≤2-3 帧)。
- 覆盖:正面、侧面**各至少 5 段**测试视频(目前各 1 段,需补拍)。
