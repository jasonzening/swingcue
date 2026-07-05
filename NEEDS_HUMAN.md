# NEEDS_HUMAN.md

---

## batch3 EET — dtl-eet-1/2/3 机位确认 (2026-07-03)

**状态**: Layer 0 感知门 → needs_human，管线暂停

**问题**: 三段 dtl-eet-* 文件名前缀为 dtl，但 VLM 采样 5 帧结果如下：

| 片段 | 面向镜头帧数 | DTL帧数 | VLM判断 |
|---|---|---|---|
| dtl-eet-1 | 3/5 (fr0,50,101) | 2/5 (fr152,202) | mixed |
| dtl-eet-2 | 4/5 (fr0,32,65,97) | 1/5 (fr129) | mixed |
| dtl-eet-3 | 4/5 (fr0,21,43,65) | 1/5 (fr86) | mixed |

VLM + 人工抽帧确认：fr0/fr50/fr101 golfer正脸朝向镜头，是真face-on。
fr152/fr202/fr129/fr86等 follow-through 帧才是DTL。

dtl-eet-1 追加异常：fr0-101 golfer穿polo+floral shorts；
fr152开始换成sweatshirt+tie-dye shorts → **疑似剪辑了不同场次素材**。

**请人工确认**：
1. dtl-eet-1：是否包含两段不同次击球/不同服装拼接？前段(fr0-101)实际是 face-on 机位？
2. dtl-eet-2：前4/5帧都是 face-on 机位，仅 fr129 是 DTL。这段是否为 face-on 素材（应归 fo-eet-*）？
3. dtl-eet-3：同上，4/5 face-on，仅最后 follow-through 是 DTL。
4. 确认后重新标注机位（dtl vs face-on），或说明是有意"混合机位"素材。

确认后 Hermes 可继续跑 DTL 管线（Step 3）。

fo-eet-1/2/3 Layer 0 已 PASS（face-on 一致），Step 4 已完成，见下方数据。

---

## Gate-1 v1.1 batch2 结果 — 待人工审核 (2026-06-11)

11段新素材全部过 Gate v1.1（address静止段帧+三票制），gate1 sheet 在桌面：
  Desktop/rtmpose_results/preview/batch2/gate1/

**异常需人工检视**:

### dtl-wrong-3 — 极度可疑
  addr=fr8, top=fr21(top_conf=0.01), impact=fr388, video=419fr
  top_conf=0.01 说明未检出有效反杆顶点。impact=fr388 几乎是视频末尾。
  可能：视频包含大量非挥杆内容，或多段间隔导致定位失败。
  请人工看 gate1_dtl-wrong-3.jpg 确认锚点是否合理。

### fo-wrong-1 — address 偏晚
  addr=fr180, top_conf=0.00, video=363fr
  地址检出在帧180（约视频中点），top_conf=0.00 degenerate。
  请人工看 gate1_fo-wrong-1.jpg 确认前180帧内容。

---

## fo-ok-2 fr75 跳变核查 — 结论更新 (2026-06-11)

**⚠️ 更正**：之前记录"fr75 single-frame jitter confirmed"为**错误结论**。

实际逐帧测量（nose_x/y 轨迹 fr60-90）：
- fr74→75 delta: (+0.3, +0.5)px — 完全平滑
- fr75→76 delta: (−0.7, +1.1)px — 完全平滑
- 无任何单帧跳变。

**因此**：
- head_lat/head_vert 在 fr75 附近的测量值是**有效的连续运动**
- 此前"head_lat=−70.9%, head_vert=+72.5% 无效"的结论**被撤销**
- 哨兵缺口（keypoint temporal coherence）问题**仍然存在**（设计层面），但本视频无此问题的实例

---

## GT 标注素材 — 待人工指认 (2026-06-11)

桌面 gate1_gt/ 目录：
  gate1_gt/201058/ — fr0180-fr0200 共 21 张（每帧标帧号，原比例）
  gate1_gt/201015/ — fr0055-fr0072 共 18 张（每帧标帧号，原比例）

**请人工指认**：
  1. 201058：fr180-200 中哪一帧是真正击球帧？
  2. 201015：fr55-72 中哪一帧是真正击球帧？

提供帧号后用于校准 gate-1 impact 锚点。

---

## Gate-1 v3/v3.4 原始段 — 已核对 (2026-06-10 → 2026-06-11)

5段原始视频 gate1 sheet 在桌面 gate1/ 目录。
v3.4 置信度已有区分度（conf 0.54~0.97，不再固死 1.00）。
锚点帧位 v3.4 vs v3.3 无变化（经 _diag_all.py 比对确认）。

请核对：
1. 8阶段分得对不对？
2. address/top/impact 缩略图是否对应正确动作
3. 201015 swing_count=3，first_swing_end=fr133 是否正确
4. 各 conf 是否有区分度（0.74~0.93）

---

## GT Line Rendering — gt_lines/ (2026-06-10)

Rendered 227 annotated frames to:
  Desktop/rtmpose_results/preview/gt_lines/

Per-video sub-folders:
  DTL (201054, 201058): Tush Line (yellow) + Spine axis (cyan)
  Face-on (201015, 201039, 201047): Head vertical/horizontal lines, forearm CW chain

**NOT RENDERED**: Shaft plane (Over-the-Top) — requires club detection (not yet built).

Human action: inspect frames in gt_lines/ and confirm or correct anchor frames.

---

## clip_129 (dlt-6) 实际错误类型 (2026-07-04)

**状态**: 需人工目视确认

**背景**:
- clip_129 = dtl-1/dlt-6.mp4, split-screen, left=✗(cross), right=○(OK)
- gate1 全量帧分析: swing_phase top=fr26
  LEFT tilt@top = -3.39° / RIGHT tilt@top = -2.83° / diff = 0.56°
- 结论: 两半脊柱侧倾几乎相同, 不是 spine_lateral_tilt 错误
- 已从 spine_lateral_tilt 基准构建中移除

**待确认**: 打开 preview/split_check/dlt-6/left.mp4 + right.mp4
目视确认该配对讲解的是什么错误类型 (候选: sway/手腕控制/杆头路径)
确认后更新本条目并记录到 docs/GT_LABELS.md。

---

## clip_035/039/041 配对筛查剔除记录 (2026-07-04)

**clip_035** (134. 不同杆数的区别): 非同人配对 — 120杆 vs 72杆不同水平球员对比，剔除。

**clip_039** (138. 读者一眼就看懂为什么要这么练): 背面机位(非face-on)，剔除。
  inventory: angle=DTL+static (与Gate3=face_on+full_swing冲突，人工目视确认为DTL，以人工为准)

**clip_041** (15. 增加20m后不击打后地的挥杆方法): 非同人配对 — Beginner vs Pro对比，剔除。
