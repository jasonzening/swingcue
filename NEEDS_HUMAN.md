# NEEDS_HUMAN.md

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
