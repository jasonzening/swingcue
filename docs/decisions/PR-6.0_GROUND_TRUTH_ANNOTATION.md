# PR-6.0 Ground Truth Annotation Spec

> Created 2026-05-19 alongside PR-6.0 Phase 1 kickoff.
> Defines the human-annotated reference data for objectively scoring pose model outputs.

## Why

CC 跑出 COMPARISON.mp4 后，需要客观判断哪个模型最准。光靠"看着像"主观容易飘。
建立标注 reference 让对比可量化（PCK@0.05, mean error）。

## 标注什么

**5 个关键帧 × 2 个视频**（开始小，验证流程）：

测试视频:
- `b3fea3f0-e248-44d7-a923-0bb43172b5bf` (Jason 的参考挥杆，face-on)
- `a735cc7d-1d4d-4b73-870f-30dca5c4aac0` (recent real-user, face-on)

每个视频 5 个 frame（来自现有 phase_markers）：
- Setup frame
- Top frame
- Transition frame
- Impact frame
- Finish frame

每个 frame 标注 **9 个 critical keypoint**（不是全 17 个，focus on golf-coaching 重要点）：

```
头部 (1):     head_crown / nose
肩膀 (2):     left_shoulder, right_shoulder
手肘 (2):     left_elbow, right_elbow
手腕 (2):     left_wrist, right_wrist
髋部 (2):     left_hip, right_hip
```

## 标注格式

每个 keypoint 标 **(x, y, visible)**:
- x, y: 像素坐标 (image space)
- visible: true / false / occluded

JSON 结构:

```json
{
  "video_id": "b3fea3f0-...",
  "frame_idx": 23,
  "phase": "top",
  "image_width": 1280,
  "image_height": 720,
  "annotator": "jason",
  "annotated_at": "2026-05-19T...",
  "keypoints": {
    "nose": {"x": 640, "y": 180, "visible": true},
    "left_shoulder": {"x": 580, "y": 290, "visible": true},
    "right_shoulder": {"x": 700, "y": 290, "visible": true},
    "left_elbow": {"x": 500, "y": 380, "visible": false, "comment": "behind body"}
  }
}
```

## 标注工具

**Option A (最简单)**: Photo + 红笔在 iPad，手动列 9 个 (x, y) 数字
- 每帧 ~3 分钟
- 10 帧 = 30 分钟
- 0 setup time

**Option B (中等)**: CVAT — 免费 in-browser 标注工具
- 上传 frame，点击 keypoints
- 自动输出 COCO JSON
- ~1 分钟每帧 once familiar

**Option C (overkill for Phase 1)**: Roboflow / Supervisely — 团队级标注平台

**推荐 Option A** 给 Phase 1 — 简单快速，先验证流程

## 评估指标

CC 把 model 输出和 ground truth 比对，输出：

```python
# 每个 (video, model, frame, keypoint) 计算:
err_px = euclidean_dist(model_kp, gt_kp)
err_normalized = err_px / image_diagonal

# 聚合:
pck_at_5 = % keypoints where err_normalized < 0.05  # PCK@0.05 标准指标
mean_err_px = mean(err_px) across all (frame, kp) pairs
```

每个 model 拿到一个 score table：

```
Model: RTMPose
─────────────────────────────────
            PCK@0.05    Mean Err
Setup       100%        4.2px
Top          88%       12.3px  ← left_elbow occluded
Transition   77%       18.7px  ← drift
Impact       66%       25.1px  ← left_wrist crossed
Finish       88%        9.4px
─────────────────────────────────
Overall      83%       13.9px
```

直接对比 4-5 个 model 的 table，picking winner 不再主观。

## 标注流程

1. **CC 输出 reference 图**: `python/benchmark/output/<video_id>/reference_frames/<phase>.png` (单 frame 高清图，无 overlay)
2. **Jason 在 iPad / Photoshop / 任何工具**: 数出 9 个 keypoint 的 (x, y)
3. **Jason 填表**: 复制下面模板，10 帧填完
4. **CC 跑 metrics**: 对比每个 model 输出 vs ground truth, 输出 score table

## Ground Truth 填写模板

```yaml
# b3fea3f0_setup.yaml (Jason 填写, ~3 分钟)
video_id: b3fea3f0-e248-44d7-a923-0bb43172b5bf
phase: setup
frame_idx: 0
image_width: 1280
image_height: 720
keypoints:
  nose:           {x: ___, y: ___, visible: true}
  left_shoulder:  {x: ___, y: ___, visible: true}
  right_shoulder: {x: ___, y: ___, visible: true}
  left_elbow:     {x: ___, y: ___, visible: true}
  right_elbow:    {x: ___, y: ___, visible: true}
  left_wrist:     {x: ___, y: ___, visible: true}
  right_wrist:    {x: ___, y: ___, visible: true}
  left_hip:       {x: ___, y: ___, visible: true}
  right_hip:      {x: ___, y: ___, visible: true}
notes: "club shaft visible, no occlusion"
```

## 时间投入估计

- CC 准备 reference frame: 5 分钟
- Jason 标 10 帧 × 9 keypoint: 30 分钟
- CC 跑 metrics: 5 分钟
- 共: **40 分钟可以拿到客观 score 对比表**

## Out of Scope (Phase 1)

- ❌ 全 17 keypoint 标注 (Phase 2 再扩)
- ❌ 多个 annotator + inter-rater agreement (个人项目无需)
- ❌ 自动 ground truth (e.g. 用 Sapiens2 作 pseudo-label) — 会污染 evaluation
