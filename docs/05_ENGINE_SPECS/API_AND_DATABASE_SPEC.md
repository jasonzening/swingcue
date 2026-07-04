# API_AND_DATABASE_SPEC.md — 接口与数据表规格

> 依据:SWINGCUE 主文档 附录 B,本文档为可直接开发的展开版。MVP-0 阶段先以本地文件/SQLite 实现同构结构,MVP-1 起上真后端;**字段一开始就按本规格命名,避免迁移返工**。

## 1. API

### POST /api/swing/analyze
创建分析任务。
- 入:`video_file`(multipart) 或 `video_id`;`user_id`(optional,MVP-0 可空);`angle`(optional,"face_on"|"down_the_line",用户手选时传)
- 出:`{ "analysis_id": "...", "status": "queued" }`
- 错误:视频过短/格式不支持 → 400 + 失败文案码(见主文档附录 A.2)

### GET /api/swing/analysis/:id
获取分析结果。
- 出:
```json
{
  "status": "done|processing|failed",
  "video": { "id": "...", "duration": 3.2, "fps": 30, "width": 1080, "height": 1920 },
  "angle": "down_the_line", "angle_confidence": 0.91,
  "phases": { "address": 10, "top": 31, "impact": 47, "finish": 68,
              "confidence": { "address": 0.88, "top": 0.81, "impact": 0.74, "finish": 0.79 },
              "warnings": [] },
  "indicators": [ /* 统一指示器 JSON,见 INDICATOR_ENGINEERING_SPEC */ ],
  "summary_message": "击球时姿势保持得不错。"
}
```

### POST /api/swing/feedback
用户"准/不准"反馈(数据飞轮机制一)。
- 入:`{ "analysis_id", "indicator_id"|"phase", "frame_index", "feedback": "accurate"|"inaccurate" }`
- 出:`{ "ok": true }`

### GET /api/swing/history?user_id=
分析历史列表(缩略图、日期、主要问题)。MVP-2 实现。

## 2. 数据表

```
swing_videos
  id, user_id, file_url, angle, angle_confidence, duration, fps,
  width, height, region, status, created_at

swing_analyses
  id, video_id, model_version, status,
  confidence_summary(json), summary_message, created_at

swing_phases                      -- 关键帧独立成表(ChatGPT 建议采纳,便于飞轮按帧检索)
  id, analysis_id,
  address_frame, top_frame, impact_frame, finish_frame,
  confidence_json, warnings_json, detector_version

swing_keypoints
  id, analysis_id, frame_index, keypoints_json, source_model

swing_indicators
  id, analysis_id, indicator_type, frame_index,
  result_json, confidence, confidence_level, status, created_at

swing_feedback
  id, user_id, analysis_id, indicator_id, frame_index, feedback, created_at

coach_annotations                 -- 预留,feature flag 隐藏(主文档 8.13)
  id, analysis_id, coach_id, frame_index, joint,
  model_xy, corrected_xy, problem_tag, media_url, created_at

users
  id, ..., region,                -- 全球化必填(主文档 8.12)
  data_training_consent(bool)     -- 数据训练授权开关(主文档 5.4)
```

## 3. 约定

- 所有指示器结果写入 `swing_indicators.result_json`,格式必须符合统一 Schema(INDICATOR_ENGINEERING_SPEC §1)
- `confidence_level=low` 的记录,`status` 必须为 null(禁止红绿)
- keypoints 原始 json 永久保留(数据集资产),不随分析删除——**但受 consent 约束,见 §4**
- MVP-0 的 CLI 输出文件结构与上述表逐一对应(phases.json ↔ swing_phases 等),保证平滑迁移
- ⚠️ **analyze.py 落库时必须自动带上 region 和 data_training_consent 字段**(即使 MVP-0 默认值也要写),否则后期 Marketplace/全球化迁移是灾难

## 4. Consent 与删除策略(上架隐私必需)

**data_training_consent = false 时:**
- 视频仍可用于**本次分析**(用户自己看结果)
- **不进入**训练集、不进入 coach/model improvement 队列
- keypoints 仅保留供本次结果展示,不纳入数据飞轮

**用户删除视频时:**
- 删除 file_url 对应的视频文件
- keypoints 是否保留按 consent 决定:有授权 → 可保留匿名 keypoints(去除身份关联);**无授权 → keypoints 一并删除或脱敏**
- swing_analyses/indicators 的结果记录按同样原则处理

> consent 默认值、措辞、留存期限上线前须经律师确认(主文档 5.4:不用"动作特征非生物特征"之类规避话术)。
