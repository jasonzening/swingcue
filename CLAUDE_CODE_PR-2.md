# SwingCue PR-2: SAM 3D Body 集成 — Claude Code Task

## Context（你必须先理解，不要跳过）

SwingCue 是一个 AI 高尔夫挥杆纠正应用，repo 在 `jasonzening/swingcue`（你现在所在的工作目录），部署到 `swingcue.ai`。

**当前任务**：把姿态识别从 MediaPipe 切换到 fal.ai 托管的 SAM 3D Body（`fal-ai/sam-3/3d-body`）。MediaPipe 输出的肩部 keypoint 实际是腋窝不是肩峰，且无遮挡推理，已被否决。SAM 3D Body 输出 70 个 MHR keypoints，包含真正的 acromion + 髋 + 手指 + 脚，5mm 精度。

**已完成**：
- ✅ fal API 测试通过（`scripts/fal_test.py`，已 commit）
- ✅ keypoint 索引识别完成（基于 `test_finish.png` finish 帧）
- ✅ PR-2A 数据库迁移已在 Supabase 跑通、verified

**本轮你要做**：
1. PR-2A 收尾：把已跑通的 migration SQL commit 到 repo
2. **STOP，等待用户审批**
3. PR-2B 实现：Railway analyzer 集成 fal API
4. **STOP，等待用户审批**

PR-2C（前端 keypointOverlays.ts 重写）不在本轮范围。

---

## 工作模式（约束，必须遵守）

1. **严格分阶段**：PR-2A commit 完 → 输出报告 → 等审批 → 才开始 PR-2B。绝对不要自己跨过审批节点。
2. **Additive only**：不修改 PR-1A 的三张表 schema，不大改现有 analyzer 主控流程。
3. **不重复提问用户**：参考下方"关键数据"独立决策。如必须问，一次问完。
4. **commit message**：用 conventional commit 格式（`feat(scope):`、`fix(scope):`），中英文均可。
5. **测试视频 ID**：`eec305a5-8758-4cf0-ad25-b81e72d3653b`（jasonzjn@hotmail.com 账号下，用于后续 PR-2B 验证）。
6. **3-way 对账**：commit 前 `git status` + `git log --oneline -5` + `git fetch && git status` 确认 local/origin 状态。
7. **API 边界**：后端写库必须用 service-role client（见 `docs/decisions/API_CLIENT_BOUNDARY.md`）。
8. **写代码前先读**：每个要修改的文件，先 `cat` 或 `view` 看一遍现状再动。

---

## PR-2A: Commit migration

### 步骤

1. 创建文件 `supabase/migrations/{TIMESTAMP}_pose_3d_phases.sql`
   - TIMESTAMP 格式：`yyyyMMddHHmmss`，用当前时间
   - 完整内容见下方"PR-2A SQL"section
2. `git status` 确认是干净的 working tree（除了新加的 migration 文件）
3. `git add supabase/migrations/` + commit + push
4. 输出"PR-2A 交付物报告"（见下方模板）
5. **STOP**

### Commit message（直接用，不要改）

```
feat(db): PR-2A pose_3d_phases for SAM 3D Body keypoints

- New table storing per-phase fal sam-3/3d-body outputs (70 kp + MHR params)
- Denormalized shoulder/hip 2D for fast disc rendering
- RLS: user reads own, service role writes
- UNIQUE(video_id, phase_name) — 5 rows per video
- Additive only, PR-1A tables untouched
```

### PR-2A SQL（完整内容，照原样写入文件）

```sql
-- ============================================================================
-- PR-2A: pose_3d_phases — SAM 3D Body keypoints per phase
-- ============================================================================
-- Strategy:
--   One row per (video_id, phase_name). 5 phases per video:
--     setup, top, transition, impact, finish
--   fal-ai/sam-3/3d-body returns 70 keypoints per frame; we store the full
--   2D + 3D arrays as JSONB (for future re-analysis without re-calling fal),
--   AND denormalize shoulder/hip 2D into typed columns for fast disc rendering.
--
-- Additive only (PR-1A schema untouched).
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.pose_3d_phases (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id        UUID NOT NULL
                  REFERENCES public.swing_videos(id) ON DELETE CASCADE,
  user_id         UUID NOT NULL,
  phase_name      TEXT NOT NULL,
  frame_idx       INTEGER NOT NULL,
  frame_timestamp_ms INTEGER,

  keypoints_2d    JSONB NOT NULL,
  keypoints_3d    JSONB NOT NULL,
  focal_length    REAL  NOT NULL,
  bbox            JSONB,
  mhr_params      JSONB,
  glb_url         TEXT,

  image_width     INTEGER NOT NULL,
  image_height    INTEGER NOT NULL,

  shoulder_left_x   REAL,
  shoulder_left_y   REAL,
  shoulder_right_x  REAL,
  shoulder_right_y  REAL,
  hip_left_x        REAL,
  hip_left_y        REAL,
  hip_right_x       REAL,
  hip_right_y       REAL,

  fal_status        TEXT NOT NULL DEFAULT 'completed',
  fal_request_id    TEXT,
  error_message     TEXT,

  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  CONSTRAINT pose_3d_phases_phase_check
    CHECK (phase_name IN ('setup', 'top', 'transition', 'impact', 'finish')),

  CONSTRAINT pose_3d_phases_fal_status_check
    CHECK (fal_status IN ('uploaded', 'processing', 'completed', 'failed')),

  CONSTRAINT pose_3d_phases_video_phase_unique
    UNIQUE (video_id, phase_name)
);

CREATE INDEX IF NOT EXISTS idx_pose_3d_phases_video
  ON public.pose_3d_phases (video_id);

CREATE INDEX IF NOT EXISTS idx_pose_3d_phases_user
  ON public.pose_3d_phases (user_id);

CREATE INDEX IF NOT EXISTS idx_pose_3d_phases_video_phase
  ON public.pose_3d_phases (video_id, phase_name);

CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_pose_3d_phases_updated_at ON public.pose_3d_phases;
CREATE TRIGGER trg_pose_3d_phases_updated_at
  BEFORE UPDATE ON public.pose_3d_phases
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

ALTER TABLE public.pose_3d_phases ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS pose_3d_phases_user_select ON public.pose_3d_phases;
CREATE POLICY pose_3d_phases_user_select
  ON public.pose_3d_phases
  FOR SELECT
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS pose_3d_phases_service_all ON public.pose_3d_phases;
CREATE POLICY pose_3d_phases_service_all
  ON public.pose_3d_phases
  FOR ALL
  USING (auth.jwt() ->> 'role' = 'service_role')
  WITH CHECK (auth.jwt() ->> 'role' = 'service_role');
```

### PR-2A 完成后的报告模板

```
## PR-2A 交付物

- 文件：supabase/migrations/{TIMESTAMP}_pose_3d_phases.sql
- Commit: <commit hash>
- Branch: main (or your branch name)
- Pushed: yes/no

### 验证（请用户在 Supabase SQL Editor 跑）
SELECT count(*) FROM information_schema.columns 
WHERE table_name = 'pose_3d_phases';
预期: 27

⏸ 等待用户审批以开始 PR-2B。
```

---

## PR-2B: Railway analyzer 集成 fal

### Step 1: 探索（必须做）

在动代码前，先做探索：

```bash
# 1. 找出 analyzer 在哪
find . -type f -name "*.py" | grep -iE "(analyz|pipeline|pose|fal)" | head -20
ls -la
cat README.md 2>/dev/null | head -50

# 2. 找 analyzer 主入口
find . -type f \( -name "*.py" -o -name "*.ts" -o -name "*.js" \) \
  -exec grep -l "phase" {} \; 2>/dev/null | head -10

# 3. 看现有 Supabase client 集成
grep -rn "create_client\|createClient" --include="*.py" --include="*.ts" 2>/dev/null | head -20

# 4. 看是否已有 Railway 相关配置
ls railway* Dockerfile* docker-compose* 2>/dev/null
cat railway.json railway.toml 2>/dev/null
```

输出探索结果，**必须告诉用户**：
- analyzer 用什么语言（Python? Node?）
- 主入口文件路径
- 相位检测的输出结构（5 个 phase 的时间戳目前以什么形式产出）
- 现有 Supabase client 写法（service role 还是 anon？）
- 是否已有 ffmpeg 调用（用于抽帧）

**如果探索后发现现有 analyzer 不是 Python**（比如是 TS/Node），告诉用户后停下来等指示，不要强行用 Python。

### Step 2: 设计方案（探索完后写出来给用户看）

基于探索结果，写一份简短的设计方案：
- 新文件清单（模块名 + 职责）
- 修改文件清单（哪一行加哪些调用）
- 数据流图（5 行文字够了）
- 失败处理策略

**STOP，等用户确认设计**，然后再写代码。

### Step 3: 实现（用户确认设计后）

基于设计实现。建议结构（Python 版，按探索结果可能要改）：

```
analyzer/
├── sam3d/
│   ├── __init__.py
│   ├── keypoints.py          # 索引常量（见下方"Keypoint 索引"）
│   ├── fal_client_wrap.py    # fal 调用 + 重试
│   ├── frame_extract.py      # ffmpeg 抽 5 帧
│   └── supabase_writer.py    # 写 pose_3d_phases
└── (现有 analyzer 主入口加 5 行集成代码)
```

**关键设计原则**：
- fal 调用必须并发（asyncio.gather），5 个相位同时调，总耗时 ~10s 不是 50s
- 单个 phase 失败不影响其他 4 个，单独标 `fal_status='failed'`
- 失败重试 2 次，指数退避
- 写库前先把 70 个 keypoints_2d 解析出 shoulder/hip 4 对坐标，存到物化列
- service role client 写库（不是 anon）

### Step 4: 测试

用测试视频 ID `eec305a5-8758-4cf0-ad25-b81e72d3653b` 跑一遍，验证：

```sql
-- 应该看到 5 行
SELECT phase_name, fal_status, shoulder_left_x, shoulder_right_x,
       hip_left_x, hip_right_x, image_width, image_height, created_at
FROM pose_3d_phases
WHERE video_id = 'eec305a5-8758-4cf0-ad25-b81e72d3653b'
ORDER BY 
  CASE phase_name 
    WHEN 'setup' THEN 1
    WHEN 'top' THEN 2
    WHEN 'transition' THEN 3
    WHEN 'impact' THEN 4
    WHEN 'finish' THEN 5
  END;
```

### PR-2B 完成后的报告模板

```
## PR-2B 交付物

### 新文件
- analyzer/sam3d/keypoints.py
- analyzer/sam3d/fal_client_wrap.py
- ...

### 修改文件
- analyzer/main.py: +XX lines (line YYY: 加入 fal 调用)
- ...

### 测试结果
[贴上 5 行 pose_3d_phases 查询输出]

### 环境变量
需要在 Railway 配置：
- FAL_KEY
- SUPABASE_SERVICE_ROLE_KEY (如尚未配置)

### Commits
- <hash> feat(analyzer): sam3d keypoints constants module
- <hash> feat(analyzer): fal client wrapper with retry
- ...

⏸ 等待用户审批 PR-2B，之后开始 PR-2C（前端重写）。
```

---

## 关键参考数据（不要再去 search）

### fal API 调用

```python
import fal_client
import asyncio

async def call_fal(image_url: str) -> dict:
    handler = await fal_client.submit_async(
        "fal-ai/sam-3/3d-body",
        arguments={"image_url": image_url},
    )
    result = await handler.get()
    return result
```

环境变量：`FAL_KEY`（格式 `uuid:hash`）。
单次推理 5-10s，$0.02/次。

### Response schema（已 verified）

```python
{
  "model_glb": {
    "url": "https://fal.media/.../combined_bodies.glb",
    "content_type": "model/gltf-binary",
    "file_name": "combined_bodies.glb",
    "file_size": 1325332,
  },
  "visualization": {...},  # 渲染预览图
  "metadata": {
    "people": [
      {
        "person_id": ...,
        "bbox": [115.7, 120.2, 473.7, 723.3],   # [xmin, ymin, xmax, ymax]
        "focal_length": 923.79,
        "pred_cam_t": [...],
        "keypoints_2d": [[x, y], ...],          # length 70, source image pixels
        "keypoints_3d": [[x, y, z], ...],       # length 70, MHR camera space
        "shape_params": [...],
        "body_pose_params": [...],
        "hand_pose_params": [...],
        "global_rot": [...],
        "pred_global_rots": [...],
        "scale_params": [...],
        "expr_params": [...],
        "pred_joint_coords": [...],
        "mhr_model_params": {...},              # store this — for future retargeting
        "pred_pose_raw": [...],
      }
    ]
  },
  "meshes": [...],
}
```

### Keypoint 索引（已 verified 通过 finish 帧）

```python
# analyzer/sam3d/keypoints.py
"""
SAM 3D Body MHR keypoint indices.

Verified 2026-05-16 via:
  scripts/fal_test.py + scripts/fal_inspect.py on test_finish.png

The 70-keypoint array is organized as:
  0-6:   head + face cluster (7 points)
  7-20:  body joints (shoulders, hips, knees, ankles, feet) (14 points)
  21-41: hand_1 (21 points)
  42-62: hand_2 (21 points)
  63-69: extra body detail — clavicle/deltoid/sternum/throat (7 points)

Total: 7 + 14 + 21 + 21 + 7 = 70 ✓

Left/Right disambiguation in finish frame uses 3D Z-coordinate:
  Z+ = toward camera (anatomical front in finish pose)
  Z- = away from camera (anatomical back)
"""

# Body anchors (primary)
LEFT_SHOULDER  = 7    # acromion, target-side (3D Z=+0.041 in finish test)
RIGHT_SHOULDER = 8    # acromion, trail-side  (3D Z=-0.238 in finish test)
LEFT_HIP       = 9
RIGHT_HIP      = 10
LEFT_KNEE      = 11
RIGHT_KNEE     = 12
LEFT_ANKLE     = 13
RIGHT_ANKLE    = 14

# Foot detail
LEFT_TOE       = 15
LEFT_TOE_OUTER = 16
LEFT_HEEL      = 17
RIGHT_TOE      = 18
RIGHT_TOE_OUTER = 19
RIGHT_HEEL     = 20

# Wrists (assumed = hand cluster origins; validate with setup-frame test later)
LEFT_WRIST     = 21
RIGHT_WRIST    = 42

# Extra detail (use when needed)
LEFT_DELTOID   = 63
RIGHT_DELTOID  = 64
LEFT_CLAVICLE  = 65
RIGHT_CLAVICLE = 66
NECK           = 67
STERNUM        = 68
THROAT         = 69
```

### Supabase

- Project ref: `ciofgtwwcgyzfafmbjxu`
- API URL: `https://ciofgtwwcgyzfafmbjxu.supabase.co`
- 必须用 service role key（从 Railway env: `SUPABASE_SERVICE_ROLE_KEY`）
- Anon key 直接读 pose_3d_phases 会返回空（RLS 限制）

### 帧抽取约定

- 每个 phase 抽 1 帧（共 5 帧/视频）
- 抽帧时间戳从现有 analyzer 的相位检测输出获取
- 用 ffmpeg 命令：
  ```bash
  ffmpeg -ss {timestamp_s} -i {video_path} -frames:v 1 -q:v 2 {output.png}
  ```
- 抽完上传到 Supabase Storage（bucket `swing-frames`，路径 `{video_id}/{phase_name}.png`），
  再把 public URL 给 fal。**不要直接传 fal_client.upload_file**，因为我们也希望
  这些 phase 帧自己保留下来供前端用。

### 数据库重要列含义

```
image_width / image_height: fal 调用时的源图尺寸（不是原始视频尺寸）。
                            前端用它把 keypoints_2d 像素坐标转到 canvas 坐标。
fal_status:                 默认 'completed'。失败时 'failed' + error_message 填写。
frame_timestamp_ms:         视频起点开始的 ms 数，用于 traceability。
mhr_params:                 整个 mhr_model_params 字段（jsonb），未来做"完美挥杆对比"用。
```

---

## 开始执行

现在开始 **PR-2A**。完成后输出报告并 STOP。
