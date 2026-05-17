# SwingCue PR-3: YOLO11x-Pose 替换 fal SAM 3D Body

## 背景

PR-2 (A/B/C) 已经走通端到端流水线 + 前端集成，但 **fal-ai/sam-3/3d-body 的 70 个 keypoints 是 MHR 内部 mesh anchors（不是 anatomical surface landmarks）**。

实测发现（Chrome DevTools verified）：
- SAM kp 7,8 (LEFT/RIGHT_SHOULDER) 落在**胸口**位置，不是肩峰 acromion
- SAM kp 9,10 (LEFT/RIGHT_HIP) 落在**腹部**位置，不是 hip joint
- 没有任何 SAM keypoint 在真正的 acromion 视觉位置

→ SAM 3D Body 设计用于 mesh recovery（输出 GLB），不适合 visual disc overlay。

**本 PR 决策**：用 **Ultralytics YOLO11x-pose**（COCO 17 keypoints）替换。COCO keypoints 是 anatomical surface ground-truth annotated，shoulder = 真正的 acromion，hip = 真正的 hip joint。

## 保留 SAM 3D Body 数据不破坏

PR-2A schema 不动。PR-2B 写入的 `keypoints_2d`(70x2) + `glb_url` 保留给未来 3D mesh / perfect swing 比对。**新增** YOLO 字段。

---

## 工作模式（与 PR-2 一致）

1. **探索先行**：读现有 Python `python/sam3d/orchestrator.py`、`python/sam3d/fal_client_wrap.py`、`Dockerfile`，输出结构报告后等审批
2. **Additive 优先**：保留 SAM 3D 路径作为 fallback，YOLO 作为新主路径
3. **3 个 STOP 节点**：探索 → 设计 → 实现 → 测试
4. **Commits**：拆 5-7 个原子 commit

---

## 范围

### PR-3A: Schema 加新字段（Additive only）

```sql
-- migration: 20260516XXXXXX_add_yolo_keypoints.sql
ALTER TABLE pose_3d_phases
  ADD COLUMN IF NOT EXISTS yolo_keypoints_2d JSONB,   -- 17 x [x, y, conf]
  ADD COLUMN IF NOT EXISTS yolo_model TEXT,            -- e.g. 'yolo11x-pose'
  ADD COLUMN IF NOT EXISTS yolo_inference_ms INTEGER;
```

不动任何现有列。RLS / triggers 不变。

### PR-3B: Python analyzer 加 YOLO pipeline

新增 `python/yolo/` 包：
- `python/yolo/__init__.py`
- `python/yolo/keypoints.py` — COCO 17 索引常量 + types
- `python/yolo/inference.py` — YOLO11x-pose 推理 wrapper
- `python/yolo/supabase_writer.py` — 写入 yolo_keypoints_2d 等字段（用 raw httpx 同 sam3d/supabase_writer.py 模式）

修改 `python/sam3d/orchestrator.py` → 重命名 `python/orchestrator.py`，整合两条 pipeline：
- SAM 3D Body（保留，可由 env flag 启用 / 禁用）
- YOLO11x-pose（新增主路径）

修改 `Dockerfile` 加 ultralytics dep。

修改 `python/requirements.txt` 加 `ultralytics>=8.3.0`。

### PR-3C: 前端集成

`src/lib/sam3d/keypoints.ts` 类型扩展（保留 SAM3D_KP，新增 COCO_KP）：

```typescript
// COCO 17 keypoint indices (anatomical surface landmarks)
export const COCO_KP = {
  NOSE: 0,
  LEFT_EYE: 1, RIGHT_EYE: 2,
  LEFT_EAR: 3, RIGHT_EAR: 4,
  LEFT_SHOULDER: 5,    // ← acromion (真正肩峰)
  RIGHT_SHOULDER: 6,
  LEFT_ELBOW: 7, RIGHT_ELBOW: 8,
  LEFT_WRIST: 9, RIGHT_WRIST: 10,
  LEFT_HIP: 11,        // ← hip joint (真正髋关节)
  RIGHT_HIP: 12,
  LEFT_KNEE: 13, RIGHT_KNEE: 14,
  LEFT_ANKLE: 15, RIGHT_ANKLE: 16,
} as const;
```

PoseRow type 扩展：
```typescript
export type PoseRow = {
  // ... 现有字段
  yolo_keypoints_2d: number[][] | null;  // 17 × [x, y, conf]
  yolo_model: string | null;
  yolo_inference_ms: number | null;
};
```

`src/lib/overlay/sparsePhaseOverlay.ts`：
- buildShoulderDisc 改用 `yolo_keypoints_2d[5]` 和 `[6]`（fallback 到旧 shoulder_left_*/right_* 字段）
- buildHipDisc 改用 `yolo_keypoints_2d[11]` 和 `[12]`

`src/components/SwingPlayer.tsx` badge：
- 'yolo' → 'YOLO 11x' (green)
- 'sam3d' → 'SAM 3D' (green)
- 'mediapipe' → 'Real keypoints' (green)
- 其他 → 'Demo overlay' (orange)

`src/app/result/[id]/page.tsx` cascade：
```
PATH 0: yolo_keypoints_2d 存在 → sparse YOLO path (新主路径)
PATH 1: shoulder_left_x 等 SAM 字段存在 → sparse SAM path (向后兼容)
PATH A: keypoint_timeline_json → dense MediaPipe (legacy)
PATH B: stub
```

---

## Step 1: 探索（必做）

```bash
# 1. Python analyzer 现有结构
ls -la python/sam3d/
cat python/sam3d/orchestrator.py | head -50
cat python/sam3d/fal_client_wrap.py | head -30
cat Dockerfile

# 2. Railway environment vars 检查（不动只看）
# 列出可能影响推理的 env 名（FAL_KEY, SUPABASE_SERVICE_ROLE_KEY 等）
grep -rn "os.getenv\|os.environ" python/ | head -20

# 3. 看 /api/analyze 现状
cat src/app/api/analyze/[id]/route.ts | head -80

# 4. 检查 SUPABASE 服务端写入 path
grep -rn "service.role\|SERVICE_ROLE" python/sam3d/ src/

# 5. 检查 frame_extract 输出 dims
cat python/sam3d/frame_extract.py
```

**探索报告必须明确**：
1. orchestrator 的入口签名（analyze_video(video_id, user_id) 还是别的）
2. fal_client_wrap 的调用方式 + retry / timeout 行为
3. supabase_writer 的写入 endpoint（用 yolo_writer 复用同 pattern）
4. frame 提取的输出尺寸（720x1280?）和命名约定
5. Railway 是否已经支持 GPU 推理（probably CPU only — 这影响 YOLO 选 nano 还是 x）

**STOP 等审批后再设计**

---

## Step 2: 设计

输出：
1. 新文件清单 + 职责
2. 修改文件清单 + 改动范围
3. YOLO 模型选择（基于 Railway CPU vs GPU）
   - **CPU only**: yolo11n-pose (2.6M params, ~50ms/frame) 或 yolo11s-pose
   - **GPU available**: yolo11x-pose (60M params, ~5ms/frame, best accuracy)
4. 串行 vs 并行（5 phase frames）
5. 失败隔离策略（YOLO 失败但 SAM 成功如何处理 / 反之）
6. Commit 拆分（5-7 个）

**STOP 等审批**

---

## Step 3: 实现

### 关键参考代码

**YOLO11x-pose 推理 Python**:
```python
# python/yolo/inference.py
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

_model = None

def _get_model():
    global _model
    if _model is None:
        logger.info("[yolo] loading yolo11x-pose model...")
        _model = YOLO('yolo11x-pose.pt')  # auto-downloads ~60MB on first call
        # or 'yolo11n-pose.pt' for ~6MB CPU-friendly
    return _model

async def infer_pose(image_path: str) -> dict:
    """Returns: {
      'keypoints_2d': [[x, y, conf], ...]  # 17 items, COCO order
      'bbox': [x1, y1, x2, y2] or None,
      'image_width': int,
      'image_height': int,
      'inference_ms': int,
      'model': 'yolo11x-pose',
    } or None if no person detected.
    """
    import time
    model = _get_model()
    t0 = time.time()
    results = model(image_path, verbose=False)
    inference_ms = int((time.time() - t0) * 1000)
    
    if not results or len(results) == 0:
        return None
    r = results[0]
    if r.keypoints is None or len(r.keypoints.data) == 0:
        return None
    
    # Take highest-confidence person (golf swing is single-person)
    kpts = r.keypoints.data[0]  # tensor [17, 3] = (x, y, conf)
    
    h, w = r.orig_shape  # (height, width)
    
    bbox = None
    if r.boxes is not None and len(r.boxes.xyxy) > 0:
        b = r.boxes.xyxy[0].tolist()
        bbox = [float(x) for x in b]
    
    return {
        'keypoints_2d': kpts.tolist(),  # 17 × [x, y, conf]
        'bbox': bbox,
        'image_width': int(w),
        'image_height': int(h),
        'inference_ms': inference_ms,
        'model': 'yolo11x-pose',
    }
```

**Supabase writer**:
```python
# python/yolo/supabase_writer.py
# Mirror python/sam3d/supabase_writer.py — raw httpx, no supabase-py SDK
# Writes yolo_keypoints_2d, yolo_model, yolo_inference_ms columns
# Uses sb_secret_* service-role key (env: SUPABASE_SERVICE_ROLE_KEY)
```

**Frontend disc builder change**:
```typescript
// sparsePhaseOverlay.ts: prefer YOLO over SAM
function getShoulderPair(row: PoseRow): { 
  lx: number, ly: number, rx: number, ry: number 
} | null {
  // PATH 0: YOLO (preferred)
  if (row.yolo_keypoints_2d && row.yolo_keypoints_2d.length === 17) {
    const ls = row.yolo_keypoints_2d[5];  // COCO_KP.LEFT_SHOULDER
    const rs = row.yolo_keypoints_2d[6];  // COCO_KP.RIGHT_SHOULDER
    if (ls && rs && ls[2] > 0.3 && rs[2] > 0.3) {
      return { lx: ls[0], ly: ls[1], rx: rs[0], ry: rs[1] };
    }
  }
  // PATH 1: SAM materialized columns (legacy fallback)
  if (row.shoulder_left_x !== null && row.shoulder_right_x !== null) {
    return {
      lx: row.shoulder_left_x, ly: row.shoulder_left_y!,
      rx: row.shoulder_right_x, ry: row.shoulder_right_y!,
    };
  }
  return null;
}
```

### Commit 拆分（建议 6 个）

1. `feat(schema): add yolo_keypoints_2d columns to pose_3d_phases`
2. `feat(python): add yolo inference module + keypoints constants`
3. `feat(python): add yolo supabase writer (mirror sam3d pattern)`
4. `feat(python): integrate yolo into orchestrator (parallel to sam3d)`
5. `feat(frontend): COCO_KP constants + PoseRow yolo fields + builder preference`
6. `feat(frontend): SwingPlayer badge supports 'yolo' source`

---

## Step 4: 测试

### 自测：本地 Python smoke test

写一个 scripts/yolo_test.py（不 commit），加载 `test_finish.png`（PR-2B 测试图，已 gitignore），调 YOLO，输出 17 个 keypoints 坐标 + 视觉对比 (acromion vs SAM kp 7)。

### 端到端测试

1. 在 swingcue.ai 重新上传一个 golf 视频
2. Railway logs 应该看到 `[yolo] inference completed in Xms`
3. SQL 验证：
   ```sql
   SELECT phase_name, yolo_model, yolo_inference_ms, 
          jsonb_array_length(yolo_keypoints_2d) AS kp_count,
          yolo_keypoints_2d->5 AS left_shoulder_kp,
          yolo_keypoints_2d->11 AS left_hip_kp
   FROM pose_3d_phases 
   WHERE video_id = '<new_video_id>'
   ORDER BY phase_name;
   ```
4. 前端 result page badge 应该显示 **"YOLO 11x"** (green)
5. 5 个 phase tab disc 视觉对位：肩盘在真肩部，胯盘在真髋部

**STOP 输出交付物报告**

---

## 验收标准

- ✅ Schema 加了 3 列，旧数据不动
- ✅ Python YOLO pipeline 跑通，写入 17 keypoints
- ✅ 前端 cascade 优先用 YOLO，fallback SAM，再 fallback dense
- ✅ Disc 中心精准对位**真正的肩峰 / 髋关节**（不是胸部 / 腹部）
- ✅ Badge "YOLO 11x" 显示
- ✅ Railway 推理 < 500ms/frame（CPU）or < 50ms/frame（GPU）

---

## 注意事项

1. **不要删 SAM 3D Body 代码**：保留作为 3D mesh 数据源 + 视觉 fallback
2. **不要破坏 PR-2A schema**：只 ADD COLUMN，不 ALTER 不 DROP
3. **Ultralytics 模型权重**：首次推理时 auto-download ~60MB (yolo11x-pose.pt)。Railway 容器**有写权限**吗？测试或预下载到 image
4. **license**：Ultralytics YOLO 是 AGPL-3.0。如果 SwingCue 商业部署，需要考虑 Enterprise License 或换 YOLO-NAS / RTMPose（Apache 2.0）。先用 YOLO 做 MVP，license 问题以后处理

---

开始 Step 1 探索，完成后 STOP 等审批。
