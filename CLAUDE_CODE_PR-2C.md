# SwingCue PR-2C: 前端 keypointOverlays 重写 — Claude Code Task

## Context

PR-2B 已合并：Railway analyzer 现在为每个上传视频生成 5 个相位的精确 SAM 3D Body 数据，存到 `pose_3d_phases` 表。

**已验证**（最新测试视频 `b4f7e911-3081-4e81-b2ad-12e82db5970b`）：
- 5/5 phase fal_status='completed'
- 70 个 keypoints_2d + keypoints_3d 完整存储（jsonb）
- 物化锚点：shoulder_left_x/y, shoulder_right_x/y, hip_left_x/y, hip_right_x/y
- 源图尺寸：image_width × image_height（用于前端 canvas 坐标变换）

**本轮你要做**：把前端 disc overlay 渲染从 MediaPipe-based 切换到 SAM 3D Body-based。

**禁止做的**：PR-2D 范围（完美挥杆比对、3D mesh 显示）不在本轮，发现就停下来问。

---

## 工作模式（与 PR-2B 一致）

1. **探索先行**：在动代码前先读现有 `keypointOverlays.ts`、disc 渲染入口、phase tab 组件，输出结构报告给用户审批后再动手。
2. **Additive 优先于 Destructive**：如果旧代码（MediaPipe inference、Phase 7.x 后处理）仍然在跑某些回退路径，先添加新路径，确认新路径稳定后再删除旧代码。
3. **每个 commit 独立可读**：拆 4-6 个原子 commit。
4. **测试方法**：用测试视频 `b4f7e911-3081-4e81-b2ad-12e82db5970b` 在 swingcue.ai 验证新 overlay 渲染。
5. **STOP 节点**：探索完输出报告 → STOP；设计方案 → STOP；实现完成 → STOP。3 个审批节点。

---

## PR-2C 范围

### 删除（destructive）
- `src/lib/overlay/keypointOverlays.ts` 中的 Phase 7.x 全部几何后处理：
  - `_prevAngle`, `_refWidth`, `_prevZAsym` 等可变状态
  - EMA (exponential moving average) 平滑逻辑
  - slew rate limiting
  - refW lock (参考宽度锁定)
  - cyShiftFactor (cy shift 补偿)
  - 所有"phase 7.x"为名的修复逻辑
- 浏览器端 MediaPipe 推理代码（如果还在跑）
- 任何依赖 frame-by-frame MediaPipe 输出的几何计算

### 新增（additive）
- `src/lib/sam3d/keypoints.ts` — keypoint 索引常量（与 Python 端 `python/sam3d/keypoints.py` 对应，**值必须一致**）
- `src/lib/sam3d/pose-fetch.ts` — 从 Supabase `pose_3d_phases` 表读取 5 个 phase 数据的服务层
- `src/lib/sam3d/coords.ts` — 源图像像素坐标 → canvas 显示坐标的变换函数
- 在 disc rendering 入口处使用 PoseRow 替代 MediaPipe-derived keypoints

### 简化（refactor）
- `buildDisc()` 函数（或类似的肩盘/胯盘构建函数）—— 输入从"复杂关键点 + 后处理状态"变成"4 个锚点坐标"，预期代码量减少 60%+

---

## Step 1: 探索

```bash
# 1. 找前端 overlay 渲染入口
find src -type f \( -name "*.ts" -o -name "*.tsx" \) | xargs grep -l "keypointOverlay\|buildDisc\|shoulder\|disc.*phase" 2>/dev/null | head -20

# 2. 找 Phase 7.x 相关代码
grep -rn "phase.*7\|_prevAngle\|_refWidth\|_prevZAsym\|cyShiftFactor\|refWLock\|slewRate" --include="*.ts" --include="*.tsx" src/ | head -30

# 3. 看 phase tab 组件结构
find src -type f -name "*.tsx" | xargs grep -l "setup\|top\|transition\|impact\|finish" 2>/dev/null | head -10

# 4. 看现有 Supabase client 用法
cat src/lib/supabase/server.ts
cat src/lib/supabase/client.ts 2>/dev/null

# 5. 看 result page 入口
find src/app -type f -name "page.tsx" | xargs grep -l "result" | head -5

# 6. 看 MediaPipe 是否在前端用
grep -rn "mediapipe\|MediaPipe\|@mediapipe" --include="*.ts" --include="*.tsx" --include="*.json" src/ package.json | head -10
```

**输出探索报告**（必须）：
1. **disc overlay 入口文件路径**（哪个 component 调 keypointOverlays?）
2. **当前数据流**（MediaPipe 输出怎么流到 disc 渲染？哪一步做几何后处理？）
3. **Phase 7.x 后处理位置**（具体哪些函数/状态变量）
4. **MediaPipe 是否还在 browser 跑**（package.json 有依赖？）
5. **phase tab UI 结构**（5 个 phase 切换时数据流如何更新）
6. **现有 Supabase client 写法**（user-scoped vs service-role）

**STOP，输出探索报告后等用户审批**

---

## Step 2: 设计方案

基于探索结果，写出：

1. **新文件清单 + 职责**
2. **修改文件清单 + 改动范围（行号粒度）**
3. **数据流图**（5 句话内）：
   ```
   result page → fetch pose_3d_phases (5 rows) → coords transform 
   → buildDisc(shoulder pair) → render SVG/canvas
   ```
4. **删除清单**：明确哪些函数/变量/import 会被删除
5. **回退策略**：如果 pose_3d_phases 没数据（旧视频还在用 MediaPipe），是否需要 fallback？
6. **Commit 拆分计划**（4-6 个）

**STOP，输出设计方案后等用户审批**

---

## Step 3: 实现

按设计 commit 顺序实现。每个 commit 完成后：
- `git diff --stat HEAD~1` 自检改动量
- 不立刻 push

所有 commit 完成后统一 push 一次。

---

## Step 4: 测试

测试视频 ID 用：`b4f7e911-3081-4e81-b2ad-12e82db5970b`（**注意不是 eec305a5**，前者才有 pose_3d_phases 数据）。

在 swingcue.ai 找到这个 video 对应的 result page。打开后：
- 5 个 phase tab 都能切换
- 每个 phase 显示精确的肩盘 + 胯盘 disc
- disc 中心对位准确（特别是 finish 帧不再有"disc 消失"问题）
- disc 半径对位身体宽度

**STOP，输出测试结果 + 截图反馈**

---

## 关键参考数据

### Keypoint 索引（必须与 Python 端 `python/sam3d/keypoints.py` 一致）

```typescript
// src/lib/sam3d/keypoints.ts
// SAM 3D Body MHR keypoint indices (verified 2026-05-16)
// Mirrors python/sam3d/keypoints.py — keep values in sync.

export const SAM3D_KP = {
  // Head
  HEAD_TOP: 1,
  NOSE: 0,
  CHIN: 5,
  NECK_BASE: 6,
  
  // Shoulders (Z-disambiguated in finish frame)
  LEFT_SHOULDER: 7,   // acromion, target-side (Z+ in finish test)
  RIGHT_SHOULDER: 8,  // acromion, trail-side  (Z- in finish test)
  
  // Hips
  LEFT_HIP: 9,
  RIGHT_HIP: 10,
  
  // Knees, ankles, feet
  LEFT_KNEE: 11,
  RIGHT_KNEE: 12,
  LEFT_ANKLE: 13,
  RIGHT_ANKLE: 14,
  LEFT_TOE: 15,
  LEFT_TOE_OUTER: 16,
  LEFT_HEEL: 17,
  RIGHT_TOE: 18,
  RIGHT_TOE_OUTER: 19,
  RIGHT_HEEL: 20,
  
  // Wrists (assumed = hand cluster origins; validate later)
  LEFT_WRIST: 21,
  RIGHT_WRIST: 42,
  
  // Extra detail
  LEFT_DELTOID: 63,
  RIGHT_DELTOID: 64,
  LEFT_CLAVICLE: 65,
  RIGHT_CLAVICLE: 66,
  NECK: 67,
  STERNUM: 68,
  THROAT: 69,
} as const;

export type PoseRow = {
  phase_name: 'setup' | 'top' | 'transition' | 'impact' | 'finish';
  fal_status: 'uploaded' | 'processing' | 'completed' | 'failed';
  frame_idx: number;
  frame_timestamp_ms: number | null;
  keypoints_2d: number[][];  // 70 × [x, y]
  keypoints_3d: number[][];  // 70 × [x, y, z]
  focal_length: number;
  bbox: [number, number, number, number] | null;
  mhr_params: Record<string, unknown> | null;
  glb_url: string | null;
  image_width: number;
  image_height: number;
  shoulder_left_x: number | null;
  shoulder_left_y: number | null;
  shoulder_right_x: number | null;
  shoulder_right_y: number | null;
  hip_left_x: number | null;
  hip_left_y: number | null;
  hip_right_x: number | null;
  hip_right_y: number | null;
};
```

### Supabase 读取（建议用 user-scoped client，遵守 RLS）

```typescript
// src/lib/sam3d/pose-fetch.ts
import { createClient } from '@/lib/supabase/server';  // adjust path to existing user-scoped client

export async function fetchPoseRows(videoId: string): Promise<PoseRow[]> {
  const supabase = await createClient();
  const { data, error } = await supabase
    .from('pose_3d_phases')
    .select('*')
    .eq('video_id', videoId)
    .order('phase_name');  // or custom ORDER BY phase sequence
  
  if (error) {
    console.error('[pose-fetch] error:', error);
    return [];
  }
  return (data ?? []) as PoseRow[];
}
```

### 坐标变换

Pose data 里的 `shoulder_left_x` 等都是**源图像像素坐标**（图像尺寸 = `image_width × image_height`）。
前端 canvas/SVG 尺寸通常不一样（受响应式布局影响）。需要变换：

```typescript
// src/lib/sam3d/coords.ts
export function srcPxToCanvas(
  srcX: number, srcY: number,
  srcW: number, srcH: number,
  canvasW: number, canvasH: number,
  fit: 'contain' | 'cover' = 'contain'
): { x: number; y: number } {
  // Compute aspect-preserving scale + letterbox offsets
  const scaleX = canvasW / srcW;
  const scaleY = canvasH / srcH;
  const scale = fit === 'contain' ? Math.min(scaleX, scaleY) : Math.max(scaleX, scaleY);
  
  const scaledW = srcW * scale;
  const scaledH = srcH * scale;
  const offsetX = (canvasW - scaledW) / 2;
  const offsetY = (canvasH - scaledH) / 2;
  
  return {
    x: offsetX + srcX * scale,
    y: offsetY + srcY * scale,
  };
}
```

### Disc 构建（简化版）

```typescript
// src/lib/overlay/buildDisc.ts (replaces complex MediaPipe-based version)
export type DiscParams = {
  centerX: number;
  centerY: number;
  radius: number;
};

export function buildShoulderDisc(pose: PoseRow): DiscParams | null {
  if (
    pose.shoulder_left_x === null || pose.shoulder_left_y === null ||
    pose.shoulder_right_x === null || pose.shoulder_right_y === null
  ) return null;
  
  const cx = (pose.shoulder_left_x + pose.shoulder_right_x) / 2;
  const cy = (pose.shoulder_left_y + pose.shoulder_right_y) / 2;
  const dx = pose.shoulder_right_x - pose.shoulder_left_x;
  const dy = pose.shoulder_right_y - pose.shoulder_left_y;
  const radius = Math.hypot(dx, dy) / 2;
  
  return { centerX: cx, centerY: cy, radius };
}

export function buildHipDisc(pose: PoseRow): DiscParams | null {
  // same pattern with hip_left_x/y + hip_right_x/y
}
```

注意：返回的 `centerX/Y/radius` 还是源图像像素坐标。canvas 渲染前再调 `srcPxToCanvas` 变换。

---

## 验收标准

PR-2C 完成的标志：
1. ✅ 5 个 phase tab 都能切换，disc 渲染稳定
2. ✅ Finish 帧 disc 不再消失
3. ✅ 没有抖动（之前是 MediaPipe frame-by-frame 推理 + EMA 平滑后还残留的抖动）
4. ✅ disc 中心精准对位 acromion（不再是腋窝）
5. ✅ 代码量：keypointOverlays.ts 净删除 200+ 行
6. ✅ `package.json` 移除 MediaPipe 相关依赖（如果只 frontend 用）—— 如果 unsure，先保留
7. ✅ 老视频回退路径（如有）记录在 docs/decisions/ 里

---

## 开始执行

现在开始 **Step 1 探索**。完成后 STOP 等审批。
