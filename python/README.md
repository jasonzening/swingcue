# SwingCue Analysis Service

Python + MediaPipe 视频分析后端。

## 本地开发

```bash
cd python/
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

测试：
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://your-supabase-url/storage/v1/object/sign/swing-videos/...",
    "view_type": "face_on"
  }'
```

## Railway 部署

1. 登录 [railway.app](https://railway.app)
2. New Project → Deploy from GitHub repo
3. 选择 `swingcue` repo，**设置 Root Directory 为 `/python`**
4. 环境变量：不需要额外变量（分析服务是无状态的）
5. Deploy → 获得 URL，例如 `https://swingcue-analyzer.up.railway.app`

## 环境变量（在 Next.js Vercel 项目里设置）

```
PYTHON_ANALYZER_URL=https://swingcue-analyzer.up.railway.app
```

## API 接口

### POST /analyze

请求：
```json
{
  "video_url": "string",    // Supabase Storage signed URL
  "view_type": "face_on",   // or "down_the_line"
  "club_type": "iron",      // or "driver" / "unknown"
  "sample_fps": 4.0         // frames per second to sample
}
```

响应：
```json
{
  "status": "success",
  "videoMetadata": {
    "durationSec": 2.87,
    "fps": 29.97,
    "width": 1080,
    "height": 1920
  },
  "phaseMarkers": {
    "setupTime": 0.05,
    "topTime": 1.42,
    "transitionTime": 1.78,
    "impactTime": 2.13,
    "finishTime": 2.58
  },
  "keypointTimeline": [
    {
      "time": 0.0,
      "landmarks": {
        "head": { "x": 0.512, "y": 0.123, "confidence": 0.98 },
        "leftShoulder": { "x": 0.623, "y": 0.278, "confidence": 0.96 },
        ...
      }
    }
  ],
  "issueDetection": {
    "issue": "early_extension",
    "confidence": 0.72,
    "metrics": {
      "hip_rise": 0.045,
      "head_x_range": 0.031
    }
  }
}
```

## 性能参数

| 视频时长 | 采样帧数(4fps) | 分析耗时 |
|--------|-------------|---------|
| 3s     | ~12 frames  | ~4-6s   |
| 5s     | ~20 frames  | ~6-9s   |
| 10s    | ~40 frames  | ~12-18s |

Railway Free tier 足够 MVP 测试。
