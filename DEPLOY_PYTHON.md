# SwingCue Python 服务部署指南

## Railway 部署（推荐，15分钟）

### 步骤

1. **登录 Railway**
   - 打开 https://railway.app
   - 用 GitHub 账号登录

2. **新建项目**
   - 点击 "New Project"
   - 选择 "Deploy from GitHub repo"
   - 选择 `jasonzening/swingcue` 仓库

3. **配置 Root Directory**
   - 在部署设置里找到 "Root Directory"
   - 设置为：`python`
   - Railway 会自动检测到 Dockerfile

4. **部署**
   - 点击 Deploy
   - 等待 3-5 分钟（首次需要构建 Docker 镜像）

5. **获取 URL**
   - 部署成功后，在 Railway dashboard 找到 "Domains"
   - 生成一个 URL，格式类似：
     `https://swingcue-analyzer-production.up.railway.app`

6. **配置 Vercel 环境变量**
   - 打开 https://vercel.com/dashboard
   - 找到 swingcue 项目 → Settings → Environment Variables
   - 添加：`PYTHON_ANALYZER_URL = https://你的railway域名`
   - 重新 Deploy（或自动触发）

### 验证部署成功

```bash
# 替换为你的 Railway URL
curl https://swingcue-analyzer-production.up.railway.app/health
# 返回: {"status": "ok", "service": "swingcue-analyzer"}
```

### 测试分析接口

```bash
curl -X POST https://swingcue-analyzer-production.up.railway.app/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "你的Supabase signed URL",
    "view_type": "face_on"
  }'
```

## 验收清单

上传一个视频后，查看 Supabase swing_analysis 表：

- `video_metadata_json.durationSec` = 真实视频时长（非固定值）
- `video_metadata_json.fps` = 真实帧率
- `video_metadata_json.dataSource` = "mediapipe"（不是"stub"）
- `phase_markers_json.topTime` = 真实腕部轨迹检测到的值
- `keypoint_timeline_json.frames` 数组有数据（不为 null）

