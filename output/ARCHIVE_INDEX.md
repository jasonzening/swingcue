# PR-6.0 姿态估计模型对比测试 — 结果归档清单
# 生成时间: 2026-06-02
# 工作目录: ~/projects/swingcue-postest/output/

## 测试概况

| 模型 | 视频 | 帧数 | FPS | 检出率 | 产出文件 |
|------|------|------|-----|--------|----------|
| RTMPose-x | test-faceon | 139 | 17.9 | 100% | rtmpose/test-faceon_* |
| RTMPose-x | test-dwontheline | 120 | 19.4 | 100% | rtmpose/test-dwontheline_* |
| RTMW3D-x | test-faceon | 139 | 24.6 | 100% | rtmw3d/test-faceon_* |
| RTMW3D-x | test-dwontheline | 120 | 26.1 | 100% | rtmw3d/test-dwontheline_* |
| ViTPose-L | test-faceon | 139 | 17.5 | 100% | vitpose/test-faceon_* |
| ViTPose-L | test-dwontheline | 120 | 18.1 | 100% | vitpose/test-dwontheline_* |
| TAR-ViTPose | — | — | 跳过 | — | tarvitpose_blocked.txt |

## 文件清单

### rtmpose/
- test-faceon_rtmpose.mp4        骨架标注视频 (COCO-17, 2D)
- test-faceon_keypoints.json     原始关键点 (139帧 x 17关节 x x/y/score)
- test-dwontheline_rtmpose.mp4
- test-dwontheline_keypoints.json
- nvidia_smi.txt                 GPU使用证明 (72% util, 89W, 模型前后对比)
- report.txt                     FPS/延迟报告

### rtmw3d/
- test-faceon_rtmw3d.mp4         骨架标注视频 (主画面 + 右侧俯视3D面板)
- test-faceon_keypoints3d.json   原始关键点 (139帧 x 17关节 x x/y/z/score)
- test-dwontheline_rtmw3d.mp4
- test-dwontheline_keypoints3d.json
- nvidia_smi.txt                 GPU使用证明 (57% util, 70W)
- report.txt

### vitpose/
- test-faceon_vitpose.mp4        骨架标注视频 (COCO-17, 2D, 青色配色)
- test-faceon_keypoints.json     原始关键点 (139帧 x 17关节)
- test-dwontheline_vitpose.mp4
- test-dwontheline_keypoints.json
- nvidia_smi.txt                 GPU使用证明 (71% util, 92W)
- report.txt

### 根目录日志
- rtmpose_gpu_run.log            RTMPose完整推理日志
- rtmw3d_run.log                 RTMW3D完整推理日志
- vitpose_run.log                ViTPose完整推理日志
- tarvitpose_blocked.txt         TAR-ViTPose跳过原因记录

## 关节抖动对比 (帧间漂移, 像素/帧)

### RTMPose-x 正面
  left_shoulder:  mean=4.1  p95=17.3  max=36.1
  right_shoulder: mean=3.6  p95=16.6  max=35.6
  left_elbow:     mean=6.9  p95=26.9  max=56.4
  right_elbow:    mean=6.1  p95=30.7  max=46.4
  left_wrist:     mean=11.1 p95=49.4  max=68.5
  right_wrist:    mean=11.3 p95=55.6  max=69.4
  left_hip:       mean=2.4  p95=7.1   max=17.9
  right_hip:      mean=2.0  p95=5.5   max=13.9

### RTMPose-x 侧面
  left_shoulder:  mean=3.7  p95=18.1  max=37.5
  right_shoulder: mean=3.8  p95=12.6  max=19.4
  left_elbow:     mean=7.2  p95=31.8  max=40.2
  right_elbow:    mean=7.7  p95=24.6  max=65.4
  left_wrist:     mean=10.8 p95=48.5  max=69.0
  right_wrist:    mean=11.1 p95=51.0  max=67.7
  left_hip:       mean=3.0  p95=10.5  max=20.7
  right_hip:      mean=2.5  p95=8.9   max=15.5

### ViTPose-L 正面
  left_shoulder:  mean=4.6  p95=17.7  max=49.9
  right_shoulder: mean=4.2  p95=19.4  max=46.4
  left_elbow:     mean=8.9  p95=32.4  max=75.4
  right_elbow:    mean=7.2  p95=32.2  max=51.6
  left_wrist:     mean=11.7 p95=46.1  max=92.1
  right_wrist:    mean=11.8 p95=60.1  max=71.0
  left_hip:       mean=2.9  p95=8.6   max=23.2
  right_hip:      mean=2.3  p95=6.3   max=16.2

### ViTPose-L 侧面
  left_shoulder:  mean=4.2  p95=14.9  max=41.4
  right_shoulder: mean=4.2  p95=12.4  max=18.8
  left_elbow:     mean=8.3  p95=36.6  max=52.2
  right_elbow:    mean=9.9  p95=36.6  max=69.6
  left_wrist:     mean=11.3 p95=46.5  max=122.3  ← ViTPose侧面腕部最不稳
  right_wrist:    mean=11.7 p95=47.6  max=120.3
  left_hip:       mean=3.1  p95=11.1  max=38.0
  right_hip:      mean=2.1  p95=8.5   max=12.8

## 关键结论

1. 检出率: 三个模型均 100% (无漏帧)
2. 速度: RTMW3D > RTMPose ≈ ViTPose (均超过视频源帧率,可实时)
3. 稳定性(肘/腕): RTMPose-x > ViTPose-L,差距在遮挡帧尤为明显
4. 侧面击球腕部: ViTPose max达122px,RTMPose仅69px,RTMPose更抗崩
5. RTMW3D: 同等场景下有Z轴深度辅助判断遮挡/漂移原因
6. GPU验证: 三个模型均通过CUDA EP真实GPU推理 (57-72% util)

## 下一步建议 (待Jason决策)
- Option A: RTMPose-x 作为 SwingCue 生产模型 (最稳、无额外依赖)
- Option B: RTMW3D 作为生产模型 (3D深度辅助修正 + 更快)
- Option C: 安装 CUDA Toolkit 后补测 TAR-ViTPose (apt install cuda-toolkit-12-4)
- Option D: 在更长视频或实际球场视频上扩大样本量

## 环境记录
  Python:    3.12.3
  PyTorch:   2.6.0+cu124
  ORT:       1.26.0 (CUDAExecutionProvider)
  rtmlib:    0.0.15
  GPU:       RTX 4060 Ti 8GB
  OS:        WSL2 Ubuntu
  关键坑:   LD_LIBRARY_PATH 需手动加 .venv/nvidia/*/lib (见 run_with_gpu.sh)
