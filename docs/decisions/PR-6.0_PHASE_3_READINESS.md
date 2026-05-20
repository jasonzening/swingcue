# PR-6.0 Phase 3 Readiness — GPU Infrastructure & Model Selection

> Researched 2026-05-19 alongside PR-6.0 spec, before Phase 1 kicked off.
> Captures model license + compute cost analysis so Phase 3 can start without re-researching.

## TL;DR

| 模型 | License | Commercial OK | 计算需求 | 推荐 |
|---|---|---|---|---|
| **RTMPose** (MMPose) | Apache-2.0 | ✅ | CPU OK (75.8 AP, 90 FPS on i7) | 🥇 主候选 |
| **ViTPose++** (ViTAE) | Apache-2.0 | ✅ | GPU 必需 (80.9 AP, transformer) | 🥈 SOTA 候选 |
| **Sapiens2** (Meta) | Meta custom (Llama-style, <700M MAU) | ✅ | GPU 必需 (1K resolution, 0.4B-5B) | 🥉 if ViTPose 不够 |
| ~~Sapiens v1~~ | cc-by-nc-4.0 | ❌ | - | **跳过 (non-commercial)** |
| WHAM (temporal) | MIT | ✅ | GPU 必需 (4D temporal) | 选择性 (如果 crossover 仍是问题) |

## Accuracy Reference (COCO test-dev AP)

```
ViTPose-G (1B params)        80.9 AP   ← SOTA on COCO
Sapiens2-1B                  ~80+ AP   ← +7.6 mAP over prior SOTA on Humans-5K
RTMPose-m                    75.8 AP   ← Industry sweet spot
RTMPose-l (COCO-Wholebody)   67.0 AP (lower b/c whole-body harder)
MediaPipe Pose (现状)        ~65 AP    ← 当前 baseline
YOLO11x-pose                 ~70 AP    ← 中等
```

实际医学场景测试（newborn study, PMC12971853, Apr 2025）排序：
**RTMPose > ViTPose > Sapiens > PCT > MediaPipe > OpenPose**

## Phase 3 Compute Cost Estimate

测试场景：3 个 test video × 4 个 GPU 模型 × ~3s 每个视频

每模型 inference 时间估算（A10G）：
- ViTPose++: ~1-2s per frame → 90 frames × 1.5s = 135s 每视频
- Sapiens2: ~2-3s per frame → 90 × 2.5 = 225s 每视频
- RTMPose-l (GPU mode): ~50ms per frame → 90 × 0.05 = 4.5s 每视频
- WHAM: ~5s per video clip (temporal model)

**Phase 3 总计算时间 ≈ 20-30 分钟 GPU 时间**

### GPU Provider 对比 (May 2026)

| Provider | A10G | H100 | 计费模式 | 适合场景 |
|---|---|---|---|---|
| **Lightning AI** | $0.50/hr | $2.99/hr | 持久 Studio + GPU 按需 attach | 🥇 Phase 3 推荐 — 持久环境调试方便 |
| **Modal** | $1.10/hr | $3.95/hr | 完全 serverless, 按秒计费 | Phase 4 生产 inference (sporadic) |
| **RunPod** | $0.69/hr | - | 按需 pod | 实验便宜，UX 差 |
| **Colab Pro** | T4 included | - | $10/月固定 | 一次性实验，session timeout 烦 |

### Phase 3 总成本

- Lightning AI A10G × 30 分钟 = **$0.25**
- 即使跑 H100 × 30 分钟 = **$1.50**

**Phase 3 整个 benchmark 花费 < $5** (包括反复实验)

## Phase 4 生产部署成本估算

假设 SwingCue 起步 ~100 swings/day：

### Option A: 赢家是 RTMPose (CPU-friendly)
- 留在 Railway 跑 ONNX runtime CPU
- **额外成本: $0** (Railway 现有 plan 足够)

### Option B: 赢家是 ViTPose / Sapiens2 (GPU-required)
- Modal serverless A10G: 每 swing ~30s = $0.009/swing
- 100 swings/day = $0.92/day = **$28/月**
- 持久 GPU (Lightning Pro $50/月 + A10G 24/7): **$410/月**
- Modal serverless 是赢家因为利用率低

### Option C: 增加 GPU 后端，CPU 后端兜底
- 默认 RTMPose CPU on Railway
- 用户付费/Pro 用户路由到 ViTPose Modal GPU
- 复杂度 +1, 但产品力 +1

## 推荐路径

1. **Phase 3 跑全部 4 个 GPU 模型** (Lightning AI A10G, ~$1)
2. **如果 RTMPose 在视觉测试上 already 显著优于 MediaPipe** → Phase 4 集成 RTMPose 到 Railway CPU
3. **如果需要 ViTPose / Sapiens2 才能解决 crossover** → 评估 Option C
4. **如果 WHAM (temporal) 是唯一解** → 重新考虑架构 (4D 模型部署复杂度高)

## Sources

- RTMPose: arxiv:2303.07399 (Mar 2023)
- ViTPose / ViTPose++: arxiv:2204.12484 / arxiv:2212.04246 (NeurIPS'22 / TPAMI'23)
- Sapiens: facebookresearch/sapiens (ECCV 2024 Oral, cc-by-nc-4.0)
- Sapiens2: facebookresearch/sapiens2 (Apr 2026, Meta custom license)
- Newborn study (real-world benchmark): pmc.ncbi.nlm.nih.gov/articles/PMC12971853
- GPU pricing: Modal docs, Lightning AI pricing page, gputracker.dev (May 2026)
