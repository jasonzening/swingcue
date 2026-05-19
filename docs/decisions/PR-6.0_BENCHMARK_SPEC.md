# PR-6.0 — Pose Model Benchmark (parallel to PR-5.9)

**Date**: 2026-05-19  
**Status**: Plan only · No benchmark runs yet  
**Owner**: TBD · Can run in parallel with PR-5.9 implementation  
**Relationship to PR-5.9**: PR-5.9 fixes the current MediaPipe pipeline (fps + smoothing + interpolation). PR-6.0 evaluates whether MediaPipe + YOLO is the right base architecture at all, or whether to switch / fuse with a stronger model.

---

## 1. Why this benchmark exists

PR-5.9 will fix lag, smoothing, and interpolation on top of the current MediaPipe + YOLO pipeline. But these fixes don't address:

- Self-occlusion (right shoulder hidden by left arm at top of backswing)
- Left/right keypoint identity swap when body rotates past 90°
- Pose model confidence collapsing on motion-blurred frames
- Inherent accuracy ceiling of MediaPipe Pose's training data (everyday motion, not athletic swings)

**Hypothesis**: even after PR-5.9 smoothing improvements, MediaPipe's per-frame pose may not be accurate enough on raw athletic motion to support SwingCue's core promise. A stronger model (or fusion) may be required.

This benchmark provides ground truth before committing to a model architecture for the next 6+ months.

---

## 2. Acceptance criteria (visual, not numerical)

Pass = on b3fea3f0 test video, the model output (raw, no smoothing) keeps shoulder and hip keypoints at the SwingCue coaching anchor positions (as defined in PR-5.8A) through all 5 phases:

| Phase | Acceptance |
|---|---|
| Setup | Shoulder/hip dots on body anchors (PR-5.8A baseline) |
| Top | Shoulder/hip dots on body, even with arm-over-shoulder occlusion |
| Transition | No keypoint jump or 100ms+ position discontinuity |
| Impact | Shoulder/hip dots track through fast motion |
| Finish | Left/right identity correctly assigned despite body rotation |

Models that pass on setup but fail at top/transition/impact are **not acceptable** — that's the current MediaPipe baseline.

Numerical metrics (recorded but not gating):
- Per-keypoint confidence distribution per phase
- Frame-level keypoint drop rate (confidence < 0.5)
- Inference time / frame
- Memory cost
- Left/right swap event count

---

## 3. Models to evaluate

### Tier A — drop-in replacements (low integration cost)

| Model | Origin | Why test |
|---|---|---|
| **MediaPipe Pose Landmarker (current)** | Google, legacy `mp.solutions.pose` | Baseline, model_complexity=1 |
| **MediaPipe Pose Landmarker Tasks API** | Google, new API | Same model, better inference pipeline, GPU acceleration |
| **MediaPipe Holistic** | Google | Includes face + hands; possibly cleaner ear/mouth/hand landmarks |
| **MoveNet Thunder** | Google, TF Lite | Trained specifically on fitness/sports motion |
| **YOLO11x-pose per-frame** | Ultralytics, already in pipeline | Currently per-phase only — what does per-frame look like? |
| **YOLOv8x-pose** | Ultralytics | Alternative YOLO variant, different speed/accuracy tradeoff |

### Tier B — research-grade (higher integration cost, GPU may be needed)

| Model | Origin | Why test |
|---|---|---|
| **RTMPose** | MMPose / OpenMMLab | Open-source SOTA, real-time variants exist |
| **ViTPose** | OpenMMLab | Transformer-based, robust to occlusion |
| **Sapiens** | Meta | Human-centric foundation model, 2D + 3D variants |
| **WHAM** | ETH | 3D-aware with motion model, robust to fast motion |

Tier B requires GPU inference. Acceptable for benchmark; production deployment is separate decision.

### Out of scope (this benchmark)
- Custom-trained models on SwingCue data (no labeled data yet)
- Closed-source competitors (Sportsbox API, etc.) — separate ecosystem evaluation
- 3D-only models (Sapiens-3D, WHAM 3D output) — useful but not directly comparable on 2D overlay

---

## 4. Methodology

### 4.1 Common test input
- Video: `b3fea3f0-e248-44d7-a923-0bb43172b5bf` (face-on, real user, currently the visual ground truth)
- Add later: 2 more videos covering down-the-line and behind-the-line angles
- All frames at native fps (no downsampling) — feeds raw frames to each model

### 4.2 Per-model run
1. Extract every frame at native fps into a directory
2. Run each model frame-by-frame, no smoothing, no temporal context (unless model has built-in temporal mode — note when used)
3. Output: JSON file per model, structure `{ frames: [{ ts, keypoints: { ... }, raw_confidence: { ... }, inference_ms }] }`
4. Map each model's native output to a common 17-keypoint schema for comparison (mapping table in §6 of this doc)

### 4.3 Visual evaluation
Frontend benchmark page (separate route, e.g. `/benchmark/[videoId]`):
- 5 columns × 5 rows = 5 phases × top 5 models
- Each cell: video frame with that model's keypoints overlaid (PR-5.8A expansion applied for fair comparison)
- Jason scores each cell pass/fail/marginal
- Best 2-3 models proceed to full-swing evaluation

### 4.4 Quantitative analysis
Python script generates summary table:
- Average confidence per keypoint per phase
- % of frames with at least one missing keypoint
- Left/right swap detection: cross-correlate left_shoulder.x with right_shoulder.x across frames; flag swaps
- Inference time distribution
- GPU vs CPU runtime if applicable

---

## 5. Infrastructure

### 5.1 Compute
- Tier A models: laptop CPU sufficient (MediaPipe, MoveNet, YOLO are tested to run real-time on consumer hardware)
- Tier B models: rent cloud GPU for benchmark only (Lambda Labs / RunPod, ~$5-20 total cost for full benchmark)

### 5.2 Storage
- Raw frames: ~150 frames × ~200KB = ~30MB per video, 90MB total for 3 videos
- Model output JSONs: small (<1MB per model per video)
- Visual comparison page can pull from local dev server

### 5.3 Code location
- Python benchmark scripts: `python/benchmark/` (new directory)
- Frontend comparison: `src/app/benchmark/[videoId]/page.tsx` (new route, dev-only, not linked from main app)
- Shared keypoint mapping: `python/benchmark/keypoint_mapping.py`

---

## 6. Keypoint mapping table (each model → SwingCue 17)

| SwingCue 17 (PR-5.8 spec) | MediaPipe 33 | MoveNet 17 | YOLO COCO 17 | RTMPose Body-17 | Notes |
|---|---|---|---|---|---|
| head_crown | derived from 7,8,9,10 | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | Need fallback for non-MediaPipe |
| shoulder_L | 11 | 5 | 5 | 5 | Map straight |
| shoulder_R | 12 | 6 | 6 | 6 | Map straight |
| elbow_L | 13 | 7 | 7 | 7 | Map straight |
| elbow_R | 14 | 8 | 8 | 8 | Map straight |
| wrist_L | 15 | 9 | 9 | 9 | Map straight |
| wrist_R | 16 | 10 | 10 | 10 | Map straight |
| hand_L | 19 | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | MediaPipe-only |
| hand_R | 20 | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | MediaPipe-only |
| hip_L | 23 | 11 | 11 | 11 | Map straight |
| hip_R | 24 | 12 | 12 | 12 | Map straight |
| knee_L | 25 | 13 | 13 | 13 | Map straight |
| knee_R | 26 | 14 | 14 | 14 | Map straight |
| ankle_L | 27 | 15 | 15 | 15 | Map straight |
| ankle_R | 28 | 16 | 16 | 16 | Map straight |
| foot_L | 31 | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | MediaPipe-only |
| foot_R | 32 | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | MediaPipe-only |

For models without head_crown / hands / feet, mark those keypoints as N/A in the benchmark display.

---

## 7. Deliverables

`docs/decisions/PR-6.0_BENCHMARK_RESULTS.md`:

1. Per-model: 5 phase screenshots with keypoint overlay
2. Pros / cons table (accuracy, cost, integration difficulty)
3. Recommendation: which model(s) to adopt for PR-6.x implementation
4. Cost analysis: estimated production inference time, infra needs
5. Fallback strategy: if no single model passes all 5 phases, propose fusion / ensemble approach

---

## 8. Decision gates

After benchmark:

- **If current MediaPipe + PR-5.9 fixes pass acceptance** → no model change needed, close PR-6.0
- **If a Tier A model clearly wins** → PR-6.1 = drop-in replacement
- **If only Tier B passes** → PR-6.2 = stronger model with GPU requirement, evaluate production cost
- **If no single model passes** → PR-6.3 = fusion architecture (per-keypoint best-of, e.g. MediaPipe for face, RTMPose for body, YOLO for occluded frames)

---

## 9. Timeline

- Week 1: Set up benchmark infra + Tier A model integration
- Week 2: Tier A runs + Jason visual review
- Week 3 (if Tier A fails): Tier B integration + GPU rental
- Week 4 (if Tier B fails): fusion architecture design

Total: 2-4 weeks depending on which tier passes.

PR-5.9 runs in parallel for the first 2 weeks; if Tier A wins quickly, PR-5.9 may absorb the model swap.

---

## 10. Prohibitions

- Do NOT start integration during benchmark phase. Benchmark is read-only model evaluation.
- Do NOT publish benchmark page on production. Dev-only route, behind feature flag.
- Do NOT optimize benchmark for any one model's strengths. Same input, same evaluation criteria, all models.
- Do NOT score numerical metrics before visual acceptance. Visual is the gate; numbers are diagnostic.
- Do NOT block PR-5.9 on benchmark completion. PR-5.9 ships independently if MediaPipe + smoothing fixes are acceptable.

---

## 11. Related docs

- `docs/decisions/PR-5.8_GOLF_17_KEYPOINTS.md` — Full SwingCue 17 spec (future work)
- `docs/decisions/PR-5.8A_COACHING_ANCHOR.md` — Render-time anchor expansion (currently in PR)
- `docs/PR-5.8_AUDIT.md` — Current pipeline audit
- `docs/PR-5.9_AUDIT.md` — TBD, pre-PR-5.9 audit
