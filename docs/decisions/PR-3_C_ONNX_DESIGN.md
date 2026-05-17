# PR-3 Option C: ONNX Export Implementation Design

**Status:** Awaiting approval
**Date:** 2026-05-16
**Audit reference:** [`docs/PR-3_AUDIT.md`](../PR-3_AUDIT.md)
**Decision:** Adopt Option C — keep ultralytics + torch in a build-time
stage only, ship `yolo11m-pose.onnx` to a slim runtime stage that uses
`onnxruntime` for inference.

---

## 1. Multi-stage Dockerfile structure

Two stages with hard isolation. Stage 1's `site-packages/` never reaches
stage 2 — only the `.onnx` file is COPYd across. **numpy 2.x and the
3-cv2-variant chaos from stage 1 cannot pollute stage 2 by Docker's
construction.**

```dockerfile
# ─────────────────────────────────────────────────────────────────────
# Stage 1: builder — install ultralytics, export .onnx, smoke-test
# ─────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS yolo-builder

# Minimal apt — only what ultralytics + cv2 import-chains need
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# CPU torch wheel (smaller than GPU; we only need it for the export call)
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

# ultralytics + minimum sibling deps for export + verify
# Note: this stage doesn't have to be ABI-clean for production — it's
# discarded. Whatever opencv tangle pip creates here is fine; we only
# need `.export()` to succeed.
RUN pip install --no-cache-dir ultralytics onnxruntime

# Export yolo11m-pose.pt → yolo11m-pose.onnx with a static 640×640 input.
# nms=False (we will implement argmax-by-confidence in Python; simpler
# than handling baked-NMS output shape variability across ultralytics
# versions). simplify=True for graph constant-folding.
RUN python -c "from ultralytics import YOLO; \
    m = YOLO('yolo11m-pose.pt'); \
    m.export(format='onnx', imgsz=640, simplify=True, nms=False, opset=12)"
# Result: /build/yolo11m-pose.onnx (~80 MB)

# Build-time smoke test: ensure our manual onnxruntime decoder produces
# keypoints within ε of ultralytics' own inference on the same image.
# Uses ultralytics' built-in sample image (no need for repo-side test
# artifact). Build fails if divergence > 5 pixels.
COPY scripts/verify_onnx_export.py /build/verify.py
RUN python /build/verify.py
# (script details in §5)

# ─────────────────────────────────────────────────────────────────────
# Stage 2: runtime — slim image, NO ultralytics, NO torch
# ─────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Apt — same as pre-PR-3 production (proven working)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pull the exported model from the builder. This is the ONLY artifact
# that crosses the stage boundary. Stage 1's PyTorch / numpy 2.x /
# ultralytics / 3-cv2-variant filesystem state is discarded.
COPY --from=yolo-builder /build/yolo11m-pose.onnx /app/yolo11m-pose.onnx

# Stage-2 ABI verify — slim (no ultralytics check)
RUN python -c "import numpy; v=numpy.__version__; \
    assert v.startswith('1.'), f'expected numpy 1.x in runtime, got {v}'; \
    print(f'[final] numpy {v}')"
RUN python -c "import cv2; print(f'[final] cv2 {cv2.__version__}')"
RUN python -c "import mediapipe; print(f'[final] mediapipe {mediapipe.__version__}')"
RUN python -c "import onnxruntime as ort; \
    print(f'[final] onnxruntime {ort.__version__}')"
RUN python -c "import os; \
    assert os.path.exists('/app/yolo11m-pose.onnx'), 'ONNX model missing'; \
    sz = os.path.getsize('/app/yolo11m-pose.onnx'); \
    print(f'[final] yolo11m-pose.onnx present ({sz / 1024 / 1024:.1f} MB)')"

# App code
COPY *.py .
COPY sam3d/ ./sam3d/
COPY yolo/  ./yolo/

CMD ["python", "main.py"]
```

**Image size estimate**:
- Stage 1: ~3.5 GB (ultralytics + torch + cv2 mess) → **discarded**
- Stage 2: ~850 MB (pre-PR-3 stack + onnxruntime 50 MB + onnx model 80 MB)
- **Runtime image ≈ 850 MB**, down from current ~1.7 GB unpacked.

**Stage 1 caching**: any code change in `python/yolo/` invalidates stage 1
unless we structure carefully. Since stage 1 only does export + verify
(not COPY of app code), `python/yolo/` changes do not invalidate it.
Stage 1 only rebuilds when `ultralytics` version changes (rare) or the
export flags change. Good caching profile.

---

## 2. `python/yolo/inference.py` rewrite

Drop `ultralytics`/`torch` imports; use `onnxruntime` + the manual decoder.

### Output tensor — verified shape

`yolo11m-pose` ONNX (exported with `nms=False`):
- **Output name**: `output0`
- **Output shape**: `[1, 56, 8400]` (float32)
  - `1` — batch dim
  - `56` channels = 4 bbox + 1 person-class confidence + 17 × 3 keypoints
  - `8400` — total grid cells across 3 detection heads (80² + 40² + 20² for 640×640 input)

Per-detection slice (after transpose to `[1, 8400, 56]`, then index `[i]`):

| Index | Field | Notes |
|---|---|---|
| `0` | `cx` | bbox centre x, in 640×640 input pixel space |
| `1` | `cy` | bbox centre y |
| `2` | `w` | bbox width |
| `3` | `h` | bbox height |
| `4` | `conf` | objectness × class confidence (single class = person) |
| `5..7` | `(x, y, vis)` of NOSE | input pixel space |
| `8..10` | `(x, y, vis)` of LEFT_EYE | |
| ... | (17 keypoints × 3) | |
| `53..55` | `(x, y, vis)` of RIGHT_ANKLE | |

Total = 4 + 1 + 51 = **56**. Keypoint visibility is a sigmoid-activated
score in [0, 1] — used directly as the confidence threshold check on the
frontend (`MIN_CONFIDENCE = 0.3`).

### Pipeline

```
PNG bytes
  → cv2.imdecode (BGR ndarray, source resolution e.g., 1080×1920)
  → letterbox to 640×640 (record scale + pad offsets)
  → BGR → RGB
  → HWC → CHW
  → /255.0, float32
  → expand dim [1, 3, 640, 640]
  → session.run() → [1, 56, 8400]
  → transpose [1, 8400, 56]
  → argmax conf (single golfer; skip full NMS)
  → top-1 detection: (cx,cy,w,h,conf, 17×(kx,ky,kvis))
  → reverse-letterbox keypoints to source image coords
  → return { keypoints_2d: [[x,y,conf],...]×17, bbox: [x1,y1,x2,y2],
             inference_ms, model: 'yolo11m-pose-onnx', image_width, image_height }
```

### Public surface (unchanged)

`async def infer_pose(png_bytes: bytes) -> Optional[dict]` — same
signature as the ultralytics-based version. Caller (`yolo/orchestrator.py`)
does not need to change.

**Confidence threshold for "no person detected"**: return `None` if the
top-1 detection's `conf < 0.25` (same heuristic ultralytics uses by
default). Orchestrator then logs `[yolo] phase X: no person detected`
and leaves `yolo_keypoints_2d` NULL — exactly the existing PR-3 frontend
fallback path to SAM.

### Singleton pattern

`InferenceSession` is heavy to construct (~200ms on CPU); load once at
first call, cache in module-level `_session`. Use `providers=['CPUExecutionProvider']`
explicitly (faster than letting onnxruntime probe for GPUs that don't
exist on Railway).

### asyncio integration

`session.run()` is synchronous and CPU-bound. Wrap in `asyncio.to_thread()`
inside `infer_pose()` so the 5 phase tasks fan out under one event loop
the same way they do today.

### Estimated LoC

| Section | Lines |
|---|---|
| Module-level singleton + config | 25 |
| `infer_pose(png_bytes)` public entry | 30 |
| `_preprocess(image)` — letterbox + BGR→RGB→CHW | 30 |
| `_postprocess(output, ...)` — transpose + argmax + reverse letterbox | 35 |
| Imports + docstring | 15 |
| **Total** | **~135 LoC** |

Roughly the same length as the current `python/yolo/inference.py` (89
lines), so similar code complexity overall.

---

## 3. `python/requirements.txt` changes

**Before** (current HEAD `d7a4ca1`):
```
fastapi==0.115.0
uvicorn[standard]==0.32.0
httpx==0.27.2
pydantic==2.9.2
mediapipe==0.10.21
opencv-python-headless==4.11.0.86
opencv-python==4.11.0.86
opencv-contrib-python==4.11.0.86
numpy>=2.0,<3.0
fal-client==0.5.6
ultralytics>=8.3.0
```

**After** (Option C):
```
fastapi==0.115.0
uvicorn[standard]==0.32.0
httpx==0.27.2
pydantic==2.9.2

# Heavy deps — back to pre-PR-3 proven-working state
mediapipe==0.10.14
opencv-python-headless==4.10.0.84
opencv-python==4.10.0.84
opencv-contrib-python==4.10.0.84
numpy==1.26.4

fal-client==0.5.6

# PR-3 (Option C): YOLO11m-pose via onnxruntime.
# ultralytics + torch are NOT in runtime — they live in the Dockerfile
# yolo-builder stage and export yolo11m-pose.onnx, which is COPYd into
# the runtime stage. See docs/decisions/PR-3_C_ONNX_DESIGN.md.
# License note: the .onnx file is a derived work of ultralytics' AGPL-3.0
# weights. Same constraints as before — see POSE_MODEL_LICENSE.md.
onnxruntime>=1.18,<2.0
```

**Why the three opencv pins are kept**: mediapipe still transitively
pulls `opencv-contrib-python`. Pinning all three to the same proven
PR-2 version prevents the same "last writer wins" trap from re-appearing
through a future mediapipe upgrade. Cost: one extra dep line; benefit:
the cv2 grief is finally bounded.

**Why `onnxruntime>=1.18,<2.0`**: 1.18 (May 2024) is the first stable
release of the post-1.x line that's been hardened on CPU; supports
numpy 1.x natively. <2.0 prevents an unannounced major bump from
breaking us. **Need to verify** in commit 3 that onnxruntime 1.18 wheels
support numpy 1.26 (expected yes — onnxruntime 1.19 was when numpy 2.x
was added, so 1.18 is numpy-1.x-only or both).

---

## 4. Dockerfile verify step changes

Runtime stage (stage 2) verify shrinks substantially:

**Drop** (no longer relevant — ultralytics is not in runtime):
- `RUN python -c "from ultralytics import YOLO; YOLO('yolo11m-pose.pt')"` (preload)
- `RUN python -c "... assert opencv-python == '4.11.0.86' ..."` (v8 assertion)
- `RUN python -c "... assert opencv-contrib-python == '4.11.0.86' ..."` (v8 assertion)
- `Layer 2.5 / 2.9 / 3.6` — all the numpy<2.0 force-install / dry-run dance is gone. Runtime never installs numpy >2.x to begin with; resolver picks 1.26.4 cleanly from explicit pin.
- ENV `YOLO_AUTOINSTALL=False` / `ULTRALYTICS_OFFLINE=True` — ultralytics isn't there.
- Layer 3.8 fresh-subprocess cv2 cold import — keep as belt-and-suspenders (cheap).

**Add**:
- `RUN python -c "import onnxruntime as ort; print(f'[final] onnxruntime {ort.__version__}')"`
- `RUN python -c "import os; assert os.path.exists('/app/yolo11m-pose.onnx'); sz = os.path.getsize(...); print(f'[final] yolo11m-pose.onnx present ({sz / 1024 / 1024:.1f} MB)')"`

**Keep** (still valuable):
- numpy 1.x assertion (now back to `startswith('1.')`)
- cv2 import check
- mediapipe import check
- Layer 3.8 fresh-subprocess cv2 cold import

`main.py` startup-verify also flips numpy assertion back to `'1.'`.

---

## 5. Build-time smoke test — `scripts/verify_onnx_export.py`

Runs at the **end of stage 1** while both `ultralytics` and `onnxruntime`
are still installed. Fails the build if the manual decoder disagrees with
ultralytics' own inference.

```python
"""
Stage-1 only. Compares the onnxruntime decoder's output against
ultralytics' YOLO() inference on the same image. Run inside the Docker
builder stage; never shipped to runtime.
"""
import numpy as np, cv2, onnxruntime as ort
from ultralytics import YOLO
from yolo_decoder import preprocess, postprocess  # imported from inference.py

REF_IMG = 'ultralytics/assets/zidane.jpg'  # ships with ultralytics
MAX_PIX_DIVERGENCE = 5.0

# ── Reference: ultralytics native path ──
m = YOLO('yolo11m-pose.pt')
ref = m(REF_IMG, verbose=False, conf=0.25)[0]
ref_kps = ref.keypoints.data[0].cpu().numpy()  # (17, 3) — x, y, conf
ref_xy = ref_kps[:, :2]
print(f'ref_xy:\n{ref_xy}')

# ── Test: our onnxruntime decoder ──
session = ort.InferenceSession(
    'yolo11m-pose.onnx',
    providers=['CPUExecutionProvider'],
)
image = cv2.imread(REF_IMG)
input_arr, scale, pad = preprocess(image)
output = session.run(None, {'images': input_arr})[0]
test_kps = postprocess(output, scale, pad, orig_h=image.shape[0], orig_w=image.shape[1])
test_xy = test_kps[:, :2]
print(f'test_xy:\n{test_xy}')

# ── Compare ──
diff = np.abs(ref_xy - test_xy).max()
print(f'[verify] max coordinate divergence: {diff:.2f}px')
assert diff < MAX_PIX_DIVERGENCE, (
    f'ONNX decoder vs ultralytics divergence too large: {diff:.2f}px '
    f'(threshold {MAX_PIX_DIVERGENCE}px). Decoder math is wrong.'
)
print('[verify] OK: ONNX decoder agrees with ultralytics native inference')
```

`yolo_decoder.py` is a thin module that imports the `preprocess` /
`postprocess` helpers from `python/yolo/inference.py` — no duplication.

**Why this is load-bearing**: the entire premise of Option C is that the
ONNX export + manual decoder produces equivalent keypoints. If we ship
without this check, we won't know until a user's swing renders with
discs in wrong places.

---

## 6. Commit split (4 commits, single push)

| # | Message | Files |
|---|---|---|
| 1 | `feat(deploy): multi-stage Dockerfile — ONNX export builder + slim runtime` | `python/Dockerfile`, `scripts/verify_onnx_export.py` (new) |
| 2 | `feat(python): rewrite yolo/inference.py to use onnxruntime + manual decoder` | `python/yolo/inference.py` |
| 3 | `feat(python): drop ultralytics+torch from runtime; add onnxruntime` | `python/yolo/__init__.py`, `python/requirements.txt`, `python/main.py` startup-verify flips numpy assertion to `'1.'` |
| 4 | `chore(deploy): clean up v1–v8 hotfix layers no longer applicable` | `python/Dockerfile` second-pass cleanup if anything remained, e.g., dead comments |

**Push strategy**: one push of all 4 after Step 3, mirroring PR-2C and
v1-of-PR-3.

Commit 4 may end up empty if commit 1's Dockerfile rewrite already does
the cleanup. If so, drop it — 3 commits is fine.

---

## 7. Risks + mitigations

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| 1 | onnxruntime 1.18 wheels require numpy 2.x (breaks our 1.26 pin) | Low | Verified at install time in commit 3. If true, drop to onnxruntime 1.17. If 1.17 still requires numpy 2.x — investigate, but this would be highly unusual; numpy 1.26 is still the LTS line for inference runtimes. |
| 2 | Decoder math wrong (wrong tensor layout, missed sigmoid, letterbox off-by-one) | Medium | `scripts/verify_onnx_export.py` at stage-1 end catches this. Threshold 5px (generous for our use; tighten later if needed). |
| 3 | ONNX export produces a different shape than documented (`nms=True` accidentally enabled, opset diff, dynamic axes) | Low | Export command pins `imgsz=640, simplify=True, nms=False, opset=12` — fully deterministic. We log the actual output shape at session-load time in the decoder. |
| 4 | yolo11m-pose.onnx size > 100 MB (Railway image size budget concern) | Low | yolo11m has 20.9M params → fp32 ONNX ≈ 84 MB. Runtime image ~850 MB total, comfortable under 3 GB budget. |
| 5 | ultralytics removes ONNX export path in a future release | Very Low | We pin `ultralytics` only in stage 1. If a breakage happens, freeze that pin or use a Python wheel snapshot of last known good. |
| 6 | We forget some ultralytics-feature we depended on (e.g., the auto-NMS, the auto-confidence sorting) | Low | Decoder implements both manually. Build-time verify catches regression. |
| 7 | Railway free tier can't build multi-stage images | Very Low | Multi-stage is a standard Docker feature, supported on every Docker-based PaaS including Railway. |

**Worst-case revert**: if Option C fails post-deploy and we need PR-3
functionality back fast, `git revert` the 4 commits in reverse order
brings us back to current `d7a4ca1` (which is still broken, but no
worse than now). Better fallback: `git reset --hard 754dabb` returns to
pre-PR-3 SAM-only state (proven working production).

---

## 8. Out of scope (explicitly)

- Replacing the AGPL Ultralytics weights with an Apache-2.0 alternative.
  Already an ADR ([`POSE_MODEL_LICENSE.md`](./POSE_MODEL_LICENSE.md));
  Option C *increases* AGPL-cleanliness (no AGPL-licensed Python ships
  at runtime, only the model weights as data), but does not eliminate it.
- Switching to a different keypoint backend entirely (RTMPose / MoveNet).
  Considered in audit §7 Option D and ADR §replacement-path; not needed
  to unblock PR-3.
- Adding 3D perspective tilt to the frontend (PR-2D follow-up).

---

## 9. What needs your approval before Step 3

1. **Multi-stage structure** — OK to install ultralytics + torch in stage
   1 with no dep-pin gymnastics, since stage 1 is discarded? (Yes
   recommended.)
2. **Decoder approach** — argmax-by-confidence for single-person golf
   swing instead of full NMS (saves ~30 LoC; topologically simpler)?
3. **Build-time smoke test** — fail the build at stage 1 if decoder
   diverges > 5px from ultralytics? Are there reasons to make this
   non-fatal (e.g., to ship even with degraded keypoint accuracy)?
4. **Three opencv pins kept** — even though ultralytics is gone, we
   still pin all three opencv distros at 4.10.0.84 because mediapipe
   transitively pulls opencv-contrib-python. OK to keep?
5. **Commit split** — 3 or 4 commits as outlined; one push at the end.
6. **onnxruntime version range** — `>=1.18,<2.0`. If commit 3 install
   fails the numpy assertion, drop to 1.17 — OK to make that call
   in-stream without re-asking?

---

⏸ **STOP** — awaiting your sign-off on this Step 2 design before any
file changes in Step 3.
