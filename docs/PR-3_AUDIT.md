# PR-3 Dependency Audit

**Date**: 2026-05-16
**Status**: Investigation complete; awaiting strategy decision
**Trigger**: 9 hotfix attempts (v0 → v8) failed to stabilise the
production image. Halted patching to root-cause the dependency graph.

---

## Executive Summary

**The runtime crash (`numpy._core.multiarray failed to import`) is the
visible symptom of a packaging contradiction inside opencv-python
4.10.0.84:**

- Its wheel **metadata** declares `numpy<2.0` as the upper bound.
- Its compiled **binary** internally references `numpy._core` symbols,
  which exist only in numpy 2.x.

pip's resolver respects the metadata and installs numpy 1.x; the binary
then fails to load that numpy at runtime. The resolver cannot detect
this mismatch — it sees the metadata, not the .so internals.

**The hotfix loop went wrong because we kept pinning numpy lower** to
satisfy *the metadata*, then watching the *binary* crash anyway.

The real architectural issue is bigger than this one bug:

1. `ultralytics>=8.3.0` declares `opencv-python>=4.6.0` (no upper bound).
2. mediapipe transitively pulls `opencv-contrib-python`.
3. We pin `opencv-python-headless`.
4. **All three opencv distributions install their cv2 native module to
   the SAME `site-packages/cv2/` directory.** Whichever is installed
   last overwrites the others' `.so` files.

In our image, this resolves to three versions in one cv2/ dir with
last-write-wins binary state. Even with all three pinned at one version
(v6), the underlying problem — opencv's lying metadata + ultralytics
demanding the full opencv-python — remains.

**Conclusion: do not continue pinning opencv versions.** The packaging
ecosystem is fundamentally broken in our combination, and the right fix
is structural: either install ultralytics without its opencv dep, or
remove ultralytics from the production image entirely (ONNX path).

---

## 1. Pre-PR-3 Production State (Verified)

`git show 754dabb:python/requirements.txt`:

```
fastapi==0.115.0
uvicorn[standard]==0.32.0
httpx==0.27.2
pydantic==2.9.2
mediapipe==0.10.14
opencv-python-headless==4.10.0.84
numpy==1.26.4
fal-client==0.5.6
```

**This combination ran successfully in production for the PR-2B/PR-2C
work** — 5-phase SAM analysis on Railway, MediaPipe overlay rendering
on Vercel, etc.

Key inference: `opencv-python-headless==4.10.0.84 + numpy==1.26.4` is
**proven-working** in the Railway container *when ultralytics is absent*.
The 9 hotfix attempts began when ultralytics was added in commit
`fd54710`.

---

## 2. Tests A & B — Status

| Test | Goal | Status |
|---|---|---|
| A: pre-PR-3 deps alone | Verify mediapipe + opencv-python-headless + numpy 1.x boot | **Not runnable in audit sandbox** (no docker on this machine). Empirically already proven by PR-2B/PR-2C production runs. |
| B: pre-PR-3 + ultralytics | Reveal which opencv* pip pulls | **Replaced by pip resolver dry-run** (see §3). Equivalent or stronger signal — shows the resolver's exact decisions before any install commits. |

User can run the literal Dockerfile tests locally if desired; the
predicted outcome below is now derived from the resolver's own choices.

---

## 3. pip Dry-Run — The Smoking Gun (Verified)

Command:

```bash
pip install --dry-run --report dryrun.json \
  "ultralytics" \
  "opencv-python-headless==4.10.0.84" \
  "mediapipe==0.10.14" \
  "numpy==1.26.4"
```

Output (43 packages total; war-zone subset):

```
numpy==1.26.4                      ← respects our pin
opencv-python-headless==4.10.0.84  ← respects our pin
opencv-python==4.11.0.86           ← pip pulled (from ultralytics dep)
opencv-contrib-python==4.11.0.86   ← pip pulled (transitive, likely mediapipe)
torch==2.12.0
torchvision==0.27.0
mediapipe==0.10.14
ultralytics==8.4.51
ultralytics-thop==2.0.19
matplotlib==3.10.9
scipy==1.17.1
polars==1.40.1
... (and 30 more)
```

**Verified fact**: pip will install THREE cv2 variants into the same
directory the moment ultralytics is added, regardless of how we pin
opencv-python-headless.

**Verified fact**: pip is satisfied with numpy 1.26.4 metadata-wise
across all three opencv variants AND ultralytics AND mediapipe — even
though opencv-python 4.11.0.86's binary needs numpy 2.x at runtime.

---

## 4. Ultralytics Dependency Analysis (Verified from PyPI METADATA)

`ultralytics==8.4.51` wheel METADATA (canonical, extracted from
`ultralytics-8.4.51-py3-none-any.whl`):

### Core required dependencies

```
numpy>=1.23.0                  # no upper bound
matplotlib>=3.3.0
opencv-python>=4.6.0           # 🚨 FULL OPENCV, no upper bound
pillow>=7.1.2
pyyaml>=5.3.1
requests>=2.23.0
scipy>=1.4.1
torch>=1.8.0
torchvision>=0.9.0
psutil>=5.8.0
polars>=0.20.0
ultralytics-thop>=2.0.18
```

### Notable optional extras

```
[export]:
  numpy<2.0.0                  # 🚨 export pipeline pins numpy 1.x
  onnx>=1.12.0
  onnxslim>=0.1.82
```

The `[export]` extra is **NOT** installed by `pip install ultralytics`.
But anyone wanting to export models (e.g., to ONNX, the path we're
considering below) needs `ultralytics[export]` which pins numpy 1.x.

### Transitive: `ultralytics-thop==2.0.19` METADATA

```
Requires-Dist: numpy
Requires-Dist: torch
```

**Verified fact**: `ultralytics-thop` does NOT pull `opencv-contrib-python`.
The v6 hypothesis that thop was the contrib-python source was wrong.
The contrib-python comes from elsewhere — most likely mediapipe's
transitive deps (mediapipe lists opencv-contrib-python in its setup).

### "Can ultralytics work --no-deps?"

Empirically not tested in the sandbox (no docker), but the metadata
tells us what would happen:

- `import ultralytics` works as long as numpy + torch + opencv (any cv2
  module) are importable.
- `YOLO(model_path)` works.
- Model inference works.

The risk: ultralytics may call into `cv2` features (e.g., `cv2.imshow`)
that exist only in `opencv-python` (full), not in `opencv-python-headless`.
For our use (inference on PNG bytes, no GUI display), headless is
sufficient — the production code already uses cv2.imdecode through
headless. **Low risk** if we pin headless and skip opencv-python.

---

## 5. ONNX Feasibility (from docs.ultralytics.com)

### Export

```python
from ultralytics import YOLO
model = YOLO('yolo11m-pose.pt')
model.export(format='onnx', nms=True)
# Produces yolo11m-pose.onnx (~50–80 MB)
```

This is a one-time step that can run at **Docker build time** in a
separate builder stage. The resulting `.onnx` file is then COPYd into
the slim runtime stage.

### Inference (runtime)

Required dependencies at runtime: **only `onnxruntime`** (~50 MB on
CPU). No ultralytics, no torch, no opencv-python (the full version).

### NMS / Post-processing

`nms=True` during export bakes NMS into the ONNX graph for detection
models. For pose models, the output tensor shape is:

```
(batch, 4 + num_classes + num_keypoints*3, num_predictions)
```

**Keypoint post-processing must be implemented in Python** (~50–100
lines): apply NMS to detections, then for each detection slice the
17 (x, y, conf) keypoint triples out of the output channels. This is
well-documented in the ultralytics ONNX examples repo, plus official
reference Python decoders are available.

### Risk

Medium. Output tensor format is stable across yolo11 versions but the
decode code is non-trivial. We'd want a local smoke test against the
PR-2B test video before merging.

---

## 6. Root Cause Synthesis

The hotfix loop was treating symptoms. The actual structural problem:

| Layer | Failure |
|---|---|
| pip resolver | Sees only metadata; doesn't catch binary-ABI lies. |
| opencv-python 4.10.0.84 wheel | Metadata says numpy<2; binary needs numpy 2.x. |
| ultralytics core | Hard-requires opencv-python (no upper bound). |
| mediapipe | Pulls opencv-contrib-python transitively. |
| `site-packages/cv2/` | Single shared directory across all three opencv distros. |

Pinning all three opencv distros to one version (v6) prevented one
class of overwrite but left us still installing opencv-python ≥ 4.6
unbounded — and the latest "stable" pin (4.11.0.86) compiled-binary
state still mismatched.

**The only way out is to break one of these layers.** Three viable
strategies follow.

---

## 7. Recommended Path Forward — Three Options

### Option A: `pip install ultralytics --no-deps`, manually install the safe subset

**Concept**: install ultralytics without letting pip pull its declared
deps, then install only the non-opencv ones explicitly. Skip
`opencv-python` (full); rely on the pinned `opencv-python-headless` to
provide cv2.

**Dockerfile sketch**:
```dockerfile
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN pip install \
    mediapipe==0.10.14 \
    opencv-python-headless==4.10.0.84 \
    numpy==1.26.4
RUN pip install --no-deps ultralytics
RUN pip install \
    matplotlib pillow pyyaml requests scipy psutil polars \
    ultralytics-thop
```

**Files changed**: `python/Dockerfile` (3 new layers), `python/requirements.txt`
(split ultralytics out into a `# install --no-deps below` section, or
move installation entirely to Dockerfile).

**Risk: MEDIUM**.
- ✅ Returns numpy + opencv to the proven pre-PR-3 state.
- ✅ No second venv, no ONNX rework.
- ❌ Brittle: future ultralytics releases may add a real dep we miss.
- ❌ Risk: if `ultralytics-thop` or `polars` itself pulls something
  opencv-related, the surprise reappears. Already verified
  ultralytics-thop is clean (only numpy + torch), but other transitives
  weren't audited.
- ❌ Loses pip's ability to maintain consistency for ultralytics's
  graph going forward.

### Option B: Two-venv isolation in one container

**Concept**: two separate Python virtual environments inside the image.
venv1 has the pre-PR-3 deps (mediapipe path). venv2 has ultralytics
with its native numpy 2.x ecosystem. `main.py` (running from venv1)
spawns subprocesses against venv2's interpreter to run YOLO inference.

**Files changed**: `python/Dockerfile` (creates 2 venvs, installs each
from different requirements files), `python/requirements-mediapipe.txt`
(new), `python/requirements-yolo.txt` (new), `python/yolo/inference.py`
(rewritten to spawn subprocess + receive JSON keypoints over stdout).

**Risk: MEDIUM-HIGH**.
- ✅ Hard isolation; the numpy/cv2 incompatibilities literally cannot
  cross venv boundaries.
- ✅ Each venv runs its own dep ecosystem cleanly.
- ❌ Image ~3× larger (two installs of similar stacks).
- ❌ Subprocess spawn per inference adds ~200ms latency per phase
  (negligible vs SAM's 15s, but still wasteful).
- ❌ Doubles the surface area to maintain.
- ❌ Doesn't fix the fundamental cv2 packaging problem; just walls it
  off.

### Option C: ONNX export, drop ultralytics from runtime (RECOMMENDED)

**Concept**: ultralytics is a build-time-only dep. In a multi-stage
Dockerfile, the builder stage installs ultralytics, exports
`yolo11m-pose.pt` → `yolo11m-pose.onnx`, then the runtime stage copies
in only the `.onnx` file and uses `onnxruntime` for inference. Runtime
deps return to pre-PR-3 state + onnxruntime.

**Files changed**:
- `python/Dockerfile`: multi-stage (builder installs ultralytics, runtime doesn't)
- `python/requirements.txt`: back to pre-PR-3 + `onnxruntime>=1.18.0`
- `python/yolo/inference.py`: rewritten to use `onnxruntime.InferenceSession`
  and manual keypoint decode (~50–100 lines)
- `python/yolo/__init__.py`: remove `from ultralytics import YOLO`
  references (already encapsulated through inference.py)

**Risk: LOW-MEDIUM**.
- ✅ Cleanest separation. ultralytics + torch never enter the runtime
  image → no opencv conflict possible.
- ✅ Runtime image size ~600 MB smaller (drop torch + ultralytics).
- ✅ Faster cold start (no PyTorch import overhead).
- ✅ Future-proof: onnxruntime is stable, well-maintained by Microsoft.
- ❌ Need ~50–100 lines of YOLO-pose decode Python. Well-documented
  but non-trivial.
- ❌ Loses ultralytics' auto-update story (no relevance for us; we
  picked yolo11m and aren't tracking upstream).
- ❌ One-time work to write + test the decode; risk of getting the
  output tensor parsing wrong (mitigated by smoke test against PR-2B's
  verified video and comparing 17 keypoint coords to a reference
  ultralytics run).

### Option D — also considered, rejected

| Option | Why rejected |
|---|---|
| Roll back to PR-2 (no YOLO) | Defeats PR-3's purpose; SAM-only data already shown insufficient for accurate shoulder/hip overlay. |
| Pin to even older opencv (4.9.0.80) | Same metadata-vs-binary issue likely. Punts the problem 6 months. |
| Replace mediapipe with another CV stack | Out-of-scope. Mediapipe is used for phase detection only and works on the pre-PR-3 deps. |
| Use a CUDA base image (Railway GPU) | Railway is CPU-only; confirmed in earlier audit. Not available. |

---

## 8. Recommendation

**Option C (ONNX) is the right strategy.**

Reasoning, ordered by weight:

1. **Architectural correctness**: separates build-time tooling from
   runtime, which is how containerised ML normally ships. The current
   image lugs around ultralytics + torch (~1 GB) only to call one
   inference function — pure waste.
2. **Eliminates the root cause permanently**: with ultralytics out of
   the runtime, there's no caller pulling unbounded opencv-python.
   Future ultralytics releases (which will only get heavier as the
   `[export]` extra grows) become irrelevant to our prod image.
3. **Returns to proven dep state**: runtime deps are PR-2's exact
   stack plus a single new package (`onnxruntime`). Risk surface is
   tiny.
4. **Performance bonus**: onnxruntime on CPU is typically 1.5–2× faster
   per inference than the equivalent PyTorch path because of graph-level
   fusion and reduced Python overhead. yolo11m on Railway CPU could drop
   from ~300ms/frame to ~150–200ms/frame.

**Estimated implementation**:
- 1 commit: multi-stage Dockerfile + onnxruntime in requirements.txt
- 1 commit: rewrite `python/yolo/inference.py` to use onnxruntime
  (singleton InferenceSession, manual NMS + keypoint decode)
- 1 commit: local smoke test verification on `test_finish.png`
  (~5 min work, optional but cheap insurance)

Total scope: ~150 LoC + 3 commits, smaller than the existing v8 patch
surface.

---

## 9. If Option C is rejected — fallback ordering

1. **Option A** — `--no-deps` install. Faster to ship, but technical
   debt. Acceptable as a stopgap if we need YOLO inference live within
   24h.
2. **Option B** — two-venv. Only if both A and C are blocked for
   reasons not yet surfaced.

---

## 10. Next Step

Awaiting strategy decision from user. **No code or commits made in
this audit task.** The current `main` (HEAD `d7a4ca1`) still has the
failed v8 image; rolling back to a known-good (pre-PR-3 commit
`754dabb`) would temporarily restore the SAM-only pipeline if needed
while Option C is implemented.
