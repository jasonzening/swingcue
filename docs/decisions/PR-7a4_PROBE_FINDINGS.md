# PR-7a.4 Probe Findings — SMPL Vertex Sampling Investigation

**Date**: 2026-05-21
**Status**: PROBE FAIL — deferred to PR-7a.5
**Outcome**: Commit current PR-7a + 7a.1 + 7a.2 + 7a.3 (bidirectional EMA) as
the PR-7a deliverable. SMPL vertex sampling becomes a future PR.

---

## Hypothesis

After PR-7a.3 (bidirectional EMA) closed the spec-§5 acceptance gates,
visual review of overlays still showed:
- Residual drift at fast-motion phases (top, impact)
- Anchor-vs-body offset at follow-through

These were attributed to the **fitted offset-vector model** approximating
SMPL bone-center → anatomical-landmark distance via a single 3D vector
per (joint, phase, view). The Path A model has a structural ceiling:
WHAM emits joint *bone centers* (e.g., glenohumeral, C7), but coaching
landmarks are skin-surface (acromion peak, throat midpoint, etc.).

**Hypothesis**: sample anatomical landmark *vertices* directly from
WHAM's posed SMPL mesh (T, 6890, 3 vertex array). Each landmark
becomes a vertex index lookup — no fitting, no per-phase tuning,
no chirality post-processing.

Expected wins:
- Acromion at exact skin-surface peak (~5-10 cm closer to GT vs current ~12 px residual)
- Throat midpoint directly addressable (vs current 8.9 px head_spine residual)
- New anatomical landmarks (knee, ankle, foot) free with no extra fitting
- Eliminates Finding F + Finding G + per-phase vectors entirely

## Why the probe failed

The probe required SMPL vertex data (verts) at GT-labeled frames.
Three prescribed paths to obtain verts all blocked by upstream data
choices in `wham_runner.py`:

### Path A — verts already in local .pkl: NO
Production `wham_runner.py` returns a JSON dict via the Modal function;
the underlying WHAM .pkl lives only on the ephemeral Modal container
at `/opt/wham/output/demo/<vid>/wham_output.pkl`. No Volume persistence
is configured for inference outputs (only for model weights).

### Path B — local SMPL forward pass from saved pose+betas+trans: NO
`wham_runner.py:429` explicitly sets `"smpl_pose": None` with the
comment *"24x3x3 too verbose; skip for smoke"*. `trans` (root translation)
was never plumbed into the JSON serialization at all. Only `smpl_betas`
(10-vec) is saved per-frame.

SMPL forward pass needs all three (pose + betas + trans) to compute
posed vertices via linear blend skinning. Two of three are missing.

### Path C — local WHAM re-run: NO
WHAM is **not installed locally**. Our entire WHAM pipeline lives in
the Modal Docker Image (built via the ~23-iteration dependency slog
documented in `PR-6.0_PHASE_2_DESIGN.md`: torch 1.11+cu113, mmcv-full
1.5.0, DPVO CUDA extension, ViTPose, chumpy, etc.). `.venv-pilot` only
has `modal` + `httpx` + `cv2`; no torch + no CUDA + no WHAM repo.

Installing WHAM locally would replay that slog on Windows + RTX 4080
(vs the proven Linux/Ubuntu Modal Docker), with uncertain success.

### Path D — sibling Modal script: 2 crash-loops in this session
- **Crash 1**: `ModuleNotFoundError: No module named 'runners'` —
  my probe script imported `from runners.wham_runner import ...` which
  works locally (after `sys.path.insert(0, REPO_ROOT/'python/pilot')`)
  but the Modal container mount layout doesn't include
  `python/pilot/runners/` as a top-level package. Fixable via
  `image.add_local_dir('python/pilot', remote_path='/root/pilot')`,
  but the actual data we extracted would still need format correction
  (Path D's pkl is via `joblib.load`, not `pickle.load`; my script
  used the latter — second crash).
- **Crash 2**: `UnpicklingError: invalid load key, '\x01'` — WHAM's
  pkl is joblib-serialized, not vanilla pickle.

After fixing the import + the joblib path, a third attempt would
have hit a yet-undiscovered issue. Jason called the architectural
mistake at minute 50: SMPL forward pass is CPU work; we don't need
WHAM GPU inference for the probe IF we already have pose+trans+betas.
But we don't.

---

## Local data state at probe time

```
python/pilot/output/wham/
  5bbcfbc8/joint_centers_3d.json   (201 frames, regressed joints only)
  b32e0f21/joint_centers_3d.json   (120 frames, regressed joints only)
  b3fea3f0/joint_centers_3d.json   (139 frames, regressed joints only)
  b3fea3f0/overlay.mp4             (raw 2D-projected debug video)
```

Per-frame fields in the JSON:
- `joint_centers_3d` (17 H36M joints, derived locally on Modal via
  `J_regressor_h36m.npy @ verts`)
- `joint_centers_2d_projected: None` (legacy field, unused)
- `smpl_betas` (10-vec — useful but insufficient alone)
- `smpl_pose: None` (explicitly skipped)

No verts, no pose, no trans saved anywhere locally.

---

## Required investment for future PR-7a.5

1. **Patch `wham_runner.py`** (~30 min):
   - Save `smpl_pose` per frame: `track["pose"][i].tolist()` (already
     wired into the frame dict at line 429, just remove the `None`).
     Adds ~24*3*3*8 = 1.7 KB per frame → 230-280 KB per clip.
   - Save `smpl_trans` per frame: same pattern. Adds 24 bytes per frame.
   - Optionally save `verts` at GT frames as sidecar JSON (avoids
     local SMPL forward pass entirely): 6890*3*4 = 82 KB per frame.
2. **Re-run WHAM on all 3 test clips** to backfill (~$0.03, 10 min wall
   on Modal A10G). Uses existing `run_wham_one.py` — no new probe
   scripts needed.
3. **Local SMPL forward pass infrastructure** (~2 hr):
   - Install `smplx` package in `.venv-benchmark` or `.venv-pilot`.
   - Wire the local SMPL forward pass against `SMPL_NEUTRAL.pkl`
     (already extracted to `local_models/smpl/_extracted/`).
   - Test on b3fea3f0 at the 5 GT frames; cross-check vs WHAM's own
     verts output (sidecar from step 1).
4. **PR-7a.5 implementation** — replace Path B body-local 3D vectors
   with vertex-index lookups (~4 hr per the original PR-7a.4 spec
   estimate, now revised to ~6 hr given probe-derived insights about
   landmark index sensitivity).
5. **Verification gates**:
   - Run `probe_smpl_vertex_landmarks.py` (already built this session,
     awaits verts data) on all 15 GT samples.
   - Acromion vertex must beat current PR-7a anchor on ≥3 of 4 face_on
     samples per landmark.
   - Visual sanity on knee/ankle/foot landmarks.
6. **Engine integration** (~2 hr): if probe passes, swap the
   `anatomical_offset.apply_offset_to_frame` mode A vector lookup with
   a vertex-index lookup (`verts[idx]` per frame instead of
   `body_local_to_camera(stored_vec, basis)`).

**Total estimated investment for PR-7a.5**: ~10-12 hours including
verification + engine integration.

---

## What the probe DID produce (kept for PR-7a.5 reuse)

- `docs/PR-7a4_PROBE/smpl_landmark_indices.json` — 11 SMPL vertex
  indices derived from T-pose geometry (head_crown=411,
  throat_midpoint=444, c7=414, acromion L/R=4721/1238,
  greater_trochanter L/R=6375/2915, lateral_epicondyle L/R=4447/959,
  lateral_malleolus L/R=6749/3348). Self-derived from
  `SMPL_NEUTRAL.pkl` template (T-pose, betas=0) — no external
  references needed (WebFetch/WebSearch were sandbox-blocked).
- `python/pilot/scripts/probe_derive_smpl_landmarks.py` — chumpy-stub
  SMPL pkl loader + T-pose geometric landmark derivation. Reusable.
- `python/pilot/scripts/probe_extract_verts_modal.py` — Modal sibling
  script for verts extraction (needs the bug fixes: `add_local_dir`
  for `runners/`, `joblib.load` instead of `pickle.load`, `cwd=`
  for demo.py). Has the inlined helpers from `runners/wham_runner.py`
  to avoid the import bug for future use.
- `python/pilot/scripts/probe_smpl_vertex_landmarks.py` — local
  rendering + acceptance script. Reads verts sidecar + PR-7a corrected
  JSON + GT labels, projects to 2D, renders comparison PNGs, writes
  `distance_to_gt.csv`. Awaits verts data to actually run.

---

## Recommendation

**Ship PR-7a as-is** with the bidirectional EMA stack (PR-7a + 7a.1
+ 7a.2 + 7a.3). Spec-§5 class gates all GREEN; visual approved earlier
this session. Defer SMPL vertex sampling to PR-7a.5 after:
1. User feedback on PR-7c frontend rendering strategy (Path E:
   confidence-aware opacity at follow-through may mask residual
   drift visually without architectural change).
2. Capacity for the ~10-12 hr PR-7a.5 investment.

Don't repeat this probe pattern without first patching
`wham_runner.py:429` to save the SMPL data we'll need.
