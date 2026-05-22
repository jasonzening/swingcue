# PR-7: Golf Keypoint Correction Layer (v2 — ChatGPT-reviewed)

**Status**: Spec v2 (ChatGPT-reviewed, awaiting Jason approval before CC implementation)
**Date**: 2026-05-20
**Predecessor**: Phase 2b WHAM smoke validated (memory #22) — WHAM SMPL skeleton has MVP-acceptable potential but raw keypoints need golf-specific calibration before consumer-facing display.
**Strategic frame**: WHAM is the **base motion signal**, NOT the final coaching layer. SwingCue's technical moat lives in this correction layer, not in pose model selection.

**v2 changes vs v1 (ChatGPT review)**:
- §5 Acceptance gates relaxed: shoulder/hip < 10 px (not 8), head/spine < 12 px, wrist/hand diagnostic-only
- §3.B Ground truth min: 15 samples (3 video × 5 phase), first pass shoulder/hip/neck only
- §3.F NEW: `coaching_anchors_2d` separate from `keypoints_2d_projected` — analysis vs visual anchor distinction
- §3.E Phase-aware smoothing **locked** — explicit prohibition of uniform-smoothing implementation
- §4 PR-7a scope tightened: **NO frontend changes** — Python module + offline corrected timeline only

---

## §1 What this PR is and is NOT

### IS
- A correction stack that runs **on top of** WHAM raw output (or any future SMPL-family pose runner)
- Implements: setup calibration baseline + per-joint anatomical offset + temporal smoothing + outlier rejection + L/R identity stability + phase-aware constraints
- **Two corrected anchor outputs**:
  - `keypoints_3d_corrected` — for analysis (angle calculations, phase detection, biomechanics)
  - `coaching_anchors_2d` — for visual overlay (disc, skeleton rendering on user-facing video)
- New schema: `pose_timeline_3d_corrected` separate from raw `pose_timeline_3d_wham`
- Frontend disc/overlay reads **coaching_anchors_2d**; raw timeline kept only for `?debug=pose`
- Configurable correction coefficients (shoulder/hip inward offset, smoothing α, outlier threshold) — empirically tuned on red-dot ground truth
- Production-grade: integration with existing analyze pipeline, Vercel + Railway/Modal cooperation

### IS NOT
- A search for a different pose model. **WHAM stays.**
- A pursuit of medical-grade anatomical bone-center accuracy
- A frontend disc redesign (current disc/SkeletonOverlay logic mostly unchanged, just consumes new schema)
- A change to Phase 2c (Human3R/SMPLest-X) — those are **deprioritized** because they're same SMPL family, same surface-anchor limit. Correction layer applies to them too if ever swapped in.

---

## §2 Architectural placement

```
Video upload
    ↓
[Track 1 production, existing] mediapipe_pose OR rtmpose_v1 → pose_timeline_2d (surface)
    ↓
[Track 2 production] WHAM Modal call → pose_timeline_3d_wham (raw SMPL joints)
    ↓
[NEW: this PR] Golf Correction Layer:
      - keypoints_3d_corrected   (analysis use)
      - coaching_anchors_2d      (visual overlay use)
    ↓
Frontend disc/SkeletonOverlay reads coaching_anchors_2d
```

Three timeline schemas coexist:

| Schema | Source | Consumer |
| --- | --- | --- |
| `pose_timeline_2d` | mediapipe_pose / rtmpose_v1 | Legacy / fallback / debug |
| `pose_timeline_3d_wham` | WHAM Modal endpoint | Debug overlay only (`?debug=raw_wham=1`) |
| **`pose_timeline_3d_corrected`** | **PR-7 correction layer** | **Production: disc + analysis** |

PR-7 implements the third schema and the layer that produces it.

---

## §3 Correction layer components

### A. Setup calibration baseline

Identify the setup frame (first stable frame before swing start) and lock anatomical references for the entire swing:

```python
@dataclass
class GolfSetupBaseline:
    setup_frame_idx: int            # which frame the baseline was extracted from
    setup_ts: float                 # timestamp
    baseShoulderWidth: float        # 3D Euclidean dist between corrected shoulders
    baseHipWidth: float
    baseSpineLength: float          # pelvis→neck
    baseHeadPosition: np.ndarray    # 3D centered on shoulderCenter
    baseStanceWidth: float          # ankle to ankle
    baseHandPosition: np.ndarray    # 3D, both hands gripped on club
    baseSpineAngle: float           # forward tilt at setup
```

**Why baseline matters**: in subsequent frames, per-frame raw `raw_shoulder_dist` will shrink/stretch with body rotation (perspective foreshortening). Discs and overlays should use `baseShoulderWidth` for size, NOT per-frame `raw_shoulder_dist` — otherwise disc grows/shrinks artificially with rotation.

**Setup frame detection** (auto, no user UI):
- First N frames (typically 0-2 sec) of analyzed video
- Identify the longest stable window (frame-to-frame movement < threshold)
- Pick the median frame of that window as setup baseline

Future PR-7.1 could add user-facing UI for setup correction nudge, but MVP is fully automatic.

### B. Per-joint anatomical offset correction

WHAM raw SMPL joints sit on body surface (deltoid, pants seam, etc.). Correct toward anatomical anchor:

```python
# Configurable correction coefficients per joint group
# Values shown are STARTING estimates; PR-7b empirical sweep determines locked values.
SHOULDER_INWARD_OFFSET = 0.14   # toward spine center; sweep range 0.10-0.18
HIP_INWARD_OFFSET     = 0.16   # toward pelvis center; sweep range 0.12-0.20
KNEE_OFFSET            = 0.05   # minor inward; mostly keep raw
ANKLE_OFFSET           = 0.03   # minimal
WRIST_OFFSET           = 0.0    # keep raw — hands are on club, diagnostic only
HEAD_OFFSET            = 0.08   # toward skull center from forehead skin
```

For each frame:

```python
def correct_shoulder(raw_left_shoulder, raw_right_shoulder, shoulder_center):
    """raw shoulder pulled inward toward shoulderCenter (midpoint of both)."""
    offset_vec = shoulder_center - raw_left_shoulder
    corrected_left = raw_left_shoulder + offset_vec * SHOULDER_INWARD_OFFSET
    offset_vec = shoulder_center - raw_right_shoulder
    corrected_right = raw_right_shoulder + offset_vec * SHOULDER_INWARD_OFFSET
    return corrected_left, corrected_right
```

**Empirical tuning protocol** (ChatGPT review §B — min 15 samples):

1. Jason hand-marks 5 keypoints per phase on 3 test videos × 5 phases = **15 ground-truth samples minimum**:
   - **Videos**: b3fea3f0, a735cc7d, 5bbcfbc8
   - **Phases per video**: Setup, Top, Transition, Impact, Finish
   - **Keypoints per phase (first pass)**: `left_shoulder`, `right_shoulder`, `left_hip`, `right_hip`, `neck_center` (or chest_center)
   - **Optional pass 2 (after first config locked)**: knee, ankle, head_crown for finer tuning
2. CC writes offset sweep harness: for each of `SHOULDER_INWARD_OFFSET in (0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20)`, project all 15 reference frames, compute mean Euclidean distance corrected-vs-redDot.
3. Pick value minimizing mean distance per joint group. Lock into `golf_correction_config.py`.

Same procedure for hip and head. Knee/ankle have lower-priority sweeps (smaller sample set OK).

Wrist/hand is diagnostic-only (see §5 acceptance) due to occlusion/crossing complexity.

### C. Temporal smoothing

```python
@dataclass
class SmoothingConfig:
    alpha_high_conf: float = 0.30        # smoothed = 0.7 * prev + 0.3 * current
    alpha_low_conf: float  = 0.10        # smoothed = 0.9 * prev + 0.1 * current
    confidence_threshold: float = 0.65
    velocity_outlier_threshold_ratio: float = 0.25  # of bodyWidth per frame
```

Per-keypoint forward smoothing:

```python
def smooth_keypoint(raw_t, prev_smoothed, confidence, body_width):
    alpha = SmoothingConfig.alpha_high_conf if confidence >= CONFIDENCE_THRESHOLD \
            else SmoothingConfig.alpha_low_conf
    
    # Outlier check
    velocity = distance(raw_t, prev_smoothed)
    if velocity > body_width * VELOCITY_OUTLIER_THRESHOLD_RATIO:
        return prev_smoothed, OUTLIER_FLAG
    
    smoothed = prev_smoothed * (1 - alpha) + raw_t * alpha
    return smoothed, NORMAL_FLAG
```

**Bidirectional smoothing optional**: forward + backward EMA averaged. Apply only after full video processed; keep forward-only as fallback.

### D. Left/Right identity stability

Hand crossing during swing (top of backswing, follow-through) causes raw pose runners to swap L/R. WHAM is better than 2D pose at this but not perfect.

```python
def stabilize_lr_identity(current_left, current_right, prev_left, prev_right):
    no_swap_distance = distance(current_left, prev_left) + distance(current_right, prev_right)
    swap_distance    = distance(current_left, prev_right) + distance(current_right, prev_left)
    
    if swap_distance < no_swap_distance * 0.7:
        return current_right, current_left, SWAPPED_FLAG
    return current_left, current_right, NORMAL_FLAG
```

Applied per-frame after raw extraction, before smoothing.

### E. Phase-aware constraints (**LOCKED — no uniform smoothing allowed**)

ChatGPT review §4: implementation must NOT apply same smoothing to all phases. Impact is fast (~50-100 ms); strong smoothing destroys impact motion fidelity. Setup is slow; weak smoothing leaves jitter.

| Phase | Smoothing α | Outlier sensitivity ratio | L/R identity |
| --- | --- | --- | --- |
| Setup | 0.20 (heavy) | 0.15 × bodyWidth (strict) | Locked, no swap |
| Backswing | 0.30 (medium) | 0.25 (moderate) | Strict swap detection |
| Top | 0.30 (medium) | 0.35 (loose for arm cross) | Trust previous identity |
| Transition | 0.30 (medium) | 0.35 (loose) | Maintain through cross |
| Downswing | 0.40 (light) | 0.40 (loose for speed) | Loose |
| Impact | 0.40 (light) | 0.40 | Loose |
| Finish | 0.30 (medium) | 0.30 (moderate) | Maintain |

**Implementation guardrail**: PR-7a/7c code review must verify per-frame phase lookup → config switch. A single "α = 0.3 for all frames" implementation is REJECTED.

Phase detection already exists in production (`phase_detector.py`). PR-7 reads phase per-frame and applies correct correction config.

### F. Output schema: `pose_timeline_3d_corrected` (v2 — with coaching_anchors)

ChatGPT review §3: separate analysis anchors from visual anchors so future visual tweaks don't pollute analysis precision.

```json
{
  "version": 1,
  "wham_source_version": "wham_vit_bedlam_w_3dpw_v1",
  "correction_version": "golf_correction_v1",
  "fps_native": 30.0,
  "video_width": 1280,
  "video_height": 720,
  "setup_baseline": {
    "setup_frame_idx": 12,
    "setup_ts": 0.4,
    "baseShoulderWidth": 0.52,
    "baseHipWidth": 0.41,
    "baseSpineLength": 0.78,
    "baseStanceWidth": 0.65,
    "baseSpineAngle_deg": 32.5
  },
  "correction_config": {
    "shoulder_inward_offset": 0.14,
    "hip_inward_offset": 0.16,
    "alpha_high_conf": 0.30,
    "velocity_outlier_threshold_ratio": 0.25
  },
  "frames": [
    {
      "ts": 0.0,
      "frame_idx": 0,
      "phase": "setup",
      "keypoints_3d_corrected": {
        "nose": [x, y, z],
        "head": [x, y, z],
        "left_shoulder": [x, y, z],
        "right_shoulder": [x, y, z],
        // ... 20 SMPL joints, post-offset + smoothed, for ANALYSIS use
      },
      "keypoints_2d_projected": {
        "nose": [px, py, depth_m],
        // ... 20 SMPL joints, same as above projected to image
      },
      "coaching_anchors_2d": {
        // Visual-overlay-specific anchors. May differ from keypoints_2d_projected
        // if the visual layer wants a slightly different position (e.g. shoulder
        // disc center prefers a midpoint between deltoid and acromion for
        // visual appeal vs strict anatomical midpoint).
        "left_shoulder_visual": [px, py],
        "right_shoulder_visual": [px, py],
        "shoulder_disc_center": [px, py],
        "left_hip_visual": [px, py],
        "right_hip_visual": [px, py],
        "hip_ring_center": [px, py],
        "head_center_visual": [px, py],
        "spine_top_visual": [px, py],
        "spine_bottom_visual": [px, py]
      },
      "correction_diagnostics": {
        "outlier_rejected": [],          # list of joint names rejected this frame
        "lr_swapped": false,
        "confidence_avg": 0.87
      }
    }
  ],
  "summary_stats": {
    "frames_with_outliers": 5,
    "outlier_rejection_rate": 0.04,
    "lr_swap_corrections": 2,
    "avg_drift_before_correction_px": 4.2,
    "avg_drift_after_correction_px": 1.3
  }
}
```

**Key principle**: `keypoints_3d_corrected` is the "truth" for analysis (angles, biomechanics). `coaching_anchors_2d` is what frontend draws. They start identical (PR-7c initial impl) and can diverge later if visual quality demands without polluting analysis.

---

## §4 File changes (sub-PR split v2)

PR-7 ships in **4 sub-PRs**. ChatGPT review §5: PR-7a scope tightened — Python module + offline output only, NO frontend.

### PR-7a — Correction layer module skeleton + setup calibration (Python only)

```
python/golf_correction/
├── __init__.py
├── config.py                ← SHOULDER_INWARD_OFFSET, ALPHA_*, thresholds, phase configs
├── setup_baseline.py        ← detect setup frame + extract baseline
├── anatomical_offset.py     ← per-joint inward offset functions
├── temporal_smoother.py     ← bidirectional EMA + outlier reject
├── lr_stability.py          ← left/right identity guard
├── phase_constraints.py     ← phase-aware config switch (LOCKED no uniform)
├── coaching_anchors.py      ← derive coaching_anchors_2d from keypoints_3d_corrected
└── golf_correction.py       ← orchestrator: raw WHAM → corrected

python/pilot/runners/wham_runner.py   ← MOD: emit raw pose_timeline_3d_wham (no production wire yet)
docs/PR-7a_OFFLINE_CORRECTION_REPORT.md  ← NEW: smoke output on b3fea3f0 with corrected timeline JSON
```

**Acceptance for 7a**:
- Module imports clean
- Setup baseline auto-detect runs on b3fea3f0 → plausible baseline values
- Per-frame correction produces corrected timeline JSON (read by Jason for visual sanity)
- **NO production schema written to DB yet** — offline JSON only
- **NO frontend changes** — `src/components/*.tsx`, `src/lib/*.ts` UNTOUCHED in 7a

7a deliverable = an offline corrected JSON Jason can inspect. Production wire happens in 7c.

### PR-7b — Empirical offset coefficient tuning + ground truth

CC writes the sweep harness; Jason hand-marks **15 ground-truth samples** (3 video × 5 phases × 5 keypoints); CC computes optimal offsets; locks into `config.py`.

```
python/golf_correction/scripts/offset_sweep.py   ← NEW: render N overlay variants
python/golf_correction/scripts/ground_truth_loader.py   ← NEW: read Jason's red-dot labels
docs/PR-7_OFFSET_TUNING_REPORT.md                 ← NEW: red-dot ground truth + sweep results
docs/PR-7_GROUND_TRUTH/                            ← NEW: Jason's annotations
  ├── b3fea3f0_setup.json
  ├── b3fea3f0_top.json
  ├── ... (15 total)
python/golf_correction/config.py                   ← MOD: final tuned coefficients
```

**Ground truth format** (Jason hand-fills using any annotation tool):

```json
{
  "video_id": "b3fea3f0-...",
  "phase": "setup",
  "frame_idx": 12,
  "video_width": 720,
  "video_height": 1280,
  "labels": {
    "left_shoulder":  {"x": 410, "y": 525},
    "right_shoulder": {"x": 510, "y": 525},
    "left_hip":       {"x": 425, "y": 720},
    "right_hip":      {"x": 495, "y": 720},
    "neck_center":    {"x": 460, "y": 450}
  }
}
```

**Acceptance for 7b**: tuned offsets produce mean px error per joint group ≤ §5 thresholds on the 15 ground-truth samples.

### PR-7c — Production integration

```
python/main.py                  ← MOD: dispatch raw WHAM result → golf_correction → corrected
python/Dockerfile               ← MOD: add python/golf_correction/ to COPY
python/requirements.txt         ← (no new deps; correction is pure numpy/scipy)
src/types/analysis.ts           ← MOD: add CorrectedKeypoint, CoachingAnchor types
src/components/SkeletonOverlay.tsx ← MOD: branch on pose_timeline_3d_corrected presence
src/components/SwingPlayer.tsx     ← MOD: disc anchors read coaching_anchors_2d
src/lib/disc/computeDiscParams.ts  ← MOD: use baseShoulderWidth + baseHipWidth from baseline
```

**Acceptance for 7c**:
- End-to-end analyze on b3fea3f0 writes all 3 schemas to DB
- Frontend disc renders from `coaching_anchors_2d`
- Skeleton overlay from `keypoints_3d_corrected`
- Visual smoke comparable to PR-6.1a era but with WHAM occlusion + correction improvements

### PR-7d — Acceptance gates + soak

Smoke-test all 3 test videos (b3fea3f0, a735cc7d, 5bbcfbc8), generate side-by-side comparisons of:
- Raw mediapipe_pose (legacy)
- Raw WHAM (pre-correction)
- Corrected WHAM with coaching anchors (PR-7 output)

Jason approves visual quality on all 3 swings × 5 phase samples. Outlier statistics within target.

If all pass → PR-7 merged. If specific phase fails (e.g. finish drift), iterate config.

---

## §5 Acceptance gates (v2 — relaxed per ChatGPT review §1)

| Gate | Threshold | Verification | Hard block? |
| --- | --- | --- | --- |
| Setup baseline auto-detected | Within first 0.5s of video, setup frame stable | Auto check | YES |
| **Shoulder / hip mean px error** | **< 10 px vs red-dot ground truth** | Sweep harness | YES |
| **Head / spine mean px error** | **< 12 px vs red-dot ground truth** | Sweep harness | YES |
| **Wrist / hand accuracy** | **Diagnostic only, not MVP gate** | Visual inspection | NO |
| Outlier rejection rate | < 10% per video | Diagnostic stats | YES |
| L/R swap correction | No visible mis-identity across full swing | Jason visual | YES |
| Drift after smoothing | < 2 px frame-to-frame avg in static phases | Auto stat output | YES |
| Disc using baseline | Disc width = baseShoulderWidth, NOT per-frame raw_shoulder_dist | Code review | YES |
| **Phase-aware smoothing applied** | Each phase shows distinct α value in correction log; no uniform 0.3 | Code review + diagnostic stats | YES |
| End-to-end analyze latency | < 90s (Railway + Modal + correction) | Production timing | YES |
| Frontend backward compat | Legacy pose_timeline_2d videos still render | Browser smoke | YES |

**v2 relaxation rationale** (ChatGPT §1): Jason's red-dot annotations themselves have ~3-5 px error. Hard-blocking on 8 px makes PR-7a/b chase the noise floor of the ground truth. 10/12 thresholds leave room for natural label noise + still represent meaningful improvement over surface-keypoint baseline. Tightening to 8 px is a PR-7.1 follow-up after MVP ships.

---

## §6 Configuration

`python/golf_correction/config.py`:

```python
# Empirically tuned per PR-7b sweep on Jason red-dot ground truth.
# Sweep range provided for future re-tuning.

ANATOMICAL_OFFSETS = {
    "shoulder_inward": 0.14,    # range 0.10-0.18
    "hip_inward":      0.16,    # range 0.12-0.20
    "head_inward":     0.08,    # range 0.05-0.12
    "knee_inward":     0.05,    # range 0.03-0.08
    "ankle_inward":    0.03,    # range 0.0-0.05
    "wrist_inward":    0.0,     # locked, hands on club
}

SMOOTHING = {
    "alpha_high_conf":         0.30,
    "alpha_low_conf":          0.10,
    "confidence_threshold":    0.65,
    "velocity_outlier_ratio":  0.25,
    "bidirectional_enabled":   True,
}

LR_STABILITY = {
    "swap_threshold_ratio": 0.70,
    "phase_strict_swap": ("setup", "backswing"),
}

# LOCKED phase-aware config — uniform smoothing not allowed
PHASE_CONFIG = {
    "setup":     {"alpha": 0.20, "outlier_ratio": 0.15},
    "backswing": {"alpha": 0.30, "outlier_ratio": 0.25},
    "top":       {"alpha": 0.30, "outlier_ratio": 0.35},
    "transition":{"alpha": 0.30, "outlier_ratio": 0.35},
    "downswing": {"alpha": 0.40, "outlier_ratio": 0.40},
    "impact":    {"alpha": 0.40, "outlier_ratio": 0.40},
    "finish":    {"alpha": 0.30, "outlier_ratio": 0.30},
}

COACHING_ANCHOR_TUNING = {
    # If visual layer wants different anchor than analysis layer in future,
    # tune here without affecting keypoints_3d_corrected.
    # First impl: coaching_anchors == keypoints_2d_projected. Divergence
    # allowed in PR-7.x follow-up if visual tweaks needed.
    "use_projected_as_coaching_anchor": True,
}
```

---

## §7 Estimated effort (v2 — by sub-PR)

| Sub-PR | Wall clock | Deliverable |
| --- | --- | --- |
| **7a** — Python module + setup baseline + offline corrected timeline (NO frontend) | 3-4 days | Module imports + corrected JSON for b3fea3f0 |
| **7b** — Ground truth labeling + offset sweep + tuning | 2-3 days | Jason: 15 samples annotated; CC: tuned config locked |
| **7c** — Production integration (Python pipeline + frontend wire) | 3-5 days | End-to-end disc renders from coaching_anchors |
| **7d** — Acceptance gates + soak | 1-2 days | All 3 videos pass v2 thresholds |
| **Total** | **9-14 days** | Production-grade Golf Correction Layer |

Parallel work:
- PR-7a starts immediately on CC; Jason simultaneously labels ground truth for PR-7b
- PR-7c blocks on PR-7a + PR-7b complete (needs tuned coefficients before production integration)

---

## §8 Strategic context (DO NOT DRIFT)

Per memory #22 + Verdict v2 §9 (updated):

- ✓ **DO**: Build correction layer on WHAM raw output
- ✓ **DO**: Treat WHAM as base motion signal, SwingCue correction as moat
- ✓ **DO**: Tune empirically against Jason red-dot ground truth on real golf swings
- ✓ **DO**: Keep correction layer model-agnostic — future SMPL family swap (Human3R etc.) reuses same layer
- ✓ **DO** (v2): Separate analysis anchors from visual anchors for future flexibility

- ✗ **DO NOT**: Search for "better anatomical bone-center model" — none exist for monocular RGB
- ✗ **DO NOT**: Pursue medical-grade anatomical accuracy as MVP blocker
- ✗ **DO NOT**: Deep-fork WHAM internals or retrain SMPL — config-level correction only
- ✗ **DO NOT**: Run Phase 2c Human3R / SMPLest-X / EasyMocap pilot — same family, same surface-anchor limit
- ✗ **DO NOT** (v2): Apply uniform smoothing across all phases — phase-aware is mandatory
- ✗ **DO NOT** (v2): Touch frontend in PR-7a — Python module + offline JSON output only in 7a

---

## §9 PR-7b ground truth labeling protocol (for Jason)

ChatGPT review §B: 15 samples min. Recommended labeling workflow:

### Tools

Any one of:
- VS Code with Image Viewer + manual coordinate noting in JSON
- Browser-based annotation tool (e.g., Roboflow free tier, makesense.ai)
- Custom: open frame in image editor, hover cursor, read pixel coords from status bar
- Simple Python script: matplotlib `ginput()` click capture

CC can write the click-capture script in PR-7b if helpful, but Jason chooses workflow.

### Frame selection (per video, 5 phases)

For each of `b3fea3f0`, `a735cc7d`, `5bbcfbc8`:

- **Setup**: first frame where player is settled, club grounded
- **Top**: club shaft at highest point in backswing
- **Transition**: club approaching parallel to ground on downswing
- **Impact**: moment of ball contact (or closest frame; can be 1-2 frames before/after)
- **Finish**: club shaft past shoulder, body rotated to target

Phase detector in production already outputs these — Jason can use phase_detector output as suggestion, then refine.

### Labels per frame (5 keypoints first pass)

```json
{
  "video_id": "b3fea3f0-e248-44d7-a923-0bb43172b5bf",
  "phase": "setup",
  "frame_idx": 12,
  "video_width": 720,
  "video_height": 1280,
  "labels": {
    "left_shoulder":  {"x": 410, "y": 525},   // pixel coords, top-left origin
    "right_shoulder": {"x": 510, "y": 525},
    "left_hip":       {"x": 425, "y": 720},
    "right_hip":      {"x": 495, "y": 720},
    "neck_center":    {"x": 460, "y": 450}    // (or chest_center, jason's choice)
  }
}
```

Save as `docs/PR-7_GROUND_TRUTH/{video_id}_{phase}.json`. 15 files total minimum.

Optional pass 2 (after first sweep complete): add `head_crown`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle` for finer tuning.

### What "left" and "right" mean here

ChatGPT §B: avoid ambiguity. Use **body-frame** (golfer's anatomical left/right), NOT screen-frame:

- `left_shoulder` = the shoulder of golfer's LEFT arm (the lead arm for a right-handed golfer)
- `right_shoulder` = the shoulder of golfer's RIGHT arm (the trail arm)

For a right-handed golfer facing the camera (down-the-line view from behind), the golfer's left arm appears on the RIGHT side of the image.

If Jason finds this confusing, alternative `screen_left_shoulder` / `screen_right_shoulder` is acceptable for 7b; CC writes mapping logic in PR-7b's ground_truth_loader.py.

---

## §10 Memory of where we landed

PR-7 represents SwingCue's actual product moat: not pose model selection (commoditized), but golf-specific calibration + smoothing stack on top of mature 3D body pose. Anyone can call WHAM Modal endpoint; only SwingCue learns from Jason's red-dot ground truth what the per-joint offset should be for golf swing analysis at setup/impact/finish anatomical anchor points.

After PR-7 merges, the architecture sequence is:

```
Video → mediapipe/rtmpose 2D surface (legacy/fast) +
        WHAM 3D motion (Modal GPU)              +
        Golf Correction Layer (this PR)        →
        Disc/Overlay frontend (reads coaching_anchors_2d)
```

Three parallel signal sources, two corrected outputs (analysis vs visual), one frontend layer. Clean separation of concerns.
