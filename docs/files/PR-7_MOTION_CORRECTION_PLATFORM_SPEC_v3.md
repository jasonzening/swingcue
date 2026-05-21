# PR-7: Motion Correction Platform — Golf as First Plugin (v3)

**Status**: Spec v3 (Platform refactor — supersedes v2)
**Date**: 2026-05-20
**Predecessor**: Spec v2 (golf-specific, ChatGPT-reviewed) — superseded
**Strategic frame**: Build a generic motion correction engine + sport-agnostic plugin architecture. Golf is the first domain plugin. Tennis / ski / PT / ergonomics are future plugins. The engine + the per-domain ground-truth datasets are SwingCue Inc.'s technical moat.

**v3 changes vs v2**:
- §2 architecture refactored: generic Engine + Domain Plugin layers
- §3 module layout: `python/motion_correction/{engine,domains,schemas}` (was `python/golf_correction/`)
- §4 NEW: `DomainPlugin` abstract interface + golf as reference implementation
- §6 schemas: add `sport` + `view` fields; output schema sport-agnostic
- §9 ground truth labeler: `--sport` flag + per-sport keypoint set
- §11 NEW: future plugins roadmap (tennis/baseball/ski/running/PT/industrial)
- §12 NEW: multi-view support (face_on / down_the_line) — coefficient sweep per view
- §13 NEW: migration path v2 → v3

---

## §1 What this PR is and is NOT

### IS
- A **generic, sport-agnostic motion correction engine** layered on top of any SMPL-family 3D pose backbone (WHAM today, Human3R/SMPLest-X tomorrow if needed)
- A **plug-in architecture** where each sport / domain registers its own phase taxonomy, anatomical offsets, coaching anchors, and (optional) equipment detector
- **Golf as the first reference implementation** — proves the platform pattern, ships SwingCue MVP
- Three corrected output schemas:
  - `pose_timeline_3d_corrected` (per-frame corrected SMPL joints, sport-agnostic)
  - `coaching_anchors_2d` (per-sport visual anchors — defined by domain plugin)
  - `analysis_metrics` (per-sport derived metrics — e.g. golf hip-shoulder separation, tennis racket head speed)
- Configurable per (sport × view × phase × joint) — empirically tuned via ground truth

### IS NOT
- A search for a different pose backbone. **WHAM stays as the default 3D source** for Phase 2.
- A pursuit of medical-grade anatomical bone-center accuracy
- A frontend disc redesign (current overlay logic mostly unchanged, just consumes new schema)
- A multi-sport product launch (golf-only MVP ships first; other plugins are platform proof-of-concepts)
- A premature B2B SDK release (engine matures via golf → tennis cycle first, then SDK packaging)

### Strategic principle (DO NOT FORGET)
> **The engine is reusable. The ground truth is the moat.** Anyone can copy this architecture. Only SwingCue accumulates Jason-labeled red-dot ground truth for golf, tennis, ski, PT, etc. That dataset compounds and is non-replicable.

---

## §2 Architectural placement

```
┌─────────────────────────────────────────────────────────────────────┐
│  Video → Pose Backbone (Layer 1, pluggable)                         │
│  ├── WHAM Modal endpoint        (default, SMPL 24 joints)           │
│  ├── RTMPose                    (PR-6.1a, 2D surface, transitional) │
│  └── Future: Human3R / SMPLest-X / 4D-Humans                        │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 2: Generic Motion Correction Engine                          │
│  ├── Setup baseline detection (sport-agnostic)                      │
│  ├── Anatomical offset framework (per-joint, per-view)              │
│  ├── Temporal smoother (phase-aware EMA + outlier reject)           │
│  ├── L/R identity guard                                             │
│  ├── View-aware projection (face_on / down_the_line / side)         │
│  └── Orchestrator: raw_timeline → corrected_timeline                │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 3: Domain Plugins (sport-specific)                           │
│  ┌───────────┬─────────┬──────────┬────────┬───────────┬─────────┐  │
│  │  Golf 🟢  │ Tennis ⚪│ Ski ⚪   │  PT ⚪  │ Industry ⚪│ ...     │  │
│  │ (PR-7)    │ (P3)    │ (P3)     │ (P4)   │ (P5)      │         │  │
│  └───────────┴─────────┴──────────┴────────┴───────────┴─────────┘  │
│  Each plugin provides:                                              │
│  - phase_taxonomy           (list of phase names)                   │
│  - keypoint_subset          (which SMPL joints this sport uses)     │
│  - offset_config            (per-view per-joint sweep-tuned values) │
│  - smoothing_config         (per-phase α + outlier thresholds)      │
│  - coaching_anchor_namespace (visual anchors for overlay)           │
│  - phase_detector           (sport-specific phase boundary logic)   │
│  - analysis_metrics         (sport-specific derived measures)       │
│  - equipment_detector       (optional, future: club/racket/ski/...) │
└─────────────────────────────────────────────────────────────────────┘
```

Layer 2 + Layer 3 together = PR-7 deliverable.
Plugins for sports beyond golf are explicit future scope (§11).

---

## §3 Module structure

```
python/motion_correction/
├── __init__.py
├── engine/                          ← Layer 2: sport-agnostic core
│   ├── __init__.py
│   ├── setup_baseline.py            ← detect stable baseline window
│   ├── anatomical_offset.py         ← per-joint inward offset framework
│   ├── temporal_smoother.py         ← phase-aware EMA + outlier reject
│   ├── lr_stability.py              ← left/right identity guard
│   ├── view_aware.py                ← view-dependent offset selection
│   ├── projection.py                ← 3D → 2D with camera params
│   └── orchestrator.py              ← assembles full pipeline
├── domains/                          ← Layer 3: per-sport plugins
│   ├── __init__.py
│   ├── base.py                      ← DomainPlugin abstract class
│   ├── registry.py                  ← name → plugin lookup
│   └── golf/                         ← FIRST PLUGIN (PR-7 deliverable)
│       ├── __init__.py
│       ├── plugin.py                ← GolfPlugin(DomainPlugin)
│       ├── phases.py                ← setup/top/transition/impact/finish
│       ├── phase_detector.py        ← per-frame phase classifier
│       ├── config.py                ← offsets / smoothing per view per phase
│       ├── coaching_anchors.py      ← shoulder_disc, hip_ring derivations
│       ├── analysis_metrics.py      ← hip-shoulder separation, COM transfer
│       └── equipment.py             ← (future: club head detector)
├── schemas/
│   ├── __init__.py
│   ├── corrected_timeline.py        ← pose_timeline_3d_corrected
│   ├── coaching_anchors.py          ← coaching_anchors_2d
│   ├── analysis_metrics.py          ← per-sport metrics envelope
│   └── ground_truth.py              ← Jason red-dot label schema (sport-aware)
└── tests/
    ├── engine/                       ← sport-agnostic test suite
    └── domains/golf/                 ← golf-specific test fixtures

python/pilot/scripts/
└── ground_truth_labeler.py           ← MODIFIED: --sport flag, per-sport keypoint set

python/pilot/runners/wham_runner.py   ← MODIFIED: emit raw pose_timeline_3d_wham only
                                         (no golf-specific logic)

src/types/analysis.ts                 ← NEW types: CorrectedKeypoint, CoachingAnchor (generic)
src/components/SkeletonOverlay.tsx    ← MODIFIED: branch on corrected schema
src/components/SwingPlayer.tsx        ← MODIFIED: disc anchors read coaching_anchors_2d

docs/files/
├── PR-7_MOTION_CORRECTION_PLATFORM_SPEC_v3.md   ← this file
├── PR-7_GROUND_TRUTH_PROTOCOL.md                ← labeling protocol (sport-aware)
└── domains/
    └── GOLF_DOMAIN_PLUGIN_SPEC.md               ← golf-specific design doc

docs/PR-7_GROUND_TRUTH/
├── golf/                             ← Jason's golf red-dot labels
│   ├── b3fea3f0_setup_face_on.json
│   ├── b3fea3f0_top_face_on.json
│   └── ... (3 video × 5 phase × 2 view = 30 samples min)
├── tennis/                           ← (future PR-8)
└── README.md                         ← per-sport labeling instructions
```

---

## §4 DomainPlugin abstract interface

`python/motion_correction/domains/base.py`:

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class PhaseSpec:
    """A swing/motion phase definition for a sport."""
    name: str                      # "setup", "backswing", "top", ...
    is_static: bool                # phase is mostly stationary (relevant to smoothing strength)
    typical_duration_s: float      # for outlier detection scaling
    requires_lr_lock: bool         # L/R identity strict in this phase


@dataclass
class CoachingAnchorSpec:
    """A visual anchor (for overlay rendering) that the plugin derives
    from corrected keypoints."""
    name: str                      # "shoulder_disc_center", "hip_ring_center", ...
    source_keypoints: list[str]    # SMPL joint names this anchor is derived from
    derivation_rule: str           # e.g. "midpoint(left_shoulder, right_shoulder)"


@dataclass
class AnalysisMetricSpec:
    """A sport-specific derived measurement for the analysis layer."""
    name: str                      # "hip_shoulder_separation", "racket_head_speed", ...
    unit: str                      # "degrees", "m/s", "ratio", ...
    computation: str               # function name in domain's analysis_metrics module


class DomainPlugin(ABC):
    """
    Abstract base for all sport / domain plugins.

    Subclass: see python/motion_correction/domains/golf/plugin.py
    """

    # ── Class attributes (each plugin overrides) ──────────────────
    sport_name: str = ""                       # "golf", "tennis", "ski"
    plugin_version: str = ""                   # "golf_v1"

    # Which SMPL joints this domain cares about (subset of 24).
    # Used to filter labeler keypoint options + skip unused joints in correction.
    keypoint_subset: list[str] = []

    # Phase taxonomy in chronological order.
    phase_taxonomy: list[PhaseSpec] = []

    # Coaching anchors this plugin produces for the frontend overlay.
    coaching_anchor_namespace: list[CoachingAnchorSpec] = []

    # Sport-specific derived metrics.
    analysis_metric_namespace: list[AnalysisMetricSpec] = []

    # ── Required methods ──────────────────────────────────────────

    @abstractmethod
    def detect_phases(
        self,
        raw_timeline: dict,
        view: str,
    ) -> list[tuple[int, int, str]]:
        """
        Per-frame phase classification.

        Args:
            raw_timeline: pose_timeline_3d_wham output (frames + keypoints).
            view: "face_on" | "down_the_line" | "side" | "back".

        Returns:
            List of (start_frame, end_frame, phase_name) tuples covering all frames.
            phase_name must be in self.phase_taxonomy.
        """
        ...

    @abstractmethod
    def get_offset_config(self, view: str) -> dict[str, float]:
        """
        View-specific anatomical offset coefficients.

        Returns:
            {"shoulder_inward": 0.14, "hip_inward": 0.16, ...}
            Keys are joint groups in self.keypoint_subset.
            Values are sweep-tuned via PR-7b empirical optimization.
        """
        ...

    @abstractmethod
    def get_smoothing_config(self, phase: str) -> dict[str, float]:
        """
        Phase-aware smoothing + outlier rejection thresholds.

        Returns:
            {"alpha": 0.30, "outlier_ratio": 0.25, "alpha_low_conf": 0.10}
        """
        ...

    @abstractmethod
    def compute_coaching_anchors(
        self,
        corrected_keypoints_2d: dict,
    ) -> dict[str, tuple[float, float]]:
        """
        Derive visual anchors for the frontend overlay layer from corrected
        keypoints. Each anchor name must be in self.coaching_anchor_namespace.
        """
        ...

    @abstractmethod
    def compute_analysis_metrics(
        self,
        corrected_timeline: dict,
    ) -> dict[str, float]:
        """
        Derive sport-specific scalar measurements (one set per swing/session).
        """
        ...

    # ── Optional methods ──────────────────────────────────────────

    def equipment_detector(self):
        """
        Optional sport equipment object detector (club, racket, bat, ski, etc.).
        Returns None if this domain doesn't track equipment.
        Future scope — not required for PR-7.
        """
        return None

    def validate_ground_truth_label(
        self,
        label_record: dict,
    ) -> tuple[bool, Optional[str]]:
        """
        Per-sport sanity check on a labeler output JSON. Default impl
        checks keypoint names are in self.keypoint_subset.
        """
        for kp_name in label_record.get("labels", {}):
            if kp_name not in self.keypoint_subset:
                return False, f"unknown keypoint for {self.sport_name}: {kp_name}"
        return True, None


# Registry pattern — plugins self-register on import.
_REGISTRY: dict[str, DomainPlugin] = {}


def register_plugin(plugin: DomainPlugin) -> None:
    if plugin.sport_name in _REGISTRY:
        raise ValueError(f"plugin {plugin.sport_name} already registered")
    _REGISTRY[plugin.sport_name] = plugin


def get_plugin(sport_name: str) -> DomainPlugin:
    if sport_name not in _REGISTRY:
        raise KeyError(
            f"no plugin for sport={sport_name!r}; "
            f"registered: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[sport_name]


def list_plugins() -> list[str]:
    return list(_REGISTRY.keys())
```

---

## §5 Golf plugin (reference implementation)

`python/motion_correction/domains/golf/plugin.py`:

```python
from motion_correction.domains.base import (
    DomainPlugin, PhaseSpec, CoachingAnchorSpec, AnalysisMetricSpec,
    register_plugin,
)
from . import config, phases, phase_detector, coaching_anchors, analysis_metrics


class GolfPlugin(DomainPlugin):
    sport_name = "golf"
    plugin_version = "golf_v1"

    keypoint_subset = [
        # Head
        "head", "neck",
        # Upper body
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        # Torso
        "spine1", "spine2", "spine3",
        # Lower body
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_foot", "right_foot",
        "pelvis",
    ]

    phase_taxonomy = [
        PhaseSpec("setup",      is_static=True,  typical_duration_s=2.0, requires_lr_lock=True),
        PhaseSpec("backswing",  is_static=False, typical_duration_s=0.8, requires_lr_lock=True),
        PhaseSpec("top",        is_static=True,  typical_duration_s=0.15, requires_lr_lock=False),
        PhaseSpec("transition", is_static=False, typical_duration_s=0.20, requires_lr_lock=False),
        PhaseSpec("downswing",  is_static=False, typical_duration_s=0.25, requires_lr_lock=False),
        PhaseSpec("impact",     is_static=False, typical_duration_s=0.05, requires_lr_lock=False),
        PhaseSpec("finish",     is_static=False, typical_duration_s=1.0,  requires_lr_lock=False),
    ]

    coaching_anchor_namespace = [
        CoachingAnchorSpec(
            "shoulder_disc_center", ["left_shoulder", "right_shoulder"],
            "midpoint",
        ),
        CoachingAnchorSpec(
            "hip_ring_center", ["left_hip", "right_hip"],
            "midpoint",
        ),
        CoachingAnchorSpec(
            "spine_top", ["neck"], "identity",
        ),
        CoachingAnchorSpec(
            "spine_bottom", ["pelvis"], "identity",
        ),
        CoachingAnchorSpec(
            "left_shoulder_visual", ["left_shoulder"], "with_offset",
        ),
        CoachingAnchorSpec(
            "right_shoulder_visual", ["right_shoulder"], "with_offset",
        ),
        CoachingAnchorSpec(
            "left_hip_visual", ["left_hip"], "with_offset",
        ),
        CoachingAnchorSpec(
            "right_hip_visual", ["right_hip"], "with_offset",
        ),
    ]

    analysis_metric_namespace = [
        AnalysisMetricSpec("hip_shoulder_separation_at_top", "degrees",
                          "analysis_metrics.hip_shoulder_separation"),
        AnalysisMetricSpec("hip_turn_at_impact", "degrees",
                          "analysis_metrics.hip_turn_at_impact"),
        AnalysisMetricSpec("spine_tilt_change_setup_to_impact", "degrees",
                          "analysis_metrics.spine_tilt_change"),
        AnalysisMetricSpec("club_path_direction", "degrees",
                          "analysis_metrics.club_path"),  # requires equipment_detector
    ]

    def detect_phases(self, raw_timeline, view):
        return phase_detector.detect(raw_timeline, view)

    def get_offset_config(self, view):
        return config.ANATOMICAL_OFFSETS[view]

    def get_smoothing_config(self, phase):
        return config.PHASE_CONFIG[phase]

    def compute_coaching_anchors(self, corrected_keypoints_2d):
        return coaching_anchors.derive(corrected_keypoints_2d, self.get_offset_config)

    def compute_analysis_metrics(self, corrected_timeline):
        return analysis_metrics.compute_all(corrected_timeline)


# Auto-register on import.
register_plugin(GolfPlugin())
```

Golf-specific config lives entirely in `python/motion_correction/domains/golf/`. Nothing about "golf" leaks into `engine/`.

---

## §6 Output schemas (sport-agnostic envelope)

### `pose_timeline_3d_corrected` (engine output)

```json
{
  "version": 1,
  "sport": "golf",
  "domain_plugin_version": "golf_v1",
  "pose_backbone": "wham_vit_bedlam_w_3dpw_v1",
  "view": "face_on",
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
    "offset_values_used": {...},      // per the locked sweep-tuned values
    "smoothing_alpha_per_phase": {...},
    "outlier_threshold_per_phase": {...}
  },
  "frames": [
    {
      "ts": 0.0,
      "frame_idx": 0,
      "phase": "setup",
      "keypoints_3d_corrected": {
        "head": [x, y, z],
        "left_shoulder": [x, y, z],
        // ... SMPL joints in keypoint_subset (per-plugin)
      },
      "keypoints_2d_projected": {
        "head": [px, py],
        // ... same set, projected to image pixel coords
      },
      "coaching_anchors_2d": {
        // Plugin-derived visual anchors. Names are from
        // plugin.coaching_anchor_namespace.
        "shoulder_disc_center": [px, py],
        "hip_ring_center": [px, py],
        "left_shoulder_visual": [px, py],
        // ...
      },
      "correction_diagnostics": {
        "outlier_rejected": [],
        "lr_swapped": false,
        "confidence_avg": 0.87
      }
    }
  ],
  "analysis_metrics": {
    // Plugin-computed scalar metrics. Names are from
    // plugin.analysis_metric_namespace.
    "hip_shoulder_separation_at_top": 38.5,
    "spine_tilt_change_setup_to_impact": -3.2,
    // ...
  },
  "summary_stats": {
    "frames_with_outliers": 5,
    "outlier_rejection_rate": 0.04,
    "lr_swap_corrections": 2,
    "avg_drift_before_correction_px": 4.2,
    "avg_drift_after_correction_px": 1.3
  }
}
```

**Key**: `sport` field is required. Frontend dispatches rendering logic per sport. Coaching anchors and analysis metrics are namespaced by sport so future plugins can add new ones without breaking existing schema.

---

## §7 Ground truth schema (sport + view aware)

```json
{
  "schema_version": "v3",
  "sport": "golf",
  "video_id": "b3fea3f0-e248-44d7-a923-0bb43172b5bf",
  "phase": "setup",
  "frame_idx": 7,
  "view": "face_on",
  "video_width": 720,
  "video_height": 1280,
  "labels": {
    "left_shoulder":  {"x": 387, "y": 476},
    "right_shoulder": {"x": 535, "y": 465},
    "left_hip":       {"x": 410, "y": 610},
    "right_hip":      {"x": 501, "y": 616},
    "neck_center":    {"x": 447, "y": 371}
  },
  "labeler_version": "labeler_v2",
  "labeled_at": "2026-05-20T22:30:00Z"
}
```

**Required new fields vs v2**:
- `schema_version` — for future migration
- `sport` — explicit, since labeler is multi-sport
- `view` — face_on / down_the_line / side / back (per spec v3 §12)
- `labeler_version` — Tk labeler v2 emits this

File location: `docs/PR-7_GROUND_TRUTH/<sport>/<video_id>_<phase>_<view>.json`

---

## §8 Acceptance gates (v3 — same as v2 + plugin pattern conformance)

| Gate | Threshold | Verification | Hard block? |
| --- | --- | --- | --- |
| Setup baseline auto-detected | Within first 0.5s of video, stable | Auto check | YES |
| Shoulder / hip mean px error | < 10 px vs red-dot ground truth | Sweep harness | YES |
| Head / spine mean px error | < 12 px vs red-dot ground truth | Sweep harness | YES |
| Wrist / hand accuracy | Diagnostic only, not MVP gate | Visual inspection | NO |
| Outlier rejection rate | < 10% per video | Diagnostic stats | YES |
| L/R swap correction | No visible mis-identity across full swing | Jason visual | YES |
| Drift after smoothing | < 2 px frame-to-frame avg in static phases | Auto stat output | YES |
| Disc using baseline | Disc width = baseShoulderWidth, NOT raw_shoulder_dist | Code review | YES |
| Phase-aware smoothing applied | Each phase shows distinct α value in correction log; no uniform 0.3 | Code review + stats | YES |
| End-to-end analyze latency | < 90s (Railway + Modal + correction) | Production timing | YES |
| Frontend backward compat | Legacy pose_timeline_2d videos still render | Browser smoke | YES |
| **NEW v3: Plugin pattern conformance** | `engine/` modules contain ZERO references to "golf" string; all sport logic lives in `domains/golf/` | grep audit | YES |
| **NEW v3: Multi-view sweep coverage** | Both face_on AND down_the_line offsets sweep-tuned (different values acceptable) | Tuning report | YES |
| **NEW v3: Ground truth sport-namespaced** | Labels under `docs/PR-7_GROUND_TRUTH/golf/`, not bare directory | Layout audit | YES |

---

## §9 Sub-PR split (v3 — same 4 sub-PRs, scope adjusted for platform)

### PR-7a — Engine + DomainPlugin abstract + Golf plugin skeleton (Python only)

```
python/motion_correction/engine/                     ← NEW: all sport-agnostic
python/motion_correction/domains/__init__.py         ← NEW: registry
python/motion_correction/domains/base.py             ← NEW: DomainPlugin ABC
python/motion_correction/domains/golf/plugin.py      ← NEW: GolfPlugin
python/motion_correction/domains/golf/phases.py
python/motion_correction/domains/golf/phase_detector.py
python/motion_correction/domains/golf/config.py      ← starting offset estimates
python/motion_correction/domains/golf/coaching_anchors.py
python/motion_correction/domains/golf/analysis_metrics.py
python/motion_correction/schemas/                    ← NEW: pydantic models
python/motion_correction/tests/                      ← test suite skeleton

python/pilot/runners/wham_runner.py                  ← MOD: emit raw schema only
docs/PR-7a_OFFLINE_CORRECTION_REPORT.md              ← smoke output JSON for b3fea3f0
```

**Acceptance for 7a**:
- `motion_correction` package imports clean, all unit tests pass
- `GolfPlugin` registered + retrievable via `domains.registry.get_plugin("golf")`
- Setup baseline auto-detect runs on b3fea3f0 → plausible baseline values
- Per-frame correction produces corrected timeline JSON for Jason offline inspection
- **NO production schema written to DB yet** — offline JSON only
- **NO frontend changes** — `src/components/*.tsx`, `src/lib/*.ts` UNTOUCHED in 7a
- **Plugin pattern conformance**: `engine/` contains zero "golf" string references (grep audit)

### PR-7b — Ground truth labeling (multi-view) + offset sweep + tuning

```
python/pilot/scripts/ground_truth_labeler.py         ← MOD: --sport flag, per-sport keypoint set
python/motion_correction/scripts/offset_sweep.py     ← NEW: per-view sweep harness
python/motion_correction/scripts/ground_truth_loader.py
docs/PR-7_OFFSET_TUNING_REPORT.md                    ← per-view sweep results
docs/PR-7_GROUND_TRUTH/golf/                          ← Jason's labels (30 files)
  ├── b3fea3f0_setup_face_on.json
  ├── b3fea3f0_top_face_on.json
  ├── ... (5 phases × face_on + 5 phases × down_the_line per video)
python/motion_correction/domains/golf/config.py      ← MOD: locked sweep-tuned values
```

**Acceptance for 7b**:
- Labeler `--sport=golf --view=face_on` flag works end-to-end
- Jason hand-labels **30 ground-truth samples** (3 video × 5 phase × 2 view)
- Sweep harness produces per-view offset values minimizing mean px error
- Tuned offsets satisfy gates: shoulder/hip < 10 px, head/spine < 12 px
- Golf `config.py` has `ANATOMICAL_OFFSETS["face_on"]` + `ANATOMICAL_OFFSETS["down_the_line"]` populated

### PR-7c — Production integration

```
python/main.py                       ← MOD: dispatch raw WHAM result → motion_correction → corrected
python/Dockerfile                    ← MOD: add motion_correction/ to COPY
src/types/analysis.ts                ← MOD: add Sport, View, CorrectedKeypoint, CoachingAnchor types
src/components/SkeletonOverlay.tsx   ← MOD: branch on pose_timeline_3d_corrected presence
src/components/SwingPlayer.tsx        ← MOD: disc anchors read coaching_anchors_2d
src/lib/disc/computeDiscParams.ts    ← MOD: use baseShoulderWidth from baseline
src/lib/sports/golfRenderer.ts       ← NEW: golf-specific frontend rendering rules
```

**Acceptance for 7c**:
- End-to-end analyze on b3fea3f0 writes all 3 schemas to DB
- Frontend disc renders from `coaching_anchors_2d` (not raw keypoints)
- Skeleton overlay from `keypoints_3d_corrected`
- View detection from `video.view_type` dispatches correct offset config
- Visual smoke comparable to PR-6.1a era but with correction improvements

### PR-7d — Acceptance gates + soak

Smoke-test all 3 test videos × 2 views × 5 phases, generate side-by-side comparisons:
- Raw mediapipe_pose (legacy)
- Raw WHAM (pre-correction)
- Corrected WHAM (PR-7 output)

Jason approves visual quality. Outlier statistics within target.

---

## §10 Configuration (v3 — multi-view structure)

`python/motion_correction/domains/golf/config.py`:

```python
"""
Golf-domain configuration.

Sweep-tuned per (view × joint × phase). Tuning protocol: PR-7b
empirical optimization against Jason red-dot ground truth.
"""

# Per-view anatomical offset coefficients.
# Sweep ranges shown for future re-tuning.
ANATOMICAL_OFFSETS = {
    "face_on": {
        "shoulder_inward": 0.14,    # sweep 0.10-0.18
        "hip_inward":      0.16,    # sweep 0.12-0.20
        "head_inward":     0.08,    # sweep 0.05-0.12
        "knee_inward":     0.05,
        "ankle_inward":    0.03,
        "wrist_inward":    0.0,     # locked, hands on club
    },
    "down_the_line": {
        "shoulder_inward": 0.18,    # typically higher than face_on (occlusion-aware)
        "hip_inward":      0.20,
        "head_inward":     0.08,
        "knee_inward":     0.05,
        "ankle_inward":    0.03,
        "wrist_inward":    0.0,
    },
    # Future: "side", "back" if added (out of MVP scope per §12)
}

# Phase-aware smoothing + outlier thresholds (LOCKED — no uniform allowed).
PHASE_CONFIG = {
    "setup":      {"alpha": 0.20, "outlier_ratio": 0.15},
    "backswing":  {"alpha": 0.30, "outlier_ratio": 0.25},
    "top":        {"alpha": 0.30, "outlier_ratio": 0.35},
    "transition": {"alpha": 0.30, "outlier_ratio": 0.35},
    "downswing":  {"alpha": 0.40, "outlier_ratio": 0.40},
    "impact":     {"alpha": 0.40, "outlier_ratio": 0.40},
    "finish":     {"alpha": 0.30, "outlier_ratio": 0.30},
}

# Smoothing common parameters.
SMOOTHING = {
    "alpha_low_conf":         0.10,   # used when keypoint confidence < threshold
    "confidence_threshold":   0.65,
    "velocity_outlier_ratio": 0.25,
    "bidirectional_enabled":  True,
}

# L/R identity stability.
LR_STABILITY = {
    "swap_threshold_ratio": 0.70,
    "phase_strict_swap": ("setup", "backswing"),
}
```

---

## §11 Future plugins roadmap (Phase 3+, NOT in PR-7 scope)

| Sport / Domain | Plugin scope | Tier | Est. effort once engine ships | Notes |
| --- | --- | --- | --- | --- |
| **Tennis** | Body + grip + racket detector | T1+T2+equipment | 1-2 weeks | Highest reuse of golf plugin patterns; serve = analog of swing |
| **Baseball batting** | Body + grip + bat detector | T1+T2+equipment | 1-2 weeks | Same family as golf/tennis swing analysis |
| **Baseball pitching** | Body + finger + ball release point | T1+T2 | 2-3 weeks | Release point + elbow torque are new analysis metrics |
| **Ski / Snowboard** | Body + equipment-as-keypoint | T1+special | 3-4 weeks | Skis treated as additional keypoint set; flight angle metrics novel |
| **Running / Gait** | Body + foot strike detail | T1 | 1-2 weeks | Phase = stride cycle (not swing-based); huge consumer market |
| **Yoga / Pilates** | Body, full SMPL alignment | T1 | 1-2 weeks | Static-pose oriented; setup baseline IS the analysis target |
| **Fitness (squat/deadlift)** | Body + barbell endpoints | T1+equipment | 1-2 weeks | Lower-body biomechanics + bar path |
| **Boxing / MMA** | Body + hands (SMPL-H) | T2 | 2-3 weeks | Impact moment detection; speed estimation |
| **Physical Therapy / Rehab** | Body + joint ROM precision | T1 high precision | 4-6 weeks | Regulatory (HIPAA), per-condition phase taxonomy |
| **Fall detection (elder)** | Simplified body | T1 simplified | 2-3 weeks | Event detection vs continuous motion; different output schema |
| **Industrial ergonomics (REBA/RULA)** | Body + lifted-object | T1+object | 4-6 weeks | OSHA compliance scoring; existing competitor (Protex/TuMeke) — differentiate via better correction |
| **Driver monitoring** | Upper body + face | T3 face-heavy | 4-6 weeks | Different output (fatigue scoring vs motion correction) |
| **Music performance** | Hand-detailed (MANO) | T4 | 6-8 weeks | Out of WHAM scope, needs different pose backbone |
| **VFX / animation** | Body + hand + face (SMPL-X) | T3 full expressive | Existing market — Move.ai/Radical compete | Lower priority due to existing solutions |
| **Animal (equine/canine)** | Different skeleton model | T6 | Major scope | Out of MVP — needs animal-specific pose backbone |

**Phase 3 starts when**: golf MVP ships + ground truth collection workflow proven + at least 50 golf swings labeled across multiple users.

**Plugin add cost** (after engine matures via golf): ~1-4 weeks per sport, depending on whether equipment detector + custom phase logic needed. Tennis is the natural Phase 3 first plugin (highest pattern reuse).

---

## §12 Multi-view support (v3 critical addition)

Per the SwingCue user record realities discussed previously:

| View | Frequency | Calibration MVP scope |
| --- | --- | --- |
| **face_on** | ~50% | ✅ MVP required |
| **down_the_line** | ~45% | ✅ MVP required |
| side | ~3% | Skip MVP |
| back | ~1% | Skip MVP |
| top-down | <1% | Skip MVP |

**View detection**: production already has `swing_videos.view_type` field populated by analyze pipeline (existing). Engine reads this field to dispatch correct offset config.

**Per-view tuning rationale**: face_on shoulder offset and down_the_line shoulder offset are NOT the same value. In face_on, both shoulders visible — offset only corrects deltoid surface bias. In down_the_line, far shoulder is occluded — WHAM SLAM infers position with different anatomical bias pattern. A single offset value across views over-fits one and under-fits the other.

**Ground truth labeling effort impact**: 15 samples (v2) → 30 samples (v3, 2 views) → +30-45 min Jason labeling time.

---

## §13 Migration path v2 → v3

Spec v2 has been written to `docs/files/PR-7_GOLF_CORRECTION_LAYER_SPEC_v2.md` (committed in `15668f2`). v3 supersedes v2 but v2 remains in git history for reference.

**No code has been written against v2 yet** (PR-7a not started). All implementation starts from v3.

**Schema migration**: not applicable — `pose_timeline_3d_corrected` only ships in PR-7c. v3 schema is what production sees from day 1.

**Labeler migration**: existing `ground_truth_labeler.py` (commit `3ac0028`) is v2-compatible (single-sport, single-view). PR-7b modifies it to add `--sport` and `--view` flags. Existing 1 label JSON (`b3fea3f0_setup.json`) is golf-face_on data — easily renamed/moved to `docs/PR-7_GROUND_TRUTH/golf/b3fea3f0_setup_face_on.json`.

---

## §14 Estimated effort (v3)

| Sub-PR | Wall clock | Deliverable |
| --- | --- | --- |
| **7a** — Engine + DomainPlugin ABC + GolfPlugin + offline JSON | 4-5 days | Generic + golf-specific code complete, offline corrected JSON for inspection |
| **7b** — Multi-view labeler + sweep + tuning | 2-3 days (CC) + 60-90 min (Jason labeling) | 30-sample ground truth + sweep report + locked config |
| **7c** — Production integration (Python + frontend) | 3-5 days | End-to-end disc renders, view-aware dispatch |
| **7d** — Acceptance gates + soak across views | 1-2 days | All v3 gates green on 3 videos × 2 views |
| **Total** | **10-15 days** | Production-grade multi-view motion correction platform with golf plugin |

vs v2 estimate (9-14 days): +1 day for platform refactor + multi-view. Marginal cost; large ROI on subsequent plugins.

Parallel work:
- PR-7a starts immediately on CC; Jason simultaneously starts labeling
- Jason needs to record + analyze 3 down-the-line videos before labeling can complete

---

## §15 Strategic context (DO NOT DRIFT)

Per memory #22 + Verdict v2 §9 + v3 platform pivot:

- ✓ **DO**: Build correction engine generic from day 1 — naming, module structure, no golf strings in `engine/`
- ✓ **DO**: Golf is the first plugin, not the only plugin
- ✓ **DO**: Treat ground truth datasets as moat — per-sport, accumulating asset
- ✓ **DO**: Keep WHAM as default pose backbone; allow Layer 1 swap
- ✓ **DO**: View-aware tuning (face_on + down_the_line) from day 1
- ✓ **DO**: Plugin pattern conformance — `engine/` modules MUST NOT reference any sport-specific identifier

- ✗ **DO NOT**: Hard-code "golf" anywhere in `engine/`
- ✗ **DO NOT**: Start tennis/ski/other plugins before golf MVP ships + ground truth workflow proven
- ✗ **DO NOT**: Add "side" or "back" views in MVP — skip until production data demands
- ✗ **DO NOT**: Search for better pose backbone — WHAM stays
- ✗ **DO NOT**: Pursue medical-grade anatomical bone-center as MVP blocker
- ✗ **DO NOT**: Apply uniform smoothing across phases — phase-aware mandatory
- ✗ **DO NOT**: Touch frontend in PR-7a — Python module + offline JSON output only
- ✗ **DO NOT**: Mix v2 vocabulary into v3 implementation (no `golf_correction/` path; use `motion_correction/`)

---

## §16 Why this architecture compounds value

PR-7 v3 implementation cost ≈ v2 cost + 1 day. But each future plugin (tennis, ski, etc.) takes 1-4 weeks instead of 9-14 days because:

- Engine reused (~70% of code)
- DomainPlugin interface forces consistent shape
- Schema envelope reused; only namespaced anchors / metrics differ
- Labeler tool reused (just `--sport=tennis`)
- Sweep harness reused
- Frontend dispatch pattern reused (just add `golfRenderer.ts` → `tennisRenderer.ts`)

After 3-4 plugins shipped, SDK packaging is mechanical (`pip install swingcue-motion`, `from swingcue_motion import get_plugin`). B2B licensing becomes realistic deliverable.

The architecture decision made in PR-7 v3 is the highest-leverage technical decision in the SwingCue roadmap. It changes the company from "golf swing app" to "motion analysis platform with golf as first market."

---

## §17 Sign-off checklist for v3

Before CC starts PR-7a implementation:

- [ ] Jason reviews this spec
- [ ] ChatGPT reviews this spec for engineering gotchas (optional but recommended)
- [ ] Spec v3 committed to `docs/files/PR-7_MOTION_CORRECTION_PLATFORM_SPEC_v3.md` on track2 branch
- [ ] Existing labeler tool tagged compatible with v2 schema (will be modified in PR-7b)
- [ ] Existing single label `b3fea3f0_setup.json` renamed to `docs/PR-7_GROUND_TRUTH/golf/b3fea3f0_setup_face_on.json` (1 of 30 total)
- [ ] If Jason wants down_the_line ground truth: 3 new videos recorded + uploaded + analyzed for phase frame extraction

Once all checked → CC starts PR-7a.
