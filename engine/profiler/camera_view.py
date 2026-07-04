"""
engine/profiler/camera_view.py
Layer 1 Video Profiler — Checkpoint 1: Camera View Detection

Geometric cross-validation approach (per blueprint §2.4, §v2.2):
  - No VLM for camera angle precision (known failure mode: misreads DTL as face-on during swing)
  - Three independent geometric evidences:
    ① sh_lat_ratio: lateral shoulder spread / torso height (face-on→wide, DTL→narrow)
    ② sh_confidence_asymmetry: symmetric in face-on, asymmetric in DTL
    ③ face_occlusion_proxy: nose+eyes confidence pattern (secondary, RTMPose extrapolates)
  - Samples from low-motion early window (address segment) only, per spec
  - Three evidences agree → high confidence; conflict → uncertain + needs_human

Output fields (subset of VideoProfile):
  camera_view: face_on / dtl / other / uncertain
  confidence.camera_view: 0.0 - 1.0
  camera_view_evidence: {sh_lat_ratio, sh_asym, face_occ, valid_frames, window_frames}
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

KP_THR: float = 0.30          # minimum keypoint confidence to use a point
MIN_TORSO_H: float = 15.0     # minimum torso height (px) for ratio to be meaningful
MIN_SHW: float = 5.0          # minimum shoulder width (px) for ratio

# Thresholds for sh_lat_ratio (lateral_shoulder_width / torso_height)
# Derived empirically from 10 verified clips (see validation notebook)
SH_LAT_FACEON_LO: float = 0.40   # above this → evidence for face-on
SH_LAT_DTL_HI:   float = 0.30   # below this → evidence for DTL
# 0.30–0.40 = uncertain zone

# Shoulder confidence asymmetry thresholds
# DTL: one shoulder faces away → lower confidence for far shoulder
SH_ASYM_DTL_THR: float = 0.06   # asym > this → evidence for DTL

# Low-motion sampling window (fraction of total frames)
WINDOW_START_FRAC: float = 0.03
WINDOW_END_FRAC:   float = 0.30
N_SAMPLES:         int   = 8


@dataclass
class CameraViewEvidence:
    sh_lat_ratio: Optional[float]        # mean lateral-shoulder / torso-height
    sh_asym:      Optional[float]        # mean shoulder confidence asymmetry (0=sym,1=full asym)
    face_occ:     Optional[float]        # mean nose+eye avg confidence (higher in face-on)
    valid_frames: int                    # frames with valid shoulder+hip kps
    window_frames: int                   # total frames sampled

    # Per-evidence votes (+1 face-on, -1 DTL, 0 uncertain)
    vote_sh_lat:  int = 0
    vote_sh_asym: int = 0
    vote_combined: int = 0


@dataclass
class CameraViewResult:
    camera_view: str                     # face_on / dtl / other / uncertain
    confidence:  float                   # 0.0 - 1.0
    evidence:    CameraViewEvidence
    needs_human: bool = False
    note:        str  = ""


def _safe_xy(kps: dict, name: str, thr: float = KP_THR):
    k = kps.get(name, {})
    if k.get("score", 0.0) >= thr:
        return k["x"], k["y"]
    return None


def _safe_score(kps: dict, name: str) -> float:
    return kps.get(name, {}).get("score", 0.0)


def _get_kps(fd: dict) -> dict:
    """Extract keypoints dict from a frame dict (handles both 'frame' and 'frame_idx' keys)."""
    persons = fd.get("persons", [])
    if not persons:
        return {}
    return persons[0].get("keypoints", {})


def _sample_frames(kp_json: dict) -> list[dict]:
    """Return N_SAMPLES frames from the low-motion early window."""
    frames = kp_json.get("frames", [])
    n = len(frames)
    if n == 0:
        return []
    start = max(0, int(n * WINDOW_START_FRAC))
    end   = min(n - 1, int(n * WINDOW_END_FRAC))
    if end <= start:
        end = min(start + 20, n - 1)
    count = N_SAMPLES
    indices = [start + int((end - start) * i / max(count - 1, 1)) for i in range(count)]
    return [frames[min(i, n - 1)] for i in indices]


def _frame_metrics(kps: dict) -> dict:
    """Compute per-frame geometric metrics. Returns {} if not enough valid KPs."""
    ls = _safe_xy(kps, "left_shoulder")
    rs = _safe_xy(kps, "right_shoulder")
    lh = _safe_xy(kps, "left_hip")
    rh = _safe_xy(kps, "right_hip")

    m = {}
    if not (ls and rs and lh and rh):
        return m

    sh_lat  = abs(rs[0] - ls[0])
    sh_mid  = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
    hp_mid  = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)
    torso_h = math.hypot(sh_mid[0]-hp_mid[0], sh_mid[1]-hp_mid[1])

    if torso_h < MIN_TORSO_H:
        return m

    m["sh_lat_ratio"] = sh_lat / torso_h
    m["sh_w"]         = math.hypot(rs[0]-ls[0], rs[1]-ls[1])
    m["torso_h"]      = torso_h

    # Shoulder confidence asymmetry (Evidence 2)
    lsc = _safe_score(kps, "left_shoulder")
    rsc = _safe_score(kps, "right_shoulder")
    denom = max(lsc, rsc, 0.01)
    m["sh_asym"] = abs(lsc - rsc) / denom   # 0=symmetric, 1=fully asymmetric

    # Face visibility (Evidence 3 — secondary, RTMPose may extrapolate)
    nose_sc = _safe_score(kps, "nose")
    le_sc   = _safe_score(kps, "left_eye")
    re_sc   = _safe_score(kps, "right_eye")
    m["face_occ"] = (nose_sc + le_sc + re_sc) / 3.0

    return m


def detect_camera_view(kp_json: dict) -> CameraViewResult:
    """
    Detect camera view (face_on / dtl / other / uncertain) from keypoint JSON.

    Uses geometric evidence from low-motion early window only (address segment).
    No VLM. Three evidences cross-validated. Returns CameraViewResult.
    """
    sampled = _sample_frames(kp_json)
    if not sampled:
        ev = CameraViewEvidence(None, None, None, 0, 0)
        return CameraViewResult("uncertain", 0.1, ev, True, "no frames in JSON")

    metrics = []
    for fd in sampled:
        kps = _get_kps(fd)
        m = _frame_metrics(kps)
        if m:
            metrics.append(m)

    ev = CameraViewEvidence(None, None, None, len(metrics), len(sampled))

    if not metrics:
        return CameraViewResult(
            "uncertain", 0.2, ev, True,
            f"all {len(sampled)} sampled frames missing valid shoulder+hip KPs"
        )

    # Aggregate
    mean_sh_lat = sum(m["sh_lat_ratio"] for m in metrics) / len(metrics)
    mean_asym   = sum(m["sh_asym"]      for m in metrics) / len(metrics)
    mean_face   = sum(m.get("face_occ", 0.5) for m in metrics) / len(metrics)

    ev.sh_lat_ratio = round(mean_sh_lat, 4)
    ev.sh_asym      = round(mean_asym,   4)
    ev.face_occ     = round(mean_face,   4)

    # ── Evidence 1: sh_lat_ratio ────────────────────────────────────────────
    if mean_sh_lat >= SH_LAT_FACEON_LO:
        ev.vote_sh_lat = +1   # face-on
    elif mean_sh_lat <= SH_LAT_DTL_HI:
        ev.vote_sh_lat = -1   # DTL
    else:
        ev.vote_sh_lat = 0    # uncertain zone (0.30–0.40)

    # ── Evidence 2: shoulder confidence asymmetry ────────────────────────────
    if mean_asym >= SH_ASYM_DTL_THR:
        ev.vote_sh_asym = -1  # asymmetric → DTL
    else:
        ev.vote_sh_asym = +1  # symmetric → face-on (or neutral)

    # ── Combined vote ────────────────────────────────────────────────────────
    # Primary: sh_lat_ratio (weight 2). Secondary: sh_asym (weight 1)
    weighted = 2 * ev.vote_sh_lat + 1 * ev.vote_sh_asym
    ev.vote_combined = weighted

    # ── Decision + Confidence ───────────────────────────────────────────────
    notes = []

    if ev.vote_combined >= 2:          # both or strong primary → face-on
        view = "face_on"
        # Confidence scales with how far ratio is from boundary
        conf = 0.70 + min(0.25, (mean_sh_lat - SH_LAT_FACEON_LO) * 1.2)
        if ev.vote_sh_asym < 0:        # conflict: ratio says face-on but asym says DTL
            conf -= 0.15
            notes.append("asym_conflict")

    elif ev.vote_combined <= -2:       # both or strong primary → DTL
        view = "dtl"
        conf = 0.70 + min(0.25, (SH_LAT_DTL_HI - mean_sh_lat) * 3.0)
        if ev.vote_sh_asym > 0:        # conflict
            conf -= 0.15
            notes.append("asym_conflict")

    elif ev.vote_sh_lat != 0:          # primary has signal, secondary conflicts
        view = "face_on" if ev.vote_sh_lat > 0 else "dtl"
        conf = 0.55
        notes.append("secondary_conflict")

    else:                              # primary uncertain
        view = "uncertain"
        conf = 0.40
        notes.append(f"sh_lat_ratio={mean_sh_lat:.3f}_in_uncertain_zone_0.30-0.40")

    # Coverage check: if fewer than 3 valid frames, lower confidence
    if len(metrics) < 3:
        conf = min(conf, 0.50)
        notes.append(f"only_{len(metrics)}_valid_frames")

    conf = round(max(0.10, min(0.98, conf)), 3)
    needs_human = view == "uncertain" or conf < 0.55 or bool(notes)

    return CameraViewResult(
        camera_view=view,
        confidence=conf,
        evidence=ev,
        needs_human=needs_human,
        note="; ".join(notes) if notes else "ok",
    )
