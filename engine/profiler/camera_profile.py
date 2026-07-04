"""
engine/profiler/camera_profile.py
Layer 1 Video Profiler — camera_profile: subject position, size, camera height.

Computes from RTMPose keypoints:
  subject_center_x: mean torso center x / frame_width (0-1, left=0 right=1)
  subject_center_y: mean torso center y / frame_height (0-1, top=0 bottom=1)
  subject_height_ratio: body bounding box height / frame height (0-1)
  camera_height: "high" | "level" | "low"

camera_height rationale (geometric):
  We use the vertical position of key joints relative to the frame:
  - At setup/address, the torso-to-hip midline establishes the subject baseline.
  - High camera: golfer appears lower in frame; hip_mid_y_frac > 0.60 (hips in lower 60%)
  - Level camera: hip_mid_y_frac in [0.40, 0.60]
  - Low camera:   hip_mid_y_frac < 0.40 (hips in upper 40%)

  Additionally: "perspectve compression" proxy.
  In face-on view, if camera is level with hips, shoulder-to-hip vertical distance
  (in px) ≈ actual torso length. If camera is high (looking down), perspective foreshortens
  the body; shoulder-hip vertical span appears compressed relative to limbs.
  We use shoulder_y_frac as secondary check:
    shoulder_y_frac < 0.30 → subject very high in frame → low camera (looking up)
    shoulder_y_frac > 0.60 → subject very low in frame → high camera (looking down)
    else → level

  Decision: primary = hip_mid_y_frac; secondary = shoulder_y_frac; majority rules.
"""

from __future__ import annotations
from typing import Optional

KP_THR: float = 0.30
STEP: int = 4   # sample every N frames

# camera_height thresholds
HIP_Y_HIGH_THR:   float = 0.62   # hip in lower 62% → camera high
HIP_Y_LOW_THR:    float = 0.38   # hip in upper 38% → camera low
SHOU_Y_HIGH_THR:  float = 0.55   # shoulder in lower 55% → camera high
SHOU_Y_LOW_THR:   float = 0.28   # shoulder in upper 28% → camera low


def _get_kps(fd: dict) -> dict:
    persons = fd.get("persons", [])
    if not persons:
        return {}
    return persons[0].get("keypoints", {})


def _safe(kps: dict, name: str, coord: str, thr: float = KP_THR) -> Optional[float]:
    k = kps.get(name, {})
    if k.get("score", 0.0) >= thr:
        return k[coord]
    return None


def compute_camera_profile(
    kp_json: dict,
    video_width: Optional[int],
    video_height: Optional[int],
    address_frame: Optional[int] = None,  # if known, sample around it
) -> dict:
    """
    Compute camera_profile from kp_json and frame dimensions.
    Returns: subject_center_x, subject_center_y, subject_height_ratio,
             camera_height, confidence, notes
    """
    frames = kp_json.get("frames", [])
    n = len(frames)
    if n == 0:
        return _empty("no frames")

    # Determine sampling window: use early 5-30% (address area) or full clip
    if address_frame is not None:
        lo = max(0, address_frame - 5)
        hi = min(n - 1, address_frame + 15)
        sample_indices = list(range(lo, hi + 1, STEP)) or [address_frame]
    else:
        # Low-motion early window (same as camera_view)
        lo = int(n * 0.05); hi = int(n * 0.30)
        if hi <= lo: hi = min(lo + 20, n - 1)
        sample_indices = list(range(lo, hi + 1, STEP))
        if not sample_indices:
            sample_indices = [lo]

    # Collect metrics per frame
    center_xs, center_ys = [], []
    shoulder_ys, hip_ys   = [], []
    bboxes_h              = []

    for fi in sample_indices:
        fi = min(fi, n - 1)
        kps = _get_kps(frames[fi])

        # Torso center
        ls_x = _safe(kps, "left_shoulder",  "x"); rs_x = _safe(kps, "right_shoulder", "x")
        lh_x = _safe(kps, "left_hip",       "x"); rh_x = _safe(kps, "right_hip",      "x")
        ls_y = _safe(kps, "left_shoulder",  "y"); rs_y = _safe(kps, "right_shoulder", "y")
        lh_y = _safe(kps, "left_hip",       "y"); rh_y = _safe(kps, "right_hip",      "y")

        torso_xs = [v for v in [ls_x, rs_x, lh_x, rh_x] if v is not None]
        torso_ys = [v for v in [ls_y, rs_y, lh_y, rh_y] if v is not None]
        if torso_xs: center_xs.append(sum(torso_xs) / len(torso_xs))
        if torso_ys: center_ys.append(sum(torso_ys) / len(torso_ys))

        # Shoulder y (mean)
        sh_ys = [v for v in [ls_y, rs_y] if v is not None]
        if sh_ys: shoulder_ys.append(sum(sh_ys) / len(sh_ys))

        # Hip y (mean)
        hp_ys = [v for v in [lh_y, rh_y] if v is not None]
        if hp_ys: hip_ys.append(sum(hp_ys) / len(hp_ys))

        # Full body bounding box height (nose to ankles)
        nose_y   = _safe(kps, "nose",         "y")
        lank_y   = _safe(kps, "left_ankle",   "y")
        rank_y   = _safe(kps, "right_ankle",  "y")
        ank_ys   = [v for v in [lank_y, rank_y] if v is not None]
        if nose_y and ank_ys:
            bbox_h = max(ank_ys) - nose_y
            if bbox_h > 0: bboxes_h.append(bbox_h)

    if not center_xs:
        return _empty("no valid torso KPs")

    mean_cx = sum(center_xs) / len(center_xs)
    mean_cy = sum(center_ys) / len(center_ys)
    mean_sh_y = sum(shoulder_ys) / len(shoulder_ys) if shoulder_ys else None
    mean_hp_y = sum(hip_ys)      / len(hip_ys)      if hip_ys      else None
    mean_bbox_h = sum(bboxes_h) / len(bboxes_h) if bboxes_h else None

    # Normalize by frame dimensions
    W = video_width  or 1920
    H = video_height or 1080
    cx_frac   = round(mean_cx / W, 4)
    cy_frac   = round(mean_cy / H, 4)
    hp_y_frac = round(mean_hp_y / H, 4) if mean_hp_y is not None else None
    sh_y_frac = round(mean_sh_y / H, 4) if mean_sh_y is not None else None
    height_ratio = round(mean_bbox_h / H, 4) if mean_bbox_h is not None else None

    # ── camera_height decision ──────────────────────────────────────────
    votes_high = 0; votes_low = 0; votes_level = 0

    if hp_y_frac is not None:
        if hp_y_frac > HIP_Y_HIGH_THR:   votes_high  += 2  # primary, weight 2
        elif hp_y_frac < HIP_Y_LOW_THR:  votes_low   += 2
        else:                             votes_level += 2

    if sh_y_frac is not None:
        if sh_y_frac > SHOU_Y_HIGH_THR:  votes_high  += 1
        elif sh_y_frac < SHOU_Y_LOW_THR: votes_low   += 1
        else:                             votes_level += 1

    if votes_high > votes_low and votes_high > votes_level:
        cam_h = "high"; cam_conf = 0.75
    elif votes_low > votes_high and votes_low > votes_level:
        cam_h = "low";  cam_conf = 0.75
    elif votes_level >= max(votes_high, votes_low):
        cam_h = "level"; cam_conf = 0.70
    else:
        cam_h = "level"; cam_conf = 0.50   # tie → level (most common)

    # Reduce confidence if dimensions were assumed (no real W/H)
    if video_width is None or video_height is None:
        cam_conf -= 0.15
        note = "frame_dims_assumed_1920x1080"
    else:
        note = "ok"

    return {
        "subject_center_x":    cx_frac,
        "subject_center_y":    cy_frac,
        "subject_height_ratio": height_ratio,
        "camera_height":       cam_h,
        "confidence":          round(max(0.10, cam_conf), 3),
        "debug": {
            "hip_y_frac":       hp_y_frac,
            "shoulder_y_frac":  sh_y_frac,
            "votes_high":       votes_high,
            "votes_level":      votes_level,
            "votes_low":        votes_low,
        },
        "note": note,
    }


def _empty(reason: str) -> dict:
    return {
        "subject_center_x": None,
        "subject_center_y": None,
        "subject_height_ratio": None,
        "camera_height": "unknown",
        "confidence": 0.1,
        "debug": {},
        "note": reason,
    }
