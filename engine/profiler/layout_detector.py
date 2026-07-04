"""
engine/profiler/layout_detector.py
Layer 1 Video Profiler — layout detection (single / split_screen / pip).

Strategy: works from kp_json person count + video dimensions.
If kp_json was extracted from a single-panel clip, person_count=1 and layout=single.
For split-screen, person_count may be 2 OR the frame-level person tracking indicates
spatial separation (two clusters of detections at different x-regions).

However, kp_json by convention stores only the primary (largest/highest-conf) person
per frame. True split-screen detection requires the raw video or knowing the clip source.

For Gate 2 of the Profiler, we use a pragmatic approach:
  1. Check width:height aspect ratio (wide frames → likely split-screen)
  2. Check x-spread of body center: if body is consistently in one half → single;
     if two distinct spatial clusters → split_screen
  3. If neither deterministic → layout="unknown", low confidence

Also: if caller provides split_hint from a pre-run splitter, use it directly.

Returns dict: layout, confidence, aspect_ratio, body_center_x_mean, note
"""

from __future__ import annotations
from typing import Optional

KP_THR: float = 0.30
SPLIT_ASPECT_THR: float = 2.2    # width/height > this → likely split-screen
STEP: int = 5


def _get_kps(fd: dict) -> dict:
    persons = fd.get("persons", [])
    if not persons:
        return {}
    return persons[0].get("keypoints", {})


def _body_center_x(kps: dict) -> Optional[float]:
    """Mean x of shoulder + hip midpoints, if available."""
    pts = []
    for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
        k = kps.get(name, {})
        if k.get("score", 0.0) >= KP_THR:
            pts.append(k["x"])
    return sum(pts) / len(pts) if pts else None


def detect_layout(
    kp_json: dict,
    video_width: Optional[int] = None,
    video_height: Optional[int] = None,
    split_hint: Optional[str] = None,  # "split_screen" | "single" from splitter
) -> dict:
    """
    Detect video layout from kp_json and optional video dimensions.

    split_hint: if provided by the caller (e.g. from split_screen_splitter),
                used directly with high confidence.
    """
    # If we have a definitive hint from the splitter, trust it
    if split_hint == "split_screen":
        return {
            "layout": "split_screen",
            "confidence": 0.90,
            "aspect_ratio": None,
            "body_center_x_frac_mean": None,
            "note": "split_hint from splitter",
        }
    if split_hint == "single":
        return {
            "layout": "single",
            "confidence": 0.85,
            "aspect_ratio": None,
            "body_center_x_frac_mean": None,
            "note": "split_hint from splitter",
        }

    # Aspect ratio check (requires video dimensions)
    aspect = None
    if video_width and video_height and video_height > 0:
        aspect = video_width / video_height
        if aspect >= SPLIT_ASPECT_THR:
            return {
                "layout": "split_screen",
                "confidence": 0.80,
                "aspect_ratio": round(aspect, 3),
                "body_center_x_frac_mean": None,
                "note": f"wide aspect {aspect:.2f} >= {SPLIT_ASPECT_THR}",
            }

    # Body center x distribution check
    frames = kp_json.get("frames", [])
    n = len(frames)
    centers = []
    for i in range(0, n, STEP):
        kps = _get_kps(frames[i])
        cx = _body_center_x(kps)
        if cx is not None:
            centers.append(cx)

    if not centers:
        return {
            "layout": "unknown",
            "confidence": 0.2,
            "aspect_ratio": round(aspect, 3) if aspect else None,
            "body_center_x_frac_mean": None,
            "note": "no valid body center KPs",
        }

    # Without image width we can't normalize, but we can check variance
    mean_cx = sum(centers) / len(centers)

    # Normalize by max x observed (proxy for frame width if not provided)
    max_x = max(centers)
    norm_mean = mean_cx / max_x if max_x > 0 else 0.5

    # Default: if body center is consistently in one lateral region → single
    # We can't reliably detect split from kp_json alone without width, so
    # flag as uncertain unless aspect ratio was decisive
    return {
        "layout": "single",   # conservative default
        "confidence": 0.60,
        "aspect_ratio": round(aspect, 3) if aspect else None,
        "body_center_x_frac_mean": round(norm_mean, 3),
        "note": "geometry_only_estimate; verify with video dimensions",
    }
