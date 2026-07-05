"""
cue_renderer/reverse_pivot.py
Visual cue renderer for Reverse Pivot / Reverse Spine Angle.

Architecture rule: PURE CONSUMER.
  - No measurements, no judgments, no threshold comparisons.
  - Input: ReversePivotPayload from diagnosis engine.
  - Output: BGR image(s) saved to disk.

Design language: CUE_DESIGN_LANGUAGE.md
Spec:            VISUAL_INDICATOR_V1.md
"""
from __future__ import annotations
import math
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional

from .payload import ReversePivotPayload

# ── colour constants (BGR) ────────────────────────────────────────────────────
_SKELETON_GREY  = (80, 80, 80)
_GREEN_ZONE     = (0, 180, 0)      # P1 correct zone
_ORANGE         = (0, 165, 255)    # P2 light deviation
_DEEP_ORANGE    = (0, 80, 200)     # P2 medium deviation
_RED            = (0, 0, 200)      # P2 heavy deviation
_ARROW_WHITE    = (255, 255, 255)  # P3 arrow
_ARROW_OUTLINE  = (20, 20, 20)     # P3 arrow outline
_TEXT_WHITE     = (255, 255, 255)
_TEXT_OUTLINE   = (20, 20, 20)
_BG_DARK        = (20, 20, 20)     # SILENT / neutral background fill

# ── skeleton edge pairs (COCO-17 names) ──────────────────────────────────────
_SKELETON_EDGES = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
    ("left_shoulder",  "nose"),
    ("right_shoulder", "nose"),
]

_KP_GUARD = 0.30   # minimum score to draw keypoint


def _pt(kps: dict, name: str) -> Optional[Tuple[int, int]]:
    """Return (x,y) int tuple if keypoint score >= guard, else None."""
    k = kps.get(name, {})
    if isinstance(k, dict):
        sc = k.get("score", 0.0)
        x, y = k.get("x", 0.0), k.get("y", 0.0)
    else:
        return None
    if sc >= _KP_GUARD and (x > 0 or y > 0):
        return (int(x), int(y))
    return None


def _midpoint(a: Tuple, b: Tuple) -> Tuple[int, int]:
    return (int((a[0]+b[0])/2), int((a[1]+b[1])/2))


# ── grey skeleton ─────────────────────────────────────────────────────────────

def _draw_skeleton(canvas: np.ndarray, kps: dict) -> None:
    """Draw all skeleton edges in low-saturation grey."""
    for n1, n2 in _SKELETON_EDGES:
        p1 = _pt(kps, n1); p2 = _pt(kps, n2)
        if p1 and p2:
            cv2.line(canvas, p1, p2, _SKELETON_GREY, 2, cv2.LINE_AA)
    for name in kps:
        p = _pt(kps, name)
        if p:
            cv2.circle(canvas, p, 3, _SKELETON_GREY, -1, cv2.LINE_AA)


# ── P1: correct-zone wedge ────────────────────────────────────────────────────

def _draw_correct_zone(canvas: np.ndarray,
                       hip_mid: Tuple[int,int],
                       shoulder_mid: Tuple[int,int],
                       band_lo: float, band_hi: float,
                       alpha: float = 0.35) -> None:
    """
    Draw P1 green semi-transparent wedge.
    The wedge opens upward from hip_mid, spanning [band_lo, band_hi] in degrees.
    Convention: 0° = straight up (shoulder directly above hip).
    Positive tilt = toward target (screen right in face-on).
    angle in screen coords: up = -Y direction, so we negate.
    """
    radius = int(math.hypot(shoulder_mid[0]-hip_mid[0],
                            shoulder_mid[1]-hip_mid[1])) + 20
    if radius < 10:
        return

    # Convert tilt degrees to OpenCV angles (0=right, CCW positive)
    # In face-on: up = angle 270°; + tilt → clockwise shift from up
    # screen_angle = 270 - tilt_deg  (because screen Y is flipped)
    angle_hi_cv = 270 - band_lo   # band_lo is the "less tilted" bound
    angle_lo_cv = 270 - band_hi   # band_hi is the "more tilted toward target" bound
    # Swap if needed so start < end for cv2.ellipse
    start_angle = min(angle_lo_cv, angle_hi_cv)
    end_angle   = max(angle_lo_cv, angle_hi_cv)

    overlay = canvas.copy()
    cv2.ellipse(overlay, hip_mid, (radius, radius), 0,
                start_angle, end_angle, _GREEN_ZONE, -1, cv2.LINE_AA)
    # Draw wedge lines
    for ang_deg in [band_lo, band_hi]:
        rad = math.radians(270 - ang_deg)
        px = int(hip_mid[0] + radius * math.cos(rad))
        py = int(hip_mid[1] - radius * math.sin(rad))  # screen Y inverted
        cv2.line(overlay, hip_mid, (px, py), _GREEN_ZONE, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, canvas, 1-alpha, 0, canvas)


# ── P2: current-state line ────────────────────────────────────────────────────

def _tilt_color(tilt_deg: float, band_hi: float) -> Tuple[int,int,int]:
    excess = tilt_deg - band_hi
    if excess > 23:
        return _RED
    elif excess > 10:
        return _DEEP_ORANGE
    else:
        return _ORANGE


def _draw_current_state(canvas: np.ndarray,
                        hip_mid: Tuple[int,int],
                        shoulder_mid: Tuple[int,int],
                        tilt_deg: float, band_hi: float) -> None:
    """Draw P2 current-state line from hip to shoulder."""
    col = _tilt_color(tilt_deg, band_hi)
    cv2.line(canvas, hip_mid, shoulder_mid, col, 4, cv2.LINE_AA)
    cv2.circle(canvas, shoulder_mid, 6, col, -1, cv2.LINE_AA)
    cv2.circle(canvas, hip_mid,      4, col, -1, cv2.LINE_AA)


# ── P3: arc direction arrow ───────────────────────────────────────────────────

def _draw_arc_arrow(canvas: np.ndarray,
                    hip_mid: Tuple[int,int],
                    shoulder_mid: Tuple[int,int],
                    band_lo: float, band_hi: float,
                    tilt_deg: float) -> None:
    """
    Draw P3 arc from shoulder_mid toward the correct zone center.
    Arc sweeps from current tilt angle toward band center angle.
    """
    radius = int(math.hypot(shoulder_mid[0]-hip_mid[0],
                            shoulder_mid[1]-hip_mid[1]))
    if radius < 10:
        return

    band_center = (band_lo + band_hi) / 2.0
    # Current angle in screen convention
    curr_angle_cv = 270 - tilt_deg
    tgt_angle_cv  = 270 - band_center

    # Only draw if there's meaningful arc
    arc_span = abs(curr_angle_cv - tgt_angle_cv)
    if arc_span < 2:
        return

    start_a = min(curr_angle_cv, tgt_angle_cv)
    end_a   = max(curr_angle_cv, tgt_angle_cv)

    # Draw arc with outline then white
    cv2.ellipse(canvas, hip_mid, (radius, radius), 0,
                start_a, end_a, _ARROW_OUTLINE, 5, cv2.LINE_AA)
    cv2.ellipse(canvas, hip_mid, (radius, radius), 0,
                start_a, end_a, _ARROW_WHITE, 3, cv2.LINE_AA)

    # Arrowhead at the target end
    arr_rad = math.radians(tgt_angle_cv)
    tip_x = int(hip_mid[0] + radius * math.cos(arr_rad))
    tip_y = int(hip_mid[1] - radius * math.sin(arr_rad))  # screen Y flip

    # Arrowhead direction: tangent to circle at tip
    tangent_rad = arr_rad + math.pi/2  # 90° offset
    if curr_angle_cv < tgt_angle_cv:   # arrow going clockwise
        tangent_rad = arr_rad - math.pi/2
    head_len = 14
    ah_x = int(tip_x - head_len * math.cos(tangent_rad + 0.4))
    ah_y = int(tip_y + head_len * math.sin(tangent_rad + 0.4))
    ah2_x = int(tip_x - head_len * math.cos(tangent_rad - 0.4))
    ah2_y = int(tip_y + head_len * math.sin(tangent_rad - 0.4))

    pts = np.array([[tip_x,tip_y],[ah_x,ah_y],[ah2_x,ah2_y]], np.int32)
    cv2.fillPoly(canvas, [pts], _ARROW_OUTLINE)
    cv2.polylines(canvas, [pts], True, _ARROW_WHITE, 2, cv2.LINE_AA)


# ── text helpers ──────────────────────────────────────────────────────────────

def _put_text_outlined(canvas: np.ndarray, text: str,
                       org: Tuple[int,int], scale: float = 0.65,
                       thickness: int = 2) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, text, org, font, scale, _TEXT_OUTLINE, thickness+2, cv2.LINE_AA)
    cv2.putText(canvas, text, org, font, scale, _TEXT_WHITE,   thickness,   cv2.LINE_AA)


def _draw_caption(canvas: np.ndarray, text: str) -> None:
    h, w = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(10, (w - tw) // 2)
    y = h - 18
    # Background bar
    cv2.rectangle(canvas, (0, h-42), (w, h), (0,0,0), -1)
    _put_text_outlined(canvas, text, (x, y), scale, thickness)


# ── main render entry ─────────────────────────────────────────────────────────

def render_reverse_pivot_cue(payload: ReversePivotPayload,
                              out_dir: Path) -> dict[str, Path]:
    """
    Render Reverse Pivot cue image(s) from verdict payload.

    Returns dict of {label: path} for produced files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conf       = payload.confidence
    stem       = payload.clip_id or "clip"
    frame      = payload.frame_bgr.copy()

    h, w = frame.shape[:2]

    hip_mid     = (int(payload.hip_mid[0]),      int(payload.hip_mid[1]))
    shoulder_mid = (int(payload.shoulder_mid[0]), int(payload.shoulder_mid[1]))

    outputs: dict[str, Path] = {}

    # ── SILENT: retake guide ─────────────────────────────────────────────────
    if conf == "SILENT":
        canvas = np.zeros_like(frame)
        canvas[:] = _BG_DARK
        lines = [
            "画面质量不足，请重新录制：",
            "正面站立，完整挥杆，确保全身入镜",
        ]
        for i, line in enumerate(lines):
            y = h//2 - 20 + i*38
            _put_text_outlined(canvas, line, (30, y), scale=0.7)
        p = out_dir / f"{stem}_top_cue.jpg"
        cv2.imwrite(str(p), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        outputs["cue"] = p
        pg = out_dir / f"{stem}_top_cue_gray.jpg"
        cv2.imwrite(str(pg), cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        outputs["cue_gray"] = pg
        return outputs

    # ── Neutral frame: Possible / None ────────────────────────────────────────
    if conf in ("Possible", "None"):
        canvas = frame.copy()
        _draw_skeleton(canvas, payload.skeleton_kps)
        _draw_caption(canvas, "此项未发现问题")
        p = out_dir / f"{stem}_top_cue.jpg"
        cv2.imwrite(str(p), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        outputs["cue"] = p
        pg = out_dir / f"{stem}_top_cue_gray.jpg"
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(pg), gray, [cv2.IMWRITE_JPEG_QUALITY, 92])
        outputs["cue_gray"] = pg
        return outputs

    # ── Complete cue: Confirmed / Likely ──────────────────────────────────────
    canvas = frame.copy()

    # 1. Grey skeleton background
    _draw_skeleton(canvas, payload.skeleton_kps)

    # 2. P1 correct-zone wedge
    _draw_correct_zone(canvas, hip_mid, shoulder_mid,
                       payload.band_lower_deg, payload.band_upper_deg)

    # 3. P2 current-state line
    _draw_current_state(canvas, hip_mid, shoulder_mid,
                        payload.tilt_deg, payload.band_upper_deg)

    # 4. P3 arc direction arrow
    _draw_arc_arrow(canvas, hip_mid, shoulder_mid,
                    payload.band_lower_deg, payload.band_upper_deg,
                    payload.tilt_deg)

    # 5. Caption
    _draw_caption(canvas, "顶点时上半身倒向了球的方向——下一杆感觉胸口留在球的后面")

    # Save colour
    p = out_dir / f"{stem}_top_cue.jpg"
    cv2.imwrite(str(p), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    outputs["cue"] = p

    # Save greyscale self-check
    pg = out_dir / f"{stem}_top_cue_gray.jpg"
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(pg), gray, [cv2.IMWRITE_JPEG_QUALITY, 92])
    outputs["cue_gray"] = pg

    return outputs
