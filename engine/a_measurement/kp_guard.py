"""
engine/a_measurement/kp_guard.py
Keypoint Validity Guard — unified sentinel for all feature extraction.

All feature extraction code must call kp_guard() instead of inline
score/coordinate checks.  This ensures consistent behavior and makes
the (0,0) class of bugs impossible at the pipeline entry point.

Validity criteria (ALL must pass):
  1. score >= threshold (default 0.3)
  2. x > 0  (not left-edge or invalid)
  3. y > 0  (not top-edge or invalid)
  4. x < frame_w  (within frame width, if provided)
  5. y < frame_h  (within frame height, if provided)

Usage
-----
  from engine.a_measurement.kp_guard import kp_guard, head_ref_v2

  # Single keypoint
  pt = kp_guard(kps, "left_hip")           # returns (x,y) or None
  pt = kp_guard(kps, "left_hip", thr=0.4)  # custom threshold

  # Head reference v2 (ears-first)
  hr = head_ref_v2(kps)   # returns HeadRef(x, y, source, flag)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any


def kp_guard(
    kps:     Dict[str, Any],
    name:    str,
    thr:     float = 0.3,
    frame_w: Optional[int] = None,
    frame_h: Optional[int] = None,
) -> Optional[tuple[float, float]]:
    """
    Return (x, y) for keypoint `name` if all validity criteria pass, else None.

    Criteria
    --------
    1. name exists in kps dict
    2. score >= thr
    3. x > 0 and y > 0  (excludes (0,0) and other invalid sentinel values)
    4. x < frame_w  (if frame_w provided)
    5. y < frame_h  (if frame_h provided)

    Parameters
    ----------
    kps     : keypoint dict, each value has "x", "y", "score" keys
    name    : keypoint name (e.g. "left_hip")
    thr     : minimum confidence score (default 0.3)
    frame_w : frame pixel width — x must be strictly less than this
    frame_h : frame pixel height — y must be strictly less than this

    Returns
    -------
    (x, y) as floats, or None if any criterion fails.
    """
    if name not in kps:
        return None
    k = kps[name]
    x = float(k.get("x", 0.0))
    y = float(k.get("y", 0.0))
    score = float(k.get("score", 0.0))

    if score < thr:
        return None
    if x <= 0 or y <= 0:           # criterion 3: excludes (0,0)
        return None
    if frame_w is not None and x >= frame_w:
        return None
    if frame_h is not None and y >= frame_h:
        return None

    return (x, y)


# ---------------------------------------------------------------------------
# Head reference v2  (ears-first)
# ---------------------------------------------------------------------------

@dataclass
class HeadRef:
    x:      float
    y:      float
    source: str   # "ears_both" | "ear_left" | "ear_right" | "nose" | "eyes" | "fallback"
    flag:   str   # "" (clean) | "single_ear" | "no_ear_nose_only" | "degraded"

    @property
    def pt(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def is_clean(self) -> bool:
        return self.flag == ""

    @property
    def needs_human(self) -> bool:
        return self.flag in ("no_ear_nose_only", "degraded")


def head_ref_v2(
    kps:     Dict[str, Any],
    thr:     float = 0.3,
    frame_w: Optional[int] = None,
    frame_h: Optional[int] = None,
) -> Optional[HeadRef]:
    """
    Head reference point v2: ears-first, with validity guard.

    Priority
    --------
    1. Both ears valid  → midpoint of left_ear + right_ear   flag=""
    2. Left ear only    → left_ear                            flag="single_ear"
    3. Right ear only   → right_ear                           flag="single_ear"
    4. Both ears invalid, nose valid → nose                   flag="no_ear_nose_only"
    5. Nose invalid, eye(s) valid   → eye midpoint            flag="degraded"
    6. Nothing valid                → None

    The source and flag are preserved in HeadRef for downstream audit.

    Parameters
    ----------
    kps     : keypoint dict
    thr     : confidence threshold (default 0.3)
    frame_w : optional frame width bound
    frame_h : optional frame height bound

    Returns
    -------
    HeadRef or None if no valid head keypoint found.
    """
    g = lambda name: kp_guard(kps, name, thr, frame_w, frame_h)

    l_ear = g("left_ear")
    r_ear = g("right_ear")

    # Case 1: both ears
    if l_ear and r_ear:
        return HeadRef(
            x=(l_ear[0] + r_ear[0]) / 2,
            y=(l_ear[1] + r_ear[1]) / 2,
            source="ears_both",
            flag="",
        )

    # Case 2/3: single ear
    if l_ear:
        return HeadRef(x=l_ear[0], y=l_ear[1], source="ear_left",  flag="single_ear")
    if r_ear:
        return HeadRef(x=r_ear[0], y=r_ear[1], source="ear_right", flag="single_ear")

    # Case 4: nose
    nose = g("nose")
    if nose:
        return HeadRef(x=nose[0], y=nose[1], source="nose", flag="no_ear_nose_only")

    # Case 5: eye midpoint
    l_eye = g("left_eye"); r_eye = g("right_eye")
    if l_eye and r_eye:
        return HeadRef(
            x=(l_eye[0] + r_eye[0]) / 2,
            y=(l_eye[1] + r_eye[1]) / 2,
            source="eyes", flag="degraded",
        )
    if l_eye:
        return HeadRef(x=l_eye[0], y=l_eye[1], source="eye_left",  flag="degraded")
    if r_eye:
        return HeadRef(x=r_eye[0], y=r_eye[1], source="eye_right", flag="degraded")

    return None
