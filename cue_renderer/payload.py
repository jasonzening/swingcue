"""
cue_renderer/payload.py
Verdict payload dataclass consumed by cue_renderer.
Produced by the diagnosis engine; renderer has zero knowledge of how values were computed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
import numpy as np


@dataclass
class ReversePivotPayload:
    """
    Structured verdict payload for Reverse Pivot / Reverse Spine Angle cue.
    All fields come from the diagnosis engine. Renderer is read-only.
    """
    fault_id:       str           # "reverse_pivot"
    confidence:     str           # "Confirmed" | "Likely" | "Possible" | "None" | "SILENT"
    tilt_deg:       float         # shoulder_lateral_tilt at top (deg, + = toward target)
    top_frame_idx:  int           # B-layer anchor.top frame index
    hip_mid:        Tuple[float, float]      # (x, y) hip midpoint at top frame
    shoulder_mid:   Tuple[float, float]      # (x, y) shoulder midpoint at top frame
    band_lower_deg: float         # correct band lower bound (e.g. -18.8)
    band_upper_deg: float         # correct band upper bound (e.g. +5.0)
    frame_bgr:      np.ndarray    # top frame BGR image (full frame)
    skeleton_kps:   Dict          # all keypoints at top frame for background skeleton
    clip_id:        str = ""      # for output naming
