"""
cue_generator/plan_schema.py
CuePlan dataclass — typed representation of Cue Plan JSON.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import datetime, json


@dataclass
class AnchorSpec:
    source: str                     # payload field name or 'fixed_px'
    coords_px: list[float]          # [x, y]
    secondary_coords_px: list[float] | None = None  # line endpoint


@dataclass
class ColorSpec:
    fill_hex: str       = "#000000"
    fill_alpha: float   = 1.0
    stroke_hex: str     = "#000000"
    stroke_alpha: float = 1.0
    stroke_width_px: int = 2


@dataclass
class AnimationTrack:
    motion_type: str       = "arc_sweep"    # arc_sweep|linear_move|fade_in|discrete_steps
    duration_s: float      = 1.8
    pause_s: float         = 0.5
    loop: bool             = True
    pauseable: bool        = True
    easing: str            = "ease_in_out"
    steps: Any             = None           # None or list of step dicts (delta sentence)


@dataclass
class CueElement:
    primitive: str          # P1~P12
    anchor: AnchorSpec
    semantic_role: str      # correct_zone|current_state|direction_instruction|...
    color: ColorSpec
    shape_params: dict      # geometry (type, angles, radius, etc.)
    animation_track: AnimationTrack | None
    layer: str              # bg|mid|fg


@dataclass
class CaptionBadge:
    text: str
    position: str   = "bottom_center"
    font: str       = "NotoSansSC"
    size_px: int    = 28
    color_hex: str  = "#FFFFFF"
    outline_hex: str = "#141414"


@dataclass
class CuePlan:
    clip_id: str
    fault_id: str
    confidence: str
    sentence_type_id: str       # alpha_angle|beta_fence|gamma_deform|delta_sequence|neutral|retake
    contrast_structure: str     # single_subject|side_by_side|sequential
    elements: list[CueElement]
    caption_badge: CaptionBadge
    static_downgrade_note: str | None = None
    schema_version: str = "cue_plan_v0.1"
    timestamp_utc: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    validator_result: dict = field(default_factory=lambda: {"passed": False, "violations": []})

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
