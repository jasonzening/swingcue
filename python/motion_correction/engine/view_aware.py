"""
view_aware.py — pick the right offset config based on which camera
view the video came from.

Per PR-7_REVIEW_RESPONSE.md Constraint 2:
  - face_on  : primary calibrated path
  - down_the_line : conservative defaults; fallback to face_on values
                     × multiplier if sweep can't converge
"""
from __future__ import annotations

from typing import Optional


# Multiplier applied to face_on offset values when falling back for
# the down_the_line view. Per Constraint 2 documented range: 1.05-1.10x.
# 1.10 picked as the midpoint default — DTL typically needs slightly
# stronger inward correction because the far-side joint is occluded
# and WHAM's SLAM-grounded inference biases outward more than face_on.
DTL_FALLBACK_MULTIPLIER: float = 1.10


def select_offset_config(
    offset_configs: dict[str, dict[str, float]],
    view: str,
    fallback_view: str = "face_on",
) -> dict[str, float]:
    """
    Look up per-view offset coefficients.

    Args:
        offset_configs: {"face_on": {"shoulder_inward": 0.14, ...},
                          "down_the_line": {...}}
        view:            "face_on" | "down_the_line" | other future view.
        fallback_view:   when `view` is missing from offset_configs, use
                          this view's values × DTL_FALLBACK_MULTIPLIER.

    Returns: flat dict {offset_name: coefficient}.

    Raises:
        KeyError if neither `view` nor `fallback_view` are in
        offset_configs (i.e., misconfigured plugin).
    """
    if view in offset_configs and offset_configs[view]:
        return dict(offset_configs[view])

    # Fallback path per Constraint 2 — apply multiplier so DTL gets
    # slightly stronger correction by default.
    if fallback_view not in offset_configs:
        raise KeyError(
            f"view={view!r} not in offset_configs and fallback_view="
            f"{fallback_view!r} also missing; available: "
            f"{list(offset_configs.keys())}"
        )
    base = offset_configs[fallback_view]
    out: dict = {}
    for name, value in base.items():
        if isinstance(value, (int, float)):
            out[name] = value * DTL_FALLBACK_MULTIPLIER
        elif isinstance(value, (list, tuple)) and len(value) == 3:
            out[name] = [
                float(value[0]) * DTL_FALLBACK_MULTIPLIER,
                float(value[1]) * DTL_FALLBACK_MULTIPLIER,
                float(value[2]) * DTL_FALLBACK_MULTIPLIER,
            ]
        elif isinstance(value, dict):
            # PR-7a.1 Fix 3: per-phase dict {phase: [d_h,d_v,d_f]}.
            scaled: dict = {}
            for phase, vec in value.items():
                if isinstance(vec, (list, tuple)) and len(vec) == 3:
                    scaled[phase] = [
                        float(vec[0]) * DTL_FALLBACK_MULTIPLIER,
                        float(vec[1]) * DTL_FALLBACK_MULTIPLIER,
                        float(vec[2]) * DTL_FALLBACK_MULTIPLIER,
                    ]
                else:
                    scaled[phase] = vec
            out[name] = scaled
        else:
            out[name] = value
    return out
