"""
cue_renderer/__init__.py
Pure-consumer rendering module. No measurements, no judgments, no thresholds.
Consumes verdict payloads produced by the diagnosis engine.
"""
from .payload import ReversePivotPayload
from .reverse_pivot import render_reverse_pivot_cue

__all__ = ["ReversePivotPayload", "render_reverse_pivot_cue"]
