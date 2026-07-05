"""
engine/reference_flywheel/__init__.py
参考基准飞轮 — 加载和查询接口

用法:
    from engine.reference_flywheel import load_baseline, query_tilt

    baseline = load_baseline()
    result = query_tilt(tilt_deg=12.5, baseline=baseline)
    print(result)  # {"confidence": "Likely", "delta_from_mu": 19.3, "in_band": False}
"""

from __future__ import annotations
import json
from pathlib import Path

_BASELINE_PATH = Path(__file__).parent / "baseline_v1.json"


def load_baseline(path: str | Path | None = None) -> dict:
    """Load baseline JSON. Returns dict."""
    p = Path(path) if path else _BASELINE_PATH
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def query_tilt(tilt_deg: float, baseline: dict | None = None) -> dict:
    """
    Query current baseline for a shoulder_lateral_tilt value at top.

    Returns:
        in_band (bool): True if tilt is within correct reference band.
        confidence (str): "None" / "Possible" / "Likely" / "Confirmed"
        delta_from_mu (float): tilt - correct_mu (positive = more toward target)
        delta_from_band_upper (float): tilt - band_upper (positive = outside band)
    """
    if baseline is None:
        baseline = load_baseline()

    band = baseline["reference_band"]
    ledger = baseline["confidence_ledger_thresholds"]
    mu = band["center_mu_deg"]
    lo = band["band_lower_deg"]
    hi = band["band_upper_deg"]

    in_band = lo <= tilt_deg <= hi
    delta_from_mu = tilt_deg - mu
    delta_from_upper = tilt_deg - hi

    if tilt_deg >= ledger["confirmed_above_deg"]:
        confidence = "Confirmed"
    elif tilt_deg >= ledger["likely_above_deg"]:
        confidence = "Likely"
    elif tilt_deg >= ledger["possible_above_deg"]:
        confidence = "Possible"
    else:
        confidence = "None"

    return {
        "in_band": in_band,
        "confidence": confidence,
        "delta_from_mu_deg": round(delta_from_mu, 2),
        "delta_from_band_upper_deg": round(delta_from_upper, 2),
        "tilt_deg": tilt_deg,
        "band": [lo, hi],
        "correct_mu_deg": mu,
    }
