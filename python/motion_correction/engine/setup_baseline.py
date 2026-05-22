"""
setup_baseline.py — detect a stable baseline window in the raw
timeline and extract anatomy metrics (shoulder width, hip width,
spine length, stance width).

Used downstream to:
  - Anchor the disc visualization to a fixed scale (so a moving wrist
    doesn't make the disc breathe).
  - Provide a reference distance for outlier rejection
    (`scale_reference` in temporal_smoother).
  - Compute baseline-relative metrics (e.g., hip turn at impact =
    current hip angle - setup baseline hip angle).

Sport-agnostic detection: takes a callable from the plugin that maps
(frame_idx → phase). Engine picks the first contiguous run of frames
whose phase is in `static_phases`, averages anatomy across that
window, and returns one SetupBaseline.

If no static phase exists (sport without a clear "setup" — e.g.,
running gait), the caller can pass static_phases=("__first_window__",)
and the engine treats the first N frames as the baseline window.
That escape hatch keeps the engine sport-agnostic.
"""
from __future__ import annotations

import math
import statistics
from typing import Callable, Optional

from ..schemas.corrected_timeline import SetupBaseline
from .anatomical_offset import body_local_basis


# Marker for "use first N frames" mode when sport has no static setup phase.
FIRST_WINDOW_SENTINEL = "__first_window__"


def _dist(a: Optional[list[float]], b: Optional[list[float]]) -> Optional[float]:
    """Euclidean distance, None-safe."""
    if a is None or b is None:
        return None
    if len(a) != 3 or len(b) != 3:
        return None
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _mean(xs: list[float]) -> Optional[float]:
    valid = [x for x in xs if x is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def find_baseline_window(
    n_frames: int,
    phase_of_frame: Callable[[int], str],
    static_phases: tuple[str, ...],
    *,
    min_window_frames: int = 3,
    first_window_size: int = 5,
) -> tuple[int, int]:
    """
    Pick the baseline frame range.

    Args:
        n_frames:           total frames in the raw timeline.
        phase_of_frame:     callable returning phase name for a frame_idx.
        static_phases:      phases the engine considers "stable enough"
                             to use as baseline.
        min_window_frames:  smallest acceptable window (engine returns
                             whatever's available even if smaller, with
                             a caller-side warning expected).
        first_window_size:  if FIRST_WINDOW_SENTINEL is in static_phases,
                             use this many initial frames as baseline.

    Returns: (start_frame_inclusive, end_frame_exclusive).
    """
    if FIRST_WINDOW_SENTINEL in static_phases:
        end = min(first_window_size, n_frames)
        return (0, end)

    # Find longest contiguous run of frames whose phase ∈ static_phases.
    best_start, best_end = 0, 0
    cur_start = None
    for i in range(n_frames):
        if phase_of_frame(i) in static_phases:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                if (i - cur_start) > (best_end - best_start):
                    best_start, best_end = cur_start, i
                cur_start = None
    # Handle trailing run.
    if cur_start is not None:
        if (n_frames - cur_start) > (best_end - best_start):
            best_start, best_end = cur_start, n_frames

    # If no static frames found at all, fall back to first window.
    if best_end - best_start == 0:
        end = min(first_window_size, n_frames)
        return (0, end)

    return (best_start, best_end)


def extract_baseline(
    frames: list[dict],
    *,
    phase_of_frame: Callable[[int], str],
    static_phases: tuple[str, ...] = ("setup",),
    compute_spine_angle: bool = False,
) -> Optional[SetupBaseline]:
    """
    Compute SetupBaseline from the chosen window.

    Args:
        frames:               raw timeline frames (each a dict with
                               "frame_idx", "ts", "joint_centers_3d"
                               per PilotRunResult schema).
        phase_of_frame:       see find_baseline_window.
        static_phases:        passed to find_baseline_window.
        compute_spine_angle:  set True for sports where spine angle is
                               meaningful (plugin-declared). Engine
                               returns None for this field otherwise.

    Returns: SetupBaseline or None if the chosen window has no usable
    anatomy data (all torso joints missing across the window).
    """
    n = len(frames)
    if n == 0:
        return None
    start, end = find_baseline_window(n, phase_of_frame, static_phases)

    # Accumulate anatomy across the window — robust to a few missing
    # frames inside (averaging handles holes).
    shoulder_widths: list[float] = []
    hip_widths: list[float] = []
    spine_lengths: list[float] = []
    stance_widths: list[float] = []
    spine_angles_deg: list[float] = []

    for i in range(start, end):
        kp = frames[i].get("joint_centers_3d", {})
        sw = _dist(kp.get("left_shoulder"), kp.get("right_shoulder"))
        if sw is not None:
            shoulder_widths.append(sw)
        hw = _dist(kp.get("left_hip"), kp.get("right_hip"))
        if hw is not None:
            hip_widths.append(hw)
        # Spine length = pelvis → neck (or pelvis → head if neck missing).
        pelvis = kp.get("pelvis")
        neck = kp.get("neck") or kp.get("head")
        sl = _dist(pelvis, neck)
        if sl is not None:
            spine_lengths.append(sl)
        # Stance = ankle to ankle.
        stance = _dist(kp.get("left_ankle"), kp.get("right_ankle"))
        if stance is not None:
            stance_widths.append(stance)
        # Spine angle (forward lean from vertical). Only if requested.
        if compute_spine_angle and pelvis is not None and neck is not None:
            dx = neck[0] - pelvis[0]
            dy = neck[1] - pelvis[1]
            dz = neck[2] - pelvis[2]
            # Forward lean = arctan( horizontal / vertical ). In camera
            # frame +y is down, so "up" is -y. Horizontal magnitude is
            # sqrt(dx² + dz²).
            horizontal = math.sqrt(dx * dx + dz * dz)
            vertical = abs(dy)
            if vertical > 1e-6:
                spine_angles_deg.append(math.degrees(math.atan2(horizontal, vertical)))

    base_shoulder = _mean(shoulder_widths)
    base_hip = _mean(hip_widths)
    base_spine = _mean(spine_lengths)
    base_stance = _mean(stance_widths)
    base_spine_angle = _mean(spine_angles_deg) if compute_spine_angle else None

    # If we couldn't compute any of the core 4, the window is unusable.
    if all(v is None for v in (base_shoulder, base_hip, base_spine, base_stance)):
        return None

    # Pick the representative frame (midpoint of the window).
    mid_idx = (start + end - 1) // 2
    mid_frame = frames[mid_idx]

    # ── PR-7a.1 Fix 1: locked pose + locked basis ─────────────────────
    # For every joint that appears in at least 50% of the window's
    # frames, take the median of each coordinate. This yields a
    # "frozen" reference pose that's robust to per-frame WHAM noise.
    locked_pose_3d = _median_pose_across_window(frames, start, end)
    locked_basis = None
    if locked_pose_3d:
        basis = body_local_basis(
            locked_pose_3d.get("pelvis"), locked_pose_3d.get("neck"),
        )
        if basis is not None:
            h, s, f = basis
            locked_basis = [list(h), list(s), list(f)]

    return SetupBaseline(
        setup_frame_idx=int(mid_frame.get("frame_idx", mid_idx)),
        setup_ts=float(mid_frame.get("ts", 0.0)),
        base_shoulder_width=base_shoulder or 0.0,
        base_hip_width=base_hip or 0.0,
        base_spine_length=base_spine or 0.0,
        base_stance_width=base_stance or 0.0,
        base_spine_angle_deg=base_spine_angle,
        setup_window_start=int(start),
        setup_window_end=int(end),
        locked_basis=locked_basis,
        locked_pose_3d=locked_pose_3d,
    )


def _median_pose_across_window(
    frames: list[dict], start: int, end: int,
) -> Optional[dict[str, list[float]]]:
    """
    Per-joint median 3D position across frames [start, end). A joint
    must appear (non-None, valid triple) in at least 50% of frames to
    be included. Returns {joint_name: [x, y, z]} or None if window empty.
    """
    if end - start < 1:
        return None
    by_joint: dict[str, list[list[float]]] = {}
    for i in range(start, end):
        kp = frames[i].get("joint_centers_3d", {})
        for name, xyz in kp.items():
            if xyz is None or len(xyz) != 3:
                continue
            if any(c is None for c in xyz):
                continue
            by_joint.setdefault(name, []).append([float(xyz[0]),
                                                   float(xyz[1]),
                                                   float(xyz[2])])
    n_window = end - start
    out: dict[str, list[float]] = {}
    for name, vals in by_joint.items():
        if len(vals) * 2 < n_window:    # appears in < 50% of window
            continue
        out[name] = [
            statistics.median(v[axis] for v in vals)
            for axis in (0, 1, 2)
        ]
    return out if out else None
