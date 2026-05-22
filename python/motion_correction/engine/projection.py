"""
projection.py — 3D camera-frame → 2D pixel coords via pinhole.

Mirror of the math in python/pilot/runners/_overlay.py:
    u = fx * X / Z + cx
    v = fy * Y / Z + cy

Default intrinsics assume `fx = fy = max(W, H)` and principal point
at image center — same as what WHAM's demo.py assumes when no
calibration is provided, so projections here line up with what the
upstream model actually saw at inference time.
"""
from __future__ import annotations

from typing import Optional


def default_intrinsics(video_width: int, video_height: int) -> dict[str, float]:
    """
    Default pinhole intrinsics for a video with unknown calibration.
    Matches WHAM demo.py + python/pilot/runners/_overlay.py.
    """
    f = float(max(video_width, video_height))
    return {
        "fx": f,
        "fy": f,
        "cx": video_width / 2.0,
        "cy": video_height / 2.0,
    }


def project_xyz_to_uv(
    xyz: list[float] | tuple[float, float, float] | None,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Optional[list[float]]:
    """
    Pinhole project one 3D point. Returns [u, v] or None if z <= 0
    (behind the camera) or xyz is None/malformed.
    """
    if xyz is None or len(xyz) != 3:
        return None
    x, y, z = xyz
    if x is None or y is None or z is None:
        return None
    if z <= 0:
        return None
    return [fx * x / z + cx, fy * y / z + cy]


def project_keypoint_dict(
    keypoints_3d: dict[str, Optional[list[float]]],
    intrinsics: dict[str, float],
) -> dict[str, Optional[list[float]]]:
    """
    Project a whole keypoint dict in one pass. Output retains keys with
    None values when projection fails (e.g., z <= 0 or input was None).
    """
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]
    return {
        name: project_xyz_to_uv(xyz, fx, fy, cx, cy)
        for name, xyz in keypoints_3d.items()
    }
