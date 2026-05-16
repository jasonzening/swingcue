"""
pose_3d_phases writer (service-role).

The Railway analyzer is a backend-controlled write context per
docs/decisions/API_CLIENT_BOUNDARY.md, so it uses the service-role key
(bypasses RLS). The Next.js API route enforces auth + video ownership before
ever invoking the analyzer, so this writer trusts the (video_id, user_id)
it receives.
"""

import logging
import os
from typing import Any, Optional

from supabase import Client, create_client

from sam3d.keypoints import (
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
)

logger = logging.getLogger(__name__)

TABLE = "pose_3d_phases"


def get_admin_client() -> Client:
    """
    Build a Supabase client with the service-role key. Reads SUPABASE_URL and
    SUPABASE_SERVICE_ROLE_KEY from env; raises if either is missing.
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set "
            "(see Railway env vars)"
        )
    return create_client(url, key)


def _extract_xy(kp2d: list, idx: int) -> tuple[Optional[float], Optional[float]]:
    """Safely pull (x, y) from the keypoints_2d array; tolerate short arrays."""
    if idx >= len(kp2d):
        return None, None
    pt = kp2d[idx]
    if not isinstance(pt, (list, tuple)) or len(pt) < 2:
        return None, None
    return float(pt[0]), float(pt[1])


def write_pose_phase(
    client: Client,
    *,
    video_id: str,
    user_id: str,
    phase_name: str,
    frame_idx: int,
    frame_timestamp_ms: int,
    fal_result: dict,
    image_width: int,
    image_height: int,
    fal_request_id: Optional[str] = None,
) -> None:
    """
    Parse one fal sam-3/3d-body response and upsert one row into
    `pose_3d_phases`. Upsert key: (video_id, phase_name).
    """
    people = (fal_result.get("metadata") or {}).get("people") or []
    if not people:
        raise ValueError(
            f"fal response for phase {phase_name!r} contained no people"
        )
    person = people[0]

    kp2d: list = person.get("keypoints_2d") or []
    kp3d: list = person.get("keypoints_3d") or []
    if len(kp2d) < 11 or len(kp3d) < 11:
        raise ValueError(
            f"fal response for phase {phase_name!r} returned only "
            f"{len(kp2d)} 2D / {len(kp3d)} 3D keypoints (need >= 11)"
        )

    ls_x, ls_y = _extract_xy(kp2d, LEFT_SHOULDER)
    rs_x, rs_y = _extract_xy(kp2d, RIGHT_SHOULDER)
    lh_x, lh_y = _extract_xy(kp2d, LEFT_HIP)
    rh_x, rh_y = _extract_xy(kp2d, RIGHT_HIP)

    row: dict[str, Any] = {
        "video_id": video_id,
        "user_id": user_id,
        "phase_name": phase_name,
        "frame_idx": frame_idx,
        "frame_timestamp_ms": frame_timestamp_ms,
        "keypoints_2d": kp2d,
        "keypoints_3d": kp3d,
        "focal_length": float(person.get("focal_length") or 0.0),
        "bbox": person.get("bbox"),
        "mhr_params": person.get("mhr_model_params"),
        "glb_url": ((fal_result.get("model_glb") or {}).get("url")),
        "image_width": image_width,
        "image_height": image_height,
        "shoulder_left_x": ls_x,
        "shoulder_left_y": ls_y,
        "shoulder_right_x": rs_x,
        "shoulder_right_y": rs_y,
        "hip_left_x": lh_x,
        "hip_left_y": lh_y,
        "hip_right_x": rh_x,
        "hip_right_y": rh_y,
        "fal_status": "completed",
        "fal_request_id": fal_request_id,
        "error_message": None,
    }

    client.table(TABLE).upsert(row, on_conflict="video_id,phase_name").execute()


def write_pose_phase_failed(
    client: Client,
    *,
    video_id: str,
    user_id: str,
    phase_name: str,
    frame_idx: int,
    frame_timestamp_ms: Optional[int],
    image_width: int,
    image_height: int,
    error_message: str,
) -> None:
    """
    Record a per-phase failure so downstream UIs can show which phase is
    missing pose data. Keypoint columns are stored as empty arrays (column
    is NOT NULL); shoulder/hip pixel cols stay NULL.
    """
    row: dict[str, Any] = {
        "video_id": video_id,
        "user_id": user_id,
        "phase_name": phase_name,
        "frame_idx": frame_idx,
        "frame_timestamp_ms": frame_timestamp_ms,
        "keypoints_2d": [],
        "keypoints_3d": [],
        "focal_length": 0.0,
        "bbox": None,
        "mhr_params": None,
        "glb_url": None,
        "image_width": image_width,
        "image_height": image_height,
        "fal_status": "failed",
        "error_message": error_message[:1000],  # keep DB row sane
    }
    client.table(TABLE).upsert(row, on_conflict="video_id,phase_name").execute()
