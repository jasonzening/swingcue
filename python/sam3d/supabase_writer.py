"""
pose_3d_phases writer (raw httpx + PostgREST upsert).

The Railway analyzer is a backend-controlled write context per
docs/decisions/API_CLIENT_BOUNDARY.md, so it uses the service-role key
(bypasses RLS). We call PostgREST directly because supabase-py v2.7.4
rejects the new sb_secret_* key format.

Upsert key is the existing UNIQUE(video_id, phase_name) constraint, so
re-running a phase replaces the previous row.
"""

import logging
import os
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

TABLE = "pose_3d_phases"


def _get_config() -> tuple[str, str]:
    """Read SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY from env."""
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY env vars are required"
        )
    return url, key


def _headers(key: str) -> dict:
    """Service-role headers + Prefer upsert (merge-duplicates on conflict)."""
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation,resolution=merge-duplicates",
    }


def write_pose_phase(
    *,
    video_id: str,
    user_id: str,
    phase_name: str,
    frame_idx: int,
    frame_timestamp_ms: int,
    keypoints_2d: list,
    keypoints_3d: list,
    focal_length: float,
    bbox: Any,
    mhr_params: Any,
    glb_url: Optional[str],
    image_width: int,
    image_height: int,
    shoulder_left_x: Optional[float],
    shoulder_left_y: Optional[float],
    shoulder_right_x: Optional[float],
    shoulder_right_y: Optional[float],
    hip_left_x: Optional[float],
    hip_left_y: Optional[float],
    hip_right_x: Optional[float],
    hip_right_y: Optional[float],
) -> None:
    """
    Upsert a completed pose_3d_phases row. Raises on non-2xx so the caller
    can record the row as failed.
    """
    url, key = _get_config()
    endpoint = f"{url}/rest/v1/{TABLE}?on_conflict=video_id,phase_name"

    payload = {
        "video_id": video_id,
        "user_id": user_id,
        "phase_name": phase_name,
        "frame_idx": frame_idx,
        "frame_timestamp_ms": frame_timestamp_ms,
        "keypoints_2d": keypoints_2d,
        "keypoints_3d": keypoints_3d,
        "focal_length": focal_length,
        "bbox": bbox,
        "mhr_params": mhr_params,
        "glb_url": glb_url,
        "image_width": image_width,
        "image_height": image_height,
        "shoulder_left_x": shoulder_left_x,
        "shoulder_left_y": shoulder_left_y,
        "shoulder_right_x": shoulder_right_x,
        "shoulder_right_y": shoulder_right_y,
        "hip_left_x": hip_left_x,
        "hip_left_y": hip_left_y,
        "hip_right_x": hip_right_x,
        "hip_right_y": hip_right_y,
        "fal_status": "completed",
        "error_message": None,
    }

    r = httpx.post(endpoint, headers=_headers(key), json=payload, timeout=15.0)
    if r.status_code not in (200, 201):
        raise RuntimeError(
            f"upsert pose_3d_phases failed {r.status_code}: {r.text[:300]}"
        )


def write_pose_phase_failed(
    *,
    video_id: str,
    user_id: str,
    phase_name: str,
    frame_idx: int,
    frame_timestamp_ms: int,
    image_width: int,
    image_height: int,
    error_message: str,
) -> None:
    """
    Upsert a failed phase row so the table always has 5 rows per video. Any
    error here is logged but NOT raised — failure to record a failure should
    never abort the orchestrator.
    """
    try:
        url, key = _get_config()
        endpoint = f"{url}/rest/v1/{TABLE}?on_conflict=video_id,phase_name"

        payload = {
            "video_id": video_id,
            "user_id": user_id,
            "phase_name": phase_name,
            "frame_idx": frame_idx,
            "frame_timestamp_ms": frame_timestamp_ms,
            "keypoints_2d": [],
            "keypoints_3d": [],
            "focal_length": 0.0,
            "image_width": image_width,
            "image_height": image_height,
            "fal_status": "failed",
            "error_message": error_message[:2000],
        }

        r = httpx.post(endpoint, headers=_headers(key), json=payload, timeout=10.0)
        if r.status_code not in (200, 201):
            logger.error(
                f"failed-row upsert returned {r.status_code}: {r.text[:300]}"
            )
    except Exception as e:
        logger.error(f"failed-row upsert exception (non-fatal): {e!r}")
