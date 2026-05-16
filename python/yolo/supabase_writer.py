"""
YOLO writer for pose_3d_phases (raw httpx + PostgREST).

Race-safe two-step write that coexists with the SAM orchestrator running
in parallel on the same (video_id, phase_name) row:

  1. PATCH the row, setting ONLY the yolo_* columns. If a row exists
     (because SAM already wrote it), PostgREST returns the updated row
     and we are done.

  2. If PATCH affects 0 rows (SAM has not written yet), INSERT a minimal
     stub containing the yolo_* columns + the NOT NULL placeholders the
     schema requires. Use Prefer: resolution=merge-duplicates so that
     when SAM later INSERTs with the real keypoints_2d/3d/etc., those
     SAM-owned columns get overwritten — but the yolo_* columns stay,
     because SAM's payload does not include them.

  fal_status='uploaded' on the stub signals "SAM has not completed yet".
  When SAM finishes it sets fal_status='completed' (and overwrites the
  placeholder keypoints_2d=[] / focal_length=0.0).

Failure path: simplified per PR-3 decision — YOLO failures do NOT write
a "failed" row. yolo_keypoints_2d stays NULL → frontend builder falls
back to SAM data. Error is logged only.
"""

import logging
import os
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

TABLE = "pose_3d_phases"


def _get_config() -> tuple[str, str]:
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY env vars are required"
        )
    return url, key


def _base_headers(key: str) -> dict[str, str]:
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def write_yolo_phase(
    *,
    video_id: str,
    user_id: str,
    phase_name: str,
    frame_idx: int,
    frame_timestamp_ms: int,
    image_width: int,
    image_height: int,
    yolo_keypoints_2d: list[list[float]],
    yolo_model: str,
    yolo_inference_ms: int,
) -> None:
    """
    Race-safe write of YOLO output. Raises on non-2xx so the caller can log.
    """
    url, key = _get_config()

    # Step 1: PATCH only yolo_* columns (no-op if SAM has not written yet).
    patch_endpoint = (
        f"{url}/rest/v1/{TABLE}"
        f"?video_id=eq.{video_id}&phase_name=eq.{phase_name}"
    )
    patch_payload: dict[str, Any] = {
        "yolo_keypoints_2d": yolo_keypoints_2d,
        "yolo_model": yolo_model,
        "yolo_inference_ms": yolo_inference_ms,
    }
    patch_headers = _base_headers(key) | {"Prefer": "return=representation"}

    r = httpx.patch(
        patch_endpoint, headers=patch_headers, json=patch_payload, timeout=10.0,
    )
    if r.status_code == 200:
        body = r.json()
        if isinstance(body, list) and len(body) > 0:
            return  # row existed; yolo_* columns updated in place.

    # Step 2: PATCH affected zero rows → INSERT stub. SAM may later merge.
    insert_endpoint = f"{url}/rest/v1/{TABLE}?on_conflict=video_id,phase_name"
    insert_payload: dict[str, Any] = {
        "video_id": video_id,
        "user_id": user_id,
        "phase_name": phase_name,
        "frame_idx": frame_idx,
        "frame_timestamp_ms": frame_timestamp_ms,
        "image_width": image_width,
        "image_height": image_height,
        # NOT NULL placeholders — SAM's later INSERT will overwrite via
        # Prefer: resolution=merge-duplicates (SAM's payload includes
        # these columns; SAM's payload does NOT include yolo_*).
        "keypoints_2d": [],
        "keypoints_3d": [],
        "focal_length": 0.0,
        # Signals "SAM has not completed yet" — flipped to 'completed'
        # when SAM finishes (SAM's payload sets fal_status='completed').
        "fal_status": "uploaded",
        # YOLO columns
        "yolo_keypoints_2d": yolo_keypoints_2d,
        "yolo_model": yolo_model,
        "yolo_inference_ms": yolo_inference_ms,
    }
    insert_headers = _base_headers(key) | {
        "Prefer": "return=representation,resolution=merge-duplicates",
    }

    r2 = httpx.post(
        insert_endpoint, headers=insert_headers, json=insert_payload, timeout=10.0,
    )
    if r2.status_code not in (200, 201):
        raise RuntimeError(
            f"[yolo] insert-stub failed {r2.status_code}: {r2.text[:300]}"
        )
