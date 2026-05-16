"""
pose3d orchestrator — runs N phases concurrently and writes SAM results.

PR-3 refactor: frame extraction is now done once at the main.py level
and the PNG bytes are passed in via `png_bytes_per_phase`, so the SAM
and YOLO pipelines share a single ffmpeg pass per phase.

Per-phase pipeline (one task):
    1. upload PNG bytes to swing-frames bucket
    2. fal sam-3/3d-body (with retries)
    3. parse fal response and upsert into pose_3d_phases

Failure isolation is two-layered:
  - Each task has its own try/except that records a `fal_status='failed'`
    row, so a single phase failure is visible in the DB.
  - `asyncio.gather(..., return_exceptions=True)` catches anything that
    somehow escapes the per-task handler (e.g. a bug in the writer itself),
    so one bad phase still cannot blow up the other four.
"""

import asyncio
import logging
from typing import Optional

from sam3d.fal_client_wrap import call_fal
from sam3d.keypoints import LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER
from sam3d.storage import ensure_bucket, upload_phase_frame
from sam3d.supabase_writer import write_pose_phase, write_pose_phase_failed

logger = logging.getLogger(__name__)


def _xy(kp2d: list, idx: int) -> tuple[Optional[float], Optional[float]]:
    """Safely pull (x, y) from a keypoints_2d entry; tolerate short arrays."""
    if idx >= len(kp2d):
        return None, None
    pt = kp2d[idx]
    if not isinstance(pt, (list, tuple)) or len(pt) < 2:
        return None, None
    return float(pt[0]), float(pt[1])


def _parse_fal_result(fal_result: dict, phase_name: str) -> dict:
    """
    Pull the fields the writer needs out of the fal sam-3/3d-body response.

    Returns a dict ready to spread into write_pose_phase as kwargs. Raises
    ValueError if the response is missing or under-populated (caught upstream
    and turned into a failed row).
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

    ls_x, ls_y = _xy(kp2d, LEFT_SHOULDER)
    rs_x, rs_y = _xy(kp2d, RIGHT_SHOULDER)
    lh_x, lh_y = _xy(kp2d, LEFT_HIP)
    rh_x, rh_y = _xy(kp2d, RIGHT_HIP)

    return {
        "keypoints_2d": kp2d,
        "keypoints_3d": kp3d,
        "focal_length": float(person.get("focal_length") or 0.0),
        "bbox": person.get("bbox"),
        "mhr_params": person.get("mhr_model_params"),
        "glb_url": ((fal_result.get("model_glb") or {}).get("url")),
        "shoulder_left_x": ls_x,
        "shoulder_left_y": ls_y,
        "shoulder_right_x": rs_x,
        "shoulder_right_y": rs_y,
        "hip_left_x": lh_x,
        "hip_left_y": lh_y,
        "hip_right_x": rh_x,
        "hip_right_y": rh_y,
    }


async def _process_one_phase(
    *,
    png_bytes: bytes,
    video_id: str,
    user_id: str,
    phase_name: str,
    timestamp_s: float,
    fps: float,
    image_width: int,
    image_height: int,
) -> dict:
    """
    Run the upload → fal → write pipeline for one phase. PNG bytes are
    supplied by the caller (extracted once at main.py and shared with
    the YOLO pipeline). Any exception is caught and turned into a
    `fal_status='failed'` row.
    """
    frame_idx = int(timestamp_s * fps) if fps > 0 else 0
    frame_timestamp_ms = int(timestamp_s * 1000)

    try:
        public_url = await upload_phase_frame(video_id, phase_name, png_bytes)
        logger.info(
            f"phase {phase_name}: uploaded "
            f"({len(png_bytes)} bytes, t={timestamp_s:.3f}s)"
        )

        fal_result = await call_fal(public_url)
        parsed = _parse_fal_result(fal_result, phase_name)

        # write_pose_phase is sync httpx; offload so siblings keep running
        await asyncio.to_thread(
            write_pose_phase,
            video_id=video_id,
            user_id=user_id,
            phase_name=phase_name,
            frame_idx=frame_idx,
            frame_timestamp_ms=frame_timestamp_ms,
            image_width=image_width,
            image_height=image_height,
            **parsed,
        )
        logger.info(f"phase {phase_name}: completed")
        return {"phase": phase_name, "status": "completed"}

    except Exception as e:
        logger.error(
            f"phase {phase_name} failed: {e!r}", exc_info=True
        )
        try:
            await asyncio.to_thread(
                write_pose_phase_failed,
                video_id=video_id,
                user_id=user_id,
                phase_name=phase_name,
                frame_idx=frame_idx,
                frame_timestamp_ms=frame_timestamp_ms,
                image_width=image_width,
                image_height=image_height,
                error_message=str(e),
            )
        except Exception as write_err:
            logger.error(
                f"phase {phase_name}: also failed to write failure row: "
                f"{write_err!r}"
            )
        return {
            "phase": phase_name,
            "status": "failed",
            "error": str(e),
        }


async def pose3d_for_all_phases(
    *,
    png_bytes_per_phase: dict[str, bytes],
    phase_timestamps: dict[str, float],
    video_id: str,
    user_id: str,
    image_width: int,
    image_height: int,
    fps: float,
) -> dict:
    """
    Run SAM 3D Body on all supplied phase frames in parallel. Returns:

        {
          "completed": int,
          "failed":    int,
          "results":   [{"phase": str, "status": "completed"|"failed", ...}]
        }

    The function never raises for per-phase failure — it always returns a
    summary. Only catastrophic setup (e.g. missing env vars) propagates.
    """
    ensure_bucket()

    tasks = []
    for phase_name, png_bytes in png_bytes_per_phase.items():
        timestamp_s = float(phase_timestamps.get(phase_name) or 0.0)
        tasks.append(
            _process_one_phase(
                png_bytes=png_bytes,
                video_id=video_id,
                user_id=user_id,
                phase_name=phase_name,
                timestamp_s=timestamp_s,
                fps=fps,
                image_width=image_width,
                image_height=image_height,
            )
        )

    # return_exceptions=True is the safety net for anything that escapes
    # the per-task try/except above (shouldn't happen, but defensive).
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    summary: dict = {"completed": 0, "failed": 0, "results": []}
    for raw in raw_results:
        if isinstance(raw, BaseException):
            logger.error(f"phase task escaped its own handler: {raw!r}")
            summary["failed"] += 1
            summary["results"].append(
                {"phase": "?", "status": "failed", "error": repr(raw)}
            )
        elif raw.get("status") == "completed":
            summary["completed"] += 1
            summary["results"].append(raw)
        else:
            summary["failed"] += 1
            summary["results"].append(raw)

    logger.info(
        f"pose3d_for_all_phases: completed={summary['completed']} "
        f"failed={summary['failed']}"
    )
    return summary
