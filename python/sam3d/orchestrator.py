"""
pose3d orchestrator — runs 5 phases concurrently and writes results.

Per-phase pipeline (one task):
    1. ffmpeg extract frame at phase timestamp
    2. upload PNG to swing-frames bucket
    3. fal sam-3/3d-body (with retries)
    4. parse + upsert into pose_3d_phases

Failure isolation is two-layered:
  - Each task has its own try/except that records a `fal_status='failed'`
    row, so a single phase failure is visible in the DB.
  - `asyncio.gather(..., return_exceptions=True)` catches anything that
    somehow escapes the per-task handler (e.g. a bug in the writer itself),
    so one bad phase still cannot blow up the other four.
"""

import asyncio
import logging
from typing import Any

from sam3d.fal_client_wrap import call_fal
from sam3d.frame_extract import extract_frame
from sam3d.storage import ensure_bucket, upload_phase_frame
from sam3d.supabase_writer import (
    get_admin_client,
    write_pose_phase,
    write_pose_phase_failed,
)

logger = logging.getLogger(__name__)

PHASES: tuple[tuple[str, str], ...] = (
    ("setup", "setupTime"),
    ("top", "topTime"),
    ("transition", "transitionTime"),
    ("impact", "impactTime"),
    ("finish", "finishTime"),
)


async def _process_one_phase(
    *,
    client: Any,
    video_path: str,
    video_id: str,
    user_id: str,
    phase_name: str,
    timestamp_s: float,
    fps: float,
    image_width: int,
    image_height: int,
) -> dict:
    """
    Run the extract → upload → fal → write pipeline for one phase. Any
    exception is caught here and converted into a `fal_status='failed'`
    row, so the orchestrator's gather() always sees a normal return.
    """
    frame_idx = int(timestamp_s * fps) if fps > 0 else 0
    frame_timestamp_ms = int(timestamp_s * 1000)

    try:
        png_bytes = await extract_frame(video_path, timestamp_s)
        public_url = await upload_phase_frame(
            client, video_id, phase_name, png_bytes
        )
        logger.info(
            f"phase {phase_name}: extracted+uploaded "
            f"({len(png_bytes)} bytes, t={timestamp_s:.3f}s)"
        )

        fal_result = await call_fal(public_url)
        fal_request_id = fal_result.get("request_id")

        await asyncio.to_thread(
            write_pose_phase,
            client,
            video_id=video_id,
            user_id=user_id,
            phase_name=phase_name,
            frame_idx=frame_idx,
            frame_timestamp_ms=frame_timestamp_ms,
            fal_result=fal_result,
            image_width=image_width,
            image_height=image_height,
            fal_request_id=fal_request_id,
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
                client,
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
    video_path: str,
    video_id: str,
    user_id: str,
    phases: dict,
    image_width: int,
    image_height: int,
    fps: float,
) -> dict:
    """
    Run all 5 phases in parallel. Returns a summary dict:

        {
          "completed": int,
          "failed":    int,
          "results":   [{"phase": str, "status": "completed"|"failed", ...}]
        }

    The function never raises for per-phase failure — it always returns a
    summary. Only catastrophic setup (e.g. missing env vars) propagates.
    """
    client = get_admin_client()
    ensure_bucket(client)

    tasks = []
    for phase_name, time_key in PHASES:
        timestamp_s = float(phases.get(time_key) or 0.0)
        tasks.append(
            _process_one_phase(
                client=client,
                video_path=video_path,
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
