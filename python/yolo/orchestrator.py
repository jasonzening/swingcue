"""
YOLO orchestrator — runs YOLO11-pose on N phase frames concurrently.

Mirrors the sam3d orchestrator pattern (asyncio.gather + per-task failure
isolation + outer return_exceptions safety net) but differs in two ways:

  1. It does NOT extract frames — png_bytes_per_phase is supplied by the
     caller, who has already paid the ffmpeg cost once for both pipelines.
  2. It does NOT write a "failed" row on per-phase failure (per PR-3
     simplification). A failure means yolo_keypoints_2d stays NULL on
     that row, and the frontend builder falls back to SAM data.
"""

import asyncio
import logging

from yolo.inference import MODEL_NAME, infer_pose
from yolo.supabase_writer import write_yolo_phase

logger = logging.getLogger(__name__)


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
    Run YOLO inference for one phase and write the result. Catches any
    exception so the orchestrator's gather() always sees a normal return.
    A failure here is logged but does NOT write a DB row — the caller
    relies on yolo_keypoints_2d staying NULL to trigger frontend fallback.
    """
    frame_idx = int(timestamp_s * fps) if fps > 0 else 0
    frame_timestamp_ms = int(timestamp_s * 1000)

    try:
        result = await infer_pose(png_bytes)
        if result is None:
            logger.warning(
                f"[yolo] phase {phase_name}: no person detected; "
                f"leaving yolo_keypoints_2d NULL"
            )
            return {
                "phase": phase_name,
                "status": "failed",
                "error": "no person detected",
            }

        await asyncio.to_thread(
            write_yolo_phase,
            video_id=video_id,
            user_id=user_id,
            phase_name=phase_name,
            frame_idx=frame_idx,
            frame_timestamp_ms=frame_timestamp_ms,
            image_width=image_width,
            image_height=image_height,
            yolo_keypoints_2d=result["keypoints_2d"],
            yolo_model=result["model"],
            yolo_inference_ms=result["inference_ms"],
        )
        logger.info(
            f"[yolo] phase {phase_name}: completed "
            f"({result['inference_ms']}ms, model={result['model']})"
        )
        return {"phase": phase_name, "status": "completed"}

    except Exception as e:
        logger.error(
            f"[yolo] phase {phase_name} failed", exc_info=True,
        )
        return {
            "phase": phase_name,
            "status": "failed",
            "error": str(e),
        }


async def yolo_for_all_phases(
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
    Run YOLO11-pose on all supplied phase frames in parallel.

    Returns a summary dict:
        {
          "completed": int,
          "failed":    int,
          "results":   [{"phase": str, "status": "completed"|"failed", ...}],
          "model":     str,    # e.g. "yolo11m-pose"
        }

    Never raises for per-phase failure — always returns a summary.
    """
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

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    summary: dict = {
        "completed": 0,
        "failed": 0,
        "results": [],
        "model": MODEL_NAME.replace(".pt", ""),
    }
    for raw in raw_results:
        if isinstance(raw, BaseException):
            logger.error(f"[yolo] phase task escaped its own handler: {raw!r}")
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
        f"[yolo] yolo_for_all_phases: completed={summary['completed']} "
        f"failed={summary['failed']}"
    )
    return summary
