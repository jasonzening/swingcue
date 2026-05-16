"""
ffmpeg-based single-frame extraction.

Async wrapper around `ffmpeg -ss <timestamp> -i <video> -frames:v 1 -q:v 2 -f
image2pipe -vcodec png -` so we can run 5 phase extractions concurrently
without blocking the event loop.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


class FrameExtractError(RuntimeError):
    """Raised when ffmpeg fails to produce a frame."""


async def extract_frame(video_path: str, timestamp_s: float) -> bytes:
    """
    Extract a single PNG frame at `timestamp_s` from `video_path`.

    Uses ffmpeg via subprocess (binary is installed in the Dockerfile).
    Returns the PNG bytes. Raises FrameExtractError on non-zero exit.

    Notes:
      - `-ss` placed BEFORE `-i` does fast (keyframe-bounded) seek; for a
        single frame extraction this is fine because we then re-decode the
        exact frame with `-frames:v 1`.
      - `-q:v 2` gives near-lossless PNG compression (PNG is lossless anyway;
        this is a no-op safety in case the codec changes).
    """
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-ss", f"{timestamp_s:.3f}",
        "-i", video_path,
        "-frames:v", "1",
        "-f", "image2pipe",
        "-vcodec", "png",
        "-",
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="replace").strip()
        raise FrameExtractError(
            f"ffmpeg exit {proc.returncode} at t={timestamp_s:.3f}s: {err}"
        )

    if not stdout:
        raise FrameExtractError(
            f"ffmpeg produced empty output at t={timestamp_s:.3f}s"
        )

    return stdout
