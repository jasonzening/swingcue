"""
Supabase Storage uploader for phase frames.

Uploads PNG bytes to bucket `swing-frames` at path `{video_id}/{phase}.png`
and returns the public URL (consumed by fal-ai/sam-3/3d-body).

The bucket itself is created by migration
`supabase/migrations/{TS}_swing_frames_bucket.sql`; `ensure_bucket()` is a
defensive check that logs a warning and best-effort-creates the bucket if it
is missing (e.g. fresh local Supabase without migrations applied).
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

BUCKET_ID = "swing-frames"


def ensure_bucket(client) -> None:
    """
    Defensive check: confirm `swing-frames` bucket exists. If missing, log
    a WARNING and attempt to create it (public read). Migrations should
    normally have created it already.
    """
    try:
        buckets = client.storage.list_buckets()
    except Exception as e:
        logger.warning(
            f"ensure_bucket: could not list buckets ({e}); "
            "assuming it exists and continuing"
        )
        return

    existing = {getattr(b, "id", None) or getattr(b, "name", None) for b in buckets}
    if BUCKET_ID in existing:
        return

    logger.warning(
        f"ensure_bucket: bucket {BUCKET_ID!r} missing — creating it now. "
        "Run swing_frames_bucket migration to make this idempotent."
    )
    try:
        client.storage.create_bucket(
            BUCKET_ID,
            options={"public": True, "file_size_limit": 5242880},
        )
    except Exception as e:
        logger.error(f"ensure_bucket: create_bucket failed: {e}")


async def upload_phase_frame(
    client,
    video_id: str,
    phase_name: str,
    png_bytes: bytes,
) -> str:
    """
    Upload one phase frame and return its public URL.

    Path: `{video_id}/{phase_name}.png`. Uses upsert=True so a re-run for the
    same (video, phase) overwrites the existing object instead of erroring.

    The supabase-py SDK is sync; we offload to a thread so the orchestrator's
    asyncio.gather can run all 5 uploads concurrently.
    """
    path = f"{video_id}/{phase_name}.png"

    def _do_upload() -> str:
        client.storage.from_(BUCKET_ID).upload(
            path=path,
            file=png_bytes,
            file_options={
                "content-type": "image/png",
                "upsert": "true",
            },
        )
        return client.storage.from_(BUCKET_ID).get_public_url(path)

    public_url: str = await asyncio.to_thread(_do_upload)
    # supabase-py sometimes returns the URL with a trailing '?' — strip it
    return public_url.rstrip("?")
