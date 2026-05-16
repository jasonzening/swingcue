"""
fal-ai/sam-3/3d-body client wrapper with retry.

Wraps `fal_client.submit_async` with 2 retries (3 total attempts) and
exponential backoff (1s, 4s). Network blips and transient fal 5xx are common
enough that surfacing them as a permanent failure is wasteful; persistent
failures still surface to the caller after retries are exhausted.

Auth: `FAL_KEY` environment variable, format `uuid:hash`. fal_client picks
this up automatically — we do not pass it explicitly.
"""

import asyncio
import logging

import fal_client

logger = logging.getLogger(__name__)

MODEL_ID = "fal-ai/sam-3/3d-body"
MAX_RETRIES = 2  # 2 retries → 3 total attempts
BACKOFF_BASE_S = 1.0
BACKOFF_FACTOR = 4.0


async def call_fal(image_url: str) -> dict:
    """
    Call fal sam-3/3d-body with the given public image URL.

    Returns the raw response dict (see CLAUDE_CODE_PR-2.md for schema).
    Raises the last exception after exhausting retries.
    """
    last_err: BaseException | None = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            handler = await fal_client.submit_async(
                MODEL_ID,
                arguments={"image_url": image_url},
            )
            result = await handler.get()
            if attempt > 0:
                logger.warning(
                    f"call_fal: succeeded on retry {attempt} for {image_url}"
                )
            return result
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                delay = BACKOFF_BASE_S * (BACKOFF_FACTOR ** attempt)
                logger.warning(
                    f"call_fal: attempt {attempt + 1}/{MAX_RETRIES + 1} "
                    f"failed for {image_url}: {e!r}; retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"call_fal: all {MAX_RETRIES + 1} attempts failed for "
                    f"{image_url}: {e!r}"
                )

    assert last_err is not None
    raise last_err
