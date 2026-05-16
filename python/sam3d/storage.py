"""
Supabase Storage uploader for phase frames (raw httpx).

We bypass supabase-py because v2.7.4's create_client() rejects the
sb_secret_* key format the project has migrated to. PostgREST + Storage
REST endpoints accept the new keys directly via the apikey + Authorization
headers, so the SDK gives us nothing we need here.
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

BUCKET = "swing-frames"


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
    """
    Build headers for sb_secret_* keys.

    Per Supabase docs: new secret keys are not JWTs. We send the key in both
    `apikey` and `Authorization: Bearer` — Supabase explicitly permits this
    case and identifies the caller as service_role.
    """
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }


def ensure_bucket() -> None:
    """
    Best-effort check that the swing-frames bucket exists.

    The migration `{TS}_swing_frames_bucket.sql` already creates it; this is
    a defensive log-only probe. Any failure to check is non-fatal and the
    pipeline proceeds.
    """
    try:
        url, key = _get_config()
        r = httpx.get(
            f"{url}/storage/v1/bucket/{BUCKET}",
            headers=_headers(key),
            timeout=5.0,
        )
        if r.status_code == 200:
            logger.info("swing-frames bucket exists")
        else:
            logger.warning(
                f"swing-frames bucket check returned {r.status_code}: "
                f"{r.text[:200]}. Make sure the migration has been run."
            )
    except Exception as e:
        logger.warning(f"bucket check failed (non-fatal): {e!r}")


async def upload_phase_frame(
    video_id: str,
    phase_name: str,
    png_bytes: bytes,
) -> str:
    """
    Upload a phase frame and return its public URL.

    Path: `{video_id}/{phase_name}.png` in bucket `swing-frames` (public).
    Uses `x-upsert: true` so re-running a phase overwrites instead of erroring.
    """
    url, key = _get_config()
    path = f"{video_id}/{phase_name}.png"

    headers = _headers(key)
    headers["Content-Type"] = "image/png"
    headers["x-upsert"] = "true"

    upload_url = f"{url}/storage/v1/object/{BUCKET}/{path}"

    async with httpx.AsyncClient() as client:
        r = await client.post(
            upload_url,
            headers=headers,
            content=png_bytes,
            timeout=30.0,
        )

    if r.status_code not in (200, 201):
        raise RuntimeError(
            f"storage upload failed {r.status_code}: {r.text[:300]}"
        )

    public_url = f"{url}/storage/v1/object/public/{BUCKET}/{path}"
    logger.info(f"uploaded {path} -> {public_url}")
    return public_url
