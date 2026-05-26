"""
wham_integration.py — PR-8c Railway-side WHAM orchestration.

Called as a FastAPI BackgroundTask from main.py /analyze AFTER the
existing MediaPipe response is returned to the user. WHAM inference
runs synchronously on Modal (~55s for a 4s video), and the result is
persisted to Supabase using the service-role key.

Per PR-8c spec:
  - Trigger AFTER MediaPipe completes (existing happy path).
  - Synchronous wait on Modal infer_video is fine.
  - MediaPipe response served while WHAM processes (BackgroundTask
    pattern achieves this in FastAPI).
  - service-role key from Railway env (existing SUPABASE_SERVICE_ROLE_KEY,
    same var used by sam3d/ and yolo/ writers).
  - Idempotency: skip if status='completed', UPDATE if 'failed', INSERT if none.
  - Failure: write meta with status='failed'+error_message; NO timeline rows.
  - MediaPipe pipeline UNAFFECTED in all failure modes (BackgroundTask
    exceptions never propagate to the response handler).

Env vars required (all already set on Railway except MODAL_TOKEN_*):
  SUPABASE_URL                    (existing — used by yolo/sam3d)
  SUPABASE_SERVICE_ROLE_KEY       (existing — used by yolo/sam3d)
  MODAL_TOKEN_ID                  (NEW — needs to be added to Railway)
  MODAL_TOKEN_SECRET              (NEW — needs to be added to Railway)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# WHAM Modal app + function names (matches python/pilot/modal_app.py +
# python/pilot/runners/wham_runner.py @app.function decorators).
MODAL_APP_NAME       = "swingcue-pilot"
MODAL_FUNCTION_NAME  = "infer_video"

# PR-8b.1 audit constant — folded into joint_index_mapping jsonb so
# wham_video_meta DDL doesn't need a wham_commit column. wham_runner
# already includes this string in meta.wham_commit, but the DB column
# `wham_model` only takes the model name; commit goes in jsonb.
WHAM_COMMIT_SHORT    = "2b54f77"

# Modal timeout — wham_runner has its own 600s timeout on the @app.function;
# we wait a little less so we get a clean Python-side timeout error if
# Modal hangs.
MODAL_REMOTE_TIMEOUT_SEC = 180

# Supabase REST timeouts.
SUPA_TIMEOUT_SEC = 30.0


# ---------------------------------------------------------------------------
# Supabase REST helpers (lifted from python/yolo/supabase_writer.py pattern).
# ---------------------------------------------------------------------------

def _get_supa_config() -> tuple[str, str]:
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY env vars are required"
        )
    return url, key


def _supa_headers(key: str) -> dict[str, str]:
    return {
        "apikey":        key,
        "Authorization": f"Bearer {key}",
        "Content-Type":  "application/json",
    }


# ---------------------------------------------------------------------------
# Public entrypoint — wired into main.py as BackgroundTask.
# ---------------------------------------------------------------------------

def run_wham_and_persist(
    video_id: str,
    user_id: str,
    signed_url: str,
    expected_seconds: int | None = None,
) -> None:
    """
    Synchronous WHAM → Supabase pipeline. Safe to call from a FastAPI
    BackgroundTask. Never raises — all failures are caught and logged
    so the MediaPipe response (already sent to the user) is never
    affected.

    Idempotency:
      - SELECT status FROM wham_video_meta WHERE video_id = ?
        - status='completed' → skip
        - status='failed'    → retry (UPDATE existing row + re-INSERT timeline)
        - none               → INSERT new row + INSERT timeline

    Args:
        video_id:    swing_videos.id (UUID string).
        user_id:     swing_videos.user_id (UUID string, RLS owner).
        signed_url:  Supabase signed URL (~1200s expiry) for Modal to fetch.
        expected_seconds: PR-8c.1 — dynamic ETA for the 'processing'
            UX hint. If provided, retry-writes wham_status='processing'
            as the FIRST step (catches the race with Next.js INSERT
            that may have caused main.py's synchronous attempt to fail).
            If None (legacy callers), skips the retry-write entirely.
    """
    try:
        logger.info(f"[wham_integration] start video_id={video_id} user={user_id[:8]}...")

        # ── PR-8c.1: ensure wham_status='processing' is set (race retry) ──
        # main.py's synchronous attempt may have failed because Next.js
        # hasn't INSERTed swing_analysis yet (~100-200ms race window).
        # Retry up to 5x with 1s backoff so the UX state machine sees
        # 'processing' even on cold first-time-analysis paths.
        if expected_seconds is not None:
            set_wham_processing_status(
                video_id,
                expected_seconds=expected_seconds,
                max_retries=5,
                retry_delay_sec=1.0,
            )

        # ── Idempotency ──────────────────────────────────────────────
        existing = _select_wham_video_meta_status(video_id)
        if existing == "completed":
            logger.info(f"[wham_integration] {video_id} already completed, skipping")
            return
        is_retry = existing == "failed"

        # ── Modal call (synchronous, ~55s) ───────────────────────────
        result = _call_modal_infer_video(video_id, signed_url)
        modal_status = result.get("status", "unknown")
        logger.info(
            f"[wham_integration] {video_id} modal returned "
            f"status={modal_status} "
            f"frames={len(result.get('frames', []))} "
            f"ms={result.get('inference_ms_total')}"
        )

        # ── Write meta row (always, even on failure) ─────────────────
        _write_wham_video_meta(video_id, user_id, result, is_retry=is_retry)

        # ── Write timeline rows (only on success or partial) ─────────
        if modal_status in ("completed", "partial"):
            # Retry path: delete existing rows for this video before re-insert.
            if is_retry:
                _delete_wham_pose_timeline(video_id)
            _write_wham_pose_timeline(video_id, user_id, result.get("frames", []))
            _set_swing_analysis_wham_status(video_id, "ready")
        else:
            # Failed Modal call — meta row already records the error.
            _set_swing_analysis_wham_status(video_id, "failed")

        logger.info(f"[wham_integration] {video_id} done (status={modal_status})")

    except Exception as exc:
        logger.error(
            f"[wham_integration] {video_id} FAILED: {exc!r}",
            exc_info=True,
        )
        # Best-effort: record the failure on swing_analysis even if the
        # meta/timeline write didn't complete. Don't re-raise — this is
        # a background task and the user response is already served.
        try:
            _set_swing_analysis_wham_status(video_id, "failed")
        except Exception:
            logger.exception(
                f"[wham_integration] {video_id} also failed to mark wham_status"
            )


# ---------------------------------------------------------------------------
# Idempotency check.
# ---------------------------------------------------------------------------

def _select_wham_video_meta_status(video_id: str) -> str | None:
    """Return existing status ('completed'|'failed') or None if no row."""
    url, key = _get_supa_config()
    r = httpx.get(
        f"{url}/rest/v1/wham_video_meta",
        headers=_supa_headers(key),
        params={"video_id": f"eq.{video_id}", "select": "status"},
        timeout=SUPA_TIMEOUT_SEC,
    )
    r.raise_for_status()
    rows = r.json()
    if rows and isinstance(rows, list) and len(rows) > 0:
        return rows[0].get("status")
    return None


# ---------------------------------------------------------------------------
# Modal invocation.
# ---------------------------------------------------------------------------

def _call_modal_infer_video(video_id: str, signed_url: str) -> dict:
    """Synchronously invoke deployed swingcue-pilot.infer_video."""
    # Lazy import — modal is heavy; only load when the BackgroundTask
    # actually runs. Keeps /analyze cold start unaffected.
    import modal

    fn = modal.Function.from_name(MODAL_APP_NAME, MODAL_FUNCTION_NAME)
    logger.info(f"[wham_integration] {video_id} invoking Modal {MODAL_APP_NAME}.{MODAL_FUNCTION_NAME}")
    result = fn.remote(video_url=signed_url, video_id=video_id)
    if not isinstance(result, dict):
        raise RuntimeError(
            f"Modal returned unexpected type {type(result).__name__}: {result!r}"
        )
    return result


# ---------------------------------------------------------------------------
# Supabase writes.
# ---------------------------------------------------------------------------

def _write_wham_video_meta(
    video_id: str,
    user_id: str,
    modal_result: dict,
    *,
    is_retry: bool,
) -> None:
    """
    INSERT new row OR UPDATE existing failed row with the Modal result.

    Schema (from MCP inspection 2026-05-25):
      wham_video_meta (id, video_id UNIQUE, user_id, source, joint_type,
        coordinate_space_2d, coordinate_space_3d, wham_model, modal_call_id,
        inference_ms_total, status, error_message, image_width, image_height,
        processed_fps, frame_count, camera jsonb, joint_index_mapping jsonb,
        created_at, updated_at)
    """
    url, key = _get_supa_config()
    meta = modal_result.get("meta", {}) or {}
    # PR-8b.1: wham_commit not a column — fold into joint_index_mapping.
    jim = dict(meta.get("joint_index_mapping") or {})
    jim["_wham_commit"] = WHAM_COMMIT_SHORT
    # Schema sanity — fall back to known constants on missing fields.
    payload: dict[str, Any] = {
        "video_id":             video_id,
        "user_id":              user_id,
        "source":               meta.get("source", "wham_smplh_v1"),
        "joint_type":           meta.get("joint_type", "bone_center"),
        "coordinate_space_2d":  meta.get("coordinate_space_2d", "video_px"),
        "coordinate_space_3d":  meta.get("coordinate_space_3d", "smpl_world_m"),
        "wham_model":           meta.get("wham_model", "wham_vit_w_3dpw"),
        "modal_call_id":        modal_result.get("modal_call_id"),
        "inference_ms_total":   modal_result.get("inference_ms_total"),
        "status":               modal_result.get("status", "failed"),
        "error_message":        modal_result.get("error_message"),
        "image_width":          int(meta.get("image_width", 0)),
        "image_height":         int(meta.get("image_height", 0)),
        "processed_fps":        float(meta.get("processed_fps", 0.0)),
        "frame_count":          int(meta.get("frame_count", 0)),
        "camera":               meta.get("camera") or {},
        "joint_index_mapping":  jim,
    }

    if is_retry:
        # Retry: UPDATE existing failed row.
        endpoint = f"{url}/rest/v1/wham_video_meta?video_id=eq.{video_id}"
        headers = _supa_headers(key) | {"Prefer": "return=minimal"}
        r = httpx.patch(endpoint, headers=headers, json=payload, timeout=SUPA_TIMEOUT_SEC)
        if r.status_code not in (200, 204):
            raise RuntimeError(
                f"wham_video_meta UPDATE failed {r.status_code}: {r.text[:400]}"
            )
        logger.info(f"[wham_integration] {video_id} wham_video_meta UPDATED (retry)")
    else:
        # First write: INSERT.
        endpoint = f"{url}/rest/v1/wham_video_meta"
        headers = _supa_headers(key) | {"Prefer": "return=minimal"}
        r = httpx.post(endpoint, headers=headers, json=payload, timeout=SUPA_TIMEOUT_SEC)
        if r.status_code not in (200, 201):
            raise RuntimeError(
                f"wham_video_meta INSERT failed {r.status_code}: {r.text[:400]}"
            )
        logger.info(f"[wham_integration] {video_id} wham_video_meta INSERTED")


def _delete_wham_pose_timeline(video_id: str) -> None:
    """Retry path: clear existing rows before re-INSERT to avoid duplicates."""
    url, key = _get_supa_config()
    endpoint = f"{url}/rest/v1/wham_pose_timeline?video_id=eq.{video_id}"
    headers = _supa_headers(key) | {"Prefer": "return=minimal"}
    r = httpx.delete(endpoint, headers=headers, timeout=SUPA_TIMEOUT_SEC)
    if r.status_code not in (200, 204):
        raise RuntimeError(
            f"wham_pose_timeline DELETE failed {r.status_code}: {r.text[:400]}"
        )
    logger.info(f"[wham_integration] {video_id} wham_pose_timeline cleared for retry")


def _write_wham_pose_timeline(
    video_id: str,
    user_id: str,
    frames: list[dict],
) -> None:
    """
    Bulk INSERT one row per frame. PostgREST accepts array of objects
    for batch INSERT.

    Schema notes:
      smpl_pose:               NOT NULL (jsonb) — Modal returns null when
                               save_smpl_params=False. Coerce None → [].
      keypoints_2d_projected:  NOT NULL (jsonb) — Modal returns null for
                               rule-1-failure frames (3D non-finite).
                               Coerce None → {}.
      smpl_shape, smpl_trans, keypoints_3d_smpl, fit_quality: nullable.
    """
    if not frames:
        logger.warning(f"[wham_integration] {video_id} no frames to write")
        return

    url, key = _get_supa_config()
    rows: list[dict[str, Any]] = []
    for f in frames:
        rows.append({
            "video_id":               video_id,
            "user_id":                user_id,
            "frame_idx":              int(f.get("frame_idx", 0)),
            "frame_timestamp_ms":     f.get("frame_timestamp_ms"),
            "smpl_pose":              f.get("smpl_pose") or [],   # NOT NULL coerce
            "smpl_shape":             f.get("smpl_shape"),
            "smpl_trans":             f.get("smpl_trans"),
            "keypoints_2d_projected": f.get("keypoints_2d_projected") or {},  # NOT NULL
            "keypoints_3d_smpl":      f.get("keypoints_3d_smpl"),
            "fit_ok":                 bool(f.get("fit_ok", True)),
            "fit_quality":            f.get("fit_quality"),
        })

    endpoint = f"{url}/rest/v1/wham_pose_timeline"
    headers = _supa_headers(key) | {"Prefer": "return=minimal"}
    # Bulk POST — PostgREST handles arrays natively.
    # Use a longer timeout because the payload can be ~2-5MB for 120 frames
    # (keypoints_2d/3d each ~500 bytes/frame × 120 ≈ 120KB; smpl_shape +
    # other fields bring it up). Network jitter on Railway → 60s safe.
    r = httpx.post(endpoint, headers=headers, json=rows, timeout=60.0)
    if r.status_code not in (200, 201):
        raise RuntimeError(
            f"wham_pose_timeline bulk INSERT failed "
            f"{r.status_code} ({len(rows)} rows): {r.text[:400]}"
        )
    logger.info(
        f"[wham_integration] {video_id} wham_pose_timeline INSERTED {len(rows)} rows"
    )


class _RowNotFound(Exception):
    """Raised by _patch_swing_analysis_meta when swing_analysis row
    doesn't exist yet (race vs Next.js INSERT). Distinguishable from
    other errors so the BackgroundTask retry loop knows when to wait
    vs when to give up."""


def _patch_swing_analysis_meta(
    video_id: str,
    updates: dict,
    raise_on_missing: bool = True,
) -> bool:
    """
    Read-modify-write swing_analysis.video_metadata_json with the
    given updates dict (merged into existing jsonb so other keys
    like durationSec / fps / dataSource are preserved).

    Returns True on successful PATCH. If swing_analysis row doesn't
    exist for this video_id:
      - raise_on_missing=True → raises _RowNotFound
      - raise_on_missing=False → logs warning + returns False
    Other errors (network / RLS / 5xx) always raise.
    """
    url, key = _get_supa_config()
    r = httpx.get(
        f"{url}/rest/v1/swing_analysis",
        headers=_supa_headers(key),
        params={"video_id": f"eq.{video_id}", "select": "id,video_metadata_json"},
        timeout=SUPA_TIMEOUT_SEC,
    )
    r.raise_for_status()
    rows = r.json()
    if not rows:
        msg = (
            f"swing_analysis row not found for video_id={video_id} "
            f"(likely race vs Next.js INSERT)"
        )
        if raise_on_missing:
            raise _RowNotFound(msg)
        logger.warning(f"[wham_integration] {msg}; PATCH skipped")
        return False
    analysis_id = rows[0]["id"]
    current = rows[0].get("video_metadata_json") or {}
    if not isinstance(current, dict):
        current = {}
    current.update(updates)
    patch_url = f"{url}/rest/v1/swing_analysis?id=eq.{analysis_id}"
    patch_headers = _supa_headers(key) | {"Prefer": "return=minimal"}
    r2 = httpx.patch(
        patch_url, headers=patch_headers,
        json={"video_metadata_json": current},
        timeout=SUPA_TIMEOUT_SEC,
    )
    if r2.status_code not in (200, 204):
        raise RuntimeError(
            f"swing_analysis PATCH failed {r2.status_code}: {r2.text[:400]}"
        )
    return True


def set_wham_processing_status(
    video_id: str,
    expected_seconds: int,
    max_retries: int = 1,
    retry_delay_sec: float = 1.0,
) -> bool:
    """
    PR-8c.1 (R1+R2): Write wham_status='processing' + wham_started_at +
    wham_expected_completion_seconds to swing_analysis.video_metadata_json.

    Returns True on success, False on any failure (per R2: UX hint, not
    correctness — caller continues regardless).

    Usage:
      - main.py synchronous attempt (max_retries=1): may fail if
        swing_analysis row doesn't exist yet (Next.js INSERT race);
        caller logs + continues.
      - BackgroundTask retry loop (max_retries=5, delay=1.0s): catches
        the Next.js INSERT race window. ~5s budget covers a ~200ms race
        with margin.

    Never raises — all failures are caught + logged.
    """
    import time
    from datetime import datetime, timezone

    updates = {
        "wham_status":                       "processing",
        "wham_started_at":                   datetime.now(timezone.utc).isoformat(),
        "wham_expected_completion_seconds":  int(expected_seconds),
    }
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            if _patch_swing_analysis_meta(video_id, updates, raise_on_missing=True):
                logger.info(
                    f"[wham_integration] {video_id} wham_status='processing' "
                    f"(expected={expected_seconds}s, attempt {attempt+1}/{max_retries})"
                )
                return True
        except _RowNotFound as exc:
            last_err = exc
            if attempt < max_retries - 1:
                time.sleep(retry_delay_sec)
                continue
        except Exception as exc:
            # Non-race error — don't retry, just give up.
            last_err = exc
            break

    logger.warning(
        f"[wham_integration] {video_id} wham_status='processing' write failed "
        f"after {max_retries} attempt(s): {last_err!r}"
    )
    return False


def _set_swing_analysis_wham_status(video_id: str, status: str) -> None:
    """
    PR-8c: write final wham_status='ready'|'failed' to
    swing_analysis.video_metadata_json. PR-8c.1 R3: failure paths
    MUST call this with 'failed' to clear the 'processing' state
    that PR-8c.1 set synchronously — otherwise frontend polling sees
    the row stuck at 'processing' forever.

    Preserves wham_started_at + wham_expected_completion_seconds set
    by set_wham_processing_status (read-modify-write merges; doesn't
    clobber). Frontend can compute wham_actual_seconds from
    (now - wham_started_at) if useful.

    Raises on PATCH failure (caller catches in the outer try in
    run_wham_and_persist + re-attempts in its own except block).
    """
    try:
        _patch_swing_analysis_meta(video_id, {"wham_status": status}, raise_on_missing=False)
        logger.info(
            f"[wham_integration] {video_id} swing_analysis.wham_status = {status}"
        )
    except Exception:
        # Re-raise with context — outer caller in run_wham_and_persist
        # has its own try/except for the final status update.
        raise
