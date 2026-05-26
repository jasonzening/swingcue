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
import traceback
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# WHAM Modal app + function names (matches python/pilot/modal_app.py +
# python/pilot/runners/wham_runner.py @app.function decorators).
MODAL_APP_NAME       = "swingcue-pilot"
MODAL_FUNCTION_NAME  = "infer_video"

# ---------------------------------------------------------------------------
# PR-8c.3: wham_failure_stage enum.
#
# PR-8d.0 frontend will switch on this stage to render targeted error
# UX (e.g., "video too short" vs "service unavailable" vs "WHAM model
# crashed"). Stable enum + free-text wham_error_message gives stage-
# specific messaging without locking us into stage-specific schema.
#
# DO NOT add new values without coordinating with PR-8d.0 frontend.
# Use STAGE_UNKNOWN as the catch-all for unexpected exception classes.
# ---------------------------------------------------------------------------

STAGE_DISPATCH      = "dispatch"        # Modal call setup/lookup/auth fail
STAGE_DOWNLOAD      = "download"        # _download_video HTTP/network fail
STAGE_PREPROCESSING = "preprocessing"   # scene-cut, duration guard (PR-8c.4)
STAGE_SLAM_INIT     = "slam_init"       # WHAM DPVO/SLAM init fail
STAGE_INFERENCE     = "inference"       # WHAM inference OOM/exception
STAGE_POSTPROCESS   = "postprocess"     # pkl load / Supabase persistence fail
STAGE_TIMEOUT       = "timeout"         # Modal function-level timeout (>180s)
STAGE_UNKNOWN       = "unknown"         # catch-all

_VALID_STAGES: frozenset[str] = frozenset({
    STAGE_DISPATCH, STAGE_DOWNLOAD, STAGE_PREPROCESSING, STAGE_SLAM_INIT,
    STAGE_INFERENCE, STAGE_POSTPROCESS, STAGE_TIMEOUT, STAGE_UNKNOWN,
})

# Max length for wham_error_message stored in jsonb — full traceback
# can be 10+ KB which bloats swing_analysis rows + jsonb queries.
# 4KB covers a deep traceback comfortably.
_WHAM_ERROR_MESSAGE_MAX_CHARS = 4000

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
    duration_sec: float | None = None,
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

        # ── PR-8c.4: pre-flight checks (duration + scene-cut) ─────────
        # Reject obvious incompatibilities BEFORE burning Modal $ on
        # WHAM inference. Rejection writes failure context to
        # swing_analysis only (no wham_video_meta row — that table is
        # for actual WHAM runs, not preprocessing rejections).
        preflight = _preflight_check_video(video_id, signed_url, duration_sec)
        if not preflight["ok"]:
            stage = preflight["stage"]
            err = preflight["error_message"]
            logger.info(
                f"[wham_integration] {video_id} REJECTED at preflight "
                f"stage={stage}: {err}"
            )
            set_wham_failed_status(video_id, stage, err)
            return  # do NOT dispatch Modal

        # ── Idempotency ──────────────────────────────────────────────
        existing = _select_wham_video_meta_status(video_id)
        if existing == "completed":
            logger.info(f"[wham_integration] {video_id} already completed, skipping")
            return
        is_retry = existing == "failed"

        # ── Modal call (synchronous, ~55s) ───────────────────────────
        # Stage-classify exceptions here so dispatch / network / WHAM-
        # internal errors all flow through set_wham_failed_status with
        # the correct stage instead of generic 'unknown'.
        try:
            result = _call_modal_infer_video(video_id, signed_url)
        except Exception as modal_exc:
            stage = classify_exception_to_stage(modal_exc)
            tb = traceback.format_exc()
            logger.error(
                f"[wham_integration] {video_id} Modal call raised "
                f"{type(modal_exc).__name__} (stage={stage})"
            )
            set_wham_failed_status(video_id, stage, tb)
            return

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
            # Modal returned status='failed'. The error_message from
            # WHAM is in result['error_message']; classify by its content
            # to pick a stage (download / slam_init / inference / etc.).
            err_msg = result.get("error_message") or "Modal returned status=failed"
            # Build a synthetic exception just to reuse the classifier.
            fake_exc = RuntimeError(err_msg)
            stage = classify_exception_to_stage(fake_exc)
            logger.info(
                f"[wham_integration] {video_id} Modal returned failed "
                f"(stage={stage})"
            )
            set_wham_failed_status(video_id, stage, err_msg)

        logger.info(f"[wham_integration] {video_id} done (status={modal_status})")

    except Exception as exc:
        # Outer catch — pkl write, wham_pose_timeline INSERT, or any
        # unexpected error after Modal returned. Stage-classify and
        # record full traceback.
        stage = classify_exception_to_stage(exc)
        tb = traceback.format_exc()
        logger.error(
            f"[wham_integration] {video_id} FAILED in outer handler "
            f"(stage={stage}): {exc!r}",
            exc_info=True,
        )
        try:
            set_wham_failed_status(video_id, stage, tb)
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
    PR-8c: write final wham_status='ready' to swing_analysis.video_metadata_json.

    For 'failed', use set_wham_failed_status (PR-8c.3) instead — it
    writes the full 4-field failure context (status + failed_at +
    failure_stage + error_message) atomically.

    Preserves wham_started_at + wham_expected_completion_seconds set
    by set_wham_processing_status. Frontend can compute
    wham_actual_seconds from (now - wham_started_at) if useful.

    Raises on PATCH failure (caller catches in the outer try in
    run_wham_and_persist + re-attempts in its own except block).
    """
    try:
        _patch_swing_analysis_meta(video_id, {"wham_status": status}, raise_on_missing=False)
        logger.info(
            f"[wham_integration] {video_id} swing_analysis.wham_status = {status}"
        )
    except Exception:
        raise


def set_wham_failed_status(
    video_id: str,
    stage: str,
    error_message: str,
) -> None:
    """
    PR-8c.3: write the FULL failure context to swing_analysis.video_metadata_json
    atomically (single PATCH). All 4 fields together so frontend can't
    poll mid-write and see partial state.

      wham_status         = 'failed'
      wham_failed_at      = UTC ISO timestamp (now)
      wham_failure_stage  = stage (validated against _VALID_STAGES)
      wham_error_message  = error_message (truncated to 4KB)

    wham_started_at + wham_expected_completion_seconds preserved
    (read-modify-write merge).

    PR-8c.3 acceptance: wham_status='failed' must NEVER be context-free.
    Every failure path in run_wham_and_persist + pre-flight rejection
    in PR-8c.4 routes through here.

    Never raises — failures are logged and swallowed (UX hint, not
    correctness).
    """
    from datetime import datetime, timezone
    if stage not in _VALID_STAGES:
        logger.warning(
            f"[wham_integration] invalid stage '{stage}' → coercing to '{STAGE_UNKNOWN}'"
        )
        stage = STAGE_UNKNOWN
    msg = (error_message or "")[:_WHAM_ERROR_MESSAGE_MAX_CHARS]
    updates = {
        "wham_status":         "failed",
        "wham_failed_at":      datetime.now(timezone.utc).isoformat(),
        "wham_failure_stage":  stage,
        "wham_error_message":  msg,
    }
    try:
        _patch_swing_analysis_meta(video_id, updates, raise_on_missing=False)
        # Truncate the logged message to a sane preview.
        preview = msg.replace("\n", " ")[:140]
        logger.info(
            f"[wham_integration] {video_id} swing_analysis.wham_status=failed "
            f"stage={stage} msg={preview!r}"
        )
    except Exception:
        logger.exception(
            f"[wham_integration] {video_id} failed to write wham_failed status"
        )


def classify_exception_to_stage(exc: BaseException) -> str:
    """
    PR-8c.3 R1: map an exception to a wham_failure_stage enum value
    using class name + message heuristics. Used by run_wham_and_persist's
    outer except so the failure stage isn't a free-form string.

    Priority order:
      1. Modal-specific exception classes (auth, timeout)
      2. Substring hints in str(exc) (download / SLAM / OOM / postprocess)
      3. STAGE_UNKNOWN fallback

    Caller still records the full traceback in wham_error_message;
    stage is the coarse-grained category for frontend dispatch.
    """
    typename = type(exc).__name__
    msg = str(exc) or ""
    msg_lower = msg.lower()

    # Modal exception classes — pattern match by name since modal lib
    # may not be available everywhere this helper is called.
    if (
        "AuthError" in typename
        or "InvalidError" in typename
        or "Authentication" in msg
        or "token" in msg_lower and "missing" in msg_lower
    ):
        return STAGE_DISPATCH
    if "FunctionTimeoutError" in typename or "TimeoutError" in typename:
        return STAGE_TIMEOUT
    if "timeout" in msg_lower and "modal" in msg_lower:
        return STAGE_TIMEOUT

    # WHAM-internal hints.
    if (
        "dpvo" in msg_lower
        or "slam" in msg_lower
        or "slam_init" in msg_lower
    ):
        return STAGE_SLAM_INIT
    if (
        "http error" in msg_lower
        or "urlretrieve" in msg_lower
        or "url error" in msg_lower
        or "_download_video" in msg
    ):
        return STAGE_DOWNLOAD
    if (
        "cuda out of memory" in msg_lower
        or "out of memory" in msg_lower
        or "oom" in msg_lower
    ):
        return STAGE_INFERENCE
    if (
        "pkl" in msg_lower
        or "joblib" in msg_lower
        or "supabase" in msg_lower
        or "PATCH failed" in msg
        or "INSERT failed" in msg
    ):
        return STAGE_POSTPROCESS

    return STAGE_UNKNOWN


def _preflight_check_video(
    video_id: str,
    signed_url: str,
    duration_sec: float | None,
) -> dict:
    """
    PR-8c.4: gate WHAM dispatch on cheap pre-checks. Saves Modal $
    when the video is obviously incompatible with single-clip WHAM
    inference.

    Checks (in order):
      1. Duration guard (R3): duration_sec < 3.0 → reject preprocessing.
      2. Scene-cut detection (R4): cv2 HSV BHATTACHARYYA over the
         downloaded video; any cut → reject preprocessing.

    Returns:
      {"ok": True}                              — pass, dispatch WHAM
      {"ok": False, "stage": str,
       "error_message": str}                    — reject; caller skips
                                                  Modal + writes failure
                                                  context to
                                                  swing_analysis only
                                                  (no wham_video_meta).

    Per PR-8c.4 R2: rejection uses wham_status='failed' + stage=
    'preprocessing' — does NOT introduce a 4th status state.
    """
    MIN_DURATION_SEC = 3.0

    if duration_sec is None:
        # Not enough info to evaluate; pass and let Modal decide.
        # (Should rarely happen since main.py always has durationSec
        # from MediaPipe by this point.)
        logger.warning(
            f"[wham_integration] {video_id} preflight: duration_sec is None; "
            f"skipping duration guard"
        )
    elif duration_sec < MIN_DURATION_SEC:
        return {
            "ok": False,
            "stage": STAGE_PREPROCESSING,
            "error_message": (
                f"Video too short for analysis ({duration_sec:.1f}s). "
                f"Please upload at least {MIN_DURATION_SEC:.0f} seconds of "
                f"a single golf swing."
            ),
        }

    # Scene-cut detection requires a local copy of the video.
    # Download to /tmp via the same urllib helper Modal uses, then
    # run the cv2 detector. ~1-3 sec for a typical 4-8s clip.
    import tempfile
    import urllib.parse
    import urllib.request

    parts = urllib.parse.urlsplit(signed_url)
    safe_path = urllib.parse.quote(parts.path, safe="/")
    encoded_url = urllib.parse.urlunsplit((
        parts.scheme, parts.netloc, safe_path, parts.query, parts.fragment,
    ))

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False, prefix=f"preflight_{video_id}_",
        ) as f:
            tmp_path = f.name
        urllib.request.urlretrieve(encoded_url, tmp_path)
        from scene_detect import detect_scene_cuts
        cuts = detect_scene_cuts(tmp_path)
    except Exception as exc:
        # Could not download or scan — treat as suspicious + reject
        # so Modal $ isn't burned. The user will get a clear message.
        logger.warning(
            f"[wham_integration] {video_id} preflight scene-cut scan failed: {exc!r}"
        )
        return {
            "ok": False,
            "stage": STAGE_PREPROCESSING,
            "error_message": (
                "Could not analyze video for preprocessing checks. "
                "Please re-upload a single continuous clip of one golf swing."
            ),
        }
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if cuts < 0:
        return {
            "ok": False,
            "stage": STAGE_PREPROCESSING,
            "error_message": (
                "Could not read video for scene-cut analysis. "
                "Please upload a single continuous clip of one golf swing."
            ),
        }
    if cuts >= 1:
        return {
            "ok": False,
            "stage": STAGE_PREPROCESSING,
            "error_message": (
                f"Multi-scene video detected ({cuts} scene cut(s)). "
                f"SwingCue MVP only supports a single continuous clip of one "
                f"golf swing — no edited/TikTok/multi-scene compilations. "
                f"Please re-upload."
            ),
        }
    logger.info(
        f"[wham_integration] {video_id} preflight OK: duration={duration_sec}s, cuts=0"
    )
    return {"ok": True}
