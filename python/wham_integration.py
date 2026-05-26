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
    """
    try:
        logger.info(f"[wham_integration] start video_id={video_id} user={user_id[:8]}...")

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


def _set_swing_analysis_wham_status(video_id: str, status: str) -> None:
    """
    UPDATE swing_analysis.video_metadata_json with a wham_status field.
    Per spec: this is the flag PR-8d (frontend) will query.

    Uses PostgREST's `||` jsonb merge via raw SQL is not natively
    available through PostgREST. Instead, fetch existing
    video_metadata_json, merge wham_status, write back.
    """
    url, key = _get_supa_config()
    # Read current video_metadata_json.
    r = httpx.get(
        f"{url}/rest/v1/swing_analysis",
        headers=_supa_headers(key),
        params={"video_id": f"eq.{video_id}", "select": "id,video_metadata_json"},
        timeout=SUPA_TIMEOUT_SEC,
    )
    r.raise_for_status()
    rows = r.json()
    if not rows:
        logger.warning(
            f"[wham_integration] {video_id} no swing_analysis row yet — "
            f"skipping wham_status update (will be set on next analyze call)"
        )
        return
    analysis_id = rows[0]["id"]
    current_meta = rows[0].get("video_metadata_json") or {}
    if not isinstance(current_meta, dict):
        current_meta = {}
    current_meta["wham_status"] = status
    # Write back merged dict.
    patch_url = f"{url}/rest/v1/swing_analysis?id=eq.{analysis_id}"
    patch_headers = _supa_headers(key) | {"Prefer": "return=minimal"}
    r2 = httpx.patch(
        patch_url, headers=patch_headers,
        json={"video_metadata_json": current_meta},
        timeout=SUPA_TIMEOUT_SEC,
    )
    if r2.status_code not in (200, 204):
        raise RuntimeError(
            f"swing_analysis wham_status PATCH failed "
            f"{r2.status_code}: {r2.text[:400]}"
        )
    logger.info(
        f"[wham_integration] {video_id} swing_analysis.wham_status = {status}"
    )
