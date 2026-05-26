# swing_videos.status state machine — documented (PR-8c.5)

## Current schema (CHECK constraint, no migration in PR-8c.5)

```
status TEXT CHECK (status IN ('uploaded', 'processing', 'completed', 'failed'))
```

PR-8c.5 documents the *intended* semantics over the existing enum
without changing the DDL. A future PR can split `'uploaded'` into
`'pending'` + `'uploaded'` if the dual-meaning becomes a real consumer
problem; PR-8c.5 hardens the upload pipeline so consumers don't trip
on the ambiguity in practice.

## Lifecycle (happy path)

```
                       INSERT (Vercel upload page,
                              swing_videos.insert at handleAnalyze)
                                |
                                v
                          status='uploaded'
                          storage_path=''
                                |
                                v          (client uploads blob to Supabase Storage)
                                |
                          [storage upload OK]
                                |
                                v
                  PATCH storage_path = '<user>/<vid>/<vid>.<ext>'
                  status='uploaded' (unchanged)
                                |
                                v       (client POSTs /api/analyze/[id])
                                |
                                v
                          status='processing'
                          processing_started_at=NOW()
                                |
                                v         (Python /analyze runs MediaPipe etc.)
                                |
                          [Railway returns success]
                                |
                                v
                          status='completed'
                          processing_completed_at=NOW()
                          pose_timeline_2d=<jsonb>
```

## Lifecycle (failure paths — all PR-8c.5 hardened)

### Upload pipeline failures (PR-8c.5 Bug 2)
Caught client-side at `src/app/upload/page.tsx`:

| When | error_code |
|---|---|
| Supabase Storage rejected the key | `invalid_storage_key` (defense-in-depth; should not fire after PR-8c.5 Bug 1 fix) |
| Network/fetch error during blob upload | `storage_network_error` |
| Storage upload error of any other shape | `storage_upload_failed` |
| `storage_path` UPDATE failed after blob OK | `storage_path_update_failed` (orphans the blob; sweeper out of scope) |
| Anything else inside `handleAnalyze` catch | `upload_pipeline_error` |

All set `status='failed'` + `error_code` + `error_message` (raw `err.message`, truncated to 2KB).

### Analysis failures (PR-8c.2)
Caught server-side at `/api/analyze/[id]` route:

| When | error_code | swing_analysis state |
|---|---|---|
| Python returned 5xx | `ANALYSIS_FAILED` | `dataSource='analysis_failed'`, `wham_status='failed'`, `wham_error_message='backend_unavailable_5xx'` |
| Python timeout / network | `ANALYSIS_FAILED` | same, with `wham_error_message='backend_timeout'` etc. |

### WHAM-only failures (PR-8c.3 + PR-8c.4)
Caught by `wham_integration.run_wham_and_persist` BackgroundTask. Don't
flip `swing_videos.status` (MediaPipe path succeeded, only the WHAM
augmentation failed). Surfaced via `swing_analysis.video_metadata_json`:

```
wham_status         = 'failed'
wham_failed_at      = <ISO>
wham_failure_stage  = one of: dispatch | download | preprocessing
                              | slam_init | inference | postprocess
                              | timeout | unknown
wham_error_message  = <truncated trace or user message>
```

## Read semantics for consumers (PR-8d.0 frontend)

| swing_videos.status | swing_videos.storage_path | swing_videos.error_code | Meaning |
|---|---|---|---|
| `'uploaded'` | `''` | NULL | Row created; blob still uploading OR upload pipeline crashed before PR-8c.5 PATCH could fire. **Should be rare post-PR-8c.5.** Frontend treats as "in progress" or "broken — retry needed". |
| `'uploaded'` | `<non-empty>` | NULL | Blob present; analysis not yet started. |
| `'processing'` | `<non-empty>` | NULL | MediaPipe / WHAM in flight. Poll `swing_analysis.video_metadata_json.wham_status` for the WHAM sub-state machine. |
| `'completed'` | `<non-empty>` | NULL | Production-ready. |
| `'failed'` | any | `<code>` | Look at `error_code` + `error_message`. Don't try to render results. |

## Why this PR didn't add `'pending'` to the CHECK

The cleanest semantic fix is:
```
status: 'pending' (row created, no blob yet)
status: 'uploaded' (blob in storage, ready for analysis)
status: 'processing'
status: 'completed'
status: 'failed'
```

Adding `'pending'` requires:
1. Supabase migration (`ALTER TABLE swing_videos DROP CONSTRAINT ... ADD CONSTRAINT ... CHECK (status IN ('pending', 'uploaded', ...))`)
2. Backfilling existing rows (decide: legacy `'uploaded'` rows that have storage_path → stay `'uploaded'`; legacy `'uploaded'` rows with empty storage_path → migrate to `'failed'` since they're ghosts?)
3. Frontend reads in `history` page + `result` page would need to handle 5-state instead of 4
4. PR-8d.0 spec lock that frozen the wham_status enum applies — adding a parallel `swing_videos.status` 5th state would echo through the UX

Out of scope for PR-8c.5 (hotfix). Future PR-8e or similar can do the
migration cleanly once PR-8d.0 frontend has shipped + the API
consumers are stable.

## Test scenarios verified after PR-8c.5

1. **CJK filename** — original_filename = `视频测试.mp4` (Chinese chars). Storage key becomes `<user>/<vid>/<vid>.mp4` — pure ASCII. Blob uploads OK. (Previously: "Invalid key" reject.)
2. **Accented filename** — `pôse.mp4`. Same outcome.
3. **Storage upload network error** — disconnect WiFi mid-upload. Row updated to `status='failed'`, `error_code='storage_network_error'`. (Previously: row stuck at `status='uploaded'`, `storage_path=''`.)
4. **Forced storage UPDATE failure** (e.g., RLS-denied row update) — `error_code='storage_path_update_failed'`. Blob exists but orphaned; row marked failed.
5. **Auth expiry mid-flight** — caught by outer catch → `error_code='upload_pipeline_error'`.

PR-8c.5 does NOT verify these via E2E here; they're surfaced for any
follow-up smoke testing.
