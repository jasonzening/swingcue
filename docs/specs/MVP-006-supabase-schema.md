# MVP-006 Supabase Schema

## swing_videos
Required columns:
- id
- user_id
- storage_path
- original_filename
- file_size_bytes
- duration_ms
- view_type
- status
- error_code
- error_message
- processing_started_at
- processing_completed_at
- created_at
- updated_at

## swing_analysis
Required columns:
- id
- video_id
- issue_type
- summary_text
- cue_text
- drill_text
- overlay_asset_url
- score
- severity
- created_at
- updated_at

## swing_phase_snapshots
Required columns:
- id
- video_id
- phase_name
- snapshot_asset_url
- frame_index
- created_at

## Rules
- additive only
- explicit status
- explicit view type
- ownership-safe
- future real analysis compatibility
- no destructive migration in first wave
