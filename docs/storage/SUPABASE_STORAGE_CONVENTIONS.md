# SUPABASE_STORAGE_CONVENTIONS

## Uploads
Suggested path:
`swings/{user_id}/{video_id}/original/{filename}`

## Stub assets
Suggested path:
- `swings/{user_id}/{video_id}/overlays/{asset_name}`
- `swings/{user_id}/{video_id}/snapshots/{phase_name}-{index}.jpg`

## Rules
- path must remain ownership-compatible
- naming should be deterministic where possible
- conventions must support later replacement by real analysis output
