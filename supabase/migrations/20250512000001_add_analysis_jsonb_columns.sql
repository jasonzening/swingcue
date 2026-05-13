-- =============================================================================
-- Migration: 20250512000001_add_analysis_jsonb_columns
-- Purpose: Add JSONB columns to swing_analysis for storing analysis pipeline
--          data (overlay timeline, phase markers, video metadata, keypoints).
--          These columns are written by the /api/analyze route.
-- Additive: YES (no destructive changes)
-- =============================================================================

alter table swing_analysis
  add column if not exists video_metadata_json     jsonb,
  add column if not exists phase_markers_json      jsonb,
  add column if not exists overlay_timeline_json   jsonb,
  add column if not exists keypoint_timeline_json  jsonb;

comment on column swing_analysis.video_metadata_json    is 'Video metadata from analysis: {durationSec, fps, width, height, dataSource}';
comment on column swing_analysis.phase_markers_json     is 'Swing phase timestamps in seconds: {setupTime, topTime, transitionTime, impactTime, finishTime}';
comment on column swing_analysis.overlay_timeline_json  is 'Pre-computed overlay frames for the player: {frames: [{time, phase, elements}]}';
comment on column swing_analysis.keypoint_timeline_json is 'Raw MediaPipe keypoint data per frame: {frames: [{time, landmarks}]}';
