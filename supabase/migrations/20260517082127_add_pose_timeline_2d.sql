-- ============================================================================
-- PR-4: pose_timeline_2d — 17-keypoint COCO frame-level timeline
-- ============================================================================
-- Data foundation for PR-5+ (rotation discs, sway markers, paths, etc.).
-- One JSONB column on swing_videos, NULL for pre-PR-4 videos.
--
-- Strategy:
--   - Additive only — swing_videos columns / RLS / triggers unchanged.
--   - Versioned JSON shape (`version: 1`) so future schema bumps don't
--     break existing readers.
--   - NULL is a valid value and means "this video has no pose timeline
--     available" — the frontend skeleton overlay toggle is disabled in
--     that case.
--
-- See docs/decisions/PR-4_DESIGN.md for the full design.
-- ============================================================================

ALTER TABLE public.swing_videos
  ADD COLUMN IF NOT EXISTS pose_timeline_2d JSONB;

COMMENT ON COLUMN public.swing_videos.pose_timeline_2d IS
  'PR-4: 17-keypoint COCO frame-level timeline (see docs/decisions/PR-4_DESIGN.md). NULL for pre-PR-4 videos and for videos where MediaPipe failed to extract any landmarks.';
