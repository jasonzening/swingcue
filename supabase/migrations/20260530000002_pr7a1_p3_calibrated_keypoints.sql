-- PR-7A.1 Phase 3: per-joint drag-to-correct calibration data.
-- Additive, nullable. No impact on existing rows or RLS.
-- Already applied to prod via MCP on 2026-05-30; this file mirrors
-- that state for fresh-local-dev / CI parity. Idempotent via IF NOT
-- EXISTS so it's safe to re-run on any DB.

ALTER TABLE landmark_validation_review
  ADD COLUMN IF NOT EXISTS calibrated_keypoints JSONB;

COMMENT ON COLUMN landmark_validation_review.calibrated_keypoints IS
  'Per-joint corrected pixel positions after drag-to-correct (PR-7A.1 Phase 3). '
  'Schema: {lead_shoulder:{x,y}, lead_elbow:{x,y}, lead_wrist:{x,y}, '
  'trail_shoulder:{x,y}, trail_elbow:{x,y}, trail_wrist:{x,y}, '
  'lead_hip:{x,y}, trail_hip:{x,y}}. '
  'Original GT lives in golf_landmark_annotations untouched; '
  'delta = calibrated - original is the PR-7B training signal.';
