-- ═══════════════════════════════════════════════════════════════════════════
-- PR-7A.1 Schema migration: anatomical-spec v2 (bone-top-centerline)
-- ───────────────────────────────────────────────────────────────────────────
-- This migration was APPLIED TO PROD via MCP on 2026-05-30 in two MCP calls
-- (initial schema + hardening). This file consolidates both into a single
-- FULLY IDEMPOTENT migration so it's safe to re-run against any DB state:
--   * fresh local dev DB → applies everything cleanly
--   * already-applied prod → all operations are no-ops
--   * partially-applied dev → catches up to final state
-- ═══════════════════════════════════════════════════════════════════════════

-- ── (A) golf_landmark_annotations: relax `arm` to nullable ─────────────────
ALTER TABLE public.golf_landmark_annotations
  ALTER COLUMN arm DROP NOT NULL;

ALTER TABLE public.golf_landmark_annotations
  DROP CONSTRAINT IF EXISTS golf_landmark_annotations_arm_check;
ALTER TABLE public.golf_landmark_annotations
  ADD CONSTRAINT golf_landmark_annotations_arm_check
  CHECK (arm IS NULL OR arm IN ('lead', 'trail'));

-- ── (B) Add hip columns ────────────────────────────────────────────────────
ALTER TABLE public.golf_landmark_annotations
  ADD COLUMN IF NOT EXISTS lead_hip_x  real,
  ADD COLUMN IF NOT EXISTS lead_hip_y  real,
  ADD COLUMN IF NOT EXISTS trail_hip_x real,
  ADD COLUMN IF NOT EXISTS trail_hip_y real;

-- ── (C) Extend task_type enum to include hip_pair ──────────────────────────
ALTER TABLE public.golf_landmark_annotations
  DROP CONSTRAINT IF EXISTS golf_landmark_annotations_task_type_check;
ALTER TABLE public.golf_landmark_annotations
  ADD CONSTRAINT golf_landmark_annotations_task_type_check
  CHECK (task_type IN (
    'manual_gt', 'manual_gt_hip_pair', 'correction_review', 'active_learning'
  ));

-- ── (D) Row-level data coherence (STRICT — hip_pair requires non-null) ─────
-- Arm tasks: arm required, hip cols MUST be null. Shoulder/elbow/wrist
-- coords stay nullable to allow visibility='occluded' partial annotations.
-- Hip-pair tasks: arm null, ALL FOUR hip coords required, no arm coords.
-- If a hip is occluded the annotator must skip the entire task (no row).
ALTER TABLE public.golf_landmark_annotations
  DROP CONSTRAINT IF EXISTS golf_landmark_annotations_task_data_match;
ALTER TABLE public.golf_landmark_annotations
  ADD CONSTRAINT golf_landmark_annotations_task_data_match
  CHECK (
    (task_type IN ('manual_gt', 'correction_review', 'active_learning')
      AND arm IS NOT NULL
      AND lead_hip_x  IS NULL AND lead_hip_y  IS NULL
      AND trail_hip_x IS NULL AND trail_hip_y IS NULL)
    OR
    (task_type = 'manual_gt_hip_pair'
      AND arm IS NULL
      AND lead_hip_x  IS NOT NULL AND lead_hip_y  IS NOT NULL
      AND trail_hip_x IS NOT NULL AND trail_hip_y IS NOT NULL
      AND shoulder_x IS NULL AND shoulder_y IS NULL
      AND elbow_x    IS NULL AND elbow_y    IS NULL
      AND wrist_x    IS NULL AND wrist_y    IS NULL)
  );

-- ── (E) Arm-task uniqueness (partial, arm IS NOT NULL) ─────────────────────
DROP INDEX IF EXISTS public.uq_gla_arm_task;
CREATE UNIQUE INDEX uq_gla_arm_task
  ON public.golf_landmark_annotations
        (video_id, frame_idx, annotator_id, task_type, arm)
  WHERE arm IS NOT NULL;

-- ── (F) Hip-pair uniqueness (partial, task_type filter; arm IS NULL) ───────
-- NULL doesn't participate in normal unique constraints, so this partial
-- index is the only way to prevent duplicate hip_pair rows per
-- (video, frame, annotator).
DROP INDEX IF EXISTS public.uq_gla_hip_pair;
CREATE UNIQUE INDEX uq_gla_hip_pair
  ON public.golf_landmark_annotations
        (video_id, frame_idx, annotator_id)
  WHERE task_type = 'manual_gt_hip_pair';

-- ── (G) Tag v1 rows as deprecated (only matters for prod migration) ────────
-- For fresh local dev DBs, the v1 rows don't exist, so this is a no-op.
-- Idempotent: only updates rows that don't already carry the deprecated tag.
UPDATE public.golf_landmark_annotations
SET source_app_version = COALESCE(source_app_version, '') || ':deprecated-v1'
WHERE source_app_version IS NULL
   OR (source_app_version NOT LIKE '%deprecated%'
       AND source_app_version NOT LIKE 'swingcue-annotate-2.0-anatomical-spec%');

-- ── (H) Column / table comments (overwrite) ────────────────────────────────
COMMENT ON COLUMN public.golf_landmark_annotations.source_app_version IS
  'App version tag. v1 rows (pre-bone-top-centerline) carry suffix ":deprecated-v1". v2 rows use "swingcue-annotate-2.0-anatomical-spec".';

COMMENT ON COLUMN public.golf_landmark_annotations.lead_hip_x IS
  'PR-7A.1: lead-side femoral head center estimate (hip_pair task). Bone-top-centerline principle: NOT the lateral trochanter bump.';
COMMENT ON COLUMN public.golf_landmark_annotations.trail_hip_x IS
  'PR-7A.1: trail-side femoral head center estimate (hip_pair task). Bone-top-centerline principle: NOT the lateral trochanter bump.';

-- ═══════════════════════════════════════════════════════════════════════════
-- (I) landmark_validation_review table (Phase 2 video-overlay verdicts)
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS public.landmark_validation_review (
  id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id            uuid NOT NULL REFERENCES public.swing_videos(id) ON DELETE CASCADE,
  reviewer_id         uuid NOT NULL REFERENCES auth.users(id),
  phase               text NOT NULL CHECK (phase IN ('setup','takeaway','top','transition','impact')),
  verdict             text NOT NULL CHECK (verdict IN ('correct','incorrect','unsure')),
  notes               text,
  compared_sources    text[] NOT NULL DEFAULT ARRAY['wham','mediapipe']::text[],
  source_app_version  text NOT NULL DEFAULT 'swingcue-review-1.0',
  reviewed_at         timestamptz NOT NULL DEFAULT now(),
  UNIQUE (video_id, reviewer_id, phase)
);

COMMENT ON TABLE public.landmark_validation_review IS
  'PR-7A.1 Phase 2: per-phase visual verdict from admin reviewer comparing Jason GT vs WHAM + MediaPipe overlays rendered on video frame. 5/5 ''correct'' per video × 2 videos = anatomical spec method validated.';

CREATE INDEX IF NOT EXISTS idx_lvr_video_phase
  ON public.landmark_validation_review(video_id, phase);

ALTER TABLE public.landmark_validation_review ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS landmark_validation_review_service_role ON public.landmark_validation_review;
CREATE POLICY landmark_validation_review_service_role
  ON public.landmark_validation_review
  FOR ALL TO service_role
  USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS landmark_validation_review_own_select ON public.landmark_validation_review;
CREATE POLICY landmark_validation_review_own_select
  ON public.landmark_validation_review
  FOR SELECT TO authenticated
  USING (reviewer_id = auth.uid());
