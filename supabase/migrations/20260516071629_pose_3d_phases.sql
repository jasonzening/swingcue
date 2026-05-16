-- ============================================================================
-- PR-2A: pose_3d_phases — SAM 3D Body keypoints per phase
-- ============================================================================
-- Strategy:
--   One row per (video_id, phase_name). 5 phases per video:
--     setup, top, transition, impact, finish
--   fal-ai/sam-3/3d-body returns 70 keypoints per frame; we store the full
--   2D + 3D arrays as JSONB (for future re-analysis without re-calling fal),
--   AND denormalize shoulder/hip 2D into typed columns for fast disc rendering.
--
-- Additive only (PR-1A schema untouched).
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.pose_3d_phases (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id        UUID NOT NULL
                  REFERENCES public.swing_videos(id) ON DELETE CASCADE,
  user_id         UUID NOT NULL,
  phase_name      TEXT NOT NULL,
  frame_idx       INTEGER NOT NULL,
  frame_timestamp_ms INTEGER,

  keypoints_2d    JSONB NOT NULL,
  keypoints_3d    JSONB NOT NULL,
  focal_length    REAL  NOT NULL,
  bbox            JSONB,
  mhr_params      JSONB,
  glb_url         TEXT,

  image_width     INTEGER NOT NULL,
  image_height    INTEGER NOT NULL,

  shoulder_left_x   REAL,
  shoulder_left_y   REAL,
  shoulder_right_x  REAL,
  shoulder_right_y  REAL,
  hip_left_x        REAL,
  hip_left_y        REAL,
  hip_right_x       REAL,
  hip_right_y       REAL,

  fal_status        TEXT NOT NULL DEFAULT 'completed',
  fal_request_id    TEXT,
  error_message     TEXT,

  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  CONSTRAINT pose_3d_phases_phase_check
    CHECK (phase_name IN ('setup', 'top', 'transition', 'impact', 'finish')),

  CONSTRAINT pose_3d_phases_fal_status_check
    CHECK (fal_status IN ('uploaded', 'processing', 'completed', 'failed')),

  CONSTRAINT pose_3d_phases_video_phase_unique
    UNIQUE (video_id, phase_name)
);

CREATE INDEX IF NOT EXISTS idx_pose_3d_phases_video
  ON public.pose_3d_phases (video_id);

CREATE INDEX IF NOT EXISTS idx_pose_3d_phases_user
  ON public.pose_3d_phases (user_id);

CREATE INDEX IF NOT EXISTS idx_pose_3d_phases_video_phase
  ON public.pose_3d_phases (video_id, phase_name);

CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_pose_3d_phases_updated_at ON public.pose_3d_phases;
CREATE TRIGGER trg_pose_3d_phases_updated_at
  BEFORE UPDATE ON public.pose_3d_phases
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

ALTER TABLE public.pose_3d_phases ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS pose_3d_phases_user_select ON public.pose_3d_phases;
CREATE POLICY pose_3d_phases_user_select
  ON public.pose_3d_phases
  FOR SELECT
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS pose_3d_phases_service_all ON public.pose_3d_phases;
CREATE POLICY pose_3d_phases_service_all
  ON public.pose_3d_phases
  FOR ALL
  USING (auth.jwt() ->> 'role' = 'service_role')
  WITH CHECK (auth.jwt() ->> 'role' = 'service_role');
