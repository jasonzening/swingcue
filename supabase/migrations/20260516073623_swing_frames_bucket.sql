-- ============================================================================
-- PR-2B: swing-frames storage bucket
-- ============================================================================
-- Phase frames extracted from videos are uploaded here, then their public URL
-- is passed to fal-ai/sam-3/3d-body. Public read is required because fal
-- fetches the image by URL; writes are service-role only (Railway analyzer).
-- ============================================================================

-- Create swing-frames bucket (idempotent via DO block)
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES ('swing-frames', 'swing-frames', true, 5242880, ARRAY['image/png','image/jpeg'])
ON CONFLICT (id) DO NOTHING;

-- RLS: anyone can read (we use public URLs for fal), service-role only writes
CREATE POLICY "Public read swing-frames"
  ON storage.objects FOR SELECT
  USING (bucket_id = 'swing-frames');

CREATE POLICY "Service role write swing-frames"
  ON storage.objects FOR INSERT
  WITH CHECK (
    bucket_id = 'swing-frames'
    AND auth.jwt() ->> 'role' = 'service_role'
  );
