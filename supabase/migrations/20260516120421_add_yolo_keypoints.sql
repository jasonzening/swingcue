-- ============================================================================
-- PR-3A: pose_3d_phases — add YOLO11x-pose keypoint columns
-- ============================================================================
-- SAM 3D Body's 70 MHR keypoints are mesh anchors (kp 7,8 land on chest, not
-- acromion). YOLO11m-pose outputs COCO 17 keypoints — anatomical surface
-- landmarks (kp 5,6 = real acromion, kp 11,12 = real hip joint).
--
-- Additive only:
--   - All three columns are nullable; no DEFAULT.
--   - Existing rows stay unchanged.
--   - PR-2 SAM columns / RLS policies / triggers are NOT touched.
-- ============================================================================

ALTER TABLE public.pose_3d_phases
  ADD COLUMN IF NOT EXISTS yolo_keypoints_2d JSONB,
  ADD COLUMN IF NOT EXISTS yolo_model        TEXT,
  ADD COLUMN IF NOT EXISTS yolo_inference_ms INTEGER;

COMMENT ON COLUMN public.pose_3d_phases.yolo_keypoints_2d IS
  '17 × [x, y, confidence] from YOLO11x-pose; COCO order (see python/yolo/keypoints.py). NULL if YOLO did not run or failed.';
COMMENT ON COLUMN public.pose_3d_phases.yolo_model IS
  'Model identifier, e.g. "yolo11m-pose". NULL if YOLO did not run.';
COMMENT ON COLUMN public.pose_3d_phases.yolo_inference_ms IS
  'Inference wall-clock ms (model only, excluding decode/upload).';
