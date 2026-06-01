-- PR-7A.2: full-body workbench expansion — head + leg joint clusters.
-- Additive: new columns nullable, existing rows untouched. Idempotent
-- via ADD COLUMN IF NOT EXISTS + CREATE UNIQUE INDEX IF NOT EXISTS so
-- it's safe to re-run on any DB. The CHECK-constraint drops are NOT
-- guarded with IF EXISTS — the prior migration created them
-- unconditionally, so any DB this file targets MUST have them.
--
-- Already applied to prod via MCP on 2026-05-31 (version
-- 20260601040738_pr7a2_full_body_calibration_schema); this file mirrors
-- that state verbatim for fresh-local-dev / CI parity.

-- 1. Add columns for new joint clusters
ALTER TABLE golf_landmark_annotations
  ADD COLUMN IF NOT EXISTS head_crown_x int,
  ADD COLUMN IF NOT EXISTS head_crown_y int,
  ADD COLUMN IF NOT EXISTS chin_x int,
  ADD COLUMN IF NOT EXISTS chin_y int,
  ADD COLUMN IF NOT EXISTS knee_x int,
  ADD COLUMN IF NOT EXISTS knee_y int,
  ADD COLUMN IF NOT EXISTS ankle_x int,
  ADD COLUMN IF NOT EXISTS ankle_y int;

-- 2. Extend task_type CHECK to allow new clusters
ALTER TABLE golf_landmark_annotations
  DROP CONSTRAINT golf_landmark_annotations_task_type_check;
ALTER TABLE golf_landmark_annotations
  ADD CONSTRAINT golf_landmark_annotations_task_type_check
  CHECK (task_type = ANY (ARRAY[
    'manual_gt'::text,             -- arm task (legacy name kept for compat)
    'manual_gt_hip_pair'::text,
    'manual_gt_head_set'::text,    -- NEW: head_crown + chin
    'manual_gt_leg'::text,          -- NEW: knee + ankle, with arm = lead/trail
    'correction_review'::text,
    'active_learning'::text
  ]));

-- 3. Extend task_data_match CHECK to validate new clusters
ALTER TABLE golf_landmark_annotations
  DROP CONSTRAINT golf_landmark_annotations_task_data_match;
ALTER TABLE golf_landmark_annotations
  ADD CONSTRAINT golf_landmark_annotations_task_data_match
  CHECK (
    -- Arm tasks (legacy name 'manual_gt'): arm + shoulder/elbow/wrist; all other clusters NULL
    (
      (task_type = ANY (ARRAY['manual_gt'::text, 'correction_review'::text, 'active_learning'::text]))
      AND (arm IS NOT NULL)
      AND (lead_hip_x IS NULL)  AND (lead_hip_y IS NULL)
      AND (trail_hip_x IS NULL) AND (trail_hip_y IS NULL)
      AND (head_crown_x IS NULL) AND (head_crown_y IS NULL)
      AND (chin_x IS NULL)       AND (chin_y IS NULL)
      AND (knee_x IS NULL)       AND (knee_y IS NULL)
      AND (ankle_x IS NULL)      AND (ankle_y IS NULL)
    )
    OR
    -- Hip pair: lead_hip + trail_hip; all other clusters NULL
    (
      (task_type = 'manual_gt_hip_pair'::text)
      AND (arm IS NULL)
      AND (lead_hip_x IS NOT NULL)  AND (lead_hip_y IS NOT NULL)
      AND (trail_hip_x IS NOT NULL) AND (trail_hip_y IS NOT NULL)
      AND (shoulder_x IS NULL) AND (shoulder_y IS NULL)
      AND (elbow_x IS NULL)    AND (elbow_y IS NULL)
      AND (wrist_x IS NULL)    AND (wrist_y IS NULL)
      AND (head_crown_x IS NULL) AND (head_crown_y IS NULL)
      AND (chin_x IS NULL)       AND (chin_y IS NULL)
      AND (knee_x IS NULL)       AND (knee_y IS NULL)
      AND (ankle_x IS NULL)      AND (ankle_y IS NULL)
    )
    OR
    -- NEW: Head set: head_crown + chin; all other clusters NULL
    (
      (task_type = 'manual_gt_head_set'::text)
      AND (arm IS NULL)
      AND (head_crown_x IS NOT NULL) AND (head_crown_y IS NOT NULL)
      AND (chin_x IS NOT NULL)        AND (chin_y IS NOT NULL)
      AND (shoulder_x IS NULL) AND (shoulder_y IS NULL)
      AND (elbow_x IS NULL)    AND (elbow_y IS NULL)
      AND (wrist_x IS NULL)    AND (wrist_y IS NULL)
      AND (lead_hip_x IS NULL)  AND (lead_hip_y IS NULL)
      AND (trail_hip_x IS NULL) AND (trail_hip_y IS NULL)
      AND (knee_x IS NULL)  AND (knee_y IS NULL)
      AND (ankle_x IS NULL) AND (ankle_y IS NULL)
    )
    OR
    -- NEW: Leg: arm + knee + ankle; all other clusters NULL
    (
      (task_type = 'manual_gt_leg'::text)
      AND (arm IS NOT NULL)
      AND (knee_x IS NOT NULL)  AND (knee_y IS NOT NULL)
      AND (ankle_x IS NOT NULL) AND (ankle_y IS NOT NULL)
      AND (shoulder_x IS NULL) AND (shoulder_y IS NULL)
      AND (elbow_x IS NULL)    AND (elbow_y IS NULL)
      AND (wrist_x IS NULL)    AND (wrist_y IS NULL)
      AND (lead_hip_x IS NULL)  AND (lead_hip_y IS NULL)
      AND (trail_hip_x IS NULL) AND (trail_hip_y IS NULL)
      AND (head_crown_x IS NULL) AND (head_crown_y IS NULL)
      AND (chin_x IS NULL)       AND (chin_y IS NULL)
    )
  );

-- 4. Partial unique indexes to prevent duplicate annotations
CREATE UNIQUE INDEX IF NOT EXISTS uq_gla_head_set
  ON golf_landmark_annotations (video_id, frame_idx, annotator_id)
  WHERE task_type = 'manual_gt_head_set';

CREATE UNIQUE INDEX IF NOT EXISTS uq_gla_leg
  ON golf_landmark_annotations (video_id, frame_idx, annotator_id, arm)
  WHERE task_type = 'manual_gt_leg' AND arm IS NOT NULL;

COMMENT ON COLUMN golf_landmark_annotations.head_crown_x IS 'PR-7A.2: 颅顶正中点 (top of skull at cranial midline above ears)';
COMMENT ON COLUMN golf_landmark_annotations.chin_x       IS 'PR-7A.2: 下颌正中点 (midpoint of mandibular symphysis)';
COMMENT ON COLUMN golf_landmark_annotations.knee_x       IS 'PR-7A.2: 外侧股骨上髁顶 (lateral femoral epicondyle apex)';
COMMENT ON COLUMN golf_landmark_annotations.ankle_x      IS 'PR-7A.2: 外踝顶 (lateral malleolus apex)';
