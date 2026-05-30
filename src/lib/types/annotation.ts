export type AnnotationArm = 'lead' | 'trail';

// 8 values — full DB-legal phase set; matches CHECK on
// public.golf_landmark_annotations.phase. The workbench-side TASK_PHASES
// is a smaller subset (5) — see below.
export type AnnotationPhase =
  | 'setup' | 'takeaway' | 'top' | 'transition'
  | 'impact' | 'post_impact' | 'finish' | 'intermediate';

// PR-7A.1: hip_pair joins the 3 v1 task types. arm-coord tasks remain
// 'manual_gt'; the new 2-click hip flow uses 'manual_gt_hip_pair'.
export type AnnotationTaskType =
  | 'manual_gt'
  | 'manual_gt_hip_pair'
  | 'correction_review'
  | 'active_learning';

export type AnnotationVisibility = 'clear' | 'occluded' | 'uncertain';
export type Handedness = 'right' | 'left';

/**
 * Umbrella row shape for golf_landmark_annotations — carries either an
 * arm-coord record (shoulder/elbow/wrist) OR a hip-pair record (lead +
 * trail femoral-head-center estimates) discriminated by task_type. The
 * DB-side STRICT CHECK constraint enforces which columns must be set vs
 * null for each task_type; TypeScript can't enforce that cross-field
 * invariant cleanly on a single interface, so we lean on the DB plus
 * the narrower HipPairAnnotation shape for new hip writes.
 *
 * PR-7A.1 deltas:
 *   - arm is now nullable (hip rows carry arm=NULL)
 *   - 4 hip coord columns added; OPTIONAL on the interface so existing
 *     arm-task body construction in the workbench stays valid without
 *     a per-call-site change (commit 3 introduces explicit hip writes).
 */
export interface AnnotationRecord {
  id?: string;
  video_id: string;
  annotator_id?: string;
  frame_idx: number;
  phase: AnnotationPhase;
  task_type: AnnotationTaskType;
  arm: AnnotationArm | null;
  visibility: AnnotationVisibility;
  shoulder_x: number | null;
  shoulder_y: number | null;
  elbow_x: number | null;
  elbow_y: number | null;
  wrist_x: number | null;
  wrist_y: number | null;
  lead_hip_x?: number | null;
  lead_hip_y?: number | null;
  trail_hip_x?: number | null;
  trail_hip_y?: number | null;
  handedness: Handedness;
  source_app_version: string;
  annotated_at?: string;
}

/**
 * Strict shape for new hip-pair writes. All four hip coords required,
 * arm explicitly null, source_app_version pinned to the v2 anatomical-
 * spec tag. The workbench builds payloads against this shape; the API
 * route discriminates on task_type and re-validates server-side.
 */
export interface HipPairAnnotation {
  video_id: string;
  frame_idx: number;
  phase: AnnotationPhase;
  task_type: 'manual_gt_hip_pair';
  lead_hip_x: number;
  lead_hip_y: number;
  trail_hip_x: number;
  trail_hip_y: number;
  arm: null;
  handedness: Handedness;
  visibility: AnnotationVisibility;
  source_app_version: 'swingcue-annotate-2.0-anatomical-spec';
}

/* ─────────────────────────────────────────────────────────────────────
   Workbench task shapes
   ─────────────────────────────────────────────────────────────────────
   AnnotationTask is the v1 single-shape interface kept for back-compat
   with the current workbench code. WorkbenchTask is the new
   discriminated union the 15-task flow in commit 3 will adopt.
   ───────────────────────────────────────────────────────────────────── */

export interface AnnotationTask {
  index: number;
  phase: AnnotationPhase;
  arm: AnnotationArm;
  frameIdx: number;
}

export interface ArmTask {
  kind: 'arm';
  index: number;
  phase: TaskPhase;
  arm: AnnotationArm;
  frameIdx: number;
}

export interface HipPairTask {
  kind: 'hip_pair';
  index: number;
  phase: TaskPhase;
  frameIdx: number;
}

export type WorkbenchTask = ArmTask | HipPairTask;

export interface VideoMetaForAnnotation {
  videoId: string;
  filename: string | null;
  width: number;
  height: number;
  fps: number;
  durationSec: number;
  frameCount: number;
  hasWhamData: boolean;
  phaseMarkers: {
    setupTime: number | null;
    topTime: number | null;
    impactTime: number | null;
    finishTime: number | null;
    transitionTime: number | null;
  };
  signedVideoUrl: string;
}

export interface VideoListEntry {
  id: string;
  original_filename: string | null;
  view_type: string;
  created_at: string;
  hasWham: boolean;
  whamMeta: { frame_count: number; processed_fps: number } | null;
  annotationCount: number;
}

/* ─────────────────────────────────────────────────────────────────────
   Constants — version + phase-set tuples
   ───────────────────────────────────────────────────────────────────── */

// PR-7A.1: bumped from 'swingcue-annotate-1.0' (v1 bone-bulge-era) to
// the bone-top-centerline tag. Stamped on every row saved by the v2
// workbench. v1 rows in the DB get the ':deprecated-v1' suffix via the
// (G) idempotent UPDATE in 20260530000001.
export const SOURCE_APP_VERSION = 'swingcue-annotate-2.0-anatomical-spec';

// Back-compat alias for older imports.
export const APP_VERSION = SOURCE_APP_VERSION;

/**
 * The 5 phases the workbench actually generates tasks for (PR-7A.1
 * reduction from v1's 7: post_impact + finish dropped to focus
 * annotation effort on the moments that matter most for WHAM
 * validation). The DB still accepts all 8 AnnotationPhase values —
 * see ANNOTATION_PHASES below — so future task types (e.g.
 * correction_review of arbitrary frames) can still emit other phases.
 */
export const TASK_PHASES = [
  'setup', 'takeaway', 'top', 'transition', 'impact',
] as const satisfies readonly AnnotationPhase[];

export type TaskPhase = typeof TASK_PHASES[number];

/**
 * Full DB-legal phase set. Used as the runtime phase whitelist in the
 * API route so it accepts ANY phase the schema allows, not just the
 * workbench-generated subset. Diverges from TASK_PHASES as of PR-7A.1.
 */
export const ANNOTATION_PHASES = [
  ...TASK_PHASES,
  'post_impact', 'finish', 'intermediate',
] as const satisfies readonly AnnotationPhase[];
