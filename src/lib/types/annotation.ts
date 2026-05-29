export type AnnotationArm = 'lead' | 'trail';
export type AnnotationPhase =
  | 'setup' | 'takeaway' | 'top' | 'transition'
  | 'impact' | 'post_impact' | 'finish' | 'intermediate';
export type AnnotationTaskType = 'manual_gt' | 'correction_review' | 'active_learning';
export type AnnotationVisibility = 'clear' | 'occluded' | 'uncertain';
export type Handedness = 'right' | 'left';

export interface AnnotationRecord {
  id?: string;
  video_id: string;
  annotator_id?: string;
  frame_idx: number;
  phase: AnnotationPhase;
  task_type: AnnotationTaskType;
  arm: AnnotationArm;
  visibility: AnnotationVisibility;
  shoulder_x: number | null;
  shoulder_y: number | null;
  elbow_x: number | null;
  elbow_y: number | null;
  wrist_x: number | null;
  wrist_y: number | null;
  handedness: Handedness;
  source_app_version: string;
  annotated_at?: string;
}

export interface AnnotationTask {
  index: number;            // 0..13
  phase: 'setup' | 'takeaway' | 'top' | 'transition'
       | 'impact' | 'post_impact' | 'finish';
  arm: AnnotationArm;
  frameIdx: number;
}

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

export const APP_VERSION = 'swingcue-annotate-1.0';

// Single source of truth for runtime phase validation. The workbench
// generates 14 tasks from TASK_PHASES (7 × lead/trail). ANNOTATION_PHASES
// is the full set that's legal in the `phase` column on
// golf_landmark_annotations — it's a superset of TASK_PHASES because
// 'intermediate' rows are valid records (future active-learning task
// type) even though no v1 task generator emits them.
export const TASK_PHASES = [
  'setup', 'takeaway', 'top', 'transition',
  'impact', 'post_impact', 'finish',
] as const satisfies readonly AnnotationPhase[];

export const ANNOTATION_PHASES = [
  ...TASK_PHASES,
  'intermediate',
] as const satisfies readonly AnnotationPhase[];
