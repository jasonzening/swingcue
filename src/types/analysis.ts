/**
 * analysis.ts — SwingCue 统一数据契约
 *
 * 所有前后端交互、组件 props、overlay 渲染，
 * 都以这里的类型为准。
 *
 * 设计原则：
 * - 前端只消费数据，不生成分析逻辑
 * - 坐标系统：归一化 0-1（相对视频宽高）
 * - 时间单位：秒（number）
 */

/* ═══════════════════════════════════════
   顶层分析结果
═══════════════════════════════════════ */

export interface SwingAnalysisPayload {
  analysisId: string;
  videoId: string;
  videoUrl: string;
  videoMetadata: VideoMetadata;
  phaseMarkers: PhaseMarkers;
  mainIssue: MainIssueType;
  cue: string;               // 一句话 cue，不要太长
  drill: string;             // 一句话 drill
  severity: 'low' | 'medium' | 'high';
  score: number;             // 0-100
  overlayTimeline: OverlayTimeline;
  keypointTimeline?: KeypointTimeline;  // MediaPipe 数据接入后填充
}

/* ═══════════════════════════════════════
   视频元数据
═══════════════════════════════════════ */

export interface VideoMetadata {
  durationSec: number;
  fps: number;
  width: number;
  height: number;
}

/* ═══════════════════════════════════════
   阶段时间节点（单位：秒）
═══════════════════════════════════════ */

export interface PhaseMarkers {
  setupTime: number;
  topTime: number;
  transitionTime: number;
  impactTime: number;
  finishTime: number;
}

/* ═══════════════════════════════════════
   问题类型枚举
═══════════════════════════════════════ */

export type MainIssueType =
  | 'steep_downswing'
  | 'hand_path_issue'
  | 'early_extension'
  | 'weight_shift_issue'
  | 'head_movement'
  | 'steep_backswing_plane'
  | 'posture_rises_too_early'
  | 'shoulder_lifts_too_early'
  | 'not_enough_hip_turn';

export const ISSUE_LABELS: Record<MainIssueType, string> = {
  steep_downswing:         'Steep Downswing',
  hand_path_issue:         'Hand Path Too Outside',
  early_extension:         'Early Extension',
  weight_shift_issue:      'Reverse Pivot',
  head_movement:           'Head Movement',
  steep_backswing_plane:   'Steep Backswing Plane',
  posture_rises_too_early: 'Posture Rises Too Early',
  shoulder_lifts_too_early:'Shoulder Lifts Too Early',
  not_enough_hip_turn:     'Not Enough Hip Turn',
};

/* ═══════════════════════════════════════
   Overlay 时间线
═══════════════════════════════════════ */

export interface OverlayTimeline {
  frames: OverlayFrame[];
}

export interface OverlayFrame {
  time: number;                                                    // 秒
  phase?: 'setup' | 'top' | 'transition' | 'impact' | 'finish';  // 可选
  elements: OverlayElement[];
}

/* ─── element 联合类型 ─── */

export type OverlayElement =
  | LineElement
  | CurveElement
  | ArrowElement
  | DotElement
  | LabelElement
  | BadgeElement
  | ZoneElement;

/* ─── 基础公共字段 ─── */

export interface BaseElement {
  id: string;
  color?: 'red' | 'green' | 'yellow' | 'white';
  opacity?: number;       // 0-1, default 0.9
  strokeWidth?: number;   // 归一化像素，default 3
  layer?: 'body' | 'arms' | 'club' | 'all';  // 用于层过滤
}

/* ─── 线段（肩线、髋线、脊柱线、杆身方向）─── */
export interface LineElement extends BaseElement {
  type: 'line';
  x1: number; y1: number;  // 归一化 0-1
  x2: number; y2: number;
  dashed?: boolean;
}

/* ─── 曲线路径（手路径、杆头路径、上下杆路径）─── */
export interface CurveElement extends BaseElement {
  type: 'curve';
  points: Array<{ x: number; y: number }>;
}

/* ─── 箭头（方向指示：该高/低/里/外）─── */
export interface ArrowElement extends BaseElement {
  type: 'arrow';
  from: { x: number; y: number };
  to: { x: number; y: number };
}

/* ─── 关节圆点（关键点位置）─── */
export interface DotElement extends BaseElement {
  type: 'dot';
  x: number; y: number;
  radius?: number;  // 归一化，default ~0.025
}

/* ─── 文字标签（尽量少用）─── */
export interface LabelElement extends BaseElement {
  type: 'label';
  x: number; y: number;
  text: string;
  size?: number;  // font size 归一化
}

/* ─── 对/错徽章 ─── */
export interface BadgeElement extends BaseElement {
  type: 'badge';
  x: number; y: number;
  variant: 'correct' | 'wrong';
}

/* ─── 区域高亮 ─── */
export interface ZoneElement extends BaseElement {
  type: 'zone';
  points: Array<{ x: number; y: number }>;
  fillOpacity?: number;
}

/* ═══════════════════════════════════════
   关键点时间线（MediaPipe 数据接口预留）
═══════════════════════════════════════ */

export interface KeypointTimeline {
  frames: KeypointFrame[];
}

export interface KeypointFrame {
  time: number;
  landmarks: BodyLandmarks;
  hands?: HandLandmarks;
  club?: ClubEstimate;
}

export type Point2D = {
  x: number;   // 归一化 0-1
  y: number;
  confidence?: number;  // 0-1
};

export interface BodyLandmarks {
  head?: Point2D;
  neck?: Point2D;
  leftShoulder?: Point2D;
  rightShoulder?: Point2D;
  leftElbow?: Point2D;
  rightElbow?: Point2D;
  leftWrist?: Point2D;
  rightWrist?: Point2D;
  leftHip?: Point2D;
  rightHip?: Point2D;
  leftKnee?: Point2D;
  rightKnee?: Point2D;
  leftAnkle?: Point2D;
  rightAnkle?: Point2D;
}

export interface HandLandmarks {
  leftHand?: Point2D[];
  rightHand?: Point2D[];
}

export interface ClubEstimate {
  grip?: Point2D;
  shaftMid?: Point2D;
  clubHead?: Point2D;
}

/* ═══════════════════════════════════════
   视觉模板类型
═══════════════════════════════════════ */

export type VisualTemplateType =
  | 'posture_structure_template'
  | 'swing_path_template'
  | 'arm_structure_template'
  | 'weight_shift_template'
  | 'setup_structure_template'
  | 'local_focus_template';

export interface TemplateInput {
  videoMetadata: VideoMetadata;
  phaseMarkers: PhaseMarkers;
  issue: MainIssueType;
  viewType?: 'face_on' | 'down_the_line';
  keypointTimeline?: KeypointTimeline;
}

/* ══════════════════════════════════════
   PR-4: pose_timeline_2d — 17-COCO frame-level timeline.
   Stored on swing_videos.pose_timeline_2d (JSONB). See
   docs/decisions/PR-4_DESIGN.md for the full design.

   This is the canonical data substrate for ALL future overlay PRs
   (PR-5 rotation, PR-6 sway, PR-7 view detect, etc.). Coordinate
   convention: video native pixels, NOT normalised 0-1.
══════════════════════════════════════ */

export type CocoKeypointName =
  | 'nose'
  | 'left_eye' | 'right_eye'
  | 'left_ear' | 'right_ear'
  | 'left_shoulder' | 'right_shoulder'
  | 'left_elbow'    | 'right_elbow'
  | 'left_wrist'    | 'right_wrist'
  | 'left_hip'      | 'right_hip'
  | 'left_knee'     | 'right_knee'
  | 'left_ankle'    | 'right_ankle';

/**
 * One keypoint: [x_px, y_px, confidence].
 * x and y are null when MediaPipe visibility < 0.3 or the keypoint was
 * outlier-rejected. Confidence is preserved in the third slot for
 * diagnostics.
 */
export type Keypoint = readonly [number | null, number | null, number];

export interface PoseFrame {
  ts: number;                                       // seconds from video start
  frame_idx: number;                                // original video frame index
  interpolated: boolean;                            // true = gap-filled, false = direct
  keypoints: Record<CocoKeypointName, Keypoint>;
  // PR-5.9 Task 4: pre-smoothing per-frame MediaPipe extract. Optional
  // (legacy v1 timelines do not have this; v1.5+ always does). Same
  // schema as `keypoints` plus the v1.5 head_crown entry.
  raw_keypoints?: Record<CocoKeypointName, Keypoint>;
}

export interface PoseTimeline {
  version: 1;
  fps_sampled: number;
  video_width: number;
  video_height: number;
  // PR-6.1a adds 'rtmpose_v1' alongside the MediaPipe family. The
  // frontend SkeletonOverlay reads 17-COCO names regardless of source,
  // so this union expansion is purely a TypeScript narrowing hint —
  // runtime rendering is unchanged for the rtmpose path in 6.1a. The
  // v1.5+ raw_keypoints field is also present on rtmpose envelopes
  // (main.py writes deepcopy snapshot for both extractor paths).
  keypoint_source:
    | 'mediapipe_pose'
    | 'mediapipe_pose_v1_5'
    | 'rtmpose_v1'
    | 'yolo'
    | 'hybrid';
  yolo_anchor_correction: {
    applied: boolean;
    anchor_phases?: string[];
    method?: string;
  };
  frames: PoseFrame[];
}
