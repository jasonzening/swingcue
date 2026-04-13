export const ViewType = {
  FACE_ON: 'face_on',
  DOWN_THE_LINE: 'down_the_line',
} as const;
export type ViewType = typeof ViewType[keyof typeof ViewType];

export const VideoStatus = {
  UPLOADED: 'uploaded',
  PROCESSING: 'processing',
  COMPLETED: 'completed',
  FAILED: 'failed',
} as const;
export type VideoStatus = typeof VideoStatus[keyof typeof VideoStatus];

export const IssueType = {
  WEIGHT_SHIFT_ISSUE: 'weight_shift_issue',
  HEAD_MOVEMENT: 'head_movement',
  EARLY_EXTENSION: 'early_extension',
  STEEP_BACKSWING_PLANE: 'steep_backswing_plane',
  STEEP_DOWNSWING: 'steep_downswing',
  HAND_PATH_ISSUE: 'hand_path_issue',
} as const;
export type IssueType = typeof IssueType[keyof typeof IssueType];

export const PhaseName = {
  ADDRESS: 'address',
  TOP: 'top',
  DOWNSWING: 'downswing',
  IMPACT: 'impact',
  FINISH: 'finish',
} as const;
export type PhaseName = typeof PhaseName[keyof typeof PhaseName];
