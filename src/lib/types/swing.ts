/**
 * swing.ts
 * TypeScript types and enums for SwingCue MVP.
 *
 * Source of truth: docs/specs/MVP-006-supabase-schema.md
 * Contract:        docs/contracts/analysis_stub_result_contract.json
 * Scope:           PR-1A — types only, no runtime logic
 */

// ---------------------------------------------------------------------------
// Enums — match SQL enum values exactly
// ---------------------------------------------------------------------------

export const SWING_VIEW_TYPES = ['face_on', 'down_the_line'] as const;
export type SwingViewType = (typeof SWING_VIEW_TYPES)[number];

export const SWING_STATUSES = ['uploaded', 'processing', 'completed', 'failed'] as const;
export type SwingStatus = (typeof SWING_STATUSES)[number];

export const SWING_SEVERITIES = ['low', 'medium', 'high'] as const;
export type SwingSeverity = (typeof SWING_SEVERITIES)[number];

export const SWING_ISSUE_TYPES = [
  'weight_shift_issue',
  'head_movement',
  'early_extension',
  'steep_backswing_plane',
  'steep_downswing',
  'hand_path_issue',
] as const;
export type SwingIssueType = (typeof SWING_ISSUE_TYPES)[number];

export const SWING_PHASE_NAMES = ['address', 'top', 'downswing', 'impact', 'finish'] as const;
export type SwingPhaseName = (typeof SWING_PHASE_NAMES)[number];

// ---------------------------------------------------------------------------
// Database row types — direct representations of table rows
// ---------------------------------------------------------------------------

export interface SwingVideoRow {
  id: string;
  user_id: string;
  storage_path: string;
  original_filename: string;
  file_size_bytes: number | null;
  duration_ms: number | null;
  view_type: SwingViewType;
  status: SwingStatus;
  error_code: string | null;
  error_message: string | null;
  processing_started_at: string | null;
  processing_completed_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface SwingAnalysisRow {
  id: string;
  video_id: string;
  issue_type: SwingIssueType;
  summary_text: string;
  cue_text: string;
  drill_text: string;
  overlay_asset_url: string | null;
  score: number | null;
  severity: SwingSeverity | null;
  created_at: string;
  updated_at: string;
}

export interface SwingPhaseSnapshotRow {
  id: string;
  video_id: string;
  phase_name: SwingPhaseName;
  snapshot_asset_url: string;
  frame_index: number;
  created_at: string;
}

// ---------------------------------------------------------------------------
// Analysis stub result contract
// Source: docs/contracts/analysis_stub_result_contract.json
// ---------------------------------------------------------------------------

export interface PhaseSnapshotPayload {
  phase_name: SwingPhaseName;
  snapshot_asset_url: string;
  frame_index: number;
}

/**
 * Completed result — requires all One-One-One fields.
 */
export interface AnalysisStubResultCompleted {
  video_id: string;
  status: 'completed';
  issue_type: SwingIssueType;
  main_issue_text: string;
  fix_cue_text: string;
  drill_text: string;
  phase_snapshots: PhaseSnapshotPayload[];
  overlay_asset_url: string | null;
  score: number | null;
  severity: SwingSeverity | null;
}

/**
 * Failed result — requires explicit error fields.
 */
export interface AnalysisStubResultFailed {
  video_id: string;
  status: 'failed';
  error_code: string;
  error_message: string;
}

export type AnalysisStubResult = AnalysisStubResultCompleted | AnalysisStubResultFailed;

// ---------------------------------------------------------------------------
// Insert payloads (used by persistence layer in PR-1B)
// ---------------------------------------------------------------------------

export type SwingVideoInsert = Omit<SwingVideoRow, 'id' | 'created_at' | 'updated_at'>;

export type SwingAnalysisInsert = Omit<SwingAnalysisRow, 'id' | 'created_at' | 'updated_at'>;

export type SwingPhaseSnapshotInsert = Omit<SwingPhaseSnapshotRow, 'id' | 'created_at'>;

// ---------------------------------------------------------------------------
// Status transition helpers — explicit only, no inference
// ---------------------------------------------------------------------------

export function isTerminalStatus(status: SwingStatus): boolean {
  return status === 'completed' || status === 'failed';
}

export function isViewTypeSupported(value: string): value is SwingViewType {
  return SWING_VIEW_TYPES.includes(value as SwingViewType);
}

export function isIssueTypeValid(value: string): value is SwingIssueType {
  return SWING_ISSUE_TYPES.includes(value as SwingIssueType);
}

export function isPhaseNameValid(value: string): value is SwingPhaseName {
  return SWING_PHASE_NAMES.includes(value as SwingPhaseName);
}
