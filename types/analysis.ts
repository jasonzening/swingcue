import type { IssueType, PhaseName } from '@/lib/constants/enums';

export interface SwingAnalysis {
  id: string;
  video_id: string;
  issue_type: IssueType;
  summary_text: string | null;
  cue_text: string | null;
  drill_text: string | null;
  overlay_asset_url: string | null;
  score: number | null;
  severity: string | null;
  created_at: string;
  updated_at: string;
}

export interface SwingPhaseSnapshot {
  id: string;
  video_id: string;
  phase_name: PhaseName;
  snapshot_asset_url: string | null;
  frame_index: number | null;
  created_at: string;
}

export interface AnalysisResult {
  analysis: SwingAnalysis;
  snapshots: SwingPhaseSnapshot[];
}
