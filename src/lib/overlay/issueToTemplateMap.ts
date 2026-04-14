/**
 * issueToTemplateMap.ts
 * 每个 issue 类型对应的视觉模板类型
 */
import type { MainIssueType, VisualTemplateType } from '@/types/analysis';

export const issueToTemplateMap: Record<MainIssueType, VisualTemplateType> = {
  steep_downswing:         'swing_path_template',
  hand_path_issue:         'arm_structure_template',
  early_extension:         'posture_structure_template',
  weight_shift_issue:      'weight_shift_template',
  head_movement:           'local_focus_template',
  steep_backswing_plane:   'swing_path_template',
  posture_rises_too_early: 'posture_structure_template',
  shoulder_lifts_too_early:'posture_structure_template',
  not_enough_hip_turn:     'weight_shift_template',
};
