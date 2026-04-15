/**
 * clubSpec.ts — 球杆显示规范
 *
 * 没有可信数据时宁可不画，也不要乱画。
 */

import type { MainIssueType } from '@/types/analysis';
import type { ViewType } from './overlayLineSpec';

export type SwingPhase = 'setup' | 'top' | 'transition' | 'impact' | 'finish';
export type ClubConfidence = 'high' | 'medium' | 'low' | 'none';

export interface ClubDisplayRule {
  showClub: boolean;
  showGrip: boolean;
  showShaft: boolean;
  showClubHead: boolean;
  minConfidence: ClubConfidence;
}

/** 哪些问题/视角/阶段允许显示球杆 */
export const CLUB_DISPLAY_RULES: {
  issues: MainIssueType[];
  viewTypes: ViewType[];
  phases: SwingPhase[];
  rule: ClubDisplayRule;
}[] = [
  {
    // steep_downswing / steep_backswing_plane — down-the-line 视角最有价值
    issues: ['steep_downswing', 'steep_backswing_plane'],
    viewTypes: ['down_the_line'],
    phases: ['top', 'transition', 'impact'],
    rule: { showClub: true, showGrip: true, showShaft: true, showClubHead: true, minConfidence: 'medium' },
  },
  {
    // 相同问题在 face-on 只做辅助
    issues: ['steep_downswing', 'steep_backswing_plane'],
    viewTypes: ['face_on'],
    phases: ['transition', 'impact'],
    rule: { showClub: true, showGrip: true, showShaft: true, showClubHead: false, minConfidence: 'high' },
  },
  {
    // hand_path_issue — 球杆只做辅助
    issues: ['hand_path_issue'],
    viewTypes: ['down_the_line'],
    phases: ['top', 'transition'],
    rule: { showClub: true, showGrip: true, showShaft: false, showClubHead: false, minConfidence: 'high' },
  },
];

/** 置信度 fallback 规则 */
export function getClubDisplayByConfidence(confidence: ClubConfidence): ClubDisplayRule {
  switch (confidence) {
    case 'high':
      return { showClub: true, showGrip: true, showShaft: true, showClubHead: true, minConfidence: 'high' };
    case 'medium':
      return { showClub: true, showGrip: true, showShaft: true, showClubHead: false, minConfidence: 'medium' };
    case 'low':
    case 'none':
    default:
      return { showClub: false, showGrip: false, showShaft: false, showClubHead: false, minConfidence: 'none' };
  }
}

export function getClubDisplaySpec(params: {
  issue: MainIssueType;
  viewType: ViewType;
  phase: SwingPhase;
  confidence: ClubConfidence;
}): ClubDisplayRule {
  const { issue, viewType, phase, confidence } = params;

  // 低置信度直接隐藏
  if (confidence === 'low' || confidence === 'none') {
    return { showClub: false, showGrip: false, showShaft: false, showClubHead: false, minConfidence: 'none' };
  }

  // 查找适用规则
  const rule = CLUB_DISPLAY_RULES.find(r =>
    r.issues.includes(issue) &&
    r.viewTypes.includes(viewType) &&
    r.phases.includes(phase)
  );

  if (!rule) {
    // 无规则 = 不显示
    return { showClub: false, showGrip: false, showShaft: false, showClubHead: false, minConfidence: 'none' };
  }

  // 按置信度降级
  const actual = getClubDisplayByConfidence(confidence);
  return {
    showClub: rule.rule.showClub && actual.showClub,
    showGrip: rule.rule.showGrip && actual.showGrip,
    showShaft: rule.rule.showShaft && actual.showShaft,
    showClubHead: rule.rule.showClubHead && actual.showClubHead,
    minConfidence: rule.rule.minConfidence,
  };
}

export interface ClubEstimate {
  grip?: { x: number; y: number };
  shaftMid?: { x: number; y: number };
  clubHead?: { x: number; y: number };
  confidence: ClubConfidence;
}

/**
 * 从手腕/前臂方向估算球杆位置（用于无真实跟踪时的近似）
 * 仅在 confidence='medium' 时使用
 */
export function estimateClubFromHands(
  rightWrist: { x: number; y: number },
  rightElbow: { x: number; y: number },
  videoWidth: number,
): ClubEstimate {
  const fvX = rightWrist.x - rightElbow.x;
  const fvY = rightWrist.y - rightElbow.y;
  const fLen = Math.hypot(fvX, fvY);
  if (fLen < 0.01) return { grip: rightWrist, confidence: 'low' };

  const nx = fvX / fLen, ny = fvY / fLen;
  const clubLen = 0.28; // 归一化约28%视频宽度

  return {
    grip: rightWrist,
    shaftMid: { x: rightWrist.x + nx * clubLen * 0.5, y: rightWrist.y + ny * clubLen * 0.5 },
    clubHead: { x: rightWrist.x + nx * clubLen, y: rightWrist.y + ny * clubLen },
    confidence: 'medium',
  };
}
