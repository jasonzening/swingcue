/**
 * overlayLineSpec.ts
 *
 * 严格定义哪些线由哪些点连接。
 * 不允许随意连线。
 */

import type { BodyPointName } from './bodyPointSpec';
import type { MainIssueType } from '@/types/analysis';

export type ViewType = 'face_on' | 'down_the_line';

/* ══════════════════════════════════════
   结构线定义
══════════════════════════════════════ */
export type StructureLineName =
  | 'shoulderLine'       // 肩线：左肩 ↔ 右肩
  | 'hipLine'            // 髋线：左髋 ↔ 右髋
  | 'spineAxis'          // 脊柱中轴：双肩中点 ↔ 双髋中点
  | 'leftArmChain'       // 左臂链：左肩→左肘→左腕
  | 'rightArmChain'      // 右臂链：右肩→右肘→右腕
  | 'armTriangle'        // 双臂三角：左肩/右肩/握把中心
  | 'leftLegChain'       // 左腿：左髋→左膝→左踝
  | 'rightLegChain'      // 右腿：右髋→右膝→右踝

export interface StructureLineDef {
  name: StructureLineName;
  points: BodyPointName[];     // 按顺序连接
  requiresAllPoints: boolean;  // true=所有点都必须存在才画线
  golfPurpose: string[];
}

export const STRUCTURE_LINE_SPEC: Record<StructureLineName, StructureLineDef> = {
  shoulderLine: {
    name: 'shoulderLine',
    points: ['leftShoulder', 'rightShoulder'],
    requiresAllPoints: true,
    golfPurpose: ['肩线', 'shoulder_tilt', 'shoulder_lifts_too_early', 'upper_body_turn'],
  },
  hipLine: {
    name: 'hipLine',
    points: ['leftHip', 'rightHip'],
    requiresAllPoints: true,
    golfPurpose: ['髋线', 'hip_turn', 'lower_body_structure', 'early_extension', 'weight_shift'],
  },
  spineAxis: {
    name: 'spineAxis',
    points: ['shoulderCenter', 'hipCenter'],
    requiresAllPoints: true,
    golfPurpose: ['脊柱中轴', 'posture', '身体是否起身', 'spine_angle_stability'],
  },
  leftArmChain: {
    name: 'leftArmChain',
    points: ['leftShoulder', 'leftElbow', 'leftWrist'],
    requiresAllPoints: true,
    golfPurpose: ['左臂结构', 'arm_structure', 'hand_path'],
  },
  rightArmChain: {
    name: 'rightArmChain',
    points: ['rightShoulder', 'rightElbow', 'rightWrist'],
    requiresAllPoints: true,
    golfPurpose: ['右臂结构', 'chicken_wing', 'trail_arm_structure'],
  },
  armTriangle: {
    name: 'armTriangle',
    points: ['leftShoulder', 'rightShoulder', 'gripCenter'],
    requiresAllPoints: true,
    golfPurpose: ['双臂整体三角', 'chicken_wing判断', '手臂结构是否塌陷'],
  },
  leftLegChain: {
    name: 'leftLegChain',
    points: ['leftHip', 'leftKnee', 'leftAnkle'],
    requiresAllPoints: false,
    golfPurpose: ['下盘稳定性', '重心参考'],
  },
  rightLegChain: {
    name: 'rightLegChain',
    points: ['rightHip', 'rightKnee', 'rightAnkle'],
    requiresAllPoints: false,
    golfPurpose: ['下盘稳定性', '重心参考'],
  },
};

/* ══════════════════════════════════════
   视角×问题×线的映射
   指定哪个视角下哪个问题显示哪些线
══════════════════════════════════════ */

export interface IssueLineSpec {
  redLines: StructureLineName[];    // 红色线（当前结构）
  auxiliaryLines: StructureLineName[];  // 辅助线（白色/淡色）
  greenLines?: StructureLineName[]; // 绿色线（目标结构，基于红线点+delta）
}

/** face_on 视角下各问题的线规范 */
const FACE_ON_LINE_SPEC: Partial<Record<MainIssueType, IssueLineSpec>> = {
  shoulder_lifts_too_early: {
    redLines: ['shoulderLine'],
    auxiliaryLines: ['spineAxis'],
    greenLines: ['shoulderLine'],  // 绿色肩线在红色肩线下方
  },
  posture_rises_too_early: {
    redLines: ['spineAxis'],
    auxiliaryLines: ['shoulderLine', 'hipLine'],
    greenLines: ['spineAxis'],
  },
  early_extension: {
    redLines: ['hipLine', 'spineAxis'],
    auxiliaryLines: ['shoulderLine'],
    greenLines: ['hipLine', 'spineAxis'],
  },
  head_movement: {
    redLines: [],
    auxiliaryLines: ['spineAxis'],
    greenLines: [],
  },
  not_enough_hip_turn: {
    redLines: ['hipLine'],
    auxiliaryLines: ['shoulderLine'],
    greenLines: ['hipLine'],
  },
  hand_path_issue: {
    redLines: ['leftArmChain'],
    auxiliaryLines: ['shoulderLine', 'armTriangle'],
    greenLines: [],
  },
  steep_downswing: {
    redLines: ['rightArmChain', 'shoulderLine'],
    auxiliaryLines: ['spineAxis'],
    greenLines: [],
  },
  weight_shift_issue: {
    redLines: ['hipLine'],
    auxiliaryLines: ['leftLegChain', 'rightLegChain'],
    greenLines: ['hipLine'],
  },
  steep_backswing_plane: {
    redLines: ['rightArmChain'],
    auxiliaryLines: ['shoulderLine'],
    greenLines: [],
  },
};

/** down_the_line 视角下各问题的线规范 */
const DTL_LINE_SPEC: Partial<Record<MainIssueType, IssueLineSpec>> = {
  posture_rises_too_early: {
    redLines: ['spineAxis'],
    auxiliaryLines: ['hipLine'],
    greenLines: ['spineAxis'],
  },
  early_extension: {
    redLines: ['spineAxis', 'hipLine'],
    auxiliaryLines: [],
    greenLines: ['spineAxis', 'hipLine'],
  },
  head_movement: {
    redLines: [],
    auxiliaryLines: ['spineAxis'],
    greenLines: [],
  },
  not_enough_hip_turn: {
    redLines: ['hipLine'],
    auxiliaryLines: ['shoulderLine'],
    greenLines: ['hipLine'],
  },
  hand_path_issue: {
    redLines: ['leftArmChain', 'rightArmChain'],
    auxiliaryLines: [],
    greenLines: ['leftArmChain'],
  },
  steep_downswing: {
    redLines: ['rightArmChain'],
    auxiliaryLines: ['shoulderLine', 'spineAxis'],
    greenLines: ['rightArmChain'],
  },
  weight_shift_issue: {
    redLines: ['hipLine', 'rightLegChain'],
    auxiliaryLines: ['leftLegChain'],
    greenLines: ['hipLine'],
  },
  steep_backswing_plane: {
    redLines: ['rightArmChain', 'leftArmChain'],
    auxiliaryLines: ['shoulderLine'],
    greenLines: ['rightArmChain'],
  },
};

export function getLineSpec(
  issue: MainIssueType,
  viewType: ViewType,
): IssueLineSpec {
  const spec = viewType === 'face_on'
    ? FACE_ON_LINE_SPEC[issue]
    : DTL_LINE_SPEC[issue];

  return spec ?? {
    redLines: ['shoulderLine', 'hipLine', 'spineAxis'],
    auxiliaryLines: [],
    greenLines: [],
  };
}
