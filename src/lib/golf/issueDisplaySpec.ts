/**
 * issueDisplaySpec.ts
 *
 * 严格定义每个问题在每个视角下必须显示哪些点、哪些线、哪些路径、哪些箭头。
 * 不允许每个问题自由发挥。
 */

import type { BodyPointName } from './bodyPointSpec';
import type { StructureLineName } from './overlayLineSpec';
import type { MainIssueType } from '@/types/analysis';
import type { CorrectionDirection } from '@/lib/golf/types';

export type ViewType = 'face_on' | 'down_the_line';
export type SwingPhase = 'setup' | 'top' | 'transition' | 'impact' | 'finish';

/* ══════════════════════════════════════
   点的显示规范
══════════════════════════════════════ */
export interface PointDisplaySpec {
  point: BodyPointName;
  showRed: boolean;           // 显示红色当前点
  showGreen: boolean;         // 显示绿色目标点
  greenDirection?: CorrectionDirection;  // 绿点相对红点的方向
  greenMagnitude?: 'small' | 'medium' | 'large';
  showArrow: boolean;         // 是否画红→绿箭头
  priority: 'must' | 'important' | 'auxiliary';
  phases: SwingPhase[];       // 在哪些阶段显示
}

/* ══════════════════════════════════════
   路径显示规范
══════════════════════════════════════ */
export interface PathDisplaySpec {
  trackedPoint: BodyPointName;   // 追踪哪个点的历史轨迹
  showRedPath: boolean;          // 显示红色当前路径
  showGreenPath: boolean;        // 显示绿色目标路径
  phases: SwingPhase[];          // 在哪些阶段显示路径
}

/* ══════════════════════════════════════
   单个问题在单个视角下的完整显示规范
══════════════════════════════════════ */
export interface IssueDisplayConfig {
  issue: MainIssueType;
  viewType: ViewType;
  userMessage: string;          // 用户一眼看懂的说明
  mustShowPoints: PointDisplaySpec[];
  mustShowLines: StructureLineName[];
  auxiliaryLines: StructureLineName[];
  paths: PathDisplaySpec[];
  showImpactTriangle: boolean;  // impact阶段是否显示手臂三角结构
}

/* ══════════════════════════════════════
   所有问题×视角的显示规范
   按文档第六章严格定义
══════════════════════════════════════ */

const ALL_PHASES: SwingPhase[] = ['setup', 'top', 'transition', 'impact', 'finish'];
const KEY_PHASES: SwingPhase[] = ['top', 'transition', 'impact'];

export const ISSUE_DISPLAY_SPEC: IssueDisplayConfig[] = [

  // ══ 1. shoulder_lifts_too_early ══
  {
    issue: 'shoulder_lifts_too_early',
    viewType: 'face_on',
    userMessage: '这个肩膀应该更低一点。',
    mustShowPoints: [
      { point: 'leftShoulder',  showRed: true,  showGreen: false, showArrow: false, priority: 'important', phases: ALL_PHASES },
      { point: 'rightShoulder', showRed: true,  showGreen: true,  greenDirection: 'lower', greenMagnitude: 'medium', showArrow: true, priority: 'must', phases: KEY_PHASES },
    ],
    mustShowLines: ['shoulderLine'],
    auxiliaryLines: ['spineAxis'],
    paths: [],
    showImpactTriangle: false,
  },
  {
    issue: 'shoulder_lifts_too_early',
    viewType: 'down_the_line',
    userMessage: '右肩在下杆时抬得太早。',
    mustShowPoints: [
      { point: 'rightShoulder', showRed: true,  showGreen: true,  greenDirection: 'lower', greenMagnitude: 'medium', showArrow: true, priority: 'must', phases: KEY_PHASES },
      { point: 'rightElbow',    showRed: true,  showGreen: false, showArrow: false, priority: 'important', phases: KEY_PHASES },
    ],
    mustShowLines: ['rightArmChain', 'shoulderLine'],
    auxiliaryLines: ['spineAxis'],
    paths: [],
    showImpactTriangle: false,
  },

  // ══ 2. posture_rises_too_early ══
  {
    issue: 'posture_rises_too_early',
    viewType: 'face_on',
    userMessage: '身体起太早了，应该保持更低更稳定。',
    mustShowPoints: [
      { point: 'shoulderCenter', showRed: true, showGreen: true, greenDirection: 'lower', greenMagnitude: 'medium', showArrow: true, priority: 'must', phases: KEY_PHASES },
      { point: 'hipCenter',      showRed: true, showGreen: false, showArrow: false, priority: 'important', phases: KEY_PHASES },
      { point: 'headCenter',     showRed: true, showGreen: true, greenDirection: 'lower', greenMagnitude: 'small', showArrow: false, priority: 'important', phases: KEY_PHASES },
    ],
    mustShowLines: ['spineAxis'],
    auxiliaryLines: ['shoulderLine', 'hipLine'],
    paths: [
      { trackedPoint: 'headCenter', showRedPath: true, showGreenPath: false, phases: ALL_PHASES },
    ],
    showImpactTriangle: false,
  },
  {
    issue: 'posture_rises_too_early',
    viewType: 'down_the_line',
    userMessage: '击球前身体站起太早，要保持脊柱角度。',
    mustShowPoints: [
      { point: 'shoulderCenter', showRed: true, showGreen: true, greenDirection: 'lower', greenMagnitude: 'medium', showArrow: true, priority: 'must', phases: KEY_PHASES },
      { point: 'hipCenter',      showRed: true, showGreen: true, greenDirection: 'more_back', greenMagnitude: 'small', showArrow: false, priority: 'must', phases: KEY_PHASES },
    ],
    mustShowLines: ['spineAxis'],
    auxiliaryLines: ['hipLine'],
    paths: [],
    showImpactTriangle: false,
  },

  // ══ 3. hand_path_issue ══
  {
    issue: 'hand_path_issue',
    viewType: 'face_on',
    userMessage: '手应该更贴近身体。',
    mustShowPoints: [
      { point: 'gripCenter',  showRed: true, showGreen: true, greenDirection: 'more_inside', greenMagnitude: 'medium', showArrow: true, priority: 'must', phases: KEY_PHASES },
      { point: 'leftWrist',   showRed: true, showGreen: false, showArrow: false, priority: 'important', phases: KEY_PHASES },
      { point: 'rightWrist',  showRed: true, showGreen: false, showArrow: false, priority: 'important', phases: KEY_PHASES },
    ],
    mustShowLines: ['leftArmChain'],
    auxiliaryLines: ['armTriangle'],
    paths: [
      { trackedPoint: 'gripCenter', showRedPath: true, showGreenPath: false, phases: ALL_PHASES },
    ],
    showImpactTriangle: true,
  },
  {
    issue: 'hand_path_issue',
    viewType: 'down_the_line',
    userMessage: '手路径太靠外，应该更往里下来。',
    mustShowPoints: [
      { point: 'gripCenter',  showRed: true, showGreen: true, greenDirection: 'more_inside', greenMagnitude: 'medium', showArrow: true, priority: 'must', phases: KEY_PHASES },
      { point: 'leftWrist',   showRed: true, showGreen: false, showArrow: false, priority: 'important', phases: KEY_PHASES },
      { point: 'rightWrist',  showRed: true, showGreen: false, showArrow: false, priority: 'important', phases: KEY_PHASES },
      { point: 'rightElbow',  showRed: true, showGreen: false, showArrow: false, priority: 'important', phases: KEY_PHASES },
    ],
    mustShowLines: ['leftArmChain', 'rightArmChain'],
    auxiliaryLines: [],
    paths: [
      { trackedPoint: 'gripCenter', showRedPath: true, showGreenPath: true, phases: ALL_PHASES },
    ],
    showImpactTriangle: false,
  },

  // ══ 4. head_movement ══
  {
    issue: 'head_movement',
    viewType: 'face_on',
    userMessage: '头不要飘，要更稳定居中。',
    mustShowPoints: [
      { point: 'headCenter', showRed: true, showGreen: true, greenDirection: 'more_centered', greenMagnitude: 'medium', showArrow: true, priority: 'must', phases: ALL_PHASES },
    ],
    mustShowLines: [],
    auxiliaryLines: ['spineAxis'],
    paths: [
      { trackedPoint: 'headCenter', showRedPath: true, showGreenPath: false, phases: ALL_PHASES },
    ],
    showImpactTriangle: false,
  },
  {
    issue: 'head_movement',
    viewType: 'down_the_line',
    userMessage: '头部位移影响了击球一致性。',
    mustShowPoints: [
      { point: 'headCenter', showRed: true, showGreen: true, greenDirection: 'more_stable', greenMagnitude: 'small', showArrow: false, priority: 'must', phases: ALL_PHASES },
    ],
    mustShowLines: [],
    auxiliaryLines: ['spineAxis'],
    paths: [
      { trackedPoint: 'headCenter', showRedPath: true, showGreenPath: false, phases: ALL_PHASES },
    ],
    showImpactTriangle: false,
  },

  // ══ 5. not_enough_hip_turn ══
  {
    issue: 'not_enough_hip_turn',
    viewType: 'face_on',
    userMessage: '髋应该转更多一点。',
    mustShowPoints: [
      { point: 'leftHip',   showRed: true, showGreen: false, showArrow: false, priority: 'important', phases: ALL_PHASES },
      { point: 'rightHip',  showRed: true, showGreen: false, showArrow: false, priority: 'important', phases: ALL_PHASES },
      { point: 'hipCenter', showRed: true, showGreen: true, greenDirection: 'more_turned', greenMagnitude: 'medium', showArrow: true, priority: 'must', phases: KEY_PHASES },
    ],
    mustShowLines: ['hipLine'],
    auxiliaryLines: ['shoulderLine'],
    paths: [],
    showImpactTriangle: false,
  },
  {
    issue: 'not_enough_hip_turn',
    viewType: 'down_the_line',
    userMessage: '髋部需要更早更完整地打开。',
    mustShowPoints: [
      { point: 'leftHip',   showRed: true, showGreen: false, showArrow: false, priority: 'important', phases: ALL_PHASES },
      { point: 'rightHip',  showRed: true, showGreen: false, showArrow: false, priority: 'important', phases: ALL_PHASES },
      { point: 'hipCenter', showRed: true, showGreen: true, greenDirection: 'more_forward', greenMagnitude: 'medium', showArrow: true, priority: 'must', phases: KEY_PHASES },
    ],
    mustShowLines: ['hipLine'],
    auxiliaryLines: [],
    paths: [],
    showImpactTriangle: false,
  },

  // ══ 6. steep_downswing ══
  {
    issue: 'steep_downswing',
    viewType: 'face_on',
    userMessage: '下杆路径太陡，手和杆应该更往里、更浅一点。',
    mustShowPoints: [
      { point: 'gripCenter',   showRed: true, showGreen: true, greenDirection: 'more_inside', greenMagnitude: 'medium', showArrow: true, priority: 'must', phases: ['transition', 'impact'] },
      { point: 'rightShoulder', showRed: true, showGreen: true, greenDirection: 'lower', greenMagnitude: 'small', showArrow: false, priority: 'important', phases: ['transition'] },
    ],
    mustShowLines: ['shoulderLine'],
    auxiliaryLines: ['spineAxis'],
    paths: [
      { trackedPoint: 'gripCenter', showRedPath: true, showGreenPath: true, phases: ['top', 'transition', 'impact'] },
    ],
    showImpactTriangle: false,
  },
  {
    issue: 'steep_downswing',
    viewType: 'down_the_line',
    userMessage: '下杆路径从外侧来，需要更从里侧浅入。',
    mustShowPoints: [
      { point: 'gripCenter',  showRed: true, showGreen: true, greenDirection: 'more_inside', greenMagnitude: 'medium', showArrow: true, priority: 'must', phases: ['transition', 'impact'] },
      { point: 'rightElbow',  showRed: true, showGreen: true, greenDirection: 'lower', greenMagnitude: 'medium', showArrow: true, priority: 'must', phases: ['transition'] },
    ],
    mustShowLines: ['rightArmChain'],
    auxiliaryLines: ['shoulderLine', 'spineAxis'],
    paths: [
      { trackedPoint: 'gripCenter', showRedPath: true, showGreenPath: true, phases: ['top', 'transition', 'impact'] },
    ],
    showImpactTriangle: false,
  },

  // ══ 7. early_extension ══
  {
    issue: 'early_extension',
    viewType: 'face_on',
    userMessage: '髋不要往球方向顶出去，身体不要太早站起。',
    mustShowPoints: [
      { point: 'leftHip',   showRed: true, showGreen: false, showArrow: false, priority: 'important', phases: KEY_PHASES },
      { point: 'rightHip',  showRed: true, showGreen: false, showArrow: false, priority: 'important', phases: KEY_PHASES },
      { point: 'hipCenter', showRed: true, showGreen: true, greenDirection: 'more_back', greenMagnitude: 'medium', showArrow: true, priority: 'must', phases: ['transition', 'impact'] },
      { point: 'shoulderCenter', showRed: true, showGreen: true, greenDirection: 'lower', greenMagnitude: 'small', showArrow: false, priority: 'important', phases: ['impact'] },
    ],
    mustShowLines: ['hipLine', 'spineAxis'],
    auxiliaryLines: ['shoulderLine'],
    paths: [],
    showImpactTriangle: false,
  },
  {
    issue: 'early_extension',
    viewType: 'down_the_line',
    userMessage: '击球时身体站起，失去了脊柱角度。',
    mustShowPoints: [
      { point: 'hipCenter',      showRed: true, showGreen: true, greenDirection: 'more_back', greenMagnitude: 'medium', showArrow: true, priority: 'must', phases: ['transition', 'impact'] },
      { point: 'shoulderCenter', showRed: true, showGreen: true, greenDirection: 'lower', greenMagnitude: 'medium', showArrow: false, priority: 'must', phases: ['impact'] },
    ],
    mustShowLines: ['spineAxis', 'hipLine'],
    auxiliaryLines: [],
    paths: [],
    showImpactTriangle: false,
  },

  // ══ 8. weight_shift_issue ══
  {
    issue: 'weight_shift_issue',
    viewType: 'face_on',
    userMessage: '重心要更早转到前脚。',
    mustShowPoints: [
      { point: 'leftHip',   showRed: true, showGreen: false, showArrow: false, priority: 'important', phases: KEY_PHASES },
      { point: 'rightHip',  showRed: true, showGreen: false, showArrow: false, priority: 'important', phases: KEY_PHASES },
      { point: 'hipCenter', showRed: true, showGreen: true, greenDirection: 'more_forward', greenMagnitude: 'medium', showArrow: true, priority: 'must', phases: ['transition', 'impact'] },
      { point: 'leftAnkle', showRed: true, showGreen: false, showArrow: false, priority: 'auxiliary', phases: ['impact'] },
      { point: 'rightAnkle', showRed: true, showGreen: false, showArrow: false, priority: 'auxiliary', phases: ['impact'] },
    ],
    mustShowLines: ['hipLine'],
    auxiliaryLines: ['leftLegChain', 'rightLegChain'],
    paths: [
      { trackedPoint: 'hipCenter', showRedPath: true, showGreenPath: false, phases: KEY_PHASES },
    ],
    showImpactTriangle: false,
  },
  {
    issue: 'weight_shift_issue',
    viewType: 'down_the_line',
    userMessage: '重心转移不足，没有充分到达前侧。',
    mustShowPoints: [
      { point: 'hipCenter', showRed: true, showGreen: true, greenDirection: 'more_forward', greenMagnitude: 'medium', showArrow: true, priority: 'must', phases: ['transition', 'impact'] },
    ],
    mustShowLines: ['hipLine'],
    auxiliaryLines: [],
    paths: [],
    showImpactTriangle: false,
  },

  // ══ 9. steep_backswing_plane ══
  {
    issue: 'steep_backswing_plane',
    viewType: 'face_on',
    userMessage: '上杆平面太陡，杆应该更绕着身体走。',
    mustShowPoints: [
      { point: 'rightElbow',  showRed: true, showGreen: true, greenDirection: 'more_inside', greenMagnitude: 'medium', showArrow: true, priority: 'must', phases: ['top'] },
      { point: 'gripCenter',  showRed: true, showGreen: true, greenDirection: 'shallower', greenMagnitude: 'medium', showArrow: false, priority: 'important', phases: ['top'] },
    ],
    mustShowLines: ['rightArmChain'],
    auxiliaryLines: ['shoulderLine'],
    paths: [
      { trackedPoint: 'gripCenter', showRedPath: true, showGreenPath: true, phases: ['setup', 'top'] },
    ],
    showImpactTriangle: false,
  },
  {
    issue: 'steep_backswing_plane',
    viewType: 'down_the_line',
    userMessage: '上杆路径太垂直，需要更绕一点。',
    mustShowPoints: [
      { point: 'rightElbow',  showRed: true, showGreen: true, greenDirection: 'more_inside', greenMagnitude: 'medium', showArrow: true, priority: 'must', phases: ['top'] },
      { point: 'gripCenter',  showRed: true, showGreen: true, greenDirection: 'shallower', greenMagnitude: 'medium', showArrow: true, priority: 'must', phases: ['top'] },
    ],
    mustShowLines: ['rightArmChain', 'leftArmChain'],
    auxiliaryLines: ['shoulderLine'],
    paths: [
      { trackedPoint: 'gripCenter', showRedPath: true, showGreenPath: true, phases: ['setup', 'top'] },
    ],
    showImpactTriangle: false,
  },
];

/**
 * 获取指定问题和视角的显示规范
 */
export function getIssueDisplaySpec(
  issue: MainIssueType,
  viewType: ViewType,
): IssueDisplayConfig | null {
  return ISSUE_DISPLAY_SPEC.find(
    s => s.issue === issue && s.viewType === viewType
  ) ?? null;
}

/**
 * 过滤出当前阶段需要显示的点
 */
export function filterPointsByPhase(
  spec: IssueDisplayConfig,
  phase: SwingPhase,
): PointDisplaySpec[] {
  return spec.mustShowPoints.filter(p => p.phases.includes(phase));
}
