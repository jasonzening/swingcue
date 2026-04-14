/**
 * golf/issues.ts — 问题定义数据库
 *
 * 每个 issue 包含：
 * - summary（一句话问题说明）
 * - cue（教练短提示，不超过 16 个单词）
 * - drill（马上能练的简单建议）
 * - targetCorrections（可视化层需要的修正指令）
 * - overlayInstructions（告诉可视化层画什么）
 * - severity / priority（评分权重）
 */

import type { MainIssueType } from '@/types/analysis';
import type {
  GolfIssueResult, TargetCorrection, OverlayInstruction
} from './types';

type IssueDefinition = Omit<GolfIssueResult, 'confidence'> & {
  priority: number;    // 0-1, 对新手的重要性
  severityDefault: GolfIssueResult['severity'];
};

export const ISSUE_DEFINITIONS: Record<MainIssueType, IssueDefinition> = {

  /* ── 1. STEEP DOWNSWING ── */
  steep_downswing: {
    issue: 'steep_downswing',
    severity: 'medium',
    severityDefault: 'medium',
    priority: 0.90,
    summary: 'Your club is coming down too steep — an over-the-top path causing pulls and fat shots.',
    cue: 'Drop your trail elbow inside before you turn.',
    drill: 'Slow transition reps: feel the club fall behind you before rotating.',
    targetCorrections: [
      {
        id: 'tc-sd-1',
        phase: 'transition',
        bodyPart: 'hands',
        direction: 'more_inside',
        magnitude: 'medium',
        message: 'Hands should work more inside — not over the top.',
      },
      {
        id: 'tc-sd-2',
        phase: 'transition',
        bodyPart: 'trailShoulder',
        direction: 'lower',
        magnitude: 'medium',
        message: 'Trail shoulder stays lower — let it stay down through transition.',
      },
      {
        id: 'tc-sd-3',
        phase: 'transition',
        bodyPart: 'club',
        direction: 'shallower',
        magnitude: 'medium',
        message: 'Club path needs to be shallower — less steep on the way down.',
      },
    ],
    overlayInstructions: [
      {
        phase: 'transition',
        focusArea: 'club_path',
        redReference: 'Current steep, outside-in downswing path',
        greenTarget: 'Target: shallower, inside-out path',
        emphasis: 'curve',
      },
      {
        phase: 'transition',
        focusArea: 'shoulder_line',
        redReference: 'Trail shoulder too high',
        greenTarget: 'Trail shoulder lower in transition',
        emphasis: 'arrow',
      },
    ],
  },

  /* ── 2. HAND PATH ISSUE ── */
  hand_path_issue: {
    issue: 'hand_path_issue',
    severity: 'medium',
    severityDefault: 'medium',
    priority: 0.70,
    summary: 'Your hands loop away from your body in the downswing, causing casting and weak contact.',
    cue: 'Keep your hands close — let them fall inside in transition.',
    drill: 'Slow swings with a towel tucked under your trail arm.',
    targetCorrections: [
      {
        id: 'tc-hp-1',
        phase: 'transition',
        bodyPart: 'hands',
        direction: 'more_inside',
        magnitude: 'medium',
        message: 'Hands should work more inside — not away from the body.',
      },
      {
        id: 'tc-hp-2',
        phase: 'transition',
        bodyPart: 'hands',
        direction: 'more_centered',
        magnitude: 'medium',
        message: 'Keep the hands closer to your body through transition.',
      },
    ],
    overlayInstructions: [
      {
        phase: 'transition',
        focusArea: 'hand_path',
        redReference: 'Current: hands loop away from body',
        greenTarget: 'Target: hands drop inside, closer to body',
        emphasis: 'curve',
      },
    ],
  },

  /* ── 3. EARLY EXTENSION ── */
  early_extension: {
    issue: 'early_extension',
    severity: 'medium',
    severityDefault: 'medium',
    priority: 0.85,
    summary: 'Your hips thrust toward the ball before impact — losing posture and robbing distance.',
    cue: 'Stay in your posture longer through impact.',
    drill: 'Wall Drill: practice swings with your backside against a wall.',
    targetCorrections: [
      {
        id: 'tc-ee-1',
        phase: 'impact',
        bodyPart: 'hips',
        direction: 'more_back',
        magnitude: 'medium',
        message: 'Hips rotate — do not thrust toward the ball.',
      },
      {
        id: 'tc-ee-2',
        phase: 'impact',
        bodyPart: 'spine',
        direction: 'more_stable',
        magnitude: 'medium',
        message: 'Maintain your spine angle from address through impact.',
      },
    ],
    overlayInstructions: [
      {
        phase: 'impact',
        focusArea: 'posture',
        redReference: 'Body rising — spine angle lost at impact',
        greenTarget: 'Posture maintained through impact',
        emphasis: 'line',
      },
    ],
  },

  /* ── 4. WEIGHT SHIFT ── */
  weight_shift_issue: {
    issue: 'weight_shift_issue',
    severity: 'high',
    severityDefault: 'high',
    priority: 0.80,
    summary: 'Your weight stays on the back foot — a reverse pivot that costs you distance and control.',
    cue: 'Shift your pressure forward before you make contact.',
    drill: 'Step Drill: step your trail foot toward lead foot on the downswing.',
    targetCorrections: [
      {
        id: 'tc-ws-1',
        phase: 'impact',
        bodyPart: 'weight',
        direction: 'more_forward',
        magnitude: 'large',
        message: 'Weight must be on the lead side at impact.',
      },
      {
        id: 'tc-ws-2',
        phase: 'transition',
        bodyPart: 'hips',
        direction: 'more_forward',
        magnitude: 'medium',
        message: 'Hips should shift toward the target in transition.',
      },
    ],
    overlayInstructions: [
      {
        phase: 'impact',
        focusArea: 'weight_shift',
        redReference: 'Weight stuck on trail foot',
        greenTarget: '70% on lead foot at impact',
        emphasis: 'zone',
      },
      {
        phase: 'transition',
        focusArea: 'hip_turn',
        redReference: 'No hip shift forward',
        greenTarget: 'Hips shift then rotate to target',
        emphasis: 'arrow',
      },
    ],
  },

  /* ── 5. HEAD MOVEMENT ── */
  head_movement: {
    issue: 'head_movement',
    severity: 'low',
    severityDefault: 'low',
    priority: 0.65,
    summary: 'Your head is drifting sideways — making consistent contact harder than it needs to be.',
    cue: 'Keep your head centered throughout the swing.',
    drill: 'Make slow swings keeping your head inside a smaller window.',
    targetCorrections: [
      {
        id: 'tc-hm-1',
        phase: 'top',
        bodyPart: 'head',
        direction: 'more_centered',
        magnitude: 'medium',
        message: 'Head should stay close to its address position.',
      },
      {
        id: 'tc-hm-2',
        phase: 'transition',
        bodyPart: 'head',
        direction: 'more_stable',
        magnitude: 'medium',
        message: 'Minimize lateral head movement through transition.',
      },
    ],
    overlayInstructions: [
      {
        phase: 'top',
        focusArea: 'head_position',
        redReference: 'Head drifts too far laterally',
        greenTarget: 'Head stays within a stable zone',
        emphasis: 'zone',
      },
    ],
  },

  /* ── 6. STEEP BACKSWING PLANE ── */
  steep_backswing_plane: {
    issue: 'steep_backswing_plane',
    severity: 'medium',
    severityDefault: 'medium',
    priority: 0.75,
    summary: 'Your backswing is too vertical — making the correct downswing path much harder.',
    cue: 'Take the club back on a flatter, more around plane.',
    drill: 'Takeaway reps keeping the club head lower and more around your body.',
    targetCorrections: [
      {
        id: 'tc-sb-1',
        phase: 'top',
        bodyPart: 'club',
        direction: 'shallower',
        magnitude: 'medium',
        message: 'Club should be on a flatter plane at the top.',
      },
      {
        id: 'tc-sb-2',
        phase: 'top',
        bodyPart: 'hands',
        direction: 'more_inside',
        magnitude: 'small',
        message: 'Hands work more around — less straight up.',
      },
    ],
    overlayInstructions: [
      {
        phase: 'top',
        focusArea: 'club_path',
        redReference: 'Club too vertical in backswing',
        greenTarget: 'Flatter, on-plane backswing',
        emphasis: 'curve',
      },
    ],
  },

  /* ── 7. POSTURE RISES TOO EARLY ── */
  posture_rises_too_early: {
    issue: 'posture_rises_too_early',
    severity: 'medium',
    severityDefault: 'medium',
    priority: 0.80,
    summary: 'Your body height increases before impact — standing up out of your posture too soon.',
    cue: 'Stay lower for longer — keep your chest down through impact.',
    drill: 'Slow swings focusing on keeping your chest angled toward the ball at impact.',
    targetCorrections: [
      {
        id: 'tc-pr-1',
        phase: 'impact',
        bodyPart: 'shoulders',
        direction: 'lower',
        magnitude: 'medium',
        message: 'Shoulders should stay lower through impact.',
      },
      {
        id: 'tc-pr-2',
        phase: 'impact',
        bodyPart: 'spine',
        direction: 'more_stable',
        magnitude: 'medium',
        message: 'Spine angle should be maintained from address.',
      },
    ],
    overlayInstructions: [
      {
        phase: 'impact',
        focusArea: 'posture',
        redReference: 'Body height increasing too early',
        greenTarget: 'Maintained posture angle to impact',
        emphasis: 'line',
      },
    ],
  },

  /* ── 8. SHOULDER LIFTS TOO EARLY ── */
  shoulder_lifts_too_early: {
    issue: 'shoulder_lifts_too_early',
    severity: 'medium',
    severityDefault: 'medium',
    priority: 0.75,
    summary: 'Your trail shoulder rises too early in transition — causing a steep, over-the-top move.',
    cue: 'Keep your trail shoulder lower and later.',
    drill: 'Transition reps feeling your trail shoulder stay down and around.',
    targetCorrections: [
      {
        id: 'tc-sl-1',
        phase: 'transition',
        bodyPart: 'trailShoulder',
        direction: 'lower',
        magnitude: 'medium',
        message: 'Trail shoulder stays lower — do not lift it early.',
      },
    ],
    overlayInstructions: [
      {
        phase: 'transition',
        focusArea: 'shoulder_line',
        redReference: 'Trail shoulder lifting too early',
        greenTarget: 'Trail shoulder lower, delayed',
        emphasis: 'arrow',
      },
    ],
  },

  /* ── 9. NOT ENOUGH HIP TURN ── */
  not_enough_hip_turn: {
    issue: 'not_enough_hip_turn',
    severity: 'medium',
    severityDefault: 'medium',
    priority: 0.70,
    summary: 'Your hips are not rotating enough — limiting your power and follow-through.',
    cue: 'Turn your hips fully — let them lead the downswing.',
    drill: 'Practice slow swings focusing on opening your hips fully before impact.',
    targetCorrections: [
      {
        id: 'tc-ht-1',
        phase: 'impact',
        bodyPart: 'hips',
        direction: 'more_turned',
        magnitude: 'medium',
        message: 'Hips need to rotate more fully through impact.',
      },
    ],
    overlayInstructions: [
      {
        phase: 'impact',
        focusArea: 'hip_turn',
        redReference: 'Limited hip rotation',
        greenTarget: 'Full hip turn through impact',
        emphasis: 'arrow',
      },
    ],
  },
};

/** 问题优先级（用于排名）*/
export const ISSUE_PRIORITY: Record<MainIssueType, number> = {
  steep_downswing:         0.90,
  early_extension:         0.85,
  weight_shift_issue:      0.80,
  posture_rises_too_early: 0.80,
  steep_backswing_plane:   0.75,
  shoulder_lifts_too_early:0.75,
  hand_path_issue:         0.70,
  not_enough_hip_turn:     0.70,
  head_movement:           0.65,
};

/** 从检测结果中获取完整 issue 定义 */
export function getIssueResult(
  issue: MainIssueType,
  confidence: number = 0.6
): GolfIssueResult {
  const def = ISSUE_DEFINITIONS[issue];
  if (!def) return getIssueResult('early_extension', confidence); // fallback
  return {
    issue: def.issue,
    severity: def.severityDefault,
    confidence,
    summary: def.summary,
    cue: def.cue,
    drill: def.drill,
    targetCorrections: def.targetCorrections,
    overlayInstructions: def.overlayInstructions,
  };
}
