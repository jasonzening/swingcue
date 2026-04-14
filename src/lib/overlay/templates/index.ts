/**
 * templates/index.ts — Overlay Timeline 生成器
 *
 * 优先级：
 * 1. 真实 keypointTimeline → 使用 keypointOverlay 生成真实点位
 * 2. 无真实数据 → 使用预设归一化坐标的示意 overlay
 *
 * 每个 OverlayFrame 包含：
 * - 来自真实 keypoints 的骨架线（白色）
 * - 当前错误点位（红色，来自真实坐标）
 * - 目标点位（绿色，真实坐标 + 修正量）
 * - 方向箭头（黄色）
 * - 阶段标签
 */

import type {
  OverlayTimeline, OverlayFrame, OverlayElement,
  PhaseMarkers, VideoMetadata, MainIssueType, TemplateInput,
  KeypointFrame,
} from '@/types/analysis';
import type { TargetCorrection } from '@/lib/golf/types';
import { ISSUE_DEFINITIONS } from '@/lib/golf/issues';
import {
  generateKeypointOverlayFrame,
  findNearestFrame,
} from '@/lib/overlay/keypointOverlay';

/* ── Shorthand builders ── */
type C = 'red' | 'green' | 'yellow' | 'white';

const dot = (id: string, x: number, y: number, color: C, r = 0.026, opacity = 0.95, lyr: OverlayElement['layer'] = 'body'): OverlayElement =>
  ({ type: 'dot', id, x, y, color, radius: r, opacity, layer: lyr });

const line = (id: string, x1: number, y1: number, x2: number, y2: number, color: C, w = 3, dashed = false, opacity = 0.85, lyr: OverlayElement['layer'] = 'body'): OverlayElement =>
  ({ type: 'line', id, x1, y1, x2, y2, color, strokeWidth: w, dashed, opacity, layer: lyr });

const arrow = (id: string, fx: number, fy: number, tx: number, ty: number, color: C, opacity = 0.92): OverlayElement =>
  ({ type: 'arrow', id, from: { x: fx, y: fy }, to: { x: tx, y: ty }, color, strokeWidth: 3, opacity });

const label = (id: string, x: number, y: number, text: string, color: C, size = 11, opacity = 0.88): OverlayElement =>
  ({ type: 'label', id, x, y, text, color, size, opacity });

const badge = (id: string, x: number, y: number, v: 'correct'|'wrong'): OverlayElement =>
  ({ type: 'badge', id, x, y, variant: v, opacity: 0.88 });

/* ══════════════════════════════════════════
   KEYPOINT-DRIVEN FRAME GENERATOR
   When real keypoints are available, use them.
══════════════════════════════════════════ */

function buildKeypointFrame(
  time: number,
  phase: OverlayFrame['phase'],
  phaseCorrections: TargetCorrection[],
  allKpFrames: KeypointFrame[],
  phaseLabel: string,
): OverlayFrame {
  const nearestKp = findNearestFrame(allKpFrames, time);

  const elements: OverlayElement[] = [];

  if (nearestKp) {
    // Use REAL keypoints
    const phaseSpecific = phaseCorrections.filter(c => c.phase === phase);
    elements.push(
      ...generateKeypointOverlayFrame(
        phaseSpecific,
        nearestKp,
        true,   // include skeleton
        true,   // include arms
      )
    );
  } else {
    // Fallback: minimal reference skeleton (pre-set)
    elements.push(
      line('sp', 0.50, 0.19, 0.49, 0.54, 'white', 2, false, 0.60),
      line('sh', 0.37, 0.27, 0.63, 0.27, 'green', 2.5, false, 0.75),
      dot('ls', 0.37, 0.27, 'green', 0.020),
      dot('rs', 0.63, 0.27, 'green', 0.020),
    );
  }

  // Phase label always visible
  elements.push(label('ph', 0.50, 0.06, phaseLabel.toUpperCase(), 'white', 10, 0.65));

  return { time, phase, elements };
}

/* ══════════════════════════════════════════
   FALLBACK TEMPLATES (no keypoints)
   Pre-set illustrative overlays per issue type.
══════════════════════════════════════════ */

function addressSkeleton(): OverlayElement[] {
  return [
    line('sp', 0.50, 0.19, 0.49, 0.54, 'white', 2.5, false, 0.65),
    line('sh', 0.37, 0.27, 0.63, 0.27, 'green', 2.5, false, 0.80),
    line('hi', 0.42, 0.54, 0.58, 0.54, 'green', 2.0, true, 0.60),
    dot('ls', 0.37, 0.27, 'green', 0.024),
    dot('rs', 0.63, 0.27, 'green', 0.024),
  ];
}

function fallbackPostureTemplate(phases: PhaseMarkers): OverlayFrame[] {
  const t = phases;
  return [
    { time: t.setupTime, phase: 'setup', elements: [...addressSkeleton(), label('ph', 0.50, 0.06, 'SETUP', 'white', 10, 0.65)] },
    { time: t.topTime, phase: 'top', elements: [
      line('sh', 0.30, 0.30, 0.64, 0.23, 'white', 2.5, false, 0.70),
      dot('ls', 0.30, 0.30, 'white', 0.022), dot('rs', 0.64, 0.23, 'white', 0.022),
      label('ph', 0.50, 0.06, 'TOP', 'white', 10, 0.65),
    ]},
    { time: t.transitionTime, phase: 'transition', elements: [
      line('sp-r', 0.50, 0.14, 0.52, 0.50, 'red', 3, false, 0.85),
      dot('h-r', 0.50, 0.10, 'red', 0.024),
      line('sp-g', 0.62, 0.17, 0.60, 0.52, 'green', 3, true, 0.80),
      dot('h-g', 0.62, 0.13, 'green', 0.022),
      arrow('fix', 0.50, 0.10, 0.50, 0.13, 'yellow'),
      label('ph', 0.50, 0.06, 'TRANSITION', 'yellow', 10),
    ]},
    { time: t.impactTime, phase: 'impact', elements: [
      line('sp-r', 0.50, 0.10, 0.53, 0.48, 'red', 3.5, false, 0.90),
      dot('h-r', 0.50, 0.07, 'red', 0.026),
      badge('wr', 0.32, 0.25, 'wrong'),
      line('sp-g', 0.68, 0.13, 0.64, 0.52, 'green', 3.5, true, 0.85),
      dot('h-g', 0.68, 0.11, 'green', 0.024),
      badge('ok', 0.78, 0.20, 'correct'),
      arrow('fix', 0.50, 0.07, 0.50, 0.12, 'yellow'),
      label('lr', 0.30, 0.06, 'Body rises ↑', 'red', 10),
      label('lg', 0.78, 0.06, 'Stay down ✓', 'green', 10),
      label('ph', 0.50, 0.97, 'IMPACT — key moment', 'yellow', 10),
    ]},
    { time: t.finishTime, phase: 'finish', elements: [label('ph', 0.50, 0.06, 'FINISH', 'white', 10, 0.60)] },
  ];
}

function fallbackSwingPathTemplate(phases: PhaseMarkers): OverlayFrame[] {
  const t = phases;
  return [
    { time: t.setupTime, phase: 'setup', elements: [...addressSkeleton(), label('ph', 0.50, 0.06, 'SETUP', 'white', 10, 0.65)] },
    { time: t.topTime, phase: 'top', elements: [
      line('sh', 0.30, 0.30, 0.64, 0.23, 'white', 2, false, 0.65),
      dot('club', 0.29, 0.12, 'yellow', 0.022),
      label('ph', 0.50, 0.06, 'TOP — watch downswing path', 'yellow', 10),
    ]},
    { time: t.transitionTime, phase: 'transition', elements: [
      { type: 'curve', id: 'pr', color: 'red', points: [{x:0.26,y:0.12},{x:0.35,y:0.20},{x:0.46,y:0.32},{x:0.50,y:0.50}], strokeWidth: 3.5, opacity: 0.88, layer: 'club' },
      dot('cr', 0.26, 0.11, 'red', 0.024, 0.88, 'club'),
      { type: 'curve', id: 'pg', color: 'green', points: [{x:0.29,y:0.12},{x:0.38,y:0.22},{x:0.50,y:0.38},{x:0.52,y:0.52}], strokeWidth: 3.5, opacity: 0.85, layer: 'club' },
      dot('cg', 0.29, 0.11, 'green', 0.022, 0.85, 'club'),
      label('lr', 0.18, 0.08, 'Over-the-top', 'red', 10),
      label('lg', 0.42, 0.08, 'Shallow ✓', 'green', 10),
      label('ph', 0.50, 0.97, 'TRANSITION — drop inside', 'yellow', 10),
    ]},
    { time: t.impactTime, phase: 'impact', elements: [
      { type: 'curve', id: 'pr', color: 'red', points: [{x:0.22,y:0.08},{x:0.35,y:0.20},{x:0.48,y:0.38},{x:0.50,y:0.56}], strokeWidth: 4, opacity: 0.88, layer: 'club' },
      badge('wr', 0.16, 0.16, 'wrong'),
      { type: 'curve', id: 'pg', color: 'green', points: [{x:0.32,y:0.14},{x:0.42,y:0.26},{x:0.52,y:0.42},{x:0.54,y:0.56}], strokeWidth: 4, opacity: 0.85, layer: 'club' },
      badge('ok', 0.60, 0.14, 'correct'),
      label('ph', 0.50, 0.97, 'IMPACT — path comparison', 'yellow', 10),
    ]},
    { time: t.finishTime, phase: 'finish', elements: [label('ph', 0.50, 0.06, 'FINISH', 'white', 10, 0.60)] },
  ];
}

function fallbackArmTemplate(phases: PhaseMarkers): OverlayFrame[] {
  const t = phases;
  return [
    { time: t.setupTime, phase: 'setup', elements: [
      ...addressSkeleton(),
      line('la', 0.37, 0.27, 0.26, 0.55, 'green', 2.5, false, 0.80, 'arms'),
      line('ra', 0.63, 0.27, 0.56, 0.55, 'red', 2.5, false, 0.80, 'arms'),
      dot('grip', 0.41, 0.55, 'green', 0.028, 0.95, 'arms'),
      label('ph', 0.50, 0.06, 'SETUP — watch hand path', 'white', 10, 0.70),
    ]},
    { time: t.transitionTime, phase: 'transition', elements: [
      { type: 'curve', id: 'pr', color: 'red', points: [{x:0.35,y:0.18},{x:0.54,y:0.24},{x:0.58,y:0.36},{x:0.52,y:0.52}], strokeWidth: 3.5, opacity: 0.88, layer: 'arms' },
      { type: 'curve', id: 'pg', color: 'green', points: [{x:0.35,y:0.18},{x:0.38,y:0.28},{x:0.40,y:0.40},{x:0.42,y:0.52}], strokeWidth: 3.5, opacity: 0.85, layer: 'arms' },
      arrow('fix', 0.54, 0.26, 0.40, 0.32, 'yellow'),
      label('lr', 0.60, 0.22, 'Away from body', 'red', 10),
      label('lg', 0.22, 0.30, 'Drop inside ✓', 'green', 10),
      label('ph', 0.50, 0.97, 'TRANSITION', 'yellow', 10),
    ]},
    { time: t.impactTime, phase: 'impact', elements: [
      badge('wr', 0.60, 0.18, 'wrong'), badge('ok', 0.28, 0.18, 'correct'),
      label('ph', 0.50, 0.97, 'IMPACT', 'yellow', 10),
    ]},
    { time: t.finishTime, phase: 'finish', elements: [label('ph', 0.50, 0.06, 'FINISH', 'white', 10, 0.60)] },
  ];
}

function fallbackWeightTemplate(phases: PhaseMarkers): OverlayFrame[] {
  const t = phases;
  return [
    { time: t.setupTime, phase: 'setup', elements: [
      ...addressSkeleton(),
      dot('lf', 0.38, 0.80, 'white', 0.020), dot('rf', 0.62, 0.80, 'white', 0.020),
      label('ph', 0.50, 0.06, 'SETUP', 'white', 10, 0.70),
    ]},
    { time: t.impactTime, phase: 'impact', elements: [
      dot('rf-r', 0.62, 0.80, 'red', 0.036), dot('lf-r', 0.38, 0.80, 'red', 0.020),
      label('rf-l', 0.62, 0.88, '70% ✗', 'red', 10),
      badge('wr', 0.65, 0.68, 'wrong'),
      dot('lf-g', 0.28, 0.80, 'green', 0.036), dot('rf-g', 0.50, 0.80, 'green', 0.020),
      label('lf-l', 0.28, 0.88, '70% ✓', 'green', 10),
      badge('ok', 0.22, 0.68, 'correct'),
      arrow('shft', 0.58, 0.70, 0.38, 0.70, 'yellow'),
      label('lbl', 0.50, 0.64, 'Shift to lead side', 'yellow', 10),
      label('ph', 0.50, 0.97, 'IMPACT — weight transfer', 'yellow', 10),
    ]},
    { time: t.finishTime, phase: 'finish', elements: [
      dot('lf', 0.38, 0.80, 'green', 0.038),
      label('ph', 0.50, 0.06, 'FINISH — full transfer ✓', 'green', 10, 0.75),
    ]},
  ];
}

function fallbackHeadTemplate(phases: PhaseMarkers): OverlayFrame[] {
  const t = phases;
  const cx = 0.50;
  return [
    { time: t.setupTime, phase: 'setup', elements: [
      ...addressSkeleton(),
      line('ctr', cx, 0.04, cx, 0.20, 'green', 1.5, true, 0.40),
      dot('head', cx, 0.11, 'green', 0.026),
      label('ph', 0.50, 0.06, 'SETUP', 'white', 10, 0.65),
    ]},
    { time: t.topTime, phase: 'top', elements: [
      line('ctr', cx, 0.04, cx, 0.20, 'green', 1.5, true, 0.40),
      dot('h-r', 0.44, 0.11, 'red', 0.028),
      dot('h-g', cx, 0.11, 'green', 0.022),
      arrow('drift', 0.48, 0.11, 0.44, 0.11, 'yellow'),
      label('lr', 0.34, 0.07, '← Drifting', 'red', 10),
      label('ph', 0.50, 0.06, 'TOP — head stays centered', 'yellow', 10),
    ]},
    { time: t.impactTime, phase: 'impact', elements: [
      line('ctr', cx, 0.04, cx, 0.20, 'green', 1.5, true, 0.40),
      dot('head', cx, 0.11, 'green', 0.026),
      badge('ok', 0.60, 0.08, 'correct'),
      label('ph', 0.50, 0.97, 'IMPACT — centered ✓', 'green', 10),
    ]},
    { time: t.finishTime, phase: 'finish', elements: [label('ph', 0.50, 0.06, 'FINISH', 'white', 10, 0.60)] },
  ];
}

/* ══════════════════════════════════════════
   MAIN EXPORT
══════════════════════════════════════════ */

export function generateOverlayTimeline(input: TemplateInput): OverlayTimeline {
  const { phaseMarkers: phases, videoMetadata: meta, issue, keypointTimeline } = input;
  const kpFrames: KeypointFrame[] = keypointTimeline?.frames ?? [];
  const hasRealKeypoints = kpFrames.length >= 3;

  // Get corrections for this issue from Golf Intelligence Layer
  const issueData = ISSUE_DEFINITIONS[issue];
  const corrections: TargetCorrection[] = issueData?.targetCorrections ?? [];

  let frames: OverlayFrame[];

  if (hasRealKeypoints) {
    // ── PATH A: Real keypoints available ──
    // Generate frames at each phase using real body positions
    frames = [
      buildKeypointFrame(phases.setupTime,      'setup',      corrections, kpFrames, 'SETUP'),
      buildKeypointFrame(phases.topTime,        'top',        corrections, kpFrames, 'TOP'),
      buildKeypointFrame(phases.transitionTime, 'transition', corrections, kpFrames, 'TRANSITION'),
      buildKeypointFrame(phases.impactTime,     'impact',     corrections, kpFrames, 'IMPACT'),
      buildKeypointFrame(phases.finishTime,     'finish',     corrections, kpFrames, 'FINISH'),
    ];

    // Add impact-specific cue
    if (issueData) {
      const impactFrame = frames.find(f => f.phase === 'impact');
      if (impactFrame) {
        impactFrame.elements.push(
          label('cue', 0.50, 0.97, `"${issueData.cue}"`, 'yellow', 10, 0.82)
        );
      }
    }

  } else {
    // ── PATH B: No keypoints — use pre-set illustrative overlays ──
    switch (issue) {
      case 'early_extension':
      case 'posture_rises_too_early':
      case 'shoulder_lifts_too_early':
        frames = fallbackPostureTemplate(phases);
        break;
      case 'steep_downswing':
      case 'steep_backswing_plane':
        frames = fallbackSwingPathTemplate(phases);
        break;
      case 'hand_path_issue':
        frames = fallbackArmTemplate(phases);
        break;
      case 'weight_shift_issue':
      case 'not_enough_hip_turn':
        frames = fallbackWeightTemplate(phases);
        break;
      case 'head_movement':
        frames = fallbackHeadTemplate(phases);
        break;
      default:
        frames = fallbackPostureTemplate(phases);
    }
  }

  return { frames };
}
