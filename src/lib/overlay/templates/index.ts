/**
 * templates/index.ts
 *
 * 模板生成器：给定 issue 类型和阶段时间节点，
 * 生成结构化 OverlayTimeline。
 *
 * 每个模板关注最重要的 1-2 个视觉对比，不要画太多线。
 * 视觉语言：红=当前错误，绿=目标正确，黄=方向/路径
 *
 * 坐标系统：归一化 0-1（以视频宽高为参考）
 * 高尔夫选手面对摄像机（Face-On视角）位置参考：
 *   头部：   (0.50, 0.11)
 *   左肩：   (0.37, 0.27)  右肩：(0.63, 0.27)
 *   左肘：   (0.30, 0.42)  右肘：(0.70, 0.42)
 *   左腕：   (0.26, 0.55)  右腕：(0.56, 0.55)
 *   左髋：   (0.42, 0.54)  右髋：(0.58, 0.54)
 *   球位：   (0.47, 0.80)
 */

import type {
  OverlayTimeline, OverlayFrame, OverlayElement,
  PhaseMarkers, VideoMetadata, MainIssueType, TemplateInput,
} from '@/types/analysis';

/* ─── Shorthand builders ─── */
type C = 'red' | 'green' | 'yellow' | 'white';

const dot = (id: string, x: number, y: number, color: C, r = 0.026, opacity = 0.95): OverlayElement =>
  ({ type: 'dot', id, x, y, color, radius: r, opacity, layer: color === 'green' || color === 'white' ? 'body' : 'arms' });

const dotArm = (id: string, x: number, y: number, color: C, r = 0.024): OverlayElement =>
  ({ type: 'dot', id, x, y, color, radius: r, opacity: 0.95, layer: 'arms' });

const line = (id: string, x1: number, y1: number, x2: number, y2: number, color: C, w = 3, dashed = false, opacity = 0.85, lyr: OverlayElement['layer'] = 'body'): OverlayElement =>
  ({ type: 'line', id, x1, y1, x2, y2, color, strokeWidth: w, dashed, opacity, layer: lyr });

const arrow = (id: string, fx: number, fy: number, tx: number, ty: number, color: C, opacity = 0.92): OverlayElement =>
  ({ type: 'arrow', id, from: { x: fx, y: fy }, to: { x: tx, y: ty }, color, strokeWidth: 3, opacity });

const label = (id: string, x: number, y: number, text: string, color: C, size = 11, opacity = 0.90): OverlayElement =>
  ({ type: 'label', id, x, y, text, color, size, opacity });

const curve = (id: string, points: {x:number;y:number}[], color: C, w = 3.5, opacity = 0.88, lyr: OverlayElement['layer'] = 'arms'): OverlayElement =>
  ({ type: 'curve', id, points, color, strokeWidth: w, opacity, layer: lyr });

const badge = (id: string, x: number, y: number, v: 'correct'|'wrong'): OverlayElement =>
  ({ type: 'badge', id, x, y, variant: v, opacity: 0.88 });

/* ─── Canonical address skeleton (shared) ─── */
function addressSkeleton(phase: string): OverlayElement[] {
  return [
    // Spine (white)
    line('spine', 0.50, 0.19, 0.49, 0.54, 'white', 2.5, false, 0.65),
    // Shoulder line (green)
    line('l-shoulder', 0.37, 0.27, 0.63, 0.27, 'green', 2.5, false, 0.80),
    // Hip line (green, dashed)
    line('hip', 0.42, 0.54, 0.58, 0.54, 'green', 2.0, true, 0.60),
    // Shoulder dots
    dot('ls', 0.37, 0.27, 'green', 0.024),
    dot('rs', 0.63, 0.27, 'green', 0.024),
    // Phase label
    label('ph', 0.50, 0.06, phase.toUpperCase(), 'white', 10, 0.70),
  ];
}

/* ══════════════════════════════════════════════
   TEMPLATE 1: posture_structure_template
   Issues: early_extension, posture_rises_too_early, shoulder_lifts_too_early
══════════════════════════════════════════════ */

function postureStructureTemplate(phases: PhaseMarkers, dur: number): OverlayFrame[] {
  const t = phases;
  return [
    {
      time: t.setupTime,
      phase: 'setup',
      elements: [
        ...addressSkeleton('Setup'),
        // Hip line and spine are same — correct starting position
        label('cue', 0.50, 0.93, 'Good address posture', 'green', 10, 0.70),
      ],
    },
    {
      time: t.topTime,
      phase: 'top',
      elements: [
        line('spine-top', 0.48, 0.17, 0.44, 0.52, 'white', 2.5, false, 0.65),
        line('sh-top', 0.30, 0.30, 0.64, 0.23, 'green', 2.5, false, 0.80),
        dot('ls-top', 0.30, 0.30, 'green', 0.024),
        dot('rs-top', 0.64, 0.23, 'green', 0.024),
        label('ph-top', 0.50, 0.06, 'TOP', 'white', 10, 0.70),
        label('cue', 0.50, 0.93, 'Hold your posture angle', 'yellow', 10),
      ],
    },
    {
      time: t.transitionTime,
      phase: 'transition',
      elements: [
        // RED: body starting to rise (incorrect)
        line('sp-red', 0.50, 0.14, 0.52, 0.50, 'red', 3, false, 0.85),
        dot('head-red', 0.50, 0.10, 'red', 0.024),
        line('sh-red', 0.33, 0.22, 0.65, 0.26, 'red', 2.5, false, 0.80),

        // GREEN: maintaining posture (correct)
        line('sp-grn', 0.62, 0.17, 0.60, 0.52, 'green', 3, true, 0.82),
        dot('head-grn', 0.62, 0.13, 'green', 0.022),
        line('sh-grn', 0.48, 0.27, 0.78, 0.27, 'green', 2.5, true, 0.75),

        arrow('fix-head', 0.50, 0.10, 0.50, 0.13, 'yellow'),
        label('ph', 0.50, 0.06, 'TRANSITION — watch posture', 'yellow', 10),
      ],
    },
    {
      time: t.impactTime,
      phase: 'impact',
      elements: [
        // RED: body risen, hips thrust — WRONG
        line('sp-red', 0.50, 0.10, 0.53, 0.48, 'red', 3.5, false, 0.90),
        dot('head-red', 0.50, 0.07, 'red', 0.026),
        line('sh-red', 0.32, 0.20, 0.64, 0.24, 'red', 3, false, 0.85),
        line('hip-red', 0.47, 0.50, 0.64, 0.50, 'red', 2.5, false, 0.80),
        badge('wrong', 0.32, 0.25, 'wrong'),

        // GREEN: posture maintained — CORRECT
        line('sp-grn', 0.68, 0.13, 0.64, 0.52, 'green', 3.5, true, 0.88),
        dot('head-grn', 0.68, 0.11, 'green', 0.024),
        line('sh-grn', 0.47, 0.24, 0.78, 0.30, 'green', 3, true, 0.82),
        badge('correct', 0.78, 0.20, 'correct'),

        // Arrow: head should stay down
        arrow('fix', 0.50, 0.07, 0.50, 0.12, 'yellow'),
        label('lbl-red', 0.30, 0.06, 'Body rises ↑', 'red', 10),
        label('lbl-grn', 0.78, 0.06, 'Stay in posture', 'green', 10),
        label('ph', 0.50, 0.97, 'IMPACT — key moment', 'yellow', 10),
      ],
    },
    {
      time: t.finishTime,
      phase: 'finish',
      elements: [
        line('sp-fin', 0.50, 0.14, 0.53, 0.52, 'white', 2, false, 0.60),
        label('ph', 0.50, 0.06, 'FINISH', 'white', 10, 0.65),
      ],
    },
  ];
}

/* ══════════════════════════════════════════════
   TEMPLATE 2: swing_path_template
   Issues: steep_downswing, steep_backswing_plane
══════════════════════════════════════════════ */

function swingPathTemplate(phases: PhaseMarkers, _dur: number): OverlayFrame[] {
  const t = phases;
  return [
    {
      time: t.setupTime,
      phase: 'setup',
      elements: [
        ...addressSkeleton('Setup'),
        // Swing plane reference line
        line('plane', 0.55, 0.78, 0.22, 0.12, 'white', 1.5, true, 0.35),
        label('cue', 0.50, 0.93, 'Watch the club path down', 'yellow', 10, 0.75),
      ],
    },
    {
      time: t.topTime,
      phase: 'top',
      elements: [
        line('sp', 0.48, 0.17, 0.44, 0.52, 'white', 2, false, 0.55),
        line('sh', 0.30, 0.30, 0.64, 0.23, 'white', 2, false, 0.65),
        dot('ls', 0.30, 0.30, 'white', 0.022),
        dot('rs', 0.64, 0.23, 'white', 0.022),
        // Club at top
        dot('club-top', 0.29, 0.12, 'yellow', 0.022),
        line('shaft-top', 0.44, 0.35, 0.29, 0.12, 'yellow', 2.5, false, 0.80, 'club'),
        label('ph', 0.50, 0.06, 'TOP — downswing path matters', 'yellow', 10),
      ],
    },
    {
      time: t.transitionTime,
      phase: 'transition',
      elements: [
        line('sp', 0.49, 0.16, 0.46, 0.52, 'white', 2, false, 0.55),
        dot('ls', 0.34, 0.27, 'white', 0.020),
        dot('rs', 0.63, 0.27, 'white', 0.020),

        // RED: over-the-top, steep path
        curve('path-red', [
          {x:0.26,y:0.12}, {x:0.35,y:0.20}, {x:0.46,y:0.32}, {x:0.50,y:0.50}
        ], 'red', 3.5, 0.88, 'club'),
        dot('club-red', 0.26, 0.11, 'red', 0.024),
        label('lbl-red', 0.18, 0.08, 'Over-the-top', 'red', 10),

        // GREEN: shallow inside path
        curve('path-grn', [
          {x:0.29,y:0.12}, {x:0.38,y:0.22}, {x:0.50,y:0.38}, {x:0.52,y:0.52}
        ], 'green', 3.5, 0.85, 'club'),
        dot('club-grn', 0.29, 0.11, 'green', 0.022),
        label('lbl-grn', 0.42, 0.08, 'Shallow', 'green', 10),

        arrow('fix', 0.30, 0.25, 0.37, 0.28, 'yellow'),
        label('ph', 0.50, 0.97, 'TRANSITION — drop it inside', 'yellow', 10),
      ],
    },
    {
      time: t.impactTime,
      phase: 'impact',
      elements: [
        line('sp', 0.48, 0.14, 0.44, 0.51, 'white', 2.5, false, 0.65),
        line('sh', 0.30, 0.25, 0.61, 0.32, 'white', 2, false, 0.65),

        // RED path (steeper, outside-in)
        curve('path-red', [
          {x:0.22,y:0.08}, {x:0.35,y:0.20}, {x:0.48,y:0.38}, {x:0.50,y:0.56}
        ], 'red', 4, 0.88, 'club'),
        badge('wrong', 0.16, 0.16, 'wrong'),

        // GREEN path (inside-out, shallower)
        curve('path-grn', [
          {x:0.32,y:0.14}, {x:0.42,y:0.26}, {x:0.52,y:0.42}, {x:0.54,y:0.56}
        ], 'green', 4, 0.85, 'club'),
        badge('correct', 0.60, 0.14, 'correct'),

        label('lbl-red', 0.14, 0.06, 'Steep path', 'red', 10),
        label('lbl-grn', 0.64, 0.06, 'Shallow path', 'green', 10),
        label('ph', 0.50, 0.97, 'IMPACT — compare paths', 'yellow', 10),
      ],
    },
    {
      time: t.finishTime,
      phase: 'finish',
      elements: [
        label('ph', 0.50, 0.06, 'FINISH', 'white', 10, 0.60),
      ],
    },
  ];
}

/* ══════════════════════════════════════════════
   TEMPLATE 3: arm_structure_template
   Issues: hand_path_issue
══════════════════════════════════════════════ */

function armStructureTemplate(phases: PhaseMarkers, _dur: number): OverlayFrame[] {
  const t = phases;
  return [
    {
      time: t.setupTime,
      phase: 'setup',
      elements: [
        ...addressSkeleton('Setup'),
        // Left arm (green = lead)
        line('la', 0.37, 0.27, 0.26, 0.55, 'green', 2.5, false, 0.80, 'arms'),
        // Right arm (red = trail)
        line('ra', 0.63, 0.27, 0.56, 0.55, 'red', 2.5, false, 0.80, 'arms'),
        // Grip point (where hands meet)
        dotArm('grip', 0.41, 0.55, 'green', 0.028),
        label('ph', 0.50, 0.06, 'SETUP — watch hand path', 'white', 10, 0.70),
      ],
    },
    {
      time: t.topTime,
      phase: 'top',
      elements: [
        line('sp', 0.48, 0.17, 0.44, 0.52, 'white', 2, false, 0.55),
        // Left arm at top
        line('la', 0.30, 0.30, 0.28, 0.20, 'green', 3, false, 0.82, 'arms'),
        dotArm('lw', 0.27, 0.18, 'green', 0.026),
        // Right arm at top
        line('ra', 0.64, 0.23, 0.42, 0.18, 'red', 3, false, 0.82, 'arms'),
        label('ph', 0.50, 0.06, 'TOP — hands here', 'white', 10, 0.70),
        label('cue', 0.50, 0.93, 'Keep hands close on the way down', 'yellow', 10),
      ],
    },
    {
      time: t.transitionTime,
      phase: 'transition',
      elements: [
        line('sp', 0.49, 0.16, 0.46, 0.52, 'white', 2, false, 0.55),

        // RED: hands loop outside
        curve('path-red', [
          {x:0.35,y:0.18}, {x:0.54,y:0.24}, {x:0.58,y:0.36}, {x:0.52,y:0.52}
        ], 'red', 3.5, 0.88, 'arms'),
        dotArm('rw-red', 0.54,0.23,'red',0.024),
        label('lbl-red', 0.60,0.22,'Away from body','red',10),

        // GREEN: hands drop inside
        curve('path-grn', [
          {x:0.35,y:0.18}, {x:0.38,y:0.28}, {x:0.40,y:0.40}, {x:0.42,y:0.52}
        ], 'green', 3.5, 0.85, 'arms'),
        dotArm('rw-grn', 0.38,0.27,'green',0.022),
        label('lbl-grn', 0.24,0.28,'Drop inside','green',10),

        arrow('fix', 0.54,0.26, 0.40,0.32, 'yellow'),
        label('ph', 0.50, 0.97, 'TRANSITION — drop hands inside', 'yellow', 10),
      ],
    },
    {
      time: t.impactTime,
      phase: 'impact',
      elements: [
        line('sp', 0.48, 0.14, 0.44, 0.51, 'white', 2.5, false, 0.65),
        line('sh', 0.28, 0.25, 0.60, 0.32, 'white', 2, false, 0.65),

        // Full path traces
        curve('path-red', [
          {x:0.35,y:0.18},{x:0.54,y:0.24},{x:0.58,y:0.36},{x:0.52,y:0.58}
        ], 'red', 3.5, 0.80, 'arms'),
        curve('path-grn', [
          {x:0.35,y:0.18},{x:0.38,y:0.28},{x:0.40,y:0.40},{x:0.42,y:0.58}
        ], 'green', 3.5, 0.80, 'arms'),

        badge('wrong', 0.60, 0.18, 'wrong'),
        badge('correct', 0.28, 0.18, 'correct'),
        label('lbl-red', 0.64, 0.06, 'Outside path', 'red', 10),
        label('lbl-grn', 0.22, 0.06, 'Inside path ✓', 'green', 10),
        label('ph', 0.50, 0.97, 'IMPACT — path result', 'yellow', 10),
      ],
    },
    {
      time: t.finishTime,
      phase: 'finish',
      elements: [
        label('ph', 0.50, 0.06, 'FINISH', 'white', 10, 0.60),
      ],
    },
  ];
}

/* ══════════════════════════════════════════════
   TEMPLATE 4: weight_shift_template
   Issues: weight_shift_issue, not_enough_hip_turn
══════════════════════════════════════════════ */

function weightShiftTemplate(phases: PhaseMarkers, _dur: number): OverlayFrame[] {
  const t = phases;
  return [
    {
      time: t.setupTime,
      phase: 'setup',
      elements: [
        ...addressSkeleton('Setup'),
        // Foot markers
        dot('lf', 0.38, 0.80, 'white', 0.020),
        dot('rf', 0.62, 0.80, 'white', 0.020),
        label('lfl', 0.38, 0.88, 'Lead', 'white', 9, 0.50),
        label('rfl', 0.62, 0.88, 'Trail', 'white', 9, 0.50),
        label('ph', 0.50, 0.06, 'SETUP — 50/50 weight', 'white', 10, 0.70),
      ],
    },
    {
      time: t.topTime,
      phase: 'top',
      elements: [
        line('sp', 0.48, 0.17, 0.44, 0.52, 'white', 2, false, 0.55),
        // Trail foot heavier (correct at top)
        dot('rf-heavy', 0.62, 0.80, 'green', 0.032),
        dot('lf', 0.38, 0.80, 'white', 0.020),
        label('rf-lbl', 0.62, 0.88, '60%', 'green', 10),
        label('lf-lbl', 0.38, 0.88, '40%', 'white', 9, 0.60),
        label('ph', 0.50, 0.06, 'TOP — weight on trail', 'green', 10),
      ],
    },
    {
      time: t.impactTime,
      phase: 'impact',
      elements: [
        line('sp', 0.47, 0.14, 0.43, 0.51, 'white', 2.5, false, 0.65),
        line('sh', 0.28, 0.25, 0.60, 0.32, 'white', 2, false, 0.65),

        // RED: weight stuck back (wrong)
        dot('rf-red', 0.62, 0.80, 'red', 0.036),
        dot('lf-red', 0.38, 0.80, 'red', 0.020),
        label('rf-lbl-red', 0.62, 0.88, '70% ✗', 'red', 10),
        badge('wrong', 0.65, 0.68, 'wrong'),

        // GREEN: weight on lead (correct)
        dot('lf-grn', 0.28, 0.80, 'green', 0.036),
        dot('rf-grn', 0.50, 0.80, 'green', 0.020),
        label('lf-lbl-grn', 0.28, 0.88, '70% ✓', 'green', 10),
        badge('correct', 0.22, 0.68, 'correct'),

        // Arrow showing shift direction
        arrow('shift', 0.58, 0.70, 0.38, 0.70, 'yellow'),
        label('lbl-yellow', 0.50, 0.64, 'Shift to lead side', 'yellow', 10),
        label('ph', 0.50, 0.97, 'IMPACT — weight transfer', 'yellow', 10),
      ],
    },
    {
      time: t.finishTime,
      phase: 'finish',
      elements: [
        dot('lf-fin', 0.38, 0.80, 'green', 0.038),
        label('fin-lbl', 0.38, 0.88, '90% ✓', 'green', 10),
        label('ph', 0.50, 0.06, 'FINISH — full transfer', 'green', 10, 0.75),
      ],
    },
  ];
}

/* ══════════════════════════════════════════════
   TEMPLATE 5: local_focus_template
   Issues: head_movement
══════════════════════════════════════════════ */

function localFocusTemplate(phases: PhaseMarkers, _dur: number): OverlayFrame[] {
  const t = phases;
  const CENTER_X = 0.50;
  return [
    {
      time: t.setupTime,
      phase: 'setup',
      elements: [
        ...addressSkeleton('Setup'),
        // Vertical center line through head
        line('ctr', CENTER_X, 0.04, CENTER_X, 0.20, 'green', 1.5, true, 0.40),
        dot('head', CENTER_X, 0.11, 'green', 0.026),
        label('ph', 0.50, 0.97, 'Head stays here throughout swing', 'green', 10, 0.80),
      ],
    },
    {
      time: t.topTime,
      phase: 'top',
      elements: [
        line('sp', 0.48, 0.17, 0.44, 0.52, 'white', 2, false, 0.55),
        line('sh', 0.30, 0.30, 0.64, 0.23, 'white', 2, false, 0.65),
        // Vertical centerline
        line('ctr', CENTER_X, 0.04, CENTER_X, 0.20, 'green', 1.5, true, 0.40),

        // RED head drifted
        dot('head-red', 0.44, 0.11, 'red', 0.028),
        arrow('drift', 0.48, 0.11, 0.44, 0.11, 'yellow'),
        label('lbl-red', 0.38, 0.07, '← Drifting', 'red', 10),

        // GREEN head centered
        dot('head-grn', CENTER_X, 0.11, 'green', 0.022),
        label('lbl-grn', 0.60, 0.07, 'Stay here', 'green', 10),
        label('ph', 0.50, 0.97, 'TOP — head must stay centered', 'yellow', 10),
      ],
    },
    {
      time: t.impactTime,
      phase: 'impact',
      elements: [
        line('sp', 0.48, 0.14, 0.44, 0.51, 'white', 2.5, false, 0.65),
        line('sh', 0.28, 0.25, 0.60, 0.32, 'white', 2, false, 0.65),
        // Centerline
        line('ctr', CENTER_X, 0.04, CENTER_X, 0.20, 'green', 1.5, true, 0.40),
        // Head centered at impact
        dot('head', CENTER_X, 0.11, 'green', 0.026),
        badge('correct', 0.60, 0.08, 'correct'),
        label('ph', 0.50, 0.97, 'IMPACT — head centered ✓', 'green', 10),
      ],
    },
    {
      time: t.finishTime,
      phase: 'finish',
      elements: [
        label('ph', 0.50, 0.06, 'FINISH', 'white', 10, 0.60),
      ],
    },
  ];
}

/* ══════════════════════════════════════════════
   MAIN GENERATOR
══════════════════════════════════════════════ */

export function generateOverlayTimeline(input: TemplateInput): OverlayTimeline {
  const { phaseMarkers: phases, videoMetadata: meta, issue } = input;
  const dur = meta.durationSec;

  let frames: OverlayFrame[];
  switch (issue) {
    case 'early_extension':
    case 'posture_rises_too_early':
    case 'shoulder_lifts_too_early':
      frames = postureStructureTemplate(phases, dur);
      break;
    case 'steep_downswing':
    case 'steep_backswing_plane':
      frames = swingPathTemplate(phases, dur);
      break;
    case 'hand_path_issue':
      frames = armStructureTemplate(phases, dur);
      break;
    case 'weight_shift_issue':
    case 'not_enough_hip_turn':
      frames = weightShiftTemplate(phases, dur);
      break;
    case 'head_movement':
      frames = localFocusTemplate(phases, dur);
      break;
    default:
      frames = postureStructureTemplate(phases, dur);
  }

  return { frames };
}
