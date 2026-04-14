/**
 * generateOverlayTimeline.ts
 *
 * Generates normalized overlay timeline data for each swing issue type.
 * Coordinates are in normalized space (0–1 relative to video width/height),
 * with the golfer centered in frame (face-on view assumed).
 *
 * This is a stub implementation — Phase 2 will replace with real MediaPipe data.
 *
 * The golfer skeleton reference positions (face-on, address stance):
 *   Head:           x=0.50, y=0.12
 *   Neck:           x=0.50, y=0.20
 *   Left shoulder:  x=0.37, y=0.28
 *   Right shoulder: x=0.63, y=0.28
 *   Left hip:       x=0.42, y=0.54
 *   Right hip:      x=0.58, y=0.54
 *   Left knee:      x=0.42, y=0.72
 *   Right knee:     x=0.58, y=0.72
 */

export type OverlayElementType = 'line' | 'circle' | 'arrow' | 'label';

export interface LineEl {
  type: 'line';
  color: 'red' | 'green' | 'white' | 'yellow';
  x1: number; y1: number;
  x2: number; y2: number;
  strokeWidth?: number;
  dashed?: boolean;
  opacity?: number;
}

export interface CircleEl {
  type: 'circle';
  color: 'red' | 'green' | 'white' | 'yellow';
  cx: number; cy: number;
  r: number;
  opacity?: number;
}

export interface ArrowEl {
  type: 'arrow';
  color: 'red' | 'green' | 'white' | 'yellow';
  fromX: number; fromY: number;
  toX: number; toY: number;
  opacity?: number;
}

export interface LabelEl {
  type: 'label';
  color: 'red' | 'green' | 'white' | 'yellow';
  x: number; y: number;
  text: string;
  size?: number;
  opacity?: number;
}

export type OverlayEl = LineEl | CircleEl | ArrowEl | LabelEl;

export interface OverlayKeyframe {
  t: number;           // normalized time 0–1 (0 = start, 1 = end of swing)
  phase: string;       // 'address' | 'takeaway' | 'top' | 'transition' | 'impact' | 'finish'
  elements: OverlayEl[];
}

export interface PhaseMarkers {
  address: number;
  takeaway: number;
  top: number;
  transition: number;
  impact: number;
  finish: number;
}

export interface OverlayTimeline {
  keyframes: OverlayKeyframe[];
  phases: PhaseMarkers;
  issueType: string;
}

/* ── Helpers ── */
function line(
  color: LineEl['color'],
  x1: number, y1: number, x2: number, y2: number,
  opts?: Partial<LineEl>
): LineEl {
  return { type: 'line', color, x1, y1, x2, y2, strokeWidth: 3, ...opts };
}

function circle(color: CircleEl['color'], cx: number, cy: number, r: number, opacity = 1): CircleEl {
  return { type: 'circle', color, cx, cy, r, opacity };
}

function arrow(color: ArrowEl['color'], fromX: number, fromY: number, toX: number, toY: number, opacity = 1): ArrowEl {
  return { type: 'arrow', color, fromX, fromY, toX, toY, opacity };
}

function label(color: LabelEl['color'], x: number, y: number, text: string, size = 12, opacity = 1): LabelEl {
  return { type: 'label', color, x, y, text, size, opacity };
}

/* ── Skeleton builder: given positions, build standard coaching lines ── */
function buildSkeleton(
  color: LineEl['color'],
  head: [number, number],
  lShoulder: [number, number],
  rShoulder: [number, number],
  lHip: [number, number],
  rHip: [number, number],
  dashed = false,
  opacity = 1.0,
): LineEl[] {
  const midHip: [number, number] = [(lHip[0] + rHip[0]) / 2, (lHip[1] + rHip[1]) / 2];
  const neck: [number, number] = [(lShoulder[0] + rShoulder[0]) / 2, (lShoulder[1] + rShoulder[1]) / 2];
  const o = { dashed, opacity, strokeWidth: dashed ? 2.5 : 3 };
  return [
    line(color, neck[0], neck[1], midHip[0], midHip[1], { ...o, strokeWidth: dashed ? 2 : 3.5 }), // spine
    line(color, lShoulder[0], lShoulder[1], rShoulder[0], rShoulder[1], o), // shoulder
    line(color, lHip[0], lHip[1], rHip[0], rHip[1], o), // hip
    line(color, neck[0], neck[1], head[0], head[1], { ...o, strokeWidth: dashed ? 1.5 : 2 }), // neck
  ];
}

/* ═════════════════════════════════════════════
   EARLY EXTENSION
   Issue: body rises + hips thrust at impact
   ═════════════════════════════════════════════ */
function earlyExtension(): OverlayTimeline {
  const phases: PhaseMarkers = {
    address: 0.0,
    takeaway: 0.15,
    top: 0.50,
    transition: 0.62,
    impact: 0.75,
    finish: 0.92,
  };

  // Canonical positions at each phase for GREEN (correct)
  const G = {
    address: {
      head: [0.50, 0.13] as [number,number], lS: [0.38, 0.28] as [number,number], rS: [0.62, 0.28] as [number,number],
      lH: [0.42, 0.54] as [number,number], rH: [0.58, 0.54] as [number,number],
    },
    top: {
      head: [0.48, 0.12] as [number,number], lS: [0.32, 0.32] as [number,number], rS: [0.66, 0.24] as [number,number],
      lH: [0.40, 0.53] as [number,number], rH: [0.60, 0.53] as [number,number],
    },
    impact: {
      // Correct: head stays, spine tilted, hips rotate
      head: [0.48, 0.13] as [number,number], lS: [0.28, 0.24] as [number,number], rS: [0.60, 0.34] as [number,number],
      lH: [0.38, 0.52] as [number,number], rH: [0.56, 0.52] as [number,number],
    },
    finish: {
      head: [0.50, 0.14] as [number,number], lS: [0.50, 0.30] as [number,number], rS: [0.70, 0.22] as [number,number],
      lH: [0.45, 0.53] as [number,number], rH: [0.62, 0.50] as [number,number],
    },
  };

  // RED (incorrect) deviates at transition + impact
  const R_impact = {
    head: [0.50, 0.09] as [number,number],   // rises up
    lS: [0.33, 0.21] as [number,number], rS: [0.63, 0.27] as [number,number],  // more upright
    lH: [0.46, 0.52] as [number,number], rH: [0.64, 0.52] as [number,number],  // hips thrust toward ball
  };

  const frames: OverlayKeyframe[] = [
    {
      t: 0.0, phase: 'address',
      elements: [
        ...buildSkeleton('white', G.address.head, G.address.lS, G.address.rS, G.address.lH, G.address.rH),
        label('white', 0.50, 0.05, 'Address', 11),
      ],
    },
    {
      t: 0.25, phase: 'takeaway',
      elements: [
        ...buildSkeleton('white', G.address.head,
          [G.address.lS[0] - 0.04, G.address.lS[1]],
          [G.address.rS[0] + 0.02, G.address.rS[1]],
          G.address.lH, G.address.rH),
        label('white', 0.50, 0.05, 'Takeaway', 11),
      ],
    },
    {
      t: 0.50, phase: 'top',
      elements: [
        ...buildSkeleton('white', G.top.head, G.top.lS, G.top.rS, G.top.lH, G.top.rH),
        label('white', 0.50, 0.05, 'Top', 11),
        // club path starts
        circle('white', 0.38, 0.22, 0.015),
      ],
    },
    {
      t: 0.65, phase: 'transition',
      elements: [
        // Green: already correct coming down
        ...buildSkeleton('green',
          [0.49, 0.12], [0.35, 0.26], [0.62, 0.30], [0.41, 0.53], [0.57, 0.53],
          true, 0.8),
        // Red: early extension starting
        ...buildSkeleton('red',
          [0.49, 0.10], [0.37, 0.23], [0.63, 0.27], [0.44, 0.52], [0.61, 0.52],
          false, 0.8),
        label('yellow', 0.50, 0.05, 'Transition — watch hips', 11),
      ],
    },
    {
      t: 0.75, phase: 'impact',
      elements: [
        // Green: correct impact posture
        ...buildSkeleton('green', G.impact.head, G.impact.lS, G.impact.rS, G.impact.lH, G.impact.rH, true),
        // Red: early extension in full effect
        ...buildSkeleton('red', R_impact.head, R_impact.lS, R_impact.rS, R_impact.lH, R_impact.rH, false),
        // Arrow showing body should stay lower
        arrow('yellow', R_impact.head[0], R_impact.head[1], G.impact.head[0], G.impact.head[1], 0.9),
        // Arrow showing hips should rotate not thrust
        arrow('yellow', R_impact.rH[0], R_impact.rH[1], G.impact.rH[0], G.impact.rH[1] - 0.02, 0.9),
        label('red', 0.75, 0.08, 'Body rising ↑', 11),
        label('green', 0.20, 0.08, 'Stay in posture ✓', 11),
        label('yellow', 0.50, 0.05, 'IMPACT — key moment', 11, 0.9),
      ],
    },
    {
      t: 0.92, phase: 'finish',
      elements: [
        ...buildSkeleton('white', G.finish.head, G.finish.lS, G.finish.rS, G.finish.lH, G.finish.rH),
        label('white', 0.50, 0.05, 'Finish', 11),
      ],
    },
  ];

  return { keyframes: frames, phases, issueType: 'early_extension' };
}

/* ═══════════════════════════════════════════
   STEEP DOWNSWING (Over-the-top)
   Issue: club comes down too steep / outside-in
   ═══════════════════════════════════════════ */
function steepDownswing(): OverlayTimeline {
  const phases: PhaseMarkers = {
    address: 0.0, takeaway: 0.15, top: 0.48, transition: 0.60, impact: 0.75, finish: 0.92,
  };

  const frames: OverlayKeyframe[] = [
    {
      t: 0.0, phase: 'address',
      elements: [
        ...buildSkeleton('white', [0.50,0.13],[0.38,0.28],[0.62,0.28],[0.42,0.54],[0.58,0.54]),
        label('white', 0.50, 0.05, 'Address', 11),
      ],
    },
    {
      t: 0.50, phase: 'top',
      elements: [
        ...buildSkeleton('white', [0.48,0.12],[0.32,0.30],[0.66,0.24],[0.40,0.53],[0.60,0.53]),
        // Club at top (handle and head markers)
        circle('white', 0.52, 0.18, 0.02),
        circle('white', 0.30, 0.14, 0.015),
        line('white', 0.30, 0.14, 0.52, 0.18, { strokeWidth: 2 }),
        label('white', 0.50, 0.05, 'Top — watch downswing path', 11),
      ],
    },
    {
      t: 0.62, phase: 'transition',
      elements: [
        // Skeleton
        ...buildSkeleton('white', [0.49,0.12],[0.35,0.27],[0.63,0.29],[0.41,0.53],[0.58,0.53]),
        // RED path: club coming over the top (steep, outside)
        line('red', 0.26, 0.16, 0.50, 0.52, { strokeWidth: 4 }),
        circle('red', 0.26, 0.14, 0.022),
        // GREEN path: inside-out, shallower
        line('green', 0.36, 0.22, 0.52, 0.54, { strokeWidth: 3, dashed: true }),
        circle('green', 0.36, 0.20, 0.018),
        label('red', 0.16, 0.12, 'Too steep ↓', 11),
        label('green', 0.55, 0.13, 'Shallow ↓', 11),
        label('yellow', 0.50, 0.05, 'Transition — club path', 11),
      ],
    },
    {
      t: 0.75, phase: 'impact',
      elements: [
        ...buildSkeleton('white', [0.48,0.13],[0.30,0.25],[0.60,0.33],[0.40,0.52],[0.56,0.52]),
        // Red: steep path trace
        line('red', 0.22, 0.08, 0.50, 0.58, { strokeWidth: 3, opacity: 0.7 }),
        arrow('red', 0.22, 0.10, 0.50, 0.56, 0.9),
        // Green: shallower path
        line('green', 0.32, 0.14, 0.52, 0.60, { strokeWidth: 3, dashed: true, opacity: 0.7 }),
        arrow('green', 0.34, 0.16, 0.52, 0.58, 0.9),
        // Angle difference indicator
        label('red', 0.15, 0.25, 'Steep path', 11),
        label('green', 0.58, 0.25, 'Shallow path', 11),
        label('yellow', 0.50, 0.05, 'IMPACT — path comparison', 11),
      ],
    },
    {
      t: 0.92, phase: 'finish',
      elements: [
        ...buildSkeleton('white', [0.50,0.14],[0.50,0.30],[0.70,0.22],[0.45,0.53],[0.62,0.50]),
        label('white', 0.50, 0.05, 'Finish', 11),
      ],
    },
  ];
  return { keyframes: frames, phases, issueType: 'steep_downswing' };
}

/* ═════════════════════════════════════
   HEAD MOVEMENT
   Issue: head drifts laterally
   ═════════════════════════════════════ */
function headMovement(): OverlayTimeline {
  const phases: PhaseMarkers = {
    address: 0.0, takeaway: 0.15, top: 0.50, transition: 0.62, impact: 0.75, finish: 0.92,
  };
  const centerX = 0.50;

  const frames: OverlayKeyframe[] = [
    {
      t: 0.0, phase: 'address',
      elements: [
        ...buildSkeleton('white', [0.50,0.13],[0.38,0.28],[0.62,0.28],[0.42,0.54],[0.58,0.54]),
        // Center line
        line('green', centerX, 0.06, centerX, 0.22, { strokeWidth: 1.5, dashed: true, opacity: 0.5 }),
        circle('green', centerX, 0.13, 0.025, 0.5),
        label('white', 0.50, 0.05, 'Address — head position', 11),
      ],
    },
    {
      t: 0.40, phase: 'takeaway',
      elements: [
        ...buildSkeleton('white', [0.50,0.13],[0.35,0.27],[0.64,0.27],[0.42,0.54],[0.58,0.54]),
        // Center line
        line('green', centerX, 0.06, centerX, 0.22, { strokeWidth: 1.5, dashed: true, opacity: 0.5 }),
        // Head drift
        circle('red', 0.45, 0.13, 0.025),   // drifted
        circle('green', centerX, 0.13, 0.02, 0.6), // target
        arrow('yellow', 0.45, 0.13, centerX, 0.13, 0.8),
        label('red', 0.35, 0.09, 'Drifts →', 11),
        label('yellow', 0.50, 0.05, 'Keep head centered', 11),
      ],
    },
    {
      t: 0.50, phase: 'top',
      elements: [
        ...buildSkeleton('white', [0.46,0.13],[0.30,0.30],[0.65,0.24],[0.40,0.53],[0.60,0.53]),
        line('green', centerX, 0.06, centerX, 0.22, { strokeWidth: 1.5, dashed: true, opacity: 0.5 }),
        circle('red', 0.44, 0.12, 0.025),
        circle('green', centerX, 0.12, 0.02, 0.6),
        arrow('yellow', 0.44, 0.12, 0.50, 0.12, 0.8),
        label('red', 0.30, 0.08, '← Drifted', 11),
        label('yellow', 0.50, 0.05, 'Top — head stays here', 11),
      ],
    },
    {
      t: 0.75, phase: 'impact',
      elements: [
        ...buildSkeleton('white', [0.50,0.13],[0.30,0.25],[0.60,0.33],[0.40,0.52],[0.56,0.52]),
        line('green', centerX, 0.06, centerX, 0.22, { strokeWidth: 1.5, dashed: true, opacity: 0.5 }),
        circle('green', centerX, 0.13, 0.025),
        label('green', 0.55, 0.09, 'Stay here ✓', 11),
        label('yellow', 0.50, 0.05, 'Impact — head centered', 11),
      ],
    },
  ];
  return { keyframes: frames, phases, issueType: 'head_movement' };
}

/* ═════════════════════════════════════
   WEIGHT SHIFT (Reverse Pivot)
   Issue: weight stays on trail foot
   ═════════════════════════════════════ */
function weightShift(): OverlayTimeline {
  const phases: PhaseMarkers = {
    address: 0.0, takeaway: 0.15, top: 0.50, transition: 0.62, impact: 0.75, finish: 0.92,
  };
  const frames: OverlayKeyframe[] = [
    {
      t: 0.0, phase: 'address',
      elements: [
        ...buildSkeleton('white', [0.50,0.13],[0.38,0.28],[0.62,0.28],[0.42,0.54],[0.58,0.54]),
        // Foot markers
        circle('white', 0.40, 0.78, 0.025, 0.5),
        circle('white', 0.60, 0.78, 0.025, 0.5),
        label('white', 0.40, 0.83, 'Lead', 10),
        label('white', 0.60, 0.83, 'Trail', 10),
        label('white', 0.50, 0.05, 'Address', 11),
      ],
    },
    {
      t: 0.50, phase: 'top',
      elements: [
        ...buildSkeleton('white', [0.48,0.12],[0.32,0.30],[0.66,0.24],[0.40,0.53],[0.60,0.53]),
        // RED: weight stuck on trail side (body tilts wrong way)
        circle('red', 0.60, 0.78, 0.035, 0.7), // heavy trail foot
        circle('green', 0.40, 0.78, 0.022, 0.5), // some lead foot
        label('red', 0.62, 0.83, '80%', 11),
        label('yellow', 0.50, 0.05, 'Top — weight should shift right', 11),
      ],
    },
    {
      t: 0.75, phase: 'impact',
      elements: [
        ...buildSkeleton('white', [0.48,0.13],[0.30,0.25],[0.60,0.33],[0.40,0.52],[0.56,0.52]),
        // GREEN: weight on lead foot
        circle('green', 0.40, 0.78, 0.038, 0.8),
        circle('red', 0.60, 0.78, 0.022, 0.4),
        // Arrow showing weight shift direction
        arrow('green', 0.60, 0.72, 0.40, 0.72, 0.9),
        label('green', 0.38, 0.83, '70% ✓', 11),
        label('red', 0.60, 0.83, '30%', 10),
        label('yellow', 0.50, 0.05, 'IMPACT — shift to lead side!', 11),
      ],
    },
    {
      t: 0.92, phase: 'finish',
      elements: [
        ...buildSkeleton('white', [0.50,0.14],[0.50,0.30],[0.70,0.22],[0.45,0.53],[0.62,0.50]),
        circle('green', 0.40, 0.78, 0.042, 0.8),
        label('green', 0.38, 0.83, 'Lead foot', 10),
        label('white', 0.50, 0.05, 'Finish — full weight transfer', 11),
      ],
    },
  ];
  return { keyframes: frames, phases, issueType: 'weight_shift_issue' };
}

/* ═════════════════════════════════════
   HAND PATH (Loop away from body)
   Issue: hands cast/loop outward
   ═════════════════════════════════════ */
function handPath(): OverlayTimeline {
  const phases: PhaseMarkers = {
    address: 0.0, takeaway: 0.15, top: 0.50, transition: 0.62, impact: 0.75, finish: 0.92,
  };
  const frames: OverlayKeyframe[] = [
    {
      t: 0.0, phase: 'address',
      elements: [
        ...buildSkeleton('white', [0.50,0.13],[0.38,0.28],[0.62,0.28],[0.42,0.54],[0.58,0.54]),
        circle('white', 0.40, 0.42, 0.02),
        label('white', 0.50, 0.05, 'Address', 11),
      ],
    },
    {
      t: 0.50, phase: 'top',
      elements: [
        ...buildSkeleton('white', [0.48,0.12],[0.32,0.30],[0.66,0.24],[0.40,0.53],[0.60,0.53]),
        // Hands at top
        circle('white', 0.32, 0.20, 0.022),
        label('white', 0.50, 0.05, 'Top', 11),
      ],
    },
    {
      t: 0.62, phase: 'transition',
      elements: [
        ...buildSkeleton('white', [0.49,0.12],[0.35,0.27],[0.63,0.29],[0.41,0.53],[0.58,0.53]),
        // RED: hands loop outside
        line('red', 0.32, 0.20, 0.52, 0.42, { strokeWidth: 3 }),
        circle('red', 0.52, 0.42, 0.022),
        // GREEN: hands drop inside
        line('green', 0.32, 0.20, 0.40, 0.38, { strokeWidth: 3, dashed: true }),
        circle('green', 0.40, 0.38, 0.018),
        arrow('yellow', 0.52, 0.42, 0.40, 0.38, 0.9),
        label('red', 0.55, 0.43, 'Away from body', 11),
        label('green', 0.25, 0.38, 'Drop inside', 11),
        label('yellow', 0.50, 0.05, 'Transition — hand path', 11),
      ],
    },
    {
      t: 0.75, phase: 'impact',
      elements: [
        ...buildSkeleton('white', [0.48,0.13],[0.30,0.25],[0.60,0.33],[0.40,0.52],[0.56,0.52]),
        // Path traces
        line('red', 0.32, 0.20, 0.52, 0.42, { strokeWidth: 2, opacity: 0.5 }),
        line('red', 0.52, 0.42, 0.46, 0.58, { strokeWidth: 3 }),
        line('green', 0.32, 0.20, 0.40, 0.38, { strokeWidth: 2, dashed: true, opacity: 0.5 }),
        line('green', 0.40, 0.38, 0.42, 0.55, { strokeWidth: 3, dashed: true }),
        label('red', 0.55, 0.55, 'Outside path', 11),
        label('green', 0.22, 0.55, 'Inside ✓', 11),
        label('yellow', 0.50, 0.05, 'IMPACT — path difference', 11),
      ],
    },
  ];
  return { keyframes: frames, phases, issueType: 'hand_path_issue' };
}

/* ═════════════════════════════════════
   STEEP BACKSWING PLANE
   ═════════════════════════════════════ */
function steepBackswing(): OverlayTimeline {
  const phases: PhaseMarkers = {
    address: 0.0, takeaway: 0.15, top: 0.50, transition: 0.62, impact: 0.75, finish: 0.92,
  };
  const frames: OverlayKeyframe[] = [
    {
      t: 0.0, phase: 'address',
      elements: [
        ...buildSkeleton('white', [0.50,0.13],[0.38,0.28],[0.62,0.28],[0.42,0.54],[0.58,0.54]),
        label('white', 0.50, 0.05, 'Address — watch backswing plane', 11),
      ],
    },
    {
      t: 0.25, phase: 'takeaway',
      elements: [
        ...buildSkeleton('white', [0.50,0.13],[0.36,0.27],[0.63,0.27],[0.42,0.54],[0.58,0.54]),
        // Club: RED too steep
        line('red', 0.46, 0.40, 0.30, 0.10, { strokeWidth: 3.5 }),
        circle('red', 0.30, 0.09, 0.022),
        // Club: GREEN flatter
        line('green', 0.46, 0.40, 0.22, 0.20, { strokeWidth: 3, dashed: true }),
        circle('green', 0.21, 0.19, 0.018),
        label('red', 0.22, 0.07, 'Too vertical ↑', 11),
        label('green', 0.10, 0.20, 'Flatter ✓', 11),
        label('yellow', 0.50, 0.05, 'Early takeaway — plane!', 11),
      ],
    },
    {
      t: 0.50, phase: 'top',
      elements: [
        ...buildSkeleton('white', [0.48,0.12],[0.32,0.30],[0.66,0.24],[0.40,0.53],[0.60,0.53]),
        // RED: steep plane
        line('red', 0.50, 0.40, 0.22, 0.08, { strokeWidth: 3.5 }),
        circle('red', 0.21, 0.07, 0.022),
        // GREEN: on-plane
        line('green', 0.50, 0.40, 0.16, 0.18, { strokeWidth: 3, dashed: true }),
        circle('green', 0.15, 0.17, 0.018),
        // Angle arc indicator
        label('red', 0.20, 0.05, 'Steep ↑', 11),
        label('green', 0.06, 0.18, 'On plane ✓', 11),
        label('yellow', 0.50, 0.05, 'Top — plane comparison', 11),
      ],
    },
  ];
  return { keyframes: frames, phases, issueType: 'steep_backswing_plane' };
}

/* ═════════════════════════════════
   MAIN EXPORT
   ═════════════════════════════════ */
export type SwingIssueTypeKey =
  | 'early_extension'
  | 'steep_downswing'
  | 'head_movement'
  | 'weight_shift_issue'
  | 'hand_path_issue'
  | 'steep_backswing_plane';

export function generateOverlayTimeline(issueType: SwingIssueTypeKey): OverlayTimeline {
  switch (issueType) {
    case 'early_extension': return earlyExtension();
    case 'steep_downswing': return steepDownswing();
    case 'head_movement': return headMovement();
    case 'weight_shift_issue': return weightShift();
    case 'hand_path_issue': return handPath();
    case 'steep_backswing_plane': return steepBackswing();
    default: return earlyExtension();
  }
}
