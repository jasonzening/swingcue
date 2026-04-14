/**
 * keypointOverlay.ts
 *
 * 将真实 MediaPipe 关键点数据转换为 OverlayElement[]。
 *
 * 核心逻辑：
 *   红色当前点/线 = 来自真实 keypoints（用户实际身体位置）
 *   绿色目标点/线 = 真实 keypoints + TargetCorrection 修正量
 *   黄色箭头      = 从红点指向绿点（告诉用户往哪里动）
 *
 * 坐标系：归一化 0-1（相对视频宽高）
 */

import type {
  OverlayElement, BodyLandmarks, KeypointFrame,
} from '@/types/analysis';
import type { TargetCorrection, BodyPartKey, CorrectionDirection } from '@/lib/golf/types';

type Pt = { x: number; y: number };

/* ══════════════════════════════════════════
   修正量映射（归一化坐标系）
   注意：y 轴向下 (0=top, 1=bottom)
   "lower" = y 增加 = 往画面下方
══════════════════════════════════════════ */
const MAGNITUDE_DELTA = {
  small:  0.030,
  medium: 0.052,
  large:  0.080,
} as const;

export function applyCorrection(
  current: Pt,
  direction: CorrectionDirection,
  magnitude: 'small' | 'medium' | 'large'
): Pt {
  const d = MAGNITUDE_DELTA[magnitude];
  const { x, y } = current;

  switch (direction) {
    case 'lower':          return { x,         y: y + d };
    case 'higher':         return { x,         y: y - d };
    case 'more_inside':    return { x: x - d,  y };        // lead side (frame left for face-on RH)
    case 'more_outside':   return { x: x + d,  y };
    case 'more_forward':   return { x: x + d * 0.6, y };
    case 'more_back':      return { x: x - d * 0.6, y };
    case 'more_turned':    return { x: x + d,  y };
    case 'less_turned':    return { x: x - d,  y };
    case 'shallower':      return { x,         y: y + d * 0.5 };  // flatter path
    case 'steeper':        return { x,         y: y - d * 0.5 };
    case 'more_centered':  return { x: 0.50,   y };               // center of frame
    case 'more_stable':    return current;                         // no positional delta
    default:               return current;
  }
}

/* ══════════════════════════════════════════
   BodyPartKey → BodyLandmarks 关键点
══════════════════════════════════════════ */
function avg(a?: Pt | null, b?: Pt | null): Pt | null {
  if (!a || !b) return a ?? b ?? null;
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

export function getBodyPartPoint(
  bodyPart: BodyPartKey,
  landmarks: BodyLandmarks
): Pt | null {
  const lm = landmarks;
  switch (bodyPart) {
    case 'head':         return lm.head ?? null;
    case 'shoulders':    return avg(lm.leftShoulder, lm.rightShoulder);
    case 'trailShoulder':return lm.rightShoulder ?? null;   // RH: trail = anatomical right
    case 'leadShoulder': return lm.leftShoulder ?? null;    // RH: lead  = anatomical left
    case 'hands':        return avg(lm.leftWrist, lm.rightWrist);
    case 'trailHand':    return lm.rightWrist ?? null;
    case 'leadHand':     return lm.leftWrist ?? null;
    case 'hips': {
      const midHip = avg(lm.leftHip, lm.rightHip);
      return midHip;
    }
    case 'spine': {
      // Midpoint between shoulder center and hip center
      const midS = avg(lm.leftShoulder, lm.rightShoulder);
      const midH = avg(lm.leftHip, lm.rightHip);
      return avg(midS, midH);
    }
    case 'weight':       return avg(lm.leftAnkle, lm.rightAnkle);
    case 'club':         return null;  // Phase 2: club tracking
    default:             return null;
  }
}

/* ══════════════════════════════════════════
   单个 TargetCorrection → OverlayElements
══════════════════════════════════════════ */
let _idCounter = 0;
function uid(prefix: string) { return `${prefix}-${++_idCounter}`; }

export function correctionToElements(
  correction: TargetCorrection,
  landmarks: BodyLandmarks,
  conf: number = 0.9
): OverlayElement[] {
  const current = getBodyPartPoint(correction.bodyPart, landmarks);
  if (!current) return [];

  // Skip very low confidence points
  const dotSize = 0.026;
  const elements: OverlayElement[] = [];

  const target = applyCorrection(current, correction.direction, correction.magnitude);

  // Red dot = current (wrong) position
  elements.push({
    type: 'dot',
    id: uid('red'),
    x: current.x, y: current.y,
    color: 'red',
    radius: dotSize,
    opacity: 0.92,
    layer: 'body',
  });

  // Green dot = target (correct) position
  elements.push({
    type: 'dot',
    id: uid('grn'),
    x: target.x, y: target.y,
    color: 'green',
    radius: dotSize * 0.88,
    opacity: 0.88,
    layer: 'body',
  });

  // Yellow arrow: current → target (only if meaningful distance)
  const dist = Math.hypot(target.x - current.x, target.y - current.y);
  if (dist > 0.015) {
    elements.push({
      type: 'arrow',
      id: uid('arr'),
      from: { x: current.x, y: current.y },
      to:   { x: target.x,  y: target.y },
      color: 'yellow',
      strokeWidth: 2.5,
      opacity: 0.88,
      layer: 'body',
    });
  }

  // Optional short label (only for "more_stable" which has no arrow)
  if (correction.direction === 'more_stable') {
    elements.push({
      type: 'label',
      id: uid('lbl'),
      x: current.x, y: current.y - 0.05,
      text: 'Hold position',
      color: 'green',
      size: 10,
      opacity: 0.75,
      layer: 'body',
    });
  }

  return elements;
}

/* ══════════════════════════════════════════
   Structure lines from real keypoints
   (肩线、髋线、脊柱线 — 来自真实坐标)
══════════════════════════════════════════ */
export function buildSkeletonLines(
  landmarks: BodyLandmarks,
  color: 'red' | 'green' | 'white',
  opacity: number = 0.80,
  dashed: boolean = false
): OverlayElement[] {
  const lm = landmarks;
  const elements: OverlayElement[] = [];
  const lw = 2.5;
  const opt = { strokeWidth: lw, opacity, dashed, layer: 'body' as const };

  const lS = lm.leftShoulder;
  const rS = lm.rightShoulder;
  const lH = lm.leftHip;
  const rH = lm.rightHip;
  const lK = lm.leftKnee;
  const rK = lm.rightKnee;

  // Shoulder line
  if (lS && rS) {
    elements.push({
      type: 'line', id: uid('sh'), color,
      x1: lS.x, y1: lS.y, x2: rS.x, y2: rS.y, ...opt,
    });
    // Shoulder dots
    elements.push({ type: 'dot', id: uid('lsd'), color, x: lS.x, y: lS.y, radius: 0.022, opacity, layer: 'body' });
    elements.push({ type: 'dot', id: uid('rsd'), color, x: rS.x, y: rS.y, radius: 0.022, opacity, layer: 'body' });
  }

  // Hip line
  if (lH && rH) {
    elements.push({
      type: 'line', id: uid('hi'), color,
      x1: lH.x, y1: lH.y, x2: rH.x, y2: rH.y,
      ...opt, strokeWidth: lw * 0.85, opacity: opacity * 0.85,
    });
  }

  // Spine (shoulder mid → hip mid)
  const midS = lS && rS ? { x: (lS.x + rS.x) / 2, y: (lS.y + rS.y) / 2 } : null;
  const midH = lH && rH ? { x: (lH.x + rH.x) / 2, y: (lH.y + rH.y) / 2 } : null;
  if (midS && midH) {
    elements.push({
      type: 'line', id: uid('sp'), color: 'white',
      x1: midS.x, y1: midS.y, x2: midH.x, y2: midH.y,
      strokeWidth: lw * 1.2, opacity: opacity * 0.70, dashed: false, layer: 'body',
    });
  }

  // Upper legs (subtle)
  if (lH && lK) {
    elements.push({ type: 'line', id: uid('ll'), color, x1: lH.x, y1: lH.y, x2: lK.x, y2: lK.y, ...opt, strokeWidth: 1.5, opacity: opacity * 0.45 });
    elements.push({ type: 'line', id: uid('rl'), color, x1: rH!.x, y1: rH!.y, x2: rK!.x, y2: rK!.y, ...opt, strokeWidth: 1.5, opacity: opacity * 0.45 });
  }

  return elements;
}

/* ══════════════════════════════════════════
   Arm chain from real keypoints
══════════════════════════════════════════ */
export function buildArmLines(landmarks: BodyLandmarks): OverlayElement[] {
  const lm = landmarks;
  const elements: OverlayElement[] = [];

  // Lead arm (green)
  if (lm.leftShoulder && lm.leftElbow && lm.leftWrist) {
    elements.push({ type: 'line', id: uid('la1'), color: 'green', x1: lm.leftShoulder.x, y1: lm.leftShoulder.y, x2: lm.leftElbow.x, y2: lm.leftElbow.y, strokeWidth: 2.8, opacity: 0.85, layer: 'arms' });
    elements.push({ type: 'line', id: uid('la2'), color: 'green', x1: lm.leftElbow.x, y1: lm.leftElbow.y, x2: lm.leftWrist.x, y2: lm.leftWrist.y, strokeWidth: 2.8, opacity: 0.85, layer: 'arms' });
    elements.push({ type: 'dot', id: uid('le'), color: 'green', x: lm.leftElbow.x, y: lm.leftElbow.y, radius: 0.018, opacity: 0.85, layer: 'arms' });
    elements.push({ type: 'dot', id: uid('lw'), color: 'green', x: lm.leftWrist.x, y: lm.leftWrist.y, radius: 0.024, opacity: 0.95, layer: 'arms' });
  }

  // Trail arm (red)
  if (lm.rightShoulder && lm.rightElbow && lm.rightWrist) {
    elements.push({ type: 'line', id: uid('ra1'), color: 'red', x1: lm.rightShoulder.x, y1: lm.rightShoulder.y, x2: lm.rightElbow.x, y2: lm.rightElbow.y, strokeWidth: 2.8, opacity: 0.85, layer: 'arms' });
    elements.push({ type: 'line', id: uid('ra2'), color: 'red', x1: lm.rightElbow.x, y1: lm.rightElbow.y, x2: lm.rightWrist.x, y2: lm.rightWrist.y, strokeWidth: 2.8, opacity: 0.85, layer: 'arms' });
    elements.push({ type: 'dot', id: uid('re'), color: 'red', x: lm.rightElbow.x, y: lm.rightElbow.y, radius: 0.018, opacity: 0.85, layer: 'arms' });
    elements.push({ type: 'dot', id: uid('rw'), color: 'red', x: lm.rightWrist.x, y: lm.rightWrist.y, radius: 0.024, opacity: 0.95, layer: 'arms' });
  }

  // Grip V-shape (both shoulders → grip midpoint)
  const lW = lm.leftWrist;
  const rW = lm.rightWrist;
  const lS = lm.leftShoulder;
  const rS = lm.rightShoulder;
  if (lW && rW && lS && rS) {
    const grip = { x: (lW.x + rW.x) / 2, y: (lW.y + rW.y) / 2 };
    elements.push({ type: 'line', id: uid('vl'), color: 'green', x1: lS.x, y1: lS.y, x2: grip.x, y2: grip.y, strokeWidth: 2, opacity: 0.65, layer: 'arms' });
    elements.push({ type: 'line', id: uid('vr'), color: 'green', x1: rS.x, y1: rS.y, x2: grip.x, y2: grip.y, strokeWidth: 2, opacity: 0.65, layer: 'arms' });
    elements.push({ type: 'dot', id: uid('grip'), color: 'green', x: grip.x, y: grip.y, radius: 0.028, opacity: 0.95, layer: 'arms' });
  }

  return elements;
}

/* ══════════════════════════════════════════
   MAIN GENERATOR
   Given a list of corrections + a keypoint frame,
   generate the full set of overlay elements.
══════════════════════════════════════════ */
export function generateKeypointOverlayFrame(
  corrections: TargetCorrection[],
  kpFrame: KeypointFrame,
  includeSkeletonLines: boolean = true,
  includeArmLines: boolean = true,
): OverlayElement[] {
  _idCounter = 0; // Reset per frame
  const elements: OverlayElement[] = [];
  const { landmarks } = kpFrame;

  // 1. Base skeleton structure (real positions)
  if (includeSkeletonLines) {
    elements.push(...buildSkeletonLines(landmarks, 'white', 0.65, false));
  }

  // 2. Arm chains (real positions)
  if (includeArmLines) {
    elements.push(...buildArmLines(landmarks));
  }

  // 3. Correction overlays (red current → arrow → green target)
  for (const correction of corrections) {
    elements.push(...correctionToElements(correction, landmarks));
  }

  return elements;
}

/* ══════════════════════════════════════════
   Find nearest keypoint frame at a given time
══════════════════════════════════════════ */
export function findNearestFrame(
  frames: KeypointFrame[],
  time: number
): KeypointFrame | null {
  if (!frames.length) return null;
  let best = frames[0];
  let bestDist = Math.abs(time - best.time);
  for (const f of frames) {
    const d = Math.abs(time - f.time);
    if (d < bestDist) { best = f; bestDist = d; }
  }
  return best;
}
