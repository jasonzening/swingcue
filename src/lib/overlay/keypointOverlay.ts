/**
 * keypointOverlay.ts
 *
 * 旋转盘核心设计原则：
 *   - 圆盘是肩膀/髋部连线旋转形成的平面
 *   - 连线（轴线）是核心，穿过圆盘中心延伸出去
 *   - 红绿两个圆盘大小完全相同，只有旋转角度不同
 *   - 红色 = 用户当前角度，绿色 = 正确水平角度
 */

import type { OverlayElement, KeypointFrame } from '@/types/analysis';
import type { MainIssueType } from '@/types/analysis';
import type { BodyPointName, Pt } from '@/lib/golf/bodyPointSpec';
import type { ViewType } from '@/lib/golf/overlayLineSpec';

type Color = 'red' | 'green' | 'yellow' | 'white';

let _uid = 0;
const uid = (p: string) => `${p}-${++_uid}`;

const mkDot = (x: number, y: number, color: Color, r = 0.010, opacity = 0.95, layer: OverlayElement['layer'] = 'body'): OverlayElement =>
  ({ type: 'dot', id: uid('d'), x, y, color, radius: r, opacity, layer });

const mkLine = (x1: number, y1: number, x2: number, y2: number, color: Color, w = 2.5, opacity = 0.92, dashed = false, layer: OverlayElement['layer'] = 'body'): OverlayElement =>
  ({ type: 'line', id: uid('l'), x1, y1, x2, y2, color, strokeWidth: w, opacity, dashed, layer });

const mkLabel = (x: number, y: number, text: string, color: Color = 'white', size = 10): OverlayElement =>
  ({ type: 'label', id: uid('t'), x, y, text, color, size, opacity: 0.92 });

const mkEllipse = (cx: number, cy: number, rx: number, ry: number, angle: number, color: Color, w = 3.5, opacity = 0.92, layer: OverlayElement['layer'] = 'body'): OverlayElement =>
  ({ type: 'ellipse' as OverlayElement['type'], id: uid('e'), cx, cy, rx, ry, angle, color, strokeWidth: w, opacity, layer } as unknown as OverlayElement);

export function getKeypoints(kpFrame: KeypointFrame): Partial<Record<BodyPointName, Pt>> {
  const lm = kpFrame.landmarks;
  const r: Partial<Record<BodyPointName, Pt>> = {};
  const toP = (pt?: { x:number; y:number; confidence?:number } | null) =>
    pt ? { x: pt.x, y: pt.y, confidence: pt.confidence ?? 0.8 } : null;
  if (lm.head)          r.headCenter     = toP(lm.head)!;
  if (lm.leftShoulder)  r.leftShoulder   = toP(lm.leftShoulder)!;
  if (lm.rightShoulder) r.rightShoulder  = toP(lm.rightShoulder)!;
  if (lm.leftElbow)     r.leftElbow      = toP(lm.leftElbow)!;
  if (lm.rightElbow)    r.rightElbow     = toP(lm.rightElbow)!;
  if (lm.leftWrist)     r.leftWrist      = toP(lm.leftWrist)!;
  if (lm.rightWrist)    r.rightWrist     = toP(lm.rightWrist)!;
  if (lm.leftHip)       r.leftHip        = toP(lm.leftHip)!;
  if (lm.rightHip)      r.rightHip       = toP(lm.rightHip)!;
  if (lm.leftKnee)      r.leftKnee       = toP(lm.leftKnee)!;
  if (lm.rightKnee)     r.rightKnee      = toP(lm.rightKnee)!;
  if (lm.leftAnkle)     r.leftAnkle      = toP(lm.leftAnkle)!;
  if (lm.rightAnkle)    r.rightAnkle     = toP(lm.rightAnkle)!;
  const ls = r.leftShoulder, rs = r.rightShoulder;
  if (ls && rs) r.shoulderCenter = { x:(ls.x+rs.x)/2, y:(ls.y+rs.y)/2, confidence:1 };
  const lh = r.leftHip, rh = r.rightHip;
  if (lh && rh) r.hipCenter = { x:(lh.x+rh.x)/2, y:(lh.y+rh.y)/2, confidence:1 };
  const lw = r.leftWrist, rw = r.rightWrist;
  if (lw && rw) r.gripCenter = { x:(lw.x+rw.x)/2, y:(lw.y+rw.y)/2, confidence:1 };
  return r;
}

/**
 * buildRotationDisc
 *
 * 肩膀线是核心：
 *   - 以两端点为轴，椭圆表示旋转平面
 *   - 红色和绿色圆盘大小完全一样
 *   - 绿色只是更接近水平（正确旋转角度）
 *   - 轴线（穿越线）比圆盘长，是视觉核心
 */
function buildRotationDisc(
  leftPt: Pt,
  rightPt: Pt,
  label: string,
  layer: OverlayElement['layer'] = 'body',
): OverlayElement[] {
  const els: OverlayElement[] = [];

  // 中心点与几何参数
  const cx = (leftPt.x + rightPt.x) / 2;
  const cy = (leftPt.y + rightPt.y) / 2;
  const dx = rightPt.x - leftPt.x;
  const dy = rightPt.y - leftPt.y;
  const angle = Math.atan2(dy, dx); // 实际旋转角度
  const bodyWidth = Math.hypot(dx, dy);

  // 圆盘尺寸（红绿完全相同）
  const rx = bodyWidth * 1.08;  // 比肩宽大8%，延伸到两侧外
  const ry = rx * 0.34;         // 厚度

  // 轴线长度 = 比圆盘长30%（穿越并延伸出去）
  const lineExt = rx * 1.32;
  const cosA = Math.cos(angle), sinA = Math.sin(angle);

  // ── 🔴 红色：用户当前位置 ──
  // 轴线（先画，视觉核心）
  els.push(mkLine(
    cx - lineExt * cosA, cy - lineExt * sinA,
    cx + lineExt * cosA, cy + lineExt * sinA,
    'red', 2.5, 0.92, false, layer,
  ));
  // 圆盘
  els.push(mkEllipse(cx, cy, rx, ry, angle, 'red', 3.5, 0.90, layer));
  // 两端关节点
  els.push(mkDot(leftPt.x,  leftPt.y,  'red', 0.011, 0.96, layer));
  els.push(mkDot(rightPt.x, rightPt.y, 'red', 0.011, 0.96, layer));

  // ── 🟢 绿色：正确目标位置（同样大小，更水平）──
  // 目标角度：从最近水平轴计算，减少65%倾斜
  const nearestH = Math.round(angle / Math.PI) * Math.PI;
  const tilt = angle - nearestH;
  const targetAngle = nearestH + tilt * 0.35; // 比当前更平

  // 绿色圆盘与红色大小完全相同
  // 位置：沿垂直方向偏移一点点，让两个圆盘能被区分
  const perpOffset = ry * 0.40; // 沿法线方向偏移
  const perpCos = -sinA, perpSin = cosA;
  const tcx = cx + perpCos * perpOffset;
  const tcy = cy + perpSin * perpOffset;

  const tCosA = Math.cos(targetAngle), tSinA = Math.sin(targetAngle);
  // 绿色轴线
  els.push(mkLine(
    tcx - lineExt * tCosA, tcy - lineExt * tSinA,
    tcx + lineExt * tCosA, tcy + lineExt * tSinA,
    'green', 2.2, 0.85, false, layer,
  ));
  // 绿色圆盘（大小与红色完全一致）
  els.push(mkEllipse(tcx, tcy, rx, ry, targetAngle, 'green', 3.0, 0.82, layer));

  // 倾斜角度标注（从水平轴的倾斜度）
  const tiltDeg = Math.round(Math.abs(tilt * 180 / Math.PI));
  els.push(mkLabel(cx, cy - ry * 1.55, label + '  ' + tiltDeg + '°', 'white', 10));

  return els;
}

export function generateSpecDrivenOverlayFrame(
  issue: MainIssueType,
  viewType: ViewType,
  phase: string,
  kpFrame: KeypointFrame,
  historyPts?: Array<{ x: number; y: number }>,
): OverlayElement[] {
  _uid = 0;
  const pts = getKeypoints(kpFrame);
  const els: OverlayElement[] = [];

  const ls = pts.leftShoulder, rs = pts.rightShoulder;
  if (ls && rs) els.push(...buildRotationDisc(ls, rs, 'SHOULDERS', 'body'));

  const lh = pts.leftHip, rh = pts.rightHip;
  if (lh && rh) els.push(...buildRotationDisc(lh, rh, 'HIPS', 'club'));

  return els;
}

export function getTrackedPoint(
  _issue: MainIssueType, _viewType: ViewType, kpFrame: KeypointFrame,
): { x: number; y: number } | null {
  const pts = getKeypoints(kpFrame);
  const sc = pts.shoulderCenter;
  return sc ? { x: sc.x, y: sc.y } : null;
}

export function findNearestFrame(frames: KeypointFrame[], time: number): KeypointFrame | null {
  if (!frames.length) return null;
  let best = frames[0], bestDist = Math.abs(time - best.time);
  for (const f of frames) {
    const d = Math.abs(time - f.time);
    if (d < bestDist) { best = f; bestDist = d; }
  }
  return best;
}

export function applyCorrection(pt: { x:number; y:number; confidence?:number }, dir: string, mag = 'medium'): { x:number; y:number; confidence?:number } {
  const DELTA: Record<string, number> = { small: 0.028, medium: 0.050, large: 0.078 };
  const d = DELTA[mag] ?? 0.050;
  switch (dir) {
    case 'lower':         return { ...pt, y: pt.y + d };
    case 'higher':        return { ...pt, y: pt.y - d };
    case 'more_inside':   return { ...pt, x: pt.x - d };
    case 'more_outside':  return { ...pt, x: pt.x + d };
    case 'more_centered': return { ...pt, x: 0.50 };
    default:              return pt;
  }
}
