/**
 * keypointOverlay.ts — 红色旋转盘（极简版）
 *
 * 只有红色圆盘跟随身体：
 *   - 圆盘大小固定（不随转身变化）
 *   - 圆盘角度跟着肩膀/髋部连线实时变化
 *   - 轴线穿过两端点延伸到圆盘外侧
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

function buildDisc(
  leftPt: Pt, rightPt: Pt,
  fixedRx: number, fixedRy: number,
  label: string,
  layer: OverlayElement['layer'] = 'body',
): OverlayElement[] {
  const els: OverlayElement[] = [];

  const cx = (leftPt.x + rightPt.x) / 2;
  const cy = (leftPt.y + rightPt.y) / 2;
  const angle = Math.atan2(rightPt.y - leftPt.y, rightPt.x - leftPt.x);

  // 轴线：比圆盘长30%，穿越并延伸出去
  const lineExt = fixedRx * 1.32;
  const cosA = Math.cos(angle), sinA = Math.sin(angle);

  // 轴线（先画，圆盘盖在上面）
  els.push(mkLine(
    cx - lineExt * cosA, cy - lineExt * sinA,
    cx + lineExt * cosA, cy + lineExt * sinA,
    'red', 2.5, 0.90, false, layer,
  ));

  // 红色圆盘
  els.push(mkEllipse(cx, cy, fixedRx, fixedRy, angle, 'red', 3.5, 0.90, layer));

  // 两端关节点
  els.push(mkDot(leftPt.x,  leftPt.y,  'red', 0.011, 0.96, layer));
  els.push(mkDot(rightPt.x, rightPt.y, 'red', 0.011, 0.96, layer));

  // 倾斜角标注
  const nearestH = Math.round(angle / Math.PI) * Math.PI;
  const tiltDeg = Math.round(Math.abs((angle - nearestH) * 180 / Math.PI));
  els.push(mkLabel(cx, cy - fixedRy * 1.55, label + '  ' + tiltDeg + '°', 'white', 10));

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

  // 固定圆盘尺寸（视频宽度的比例，不随转身变化）
  const SHOULDER_RX = 0.26;
  const SHOULDER_RY = SHOULDER_RX * 0.35;
  const HIP_RX = 0.20;
  const HIP_RY = HIP_RX * 0.35;

  const ls = pts.leftShoulder, rs = pts.rightShoulder;
  if (ls && rs) els.push(...buildDisc(ls, rs, SHOULDER_RX, SHOULDER_RY, 'SHOULDERS', 'body'));

  const lh = pts.leftHip, rh = pts.rightHip;
  if (lh && rh) els.push(...buildDisc(lh, rh, HIP_RX, HIP_RY, 'HIPS', 'club'));

  return els;
}

export function getTrackedPoint(
  _issue: MainIssueType, _viewType: ViewType, kpFrame: KeypointFrame,
): { x: number; y: number } | null {
  const pts = getKeypoints(kpFrame);
  return pts.shoulderCenter ? { x: pts.shoulderCenter.x, y: pts.shoulderCenter.y } : null;
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
