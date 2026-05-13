/**
 * keypointOverlay.ts — 大椭圆旋转盘，参考图风格
 * 肩部红色大椭圆（延伸到身体外）+ 绿色目标椭圆
 * 髋部黄色椭圆 + 绿色目标
 * 头部大圆圈 + 轨迹
 */

import type { OverlayElement, KeypointFrame } from '@/types/analysis';
import type { MainIssueType } from '@/types/analysis';
import type { BodyPointName, Pt } from '@/lib/golf/bodyPointSpec';
import type { ViewType } from '@/lib/golf/overlayLineSpec';

type Color = 'red' | 'green' | 'yellow' | 'white';

let _uid = 0;
const uid = (p: string) => `${p}-${++_uid}`;
const mkDot   = (x: number, y: number, color: Color, r = 0.007, opacity = 0.92, layer: OverlayElement['layer'] = 'body'): OverlayElement =>
  ({ type: 'dot',   id: uid('d'), x, y, color, radius: r, opacity, layer });
const mkLine  = (x1: number, y1: number, x2: number, y2: number, color: Color, w = 1.0, opacity = 0.85, dashed = false, layer: OverlayElement['layer'] = 'body'): OverlayElement =>
  ({ type: 'line',  id: uid('l'), x1, y1, x2, y2, color, strokeWidth: w, opacity, dashed, layer });
const mkArrow = (fx: number, fy: number, tx: number, ty: number, color: Color = 'red'): OverlayElement =>
  ({ type: 'arrow', id: uid('a'), from: { x: fx, y: fy }, to: { x: tx, y: ty }, color, strokeWidth: 1.2, opacity: 0.90 });
const mkLabel = (x: number, y: number, text: string, color: Color = 'white', size = 10): OverlayElement =>
  ({ type: 'label', id: uid('t'), x, y, text, color, size, opacity: 0.92 });
const mkCurve = (points: {x:number;y:number}[], color: Color, w = 1.2, opacity = 0.80): OverlayElement =>
  ({ type: 'curve', id: uid('c'), points, color, strokeWidth: w, opacity, layer: 'arms' });
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

// 肩部旋转盘
function buildShoulderLayer(pts: Partial<Record<BodyPointName, Pt>>, historyPts?: Array<{x:number;y:number}>): OverlayElement[] {
  const els: OverlayElement[] = [];
  const ls = pts.leftShoulder, rs = pts.rightShoulder;
  if (!ls || !rs) return els;
  const cx = (ls.x+rs.x)/2, cy = (ls.y+rs.y)/2;
  const dx = rs.x-ls.x, dy = rs.y-ls.y;
  const angle = Math.atan2(dy, dx);
  const sw = Math.hypot(dx, dy);
  // 椭圆尺寸：长轴 = 肩宽 × 1.1（大幅延伸到身体两侧外）
  const rx = sw * 1.1;
  const ry = rx * 0.38; // 厚度

  // 🔴 红色大椭圆 = 当前肩部旋转平面
  els.push(mkEllipse(cx, cy, rx, ry, angle, 'red', 3.5, 0.92, 'body'));
  // 🟢 绿色目标椭圆（更平 = 角度更小）
  els.push(mkEllipse(cx, cy-0.018, rx*1.06, ry*0.82, angle*0.30, 'green', 3.0, 0.82, 'body'));
  // 白色水平参考虚线（横穿椭圆中心）
  els.push(mkLine(cx-rx*1.25, cy, cx+rx*1.25, cy, 'white', 1.0, 0.50, true, 'body'));
  // 端点
  els.push(mkDot(ls.x, ls.y, 'red', 0.011, 0.95));
  els.push(mkDot(rs.x, rs.y, 'red', 0.011, 0.95));
  // 旋转角度标注
  const deg = Math.round(Math.abs(angle * 180 / Math.PI));
  els.push(mkLabel(cx, cy - ry*1.4 - 0.02, `SHOULDERS  ${deg}°`, 'white', 10));
  if (historyPts && historyPts.length >= 2) els.push(mkCurve(historyPts, 'red', 1.0, 0.60));
  return els;
}

// 髋部旋转盘（黄色）
function buildHipLayer(pts: Partial<Record<BodyPointName, Pt>>, shoulderAngle: number): OverlayElement[] {
  const els: OverlayElement[] = [];
  const lh = pts.leftHip, rh = pts.rightHip;
  if (!lh || !rh) return els;
  const cx = (lh.x+rh.x)/2, cy = (lh.y+rh.y)/2;
  const dx = rh.x-lh.x, dy = rh.y-lh.y;
  const angle = Math.atan2(dy, dx);
  const hw = Math.hypot(dx, dy);
  const rx = hw * 1.0;
  const ry = rx * 0.32;

  // 🟡 黄色大椭圆 = 当前髋部
  els.push(mkEllipse(cx, cy, rx, ry, angle, 'yellow', 3.5, 0.92, 'body'));
  // 🟢 绿色目标
  els.push(mkEllipse(cx, cy-0.012, rx*1.08, ry*0.75, angle*0.20, 'green', 2.8, 0.75, 'body'));
  // 参考线
  els.push(mkLine(cx-rx*1.25, cy, cx+rx*1.25, cy, 'white', 1.0, 0.45, true, 'body'));
  els.push(mkDot(lh.x, lh.y, 'yellow', 0.011, 0.92));
  els.push(mkDot(rh.x, rh.y, 'yellow', 0.011, 0.92));
  const hipDeg = Math.round(Math.abs(angle * 180 / Math.PI));
  const xFactor = Math.max(0, Math.round(Math.abs(shoulderAngle*180/Math.PI) - hipDeg));
  els.push(mkLabel(cx, cy - ry*1.4 - 0.02, `HIPS  ${hipDeg}°  X:${xFactor}°`, 'yellow', 10));
  return els;
}

// 头部大圆圈
function buildHeadLayer(pts: Partial<Record<BodyPointName, Pt>>, headHistory: Array<{x:number;y:number}>): OverlayElement[] {
  const els: OverlayElement[] = [];
  const head = pts.headCenter;
  if (!head) return els;
  const bx = headHistory.length > 0 ? headHistory[0].x : head.x;
  const by = headHistory.length > 0 ? headHistory[0].y : head.y;
  const r = 0.075;
  els.push(mkEllipse(bx, by, r, r, 0, 'green', 2.5, 0.75, 'body'));
  els.push(mkLine(bx-r*1.3, by, bx+r*1.3, by, 'green', 0.8, 0.40, true, 'body'));
  els.push(mkLine(bx, by-r*1.3, bx, by+r*1.3, 'green', 0.8, 0.40, true, 'body'));
  els.push(mkDot(head.x, head.y, 'red', 0.013, 0.95));
  if (headHistory.length >= 2) els.push(mkCurve(headHistory, 'red', 1.0, 0.75));
  const ox = head.x - bx;
  if (Math.abs(ox) > 0.012) {
    els.push(mkArrow(bx, head.y, head.x, head.y, 'red'));
    els.push(mkLabel(head.x, head.y-0.09, (ox>0?'→':'←')+' '+Math.round(Math.abs(ox)*100)+'%', 'red', 10));
  }
  els.push(mkLabel(bx, by-r-0.05, 'HEAD', 'white', 10));
  return els;
}

// 挥杆平面
function buildSwingPlaneLayer(pts: Partial<Record<BodyPointName, Pt>>, handHistory: Array<{x:number;y:number}>): OverlayElement[] {
  const els: OverlayElement[] = [];
  const grip = pts.gripCenter;
  const ls = pts.leftShoulder, rs = pts.rightShoulder;
  if (ls && rs) {
    const sc = { x:(ls.x+rs.x)/2, y:(ls.y+rs.y)/2 };
    els.push(mkLine(sc.x, sc.y, sc.x+0.02, sc.y+0.32, 'green', 1.5, 0.65, true, 'arms'));
    els.push(mkLabel(sc.x+0.08, sc.y+0.18, 'PLANE', 'green', 10));
    if (grip) {
      els.push(mkLine(ls.x, ls.y, grip.x, grip.y, 'red', 1.2, 0.82, false, 'arms'));
      els.push(mkLine(rs.x, rs.y, grip.x, grip.y, 'red', 1.2, 0.82, false, 'arms'));
      els.push(mkLine(ls.x, ls.y, rs.x, rs.y, 'red', 1.0, 0.72, false, 'arms'));
      const gGrip = { x: grip.x+(sc.x-grip.x)*0.12, y: grip.y-0.02 };
      els.push(mkLine(ls.x, ls.y, gGrip.x, gGrip.y, 'green', 1.0, 0.65, true, 'arms'));
      els.push(mkLine(rs.x, rs.y, gGrip.x, gGrip.y, 'green', 1.0, 0.65, true, 'arms'));
      els.push(mkDot(grip.x, grip.y, 'red', 0.012, 0.95, 'arms'));
      els.push(mkDot(gGrip.x, gGrip.y, 'green', 0.009, 0.82, 'arms'));
    }
  }
  if (handHistory.length >= 2) els.push(mkCurve(handHistory, 'red', 1.2, 0.82));
  if (ls) els.push(mkDot(ls.x, ls.y, 'red', 0.008, 0.88, 'arms'));
  if (rs) els.push(mkDot(rs.x, rs.y, 'red', 0.008, 0.88, 'arms'));
  return els;
}

export function generateSpecDrivenOverlayFrame(
  issue: MainIssueType, viewType: ViewType, phase: string,
  kpFrame: KeypointFrame, historyPts?: Array<{ x:number; y:number }>,
): OverlayElement[] {
  _uid = 0;
  const pts = getKeypoints(kpFrame);
  const els: OverlayElement[] = [];
  const ls = pts.leftShoulder, rs = pts.rightShoulder;
  let shoulderAngle = 0;
  if (ls && rs) shoulderAngle = Math.atan2(rs.y-ls.y, rs.x-ls.x);

  els.push(...buildShoulderLayer(pts, historyPts));
  els.push(...buildHipLayer(pts, shoulderAngle));
  els.push(...buildHeadLayer(pts, historyPts ? historyPts.slice(-10) : []));
  els.push(...buildSwingPlaneLayer(pts, historyPts || []));

  // 腿部辅助
  const lh = pts.leftHip, rh = pts.rightHip;
  const lk = pts.leftKnee, rk = pts.rightKnee;
  const la = pts.leftAnkle, ra = pts.rightAnkle;
  if (lh && lk) els.push(mkLine(lh.x, lh.y, lk.x, lk.y, 'red', 0.8, 0.48, false, 'body'));
  if (lk && la) els.push(mkLine(lk.x, lk.y, la.x, la.y, 'red', 0.8, 0.48, false, 'body'));
  if (rh && rk) els.push(mkLine(rh.x, rh.y, rk.x, rk.y, 'red', 0.8, 0.48, false, 'body'));
  if (rk && ra) els.push(mkLine(rk.x, rk.y, ra.x, ra.y, 'red', 0.8, 0.48, false, 'body'));
  const legPts: BodyPointName[] = ['leftKnee','rightKnee','leftAnkle','rightAnkle'];
  for (const pn of legPts) { const p = pts[pn]; if (p) els.push(mkDot(p.x, p.y, 'red', 0.006, 0.62, 'body')); }
  return els;
}

export function getTrackedPoint(_issue: MainIssueType, _viewType: ViewType, kpFrame: KeypointFrame): { x:number; y:number } | null {
  const pts = getKeypoints(kpFrame);
  const sc = pts.shoulderCenter;
  return sc ? { x: sc.x, y: sc.y } : null;
}

export function findNearestFrame(frames: KeypointFrame[], time: number): KeypointFrame | null {
  if (!frames.length) return null;
  let best = frames[0], bestDist = Math.abs(time - best.time);
  for (const f of frames) { const d = Math.abs(time - f.time); if (d < bestDist) { best = f; bestDist = d; } }
  return best;
}

export function applyCorrection(pt: { x:number; y:number; confidence?:number }, dir: string, mag = 'medium'): { x:number; y:number; confidence?:number } {
  const DELTA: Record<string, number> = { small: 0.028, medium: 0.050, large: 0.078 };
  const d = DELTA[mag] ?? 0.050;
  switch (dir) {
    case 'lower': return { ...pt, y: pt.y + d };
    case 'higher': return { ...pt, y: pt.y - d };
    case 'more_inside': return { ...pt, x: pt.x - d };
    case 'more_outside': return { ...pt, x: pt.x + d };
    case 'more_centered': return { ...pt, x: 0.50 };
    default: return pt;
  }
}
