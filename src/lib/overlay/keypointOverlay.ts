/**
 * keypointOverlay.ts
 *
 * 极简双色旋转盘设计：
 *   红色 = 用户当前位置（椭圆 + 穿越线）
 *   绿色 = 专业高尔夫球手的正确位置（椭圆 + 穿越线）
 *   穿越线比椭圆长20%，延伸到身体两侧外面
 *   只显示肩部和髋部，其余全部隐藏
 */

import type { OverlayElement, KeypointFrame } from '@/types/analysis';
import type { MainIssueType } from '@/types/analysis';
import type { BodyPointName, Pt } from '@/lib/golf/bodyPointSpec';
import type { ViewType } from '@/lib/golf/overlayLineSpec';

type Color = 'red' | 'green' | 'yellow' | 'white';

let _uid = 0;
const uid = (p: string) => `${p}-${++_uid}`;

const mkDot = (x: number, y: number, color: Color, r = 0.009, opacity = 0.95, layer: OverlayElement['layer'] = 'body'): OverlayElement =>
  ({ type: 'dot', id: uid('d'), x, y, color, radius: r, opacity, layer });

const mkLine = (x1: number, y1: number, x2: number, y2: number, color: Color, w = 2.0, opacity = 0.90, dashed = false, layer: OverlayElement['layer'] = 'body'): OverlayElement =>
  ({ type: 'line', id: uid('l'), x1, y1, x2, y2, color, strokeWidth: w, opacity, dashed, layer });

const mkLabel = (x: number, y: number, text: string, color: Color = 'white', size = 10): OverlayElement =>
  ({ type: 'label', id: uid('t'), x, y, text, color, size, opacity: 0.90 });

const mkEllipse = (cx: number, cy: number, rx: number, ry: number, angle: number, color: Color, w = 3.5, opacity = 0.92, layer: OverlayElement['layer'] = 'body'): OverlayElement =>
  ({ type: 'ellipse' as OverlayElement['type'], id: uid('e'), cx, cy, rx, ry, angle, color, strokeWidth: w, opacity, layer } as unknown as OverlayElement);

/* ── 关键点解析 ── */
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
 * buildRotationDisc — 旋转盘核心函数
 *
 * 根据两端点画旋转盘：
 *   - 红色椭圆 = 实际旋转平面（连接两端点）
 *   - 红色穿越线 = 沿旋转轴延伸，比椭圆长20%
 *   - 红色端点 = 两端小圆点
 *   - 绿色椭圆 = 目标旋转平面（更平/正确角度）
 *   - 绿色穿越线 = 绿色旋转轴
 *
 * 绿色目标的计算逻辑：
 *   高尔夫标准：address时肩膀应接近水平，
 *   倾斜角应比用户更平（偏向水平轴）
 */
function buildRotationDisc(
  leftPt: Pt, rightPt: Pt,
  label: string,
  layer: OverlayElement['layer'] = 'body',
): OverlayElement[] {
  const els: OverlayElement[] = [];

  // ── 当前实际位置（红色）──
  const cx = (leftPt.x + rightPt.x) / 2;
  const cy = (leftPt.y + rightPt.y) / 2;
  const dx = rightPt.x - leftPt.x;
  const dy = rightPt.y - leftPt.y;
  const angle = Math.atan2(dy, dx); // 实际旋转角度（弧度）
  const width = Math.hypot(dx, dy); // 两端点距离

  // 椭圆尺寸：比身体宽，延伸到外面
  const rx = width * 1.05;  // 长轴比肩宽大10%
  const ry = rx * 0.36;     // 短轴 = 36%长轴，有足够厚度

  // 穿越线端点：沿旋转轴方向，比椭圆长20%
  const lineExt = rx * 1.25;
  const cosA = Math.cos(angle), sinA = Math.sin(angle);
  const lx1 = cx - lineExt * cosA, ly1 = cy - lineExt * sinA;
  const lx2 = cx + lineExt * cosA, ly2 = cy + lineExt * sinA;

  // 红色穿越线（先画，椭圆覆盖在上面）
  els.push(mkLine(lx1, ly1, lx2, ly2, 'red', 2.0, 0.88, false, layer));
  // 红色椭圆（用户当前平面）
  els.push(mkEllipse(cx, cy, rx, ry, angle, 'red', 3.5, 0.92, layer));
  // 红色端点
  els.push(mkDot(leftPt.x, leftPt.y, 'red', 0.010, 0.95, layer));
  els.push(mkDot(rightPt.x, rightPt.y, 'red', 0.010, 0.95, layer));

  // ── 目标正确位置（绿色）──
  // 专业球手的旋转平面更平（倾斜角更小）
  // 绿色中心点与红色相同，但角度更接近水平（减少60%倾斜）
  const targetAngle = angle * 0.4; // 目标角度 = 当前角度 × 40%（更平）
  const targetRx = rx * 1.08;       // 绿色椭圆稍大，表示更充分旋转
  const targetRy = ry * 0.80;       // 更扁，旋转更平

  // 目标中心：沿法线方向偏移一点（视觉上分开）
  const perpCos = Math.cos(angle + Math.PI / 2);
  const perpSin = Math.sin(angle + Math.PI / 2);
  const offset = ry * 0.3;
  const tcx = cx + perpCos * offset;
  const tcy = cy + perpSin * offset;

  // 绿色穿越线
  const tLineExt = targetRx * 1.25;
  const tCosA = Math.cos(targetAngle), tSinA = Math.sin(targetAngle);
  const tlx1 = tcx - tLineExt * tCosA, tly1 = tcy - tLineExt * tSinA;
  const tlx2 = tcx + tLineExt * tCosA, tly2 = tcy + tLineExt * tSinA;

  els.push(mkLine(tlx1, tly1, tlx2, tly2, 'green', 2.0, 0.82, false, layer));
  // 绿色椭圆（目标平面）
  els.push(mkEllipse(tcx, tcy, targetRx, targetRy, targetAngle, 'green', 3.0, 0.82, layer));

  // 标注
  const deg = Math.round(Math.abs(angle * 180 / Math.PI));
  els.push(mkLabel(cx, cy - ry * 1.6, label + '  ' + deg + '°', 'white', 10));

  return els;
}

/* ── 主函数 ── */
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

  // 肩部旋转盘（Shoulders layer → body）
  const ls = pts.leftShoulder, rs = pts.rightShoulder;
  if (ls && rs) {
    els.push(...buildRotationDisc(ls, rs, 'SHOULDERS', 'body'));
  }

  // 髋部旋转盘（Hips layer → club）
  const lh = pts.leftHip, rh = pts.rightHip;
  if (lh && rh) {
    els.push(...buildRotationDisc(lh, rh, 'HIPS', 'club'));
  }

  return els;
}

/* ── 工具函数（保持兼容） ── */
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
    case 'lower': return { ...pt, y: pt.y + d };
    case 'higher': return { ...pt, y: pt.y - d };
    case 'more_inside': return { ...pt, x: pt.x - d };
    case 'more_outside': return { ...pt, x: pt.x + d };
    case 'more_centered': return { ...pt, x: 0.50 };
    default: return pt;
  }
}
