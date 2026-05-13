/**
 * keypointOverlay.ts — 旋转盘双色显示
 *
 * 红色 = 当前位置（椭圆 + 穿越线）
 * 绿色 = 正确目标位置（椭圆 + 穿越线）
 *
 * 关键修正：angle 可能接近 ±π（比如 180°），
 * 目标角度不能简单乘以 0.4，否则绿色线会变成斜线。
 * 正确做法：从最近的水平轴（0 或 ±π）计算偏斜量，再减少偏斜。
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
const mkLine = (x1: number, y1: number, x2: number, y2: number, color: Color, w = 2.2, opacity = 0.90, dashed = false, layer: OverlayElement['layer'] = 'body'): OverlayElement =>
  ({ type: 'line', id: uid('l'), x1, y1, x2, y2, color, strokeWidth: w, opacity, dashed, layer });
const mkLabel = (x: number, y: number, text: string, color: Color = 'white', size = 10): OverlayElement =>
  ({ type: 'label', id: uid('t'), x, y, text, color, size, opacity: 0.92 });
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
 * computeTargetAngle
 *
 * 从 angle 计算"更水平"的目标角度。
 * 关键：angle 可能是任意弧度，需要找到最近的水平轴（0 或 ±π），
 * 然后在该轴附近减少倾斜量（倾斜量乘以 0.35）。
 *
 * 例子：
 *   angle =  π  (180°) → nearestH = π,  tilt = 0  → targetAngle = π   ✅
 *   angle = 2.9 (166°) → nearestH = π,  tilt = 2.9-π ≈ -0.24 → targetAngle = π + (-0.24 × 0.35) ≈ 3.05 ✅
 *   angle = 0.3 ( 17°) → nearestH = 0,  tilt = 0.3 → targetAngle = 0 + 0.3×0.35 ≈ 0.11 ✅
 */
function computeTargetAngle(angle: number): number {
  // 找最近的水平轴：0 或 ±π
  const nearestH = Math.round(angle / Math.PI) * Math.PI;
  // 当前对水平轴的偏斜量
  const tilt = angle - nearestH;
  // 目标：减少 65% 的偏斜（高尔夫球手旋转更平）
  return nearestH + tilt * 0.35;
}

/**
 * buildRotationDisc — 旋转盘核心
 *
 * 对于每组端点（左肩/右肩 或 左髋/右髋）：
 *   1. 计算实际旋转角和椭圆参数
 *   2. 画红色椭圆 + 红色穿越线（比椭圆长 25%）+ 红色端点
 *   3. 计算目标旋转角（更水平）
 *   4. 画绿色椭圆 + 绿色穿越线 + 无端点（避免干扰）
 */
function buildRotationDisc(
  leftPt: Pt,
  rightPt: Pt,
  label: string,
  layer: OverlayElement['layer'] = 'body',
): OverlayElement[] {
  const els: OverlayElement[] = [];

  // ── 几何计算 ──
  const cx = (leftPt.x + rightPt.x) / 2;
  const cy = (leftPt.y + rightPt.y) / 2;
  const dx = rightPt.x - leftPt.x;
  const dy = rightPt.y - leftPt.y;
  const angle = Math.atan2(dy, dx);
  const bodyWidth = Math.hypot(dx, dy);

  // 椭圆：比身体宽度更大，延伸到两侧外面
  const rx = bodyWidth * 1.08;
  const ry = rx * 0.35;

  // 穿越线端点（比椭圆长 25%）
  const lineExt = rx * 1.28;
  const cosA = Math.cos(angle), sinA = Math.sin(angle);

  // ── 🔴 红色：当前位置 ──
  // 穿越线（先画，让椭圆压在上面）
  els.push(mkLine(
    cx - lineExt * cosA, cy - lineExt * sinA,
    cx + lineExt * cosA, cy + lineExt * sinA,
    'red', 2.2, 0.88, false, layer,
  ));
  // 椭圆
  els.push(mkEllipse(cx, cy, rx, ry, angle, 'red', 3.5, 0.92, layer));
  // 端点
  els.push(mkDot(leftPt.x,  leftPt.y,  'red', 0.010, 0.95, layer));
  els.push(mkDot(rightPt.x, rightPt.y, 'red', 0.010, 0.95, layer));

  // ── 🟢 绿色：目标正确位置 ──
  const targetAngle = computeTargetAngle(angle);
  const targetRx = rx * 1.06;
  const targetRy = ry * 0.78; // 更扁、更平

  // 目标垂直偏移（沿椭圆法线方向，让两个椭圆稍微分开）
  const perpCos = -sinA, perpSin = cosA; // 垂直于旋转轴
  const offsetDist = ry * 0.35;
  const tcx = cx + perpCos * offsetDist;
  const tcy = cy + perpSin * offsetDist;

  // 绿色穿越线
  const tLineExt = targetRx * 1.28;
  const tCosA = Math.cos(targetAngle), tSinA = Math.sin(targetAngle);
  els.push(mkLine(
    tcx - tLineExt * tCosA, tcy - tLineExt * tSinA,
    tcx + tLineExt * tCosA, tcy + tLineExt * tSinA,
    'green', 2.0, 0.82, false, layer,
  ));
  // 绿色椭圆
  els.push(mkEllipse(tcx, tcy, targetRx, targetRy, targetAngle, 'green', 3.0, 0.82, layer));

  // ── 角度标注 ──
  // 实际倾斜角：从水平轴的偏斜度数
  const nearestH = Math.round(angle / Math.PI) * Math.PI;
  const tiltDeg = Math.round(Math.abs((angle - nearestH) * 180 / Math.PI));
  els.push(mkLabel(cx, cy - ry * 1.55, label + '  ' + tiltDeg + '°', 'white', 10));

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
    case 'lower':        return { ...pt, y: pt.y + d };
    case 'higher':       return { ...pt, y: pt.y - d };
    case 'more_inside':  return { ...pt, x: pt.x - d };
    case 'more_outside': return { ...pt, x: pt.x + d };
    case 'more_centered':return { ...pt, x: 0.50 };
    default:             return pt;
  }
}
