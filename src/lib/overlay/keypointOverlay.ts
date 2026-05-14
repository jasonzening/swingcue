/**
 * keypointOverlay.ts
 *
 * 旋转盘显示原则（美术透视规则）：
 *
 * ① 颜色：灰色（rgba(200,200,200,0.82)），不刺眼
 *
 * ② 透视压缩：
 *    圆盘是肩膀连线绕垂直轴旋转形成的平面。
 *    正面时：ry = 最大值（圆盘正面朝镜头）
 *    转45°时：ry = ry_max × cos(45°) ≈ 70%（压扁）
 *    转90°时：ry → 0（侧面看只剩一条线）
 *    visRatio = apparentWidth / refWidth ≈ cos(旋转角)
 *    perspRy  = fixedRy × visRatio
 *
 * ③ 倾斜角度（关键修正）：
 *    问题：atan2(dy, dx) 当 dx 很小时（身体转侧）dy/dx 被放大，
 *          导致圆盘极端倾斜——这不符合透视原则。
 *    解决：用 Y 方向的差值除以"参考宽度"（而不是当前 dx），
 *          得到稳定的倾斜角。
 *    tiltAngle = atan2(dy, refWidth) → 只反映实际脊柱倾斜
 *    旋转角（身体转向）不影响 tiltAngle，只影响 visRatio。
 *
 * ④ 隐藏条件：
 *    visRatio < 0.22 → 身体已转超过约 75° → 圆盘失去参考意义
 *    confidence < 0.40 → 关键点被遮挡
 */

import type { OverlayElement, KeypointFrame } from '@/types/analysis';
import type { MainIssueType } from '@/types/analysis';
import type { BodyPointName, Pt } from '@/lib/golf/bodyPointSpec';
import type { ViewType } from '@/lib/golf/overlayLineSpec';

type Color = 'red' | 'green' | 'yellow' | 'white' | 'gray';

let _uid = 0;
const uid = (p: string) => `${p}-${++_uid}`;

const DISC_COLOR: Color = 'gray';

const mkDot = (x: number, y: number, color: Color, r = 0.009, opacity = 0.78, layer: OverlayElement['layer'] = 'body'): OverlayElement =>
  ({ type: 'dot', id: uid('d'), x, y, color, radius: r, opacity, layer });
const mkLine = (x1: number, y1: number, x2: number, y2: number, color: Color, w = 2.0, opacity = 0.75, dashed = false, layer: OverlayElement['layer'] = 'body'): OverlayElement =>
  ({ type: 'line', id: uid('l'), x1, y1, x2, y2, color, strokeWidth: w, opacity, dashed, layer });
const mkLabel = (x: number, y: number, text: string, color: Color = 'gray', size = 9): OverlayElement =>
  ({ type: 'label', id: uid('t'), x, y, text, color, size, opacity: 0.72 });
const mkEllipse = (cx: number, cy: number, rx: number, ry: number, angle: number, color: Color, w = 2.5, opacity = 0.80, layer: OverlayElement['layer'] = 'body'): OverlayElement =>
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
 * buildDisc
 *
 * @param leftPt    左端点（左肩/左髋）
 * @param rightPt   右端点（右肩/右髋）
 * @param fixedRx   固定长轴（不随转身变化）
 * @param fixedRy   固定短轴（正面朝向时最大值）
 * @param refWidth  参考宽度（正面时两点水平间距）
 * @param label     标注文字
 * @param layer     层
 */
function buildDisc(
  leftPt: Pt, rightPt: Pt,
  fixedRx: number, fixedRy: number,
  refWidth: number,
  label: string,
  layer: OverlayElement['layer'] = 'body',
): OverlayElement[] {

  // ── 遮挡检测 ──
  if ((leftPt.confidence ?? 0.8) < 0.40 || (rightPt.confidence ?? 0.8) < 0.40) return [];

  const cx = (leftPt.x + rightPt.x) / 2;
  const cy = (leftPt.y + rightPt.y) / 2;
  const rawDx = rightPt.x - leftPt.x; // 当前水平间距
  const rawDy = rightPt.y - leftPt.y; // Y差（脊柱倾斜）

  // 转身可见度：当前水平间距 / 参考宽度（正面时 = 1.0）
  const apparentW = Math.abs(rawDx);
  const visRatio  = Math.min(1.0, apparentW / refWidth);

  // 转身超过75°（visRatio < 0.22）→ 隐藏
  if (visRatio < 0.22) return [];

  // ── 透视压缩短轴 ──
  const perspRy = fixedRy * visRatio;

  // ── 倾斜角度（美术透视修正）──
  // 关键：用 refWidth 作为 dx 基准，而不是当前 rawDx
  // 这样 tiltAngle 只反映实际脊柱倾斜（rawDy），
  // 不会因为转身后 dx 变小而被放大
  const sign = rawDx >= 0 ? 1 : -1;
  const tiltAngle = Math.atan2(rawDy, sign * refWidth);

  // 透明度随可见度渐变（转身时自然淡出）
  const alpha = 0.45 + visRatio * 0.35; // 0.45~0.80

  const els: OverlayElement[] = [];

  // 轴线：从左端点到右端点，再延伸到圆盘外侧
  // 轴线长 = 圆盘长轴 × 1.3（穿过并延伸出去）
  const lineExt = fixedRx * 1.32;
  const cosT = Math.cos(tiltAngle), sinT = Math.sin(tiltAngle);

  els.push(mkLine(
    cx - lineExt * cosT, cy - lineExt * sinT,
    cx + lineExt * cosT, cy + lineExt * sinT,
    DISC_COLOR, 1.8, alpha * 0.90, false, layer,
  ));

  // 椭圆（透视压缩）
  if (perspRy > 0.015) {
    els.push(mkEllipse(cx, cy, fixedRx, perspRy, tiltAngle, DISC_COLOR, 2.5, alpha, layer));
  }

  // 端点（仅在可见度足够时显示）
  if ((leftPt.confidence  ?? 0.8) > 0.50) els.push(mkDot(leftPt.x,  leftPt.y,  DISC_COLOR, 0.009, alpha * 0.90, layer));
  if ((rightPt.confidence ?? 0.8) > 0.50) els.push(mkDot(rightPt.x, rightPt.y, DISC_COLOR, 0.009, alpha * 0.90, layer));

  // 倾斜角标注（仅正面时显示，转身后角度已无参考意义）
  if (visRatio > 0.60) {
    const tiltDeg = Math.round(Math.abs(rawDy / refWidth) * 90); // 简单倾斜估算
    els.push(mkLabel(cx, cy - perspRy * 1.6, label + '  ' + tiltDeg + '°', DISC_COLOR, 9));
  }

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

  // 固定圆盘尺寸
  const SHOULDER_RX  = 0.26;
  const SHOULDER_RY  = SHOULDER_RX * 0.34;
  const SHOULDER_REF = 0.22; // 正面朝向时肩宽约 22% 视频宽

  const HIP_RX  = 0.20;
  const HIP_RY  = HIP_RX * 0.34;
  const HIP_REF = 0.16; // 正面朝向时髋宽约 16%

  const ls = pts.leftShoulder, rs = pts.rightShoulder;
  if (ls && rs) els.push(...buildDisc(ls, rs, SHOULDER_RX, SHOULDER_RY, SHOULDER_REF, 'SHOULDERS', 'body'));

  const lh = pts.leftHip, rh = pts.rightHip;
  if (lh && rh) els.push(...buildDisc(lh, rh, HIP_RX, HIP_RY, HIP_REF, 'HIPS', 'club'));

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
  for (const f of frames) { const d = Math.abs(time - f.time); if (d < bestDist) { best = f; bestDist = d; } }
  return best;
}

export function applyCorrection(pt: { x:number; y:number; confidence?:number }, dir: string, mag = 'medium'): { x:number; y:number; confidence?:number } {
  const DELTA: Record<string, number> = { small:0.028, medium:0.050, large:0.078 };
  const d = DELTA[mag] ?? 0.050;
  switch (dir) {
    case 'lower':          return { ...pt, y: pt.y + d };
    case 'higher':         return { ...pt, y: pt.y - d };
    case 'more_inside':    return { ...pt, x: pt.x - d };
    case 'more_outside':   return { ...pt, x: pt.x + d };
    case 'more_centered':  return { ...pt, x: 0.50 };
    default:               return pt;
  }
}
