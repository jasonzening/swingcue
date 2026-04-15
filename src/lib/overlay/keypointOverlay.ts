/**
 * keypointOverlay.ts — 规范驱动的 overlay 生成器
 *
 * 严格按 bodyPointSpec / overlayLineSpec / issueDisplaySpec 生成 overlay。
 * 不再自由选点，不再随意连线。
 * 绿色目标点 = 真实当前点 + correction delta（基于规范）
 */

import type { OverlayElement, KeypointFrame } from '@/types/analysis';
import type { MainIssueType } from '@/types/analysis';
import type { BodyPointName, Pt } from '@/lib/golf/bodyPointSpec';
import type { CorrectionDirection } from '@/lib/golf/types';
import type { ViewType } from '@/lib/golf/overlayLineSpec';
import type { SwingPhase, PointDisplaySpec } from '@/lib/golf/issueDisplaySpec';
import { resolveAllPoints } from '@/lib/golf/bodyPointSpec';
import { STRUCTURE_LINE_SPEC, type StructureLineName } from '@/lib/golf/overlayLineSpec';
import { getIssueDisplaySpec, filterPointsByPhase } from '@/lib/golf/issueDisplaySpec';

type Color = 'red' | 'green' | 'yellow' | 'white';

/* ── delta 修正量 ── */
const DELTA = { small: 0.028, medium: 0.050, large: 0.078 } as const;

export function applyCorrection(pt: Pt, dir: CorrectionDirection, mag: 'small'|'medium'|'large' = 'medium'): Pt {
  const d = DELTA[mag];
  const { x, y } = pt;
  switch (dir) {
    case 'lower':         return { x, y: y + d };
    case 'higher':        return { x, y: y - d };
    case 'more_inside':   return { x: x - d,       y };
    case 'more_outside':  return { x: x + d,       y };
    case 'more_forward':  return { x: x + d * 0.7, y };
    case 'more_back':     return { x: x - d * 0.7, y };
    case 'more_turned':   return { x: x + d,       y };
    case 'less_turned':   return { x: x - d,       y };
    case 'shallower':     return { x,              y: y + d * 0.4 };
    case 'steeper':       return { x,              y: y - d * 0.4 };
    case 'more_centered': return { x: 0.50,        y };
    case 'more_stable':   return pt;
    default:              return pt;
  }
}

/* ── element builders ── */
let _uid = 0;
const uid = (p: string) => `${p}-${++_uid}`;
const mkDot = (x: number, y: number, color: Color, r = 0.026, opacity = 0.94, layer: OverlayElement['layer'] = 'body'): OverlayElement =>
  ({ type: 'dot', id: uid('d'), x, y, color, radius: r, opacity, layer });
const mkLine = (x1: number, y1: number, x2: number, y2: number, color: Color, w = 2.8, opacity = 0.82, dashed = false, layer: OverlayElement['layer'] = 'body'): OverlayElement =>
  ({ type: 'line', id: uid('l'), x1, y1, x2, y2, color, strokeWidth: w, opacity, dashed, layer });
const mkArrow = (fx: number, fy: number, tx: number, ty: number): OverlayElement =>
  ({ type: 'arrow', id: uid('a'), from: { x: fx, y: fy }, to: { x: tx, y: ty }, color: 'yellow', strokeWidth: 3, opacity: 0.90 });
const mkCurve = (points: {x:number;y:number}[], color: Color, w = 3.5, opacity = 0.82): OverlayElement =>
  ({ type: 'curve', id: uid('c'), points, color, strokeWidth: w, opacity, layer: 'arms' });
const mkLabel = (x: number, y: number, text: string, color: Color = 'white', size = 10): OverlayElement =>
  ({ type: 'label', id: uid('t'), x, y, text, color, size, opacity: 0.80 });

/* ── 从 KeypointFrame 解析规范点位（直接按名字映射，不走 MediaPipe 索引）── */
export function getKeypoints(kpFrame: KeypointFrame): Partial<Record<BodyPointName, Pt>> {
  const lm = kpFrame.landmarks;
  const result: Partial<Record<BodyPointName, Pt>> = {};

  const toP = (pt?: { x: number; y: number; confidence?: number } | null): Pt | null =>
    pt ? { x: pt.x, y: pt.y, confidence: pt.confidence ?? 0.8 } : null;

  // 直接按字段名赋值，置信度直接来自 Python 输出
  if (lm.head)          result.headCenter     = toP(lm.head)!;
  if (lm.leftShoulder)  result.leftShoulder   = toP(lm.leftShoulder)!;
  if (lm.rightShoulder) result.rightShoulder  = toP(lm.rightShoulder)!;
  if (lm.leftElbow)     result.leftElbow      = toP(lm.leftElbow)!;
  if (lm.rightElbow)    result.rightElbow     = toP(lm.rightElbow)!;
  if (lm.leftWrist)     result.leftWrist      = toP(lm.leftWrist)!;
  if (lm.rightWrist)    result.rightWrist     = toP(lm.rightWrist)!;
  if (lm.leftHip)       result.leftHip        = toP(lm.leftHip)!;
  if (lm.rightHip)      result.rightHip       = toP(lm.rightHip)!;
  if (lm.leftKnee)      result.leftKnee       = toP(lm.leftKnee)!;
  if (lm.rightKnee)     result.rightKnee      = toP(lm.rightKnee)!;
  if (lm.leftAnkle)     result.leftAnkle      = toP(lm.leftAnkle)!;
  if (lm.rightAnkle)    result.rightAnkle     = toP(lm.rightAnkle)!;

  // 派生点：直接算
  const ls = result.leftShoulder, rs = result.rightShoulder;
  if (ls && rs) result.shoulderCenter = { x: (ls.x + rs.x) / 2, y: (ls.y + rs.y) / 2, confidence: 1 };

  const lh = result.leftHip, rh = result.rightHip;
  if (lh && rh) result.hipCenter = { x: (lh.x + rh.x) / 2, y: (lh.y + rh.y) / 2, confidence: 1 };

  const lw = result.leftWrist, rw = result.rightWrist;
  if (lw && rw) result.gripCenter = { x: (lw.x + rw.x) / 2, y: (lw.y + rw.y) / 2, confidence: 1 };

  return result;
}

/* ── 结构线（两点都存在才画）── */
function buildStructureLine(
  lineName: StructureLineName,
  pts: Partial<Record<BodyPointName, Pt>>,
  color: Color, opacity: number, dashed = false,
): OverlayElement[] {
  const spec = STRUCTURE_LINE_SPEC[lineName];
  const resolved = spec.points.map(pn => pts[pn] ?? null);
  if (spec.requiresAllPoints && resolved.some(p => !p)) return [];
  const valid = resolved.filter((p): p is Pt => p !== null);
  if (valid.length < 2) return [];
  const els: OverlayElement[] = [];
  for (let i = 0; i < valid.length - 1; i++) {
    els.push(mkLine(valid[i].x, valid[i].y, valid[i+1].x, valid[i+1].y, color, 2.8, opacity, dashed));
  }
  valid.forEach(p => els.push(mkDot(p.x, p.y, color, 0.020, opacity)));
  return els;
}

/* ── 绿色目标线（红点 + delta）── */
function buildGreenLine(
  lineName: StructureLineName,
  pts: Partial<Record<BodyPointName, Pt>>,
  pointSpecs: PointDisplaySpec[],
): OverlayElement[] {
  const spec = STRUCTURE_LINE_SPEC[lineName];
  const greenPts: Pt[] = [];
  for (const pn of spec.points) {
    const redPt = pts[pn];
    if (!redPt) return [];
    const correction = pointSpecs.find(s => s.point === pn && s.showGreen);
    greenPts.push(
      correction?.greenDirection
        ? applyCorrection(redPt, correction.greenDirection, correction.greenMagnitude ?? 'medium')
        : redPt
    );
  }
  if (greenPts.length < 2) return [];
  const els: OverlayElement[] = [];
  for (let i = 0; i < greenPts.length - 1; i++) {
    els.push(mkLine(greenPts[i].x, greenPts[i].y, greenPts[i+1].x, greenPts[i+1].y, 'green', 2.8, 0.78, true));
  }
  greenPts.forEach(p => els.push(mkDot(p.x, p.y, 'green', 0.020, 0.80)));
  return els;
}

/* ── impact 手臂三角结构 ── */
function buildArmTriangle(pts: Partial<Record<BodyPointName, Pt>>, color: Color, opacity: number): OverlayElement[] {
  const ls = pts.leftShoulder, rs = pts.rightShoulder, grip = pts.gripCenter;
  if (!ls || !rs || !grip) return [];
  return [
    mkLine(ls.x, ls.y, grip.x, grip.y, color, 3.0, opacity),
    mkLine(rs.x, rs.y, grip.x, grip.y, color, 3.0, opacity),
    mkLine(ls.x, ls.y, rs.x, rs.y, color, 2.5, opacity * 0.80),
    mkDot(ls.x, ls.y, color, 0.026, opacity),
    mkDot(rs.x, rs.y, color, 0.026, opacity),
    mkDot(grip.x, grip.y, color, 0.032, opacity),
  ];
}

/* ── 基础骨架 fallback ── */
function buildBasicSkeleton(pts: Partial<Record<BodyPointName, Pt>>): OverlayElement[] {
  const els: OverlayElement[] = [];
  const ls = pts.leftShoulder, rs = pts.rightShoulder;
  const lh = pts.leftHip, rh = pts.rightHip;
  const sc = pts.shoulderCenter, hc = pts.hipCenter;
  if (ls && rs) { els.push(mkLine(ls.x, ls.y, rs.x, rs.y, 'green', 2.5, 0.72)); mkDot(ls.x, ls.y, 'green', 0.022, 0.80); mkDot(rs.x, rs.y, 'green', 0.022, 0.80); }
  if (sc && hc) els.push(mkLine(sc.x, sc.y, hc.x, hc.y, 'white', 2.0, 0.52));
  if (lh && rh) els.push(mkLine(lh.x, lh.y, rh.x, rh.y, 'white', 2.0, 0.48, true));
  return els;
}

/* ── 始终绘制的基础身体骨架（红点 + 红线）── */
function buildFullBodySkeleton(pts: Partial<Record<BodyPointName, Pt>>): OverlayElement[] {
  const els: OverlayElement[] = [];

  // 所有关键身体点 → 红色小圆点
  const bodyPoints: BodyPointName[] = [
    'headCenter',
    'leftShoulder', 'rightShoulder',
    'leftElbow', 'rightElbow',
    'leftWrist', 'rightWrist',
    'leftHip', 'rightHip',
    'gripCenter', 'hipCenter', 'shoulderCenter',
  ];
  for (const pn of bodyPoints) {
    const p = pts[pn];
    if (p) els.push(mkDot(p.x, p.y, 'red', 0.022, 0.90));
  }

  // 肩线
  const ls = pts.leftShoulder, rs = pts.rightShoulder;
  if (ls && rs) els.push(mkLine(ls.x, ls.y, rs.x, rs.y, 'red', 2.5, 0.85));

  // 髋线
  const lh = pts.leftHip, rh = pts.rightHip;
  if (lh && rh) els.push(mkLine(lh.x, lh.y, rh.x, rh.y, 'red', 2.5, 0.80));

  // 脊柱线（白色半透明）
  const sc = pts.shoulderCenter, hc = pts.hipCenter;
  if (sc && hc) els.push(mkLine(sc.x, sc.y, hc.x, hc.y, 'white', 1.8, 0.45, true));

  // 左臂链：左肩 → 左肘 → 左腕
  const le = pts.leftElbow, lw = pts.leftWrist;
  if (ls && le) els.push(mkLine(ls.x, ls.y, le.x, le.y, 'red', 2.2, 0.82));
  if (le && lw) els.push(mkLine(le.x, le.y, lw.x, lw.y, 'red', 2.2, 0.82));

  // 右臂链：右肩 → 右肘 → 右腕
  const re = pts.rightElbow, rw = pts.rightWrist;
  if (rs && re) els.push(mkLine(rs.x, rs.y, re.x, re.y, 'red', 2.2, 0.82));
  if (re && rw) els.push(mkLine(re.x, re.y, rw.x, rw.y, 'red', 2.2, 0.82));

  // 双手中点（较大红点）
  const grip = pts.gripCenter;
  if (grip) els.push(mkDot(grip.x, grip.y, 'red', 0.030, 0.95));

  return els;
}

/* ══════════════════════════════════════
   主函数：规范驱动的 overlay frame 生成
══════════════════════════════════════ */
export function generateSpecDrivenOverlayFrame(
  issue: MainIssueType,
  viewType: ViewType,
  phase: SwingPhase,
  kpFrame: KeypointFrame,
  historyPts?: Array<{ x: number; y: number }>,
): OverlayElement[] {
  _uid = 0;
  const elements: OverlayElement[] = [];
  const pts = getKeypoints(kpFrame);

  // ── STEP 1: 始终渲染完整身体骨架（红点 + 红线）──
  elements.push(...buildFullBodySkeleton(pts));

  const displaySpec = getIssueDisplaySpec(issue, viewType);
  if (!displaySpec) return elements; // 没有 spec 就只返回骨架

  const phasePoints = filterPointsByPhase(displaySpec, phase);

  // ── STEP 2: Issue 专项 overlay（绿色目标 + 箭头）──
  // 辅助线（白色淡）
  for (const ln of displaySpec.auxiliaryLines) {
    elements.push(...buildStructureLine(ln, pts, 'white', 0.42, false));
  }

  // 必须显示的红色结构线
  for (const ln of displaySpec.mustShowLines) {
    elements.push(...buildStructureLine(ln, pts, 'red', 0.85, false));
  }

  // 绿色目标线（仅当有点需要 showGreen 时）
  if (phasePoints.some(p => p.showGreen)) {
    for (const ln of displaySpec.mustShowLines) {
      elements.push(...buildGreenLine(ln, pts, phasePoints));
    }
  }

  // 每个规范点位：红点 + 绿点 + 箭头
  for (const ps of phasePoints) {
    const redPt = pts[ps.point];
    if (!redPt) continue;

    if (ps.showRed) {
      elements.push(mkDot(redPt.x, redPt.y, 'red', ps.priority === 'must' ? 0.028 : 0.022, 0.95));
    }

    if (ps.showGreen && ps.greenDirection) {
      const greenPt = applyCorrection(redPt, ps.greenDirection, ps.greenMagnitude ?? 'medium');
      elements.push(mkDot(greenPt.x, greenPt.y, 'green', 0.024, 0.88));

      if (ps.showArrow && Math.hypot(greenPt.x - redPt.x, greenPt.y - redPt.y) > 0.012) {
        elements.push(mkArrow(redPt.x, redPt.y, greenPt.x, greenPt.y));
      }
    }
  }

  // impact 手臂三角
  if (displaySpec.showImpactTriangle && phase === 'impact') {
    elements.push(...buildArmTriangle(pts, 'red', 0.82));
    const grip = pts.gripCenter;
    if (grip) {
      const greenGrip = applyCorrection(grip, 'more_inside', 'small');
      elements.push(...buildArmTriangle({ ...pts, gripCenter: greenGrip }, 'green', 0.68));
    }
  }

  // 历史路径
  if (historyPts && historyPts.length >= 2) {
    elements.push(mkCurve(historyPts, 'red', 3.0, 0.70));
  }

  // 阶段标签
  elements.push(mkLabel(0.50, 0.06, phase.toUpperCase(), 'white', 10));

  return elements;
}

/* ── 路径追踪：获取当前帧追踪点坐标 ── */
export function getTrackedPoint(
  issue: MainIssueType, viewType: ViewType, kpFrame: KeypointFrame,
): { x: number; y: number } | null {
  const displaySpec = getIssueDisplaySpec(issue, viewType);
  if (!displaySpec?.paths.length) return null;
  const pts = getKeypoints(kpFrame);
  const pt = pts[displaySpec.paths[0].trackedPoint];
  return pt ? { x: pt.x, y: pt.y } : null;
}

/* ── nearest-frame 查找 ── */
export function findNearestFrame(frames: KeypointFrame[], time: number): KeypointFrame | null {
  if (!frames.length) return null;
  let best = frames[0], bestDist = Math.abs(time - best.time);
  for (const f of frames) {
    const d = Math.abs(time - f.time);
    if (d < bestDist) { best = f; bestDist = d; }
  }
  return best;
}
