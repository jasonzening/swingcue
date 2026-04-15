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

/* ── 从 KeypointFrame 解析规范点位 ── */
export function getKeypoints(kpFrame: KeypointFrame): Partial<Record<BodyPointName, Pt>> {
  const lm = kpFrame.landmarks;
  const mp = new Array(33).fill(null).map(() => ({ x: 0.5, y: 0.5, visibility: 0 }));

  const setMp = (idx: number, pt?: { x: number; y: number; confidence?: number } | null) => {
    if (!pt) return;
    mp[idx] = { x: pt.x, y: pt.y, visibility: pt.confidence ?? 0.8 };
  };

  setMp(0,  lm.head);
  setMp(11, lm.leftShoulder);
  setMp(12, lm.rightShoulder);
  setMp(13, lm.leftElbow);
  setMp(14, lm.rightElbow);
  setMp(15, lm.leftWrist);
  setMp(16, lm.rightWrist);
  setMp(23, lm.leftHip);
  setMp(24, lm.rightHip);
  setMp(25, lm.leftKnee);
  setMp(26, lm.rightKnee);
  setMp(27, lm.leftAnkle);
  setMp(28, lm.rightAnkle);

  return resolveAllPoints(mp);
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

  const displaySpec = getIssueDisplaySpec(issue, viewType);
  if (!displaySpec) return buildBasicSkeleton(pts);

  const phasePoints = filterPointsByPhase(displaySpec, phase);

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
