/**
 * keypointOverlay.ts â è§èé©±å¨ç overlay çæå¨
 *
 * ä¸¥æ ¼æ bodyPointSpec / overlayLineSpec / issueDisplaySpec çæ overlayã
 * ä¸åèªç±éç¹ï¼ä¸åéæè¿çº¿ã
 * ç»¿è²ç®æ ç¹ = çå®å½åç¹ + correction deltaï¼åºäºè§èï¼
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

/* ââ delta ä¿®æ­£é ââ */
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

/* ââ element builders ââ */
let _uid = 0;
const uid = (p: string) => `${p}-${++_uid}`;
const mkDot = (x: number, y: number, color: Color, r = 0.007, opacity = 0.90, layer: OverlayElement['layer'] = 'body'): OverlayElement =>
  ({ type: 'dot', id: uid('d'), x, y, color, radius: r, opacity, layer });
const mkLine = (x1: number, y1: number, x2: number, y2: number, color: Color, w = 1.0, opacity = 0.85, dashed = false, layer: OverlayElement['layer'] = 'body'): OverlayElement =>
  ({ type: 'line', id: uid('l'), x1, y1, x2, y2, color, strokeWidth: w, opacity, dashed, layer });
const mkArrow = (fx: number, fy: number, tx: number, ty: number): OverlayElement =>
  ({ type: 'arrow', id: uid('a'), from: { x: fx, y: fy }, to: { x: tx, y: ty }, color: 'yellow', strokeWidth: 1.5, opacity: 0.90 });
const mkCurve = (points: {x:number;y:number}[], color: Color, w = 3.5, opacity = 0.82): OverlayElement =>
  ({ type: 'curve', id: uid('c'), points, color, strokeWidth: w, opacity, layer: 'arms' });
const mkLabel = (x: number, y: number, text: string, color: Color = 'white', size = 10): OverlayElement =>
  ({ type: 'label', id: uid('t'), x, y, text, color, size, opacity: 0.80 });

/* ââ ä» KeypointFrame è§£æè§èç¹ä½ï¼ç´æ¥æåå­æ å°ï¼ä¸èµ° MediaPipe ç´¢å¼ï¼ââ */
export function getKeypoints(kpFrame: KeypointFrame): Partial<Record<BodyPointName, Pt>> {
  const lm = kpFrame.landmarks;
  const result: Partial<Record<BodyPointName, Pt>> = {};

  const toP = (pt?: { x: number; y: number; confidence?: number } | null): Pt | null =>
    pt ? { x: pt.x, y: pt.y, confidence: pt.confidence ?? 0.8 } : null;

  // ç´æ¥æå­æ®µåèµå¼ï¼ç½®ä¿¡åº¦ç´æ¥æ¥èª Python è¾åº
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

  // æ´¾çç¹ï¼ç´æ¥ç®
  const ls = result.leftShoulder, rs = result.rightShoulder;
  if (ls && rs) result.shoulderCenter = { x: (ls.x + rs.x) / 2, y: (ls.y + rs.y) / 2, confidence: 1 };

  const lh = result.leftHip, rh = result.rightHip;
  if (lh && rh) result.hipCenter = { x: (lh.x + rh.x) / 2, y: (lh.y + rh.y) / 2, confidence: 1 };

  const lw = result.leftWrist, rw = result.rightWrist;
  if (lw && rw) result.gripCenter = { x: (lw.x + rw.x) / 2, y: (lw.y + rw.y) / 2, confidence: 1 };

  return result;
}

/* ââ ç»æçº¿ï¼ä¸¤ç¹é½å­å¨æç»ï¼ââ */
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
  valid.forEach(p => els.push(mkDot(p.x, p.y, color, 0.006, opacity)));
  return els;
}

/* ââ ç»¿è²ç®æ çº¿ï¼çº¢ç¹ + deltaï¼ââ */
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

/* ââ impact æèä¸è§ç»æ ââ */
function buildArmTriangle(pts: Partial<Record<BodyPointName, Pt>>, color: Color, opacity: number): OverlayElement[] {
  const ls = pts.leftShoulder, rs = pts.rightShoulder, grip = pts.gripCenter;
  if (!ls || !rs || !grip) return [];
  return [
    mkLine(ls.x, ls.y, grip.x, grip.y, color, 1.2, opacity),
    mkLine(rs.x, rs.y, grip.x, grip.y, color, 1.2, opacity),
    mkLine(ls.x, ls.y, rs.x, rs.y, color, 1.0, opacity * 0.80),
    mkDot(ls.x, ls.y, color, 0.008, opacity),
    mkDot(rs.x, rs.y, color, 0.008, opacity),
    mkDot(grip.x, grip.y, color, 0.010, opacity),
  ];
}

/* ââ åºç¡éª¨æ¶ fallback ââ */
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

/* ââ å§ç»ç»å¶çåºç¡èº«ä½éª¨æ¶ï¼çº¢ç¹ + çº¢çº¿ï¼ââ */
function buildFullBodySkeleton(pts: Partial<Record<BodyPointName, Pt>>): OverlayElement[] {
  const els: OverlayElement[] = [];
  const DOT_R = 0.007;

  // ææå³é®èº«ä½ç¹ â çº¢è²å°åç¹
  const bodyPoints: BodyPointName[] = [
    'headCenter',
    'leftShoulder', 'rightShoulder',
    'leftElbow', 'rightElbow',
    'leftWrist', 'rightWrist',
    'leftHip', 'rightHip',
    'leftKnee', 'rightKnee',
    'leftAnkle', 'rightAnkle',
    'gripCenter', 'hipCenter', 'shoulderCenter',
  ];
  for (const pn of bodyPoints) {
    const p = pts[pn];
    if (p) els.push(mkDot(p.x, p.y, 'red', DOT_R, 0.90));
  }

  // è©çº¿
  const ls = pts.leftShoulder, rs = pts.rightShoulder;
  if (ls && rs) els.push(mkLine(ls.x, ls.y, rs.x, rs.y, 'red', 1.0, 0.88));

  // é«çº¿
  const lh = pts.leftHip, rh = pts.rightHip;
  if (lh && rh) els.push(mkLine(lh.x, lh.y, rh.x, rh.y, 'red', 1.0, 0.85));

  // èæ±çº¿ï¼ç½è²åéæï¼
  const sc = pts.shoulderCenter, hc = pts.hipCenter;
  if (sc && hc) els.push(mkLine(sc.x, sc.y, hc.x, hc.y, 'white', 0.8, 0.40, true));

  // å¤´ â åè©
  const head = pts.headCenter;
  if (head && ls) els.push(mkLine(head.x, head.y, ls.x, ls.y, 'red', 1.0, 0.82));
  if (head && rs) els.push(mkLine(head.x, head.y, rs.x, rs.y, 'red', 1.0, 0.82));

  // å·¦èé¾ï¼å·¦è© â å·¦è â å·¦è
  const le = pts.leftElbow, lw = pts.leftWrist;
  if (ls && le) els.push(mkLine(ls.x, ls.y, le.x, le.y, 'red', 1.0, 0.85));
  if (le && lw) els.push(mkLine(le.x, le.y, lw.x, lw.y, 'red', 1.0, 0.85));

  // å³èé¾ï¼å³è© â å³è â å³è
  const re = pts.rightElbow, rw = pts.rightWrist;
  if (rs && re) els.push(mkLine(rs.x, rs.y, re.x, re.y, 'red', 1.0, 0.85));
  if (re && rw) els.push(mkLine(re.x, re.y, rw.x, rw.y, 'red', 1.0, 0.85));

  // å·¦è¿é¾ï¼å·¦é« â å·¦è â å·¦è¸
  const lk = pts.leftKnee, la = pts.leftAnkle;
  if (lh && lk) els.push(mkLine(lh.x, lh.y, lk.x, lk.y, 'red', 1.0, 0.85));
  if (lk && la) els.push(mkLine(lk.x, lk.y, la.x, la.y, 'red', 1.0, 0.85));

  // å³è¿é¾ï¼å³é« â å³è â å³è¸
  const rk = pts.rightKnee, ra = pts.rightAnkle;
  if (rh && rk) els.push(mkLine(rh.x, rh.y, rk.x, rk.y, 'red', 1.0, 0.85));
  if (rk && ra) els.push(mkLine(rk.x, rk.y, ra.x, ra.y, 'red', 1.0, 0.85));

  // åæä¸­ç¹ï¼è¾å¤§çº¢ç¹ï¼
  const grip = pts.gripCenter;
  if (grip) els.push(mkDot(grip.x, grip.y, 'red', 0.010, 0.92));

  return els;
}

/* ââââââââââââââââââââââââââââââââââââââ
   ä¸»å½æ°ï¼è§èé©±å¨ç overlay frame çæ
ââââââââââââââââââââââââââââââââââââââ */
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

  // ââ STEP 1: å§ç»æ¸²æå®æ´èº«ä½éª¨æ¶ï¼çº¢ç¹ + çº¢çº¿ï¼ââ
  elements.push(...buildFullBodySkeleton(pts));

  const displaySpec = getIssueDisplaySpec(issue, viewType);
  if (!displaySpec) return elements; // æ²¡æ spec å°±åªè¿åéª¨æ¶

  const phasePoints = filterPointsByPhase(displaySpec, phase);

  // ââ STEP 2: Issue ä¸é¡¹ overlayï¼ç»¿è²ç®æ  + ç®­å¤´ï¼ââ
  // è¾å©çº¿ï¼ç½è²æ·¡ï¼
  for (const ln of displaySpec.auxiliaryLines) {
    elements.push(...buildStructureLine(ln, pts, 'white', 0.42, false));
  }

  // å¿é¡»æ¾ç¤ºççº¢è²ç»æçº¿
  for (const ln of displaySpec.mustShowLines) {
    elements.push(...buildStructureLine(ln, pts, 'red', 0.85, false));
  }

  // ç»¿è²ç®æ çº¿ï¼ä»å½æç¹éè¦ showGreen æ¶ï¼
  if (phasePoints.some(p => p.showGreen)) {
    for (const ln of displaySpec.mustShowLines) {
      elements.push(...buildGreenLine(ln, pts, phasePoints));
    }
  }

  // æ¯ä¸ªè§èç¹ä½ï¼çº¢ç¹ + ç»¿ç¹ + ç®­å¤´
  for (const ps of phasePoints) {
    const redPt = pts[ps.point];
    if (!redPt) continue;

    if (ps.showRed) {
      elements.push(mkDot(redPt.x, redPt.y, 'red', ps.priority === 'must' ? 0.009 : 0.007, 0.95));
    }

    if (ps.showGreen && ps.greenDirection) {
      const greenPt = applyCorrection(redPt, ps.greenDirection, ps.greenMagnitude ?? 'medium');
      elements.push(mkDot(greenPt.x, greenPt.y, 'green', 0.008, 0.88));

      if (ps.showArrow && Math.hypot(greenPt.x - redPt.x, greenPt.y - redPt.y) > 0.012) {
        elements.push(mkArrow(redPt.x, redPt.y, greenPt.x, greenPt.y));
      }
    }
  }

  // impact æèä¸è§
  if (displaySpec.showImpactTriangle && phase === 'impact') {
    elements.push(...buildArmTriangle(pts, 'red', 0.82));
    const grip = pts.gripCenter;
    if (grip) {
      const greenGrip = applyCorrection(grip, 'more_inside', 'small');
      elements.push(...buildArmTriangle({ ...pts, gripCenter: greenGrip }, 'green', 0.68));
    }
  }

  // åå²è·¯å¾
  if (historyPts && historyPts.length >= 2) {
    elements.push(mkCurve(historyPts, 'red', 3.0, 0.70));
  }

  // é¶æ®µæ ç­¾
  elements.push(mkLabel(0.50, 0.06, phase.toUpperCase(), 'white', 10));

  return elements;
}

/* ââ è·¯å¾è¿½è¸ªï¼è·åå½åå¸§è¿½è¸ªç¹åæ  ââ */
export function getTrackedPoint(
  issue: MainIssueType, viewType: ViewType, kpFrame: KeypointFrame,
): { x: number; y: number } | null {
  const displaySpec = getIssueDisplaySpec(issue, viewType);
  if (!displaySpec?.paths.length) return null;
  const pts = getKeypoints(kpFrame);
  const pt = pts[displaySpec.paths[0].trackedPoint];
  return pt ? { x: pt.x, y: pt.y } : null;
}

/* ââ nearest-frame æ¥æ¾ ââ */
export function findNearestFrame(frames: KeypointFrame[], time: number): KeypointFrame | null {
  if (!frames.length) return null;
  let best = frames[0], bestDist = Math.abs(time - best.time);
  for (const f of frames) {
    const d = Math.abs(time - f.time);
    if (d < bestDist) { best = f; bestDist = d; }
  }
  return best;
}
