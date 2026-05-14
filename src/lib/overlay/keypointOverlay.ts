/**
 * keypointOverlay.ts
 *
 * Shoulder / Hip Rotation Disc — 透视绑定修复版
 *
 * 核心原则：
 * 1. 圆盘锚定关键点：rx = bodyDist × 0.78，clamp [rxMin, rxMax]
 * 2. rotationAngle clamped：face_on ±30°，dtl ±45°
 * 3. 帧间平滑：max delta 8° per frame
 * 4. perspectiveRatio 控制 ry（随转身压扁）
 * 5. confidence < 0.4 不画
 * 6. 椭圆用 zone polygon 模拟（兼容现有 OverlayElement 类型）
 * 7. 颜色：current=red, target=green，均在 BaseElement.color 范围内
 */

import type {
  OverlayElement, KeypointFrame,
  LineElement, ZoneElement, DotElement, LabelElement,
} from '@/types/analysis';
import type { MainIssueType } from '@/types/analysis';
import type { BodyPointName, Pt } from '@/lib/golf/bodyPointSpec';
import type { ViewType } from '@/lib/golf/overlayLineSpec';

/* ── 帧间平滑状态 ── */
const _prevAngle: Record<string, number> = {};

let _uid = 0;
const uid = (p: string) => `${p}-${++_uid}`;

const clamp = (v: number, lo: number, hi: number) =>
  Math.max(lo, Math.min(hi, v));
const toRad = (deg: number) => deg * Math.PI / 180;
const dist2D = (a: Pt, b: Pt) => Math.hypot(b.x - a.x, b.y - a.y);
const mid2D  = (a: Pt, b: Pt): Pt => ({
  x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, confidence: 1,
});

type AllowedColor = 'red' | 'green' | 'yellow' | 'white';

const mkLine = (
  x1: number, y1: number, x2: number, y2: number,
  color: AllowedColor, w = 1.8, opacity = 0.85, dashed = false,
  layer: OverlayElement['layer'] = 'body',
): LineElement => ({ type: 'line', id: uid('l'), x1, y1, x2, y2, color, strokeWidth: w, opacity, dashed, layer });

const mkZone = (
  points: Array<{ x: number; y: number }>,
  color: AllowedColor, fillOpacity = 0.14, opacity = 0.88,
  layer: OverlayElement['layer'] = 'body',
): ZoneElement => ({ type: 'zone', id: uid('z'), points, color, fillOpacity, opacity, layer });

const mkDot = (
  x: number, y: number,
  color: AllowedColor, r = 0.009, opacity = 0.90,
  layer: OverlayElement['layer'] = 'body',
): DotElement => ({ type: 'dot', id: uid('d'), x, y, color, radius: r, opacity, layer });

const mkLabel = (
  x: number, y: number, text: string,
  color: AllowedColor = 'white', size = 9, opacity = 0.75,
): LabelElement => ({ type: 'label', id: uid('t'), x, y, text, color, size, opacity });

/* ── 椭圆近似（40 点 polygon）── */
function ellipsePoints(
  cx: number, cy: number,
  rx: number, ry: number,
  angleDeg: number,
  n = 40,
): Array<{ x: number; y: number }> {
  const ar = toRad(angleDeg);
  const cosA = Math.cos(ar), sinA = Math.sin(ar);
  const pts: Array<{ x: number; y: number }> = [];
  for (let i = 0; i < n; i++) {
    const t  = (2 * Math.PI * i) / n;
    const ex = rx * Math.cos(t);
    const ey = ry * Math.sin(t);
    pts.push({ x: cx + ex * cosA - ey * sinA, y: cy + ex * sinA + ey * cosA });
  }
  return pts;
}

/* ══════════════════════════════════════════════════════
   computePerspectiveDisc
══════════════════════════════════════════════════════ */
export interface DiscParams {
  cx: number; cy: number;
  rx: number; ry: number;
  rotationDeg: number;
  guideStart: { x: number; y: number };
  guideEnd:   { x: number; y: number };
  alpha: number;
  visible: boolean;
}

export function computePerspectiveDisc(args: {
  leftPoint:      Pt;
  rightPoint:     Pt;
  viewType:       ViewType;
  kind:           'shoulder' | 'hip';
  previousAngle?: number;
}): DiscParams {
  const { leftPoint: lp, rightPoint: rp, viewType, kind, previousAngle } = args;

  const FAIL: DiscParams = {
    cx: 0, cy: 0, rx: 0, ry: 0, rotationDeg: 0,
    guideStart: { x: 0, y: 0 }, guideEnd: { x: 0, y: 0 },
    alpha: 0, visible: false,
  };

  if ((lp.confidence ?? 0.8) < 0.40 || (rp.confidence ?? 0.8) < 0.40) return FAIL;

  const center   = mid2D(lp, rp);
  const bodyDist = dist2D(lp, rp);

  /* rx: 动态绑定到关键点间距 */
  const rxMin = kind === 'shoulder' ? 0.12 : 0.09;
  const rxMax = kind === 'shoulder' ? 0.32 : 0.24;
  const rx = clamp(bodyDist * 0.78, rxMin, rxMax);

  /* rotation angle: clamp + 帧间平滑 */
  const rawDeg = Math.atan2(rp.y - lp.y, rp.x - lp.x) * 180 / Math.PI;
  const maxAng = viewType === 'face_on' ? 30 : 45;
  const clampedDeg = clamp(rawDeg, -maxAng, maxAng);

  let smoothedDeg = clampedDeg;
  if (previousAngle !== undefined) {
    const delta = clamp(clampedDeg - previousAngle, -8, 8);
    smoothedDeg = previousAngle + delta;
  }

  /* perspective ratio → ry */
  const refW = kind === 'shoulder'
    ? (viewType === 'face_on' ? 0.22 : 0.14)
    : (viewType === 'face_on' ? 0.16 : 0.10);
  const visRatio = clamp(Math.abs(rp.x - lp.x) / refW, 0, 1.0);

  if (visRatio < 0.22) return FAIL;

  const ryRatioMax = viewType === 'face_on'
    ? (kind === 'shoulder' ? 0.36 : 0.32)
    : (kind === 'shoulder' ? 0.28 : 0.24);
  const ryRatio = clamp(ryRatioMax * visRatio, 0.06, ryRatioMax);
  const ry = rx * ryRatio;

  /* guide line 端点（比 rx 长 30%）*/
  const guideExt = rx * 1.30;
  const ar    = toRad(smoothedDeg);
  const cosA  = Math.cos(ar), sinA = Math.sin(ar);
  const guideStart = { x: center.x - guideExt * cosA, y: center.y - guideExt * sinA };
  const guideEnd   = { x: center.x + guideExt * cosA, y: center.y + guideExt * sinA };

  const alpha = 0.45 + visRatio * 0.40;

  return { cx: center.x, cy: center.y, rx, ry, rotationDeg: smoothedDeg, guideStart, guideEnd, alpha, visible: true };
}

/* ── 绘制单个圆盘 ── */
function drawDisc(
  params: DiscParams,
  color: AllowedColor,
  lp: Pt, rp: Pt,
  label: string,
  visRefW: number,
  layer: OverlayElement['layer'] = 'body',
): OverlayElement[] {
  if (!params.visible) return [];
  const { cx, cy, rx, ry, rotationDeg, guideStart, guideEnd, alpha } = params;
  const els: OverlayElement[] = [];

  /* guide line */
  els.push(mkLine(guideStart.x, guideStart.y, guideEnd.x, guideEnd.y, color, 1.8, alpha * 0.85, false, layer));

  /* 椭圆填充 + 轮廓 */
  const pts = ellipsePoints(cx, cy, rx, ry, rotationDeg, 40);
  const fillOp  = color === 'red' ? 0.14 : 0.16;
  els.push(mkZone(pts, color, fillOp, alpha, layer));

  /* 端点 */
  if ((lp.confidence ?? 0.8) > 0.50) els.push(mkDot(lp.x, lp.y, color, 0.009, alpha * 0.92, layer));
  if ((rp.confidence ?? 0.8) > 0.50) els.push(mkDot(rp.x, rp.y, color, 0.009, alpha * 0.92, layer));

  /* label（仅正面时）*/
  const vis = Math.abs(rp.x - lp.x) / visRefW;
  if (vis > 0.60) {
    els.push(mkLabel(cx, cy - ry * 1.7, label + '  ' + Math.round(Math.abs(rotationDeg)) + '°', 'white', 9, 0.72));
  }

  return els;
}

/* ══════════════════════════════════════════════════════
   关键点解析
══════════════════════════════════════════════════════ */
export function getKeypoints(kpFrame: KeypointFrame): Partial<Record<BodyPointName, Pt>> {
  const lm = kpFrame.landmarks;
  const r: Partial<Record<BodyPointName, Pt>> = {};
  const toP = (pt?: { x: number; y: number; confidence?: number } | null) =>
    pt ? { x: pt.x, y: pt.y, confidence: pt.confidence ?? 0.8 } : null;
  if (lm.head)          r.headCenter    = toP(lm.head)!;
  if (lm.leftShoulder)  r.leftShoulder  = toP(lm.leftShoulder)!;
  if (lm.rightShoulder) r.rightShoulder = toP(lm.rightShoulder)!;
  if (lm.leftElbow)     r.leftElbow     = toP(lm.leftElbow)!;
  if (lm.rightElbow)    r.rightElbow    = toP(lm.rightElbow)!;
  if (lm.leftWrist)     r.leftWrist     = toP(lm.leftWrist)!;
  if (lm.rightWrist)    r.rightWrist    = toP(lm.rightWrist)!;
  if (lm.leftHip)       r.leftHip       = toP(lm.leftHip)!;
  if (lm.rightHip)      r.rightHip      = toP(lm.rightHip)!;
  if (lm.leftKnee)      r.leftKnee      = toP(lm.leftKnee)!;
  if (lm.rightKnee)     r.rightKnee     = toP(lm.rightKnee)!;
  if (lm.leftAnkle)     r.leftAnkle     = toP(lm.leftAnkle)!;
  if (lm.rightAnkle)    r.rightAnkle    = toP(lm.rightAnkle)!;
  const ls = r.leftShoulder, rs = r.rightShoulder;
  if (ls && rs) r.shoulderCenter = mid2D(ls, rs);
  const lh = r.leftHip, rh = r.rightHip;
  if (lh && rh) r.hipCenter = mid2D(lh, rh);
  const lw = r.leftWrist, rw = r.rightWrist;
  if (lw && rw) r.gripCenter = mid2D(lw, rw);
  return r;
}

/* ══════════════════════════════════════════════════════
   主生成函数
══════════════════════════════════════════════════════ */
export function generateSpecDrivenOverlayFrame(
  issue: MainIssueType,
  viewType: ViewType,
  phase: string,
  kpFrame: KeypointFrame,
  _historyPts?: Array<{ x: number; y: number }>,
): OverlayElement[] {
  _uid = 0;
  const pts = getKeypoints(kpFrame);
  const els: OverlayElement[] = [];

  /* Shoulder Disc */
  const ls = pts.leftShoulder, rs = pts.rightShoulder;
  if (ls && rs) {
    const sp = computePerspectiveDisc({ leftPoint: ls, rightPoint: rs, viewType, kind: 'shoulder', previousAngle: _prevAngle['shoulder'] });
    if (sp.visible) _prevAngle['shoulder'] = sp.rotationDeg;
    const refW = viewType === 'face_on' ? 0.22 : 0.14;
    els.push(...drawDisc(sp, 'red', ls, rs, 'SHOULDERS', refW, 'body'));
  }

  /* Hip Ring */
  const lh = pts.leftHip, rh = pts.rightHip;
  if (lh && rh) {
    const hp = computePerspectiveDisc({ leftPoint: lh, rightPoint: rh, viewType, kind: 'hip', previousAngle: _prevAngle['hip'] });
    if (hp.visible) _prevAngle['hip'] = hp.rotationDeg;
    const refW = viewType === 'face_on' ? 0.16 : 0.10;
    els.push(...drawDisc(hp, 'red', lh, rh, 'HIPS', refW, 'club'));
  }

  return els;
}

/* ── 兼容性导出 ── */
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

export function applyCorrection(
  pt: { x: number; y: number; confidence?: number }, dir: string, mag = 'medium',
): { x: number; y: number; confidence?: number } {
  const DELTA: Record<string, number> = { small: 0.028, medium: 0.050, large: 0.078 };
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
