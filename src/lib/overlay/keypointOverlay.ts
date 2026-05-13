/**
 * keypointOverlay.ts
 *
 * 高尔夫分部位可视化系统
 *
 * Layer 分层：
 *   shoulders — 肩部旋转盘（椭圆）
 *   hips      — 髋部旋转盘（椭圆）
 *   head      — 头部轨迹（圆圈 + 路径）
 *   plane     — 挥杆平面（弧线）
 *   all       — 全部叠加
 *
 * 颜色规则：
 *   红色 = 用户当前轨迹/位置
 *   绿色 = 正确目标轨迹/位置
 */

import type { OverlayElement, KeypointFrame } from '@/types/analysis';
import type { MainIssueType } from '@/types/analysis';
import type { BodyPointName, Pt } from '@/lib/golf/bodyPointSpec';
import type { ViewType } from '@/lib/golf/overlayLineSpec';

type Color = 'red' | 'green' | 'yellow' | 'white';

/* ── element 构建器 ── */
let _uid = 0;
const uid = (p: string) => `${p}-${++_uid}`;

const mkDot = (x: number, y: number, color: Color, r = 0.007, opacity = 0.92, layer: OverlayElement['layer'] = 'body'): OverlayElement =>
  ({ type: 'dot', id: uid('d'), x, y, color, radius: r, opacity, layer });

const mkLine = (x1: number, y1: number, x2: number, y2: number, color: Color, w = 1.0, opacity = 0.85, dashed = false, layer: OverlayElement['layer'] = 'body'): OverlayElement =>
  ({ type: 'line', id: uid('l'), x1, y1, x2, y2, color, strokeWidth: w, opacity, dashed, layer });

const mkArrow = (fx: number, fy: number, tx: number, ty: number, color: Color = 'green'): OverlayElement =>
  ({ type: 'arrow', id: uid('a'), from: { x: fx, y: fy }, to: { x: tx, y: ty }, color, strokeWidth: 1.2, opacity: 0.88 });

const mkLabel = (x: number, y: number, text: string, color: Color = 'white', size = 9): OverlayElement =>
  ({ type: 'label', id: uid('t'), x, y, text, color, size, opacity: 0.82 });

const mkCurve = (points: {x:number;y:number}[], color: Color, w = 1.2, opacity = 0.80): OverlayElement =>
  ({ type: 'curve', id: uid('c'), points, color, strokeWidth: w, opacity, layer: 'arms' });

/* ── ellipse element（新类型，OverlayRenderer 需支持）── */
const mkEllipse = (cx: number, cy: number, rx: number, ry: number, angle: number, color: Color, w = 1.5, opacity = 0.85, layer: OverlayElement['layer'] = 'body'): OverlayElement =>
  ({ type: 'ellipse' as OverlayElement['type'], id: uid('e'), cx, cy, rx, ry, angle, color, strokeWidth: w, opacity, layer } as unknown as OverlayElement);

/* ── 从 KeypointFrame 解析关节点 ── */
export function getKeypoints(kpFrame: KeypointFrame): Partial<Record<BodyPointName, Pt>> {
  const lm = kpFrame.landmarks;
  const result: Partial<Record<BodyPointName, Pt>> = {};
  const toP = (pt?: { x:number; y:number; confidence?:number } | null): Pt | null =>
    pt ? { x: pt.x, y: pt.y, confidence: pt.confidence ?? 0.8 } : null;

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

  const ls = result.leftShoulder, rs = result.rightShoulder;
  if (ls && rs) result.shoulderCenter = { x:(ls.x+rs.x)/2, y:(ls.y+rs.y)/2, confidence:1 };
  const lh = result.leftHip, rh = result.rightHip;
  if (lh && rh) result.hipCenter = { x:(lh.x+rh.x)/2, y:(lh.y+rh.y)/2, confidence:1 };
  const lw = result.leftWrist, rw = result.rightWrist;
  if (lw && rw) result.gripCenter = { x:(lw.x+rw.x)/2, y:(lw.y+rw.y)/2, confidence:1 };

  return result;
}

/* ══════════════════════════════════════════
  LAYER 1: 肩部旋转盘
  - 红色椭圆 = 当前肩线旋转平面
  - 绿色椭圆 = 目标（更平的旋转平面）
  - 白色标注旋转角度
══════════════════════════════════════════ */
function buildShoulderLayer(pts: Partial<Record<BodyPointName, Pt>>, historyPts?: Array<{x:number;y:number}>): OverlayElement[] {
  const els: OverlayElement[] = [];
  const ls = pts.leftShoulder, rs = pts.rightShoulder;
  if (!ls || !rs) return els;

  const cx = (ls.x + rs.x) / 2;
  const cy = (ls.y + rs.y) / 2;
  const dx = rs.x - ls.x;
  const dy = rs.y - ls.y;
  const angle = Math.atan2(dy, dx); // 实际肩线角度（弧度）
  const dist = Math.hypot(dx, dy);
  const rx = dist / 2;             // 椭圆长轴 = 肩宽一半
  const ry = rx * 0.22;            // 椭圆短轴 = 旋转盘的"厚度"

  // 红色椭圆 = 当前肩部旋转平面
  els.push(mkEllipse(cx, cy, rx, ry, angle, 'red', 1.8, 0.88, 'body'));

  // 目标角度：比当前更平（接近水平）
  const targetAngle = angle * 0.4; // 减少倾斜
  const targetCy = cy + 0.01;
  els.push(mkEllipse(cx, targetCy, rx * 1.05, ry * 0.85, targetAngle, 'green', 1.5, 0.75, 'body'));

  // 肩部关节点
  els.push(mkDot(ls.x, ls.y, 'red', 0.009, 0.92));
  els.push(mkDot(rs.x, rs.y, 'red', 0.009, 0.92));
  els.push(mkDot(cx, cy, 'white', 0.005, 0.60));

  // 角度标注
  const deg = Math.round(Math.abs(angle * 180 / Math.PI));
  els.push(mkLabel(cx, cy - 0.06, `${deg}°`, 'white', 9));
  els.push(mkLabel(cx, cy - 0.10, 'SHOULDERS', 'white', 8));

  // 肩部历史路径（追踪肩膀中心移动）
  if (historyPts && historyPts.length >= 2) {
    els.push(mkCurve(historyPts, 'red', 1.0, 0.65));
  }

  return els;
}

/* ══════════════════════════════════════════
  LAYER 2: 髋部旋转盘
  - 红色椭圆 = 当前髋线旋转平面
  - 绿色椭圆 = 目标（髋部旋转应领先肩部）
  - X-Factor: 髋比肩多转15-20°
══════════════════════════════════════════ */
function buildHipLayer(pts: Partial<Record<BodyPointName, Pt>>, shoulderAngle: number): OverlayElement[] {
  const els: OverlayElement[] = [];
  const lh = pts.leftHip, rh = pts.rightHip;
  if (!lh || !rh) return els;

  const cx = (lh.x + rh.x) / 2;
  const cy = (lh.y + rh.y) / 2;
  const dx = rh.x - lh.x;
  const dy = rh.y - lh.y;
  const angle = Math.atan2(dy, dx);
  const dist = Math.hypot(dx, dy);
  const rx = dist / 2;
  const ry = rx * 0.20;

  // 红色椭圆 = 当前髋部平面
  els.push(mkEllipse(cx, cy, rx, ry, angle, 'red', 1.8, 0.88, 'body'));

  // 绿色椭圆 = 目标（髋部应该转更多，角度更平）
  const targetAngle = angle * 0.3;
  els.push(mkEllipse(cx, cy + 0.01, rx * 1.08, ry * 0.80, targetAngle, 'green', 1.5, 0.72, 'body'));

  // 髋部关节点
  els.push(mkDot(lh.x, lh.y, 'red', 0.009, 0.92));
  els.push(mkDot(rh.x, rh.y, 'red', 0.009, 0.92));

  // X-Factor 标注（髋与肩的旋转差）
  const hipDeg = Math.round(Math.abs(angle * 180 / Math.PI));
  const shoulderDeg = Math.round(Math.abs(shoulderAngle * 180 / Math.PI));
  const xFactor = Math.abs(shoulderDeg - hipDeg);
  els.push(mkLabel(cx, cy - 0.04, `HIPS ${hipDeg}°`, 'white', 9));
  if (xFactor > 0) {
    els.push(mkLabel(cx, cy - 0.08, `X: ${xFactor}°`, 'yellow', 8));
  }

  return els;
}

/* ══════════════════════════════════════════
  LAYER 3: 头部追踪
  - 绿色圆圈 = 头部应该保持的位置（地址位）
  - 红色圆圈 + 轨迹 = 头部实际移动路径
  - 横向偏移量标注
══════════════════════════════════════════ */
function buildHeadLayer(pts: Partial<Record<BodyPointName, Pt>>, headHistory: Array<{x:number;y:number}>): OverlayElement[] {
  const els: OverlayElement[] = [];
  const head = pts.headCenter;
  if (!head) return els;

  // 绿色圆圈 = 理想位置（用历史第一帧作为基准）
  const baseX = headHistory.length > 0 ? headHistory[0].x : head.x;
  const baseY = headHistory.length > 0 ? headHistory[0].y : head.y;

  // 绘制绿色目标圆圈（较大、半透明）
  els.push(mkEllipse(baseX, baseY, 0.055, 0.055, 0, 'green', 1.2, 0.55, 'body'));
  // 绿色十字标记中心
  els.push(mkDot(baseX, baseY, 'green', 0.006, 0.70));

  // 红色当前头部位置
  els.push(mkDot(head.x, head.y, 'red', 0.010, 0.95));

  // 头部移动轨迹
  if (headHistory.length >= 2) {
    els.push(mkCurve(headHistory, 'red', 1.0, 0.72));
  }

  // 横向偏移箭头（如果偏移明显）
  const offsetX = head.x - baseX;
  if (Math.abs(offsetX) > 0.015) {
    els.push(mkArrow(baseX, head.y, head.x, head.y, offsetX > 0 ? 'red' : 'red'));
    const offsetPx = Math.round(Math.abs(offsetX) * 100);
    els.push(mkLabel(head.x, head.y - 0.06, `±${offsetPx}%`, 'red', 8));
  }

  els.push(mkLabel(baseX, baseY - 0.09, 'HEAD', 'white', 8));

  return els;
}

/* ══════════════════════════════════════════
  LAYER 4: 挥杆平面
  - 红色弧线 = 手部实际运动轨迹
  - 绿色线 = 标准 Ben Hogan 挥杆平面
  - 双臂三角
══════════════════════════════════════════ */
function buildSwingPlaneLayer(pts: Partial<Record<BodyPointName, Pt>>, handHistory: Array<{x:number;y:number}>): OverlayElement[] {
  const els: OverlayElement[] = [];
  const grip = pts.gripCenter;
  const ls = pts.leftShoulder, rs = pts.rightShoulder;

  // 标准挥杆平面线（肩部连到球的理想角度）
  if (ls && rs) {
    const sc = { x:(ls.x+rs.x)/2, y:(ls.y+rs.y)/2 };
    // 绿色挥杆平面参考线（从肩中心向下延伸到球的方向）
    els.push(mkLine(sc.x, sc.y, sc.x + 0.02, sc.y + 0.30, 'green', 1.2, 0.65, true, 'arms'));
    els.push(mkLabel(sc.x + 0.06, sc.y + 0.15, 'PLANE', 'green', 8));

    // 双臂三角
    if (grip) {
      els.push(mkLine(ls.x, ls.y, grip.x, grip.y, 'red', 1.0, 0.80, false, 'arms'));
      els.push(mkLine(rs.x, rs.y, grip.x, grip.y, 'red', 1.0, 0.80, false, 'arms'));
      els.push(mkLine(ls.x, ls.y, rs.x, rs.y, 'red', 1.0, 0.75, false, 'arms'));
      // 绿色目标三角（更靠近身体）
      const gGrip = { x: grip.x + (sc.x - grip.x) * 0.12, y: grip.y - 0.02 };
      els.push(mkLine(ls.x, ls.y, gGrip.x, gGrip.y, 'green', 1.0, 0.68, true, 'arms'));
      els.push(mkLine(rs.x, rs.y, gGrip.x, gGrip.y, 'green', 1.0, 0.68, true, 'arms'));
      els.push(mkDot(grip.x, grip.y, 'red', 0.010, 0.95, 'arms'));
      els.push(mkDot(gGrip.x, gGrip.y, 'green', 0.008, 0.82, 'arms'));
    }
  }

  // 手部实际运动轨迹（红色曲线）
  if (handHistory.length >= 2) {
    els.push(mkCurve(handHistory, 'red', 1.2, 0.80));
  }

  // 关节点
  if (ls) els.push(mkDot(ls.x, ls.y, 'red', 0.007, 0.88, 'arms'));
  if (rs) els.push(mkDot(rs.x, rs.y, 'red', 0.007, 0.88, 'arms'));

  return els;
}

/* ══════════════════════════════════════════
  主导出函数：按 layer 分路输出
══════════════════════════════════════════ */
export function generateSpecDrivenOverlayFrame(
  issue: MainIssueType,
  viewType: ViewType,
  phase: string,
  kpFrame: KeypointFrame,
  historyPts?: Array<{ x: number; y: number }>,
): OverlayElement[] {
  _uid = 0;
  const pts = getKeypoints(kpFrame);
  const elements: OverlayElement[] = [];

  // 计算肩部角度（供髋部X-Factor使用）
  const ls = pts.leftShoulder, rs = pts.rightShoulder;
  let shoulderAngle = 0;
  if (ls && rs) {
    shoulderAngle = Math.atan2(rs.y - ls.y, rs.x - ls.x);
  }

  // LAYER: shoulders — 肩部旋转盘
  elements.push(...buildShoulderLayer(pts, historyPts));

  // LAYER: hips (club) — 髋部旋转盘
  elements.push(...buildHipLayer(pts, shoulderAngle));

  // LAYER: head (body) — 头部追踪
  const headHistory = historyPts ? historyPts.slice(-8) : [];
  elements.push(...buildHeadLayer(pts, headHistory));

  // LAYER: plane (arms) — 挥杆平面
  elements.push(...buildSwingPlaneLayer(pts, historyPts || []));

  // 全身关键点（轻量辅助）
  const allJoints: BodyPointName[] = [
    'leftKnee','rightKnee','leftAnkle','rightAnkle',
    'leftHip','rightHip',
  ];
  for (const pn of allJoints) {
    const p = pts[pn];
    if (p) elements.push(mkDot(p.x, p.y, 'red', 0.006, 0.70, 'body'));
  }

  // 腿部连线（辅助参考）
  const lh = pts.leftHip, rh = pts.rightHip;
  const lk = pts.leftKnee, rk = pts.rightKnee;
  const la = pts.leftAnkle, ra = pts.rightAnkle;
  if (lh && lk) elements.push(mkLine(lh.x, lh.y, lk.x, lk.y, 'red', 0.8, 0.55, false, 'body'));
  if (lk && la) elements.push(mkLine(lk.x, lk.y, la.x, la.y, 'red', 0.8, 0.55, false, 'body'));
  if (rh && rk) elements.push(mkLine(rh.x, rh.y, rk.x, rk.y, 'red', 0.8, 0.55, false, 'body'));
  if (rk && ra) elements.push(mkLine(rk.x, rk.y, ra.x, ra.y, 'red', 0.8, 0.55, false, 'body'));

  return elements;
}

/* ── 路径追踪点 ── */
export function getTrackedPoint(
  issue: MainIssueType, viewType: ViewType, kpFrame: KeypointFrame,
): { x: number; y: number } | null {
  const pts = getKeypoints(kpFrame);
  // 追踪肩膀中心（最能体现旋转）
  const sc = pts.shoulderCenter;
  return sc ? { x: sc.x, y: sc.y } : null;
}

/* ── 最近帧查找 ── */
export function findNearestFrame(frames: KeypointFrame[], time: number): KeypointFrame | null {
  if (!frames.length) return null;
  let best = frames[0], bestDist = Math.abs(time - best.time);
  for (const f of frames) {
    const d = Math.abs(time - f.time);
    if (d < bestDist) { best = f; bestDist = d; }
  }
  return best;
}

/* ── applyCorrection (保持兼容性) ── */
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
