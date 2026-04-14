/**
 * OverlayRenderer.ts
 *
 * 统一 canvas 绘图组件库。
 * 所有坐标输入为归一化 0-1，乘以 canvas 宽高得到像素坐标。
 * 视觉风格完全对齐参考图（大填充圆点 + 清晰结构线 + 曲线路径 + 方向箭头）。
 */

import type {
  OverlayElement,
  LineElement, CurveElement, ArrowElement,
  DotElement, LabelElement, BadgeElement, ZoneElement,
} from '@/types/analysis';

type Ctx = CanvasRenderingContext2D;
type Pt = { x: number; y: number };

/* ══════════════════════════════════════════════
   COLOR PALETTE
══════════════════════════════════════════════ */
export const COLORS = {
  red:    '#ff3c3c',
  green:  '#3cee3c',
  yellow: '#ffd040',
  white:  'rgba(255,255,255,0.90)',
  black:  'rgba(0,0,0,0.80)',
} as const;

function resolveColor(c?: string): string {
  if (c === 'red')    return COLORS.red;
  if (c === 'green')  return COLORS.green;
  if (c === 'yellow') return COLORS.yellow;
  return COLORS.white;
}

/* ══════════════════════════════════════════════
   PRIMITIVES
══════════════════════════════════════════════ */

/**
 * JointDot — 关节圆点，视觉核心（参考图 Image 1 大圆点）
 * 填充色 + 黑色外环 + 阴影，确保在任何视频背景上都清晰
 */
export function drawJointDot(
  ctx: Ctx,
  x: number, y: number,    // 归一化
  W: number, H: number,
  color: string,
  radius: number = 0.028,  // 归一化半径
  opacity: number = 0.95
) {
  const px = x * W, py = y * H;
  const r = radius * Math.min(W, H);

  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.shadowColor = 'rgba(0,0,0,0.75)';
  ctx.shadowBlur = 8;

  // Fill
  ctx.beginPath();
  ctx.arc(px, py, r, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();

  // Outer ring
  ctx.shadowBlur = 0;
  ctx.strokeStyle = 'rgba(0,0,0,0.60)';
  ctx.lineWidth = Math.max(1.5, r * 0.22);
  ctx.stroke();

  ctx.restore();
}

/**
 * GuideLine / AxisLine — 结构线（肩线、髋线、脊柱线）
 */
export function drawStructureLine(
  ctx: Ctx,
  x1: number, y1: number, x2: number, y2: number,
  W: number, H: number,
  color: string,
  strokeWidth: number = 3,
  opacity: number = 0.85,
  dashed: boolean = false
) {
  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.strokeStyle = color;
  ctx.lineWidth = strokeWidth * (Math.min(W, H) / 320);
  ctx.lineCap = 'round';
  ctx.shadowColor = 'rgba(0,0,0,0.65)';
  ctx.shadowBlur = 5;

  if (dashed) ctx.setLineDash([strokeWidth * 2.5, strokeWidth * 1.5]);
  ctx.beginPath();
  ctx.moveTo(x1 * W, y1 * H);
  ctx.lineTo(x2 * W, y2 * H);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
}

/**
 * CurvePath — 平滑曲线路径（手路径、杆头轨迹、参考图 Image 2）
 * 贝塞尔曲线，末端有运动圆点
 */
export function drawCurvePath(
  ctx: Ctx,
  points: Pt[],
  W: number, H: number,
  color: string,
  strokeWidth: number = 3.5,
  opacity: number = 0.88
) {
  if (points.length < 2) return;

  const px = (pt: Pt): [number, number] => [pt.x * W, pt.y * H];
  const lw = strokeWidth * (Math.min(W, H) / 320);

  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.strokeStyle = color;
  ctx.lineWidth = lw;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.shadowColor = 'rgba(0,0,0,0.60)';
  ctx.shadowBlur = 6;

  ctx.beginPath();
  const [sx, sy] = px(points[0]);
  ctx.moveTo(sx, sy);

  for (let i = 1; i < points.length - 1; i++) {
    const [cx, cy] = px(points[i]);
    const [nx, ny] = px(points[i + 1]);
    ctx.quadraticCurveTo(cx, cy, (cx + nx) / 2, (cy + ny) / 2);
  }
  const last = px(points[points.length - 1]);
  ctx.lineTo(last[0], last[1]);
  ctx.stroke();

  // Terminal dot — shows current position
  ctx.shadowBlur = 0;
  ctx.beginPath();
  ctx.arc(last[0], last[1], lw * 2, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();

  ctx.restore();
}

/**
 * ArrowMarker — 方向箭头（参考图 Image 3：该往哪移动）
 */
export function drawArrow(
  ctx: Ctx,
  fromX: number, fromY: number,
  toX: number, toY: number,
  W: number, H: number,
  color: string,
  strokeWidth: number = 3,
  opacity: number = 0.92
) {
  const fx = fromX * W, fy = fromY * H;
  const tx = toX * W, ty = toY * H;
  const angle = Math.atan2(ty - fy, tx - fx);
  const dist = Math.hypot(tx - fx, ty - fy);
  const headLen = Math.min(22, dist * 0.42);
  const lw = strokeWidth * (Math.min(W, H) / 320);

  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = lw;
  ctx.lineCap = 'round';
  ctx.shadowColor = 'rgba(0,0,0,0.70)';
  ctx.shadowBlur = 6;

  // Shaft
  ctx.beginPath();
  ctx.moveTo(fx, fy);
  ctx.lineTo(tx, ty);
  ctx.stroke();

  // Head
  ctx.shadowBlur = 0;
  ctx.beginPath();
  ctx.moveTo(tx, ty);
  ctx.lineTo(
    tx - headLen * Math.cos(angle - 0.40),
    ty - headLen * Math.sin(angle - 0.40)
  );
  ctx.lineTo(
    tx - headLen * Math.cos(angle + 0.40),
    ty - headLen * Math.sin(angle + 0.40)
  );
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

/**
 * LabelTag — 文字标签（尽量少用，最多每帧 2 个）
 */
export function drawLabel(
  ctx: Ctx,
  x: number, y: number,
  W: number, H: number,
  text: string,
  color: string,
  fontSize: number = 12,
  opacity: number = 0.92
) {
  const size = fontSize * Math.min(W, H) / 320;
  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.font = `800 ${size}px "DM Sans", system-ui, sans-serif`;
  ctx.textAlign = 'center';
  ctx.shadowColor = 'rgba(0,0,0,0.95)';
  ctx.shadowBlur = 7;
  ctx.fillStyle = color;
  ctx.fillText(text, x * W, y * H);
  ctx.restore();
}

/**
 * CorrectBadge / WrongBadge — 大对勾 / 大错号
 */
export function drawBadge(
  ctx: Ctx,
  x: number, y: number,
  W: number, H: number,
  variant: 'correct' | 'wrong',
  opacity: number = 0.9
) {
  const px = x * W, py = y * H;
  const r = Math.min(W, H) * 0.055;

  ctx.save();
  ctx.globalAlpha = opacity;

  // Circle background
  ctx.beginPath();
  ctx.arc(px, py, r, 0, Math.PI * 2);
  ctx.fillStyle = variant === 'correct' ? COLORS.green : COLORS.red;
  ctx.fill();

  // Symbol
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = r * 0.24;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.beginPath();
  if (variant === 'correct') {
    // ✓
    ctx.moveTo(px - r * 0.5, py);
    ctx.lineTo(px - r * 0.1, py + r * 0.4);
    ctx.lineTo(px + r * 0.55, py - r * 0.35);
  } else {
    // ✗
    ctx.moveTo(px - r * 0.4, py - r * 0.4);
    ctx.lineTo(px + r * 0.4, py + r * 0.4);
    ctx.moveTo(px + r * 0.4, py - r * 0.4);
    ctx.lineTo(px - r * 0.4, py + r * 0.4);
  }
  ctx.stroke();
  ctx.restore();
}

/**
 * HighlightZone — 区域高亮
 */
export function drawZone(
  ctx: Ctx,
  points: Pt[],
  W: number, H: number,
  color: string,
  fillOpacity: number = 0.12,
  strokeOpacity: number = 0.5
) {
  if (points.length < 3) return;
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(points[0].x * W, points[0].y * H);
  for (let i = 1; i < points.length; i++) {
    ctx.lineTo(points[i].x * W, points[i].y * H);
  }
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.globalAlpha = fillOpacity;
  ctx.fill();
  ctx.globalAlpha = strokeOpacity;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.restore();
}

/* ══════════════════════════════════════════════
   MAIN DISPATCHER
   单个 OverlayElement → 对应绘图函数
══════════════════════════════════════════════ */

export function renderElement(
  ctx: Ctx,
  el: OverlayElement,
  W: number, H: number,
  layer: string = 'all'
) {
  // Layer filter
  if (el.layer && el.layer !== 'all' && layer !== 'all' && el.layer !== layer) return;

  const color = resolveColor(el.color);
  const opacity = el.opacity ?? 0.9;

  switch (el.type) {
    case 'line': {
      const e = el as LineElement;
      drawStructureLine(ctx, e.x1, e.y1, e.x2, e.y2, W, H, color, e.strokeWidth ?? 3, opacity, e.dashed);
      break;
    }
    case 'curve': {
      const e = el as CurveElement;
      drawCurvePath(ctx, e.points, W, H, color, e.strokeWidth ?? 3.5, opacity);
      break;
    }
    case 'arrow': {
      const e = el as ArrowElement;
      drawArrow(ctx, e.from.x, e.from.y, e.to.x, e.to.y, W, H, color, e.strokeWidth ?? 3, opacity);
      break;
    }
    case 'dot': {
      const e = el as DotElement;
      drawJointDot(ctx, e.x, e.y, W, H, color, e.radius ?? 0.028, opacity);
      break;
    }
    case 'label': {
      const e = el as LabelElement;
      drawLabel(ctx, e.x, e.y, W, H, e.text, color, e.size ?? 12, opacity);
      break;
    }
    case 'badge': {
      const e = el as BadgeElement;
      drawBadge(ctx, e.x, e.y, W, H, e.variant, opacity);
      break;
    }
    case 'zone': {
      const e = el as ZoneElement;
      drawZone(ctx, e.points, W, H, color, e.fillOpacity ?? 0.12);
      break;
    }
  }
}

/**
 * 渲染一帧的所有 elements
 */
export function renderFrame(
  ctx: Ctx,
  elements: OverlayElement[],
  W: number, H: number,
  layer: string = 'all'
) {
  ctx.clearRect(0, 0, W, H);
  for (const el of elements) {
    renderElement(ctx, el, W, H, layer);
  }
}
