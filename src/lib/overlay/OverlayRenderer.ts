/**
 * OverlayRenderer.ts
 *
 * AlignSnow 风格：极细线条 + 极小关节点，干净简洁。
 * 归一化坐标输入 (0-1)，乘以 canvas 宽高得到像素坐标。
 */

import type {
  OverlayElement,
  LineElement, CurveElement, ArrowElement,
  DotElement, LabelElement, BadgeElement, ZoneElement,
} from '@/types/analysis';

type Ctx = CanvasRenderingContext2D;
type Pt = { x: number; y: number };

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

/**
 * JointDot — 极小关节点（AlignSnow 风格）
 * radius 默认 0.008（比原来 0.028 小了 3.5 倍）
 */
export function drawJointDot(
  ctx: Ctx,
  x: number, y: number,
  W: number, H: number,
  color: string,
  radius: number = 0.008,
  opacity: number = 0.95,
) {
  const px = x * W, py = y * H;
  const r = radius * Math.min(W, H);
  if (r < 0.5) return;

  ctx.save();
  ctx.globalAlpha = opacity;

  ctx.beginPath();
  ctx.arc(px, py, r, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();

  // 极细外环增加对比
  ctx.strokeStyle = 'rgba(0,0,0,0.45)';
  ctx.lineWidth = Math.max(0.5, r * 0.15);
  ctx.stroke();

  ctx.restore();
}

/**
 * StructureLine — 极细骨骼连线（AlignSnow 风格）
 * strokeWidth 默认 1.0（比原来 3.0 小了 3 倍）
 */
export function drawStructureLine(
  ctx: Ctx,
  x1: number, y1: number, x2: number, y2: number,
  W: number, H: number,
  color: string,
  strokeWidth: number = 1.0,
  opacity: number = 0.85,
  dashed: boolean = false,
) {
  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.strokeStyle = color;
  ctx.lineWidth = strokeWidth * (Math.min(W, H) / 320);
  ctx.lineCap = 'round';
  ctx.shadowColor = 'rgba(0,0,0,0.40)';
  ctx.shadowBlur = 2;

  if (dashed) ctx.setLineDash([strokeWidth * 2, strokeWidth * 2]);
  ctx.beginPath();
  ctx.moveTo(x1 * W, y1 * H);
  ctx.lineTo(x2 * W, y2 * H);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
}

/**
 * CurvePath — 路径曲线（手/杆头轨迹）
 */
export function drawCurvePath(
  ctx: Ctx,
  points: Pt[],
  W: number, H: number,
  color: string,
  strokeWidth: number = 1.5,
  opacity: number = 0.80,
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
  ctx.shadowColor = 'rgba(0,0,0,0.40)';
  ctx.shadowBlur = 2;

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

  // 末端极小点
  ctx.shadowBlur = 0;
  ctx.beginPath();
  ctx.arc(last[0], last[1], lw * 1.5, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();

  ctx.restore();
}

/**
 * Arrow — 方向箭头
 */
export function drawArrow(
  ctx: Ctx,
  fromX: number, fromY: number,
  toX: number, toY: number,
  W: number, H: number,
  color: string,
  strokeWidth: number = 1.5,
  opacity: number = 0.90,
) {
  const fx = fromX * W, fy = fromY * H;
  const tx = toX * W, ty = toY * H;
  const angle = Math.atan2(ty - fy, tx - fx);
  const dist = Math.hypot(tx - fx, ty - fy);
  const headLen = Math.min(14, dist * 0.38);
  const lw = strokeWidth * (Math.min(W, H) / 320);

  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = lw;
  ctx.lineCap = 'round';
  ctx.shadowColor = 'rgba(0,0,0,0.50)';
  ctx.shadowBlur = 3;

  ctx.beginPath();
  ctx.moveTo(fx, fy);
  ctx.lineTo(tx, ty);
  ctx.stroke();

  ctx.shadowBlur = 0;
  ctx.beginPath();
  ctx.moveTo(tx, ty);
  ctx.lineTo(tx - headLen * Math.cos(angle - 0.40), ty - headLen * Math.sin(angle - 0.40));
  ctx.lineTo(tx - headLen * Math.cos(angle + 0.40), ty - headLen * Math.sin(angle + 0.40));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

/**
 * Label — 文字标签
 */
export function drawLabel(
  ctx: Ctx,
  x: number, y: number,
  W: number, H: number,
  text: string,
  color: string,
  fontSize: number = 10,
  opacity: number = 0.88,
) {
  const size = fontSize * Math.min(W, H) / 320;
  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.font = `700 ${size}px "DM Sans", system-ui, sans-serif`;
  ctx.textAlign = 'center';
  ctx.shadowColor = 'rgba(0,0,0,0.90)';
  ctx.shadowBlur = 5;
  ctx.fillStyle = color;
  ctx.fillText(text, x * W, y * H);
  ctx.restore();
}

/**
 * Badge — 对勾/错号
 */
export function drawBadge(
  ctx: Ctx,
  x: number, y: number,
  W: number, H: number,
  variant: 'correct' | 'wrong',
  opacity: number = 0.88,
) {
  const px = x * W, py = y * H;
  const r = Math.min(W, H) * 0.032;

  ctx.save();
  ctx.globalAlpha = opacity;

  ctx.beginPath();
  ctx.arc(px, py, r, 0, Math.PI * 2);
  ctx.fillStyle = variant === 'correct' ? COLORS.green : COLORS.red;
  ctx.fill();

  ctx.strokeStyle = '#fff';
  ctx.lineWidth = r * 0.22;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.beginPath();
  if (variant === 'correct') {
    ctx.moveTo(px - r * 0.5, py);
    ctx.lineTo(px - r * 0.1, py + r * 0.4);
    ctx.lineTo(px + r * 0.55, py - r * 0.35);
  } else {
    ctx.moveTo(px - r * 0.4, py - r * 0.4);
    ctx.lineTo(px + r * 0.4, py + r * 0.4);
    ctx.moveTo(px + r * 0.4, py - r * 0.4);
    ctx.lineTo(px - r * 0.4, py + r * 0.4);
  }
  ctx.stroke();
  ctx.restore();
}

/**
 * Zone — 区域高亮
 */
export function drawZone(
  ctx: Ctx,
  points: Pt[],
  W: number, H: number,
  color: string,
  fillOpacity: number = 0.08,
  strokeOpacity: number = 0.35,
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
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.restore();
}

/* ══ 主分发函数 ══ */
export function renderElement(
  ctx: Ctx,
  el: OverlayElement,
  W: number, H: number,
  layer: string = 'all',
) {
  if (el.layer && el.layer !== 'all' && layer !== 'all' && el.layer !== layer) return;

  const color = resolveColor(el.color);
  const opacity = el.opacity ?? 0.88;

  switch (el.type) {
    case 'line': {
      const e = el as LineElement;
      drawStructureLine(ctx, e.x1, e.y1, e.x2, e.y2, W, H, color, e.strokeWidth ?? 1.0, opacity, e.dashed);
      break;
    }
    case 'curve': {
      const e = el as CurveElement;
      drawCurvePath(ctx, e.points, W, H, color, e.strokeWidth ?? 1.5, opacity);
      break;
    }
    case 'arrow': {
      const e = el as ArrowElement;
      drawArrow(ctx, e.from.x, e.from.y, e.to.x, e.to.y, W, H, color, e.strokeWidth ?? 1.5, opacity);
      break;
    }
    case 'dot': {
      const e = el as DotElement;
      drawJointDot(ctx, e.x, e.y, W, H, color, e.radius ?? 0.008, opacity);
      break;
    }
    case 'label': {
      const e = el as LabelElement;
      drawLabel(ctx, e.x, e.y, W, H, e.text, color, e.size ?? 10, opacity);
      break;
    }
    case 'badge': {
      const e = el as BadgeElement;
      drawBadge(ctx, e.x, e.y, W, H, e.variant, opacity);
      break;
    }
    case 'zone': {
      const e = el as ZoneElement;
      drawZone(ctx, e.points, W, H, color, e.fillOpacity ?? 0.08);
      break;
    }
  }
}

export function renderFrame(
  ctx: Ctx,
  elements: OverlayElement[],
  W: number, H: number,
  layer: string = 'all',
) {
  ctx.clearRect(0, 0, W, H);
  for (const el of elements) {
    renderElement(ctx, el, W, H, layer);
  }
}
