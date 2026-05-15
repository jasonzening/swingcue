/**
 * OverlayRenderer.ts Ã¢ÂÂ Ã¦ÂÂÃ¨Â½Â¬Ã§ÂÂÃ¦Â¸Â²Ã¦ÂÂÃ¥ÂÂ¨Ã¯Â¼ÂÃ¦Â Â·Ã¦ÂÂ¿Ã¥ÂÂ¾1:1Ã¥Â¤ÂÃ¥ÂÂ»Ã¯Â¼Â
 *
 * drawEllipse Ã¤Â¸ÂÃ¥Â±ÂÃ¦Â¸Â²Ã¦ÂÂ + 3DÃ¥ÂÂÃ¥ÂÂÃ¥Â¼Â§Ã¥ÂÂºÃ¥ÂÂÃ¯Â¼Â
 *   Ã¥ÂÂÃ¥Â¼Â§Ã¯Â¼ÂÃ¦ÂÂÃ©ÂÂÃ¥Â¤Â´Ã¯Â¼Â= Ã§Â²ÂÃ¥Â®ÂÃ§ÂºÂ¿ + Ã¥Â¼ÂºÃ¥ÂÂÃ¥ÂÂ
 *   Ã¥ÂÂÃ¥Â¼Â§Ã¯Â¼ÂÃ¨ÂÂÃ©ÂÂÃ¥Â¤Â´Ã¯Â¼Â= Ã§Â»ÂÃ¨ÂÂÃ§ÂºÂ¿ + Ã¦Â·Â¡Ã¯Â¼ÂÃ¦Â¨Â¡Ã¦ÂÂÃ¨Â¢Â«Ã¨ÂºÂ«Ã¤Â½ÂÃ©ÂÂ®Ã¦ÂÂ¡Ã¯Â¼Â
 *   Ã¥Â¡Â«Ã¥ÂÂ = Ã¥ÂÂÃ©ÂÂÃ¦ÂÂÃ¨Â¦ÂÃ§ÂÂÃ¦ÂÂ´Ã§ÂÂ
 */

import type {
  OverlayElement,
  LineElement, CurveElement, ArrowElement,
  DotElement, LabelElement, BadgeElement, ZoneElement,
} from '@/types/analysis';

type Ctx = CanvasRenderingContext2D;
type Pt = { x: number; y: number };

export const COLORS = {
  red:    '#ff3030',
  green:  '#32ff50',
  yellow: '#ffe040',
  white:  'rgba(255,255,255,0.92)',
  gray:   'rgba(200,200,200,0.82)',
  black:  'rgba(0,0,0,0.80)',
} as const;

function resolveColor(c?: string): string {
  if (c === 'red')    return COLORS.red;
  if (c === 'green')  return COLORS.green;
  if (c === 'yellow') return COLORS.yellow;
  if (c === 'gray')   return COLORS.gray;
  return COLORS.white;
}

export function drawJointDot(
  ctx: Ctx, x: number, y: number, W: number, H: number,
  color: string, radius = 0.008, opacity = 0.92,
) {
  const px = x * W, py = y * H, r = radius * Math.min(W, H);
  if (r < 0.5) return;
  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI * 2);
  ctx.fillStyle = color; ctx.fill();
  ctx.strokeStyle = 'rgba(0,0,0,0.40)';
  ctx.lineWidth = Math.max(0.5, r * 0.15); ctx.stroke();
  ctx.restore();
}

export function drawStructureLine(
  ctx: Ctx, x1: number, y1: number, x2: number, y2: number,
  W: number, H: number, color: string, strokeWidth = 1.0, opacity = 0.85, dashed = false,
) {
  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.strokeStyle = color;
  ctx.lineWidth = strokeWidth * (Math.min(W, H) / 320);
  ctx.lineCap = 'round';
  if (dashed) ctx.setLineDash([strokeWidth * 2.5, strokeWidth * 2]);
  ctx.beginPath(); ctx.moveTo(x1 * W, y1 * H); ctx.lineTo(x2 * W, y2 * H); ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
}

/**
 * drawEllipse Ã¢ÂÂ 3DÃ¦ÂÂÃ¨Â½Â¬Ã§ÂÂÃ¯Â¼ÂÃ¤Â¸ÂÃ¦Â¯ÂÃ¤Â¸ÂÃ¥Â¯Â¹Ã¦Â ÂÃ¦Â Â·Ã¦ÂÂ¿Ã¥ÂÂ¾Ã¯Â¼Â
 *
 * Ã¥ÂÂÃ¥Â¼Â§ = Ã¦ÂÂÃ¥ÂÂÃ©ÂÂÃ¥Â¤Â´Ã§ÂÂÃ¥ÂÂÃ¥Â¼Â§Ã¯Â¼ÂÃ§Â²Â + Ã¥Â¼ÂºÃ¥ÂÂÃ¥ÂÂÃ¯Â¼Â
 * Ã¥ÂÂÃ¥Â¼Â§ = Ã¨ÂÂÃ¥ÂÂÃ©ÂÂÃ¥Â¤Â´Ã§ÂÂÃ¥ÂÂÃ¥Â¼Â§Ã¯Â¼ÂÃ§Â»ÂÃ¨ÂÂÃ§ÂºÂ¿ + Ã¦Â·Â¡Ã¯Â¼ÂÃ¦Â¨Â¡Ã¦ÂÂÃ¨Â¢Â«Ã¨ÂºÂ«Ã¤Â½ÂÃ©ÂÂ®Ã¦ÂÂ¡Ã¯Â¼Â
 * Ã¥Â¡Â«Ã¥ÂÂ = Ã¦ÂÂ´Ã§ÂÂÃ¥ÂÂÃ©ÂÂÃ¦ÂÂ
 *
 * face_on Ã¨Â§ÂÃ¨Â§ÂÃ¯Â¼Â
 *   Ã¥ÂÂÃ¥Â¼Â§ = ellipse Ã¤Â¸ÂÃ¥ÂÂÃ¥ÂÂÃ¯Â¼Ây+Ã¦ÂÂ¹Ã¥ÂÂÃ¯Â¼ÂÃ©ÂÂ Ã¨Â¿ÂÃ©ÂÂÃ¥Â¤Â´Ã¯Â¼Â
 *   Ã¥ÂÂÃ¥Â¼Â§ = ellipse Ã¤Â¸ÂÃ¥ÂÂÃ¥ÂÂÃ¯Â¼Ây-Ã¦ÂÂ¹Ã¥ÂÂÃ¯Â¼ÂÃ§Â©Â¿Ã¨Â¿ÂÃ¨ÂºÂ«Ã¤Â½ÂÃ¥ÂÂÃ©ÂÂ¢Ã¯Â¼Â
 */
export function drawEllipse(
  ctx: Ctx,
  cx: number, cy: number,
  rx: number, ry: number,
  angleDeg: number,
  W: number, H: number,
  color: string,
  strokeWidth = 5.0,
  opacity = 0.92,
) {
  const pcx = cx * W, pcy = cy * H;
  const prx = rx * W;
  const pry = ry * W;
  const angle = angleDeg * Math.PI / 180;

  ctx.save();
  ctx.translate(pcx, pcy);
  ctx.rotate(angle);

  // ── 1. 整盘半透明填充（身体透过来显示）──
  ctx.globalAlpha = opacity * 0.20;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.ellipse(0, 0, prx, pry, 0, 0, Math.PI * 2);
  ctx.fill();

  // ── 2. 前弧（下半弧 0→π，朝向镜头）：粗实线 + 强发光 ──
  // 这是观众能看到的部分（圆盘朝向镜头的那一侧）
  // 外层 halo
  ctx.globalAlpha = opacity * 0.32;
  ctx.shadowColor = color;
  ctx.shadowBlur  = 26;
  ctx.strokeStyle = color;
  ctx.lineWidth   = strokeWidth * 3.0;
  ctx.beginPath();
  ctx.ellipse(0, 0, prx, pry, 0, 0, Math.PI);
  ctx.stroke();

  // 内层清晰主线
  ctx.globalAlpha = opacity;
  ctx.shadowColor = color;
  ctx.shadowBlur  = 8;
  ctx.strokeStyle = color;
  ctx.lineWidth   = strokeWidth;
  ctx.beginPath();
  ctx.ellipse(0, 0, prx, pry, 0, 0, Math.PI);
  ctx.stroke();

  // ── 3. 后弧（上半弧 π→2π，被身体遮挡）：完全不画 ──
  // 不渲染后弧，视觉上像圆盘穿过身体后消失在背后

  ctx.restore();
}
export function drawCurvePath(
  ctx: Ctx, points: Pt[], W: number, H: number,
  color: string, strokeWidth = 1.5, opacity = 0.80,
) {
  if (points.length < 2) return;
  const px = (pt: Pt): [number, number] => [pt.x * W, pt.y * H];
  const lw = strokeWidth * (Math.min(W, H) / 320);
  ctx.save();
  ctx.globalAlpha = opacity; ctx.strokeStyle = color;
  ctx.lineWidth = lw; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  ctx.beginPath();
  const [sx, sy] = px(points[0]); ctx.moveTo(sx, sy);
  for (let i = 1; i < points.length - 1; i++) {
    const [cx2, cy2] = px(points[i]), [nx, ny] = px(points[i + 1]);
    ctx.quadraticCurveTo(cx2, cy2, (cx2 + nx) / 2, (cy2 + ny) / 2);
  }
  const last = px(points[points.length - 1]); ctx.lineTo(last[0], last[1]);
  ctx.stroke(); ctx.restore();
}

export function drawArrow(
  ctx: Ctx, fromX: number, fromY: number, toX: number, toY: number,
  W: number, H: number, color: string, strokeWidth = 1.5, opacity = 0.90,
) {
  const fx=fromX*W,fy=fromY*H,tx=toX*W,ty=toY*H;
  const a=Math.atan2(ty-fy,tx-fx),dist=Math.hypot(tx-fx,ty-fy);
  const headLen=Math.min(14,dist*0.38),lw=strokeWidth*(Math.min(W,H)/320);
  ctx.save(); ctx.globalAlpha=opacity;
  ctx.strokeStyle=color; ctx.fillStyle=color; ctx.lineWidth=lw; ctx.lineCap='round';
  ctx.beginPath(); ctx.moveTo(fx,fy); ctx.lineTo(tx,ty); ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(tx,ty);
  ctx.lineTo(tx-headLen*Math.cos(a-0.40),ty-headLen*Math.sin(a-0.40));
  ctx.lineTo(tx-headLen*Math.cos(a+0.40),ty-headLen*Math.sin(a+0.40));
  ctx.closePath(); ctx.fill(); ctx.restore();
}

export function drawLabel(
  ctx: Ctx, x: number, y: number, W: number, H: number,
  text: string, color: string, fontSize = 10, opacity = 0.88,
) {
  const size = fontSize * Math.min(W, H) / 320;
  ctx.save(); ctx.globalAlpha=opacity;
  ctx.font=`700 ${size}px "DM Sans", system-ui, sans-serif`;
  ctx.textAlign='center'; ctx.shadowColor='rgba(0,0,0,0.90)'; ctx.shadowBlur=5;
  ctx.fillStyle=color; ctx.fillText(text, x*W, y*H); ctx.restore();
}

export function drawBadge(
  ctx: Ctx, x: number, y: number, W: number, H: number,
  variant: 'correct'|'wrong', opacity = 0.88,
) {
  const px2=x*W,py2=y*H,r=Math.min(W,H)*0.032;
  ctx.save(); ctx.globalAlpha=opacity;
  ctx.beginPath(); ctx.arc(px2,py2,r,0,Math.PI*2);
  ctx.fillStyle=variant==='correct'?COLORS.green:COLORS.red; ctx.fill();
  ctx.strokeStyle='#fff'; ctx.lineWidth=r*0.22; ctx.lineCap='round'; ctx.lineJoin='round';
  ctx.beginPath();
  if(variant==='correct'){ctx.moveTo(px2-r*0.5,py2);ctx.lineTo(px2-r*0.1,py2+r*0.4);ctx.lineTo(px2+r*0.55,py2-r*0.35);}
  else{ctx.moveTo(px2-r*0.4,py2-r*0.4);ctx.lineTo(px2+r*0.4,py2+r*0.4);ctx.moveTo(px2+r*0.4,py2-r*0.4);ctx.lineTo(px2-r*0.4,py2+r*0.4);}
  ctx.stroke(); ctx.restore();
}

export function drawZone(
  ctx: Ctx, points: Pt[], W: number, H: number,
  color: string, fillOpacity = 0.18, strokeOpacity = 0.88, strokeWidth = 4.5,
) {
  if (points.length < 3) return;
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(points[0].x*W,points[0].y*H);
  for(let i=1;i<points.length;i++) ctx.lineTo(points[i].x*W,points[i].y*H);
  ctx.closePath();
  ctx.fillStyle=color; ctx.globalAlpha=fillOpacity; ctx.fill();
  ctx.shadowColor=color; ctx.shadowBlur=10;
  ctx.globalAlpha=strokeOpacity; ctx.strokeStyle=color;
  ctx.lineWidth=strokeWidth; ctx.lineJoin='round'; ctx.stroke();
  ctx.restore();
}

export function renderElement(
  ctx: Ctx, el: OverlayElement, W: number, H: number, layer = 'all',
) {
  if (el.layer && el.layer !== 'all' && layer !== 'all' && el.layer !== layer) return;
  const color = resolveColor(el.color);
  const opacity = el.opacity ?? 0.88;

  const elAny = el as unknown as Record<string, unknown>;
  if (elAny['type'] === 'ellipse') {
    drawEllipse(
      ctx,
      elAny['cx'] as number, elAny['cy'] as number,
      elAny['rx'] as number, elAny['ry'] as number,
      elAny['angleDeg'] as number,
      W, H, color,
      (elAny['strokeWidth'] as number) ?? 5.0,
      opacity,
    );
    return;
  }

  switch (el.type) {
    case 'line': {
      const e = el as LineElement;
      drawStructureLine(ctx,e.x1,e.y1,e.x2,e.y2,W,H,color,e.strokeWidth??1.0,opacity,e.dashed);
      break;
    }
    case 'curve': {
      const e = el as CurveElement;
      drawCurvePath(ctx,e.points,W,H,color,e.strokeWidth??1.5,opacity);
      break;
    }
    case 'arrow': {
      const e = el as ArrowElement;
      drawArrow(ctx,e.from.x,e.from.y,e.to.x,e.to.y,W,H,color,e.strokeWidth??1.5,opacity);
      break;
    }
    case 'dot': {
      const e = el as DotElement;
      drawJointDot(ctx,e.x,e.y,W,H,color,e.radius??0.008,opacity);
      break;
    }
    case 'label': {
      const e = el as LabelElement;
      drawLabel(ctx,e.x,e.y,W,H,e.text,color,e.size??10,opacity);
      break;
    }
    case 'badge': {
      const e = el as BadgeElement;
      drawBadge(ctx,e.x,e.y,W,H,e.variant,opacity);
      break;
    }
    case 'zone': {
      const e = el as ZoneElement;
      const zo = el as unknown as { strokeWidth?: number };
      drawZone(ctx,e.points,W,H,color,e.fillOpacity??0.18,opacity,zo.strokeWidth??4.5);
      break;
    }
  }
}

export function renderFrame(
  ctx: Ctx, elements: OverlayElement[], W: number, H: number, layer = 'all',
) {
  ctx.clearRect(0, 0, W, H);
  for (const el of elements) renderElement(ctx, el, W, H, layer);
}
