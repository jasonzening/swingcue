/**
 * playerSync.ts
 *
 * 根据当前视频时间从 OverlayTimeline 中取 overlay 元素。
 * 支持帧间线性插值，实现 60fps 平滑追踪。
 */
import type { OverlayTimeline, OverlayElement, PhaseMarkers } from '@/types/analysis';

/* ── 线性插值 ── */
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

function interpolateElement(a: OverlayElement, b: OverlayElement, t: number): OverlayElement {
  if (a.type === 'dot' && b.type === 'dot' && 'x' in a && 'x' in b) {
    const ax = (a as { x: number }).x, ay = (a as { y: number }).y;
    const bx = (b as { x: number }).x, by = (b as { y: number }).y;
    return { ...a, x: lerp(ax, bx, t), y: lerp(ay, by, t) } as OverlayElement;
  }
  if (a.type === 'line' && b.type === 'line' && 'x1' in a && 'x1' in b) {
    const al = a as { x1:number;y1:number;x2:number;y2:number } & OverlayElement;
    const bl = b as { x1:number;y1:number;x2:number;y2:number } & OverlayElement;
    return { type:'line', id:al.id, color:al.color, strokeWidth:al.strokeWidth, opacity:al.opacity, layer:al.layer, dashed:al.dashed,
      x1:lerp(al.x1,bl.x1,t), y1:lerp(al.y1,bl.y1,t),
      x2:lerp(al.x2,bl.x2,t), y2:lerp(al.y2,bl.y2,t) } as OverlayElement;
  }
  if (a.type === 'arrow' && b.type === 'arrow' && 'from' in a && 'from' in b) {
    const aa = a as { from:{x:number;y:number}; to:{x:number;y:number} } & OverlayElement;
    const ba = b as { from:{x:number;y:number}; to:{x:number;y:number} } & OverlayElement;
    return { ...aa,
      from: { x:lerp(aa.from.x,ba.from.x,t), y:lerp(aa.from.y,ba.from.y,t) },
      to:   { x:lerp(aa.to.x,  ba.to.x,  t), y:lerp(aa.to.y,  ba.to.y,  t) } } as OverlayElement;
  }
  return t < 0.5 ? a : b;
}

function interpolateFrames(aEls: OverlayElement[], bEls: OverlayElement[], t: number): OverlayElement[] {
  if (!aEls.length || !bEls.length) return t < 0.5 ? aEls : bEls;
  const len = Math.min(aEls.length, bEls.length);
  const result: OverlayElement[] = [];
  for (let i = 0; i < len; i++) {
    const a = aEls[i], b = bEls[i];
    result.push(a.type === b.type ? interpolateElement(a, b, t) : (t < 0.5 ? a : b));
  }
  if (aEls.length > len) result.push(...aEls.slice(len));
  if (bEls.length > len) result.push(...bEls.slice(len));
  return result;
}

export function getOverlayAtTime(
  timeline: OverlayTimeline,
  currentTime: number,
): OverlayElement[] {
  const frames = timeline.frames;
  if (!frames.length) return [];
  if (currentTime <= frames[0].time) return frames[0].elements;
  if (currentTime >= frames[frames.length - 1].time) return frames[frames.length - 1].elements;
  for (let i = 0; i < frames.length - 1; i++) {
    const a = frames[i], b = frames[i + 1];
    if (currentTime >= a.time && currentTime <= b.time) {
      const span = b.time - a.time;
      const frac = span > 0 ? (currentTime - a.time) / span : 0;
      return interpolateFrames(a.elements, b.elements, frac);
    }
  }
  return frames[frames.length - 1].elements;
}

export function getCurrentPhase(
  phases: PhaseMarkers,
  currentTime: number,
  duration: number,
): 'setup' | 'top' | 'transition' | 'impact' | 'finish' {
  const normT = duration > 0 ? currentTime / duration : 0;
  const p = {
    setup: phases.setupTime / duration,
    top: phases.topTime / duration,
    transition: phases.transitionTime / duration,
    impact: phases.impactTime / duration,
    finish: phases.finishTime / duration,
  };
  if (normT >= p.finish) return 'finish';
  if (normT >= p.impact) return 'impact';
  if (normT >= p.transition) return 'transition';
  if (normT >= p.top) return 'top';
  return 'setup';
}

export function getPhaseTime(phases: PhaseMarkers, phase: keyof PhaseMarkers): number {
  return phases[phase];
}

export function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}
