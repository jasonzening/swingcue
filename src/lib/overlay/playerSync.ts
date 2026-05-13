/**
 * playerSync.ts
 *
 * 根据当前视频时间从 OverlayTimeline 中取 overlay 元素。
 * 支持帧间线性插值，实现 60fps 平滑追踪。
 */
import type { OverlayTimeline, OverlayElement, PhaseMarkers } from '@/types/analysis';
import type { DotElement, LineElement, ArrowElement } from '@/types/analysis';

/* ── 线性插值 ── */
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

function interpolateElement(a: OverlayElement, b: OverlayElement, t: number): OverlayElement {
  if (a.type === 'dot' && b.type === 'dot') {
    const ae = a as DotElement;
    const be = b as DotElement;
    return { ...ae, x: lerp(ae.x, be.x, t), y: lerp(ae.y, be.y, t) } as OverlayElement;
  }
  if (a.type === 'line' && b.type === 'line') {
    const ae = a as LineElement;
    const be = b as LineElement;
    return { ...ae, x1: lerp(ae.x1, be.x1, t), y1: lerp(ae.y1, be.y1, t), x2: lerp(ae.x2, be.x2, t), y2: lerp(ae.y2, be.y2, t) } as OverlayElement;
  }
  if (a.type === 'arrow' && b.type === 'arrow') {
    const ae = a as ArrowElement;
    const be = b as ArrowElement;
    return { ...ae, from: { x: lerp(ae.from.x, be.from.x, t), y: lerp(ae.from.y, be.from.y, t) }, to: { x: lerp(ae.to.x, be.to.x, t), y: lerp(ae.to.y, be.to.y, t) } } as OverlayElement;
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

/**
 * getOverlayAtTime — 帧间线性插值
 * 4fps keypoint 数据 + 60fps render loop = 平滑连续追踪
 */
export function getOverlayAtTime(
  timeline: OverlayTimeline,
  currentTime: number,
): OverlayElement[] {
  const frames = timeline.frames;
  if (!frames.length) return [];

  if (currentTime <= frames[0].time) return frames[0].elements;
  if (currentTime >= frames[frames.length - 1].time) return frames[frames.length - 1].elements;

  for (let i = 0; i < frames.length - 1; i++) {
    const a = frames[i];
    const b = frames[i + 1];
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
