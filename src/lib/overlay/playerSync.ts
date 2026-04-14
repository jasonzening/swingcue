/**
 * playerSync.ts
 *
 * 根据当前视频时间（秒）从 OverlayTimeline 中取对应的 overlay 元素。
 *
 * Phase 1：最近时间片匹配（离散帧）
 * Phase 2：将在这里加入线性插值，不影响调用方
 */
import type { OverlayTimeline, OverlayFrame, OverlayElement, PhaseMarkers } from '@/types/analysis';

/**
 * 从 timeline 中找到当前时间最近的帧，返回它的 elements。
 * 如果相邻帧之间有过渡区（frac > 0.4），将两帧 elements 合并渲染。
 */
export function getOverlayAtTime(
  timeline: OverlayTimeline,
  currentTime: number
): OverlayElement[] {
  const frames = timeline.frames;
  if (!frames.length) return [];

  // 边界情况
  if (currentTime <= frames[0].time) return frames[0].elements;
  if (currentTime >= frames[frames.length - 1].time) return frames[frames.length - 1].elements;

  // 找到包含当前时间的区间
  for (let i = 0; i < frames.length - 1; i++) {
    const a = frames[i];
    const b = frames[i + 1];
    if (currentTime >= a.time && currentTime <= b.time) {
      const span = b.time - a.time;
      const frac = span > 0 ? (currentTime - a.time) / span : 0;

      // 过渡前半段只显示 a 帧
      if (frac < 0.45) return a.elements;
      // 过渡后半段混合显示（透明度渐变由 renderer 处理）
      if (frac < 0.55) return [...a.elements, ...b.elements.map(el => ({ ...el, opacity: (el.opacity ?? 0.9) * ((frac - 0.45) / 0.1) }))];
      // 后半段只显示 b 帧
      return b.elements;
    }
  }

  return frames[frames.length - 1].elements;
}

/**
 * 根据当前时间判断挥杆阶段
 */
export function getCurrentPhase(
  phases: PhaseMarkers,
  currentTime: number,
  duration: number
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

/**
 * 找到阶段对应的时间（秒）
 */
export function getPhaseTime(phases: PhaseMarkers, phase: keyof PhaseMarkers): number {
  return phases[phase];
}

/**
 * 格式化时间显示
 */
export function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}
