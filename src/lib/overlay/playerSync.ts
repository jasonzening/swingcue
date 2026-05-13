/**
 * playerSync.ts
 *
 * 根据当前视频时间从 OverlayTimeline 中取 overlay 元素。
 * 使用最近帧策略 — 无类型转换问题。
 */
import type { OverlayTimeline, OverlayElement, PhaseMarkers } from '@/types/analysis';

/**
 * getOverlayAtTime — 找最近帧
 * dense timeline (29帧) + 插值已在数据生成层完成，这里只做帧查找
 */
export function getOverlayAtTime(
  timeline: OverlayTimeline,
  currentTime: number,
): OverlayElement[] {
  const frames = timeline.frames;
  if (!frames.length) return [];

  // 二分查找最近帧
  if (currentTime <= frames[0].time) return frames[0].elements;
  if (currentTime >= frames[frames.length - 1].time) return frames[frames.length - 1].elements;

  let best = frames[0];
  let bestDist = Math.abs(currentTime - best.time);

  for (const frame of frames) {
    const dist = Math.abs(currentTime - frame.time);
    if (dist < bestDist) {
      bestDist = dist;
      best = frame;
    }
    // 一旦距离开始增大，提前退出（frames 有序）
    if (frame.time > currentTime && dist > bestDist) break;
  }

  return best.elements;
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
