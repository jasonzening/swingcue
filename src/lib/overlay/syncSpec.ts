/**
 * syncSpec.ts — overlay 同步与稳定性规范
 *
 * 核心原则：video.currentTime 是唯一真相。
 * overlay 永远跟视频时间走，不自己跑独立时钟。
 */

import type { BodyPointName } from '@/lib/golf/bodyPointSpec';
import type { OverlayTimeline, OverlayFrame, OverlayElement } from '@/types/analysis';

/* ══════════════════════════════════════
   播放器状态
══════════════════════════════════════ */
export type PlayerMode = 'playing' | 'slow' | 'scrubbing' | 'frameStep' | 'paused';

export interface PlayerOverlayState {
  isPlaying: boolean;
  isScrubbing: boolean;
  playbackRate: number;
  currentTime: number;
  currentPhase?: 'setup' | 'top' | 'transition' | 'impact' | 'finish';
}

/* ══════════════════════════════════════
   平滑配置
  high:   EMA alpha=0.65（强平滑，响应稍慢但极稳定）
  medium: EMA alpha=0.50
  low:    EMA alpha=0.35（弱平滑，响应快）
  none:   直接透传（scrub时使用）
══════════════════════════════════════ */
export const SMOOTHING_CONFIG: Record<BodyPointName, 'high' | 'medium' | 'low'> = {
  headCenter:      'high',
  headTop:         'high',
  neckCenter:      'high',
  leftShoulder:    'high',
  rightShoulder:   'high',
  leftElbow:       'medium',
  rightElbow:      'medium',
  leftWrist:       'high',
  rightWrist:      'high',
  leftHip:         'high',
  rightHip:        'high',
  leftKnee:        'medium',
  rightKnee:       'medium',
  leftAnkle:       'low',
  rightAnkle:      'low',
  gripCenter:      'high',
  hipCenter:       'high',
  shoulderCenter:  'high',
};

export const SMOOTHING_ALPHA: Record<'high' | 'medium' | 'low', number> = {
  high:   0.65,
  medium: 0.50,
  low:    0.35,
};

/* ══════════════════════════════════════
   置信度阈值
══════════════════════════════════════ */
export const CONFIDENCE_THRESHOLDS = {
  /** 低于此值：尝试用前帧插值 */
  useInterpolation: 0.30,
  /** 低于此值：隐藏点，不画 */
  hidePoint: 0.20,
  /** 短暂缺失帧数（用前帧填充） */
  maxFillFrames: 3,
};

/* ══════════════════════════════════════
   核心：按当前时间取 overlay 帧
  - nearest-frame 匹配（MVP阶段）
  - 不做插值（先准确，再平滑）
══════════════════════════════════════ */
export function getOverlayAtTime(
  timeline: OverlayTimeline,
  currentTime: number,
  mode: PlayerMode = 'playing',
): OverlayElement[] {
  const frames = timeline.frames;
  if (!frames.length) return [];

  // 边界情况
  if (currentTime <= frames[0].time) return frames[0].elements;
  if (currentTime >= frames[frames.length - 1].time) return frames[frames.length - 1].elements;

  // scrubbing / frameStep 模式：严格 nearest-frame，不过渡
  if (mode === 'scrubbing' || mode === 'frameStep') {
    return nearestFrame(frames, currentTime).elements;
  }

  // 播放/慢放模式：nearest-frame（MVP），后续可升级插值
  return nearestFrame(frames, currentTime).elements;
}

function nearestFrame(frames: OverlayFrame[], t: number): OverlayFrame {
  let best = frames[0];
  let bestDist = Math.abs(t - best.time);
  for (const f of frames) {
    const d = Math.abs(t - f.time);
    if (d < bestDist) { best = f; bestDist = d; }
  }
  return best;
}

/* ══════════════════════════════════════
   逐步前进/后退的时间单位
  - 有真实fps：按 1/fps
  - 无fps / 离散帧：跳到下一个 overlay frame
══════════════════════════════════════ */
export function getFrameStepTime(
  currentTime: number,
  direction: 1 | -1,
  fps: number,
  timeline: OverlayTimeline,
): number {
  if (fps > 0) {
    // 按真实帧率步进
    return Math.max(0, currentTime + direction / fps);
  }

  // 按 overlay frame 步进
  const frames = timeline.frames;
  if (!frames.length) return currentTime;

  if (direction === 1) {
    const next = frames.find(f => f.time > currentTime + 0.001);
    return next?.time ?? frames[frames.length - 1].time;
  } else {
    const prev = [...frames].reverse().find(f => f.time < currentTime - 0.001);
    return prev?.time ?? frames[0].time;
  }
}

/* ══════════════════════════════════════
   EMA 平滑器（指数移动平均）
  用于对 keypoint 坐标做时序平滑
══════════════════════════════════════ */
export class EmaSmoothor {
  private prev: { x: number; y: number } | null = null;
  private readonly alpha: number;

  constructor(alpha: number) {
    this.alpha = alpha;
  }

  update(pt: { x: number; y: number } | null, isScrubbing: boolean): { x: number; y: number } | null {
    if (!pt) {
      // 点丢失：保持上一帧（最多 CONFIDENCE_THRESHOLDS.maxFillFrames 帧）
      return this.prev;
    }

    if (isScrubbing || !this.prev) {
      // scrub 时直接跳，不做平滑（准确优先）
      this.prev = pt;
      return pt;
    }

    // EMA 平滑
    this.prev = {
      x: this.alpha * pt.x + (1 - this.alpha) * this.prev.x,
      y: this.alpha * pt.y + (1 - this.alpha) * this.prev.y,
    };
    return this.prev;
  }

  reset() { this.prev = null; }
}

/* ══════════════════════════════════════
   低置信度/丢点处理
══════════════════════════════════════ */
export function resolvePointWithFallback(
  current: { x: number; y: number; confidence?: number } | null,
  prev: { x: number; y: number } | null,
  smoothhor: EmaSmoothor,
  isScrubbing: boolean,
): { x: number; y: number; visible: boolean } | null {
  const conf = current?.confidence ?? 0;

  if (!current || conf < CONFIDENCE_THRESHOLDS.hidePoint) {
    // 低置信度：尝试用上一帧填充
    if (prev) return { ...prev, visible: false };
    return null;
  }

  const smoothed = smoothhor.update(current, isScrubbing);
  if (!smoothed) return null;
  return { ...smoothed, visible: true };
}

/* ══════════════════════════════════════
   线条可见性规则：
  两个端点都必须可信才画线
══════════════════════════════════════ */
export function canDrawLine(
  pt1: { visible: boolean } | null,
  pt2: { visible: boolean } | null,
): boolean {
  return !!(pt1?.visible && pt2?.visible);
}

/* ══════════════════════════════════════
   路径管理：只保留当前阶段窗口
  不允许全视频无限累积历史路径
══════════════════════════════════════ */
export class PathTracker {
  private history: Array<{ x: number; y: number; time: number }> = [];
  private readonly maxDuration: number; // 秒

  constructor(maxDuration = 1.5) {
    this.maxDuration = maxDuration;
  }

  push(pt: { x: number; y: number }, time: number) {
    this.history.push({ ...pt, time });
    // 截断超出窗口的历史
    const cutoff = time - this.maxDuration;
    this.history = this.history.filter(p => p.time >= cutoff);
  }

  getPath(): Array<{ x: number; y: number }> {
    return this.history.map(({ x, y }) => ({ x, y }));
  }

  reset() { this.history = []; }
}

/* ══════════════════════════════════════
   当前时间 → 阶段判断
══════════════════════════════════════ */
export function getCurrentPhase(
  phases: { setupTime: number; topTime: number; transitionTime: number; impactTime: number; finishTime: number },
  currentTime: number,
): 'setup' | 'top' | 'transition' | 'impact' | 'finish' {
  if (currentTime >= phases.finishTime)     return 'finish';
  if (currentTime >= phases.impactTime)     return 'impact';
  if (currentTime >= phases.transitionTime) return 'transition';
  if (currentTime >= phases.topTime)        return 'top';
  return 'setup';
}

export function formatTime(s: number): string {
  const m = Math.floor(s / 60);
  const ss = Math.floor(s % 60);
  return `${m}:${ss.toString().padStart(2, '0')}`;
}
