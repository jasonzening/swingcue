/**
 * templates/index.ts — OverlayTimeline 生成器
 *
 * PATH A: 有真实 keypoints → 严格按 spec 渲染真实身体点
 * PATH B: 无真实数据 → 预设示意 overlay（明确标注为 stub）
 */

import type {
  OverlayTimeline, OverlayFrame, OverlayElement,
  PhaseMarkers, VideoMetadata, MainIssueType, TemplateInput, KeypointFrame,
} from '@/types/analysis';
import { ISSUE_DEFINITIONS } from '@/lib/golf/issues';
import { generateSpecDrivenOverlayFrame, findNearestFrame, getTrackedPoint } from '@/lib/overlay/keypointOverlay';
import type { ViewType } from '@/lib/golf/overlayLineSpec';

type Phase = 'setup' | 'top' | 'transition' | 'impact' | 'finish';
type C = 'red' | 'green' | 'yellow' | 'white';

/* ── 小型 element 构建器（仅用于 PATH B fallback）── */
const dot = (id: string, x: number, y: number, color: C, r = 0.026, opacity = 0.95): OverlayElement =>
  ({ type: 'dot', id, x, y, color, radius: r, opacity, layer: 'body' });
const line = (id: string, x1: number, y1: number, x2: number, y2: number, color: C, w = 3, dashed = false, opacity = 0.85): OverlayElement =>
  ({ type: 'line', id, x1, y1, x2, y2, color, strokeWidth: w, dashed, opacity, layer: 'body' });
const arrow = (id: string, fx: number, fy: number, tx: number, ty: number, color: C = 'yellow', opacity = 0.90): OverlayElement =>
  ({ type: 'arrow', id, from: { x: fx, y: fy }, to: { x: tx, y: ty }, color, strokeWidth: 3, opacity });
const label = (id: string, x: number, y: number, text: string, color: C = 'white', size = 10): OverlayElement =>
  ({ type: 'label', id, x, y, text, color, size, opacity: 0.80 });
const badge = (id: string, x: number, y: number, v: 'correct' | 'wrong'): OverlayElement =>
  ({ type: 'badge', id, x, y, variant: v, opacity: 0.88 });

/* ── PATH A：真实 keypoints 驱动 ── */
function buildRealFrame(
  time: number, phase: Phase,
  issue: MainIssueType, viewType: ViewType,
  allKpFrames: KeypointFrame[],
  pathHistory: Array<{ x: number; y: number }>,
): OverlayFrame {
  const nearestKp = findNearestFrame(allKpFrames, time);
  if (!nearestKp) {
    return { time, phase, elements: [label('ph', 0.50, 0.06, phase.toUpperCase(), 'white', 10)] };
  }

  // 收集路径历史点（用于手路径/头路径显示）
  const trackedPt = getTrackedPoint(issue, viewType, nearestKp);
  if (trackedPt) pathHistory.push(trackedPt);
  // 只保留最近 1.5s 的路径（约 6 个采样点）
  if (pathHistory.length > 8) pathHistory.splice(0, pathHistory.length - 8);

  const elements = generateSpecDrivenOverlayFrame(
    issue, viewType, phase, nearestKp,
    pathHistory.length >= 2 ? [...pathHistory] : undefined,
  );

  // 加 cue 标签（只在 impact 阶段）
  if (phase === 'impact') {
    const issueData = ISSUE_DEFINITIONS[issue];
    if (issueData) {
      elements.push(label('cue', 0.50, 0.96, `"${issueData.cue}"`, 'yellow', 10));
    }
  }

  return { time, phase, elements };
}

/* ── PATH B fallback 示意图 ── */
function buildFallbackFrame(
  time: number, phase: Phase, issue: MainIssueType,
): OverlayFrame {
  const elements: OverlayElement[] = [];

  // 通用基础骨架
  elements.push(
    line('sp', 0.50, 0.19, 0.49, 0.54, 'white', 2, false, 0.55),
    line('sh', 0.37, 0.27, 0.63, 0.27, 'green', 2.5, false, 0.72),
    dot('ls', 0.37, 0.27, 'green', 0.022),
    dot('rs', 0.63, 0.27, 'green', 0.022),
  );

  // 按问题在关键阶段加方向指示
  if (phase === 'impact' || phase === 'transition') {
    switch (issue) {
      case 'shoulder_lifts_too_early':
        elements.push(
          dot('rs-r', 0.63, 0.22, 'red', 0.028),
          dot('rs-g', 0.63, 0.30, 'green', 0.024),
          arrow('arr', 0.63, 0.22, 0.63, 0.30),
          label('msg', 0.72, 0.18, 'Lower ↓', 'green', 10),
        );
        break;
      case 'early_extension':
      case 'posture_rises_too_early':
        elements.push(
          line('sp-r', 0.50, 0.13, 0.52, 0.50, 'red', 3, false, 0.85),
          line('sp-g', 0.64, 0.16, 0.62, 0.53, 'green', 3, true, 0.80),
          arrow('arr', 0.50, 0.13, 0.50, 0.18, 'yellow'),
          badge('wr', 0.34, 0.25, 'wrong'),
          badge('ok', 0.75, 0.22, 'correct'),
        );
        break;
      case 'head_movement':
        elements.push(
          dot('h-r', 0.44, 0.11, 'red', 0.028),
          dot('h-g', 0.50, 0.11, 'green', 0.024),
          arrow('arr', 0.44, 0.11, 0.50, 0.11),
          label('msg', 0.50, 0.07, 'Stay centered', 'green', 10),
        );
        break;
      case 'hand_path_issue':
        elements.push(
          { type: 'curve', id: 'pr', color: 'red', points: [{x:0.35,y:0.18},{x:0.54,y:0.24},{x:0.56,y:0.50}], strokeWidth: 3.5, opacity: 0.85, layer: 'arms' },
          { type: 'curve', id: 'pg', color: 'green', points: [{x:0.35,y:0.18},{x:0.38,y:0.30},{x:0.42,y:0.52}], strokeWidth: 3.5, opacity: 0.82, layer: 'arms' },
          arrow('arr', 0.54, 0.26, 0.40, 0.32),
          label('msg', 0.24, 0.30, 'Drop inside ✓', 'green', 10),
        );
        break;
      case 'not_enough_hip_turn':
        elements.push(
          line('hi-r', 0.38, 0.54, 0.60, 0.54, 'red', 3, false, 0.85),
          line('hi-g', 0.34, 0.52, 0.58, 0.58, 'green', 3, true, 0.80),
          arrow('arr', 0.60, 0.54, 0.64, 0.52),
        );
        break;
      default:
        elements.push(label('ph', 0.50, 0.93, 'Analyzing...', 'yellow', 10));
    }
  }

  elements.push(
    label('ph', 0.50, 0.06, `${phase.toUpperCase()} [GUIDE]`, 'white', 10),
  );

  return { time, phase, elements };
}

/* ══════════════════════════════════════
   主导出函数
══════════════════════════════════════ */
export function generateOverlayTimeline(input: TemplateInput): OverlayTimeline {
  const {
    phaseMarkers: phases, issue, keypointTimeline, viewType: inputViewType,
  } = input;

  const kpFrames: KeypointFrame[] = keypointTimeline?.frames ?? [];
  const hasRealKeypoints = kpFrames.length >= 3;

  const viewType: ViewType = (inputViewType as ViewType) ?? 'face_on';

  const phasePairs: [number, Phase][] = [
    [phases.setupTime,      'setup'],
    [phases.topTime,        'top'],
    [phases.transitionTime, 'transition'],
    [phases.impactTime,     'impact'],
    [phases.finishTime,     'finish'],
  ];

  const frames: OverlayFrame[] = [];
  const pathHistory: Array<{ x: number; y: number }> = [];

  for (const [time, phase] of phasePairs) {
    if (hasRealKeypoints) {
      frames.push(buildRealFrame(time, phase, issue, viewType, kpFrames, pathHistory));
    } else {
      frames.push(buildFallbackFrame(time, phase, issue));
    }
  }

  return { frames };
}


/* ══════════════════════════════════════
Dense overlay：每个 keypoint 帧生成一帧 overlay
用于前端连续追踪（解决5帧快照脱节问题）
══════════════════════════════════════ */

/**
 * 计算给定时间属于哪个挥杆阶段
 */
function computePhase(phases: PhaseMarkers, time: number, duration: number): Phase {
  if (duration <= 0) return 'setup';
  const norm = time / duration;
  const p = {
    setup: phases.setupTime / duration,
    top: phases.topTime / duration,
    transition: phases.transitionTime / duration,
    impact: phases.impactTime / duration,
    finish: phases.finishTime / duration,
  };
  if (norm >= p.finish) return 'finish';
  if (norm >= p.impact) return 'impact';
  if (norm >= p.transition) return 'transition';
  if (norm >= p.top) return 'top';
  return 'setup';
}

/**
 * generateDenseOverlayTimeline
 *
 * 每个 keypoint 帧都生成一帧 overlay。
 * 结果约 25-30 帧（4fps 采样），SwingPlayer 可以流畅追踪身体。
 * 仅在有真实 keypoint 数据时使用，否则退化到 generateOverlayTimeline。
 */
export function generateDenseOverlayTimeline(input: {
  keypointFrames: KeypointFrame[];
  phaseMarkers: PhaseMarkers;
  issue: MainIssueType;
  viewType: string;
  duration: number;
}): OverlayTimeline {
  const { keypointFrames, phaseMarkers, issue, viewType, duration } = input;

  if (!keypointFrames || keypointFrames.length < 3) {
    return { frames: [] };
  }

  const vt: ViewType = (viewType as ViewType) ?? 'face_on';
  const pathHistory: Array<{ x: number; y: number }> = [];
  const frames: OverlayFrame[] = [];

  for (const kpFrame of keypointFrames) {
    const phase = computePhase(phaseMarkers, kpFrame.time, duration);

    // 收集路径历史（手路径等）
    const trackedPt = getTrackedPoint(issue, vt, kpFrame);
    if (trackedPt) pathHistory.push(trackedPt);
    if (pathHistory.length > 8) pathHistory.splice(0, pathHistory.length - 8);

    const elements = generateSpecDrivenOverlayFrame(
      issue,
      vt,
      phase,
      kpFrame,
      pathHistory.length >= 2 ? [...pathHistory] : undefined,
    );

    frames.push({ time: kpFrame.time, phase, elements });
  }

  return { frames };
}
