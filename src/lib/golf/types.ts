/**
 * golf/types.ts — Golf Intelligence Layer 核心类型
 *
 * 这一层把底层运动数据（关键点、角度、速度）
 * 转化成新手看得懂的高尔夫纠错逻辑。
 *
 * 层次结构：
 *   Foundation Model (MediaPipe) → Motion Understanding → Golf Intelligence → Visual Coaching
 */

import type { MainIssueType } from '@/types/analysis';

/* ══════════════════════════════════════
   目标修正：核心抽象
   "这里高一点 / 低一点 / 里一点 / 外一点"
══════════════════════════════════════ */

export type BodyPartKey =
  | 'head'
  | 'shoulders'
  | 'trailShoulder'
  | 'leadShoulder'
  | 'hands'
  | 'trailHand'
  | 'leadHand'
  | 'hips'
  | 'spine'
  | 'weight'
  | 'club';

export type CorrectionDirection =
  | 'higher'         // 更高一点
  | 'lower'          // 更低一点
  | 'more_inside'    // 更靠里一点
  | 'more_outside'   // 更靠外一点
  | 'more_forward'   // 更向前一点
  | 'more_back'      // 更向后一点
  | 'more_turned'    // 更转一点
  | 'less_turned'    // 少转一点
  | 'steeper'        // 更陡
  | 'shallower'      // 更平（浅）
  | 'more_centered'  // 更居中
  | 'more_stable';   // 更稳定

export type SwingPhase = 'setup' | 'top' | 'transition' | 'impact' | 'finish';

export interface TargetCorrection {
  id: string;
  phase: SwingPhase;
  bodyPart: BodyPartKey;
  direction: CorrectionDirection;
  magnitude: 'small' | 'medium' | 'large';
  message: string;   // 简洁说明（英文，教练语言）
}

/* ══════════════════════════════════════
   Overlay 指令（规则层 → 可视化层）
══════════════════════════════════════ */

export type FocusArea =
  | 'posture'
  | 'hand_path'
  | 'club_path'
  | 'shoulder_line'
  | 'hip_turn'
  | 'weight_shift'
  | 'head_position';

export interface OverlayInstruction {
  phase: SwingPhase;
  focusArea: FocusArea;
  redReference: string;    // 红色表示什么（当前错误）
  greenTarget: string;     // 绿色表示什么（目标）
  emphasis: 'line' | 'curve' | 'arrow' | 'dot' | 'zone' | 'badge';
}

/* ══════════════════════════════════════
   问题分析结果
══════════════════════════════════════ */

export interface GolfIssueResult {
  issue: MainIssueType;
  severity: 'low' | 'medium' | 'high';
  confidence: number;        // 0–1
  summary: string;           // 一句话给用户看的问题说明
  cue: string;               // 一句话 cue（教练短提示）
  drill: string;             // 一句话 drill
  targetCorrections: TargetCorrection[];
  overlayInstructions: OverlayInstruction[];
}

/* ══════════════════════════════════════
   问题排名（优先级系统）
══════════════════════════════════════ */

export interface RankedIssue {
  issue: MainIssueType;
  severity: number;      // 0–1
  confidence: number;    // 0–1
  priority: number;      // 0–1 (preset per issue type)
  finalScore: number;    // combined score
}
