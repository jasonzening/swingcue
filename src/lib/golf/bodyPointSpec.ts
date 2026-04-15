/**
 * bodyPointSpec.ts
 *
 * 严格按规范定义SwingCue必须追踪的身体关键点。
 * 不允许自由选点。所有overlay必须从这里取点定义。
 *
 * MediaPipe Pose 33点模型，我们使用以下点：
 */

/* ══════════════════════════════════════
   原始 MediaPipe 关键点索引
══════════════════════════════════════ */
export const MP_IDX = {
  NOSE:            0,
  LEFT_EYE_INNER:  1,
  LEFT_EYE:        2,
  LEFT_EAR:        7,
  RIGHT_EAR:       8,
  LEFT_SHOULDER:   11,
  RIGHT_SHOULDER:  12,
  LEFT_ELBOW:      13,
  RIGHT_ELBOW:     14,
  LEFT_WRIST:      15,
  RIGHT_WRIST:     16,
  LEFT_HIP:        23,
  RIGHT_HIP:       24,
  LEFT_KNEE:       25,
  RIGHT_KNEE:      26,
  LEFT_ANKLE:      27,
  RIGHT_ANKLE:     28,
} as const;

/* ══════════════════════════════════════
   SwingCue 点位语义命名
   对应规范文档 A-F 章节
══════════════════════════════════════ */

export type BodyPointName =
  // A. 头部与身体中轴
  | 'headCenter'       // 头部中心 (≈ nose 或眼耳中点)
  | 'headTop'          // 头顶近似 (headCenter上方一个头高)
  | 'neckCenter'       // 颈部/上胸中心 (双肩中点)
  // B. 肩部
  | 'leftShoulder'     // 左肩
  | 'rightShoulder'    // 右肩
  // C. 手臂
  | 'leftElbow'        // 左肘
  | 'rightElbow'       // 右肘
  | 'leftWrist'        // 左手腕
  | 'rightWrist'       // 右手腕
  // D. 髋部
  | 'leftHip'          // 左髋
  | 'rightHip'         // 右髋
  // E. 下肢
  | 'leftKnee'         // 左膝
  | 'rightKnee'        // 右膝
  | 'leftAnkle'        // 左踝
  | 'rightAnkle'       // 右踝
  // F. 派生点（计算得出，非模型直接输出）
  | 'gripCenter'       // 双手握把中心 = (左腕+右腕)/2
  | 'hipCenter'        // 双髋中点 = (左髋+右髋)/2
  | 'shoulderCenter'   // 双肩中点 = (左肩+右肩)/2

export interface BodyPointDef {
  name: BodyPointName;
  mpIdx?: number | number[];  // MediaPipe索引，派生点无此字段
  isDerived: boolean;          // 是否为派生点（计算得出）
  derivedFrom?: BodyPointName[];  // 派生点的来源点
  smoothingPriority: 'high' | 'medium' | 'low';  // 平滑优先级
  golfPurpose: string[];       // 高尔夫用途说明
  minConfidence: number;       // 最低置信度阈值
}

export const BODY_POINT_SPEC: Record<BodyPointName, BodyPointDef> = {
  // ── A. 头部与身体中轴 ──────────────────────────
  headCenter: {
    name: 'headCenter',
    mpIdx: MP_IDX.NOSE,
    isDerived: false,
    smoothingPriority: 'high',
    golfPurpose: ['head_movement', 'centered判断', 'stable判断'],
    minConfidence: 0.4,
  },
  headTop: {
    name: 'headTop',
    isDerived: true,
    derivedFrom: ['headCenter', 'neckCenter'],
    smoothingPriority: 'high',
    golfPurpose: ['身体高度变化', '是否过早站起', 'posture'],
    minConfidence: 0,
  },
  neckCenter: {
    name: 'neckCenter',
    isDerived: true,
    derivedFrom: ['leftShoulder', 'rightShoulder'],
    smoothingPriority: 'high',
    golfPurpose: ['脊柱上端参考', '上半身整体稳定性'],
    minConfidence: 0,
  },
  // ── B. 肩部 ──────────────────────────────────
  leftShoulder: {
    name: 'leftShoulder',
    mpIdx: MP_IDX.LEFT_SHOULDER,
    isDerived: false,
    smoothingPriority: 'high',
    golfPurpose: ['肩线', 'shoulder_tilt', 'shoulder_lifts_too_early', 'upper_body_structure'],
    minConfidence: 0.4,
  },
  rightShoulder: {
    name: 'rightShoulder',
    mpIdx: MP_IDX.RIGHT_SHOULDER,
    isDerived: false,
    smoothingPriority: 'high',
    golfPurpose: ['肩线', 'trail_shoulder判断', 'shoulder_lifts_too_early'],
    minConfidence: 0.4,
  },
  // ── C. 手臂 ──────────────────────────────────
  leftElbow: {
    name: 'leftElbow',
    mpIdx: MP_IDX.LEFT_ELBOW,
    isDerived: false,
    smoothingPriority: 'medium',
    golfPurpose: ['手臂结构', 'chicken_wing', 'arm_structure'],
    minConfidence: 0.35,
  },
  rightElbow: {
    name: 'rightElbow',
    mpIdx: MP_IDX.RIGHT_ELBOW,
    isDerived: false,
    smoothingPriority: 'medium',
    golfPurpose: ['手臂结构', 'chicken_wing', 'trail_arm'],
    minConfidence: 0.35,
  },
  leftWrist: {
    name: 'leftWrist',
    mpIdx: MP_IDX.LEFT_WRIST,
    isDerived: false,
    smoothingPriority: 'high',
    golfPurpose: ['hand_path', 'grip判断', 'arm_structure'],
    minConfidence: 0.35,
  },
  rightWrist: {
    name: 'rightWrist',
    mpIdx: MP_IDX.RIGHT_WRIST,
    isDerived: false,
    smoothingPriority: 'high',
    golfPurpose: ['hand_path', 'grip判断', 'arm_structure'],
    minConfidence: 0.35,
  },
  // ── D. 髋部 ──────────────────────────────────
  leftHip: {
    name: 'leftHip',
    mpIdx: MP_IDX.LEFT_HIP,
    isDerived: false,
    smoothingPriority: 'high',
    golfPurpose: ['髋线', 'hip_turn', 'early_extension', 'weight_shift'],
    minConfidence: 0.4,
  },
  rightHip: {
    name: 'rightHip',
    mpIdx: MP_IDX.RIGHT_HIP,
    isDerived: false,
    smoothingPriority: 'high',
    golfPurpose: ['髋线', 'hip_turn', 'early_extension', 'weight_shift'],
    minConfidence: 0.4,
  },
  // ── E. 下肢 ──────────────────────────────────
  leftKnee: {
    name: 'leftKnee',
    mpIdx: MP_IDX.LEFT_KNEE,
    isDerived: false,
    smoothingPriority: 'medium',
    golfPurpose: ['基础姿态稳定性', '下盘结构'],
    minConfidence: 0.3,
  },
  rightKnee: {
    name: 'rightKnee',
    mpIdx: MP_IDX.RIGHT_KNEE,
    isDerived: false,
    smoothingPriority: 'medium',
    golfPurpose: ['基础姿态稳定性', '下盘结构'],
    minConfidence: 0.3,
  },
  leftAnkle: {
    name: 'leftAnkle',
    mpIdx: MP_IDX.LEFT_ANKLE,
    isDerived: false,
    smoothingPriority: 'low',
    golfPurpose: ['重心参考', '下盘结构'],
    minConfidence: 0.25,
  },
  rightAnkle: {
    name: 'rightAnkle',
    mpIdx: MP_IDX.RIGHT_ANKLE,
    isDerived: false,
    smoothingPriority: 'low',
    golfPurpose: ['重心参考', '下盘结构'],
    minConfidence: 0.25,
  },
  // ── F. 派生点 ──────────────────────────────────
  gripCenter: {
    name: 'gripCenter',
    isDerived: true,
    derivedFrom: ['leftWrist', 'rightWrist'],
    smoothingPriority: 'high',
    golfPurpose: ['hand_path主点', '手应更高/低/里/外判断'],
    minConfidence: 0,
  },
  hipCenter: {
    name: 'hipCenter',
    isDerived: true,
    derivedFrom: ['leftHip', 'rightHip'],
    smoothingPriority: 'high',
    golfPurpose: ['脊柱线下端', '重心位移参考', 'early_extension', 'weight_shift'],
    minConfidence: 0,
  },
  shoulderCenter: {
    name: 'shoulderCenter',
    isDerived: true,
    derivedFrom: ['leftShoulder', 'rightShoulder'],
    smoothingPriority: 'high',
    golfPurpose: ['脊柱线上端', '中轴上端参考'],
    minConfidence: 0,
  },
};

/**
 * 从 MediaPipe 关键点数组中提取指定点位
 * 支持原始点和派生点
 */
export type Pt = { x: number; y: number; confidence?: number };

export function resolvePoint(
  name: BodyPointName,
  mpKeypoints: { x: number; y: number; visibility?: number; score?: number }[],
): Pt | null {
  const spec = BODY_POINT_SPEC[name];

  if (spec.isDerived && spec.derivedFrom) {
    // 派生点：计算来源点的平均值
    const srcs = spec.derivedFrom
      .map(src => resolvePoint(src, mpKeypoints))
      .filter((p): p is Pt => p !== null);
    if (srcs.length === 0) return null;
    const x = srcs.reduce((s, p) => s + p.x, 0) / srcs.length;
    const y = srcs.reduce((s, p) => s + p.y, 0) / srcs.length;

    // headTop: 在headCenter上方约0.5个"头高"
    if (name === 'headTop') {
      const headCenter = resolvePoint('headCenter', mpKeypoints);
      const neckCenter = resolvePoint('neckCenter', mpKeypoints);
      if (headCenter && neckCenter) {
        const headHeight = Math.abs(neckCenter.y - headCenter.y) * 0.8;
        return { x: headCenter.x, y: headCenter.y - headHeight };
      }
    }

    return { x, y, confidence: 1 };
  }

  // 原始点
  const idx = typeof spec.mpIdx === 'number' ? spec.mpIdx : (spec.mpIdx?.[0] ?? -1);
  if (idx < 0 || idx >= mpKeypoints.length) return null;

  const kp = mpKeypoints[idx];
  const conf = kp.visibility ?? kp.score ?? 1;
  if (conf < spec.minConfidence) return null;

  return { x: kp.x, y: kp.y, confidence: conf };
}

/**
 * 批量解析所有需要的点位
 */
export function resolveAllPoints(
  mpKeypoints: { x: number; y: number; visibility?: number; score?: number }[],
): Partial<Record<BodyPointName, Pt>> {
  const result: Partial<Record<BodyPointName, Pt>> = {};
  for (const name of Object.keys(BODY_POINT_SPEC) as BodyPointName[]) {
    const pt = resolvePoint(name, mpKeypoints);
    if (pt) result[name] = pt;
  }
  return result;
}

/**
 * 平滑权重（按优先级）
 */
export const SMOOTHING_ALPHA: Record<'high' | 'medium' | 'low', number> = {
  high:   0.65,  // 强平滑：头/肩/手/髋
  medium: 0.50,  // 中平滑：肘/膝
  low:    0.35,  // 弱平滑：踝
};
