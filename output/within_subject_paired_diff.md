# within_subject_paired_diff.md
# 同人配对差值分析 — DTL 七段

**生成时间**: 2026-07-03
**被试**: Jason（全部七段均为同一人）
**机位**: 全部 DTL（球后方）
**数据来源**:
- spine_delta / hip_mid: pipeline_output/ dtl_measurement_table.json (batch2) + output/batch3_eet/*_diagnosis.json (batch3)
- hip_rear (R2' v1.1): dtl_hip_rear_table.json (batch2) + batch3 diagnosis JSONs
- 窗口: P5(transition开始)→impact（两组脚本均如此，batch2标注"downswing→impact"与P5等价）
- intent: GT_LABELS.md 意图申报照录

**GT铁律**: 本文无缺陷结论标签，intent 以 GT_LABELS.md 登记为准，非引擎自判。
**dtl-eet-1**: 未跑（Jason 待确认拼接），不纳入。

---

## 第1步: 汇总总表

| 文件 | intent(GT_LABELS登记) | spine_delta峰(°) | hip_mid峰(%) | hip_rear峰(%) | 备注 |
|---|---|---|---|---|---|
| dtl-ok-1   | ok(无意图申报)              | +3.93 | +17.6 | +15.4 | 基线 |
| dtl-ok-2   | ok(无意图申报)              | -8.24 | +16.3 | +24.6 | 基线 |
| dtl-wrong-1 | wrong(晃动+后蹲+臂翅)       | +4.67 | +17.0 | +17.5 | |
| dtl-wrong-2 | wrong(晃动+后蹲+臂翅)       | -3.48 | +28.4 | +25.1 | |
| dtl-wrong-3 | wrong(晃动+后蹲+臂翅)       | +1.61 | +17.3 | N/A   | EXCLUDED: top_conf=0.012, impact=fr388 |
| dtl-eet-2  | EET(Early Extension, batch3) | -15.60 | +28.2 | +33.9 | |
| dtl-eet-3  | EET(Early Extension, batch3) | -25.60 | +36.5 | +36.5 | |

spine_delta 符号规定: + = 脊柱前倾角减小(变直); - = 前倾角增大(加深前倾)
hip_mid/hip_rear 符号规定: + = 髋向球侧位移 (fraction of torso_h × 100)

---

## 第2步: 配对差值

正常基线 = dtl-ok-1/2 均值:
  spine_delta 基线: -2.16°
  hip_mid 基线:     +16.95%
  hip_rear 基线:    +20.00%

| 文件 | intent | Δspine(°) | Δhip_mid(%) | Δhip_rear(%) |
|---|---|---|---|---|
| dtl-wrong-1 | wrong(晃动+后蹲+臂翅) | +6.83 | +0.0 | -2.5 |
| dtl-wrong-2 | wrong(晃动+后蹲+臂翅) | -1.32 | +11.4 | +5.1 |
| dtl-wrong-3 | wrong(晃动+后蹲+臂翅) | +3.77 | +0.3 | N/A |
| dtl-eet-2  | EET | -13.45 | +11.2 | +13.9 |
| dtl-eet-3  | EET | -23.45 | +19.5 | +16.5 |

---

## 第3步: 观察 (纯数字，无诊断标签)

**spine_delta 差值 (Δ vs 基线 -2.16°)**:
- 最大负偏离 (前倾角增大最多): dtl-eet-3  Δ=-23.45°
- 次大负偏离:                   dtl-eet-2  Δ=-13.45°
- 正偏离 (前倾角减小):          dtl-wrong-1 Δ=+6.83°, dtl-wrong-3 Δ=+3.77°
- 接近基线:                     dtl-wrong-2 Δ=-1.32°

**hip_mid 差值 (Δ vs 基线 +16.95%)**:
- 最大正偏离: dtl-eet-3   Δ=+19.5%
- 次大 (接近): dtl-eet-2  Δ=+11.2%,  dtl-wrong-2 Δ=+11.4%
- 接近基线:   dtl-wrong-1 Δ=+0.0%,  dtl-wrong-3 Δ=+0.3%

**hip_rear 差值 (Δ vs 基线 +20.00%)**:
- 最大正偏离: dtl-eet-3   Δ=+16.5%
- 次大:       dtl-eet-2   Δ=+13.9%
- 小偏离:     dtl-wrong-2 Δ=+5.1%,  dtl-wrong-1 Δ=-2.5%
- dtl-wrong-3: N/A (EXCLUDED)

**特征值域**:
- spine_delta: +4.67° ~ -25.60°  (跨度 30.27°)
- hip_mid:    +17.0% ~ +36.5%   (跨度 19.5%)
- hip_rear:   +15.4% ~ +36.5%   (跨度 21.1%; dtl-wrong-3 除外)

**各杆三特征同时偏离情况**:
- dtl-eet-2/3: 三特征同时正向偏离基线，spine_delta 负偏离尤大
- dtl-wrong-2: hip_mid/hip_rear 偏离明显，spine_delta 接近基线
- dtl-wrong-1/3: 三特征均接近基线

---

*数据版本: batch2 dtl_measurement_table 2026-06-11 + batch3 diagnosis 2026-07-03*
*GT确认 = 意图申报 + 人工看图，非引擎自判*
