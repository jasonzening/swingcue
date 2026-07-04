# INDICATOR_ENGINEERING_SPEC.md — 指示器工程规格

> 产品规格见主文档 Ⅲ;本文档是工程实现层:输入点位、计算公式、输出 JSON、置信度规则、渲染要求、失败状态、验收。MVP-0 只实现 §2 spine_angle;其余按切片推进。

## 1. 统一输出 Schema(所有指示器必须遵守)

```json
{
  "indicator_type": "spine_angle",
  "frame_index": 47,
  "current_value": 39.2,
  "target_range": [, ],            // 来自分层模板;⚙️ 阈值待坎2后用真实数据校准
  "status": "good"|"warning"|"off"|null,   // 绿/黄/红;low confidence 时强制 null
  "confidence": 0.84,
  "confidence_level": "high"|"medium"|"low",
  "message": "一句大白话",
  "geometry": { ... }              // 渲染坐标;与 coach_annotations 渲染共用同结构
}
```

铁律:`confidence_level=low` → `status=null`,前端不渲染红绿,只显示不确定文案。

## 2. spine_angle 脊柱角/起身检测(MVP-0 唯一实现,侧面)

> ⚠️ **命名与边界:** MVP-0 算的是 **2D 画面前倾角**,不等于真实 3D 脊柱角。产品文案**不要叫"真实脊柱角"**,叫 **Posture Line / 起身检测 / Spine line stability**,避免过度承诺(与"近似要标注"原则一致)。

**坐标系(必须固定,否则不同实现会算出 30° 和 150° 两种结果):**
- x 向右,**y 向下**(视频坐标系)
- `vertical_vector = (0, 1)` 固定
- `spine_vector = shoulder_center - hip_center`
- `angle = arccos( dot(normalize(spine_vector), vertical_vector) )`,取画面前倾角

**输入点位:** neck/shoulder_center(双肩中点)、hip_center(双髋中点);可选 ankle 作地面参考。

**计算:**
1. address 帧:按上式算 `address_angle`(初始前倾角)
2. impact 帧(及全程逐帧):同式算 `current_angle`
3. `delta = current_angle - address_angle`
4. 判定(⚙️ 初版阈值,待校准):|delta| ≤ 5° → good;5-10° → warning("有一点起身");>10° → off("明显抬头/起身")

**置信度:** 取两帧中 neck/shoulder/hip 关键点检测置信度的最小均值;若 impact 帧本身 phase confidence 为 low,则本指示器整体 low。

**输出示例:**
```json
{ "indicator_type": "spine_angle", "frame_index": 47,
  "current_value": 39.2, "address_angle": 36.5, "delta": 2.7,
  "target_range": [-5, 5], "status": "good",
  "confidence": 0.84, "confidence_level": "high",
  "message": "姿势保持得不错,击球时没有明显起身。",
  "geometry": { "line": [[x1,y1],[x2,y2]], "ghost_line": [[gx1,gy1],[gx2,gy2]] } }
```

**渲染:** 用户当前脊柱线=黄实线;address 时的脊柱线=绿半透明虚线幽灵(贴身法,主文档 4.2);delta 超阈值时当前线黄→红。

**失败状态:** 肩/髋点缺失 → "请确保全身入镜";impact 不可信 → 用 top→finish 间置信度最高帧替代并 warning,或直接 low。

**生理极限测试:** 前倾角应在 0-70° 区间;超出 → 报错记录该帧,不输出给用户。

**验收:** test-dwontheline 上输出合理 delta;人为制造低置信(裁掉半身的视频)能正确触发 low + 不渲染红绿;overlay 线贴身不漂。

## 3. 后续指示器(实现排期,规格见主文档 Ⅲ,工程展开按本模板补)

| 切片 | 指示器 | 关键点位 | 核心公式 |
|---|---|---|---|
| MVP-1 | wrist_v(侧面) | 肘-腕-手方向(无杆头时近似) | V 夹角,下杆段监控提前释放 |
| MVP-1 | center_line(正面) | 头/颈+髋中点 vs address 锁定线 | 水平偏移量/肩宽比 |
| MVP-2 | shoulder_disk / hip_disk(正面) | 双肩/双髋连线 | 旋转区间感(不标精确度数) |
| MVP-2 | weight_point(正面) | 髋中点地面投影 | 左右重心位置与转移 |
| MVP-2 | swing_plane(侧面,近似) | 腕+杆向 | 平面色面,明确标注"近似" |
