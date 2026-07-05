# CUE 设计迭代档案库

> 目录: `docs/07_CUE_DESIGN/design_iterations/`
> 用途: 存放每个 cue 指示器的历版设计稿（失败图、中间稿、定版图），记录每版推翻原因
> 维护规则:
>   - 每版设计稿由 Jason 人工拷入本目录，按命名规范放置
>   - Hermes 只负责更新本 INDEX，不自动生成设计图
>   - 版本号与推翻原因由 Jason 口述，Hermes 写入表格
>   - 定版图不在此存档（定版见 cue_renderer/ 输出）

---

## 命名规范

```
<error_code>_v<N>_<状态>.jpg
```

- error_code: 错误简称（reverse_pivot / sway / hip_slide 等）
- N: 版本号（1, 2, 3 ...）
- 状态: failed / draft / approved

示例:
- `reverse_pivot_v1_failed.jpg` — Reverse Pivot 第 1 版失败稿
- `reverse_pivot_v2_draft.jpg` — 第 2 版中间稿
- `reverse_pivot_v3_approved.jpg` — 第 3 版定版（正式版）

---

## 迭代记录表

> **注**: v1 指示器已渲染（见 `cue_renderer/` 输出目录），暂无失败迭代稿。
> Jason 后续可将失败版本图片拷入本目录，由 Hermes 补录推翻原因。

| 文件名 | 错误类型 | 版本 | 状态 | 推翻原因 / 备注 |
|--------|---------|------|------|----------------|
| *(待 Jason 拷入)* | — | — | — | — |

---

## 已知待立项迭代题（来自解构报告 v0.2 §6）

以下为生成器专项设计题，一旦有 mock 草稿请拷入本目录并更新上表：

1. **时序类单主体动画句型**（源 _119）: 髋部绿弧箭头先动 + 手部标记静止，动画脚本规范
2. **换部位/时序复合错误元素分工规范**（错误锚点 ≠ 正确锚点时的构图规则）

---

## 目录文件列表（持续更新）

*(当前为空，等待 Jason 拷入设计稿)*

---

*DESIGN_ITERATIONS_INDEX.md — 创建于 CUE-002 追加§7 / 2026-07-05*
*此文件随每次新增设计稿由 Hermes 更新*
