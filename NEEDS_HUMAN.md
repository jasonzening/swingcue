# NEEDS_HUMAN.md

## Gate-1 v3 ready for review

B层8阶段标注已完成（v3修复版）。5段视频的阶段摘要图放在：
- WSL: ~/projects/swingcue-postest/keyframes/gate1_preview/
- 桌面: C:\Users\jason\Desktop\rtmpose_results\preview\gate1\

v3修复内容：
- 多挥杆检测（201015含3次挥杆，已截取第一次挥杆范围做检测）
- 输出 swing_count 与 first_swing_end，显示在 gate1 图头
- impact 改取第一个峰（chronological），不再取最高prominence峰
- 置信度三因子公式：信号显著度50%+多挥杆歧义30%+关节质量20%
- 等待人工 GT 标注（201058 fr180-200 / 201015 fr55-72 单帧图在桌面 gate1_gt/）

**注意：GT 只来自人工标注，不得使用检测值自造基准。**

请核对每张图：
1. 8阶段分得对不对？
2. address/top/impact 缩略图是否对应正确动作
3. 201015 的 swing_count 是否正确
4. 各 conf 是否有区分度（不再全=1.00）


## GT Line Rendering — gt_lines/ (2026-06-10)

Rendered 227 annotated frames to:
  Desktop/rtmpose_results/preview/gt_lines/

Per-video sub-folders:
  DTL (201054, 201058): Tush Line (yellow) + Spine axis (cyan)
    -> Window: P5 transition through impact+5
  Face-on (201015, 201039, 201047):
    backswing/   : Head vertical line (magenta) — RP check
    downswing/   : Head horizontal line (orange) — LoP check
    followthrough/: Lead forearm chain (green) + elbow angle — CW check

**NOT RENDERED** (deferred):
  - Shaft plane line (Over-the-Top check): requires ball/club detection pipeline
    which is not yet built. Will be added when club detection flow is complete.

Human action: inspect frames in gt_lines/ and confirm or correct anchor frames.
