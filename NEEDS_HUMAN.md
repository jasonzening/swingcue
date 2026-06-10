# NEEDS_HUMAN.md

## Gate-1 v2 ready for review

B层8阶段标注已完成（v2修复版）。5段视频的阶段摘要图放在：
- WSL: ~/projects/swingcue-postest/keyframes/gate1_preview/
- 桌面: C:\Users\jason\Desktop\rtmpose_results\preview\gate1\

v2修复内容:
- Fix1: impact用wrist-Y-max(face-on)/wrist-X-max(DTL)，取第一个峰（非全局最大）
- Fix2: transition阶段不再为空，top阶段窗口[TOP-1, TOP+2]
- Fix3: impact置信度改用峰值prominence/torso_height
- Fix4: 201015_wrist_y_curve.png已生成在桌面

请核对每张图：
1. 8个阶段分得对不对？
2. 特别看 address/top/impact 的缩略图是否对应正确的动作
3. 如有问题请指出具体视频和阶段

人验收通过前，E2代码可以继续写但不得用这些帧号做C层计算。
