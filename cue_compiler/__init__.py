"""
cue_compiler — CUE-004 关卡A
Cue Plan JSON → Lottie JSON / .lottie (dotLottie) / .mp4 preview

供应链纪律 (Jason 裁决 2026-07-05, CUE-004):
  Lottie JSON 格式完全公开 (https://lottiefiles.github.io/lottie-docs/)
  本模块直接手写 Lottie JSON dict，不依赖任何第三方 lottie 库。
  原因: 2024年 lottie-player npm 包投毒事件 (CVE-2024-1548 相关供应链攻击)
  证明第三方 lottie 工具链存在供应链风险。Python pip 侧亦有 lottie/lottie-python
  等包，版本未充分审计，故一律不引入。
  所有 lottie 产物由本模块纯 Python dict 构建，无外部 lottie 依赖。

依赖列表 (均已在 requirements.txt 中锁版本):
  Pillow >= 10.0      图像处理
  opencv-python       视频读写 / MP4 渲染
  numpy               数组操作
  (全部已安装，版本见 requirements.txt)
"""

from .lottie_builder import compile_lottie
from .mp4_renderer import render_mp4_preview
from .neutral_renderer import render_neutral_frame, render_silent_card

__all__ = [
    "compile_lottie",
    "render_mp4_preview",
    "render_neutral_frame",
    "render_silent_card",
]
