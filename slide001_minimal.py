#!/usr/bin/env python3
"""
CUE-SLIDE-001 极简静态原型
3个元素: 红竖线(髋错误位) + 绿竖线(髋正确位) + barb箭头(红推向绿)
无任何数字/标注/文字
"""
import json, cv2, numpy as np
from pathlib import Path

CACHE   = Path("engine/kp_cache/batch2/fo-wrong-4.json")
VID     = Path("/mnt/c/Users/jason/Zening/Swingcue/Video/fo-wrong-4.mp4")
PREVIEW = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview")
PREVIEW.mkdir(parents=True, exist_ok=True)

with open(CACHE) as f:
    data = json.load(f)
frames_data = data['frames']

A=85; IMP=147

def kpt(fi, name):
    fr = frames_data[fi]
    if not fr['persons']: return None
    v = fr['persons'][0]['keypoints'].get(name)
    return np.array([v['x'], v['y']]) if v else None

lh_A   = kpt(A,   'left_hip');  rh_A   = kpt(A,   'right_hip')
lh_IMP = kpt(IMP, 'left_hip');  rh_IMP = kpt(IMP, 'right_hip')

hi_A_x   = float((lh_A[0]   + rh_A[0])   / 2)   # 绿线 x
hi_IMP_x = float((lh_IMP[0] + rh_IMP[0]) / 2)   # 红线 x

# ── load impact frame ────────────────────────────────────────────────────────
cap = cv2.VideoCapture(str(VID))
for _ in range(IMP):
    cap.read()
ret, frame_raw = cap.read()
cap.release()
assert ret
VH, VW = frame_raw.shape[:2]

# 颜色 (BGR)
RED   = (17,  15, 228)
GREEN = (12, 220,  48)

LINE_W = 2       # 竖线宽度 (同鸡翅膀)
STANDOFF = 35    # 箭头距目标线距离

def ip(p): return (int(round(float(p[0]))), int(round(float(p[1]))))

def draw_barb(img, tip, fwd, col, length=22, hw=14, notch=0.38):
    fwd = np.array(fwd, float); fwd /= np.linalg.norm(fwd) + 1e-9
    perp = np.array([-fwd[1], fwd[0]])
    tail = tip - fwd * length
    notch_pt = tip - fwd * (length * (1 - notch))
    pts = np.array([ip(tip), ip(tail + perp*hw), ip(notch_pt), ip(tail - perp*hw)], dtype=np.int32)
    cv2.fillPoly(img, [pts], col)
    cv2.polylines(img, [pts], True, col, 1, cv2.LINE_AA)

# ── 竖线范围: 从头部上方到脚踝下方 ─────────────────────────────────────────
nose_y  = kpt(IMP, 'nose')
la_y    = kpt(IMP, 'left_ankle')
line_top = int((nose_y[1] if nose_y is not None else 200) - 40)
line_bot = int((la_y[1]  if la_y  is not None else VH-80) + 40)
line_top = max(line_top, 20)
line_bot = min(line_bot, VH - 20)

# 箭头高度: 髋中心 y
hip_y = float((lh_IMP[1] + rh_IMP[1]) / 2)

# 箭头方向: 从红(右)→绿(左) = 负x方向
fwd = np.array([-1.0, 0.0])   # 推回方向

# 箭头起点: 红线右侧 STANDOFF 处 → tip 指向红线
tip_x = hi_IMP_x + STANDOFF
tip   = np.array([tip_x, hip_y])

img = frame_raw.copy()

# ── 红竖线 ──────────────────────────────────────────────────────────────────
x_red = int(hi_IMP_x)
cv2.line(img, (x_red, line_top), (x_red, line_bot), RED, LINE_W, cv2.LINE_AA)

# ── 绿竖线 ──────────────────────────────────────────────────────────────────
x_grn = int(hi_A_x)
cv2.line(img, (x_grn, line_top), (x_grn, line_bot), GREEN, LINE_W, cv2.LINE_AA)

# ── barb 箭头 (红色, 指向红线) ──────────────────────────────────────────────
draw_barb(img, tip, fwd, RED)

out = PREVIEW / "slide001_v3_minimal.jpg"
cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"=> {out}")
print(f"red_x={x_red}  green_x={x_grn}  gap={x_red-x_grn:.0f}px")
