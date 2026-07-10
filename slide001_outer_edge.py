#!/usr/bin/env python3
"""
CUE-SLIDE-001 v5 — 双脚绿基准线 + 红线(lead侧髋关节外缘) + barb箭头顶回
- 绿线: address 双脚 x 位置 (正常范围)
- 红线: impact 时 left_hip x = lead侧身体外缘 (slide后冲到的位置)
- 箭头: 从红线 lead侧压入, 指向 trail 方向 (纠正slide)
- 画面零标注
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

# ── geometry ─────────────────────────────────────────────────────────────────
la_A  = kpt(A,   'left_ankle')    # lead foot  @ address → 绿线位置
ra_A  = kpt(A,   'right_ankle')   # trail foot @ address → 绿线位置
lh_I  = kpt(IMP, 'left_hip')      # lead side hip @ impact = body outer edge (红线)
nose_I = kpt(IMP, 'nose')
la_I   = kpt(IMP, 'left_ankle')

x_grn_lead  = int(la_A[0])   # 绿线: lead foot  (397)
x_grn_trail = int(ra_A[0])   # 绿线: trail foot (262)
x_red       = int(lh_I[0])   # 红线: lead hip @ impact (385), lead侧身体外缘

# vertical extent: 头上方 → 脚下方
line_top = max(int((nose_I[1] if nose_I is not None else 250) - 50), 15)
line_bot = min(int((la_I[1]  if la_I  is not None else 900) + 50), 960)

# arrow: 纠正 slide = 把 lead 侧外缘往 trail 推 (图像中 -x 方向)
# tip 落在红线上, arrow tail 在红线 lead 侧 (trail←[arrow tail]····[tip]→lead )
# fwd = (-1, 0): 推力方向 = trail 方向
hip_y = float(lh_I[1])
STANDOFF = 35
tip = np.array([x_red + STANDOFF, hip_y])   # tail 在 lead 侧 35px 处
fwd = np.array([-1.0, 0.0])                  # 推向 trail

# ── colors & style ───────────────────────────────────────────────────────────
RED   = (17,  15, 228)
GREEN = (12, 220,  48)
LINE_W = 2

def ip(p): return (int(round(float(p[0]))), int(round(float(p[1]))))

def draw_barb(img, tip, fwd, col, length=22, hw=14, notch=0.38):
    fwd = np.array(fwd, float); fwd /= np.linalg.norm(fwd) + 1e-9
    perp = np.array([-fwd[1], fwd[0]])
    tail = tip - fwd * length
    notch_pt = tip - fwd * (length * (1 - notch))
    pts = np.array([ip(tip), ip(tail+perp*hw), ip(notch_pt), ip(tail-perp*hw)], dtype=np.int32)
    cv2.fillPoly(img, [pts], col)
    cv2.polylines(img, [pts], True, col, 1, cv2.LINE_AA)

# ── load impact frame ────────────────────────────────────────────────────────
cap = cv2.VideoCapture(str(VID))
for _ in range(IMP):
    cap.read()
ret, frame_raw = cap.read()
cap.release()
assert ret

img = frame_raw.copy()

# ── 两条绿竖线 ───────────────────────────────────────────────────────────────
cv2.line(img, (x_grn_lead,  line_top), (x_grn_lead,  line_bot), GREEN, LINE_W, cv2.LINE_AA)
cv2.line(img, (x_grn_trail, line_top), (x_grn_trail, line_bot), GREEN, LINE_W, cv2.LINE_AA)

# ── 红竖线 ─────────────────────────────────────────────────────────────────
cv2.line(img, (x_red, line_top), (x_red, line_bot), RED, LINE_W, cv2.LINE_AA)

# ── barb 箭头: lead侧压入 → 往trail推 ────────────────────────────────────────
draw_barb(img, tip, fwd, RED)

# ── save ─────────────────────────────────────────────────────────────────────
out = PREVIEW / "slide001_v5_outer_edge.jpg"
cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"=> {out}")

print(f"\n几何说明 (不上图):")
print(f"  trail green  x={x_grn_trail}  (right_ankle@addr)")
print(f"  red          x={x_red}          (left_hip@impact = lead侧髋关节外缘)")
print(f"  lead  green  x={x_grn_lead}  (left_ankle@addr)")
print(f"  red 距 lead green: {x_grn_lead - x_red}px (正=在绿线内侧)")
print(f"  red 相对 addr left_hip: +{x_red - int(kpt(A,'left_hip')[0])}px (slide位移)")
print(f"\n  箭头方向: trail (−x), 纠正 slide")
print(f"  arrow tip x={int(tip[0])}, 指向红线 x={x_red}")
