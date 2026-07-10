#!/usr/bin/env python3
"""
CUE-SLIDE-001 v4 — 双脚绿基准线 + 红线(髋冲出位) + barb箭头顶回
极简: 画面只有 2条绿竖线 + 1条红竖线 + 箭头。无任何文字/数字。
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
# green lines: address双脚x位置
la_A = kpt(A, 'left_ankle')    # lead foot (x大)
ra_A = kpt(A, 'right_ankle')   # trail foot (x小)

# red line: impact髋中心x
lh_I = kpt(IMP, 'left_hip'); rh_I = kpt(IMP, 'right_hip')
hi_IMP_x = float((lh_I[0] + rh_I[0]) / 2)

# nose and ankle for line extent
nose_I  = kpt(IMP, 'nose')
la_I    = kpt(IMP, 'left_ankle')

x_lead  = int(la_A[0])      # 绿线 — lead foot
x_trail = int(ra_A[0])      # 绿线 — trail foot
x_red   = int(hi_IMP_x)     # 红线 — 髋实际位

# vertical extent: head top → ankle bottom
line_top = max(int((nose_I[1] if nose_I is not None else 250) - 50), 15)
line_bot = min(int((la_I[1]  if la_I  is not None else 900) + 50), 960)

# arrow at hip height
hip_y = float((lh_I[1] + rh_I[1]) / 2)

# ── colors ───────────────────────────────────────────────────────────────────
RED   = (17,  15, 228)
GREEN = (12, 220,  48)
LINE_W = 2
STANDOFF = 35    # 箭头 tip 距红线的距离

def ip(p): return (int(round(float(p[0]))), int(round(float(p[1]))))

def draw_barb(img, tip, fwd, col, length=22, hw=14, notch=0.38):
    fwd = np.array(fwd, float); fwd /= np.linalg.norm(fwd) + 1e-9
    perp = np.array([-fwd[1], fwd[0]])
    tail = tip - fwd * length
    notch_pt = tip - fwd * (length * (1 - notch))
    pts = np.array([ip(tip), ip(tail + perp*hw), ip(notch_pt), ip(tail - perp*hw)], dtype=np.int32)
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

# ── draw 2 green lines ───────────────────────────────────────────────────────
cv2.line(img, (x_lead,  line_top), (x_lead,  line_bot), GREEN, LINE_W, cv2.LINE_AA)
cv2.line(img, (x_trail, line_top), (x_trail, line_bot), GREEN, LINE_W, cv2.LINE_AA)

# ── draw red line ─────────────────────────────────────────────────────────────
cv2.line(img, (x_red, line_top), (x_red, line_bot), RED, LINE_W, cv2.LINE_AA)

# ── barb arrow: from lead-side of red line, pointing toward lead green line ──
# slide = hip pushed toward lead (left, x增大); arrow pushes it BACK (右→左 i.e. -x)
# tip points at red line from the outside (trail side of red line, i.e. x_red + STANDOFF)
tip  = np.array([x_red + STANDOFF, hip_y])
fwd  = np.array([-1.0, 0.0])   # push direction: trail→lead (向绿线方向: 减小x)
draw_barb(img, tip, fwd, RED)

# ── save ─────────────────────────────────────────────────────────────────────
out = PREVIEW / "slide001_v4_feet_baseline.jpg"
cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"=> {out}")

# diagnostic (not on image)
print(f"\n几何说明 (不上图):")
print(f"  lead foot (左脚) x={x_lead}  trail foot (右脚) x={x_trail}")
print(f"  hip@impact x={x_red}")
print(f"  hip到lead绿线距离: {x_lead - x_red:.0f}px (正=未出界, 负=已出界)")
print(f"  ← 本例髋未冲出脚线({x_lead-x_red:.0f}px inside), 但已明显向lead侧滑动")
print(f"  address髋x=326  impact髋x={x_red}  Δ=+{x_red-326:.0f}px toward lead")
print(f"\n  Jason可选框架选项:")
print(f"  (A) 严格\"脚线出界\"框架: 本例未触发(hip={x_red} < lead_foot={x_lead})")
print(f"  (B) \"髋相对address位移\"框架: 触发(+29px/0.26×SW0, TPI明显阈值0.25)")
print(f"  两者结合: 脚线作视觉参考, 位移量作触发判据")
