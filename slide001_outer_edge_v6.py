#!/usr/bin/env python3
"""
CUE-SLIDE-001 v6 — 红线贴lead侧髋关节外缘(绿线外侧)
修正: red_x = lead_hip_kp_x + 0.18*SW0 (body outer edge offset)
     → 红线在 lead 绿线外侧约9px, 明确越界感
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
la_A  = kpt(A,   'left_ankle')    # lead  foot @ address  → 绿线
ra_A  = kpt(A,   'right_ankle')   # trail foot @ address  → 绿线
lh_I  = kpt(IMP, 'left_hip')      # lead  hip  @ impact   (RTMPose KP)
ls_A  = kpt(A,   'left_shoulder')
rs_A  = kpt(A,   'right_shoulder')
nose_I = kpt(IMP, 'nose')
la_I   = kpt(IMP, 'left_ankle')

SW0 = float(np.linalg.norm(rs_A - ls_A))   # address 肩宽 = 尺度基准

# ── 关键 x 坐标 ──────────────────────────────────────────────────────────────
x_grn_lead  = int(la_A[0])   # 绿线 lead  side (左脚): 397
x_grn_trail = int(ra_A[0])   # 绿线 trail side (右脚): 262

# 红线 = lead侧髋关节KP + body edge offset
# RTMPose hip KP = 关节窝; 外表面 ≈ 关节窝x + 0.18×SW0
BODY_EDGE = 0.18 * SW0       # ≈ 20px, hip KP → 身体外表面
x_red = int(lh_I[0] + BODY_EDGE)   # ≈ 406, 在 lead 绿线(397) 外侧 9px

# ── 线段高度范围 ──────────────────────────────────────────────────────────────
line_top = max(int((nose_I[1] if nose_I is not None else 250) - 50), 15)
line_bot = min(int((la_I[1]   if la_I  is not None else 900) + 50), 960)

# ── 箭头: 从红线外侧压入, 推向 trail(纠正slide) ──────────────────────────────
hip_y    = float(lh_I[1])
STANDOFF = 35
tip = np.array([x_red + STANDOFF, hip_y])   # tail 在 lead 侧 35px
fwd = np.array([-1.0, 0.0])                  # 推向 trail 方向

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

# ── 2 绿竖线 ─────────────────────────────────────────────────────────────────
cv2.line(img, (x_grn_lead,  line_top), (x_grn_lead,  line_bot), GREEN, LINE_W, cv2.LINE_AA)
cv2.line(img, (x_grn_trail, line_top), (x_grn_trail, line_bot), GREEN, LINE_W, cv2.LINE_AA)

# ── 红竖线 (lead侧髋外缘, 绿线外侧) ─────────────────────────────────────────
cv2.line(img, (x_red, line_top), (x_red, line_bot), RED, LINE_W, cv2.LINE_AA)

# ── barb 箭头 ─────────────────────────────────────────────────────────────────
draw_barb(img, tip, fwd, RED)

# ── save ─────────────────────────────────────────────────────────────────────
out = PREVIEW / "slide001_v6_outer_edge.jpg"
cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"=> {out}")
print(f"\n坐标断言:")
print(f"  trail green x={x_grn_trail}")
print(f"  lead  green x={x_grn_lead}")
print(f"  red         x={x_red}  (lead_hip_kp={int(lh_I[0])} + BODY_EDGE={int(BODY_EDGE)})")
print(f"  red 在 lead green 外侧: {x_red - x_grn_lead:+d}px  {'✓ 越界感清晰' if x_red > x_grn_lead else '✗ 仍在绿线内'}")
