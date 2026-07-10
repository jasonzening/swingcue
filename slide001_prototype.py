#!/usr/bin/env python3
"""
CUE-SLIDE-001 静态原型
两版设计: 
  v1 — 竖直基准轴 + 红髋实际位置 + 绿髋目标 + 横向误差箭头
  v2 — 髋部轨迹弧线 + addr/top/impact三帧骨架 + 超阈值标注
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

A=85; TOP=126; IMP=147
NF=209

# ── geometry ────────────────────────────────────────────────────────────────
def kpt(fi, name):
    fr = frames_data[fi]
    if not fr['persons']: return None
    v = fr['persons'][0]['keypoints'].get(name)
    return np.array([v['x'], v['y']]) if v else None

def arr(name):
    return np.array([kpt(fi,name) if kpt(fi,name) is not None
                     else np.array([np.nan,np.nan]) for fi in range(NF)])

ls = arr('left_shoulder'); rs = arr('right_shoulder')
lh = arr('left_hip');      rh = arr('right_hip')
le = arr('left_elbow');    re = arr('right_elbow')
lw = arr('left_wrist');    rw = arr('right_wrist')
lk = arr('left_knee');     rk = arr('right_knee')
la = arr('left_ankle');    ra = arr('right_ankle')

hi_mid = (lh + rh) / 2
sh_mid = (ls + rs) / 2

SW0 = float(np.linalg.norm(rs[A] - ls[A]))
HW0 = float(np.linalg.norm(rh[A] - lh[A]))

hi_A   = hi_mid[A].copy()
hi_TOP = hi_mid[TOP].copy()
hi_IMP = hi_mid[IMP].copy()

slide_val  = (hi_IMP[0] - hi_A[0]) / SW0      # 0.258
thr_15     = 0.15                              # TPI疑似
thr_25     = 0.25                              # TPI明显
green_x    = hi_A[0] + thr_15 * SW0           # 最大可接受位
excess_px  = hi_IMP[0] - green_x              # 超出量
derot      = 17.4                              # 解旋角

RED   = (17, 15, 228)
GREEN = (12, 220, 48)
WHITE = (255, 255, 255)
GRAY  = (160, 160, 160)
ORANGE= (30, 140, 255)
CYAN  = (220, 200, 0)

def ip(p): return (int(round(float(p[0]))), int(round(float(p[1]))))

# ── load video frame ─────────────────────────────────────────────────────────
cap = cv2.VideoCapture(str(VID))
for _ in range(IMP):
    cap.read()
ret, frame_raw = cap.read()
cap.release()
assert ret, "failed to read impact frame"

VH, VW = frame_raw.shape[:2]

# ── skeleton helper ──────────────────────────────────────────────────────────
SKEL = [
    ('left_shoulder','left_elbow'),('left_elbow','left_wrist'),
    ('right_shoulder','right_elbow'),('right_elbow','right_wrist'),
    ('left_shoulder','right_shoulder'),
    ('left_shoulder','left_hip'),('right_shoulder','right_hip'),
    ('left_hip','right_hip'),
    ('left_hip','left_knee'),('right_hip','right_knee'),
    ('left_knee','left_ankle'),('right_knee','right_ankle'),
]

def draw_skel(img, fi, col, alpha=1.0, lw=1):
    fr = frames_data[fi]
    if not fr['persons']: return
    kp = {k: np.array([v['x'],v['y']]) for k,v in fr['persons'][0]['keypoints'].items()}
    if alpha < 1.0:
        ov = img.copy()
        for a,b in SKEL:
            pa=kp.get(a); pb=kp.get(b)
            if pa is None or pb is None: continue
            cv2.line(ov, ip(pa), ip(pb), col, lw, cv2.LINE_AA)
        for p in kp.values():
            cv2.circle(ov, ip(p), 3, col, -1, cv2.LINE_AA)
        cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)
    else:
        for a,b in SKEL:
            pa=kp.get(a); pb=kp.get(b)
            if pa is None or pb is None: continue
            cv2.line(img, ip(pa), ip(pb), col, lw, cv2.LINE_AA)
        for p in kp.values():
            cv2.circle(img, ip(p), 3, col, -1, cv2.LINE_AA)

def draw_barb(img, tip, fwd, col, length=22, hw=14, notch=0.38):
    fwd = np.array(fwd, float); fwd /= np.linalg.norm(fwd)+1e-9
    perp = np.array([-fwd[1], fwd[0]])
    tail = tip - fwd*length
    notch_pt = tip - fwd*(length*(1-notch))
    pts = np.array([ip(tip), ip(tail+perp*hw), ip(notch_pt), ip(tail-perp*hw)], dtype=np.int32)
    cv2.fillPoly(img, [pts], col)
    cv2.polylines(img, [pts], True, col, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# VERSION 1: 竖直基准轴 + 红位/绿位 + 超标箭头
# ═══════════════════════════════════════════════════════════════════════════════
img1 = (frame_raw * 0.50).astype(np.uint8)
draw_skel(img1, IMP, (130,130,130), alpha=1.0, lw=1)

hy = int(hi_IMP[1])   # hip y at impact

# ── vertical reference: address hip x (旋转轴) ──────────────────────────────
x_addr  = int(hi_A[0])
x_thr   = int(green_x)         # 0.15×SW0 threshold (绿线位置)
x_act   = int(hi_IMP[0])       # actual hip x at impact (红色)

# dashed vertical axis at address x
for y in range(60, VH-60, 14):
    cv2.line(img1, (x_addr, y), (x_addr, y+7), GRAY, 1, cv2.LINE_AA)
cv2.putText(img1, "ADDR HIP AXIS", (x_addr+4, 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, GRAY, 1, cv2.LINE_AA)

# green threshold line (0.15×SW0 = max acceptable)
for y in range(60, VH-60, 14):
    cv2.line(img1, (x_thr, y), (x_thr, y+7), GREEN, 1, cv2.LINE_AA)
cv2.putText(img1, f"MAX OK (+0.15×SW)", (x_thr+4, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, GREEN, 1, cv2.LINE_AA)

# red actual hip circle + fill
cv2.circle(img1, (x_act, hy), 14, RED, -1, cv2.LINE_AA)
cv2.circle(img1, (x_act, hy), 14, WHITE, 1, cv2.LINE_AA)

# green target hip circle
cv2.circle(img1, (x_thr, hy), 12, GREEN, -1, cv2.LINE_AA)
cv2.circle(img1, (x_thr, hy), 12, WHITE, 1, cv2.LINE_AA)

# red error bracket from green→red with barb arrow
mid_x = (x_thr + x_act) // 2
cv2.line(img1, (x_thr, hy), (x_act, hy), RED, 2, cv2.LINE_AA)
draw_barb(img1, np.array([x_act, hy], float),
          np.array([1.0, 0.0]), RED, length=20, hw=12)

# excess annotation
cv2.putText(img1, f"SLIDE +{excess_px:.0f}px", (x_act+4, hy-20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1, cv2.LINE_AA)
cv2.putText(img1, f"(+{slide_val:.2f}xSW)", (x_act+4, hy-4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, RED, 1, cv2.LINE_AA)

# derotation annotation
hi_w_IMP = float(np.linalg.norm(rh[IMP]-lh[IMP]))
cv2.putText(img1, f"HIP DEROT: only {derot:.0f}deg", (x_addr-80, hy+35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, ORANGE, 1, cv2.LINE_AA)
cv2.putText(img1, f"(should be ~40+deg)", (x_addr-80, hy+52),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, GRAY, 1, cv2.LINE_AA)

# hip line at impact (red)
cv2.line(img1, ip(lh[IMP]), ip(rh[IMP]), RED, 2, cv2.LINE_AA)

# title
cv2.putText(img1, "CUE-SLIDE-001 v1: AXIS + THRESHOLD", (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
cv2.putText(img1, f"fr{IMP} impact | slide={slide_val:.2f}xSW (TPI明显>{thr_25})", (14, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRAY, 1, cv2.LINE_AA)

out1 = PREVIEW / "slide001_v1_static.jpg"
cv2.imwrite(str(out1), img1, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"v1 => {out1}")


# ═══════════════════════════════════════════════════════════════════════════════
# VERSION 2: 三帧叠加 + 髋轨迹 + 简洁"髋应停、已滑"
# ═══════════════════════════════════════════════════════════════════════════════
img2 = (frame_raw * 0.50).astype(np.uint8)

# ghost skeleton at address (light gray)
draw_skel(img2, A,   (80,80,80),    alpha=0.6, lw=1)
# ghost skeleton at top (dim blue)
draw_skel(img2, TOP, (120,80,40),   alpha=0.5, lw=1)
# live skeleton at impact (mid gray)
draw_skel(img2, IMP, (140,140,140), alpha=1.0, lw=1)

# hip trajectory: addr → every 5 frames → impact
traj_frs = list(range(A, IMP+1, 4)) + [IMP]
prev = None
for fi in traj_frs:
    curr = ip(hi_mid[fi])
    if prev is not None:
        # color from gray→red as slide increases
        t = max(0.0, (hi_mid[fi][0] - hi_A[0]) / (hi_IMP[0]-hi_A[0]+1e-9))
        col = tuple(int(RED[i]*t + GRAY[i]*(1-t)) for i in range(3))
        cv2.line(img2, prev, curr, col, 2, cv2.LINE_AA)
        cv2.circle(img2, curr, 3, col, -1, cv2.LINE_AA)
    prev = curr

# address hip: gray circle (origin)
cv2.circle(img2, ip(hi_A),   10, GRAY,  -1, cv2.LINE_AA)
cv2.circle(img2, ip(hi_A),   10, WHITE,  1, cv2.LINE_AA)
cv2.putText(img2, "A", (ip(hi_A)[0]-4, ip(hi_A)[1]+4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1)

# top hip: dim orange circle
cv2.circle(img2, ip(hi_TOP), 10, ORANGE, -1, cv2.LINE_AA)
cv2.circle(img2, ip(hi_TOP), 10, WHITE,   1, cv2.LINE_AA)
cv2.putText(img2, "T", (ip(hi_TOP)[0]-4, ip(hi_TOP)[1]+4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1)

# impact hip actual: RED circle (大)
cv2.circle(img2, ip(hi_IMP), 14, RED,   -1, cv2.LINE_AA)
cv2.circle(img2, ip(hi_IMP), 14, WHITE,  1, cv2.LINE_AA)
cv2.putText(img2, "I", (ip(hi_IMP)[0]-4, ip(hi_IMP)[1]+5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1)

# green target (threshold position)
tgt = np.array([green_x, hi_IMP[1]])
cv2.circle(img2, ip(tgt), 12, GREEN, -1, cv2.LINE_AA)
cv2.circle(img2, ip(tgt), 12, WHITE,  1, cv2.LINE_AA)
cv2.putText(img2, "OK", (ip(tgt)[0]-8, ip(tgt)[1]+5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.28, WHITE, 1)

# error arrow from green target → red actual
cv2.line(img2, ip(tgt), ip(hi_IMP), RED, 2, cv2.LINE_AA)
draw_barb(img2, hi_IMP.copy(), np.array([1.0, 0.0]), RED, length=18, hw=11)

# hip line at impact (red, bold)
cv2.line(img2, ip(lh[IMP]), ip(rh[IMP]), RED, 3, cv2.LINE_AA)

# hip line at address (gray reference)
cv2.line(img2, ip(lh[A]), ip(rh[A]), GRAY, 1, cv2.LINE_AA)

# annotation block
ax = int(hi_A[0]) - 95
ay = int(hi_A[1]) - 60
cv2.rectangle(img2, (ax-5, ay-16), (ax+210, ay+65), (20,20,30), -1)
cv2.rectangle(img2, (ax-5, ay-16), (ax+210, ay+65), (60,60,80), 1)
cv2.putText(img2, f"SLIDE: +{slide_val:.2f}xSW  (TPI>{thr_25}=明显)", (ax, ay),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, RED, 1, cv2.LINE_AA)
cv2.putText(img2, f"髋横移: +{hi_IMP[0]-hi_A[0]:.0f}px  超阈值: +{excess_px:.0f}px",
            (ax, ay+18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, GRAY, 1, cv2.LINE_AA)
cv2.putText(img2, f"髋解旋: only {derot:.0f}deg (expected >40deg)",
            (ax, ay+36), cv2.FONT_HERSHEY_SIMPLEX, 0.36, ORANGE, 1, cv2.LINE_AA)
cv2.putText(img2, f"= 滑而不转 (slide, not rotate)",
            (ax, ay+54), cv2.FONT_HERSHEY_SIMPLEX, 0.38, ORANGE, 1, cv2.LINE_AA)

# title
cv2.putText(img2, "CUE-SLIDE-001 v2: TRAJECTORY + 3-FRAME", (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

out2 = PREVIEW / "slide001_v2_static.jpg"
cv2.imwrite(str(out2), img2, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"v2 => {out2}")

# ── detection report ─────────────────────────────────────────────────────────
print()
print("="*68)
print("CUE-SLIDE-001 检测结果 — fo-wrong-4")
print("="*68)
print(f"  判据: 髋中心 x位移/SW0, addr→impact, lead方向为正")
print(f"  SW0 (address肩宽): {SW0:.1f}px")
print(f"  髋位移 addr→impact: +{hi_IMP[0]-hi_A[0]:.1f}px = +{slide_val:.3f}×SW0")
print(f"  髋位移 top→impact:  +{hi_IMP[0]-hi_TOP[0]:.1f}px = +{(hi_IMP[0]-hi_TOP[0])/SW0:.3f}×SW0")
print(f"  髋解旋角 top→impact: {derot:.1f}° (充分解旋应≥40°)")
print()
print(f"  TPI疑似阈值  0.15×SW0 = +{0.15*SW0:.1f}px")
print(f"  TPI明显阈值  0.25×SW0 = +{0.25*SW0:.1f}px")
print(f"  本例实测:              +{slide_val:.3f}×SW0 ← 超明显阈值")
print(f"  超出量:                +{excess_px:.1f}px (+{excess_px/SW0:.2f}×SW0 beyond 0.15)")
print()
print(f"  综合判断: slide_val={slide_val:.2f} + derot仅{derot:.0f}° = 典型'滑而不转'")
print(f"  触发: ✓ (超明显阈值0.25)")
print()
print(f"  请Jason定阈值: 建议≥0.15触发(TPI疑似) 或 ≥0.20(保守)")
