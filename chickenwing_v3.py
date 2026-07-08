"""
CUE-CHICKENWING-001 Step 2 — Indicator V3
三色线: 红(错误手臂折线) + 绿(肩→腕直线) + 黄(肘→绿线箭头)
同起终点: lead 肩 / lead 腕
零文字 零幽灵 零虚线
"""

import json, cv2, numpy as np
from pathlib import Path

ROOT       = Path("/home/jason/projects/swingcue-postest")
USER_CACHE = ROOT / "engine/kp_cache/batch2/fo-ok-1.json"
USER_VID   = ROOT / "input/fo-ok-1.mp4"
OUT_STATIC = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/cw001_v3_static.jpg")
OUT_STATIC.parent.mkdir(parents=True, exist_ok=True)

WORST_FR = 96

with open(USER_CACHE) as f:
    user_frames = json.load(f)['frames']

def get_kp(frames, fi):
    fr = frames[fi]
    if not fr['persons']: return {}
    return {k: (v['x'], v['y'], v['score'])
            for k, v in fr['persons'][0]['keypoints'].items()}

def pt(kp, name):
    return np.array(kp[name][:2]) if name in kp else None

def ip_(a): return (int(a[0]), int(a[1]))

def extract_frame(vid, fi):
    cap = cv2.VideoCapture(str(vid))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

# ── keypoints at worst frame ──────────────────────────────────────────────────
kp = get_kp(user_frames, WORST_FR)
ls = pt(kp, 'left_shoulder')
le = pt(kp, 'left_elbow')
lw = pt(kp, 'left_wrist')
rs = pt(kp, 'right_shoulder')
re = pt(kp, 'right_elbow')
rw = pt(kp, 'right_wrist')
lh = pt(kp, 'left_hip')
rh = pt(kp, 'right_hip')

print(f"fr{WORST_FR}  ls={ip_(ls)}  le={ip_(le)}  lw={ip_(lw)}")

img = extract_frame(USER_VID, WORST_FR)
assert img is not None

# ── 1. muted background skeleton ─────────────────────────────────────────────
MUTED = (65, 65, 65)
for a, b in [(rs,re),(re,rw),(lh,rh),(ls,rs),(ls,lh),(rs,rh),(lw,rw)]:
    if a is not None and b is not None:
        cv2.line(img, ip_(a), ip_(b), MUTED, 2)
for p in [rs, re, rw, lh, rh, ls, lw]:
    if p is not None:
        cv2.circle(img, ip_(p), 4, MUTED, -1)

# ── 2. RED line — current wrong arm: shoulder → elbow → wrist ────────────────
RED   = (40, 60, 230)   # BGR red
LINE_W = 5
cv2.line(img, ip_(ls), ip_(le), RED, LINE_W)
cv2.line(img, ip_(le), ip_(lw), RED, LINE_W)
cv2.circle(img, ip_(ls), 9, RED, -1)
cv2.circle(img, ip_(le), 9, RED, -1)   # the "wrong" elbow — prominent
cv2.circle(img, ip_(lw), 9, RED, -1)

# ── 3. GREEN line — correct: same shoulder → same wrist (straight) ────────────
GREEN = (40, 200, 60)   # BGR green
cv2.line(img, ip_(ls), ip_(lw), GREEN, LINE_W)
# endpoints shared — draw over red dots with green to make sharing clear
cv2.circle(img, ip_(ls), 9, GREEN, -1)   # same shoulder
cv2.circle(img, ip_(lw), 9, GREEN, -1)   # same wrist

# ── 4. YELLOW arrow — from red elbow → green line (nearest point) ────────────
YELLOW = (0, 210, 255)  # BGR yellow

# find nearest point on green line (ls→lw) to current elbow
sw_vec = lw - ls
t      = np.dot(le - ls, sw_vec) / (np.dot(sw_vec, sw_vec) + 1e-9)
t      = float(np.clip(t, 0.05, 0.95))
target_on_green = ls + t * sw_vec          # point on green line closest to elbow

print(f"yellow arrow: {ip_(le)} → {ip_(target_on_green)}")

# draw a bold arrowedLine
cv2.arrowedLine(img,
    ip_(le),
    ip_(target_on_green),
    YELLOW, 4, tipLength=0.22)

cv2.imwrite(str(OUT_STATIC), img)
print(f"\n=> {OUT_STATIC}")
