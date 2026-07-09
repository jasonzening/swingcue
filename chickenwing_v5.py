"""
CUE-CHICKENWING-001 — v5 全绿版 修正4
- 绿线有肘节点 (投影点圆点)
- 箭头终点 = 绿肘节点
- 绿线收细
- 红绿粗细平衡
"""

import json, cv2, numpy as np
from pathlib import Path
from PIL import Image as PILImage

ROOT  = Path("/home/jason/projects/swingcue-postest")
CACHE = ROOT / "engine/kp_cache/batch2/fo-wrong-4.json"
VID   = Path("/mnt/c/Users/jason/Zening/Swingcue/Video/fo-wrong-4.mp4")
OUT   = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/cw001_v5_green.gif")
OUT.parent.mkdir(parents=True, exist_ok=True)

DISPLAY_FR = 149
STANDOFF   = 90

with open(CACHE) as f:
    frames = json.load(f)['frames']

def get_kp(fi):
    fr = frames[fi]
    if not fr['persons']: return {}
    return {k: (v['x'], v['y'], v['score'])
            for k, v in fr['persons'][0]['keypoints'].items()}

def pt(kp, name):
    return np.array(kp[name][:2], dtype=float) if name in kp else None

def extract_frame(fi):
    cap = cv2.VideoCapture(str(VID))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

kp = get_kp(DISPLAY_FR)
ls = pt(kp,'left_shoulder'); le = pt(kp,'left_elbow')
lw = pt(kp,'left_wrist');    rs = pt(kp,'right_shoulder')
re = pt(kp,'right_elbow');   rw = pt(kp,'right_wrist')
lh = pt(kp,'left_hip');      rh = pt(kp,'right_hip')

# ── geometry ──────────────────────────────────────────────────────────────────
sw_vec = lw - ls
# projection of real elbow onto shoulder-wrist line = green elbow node
t_elbow = float(np.clip(
    np.dot(le - ls, sw_vec) / (np.dot(sw_vec, sw_vec) + 1e-9), 0.1, 0.9))
green_elbow = ls + t_elbow * sw_vec      # ← green elbow node (target)

outward = le - green_elbow
outward_norm = outward / (np.linalg.norm(outward) + 1e-9)

travel_start = le + outward_norm * STANDOFF
travel_end   = green_elbow               # ← arrow stops HERE (not past green line)
travel_vec   = travel_end - travel_start
travel_dir   = travel_vec / (np.linalg.norm(travel_vec) + 1e-9)

print(f"green_elbow node: {tuple(green_elbow.astype(int))}")
print(f"travel: {tuple(travel_start.astype(int))} -> {tuple(travel_end.astype(int))}")
print(f"travel_len={np.linalg.norm(travel_vec):.1f}px")

# ── colors ────────────────────────────────────────────────────────────────────
RED          = (50,  70, 220)   # slightly softer red
BRIGHT_GREEN = (50, 255,  80)   # chevron + node + line core
MUTED        = (60,  60,  60)

def ip(a): return (int(round(a[0])), int(round(a[1])))

# ── draw green glow line (thinner) ────────────────────────────────────────────
def draw_green_glow(img, p1, p2, elbow_node):
    ip1, ip2, ipe = ip(p1), ip(p2), ip(elbow_node)
    # outer glow — narrow (was 18, now 10)
    ov1 = img.copy()
    cv2.line(ov1, ip1, ip2, (30, 180, 50), 10, cv2.LINE_AA)
    cv2.addWeighted(ov1, 0.30, img, 0.70, 0, img)
    # mid glow
    ov2 = img.copy()
    cv2.line(ov2, ip1, ip2, (40, 230, 65), 6, cv2.LINE_AA)
    cv2.addWeighted(ov2, 0.45, img, 0.55, 0, img)
    # bright core (was 5, now 3)
    cv2.line(img, ip1, ip2, BRIGHT_GREEN, 3, cv2.LINE_AA)
    # endpoints
    cv2.circle(img, ip1, 7, BRIGHT_GREEN, -1)
    cv2.circle(img, ip2, 7, BRIGHT_GREEN, -1)
    # elbow node — larger circle, with inner dot
    cv2.circle(img, ipe, 10, BRIGHT_GREEN, 2)     # ring
    cv2.circle(img, ipe,  4, BRIGHT_GREEN, -1)    # center dot

# ── chevron ───────────────────────────────────────────────────────────────────
def draw_chevron(img, tip, fwd, size=13, thickness=4, color=BRIGHT_GREEN):
    perp = np.array([-fwd[1], fwd[0]])
    arm_root = tip - fwd * size
    cv2.line(img, ip(tip), ip(arm_root + perp*size), color, thickness, cv2.LINE_AA)
    cv2.line(img, ip(tip), ip(arm_root - perp*size), color, thickness, cv2.LINE_AA)

N_CHEV = 3; GAP = 20

def draw_series(img, anchor):
    for i in range(N_CHEV):
        draw_chevron(img, anchor + travel_dir * i * GAP, travel_dir)

# ── base layer ────────────────────────────────────────────────────────────────
base_raw = extract_frame(DISPLAY_FR)
assert base_raw is not None

def build_base(img):
    for a,b in [(rs,re),(re,rw),(lh,rh),(ls,rs),(ls,lh),(rs,rh),(lw,rw)]:
        if a is not None and b is not None:
            cv2.line(img, ip(a), ip(b), MUTED, 2)
    for p in [rs,re,rw,lh,rh]:
        if p is not None: cv2.circle(img, ip(p), 4, MUTED, -1)
    # red arm — balanced thickness (4, was 5)
    cv2.line(img, ip(ls), ip(le), RED, 4)
    cv2.line(img, ip(le), ip(lw), RED, 4)
    for p,r in [(ls,7),(le,10),(lw,7)]: cv2.circle(img, ip(p), r, RED, -1)

base_with_green = base_raw.copy()
build_base(base_with_green)
draw_green_glow(base_with_green, ls, lw, green_elbow)

# ── timeline: push×2, stop at start ──────────────────────────────────────────
def smoothstep(t): return t*t*(3-2*t)

PUSH=14; PAUSE_END=5; RETREAT=10; PAUSE_MID=4; PAUSE_LONG=10

def make_timeline():
    tl = []
    for cycle in range(2):
        for i in range(PUSH):    tl.append(smoothstep(i/(PUSH-1)))
        for _ in range(PAUSE_END): tl.append(1.0)
        for i in range(RETREAT): tl.append(1.0 - smoothstep(i/(RETREAT-1)))
        pause = PAUSE_LONG if cycle==1 else PAUSE_MID
        for _ in range(pause):   tl.append(0.0)
    return tl

timeline = make_timeline()   # list of floats 0.0(at start)~1.0(at green node)

gif_frames = []; durations = []
for e in timeline:
    img    = base_with_green.copy()
    anchor = travel_start + e * travel_vec
    draw_series(img, anchor)
    gif_frames.append(PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    dur = 50 if (0 < e < 1) else 200
    durations.append(dur)

gif_frames[0].save(str(OUT), save_all=True,
                   append_images=gif_frames[1:],
                   loop=0, duration=durations, optimize=False)
print(f"=> {OUT}")
