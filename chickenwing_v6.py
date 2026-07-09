"""
CUE-CHICKENWING-001 — v6
1. 箭头起点近肘 (standoff=25px)
2. 终点=绿肘节点精确
3. 纯正红 (0,0,220)
4. 肘节点改小点 (r=5)
5. 所有节点统一小 (r=6)
"""

import json, cv2, numpy as np
from pathlib import Path
from PIL import Image as PILImage

ROOT  = Path("/home/jason/projects/swingcue-postest")
CACHE = ROOT / "engine/kp_cache/batch2/fo-wrong-4.json"
VID   = Path("/mnt/c/Users/jason/Zening/Swingcue/Video/fo-wrong-4.mp4")
OUT   = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/cw001_v6_green.gif")
OUT.parent.mkdir(parents=True, exist_ok=True)

DISPLAY_FR = 149
STANDOFF   = 25    # close to elbow — user sees "from elbow outward"

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
sw_vec   = lw - ls
t_elbow  = float(np.clip(
    np.dot(le - ls, sw_vec) / (np.dot(sw_vec, sw_vec) + 1e-9), 0.1, 0.9))
green_elbow = ls + t_elbow * sw_vec      # green elbow node

outward      = le - green_elbow
outward_norm = outward / (np.linalg.norm(outward) + 1e-9)

travel_start = le + outward_norm * STANDOFF   # close to elbow
travel_end   = green_elbow                    # stops exactly at node
travel_vec   = travel_end - travel_start
travel_dir   = travel_vec / (np.linalg.norm(travel_vec) + 1e-9)

print(f"green_elbow:  {tuple(green_elbow.astype(int))}")
print(f"travel_start: {tuple(travel_start.astype(int))}")
print(f"travel_len:   {np.linalg.norm(travel_vec):.1f}px")

# ── colors ────────────────────────────────────────────────────────────────────
PURE_RED     = (0,   0, 220)   # pure red BGR
BRIGHT_GREEN = (50, 255,  80)
MUTED        = (60,  60,  60)
NODE_R       = 6               # unified node radius for all joints

def ip(a): return (int(round(a[0])), int(round(a[1])))

# ── green glow line + small nodes ─────────────────────────────────────────────
def draw_green_glow(img):
    ip_ls, ip_lw, ip_ge = ip(ls), ip(lw), ip(green_elbow)
    # glow layers
    ov1 = img.copy()
    cv2.line(ov1, ip_ls, ip_lw, (30,180,50), 10, cv2.LINE_AA)
    cv2.addWeighted(ov1, 0.28, img, 0.72, 0, img)
    ov2 = img.copy()
    cv2.line(ov2, ip_ls, ip_lw, (40,230,65), 6, cv2.LINE_AA)
    cv2.addWeighted(ov2, 0.42, img, 0.58, 0, img)
    # core line
    cv2.line(img, ip_ls, ip_lw, BRIGHT_GREEN, 3, cv2.LINE_AA)
    # unified small nodes: shoulder, wrist, elbow-node
    for p in [ip_ls, ip_lw]:
        cv2.circle(img, p, NODE_R, BRIGHT_GREEN, -1)
    # elbow node = small dot only (no ring)
    cv2.circle(img, ip_ge, 5, BRIGHT_GREEN, -1)

# ── red arm + small nodes ──────────────────────────────────────────────────────
def draw_red_arm(img):
    cv2.line(img, ip(ls), ip(le), PURE_RED, 4, cv2.LINE_AA)
    cv2.line(img, ip(le), ip(lw), PURE_RED, 4, cv2.LINE_AA)
    for p in [ls, le, lw]:
        cv2.circle(img, ip(p), NODE_R, PURE_RED, -1)

# ── muted skeleton ────────────────────────────────────────────────────────────
def draw_muted(img):
    for a,b in [(rs,re),(re,rw),(lh,rh),(ls,rs),(ls,lh),(rs,rh),(lw,rw)]:
        if a is not None and b is not None:
            cv2.line(img, ip(a), ip(b), MUTED, 2)
    for p in [rs,re,rw,lh,rh]:
        if p is not None: cv2.circle(img, ip(p), 4, MUTED, -1)

# ── chevron ───────────────────────────────────────────────────────────────────
def draw_chevron(img, tip, fwd, size=13, thickness=4):
    perp = np.array([-fwd[1], fwd[0]])
    root = tip - fwd * size
    cv2.line(img, ip(tip), ip(root + perp*size), BRIGHT_GREEN, thickness, cv2.LINE_AA)
    cv2.line(img, ip(tip), ip(root - perp*size), BRIGHT_GREEN, thickness, cv2.LINE_AA)

N_CHEV = 3; GAP = 18

def draw_series(img, anchor):
    for i in range(N_CHEV):
        draw_chevron(img, anchor + travel_dir * i * GAP, travel_dir)

# ── bake base (won't change per frame) ───────────────────────────────────────
base_raw = extract_frame(DISPLAY_FR)
assert base_raw is not None

base = base_raw.copy()
draw_muted(base)
draw_red_arm(base)
draw_green_glow(base)

# ── animation timeline ────────────────────────────────────────────────────────
def smoothstep(t): return t*t*(3-2*t)

PUSH=12; PAUSE_END=5; RETREAT=9; PAUSE_MID=4; PAUSE_LONG=10

def make_timeline():
    tl = []
    for cycle in range(2):
        for i in range(PUSH):      tl.append(smoothstep(i/(PUSH-1)))
        for _ in range(PAUSE_END): tl.append(1.0)
        for i in range(RETREAT):   tl.append(1.0 - smoothstep(i/(RETREAT-1)))
        for _ in range(PAUSE_LONG if cycle==1 else PAUSE_MID): tl.append(0.0)
    return tl

timeline = make_timeline()

gif_frames = []; durations = []
for e in timeline:
    img    = base.copy()
    anchor = travel_start + e * travel_vec
    draw_series(img, anchor)
    gif_frames.append(PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    durations.append(50 if (0 < e < 1) else 200)

gif_frames[0].save(str(OUT), save_all=True,
                   append_images=gif_frames[1:],
                   loop=0, duration=durations, optimize=False)
print(f"=> {OUT}")
