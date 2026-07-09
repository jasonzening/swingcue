"""
CUE-CHICKENWING-001 — v7
箭头: 单向推进 → 碰绿肘节点消失 → 起点重现 → 循环 (无退回)
红色: RGB(255,0,0) = BGR(0,0,255) 纯红
其余 v6 保持
"""

import json, cv2, numpy as np
from pathlib import Path
from PIL import Image as PILImage

ROOT  = Path("/home/jason/projects/swingcue-postest")
CACHE = ROOT / "engine/kp_cache/batch2/fo-wrong-4.json"
VID   = Path("/mnt/c/Users/jason/Zening/Swingcue/Video/fo-wrong-4.mp4")
OUT   = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/cw001_v8_green.gif")
OUT.parent.mkdir(parents=True, exist_ok=True)

DISPLAY_FR = 149
STANDOFF   = 20

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

# geometry
sw_vec  = lw - ls
t_elbow = float(np.clip(
    np.dot(le-ls, sw_vec)/(np.dot(sw_vec,sw_vec)+1e-9), 0.1, 0.9))
green_elbow  = ls + t_elbow * sw_vec
outward      = le - green_elbow
outward_norm = outward / (np.linalg.norm(outward)+1e-9)
travel_start = le + outward_norm * STANDOFF
travel_end   = green_elbow
travel_vec   = travel_end - travel_start
travel_dir   = travel_vec / (np.linalg.norm(travel_vec)+1e-9)

print(f"green_elbow:  {tuple(green_elbow.astype(int))}")
print(f"travel_start: {tuple(travel_start.astype(int))}")
print(f"travel_len:   {np.linalg.norm(travel_vec):.1f}px")

# colors
PURE_RED     = (0,   0, 255)   # BGR = RGB(255,0,0) 纯正红
BRIGHT_GREEN = (50, 255,  80)
MUTED        = (60,  60,  60)
NODE_R       = 6

def ip(a): return (int(round(a[0])), int(round(a[1])))

def draw_green_glow(img):
    ip_ls, ip_lw, ip_ge = ip(ls), ip(lw), ip(green_elbow)
    ov1 = img.copy()
    cv2.line(ov1, ip_ls, ip_lw, (30,180,50), 10, cv2.LINE_AA)
    cv2.addWeighted(ov1, 0.28, img, 0.72, 0, img)
    ov2 = img.copy()
    cv2.line(ov2, ip_ls, ip_lw, (40,230,65), 6, cv2.LINE_AA)
    cv2.addWeighted(ov2, 0.42, img, 0.58, 0, img)
    cv2.line(img, ip_ls, ip_lw, BRIGHT_GREEN, 3, cv2.LINE_AA)
    cv2.circle(img, ip_ls, NODE_R, BRIGHT_GREEN, -1)
    cv2.circle(img, ip_lw, NODE_R, BRIGHT_GREEN, -1)
    cv2.circle(img, ip_ge, 5,      BRIGHT_GREEN, -1)   # elbow node: small dot

def draw_red_arm(img):
    cv2.line(img, ip(ls), ip(le), PURE_RED, 4, cv2.LINE_AA)
    cv2.line(img, ip(le), ip(lw), PURE_RED, 4, cv2.LINE_AA)
    for p in [ls, le, lw]:
        cv2.circle(img, ip(p), NODE_R, PURE_RED, -1)

def draw_muted(img):
    for a,b in [(rs,re),(re,rw),(lh,rh),(ls,rs),(ls,lh),(rs,rh),(lw,rw)]:
        if a is not None and b is not None:
            cv2.line(img, ip(a), ip(b), MUTED, 2)
    for p in [rs,re,rw,lh,rh]:
        if p is not None: cv2.circle(img, ip(p), 4, MUTED, -1)

def draw_chevron(img, tip, fwd):
    perp = np.array([-fwd[1], fwd[0]])
    root = tip - fwd * 13
    cv2.line(img, ip(tip), ip(root + perp*13), BRIGHT_GREEN, 4, cv2.LINE_AA)
    cv2.line(img, ip(tip), ip(root - perp*13), BRIGHT_GREEN, 4, cv2.LINE_AA)

N_CHEV = 3; GAP = 18

def draw_series(img, anchor):
    """Draw chevrons only if they fit within travel range (don't overshoot)."""
    for i in range(N_CHEV):
        tip = anchor + travel_dir * i * GAP
        # clip: don't draw past travel_end
        overshoot = np.dot(tip - travel_end, travel_dir)
        if overshoot > 0:
            break
        draw_chevron(img, tip, travel_dir)

# bake static base
base_raw = extract_frame(DISPLAY_FR)
assert base_raw is not None
base = base_raw.copy()
draw_muted(base)
draw_red_arm(base)
draw_green_glow(base)

# ── timeline: single-direction, disappear at node, reappear at start ─────────
def smoothstep(t): return t*t*(3-2*t)

PUSH        = 14   # frames to travel from start→end
PAUSE_END   = 3    # brief flash at end before disappear
INVISIBLE   = 2    # frames with no arrow (disappeared)
PAUSE_START = 4    # pause after reappear at start

def make_timeline():
    """Returns list of (e, visible) where e=0.0(start)..1.0(end)."""
    tl = []
    # push
    for i in range(PUSH):
        tl.append((smoothstep(i/(PUSH-1)), True))
    # pause at end (fully arrived)
    for _ in range(PAUSE_END):
        tl.append((1.0, True))
    # disappear
    for _ in range(INVISIBLE):
        tl.append((0.0, False))
    # reappear at start + brief pause
    for _ in range(PAUSE_START):
        tl.append((0.0, True))
    return tl

one_cycle = make_timeline()

# duplicate for loop (GIF loop=0 handles infinite, but duplicate gives
# smoother perceived loop by ensuring GIF encoder sees repeated frames)
timeline = one_cycle   # single cycle; loop=0 repeats forever

gif_frames = []; durations = []
for (e, visible) in timeline:
    img    = base.copy()
    anchor = travel_start + e * travel_vec
    if visible:
        draw_series(img, anchor)
    gif_frames.append(PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    # timing
    if not visible:
        dur = 80    # brief blank
    elif e == 0.0:
        dur = 220   # pause at start (reappear)
    elif e == 1.0:
        dur = 180   # pause at end (about to vanish)
    else:
        dur = 55    # motion frames
    durations.append(dur)

gif_frames[0].save(str(OUT), save_all=True,
                   append_images=gif_frames[1:],
                   loop=0, duration=durations, optimize=False)
print(f"=> {OUT}")
print(f"cycle frames: {len(timeline)}  total dur: {sum(durations)}ms/cycle")
