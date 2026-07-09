"""
CUE-CHICKENWING-001 — fr149 动画修正3
- 终点精确停在绿线 (proj), 不越过
- 起点更靠外 (standoff=90px)
- 推两次循环: push→pause→retreat→pause→push→... 停在起点
- 绿线加亮发光 (glow)
- 版本A: 黄色雪佛龙  版本B: 绿色雪佛龙
"""

import json, math, cv2, numpy as np
from pathlib import Path
from PIL import Image as PILImage

ROOT  = Path("/home/jason/projects/swingcue-postest")
CACHE = ROOT / "engine/kp_cache/batch2/fo-wrong-4.json"
VID   = Path("/mnt/c/Users/jason/Zening/Swingcue/Video/fo-wrong-4.mp4")
OUT_A = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/cw001_v4A_yellow.gif")
OUT_B = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/cw001_v4B_green.gif")
OUT_A.parent.mkdir(parents=True, exist_ok=True)

DISPLAY_FR = 149
STANDOFF   = 90     # further out than before

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
t = float(np.clip(np.dot(le-ls, sw_vec)/(np.dot(sw_vec,sw_vec)+1e-9), 0.05, 0.95))
proj = ls + t * sw_vec                      # target = point on green line

outward = le - proj
outward_norm = outward / (np.linalg.norm(outward)+1e-9)

# travel: start further out, end exactly at proj (green line)
travel_start = le + outward_norm * STANDOFF  # further out
travel_end   = proj                          # green line — STOP HERE
travel_vec   = travel_end - travel_start
travel_len   = np.linalg.norm(travel_vec)
travel_dir   = travel_vec / (travel_len+1e-9)

print(f"travel: {tuple(travel_start.astype(int))} -> {tuple(travel_end.astype(int))}")
print(f"travel_len={travel_len:.1f}px")

# ── glow green line ───────────────────────────────────────────────────────────
BRIGHT_GREEN = (50, 255, 80)   # very bright, saturated (BGR)

def draw_green_glow(img, p1, p2):
    """Draw green line with glow: wide dim → medium → bright core."""
    ip1 = (int(round(p1[0])), int(round(p1[1])))
    ip2 = (int(round(p2[0])), int(round(p2[1])))
    # outer glow (widest, blended)
    overlay = img.copy()
    cv2.line(overlay, ip1, ip2, (30, 180, 50), 18, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    # mid glow
    overlay2 = img.copy()
    cv2.line(overlay2, ip1, ip2, (40, 230, 65), 10, cv2.LINE_AA)
    cv2.addWeighted(overlay2, 0.50, img, 0.50, 0, img)
    # bright core
    cv2.line(img, ip1, ip2, BRIGHT_GREEN, 5, cv2.LINE_AA)
    # endpoints
    cv2.circle(img, ip1, 9, BRIGHT_GREEN, -1)
    cv2.circle(img, ip2, 9, BRIGHT_GREEN, -1)

# ── chevron draw ──────────────────────────────────────────────────────────────
def draw_chevron(img, tip, fwd, size=14, thickness=5, color=(0,210,255)):
    perp = np.array([-fwd[1], fwd[0]])
    arm_root = tip - fwd * size
    p1 = arm_root + perp * size
    p2 = arm_root - perp * size
    def ip(a): return (int(round(a[0])), int(round(a[1])))
    cv2.line(img, ip(tip), ip(p1), color, thickness, cv2.LINE_AA)
    cv2.line(img, ip(tip), ip(p2), color, thickness, cv2.LINE_AA)

N_CHEVRONS  = 3
CHEVRON_GAP = 20

def draw_series(img, anchor, color):
    for i in range(N_CHEVRONS):
        tip = anchor + travel_dir * (i * CHEVRON_GAP)
        draw_chevron(img, tip, travel_dir, size=14, thickness=5, color=color)

# ── base layer (skeleton without green line or chevrons) ──────────────────────
RED   = (40,  60, 230)
MUTED = (60,  60,  60)

base_raw = extract_frame(DISPLAY_FR)
assert base_raw is not None

def build_base(img):
    """Draw muted skeleton + red arm. No green, no chevrons."""
    def ip(a): return (int(round(a[0])), int(round(a[1])))
    for a,b in [(rs,re),(re,rw),(lh,rh),(ls,rs),(ls,lh),(rs,rh),(lw,rw)]:
        if a is not None and b is not None:
            cv2.line(img, ip(a), ip(b), MUTED, 2)
    for p in [rs,re,rw,lh,rh]:
        if p is not None: cv2.circle(img, ip(p), 4, MUTED, -1)
    cv2.line(img, ip(ls), ip(le), RED, 5)
    cv2.line(img, ip(le), ip(lw), RED, 5)
    for p,r in [(ls,8),(le,11),(lw,8)]: cv2.circle(img, ip(p), r, RED, -1)

# pre-bake base + green (these don't change per frame)
base_with_green = base_raw.copy()
build_base(base_with_green)
draw_green_glow(base_with_green, ls, lw)

# ── animation timeline (push×2, stop at start) ───────────────────────────────
# One cycle = push → pause_end → retreat → pause_start → push → pause_end → retreat → pause_start(long)
# smooth_step for easing
def smoothstep(t): return t*t*(3-2*t)

# build per-frame anchor positions
PUSH_FRAMES    = 14   # frames to travel start→end
PAUSE_END      = 5    # hold at green line
RETREAT_FRAMES = 10   # frames to travel end→start
PAUSE_START    = 4    # brief pause at start between pushes
PAUSE_START_LONG = 10 # final rest at start before loop

def make_timeline():
    timeline = []
    for cycle in range(2):
        # push
        for i in range(PUSH_FRAMES):
            e = smoothstep(i/(PUSH_FRAMES-1))
            timeline.append(('push', e))
        # pause at end
        for _ in range(PAUSE_END):
            timeline.append(('push', 1.0))
        # retreat
        for i in range(RETREAT_FRAMES):
            e = smoothstep(i/(RETREAT_FRAMES-1))
            timeline.append(('retreat', e))
        # pause at start
        pause = PAUSE_START_LONG if cycle==1 else PAUSE_START
        for _ in range(pause):
            timeline.append(('retreat', 1.0))
    return timeline

timeline = make_timeline()

def anchor_from_state(state, e):
    if state == 'push':
        return travel_start + e * travel_vec
    else:  # retreat
        return travel_end - e * travel_vec   # e=1.0 means fully back at start

# ── render both versions ──────────────────────────────────────────────────────
def render_gif(out_path, chevron_color):
    gif_frames = []
    durations  = []
    for (state, e) in timeline:
        img    = base_with_green.copy()
        anchor = anchor_from_state(state, e)
        draw_series(img, anchor, chevron_color)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gif_frames.append(PILImage.fromarray(rgb))
        # faster during motion, slower during pauses
        if state == 'push' and e > 0 and e < 1:
            dur = 50
        elif state == 'retreat' and e > 0 and e < 1:
            dur = 45
        else:
            dur = 220
        durations.append(dur)

    gif_frames[0].save(
        str(out_path),
        save_all=True,
        append_images=gif_frames[1:],
        loop=0,
        duration=durations,
        optimize=False,
    )
    print(f"=> {out_path}")

YELLOW_COL     = (0, 210, 255)   # BGR yellow
GREEN_CHEV_COL = (50, 255, 80)   # BGR bright green (same as glow core)

render_gif(OUT_A, YELLOW_COL)
render_gif(OUT_B, GREEN_CHEV_COL)
print("done")
