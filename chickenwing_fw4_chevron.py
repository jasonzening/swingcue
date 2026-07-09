"""
CUE-CHICKENWING-001 — fr149 黄色雪佛龙串动画
整串3个 chevron, 黄色加粗, 从肘外侧整体平移推向绿线, loop
"""

import json, math, cv2, numpy as np
from pathlib import Path
from PIL import Image as PILImage

ROOT  = Path("/home/jason/projects/swingcue-postest")
CACHE = ROOT / "engine/kp_cache/batch2/fo-wrong-4.json"
VID   = Path("/mnt/c/Users/jason/Zening/Swingcue/Video/fo-wrong-4.mp4")
OUT   = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/cw001_fw4_chevron.gif")
OUT.parent.mkdir(parents=True, exist_ok=True)

DISPLAY_FR = 149

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
ls = pt(kp, 'left_shoulder'); le = pt(kp, 'left_elbow')
lw = pt(kp, 'left_wrist');    rs = pt(kp, 'right_shoulder')
re = pt(kp, 'right_elbow');   rw = pt(kp, 'right_wrist')
lh = pt(kp, 'left_hip');      rh = pt(kp, 'right_hip')

# ── geometry (same as previous) ───────────────────────────────────────────────
sw_vec = lw - ls
t = float(np.clip(np.dot(le - ls, sw_vec) / (np.dot(sw_vec, sw_vec) + 1e-9), 0.05, 0.95))
proj = ls + t * sw_vec                          # target on green line

outward = le - proj
outward_norm = outward / (np.linalg.norm(outward) + 1e-9)

# travel: from 50px outside elbow → proj (on green line)
STANDOFF     = 50.0
travel_start = le + outward_norm * STANDOFF     # series origin (outside)
travel_end   = proj                             # series destination (green line)
travel_vec   = travel_end - travel_start        # full translation vector
travel_len   = np.linalg.norm(travel_vec)
travel_dir   = travel_vec / (travel_len + 1e-9) # unit vector toward green line

print(f"travel: {tuple(travel_start.astype(int))} -> {tuple(travel_end.astype(int))}")
print(f"travel_len={travel_len:.1f}px  dir={travel_dir}")

# ── draw one chevron (V shape) ─────────────────────────────────────────────────
def draw_chevron(img, tip, fwd, size=14, thickness=4, color=(0, 210, 255)):
    """
    Draw one chevron (>) at position `tip`, pointing in direction `fwd`.
    `size`      = half-width of the V
    `thickness` = line thickness
    """
    # perpendicular to fwd
    perp = np.array([-fwd[1], fwd[0]])
    # two arms of the V: tip is the leading point
    # arm back-left:  tip - fwd*size + perp*size
    # arm back-right: tip - fwd*size - perp*size
    arm_root = tip - fwd * size
    p1 = arm_root + perp * size
    p2 = arm_root - perp * size

    def ip(a): return (int(round(a[0])), int(round(a[1])))
    cv2.line(img, ip(tip), ip(p1), color, thickness, cv2.LINE_AA)
    cv2.line(img, ip(tip), ip(p2), color, thickness, cv2.LINE_AA)

# ── draw chevron series at a given offset along travel path ───────────────────
N_CHEVRONS   = 3      # number of chevrons in series
CHEVRON_GAP  = 18     # spacing between chevrons (along travel_dir)
CHEVRON_SIZE = 13     # arm length
THICKNESS    = 5      # line thickness —加粗

def draw_series(img, series_anchor):
    """
    Draw N_CHEVRONS chevrons starting at series_anchor,
    spaced CHEVRON_GAP apart along travel_dir.
    """
    for i in range(N_CHEVRONS):
        tip = series_anchor + travel_dir * (i * CHEVRON_GAP)
        draw_chevron(img, tip, travel_dir,
                     size=CHEVRON_SIZE,
                     thickness=THICKNESS,
                     color=(0, 210, 255))   # YELLOW (BGR)

# ── base frame (skeleton without chevrons) ────────────────────────────────────
RED   = (40,  60, 230)
GREEN = (40, 200,  60)
MUTED = (60,  60,  60)

base_raw = extract_frame(DISPLAY_FR)
assert base_raw is not None

def draw_base(img):
    def ip(a): return (int(round(a[0])), int(round(a[1])))
    for a,b in [(rs,re),(re,rw),(lh,rh),(ls,rs),(ls,lh),(rs,rh),(lw,rw)]:
        if a is not None and b is not None:
            cv2.line(img, ip(a), ip(b), MUTED, 2)
    for p in [rs,re,rw,lh,rh]:
        if p is not None: cv2.circle(img, ip(p), 4, MUTED, -1)

    # red: shoulder→elbow→wrist
    cv2.line(img, ip(ls), ip(le), RED, 5)
    cv2.line(img, ip(le), ip(lw), RED, 5)
    for p,r in [(ls,8),(le,11),(lw,8)]: cv2.circle(img, ip(p), r, RED, -1)

    # green: shoulder→wrist
    cv2.line(img, ip(ls), ip(lw), GREEN, 5)
    cv2.circle(img, ip(ls), 8, GREEN, -1)
    cv2.circle(img, ip(lw), 8, GREEN, -1)

# ── animate: series translates from travel_start toward travel_end ────────────
# The anchor of the series (tip of first chevron) moves along travel_vec.
# Start: anchor = travel_start
# End:   anchor = travel_end  (series has arrived at green line)
# Then loop back instantly.

N_ANIM    = 24    # animation frames per cycle
PAUSE_END = 6     # hold at destination

gif_frames = []
durations  = []

for i in range(N_ANIM + PAUSE_END):
    img = base_raw.copy()
    draw_base(img)

    if i < N_ANIM:
        # ease-in-out: series starts slow, speeds up, slows at end
        t_norm = i / (N_ANIM - 1)
        ease   = t_norm * t_norm * (3 - 2 * t_norm)   # smoothstep
        anchor = travel_start + ease * travel_vec
        dur    = 55
    else:
        # hold at destination
        anchor = travel_end.copy()
        dur    = 280

    draw_series(img, anchor)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gif_frames.append(PILImage.fromarray(rgb))
    durations.append(dur)

gif_frames[0].save(
    str(OUT),
    save_all=True,
    append_images=gif_frames[1:],
    loop=0,
    duration=durations,
    optimize=False
)
print(f"=> {OUT}")
