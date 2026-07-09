"""
CUE-CHICKENWING-001 — v11
修正:
1. 颜色全程恒定: 每帧从 base_raw 重建(不预烘焙), glow改实色叠加; GIF强制调色板含目标色
2. 红色 = REF_RED 精确值, 全帧恒定
3. 箭头起点 STANDOFF=35px (v10 70px 再减一半)
4. 停顿减半: PAUSE_START 5帧→2帧, INVISIBLE 2帧→1帧
"""

import json, cv2, numpy as np
from pathlib import Path
from PIL import Image as PILImage, ImagePalette

ROOT  = Path("/home/jason/projects/swingcue-postest")
CACHE = ROOT / "engine/kp_cache/batch2/fo-wrong-4.json"
VID   = Path("/mnt/c/Users/jason/Zening/Swingcue/Video/fo-wrong-4.mp4")
OUT   = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/cw001_v11_final.gif")
OUT.parent.mkdir(parents=True, exist_ok=True)

DISPLAY_FR = 149
STANDOFF   = 35     # half of v10's 70px — "1/8 arrow body" gap

# sampled from Jason's reference images
REF_RED   = (17,  15, 228)   # BGR — RGB(228,15,17)
REF_GREEN = (12, 220,  48)   # BGR — RGB(48,220,12)
MUTED     = (60,  60,  60)
NODE_R    = 6

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

ARROW_LENGTH = 22   # barb arrow body length
STD_GAP_PX   = STANDOFF - ARROW_LENGTH   # gap between arrow tail and elbow when at start
STD_GAP_RATIO = (STANDOFF - ARROW_LENGTH) / ARROW_LENGTH

print(f"REF_RED   BGR={REF_RED}   RGB={REF_RED[::-1]}")
print(f"REF_GREEN BGR={REF_GREEN} RGB={REF_GREEN[::-1]}")
print(f"STANDOFF={STANDOFF}px  arrow_length={ARROW_LENGTH}px")
print(f"gap(arrow_tail to elbow) = {STD_GAP_PX:.1f}px = {STD_GAP_RATIO:.2f}x arrow_length")
print(f"travel_len: {np.linalg.norm(travel_vec):.1f}px")

def ip(a): return (int(round(a[0])), int(round(a[1])))

# ── draw layers (rebuilt every frame from base_raw) ───────────────────────────
def draw_muted(img):
    for a,b in [(rs,re),(re,rw),(lh,rh),(ls,rs),(ls,lh),(rs,rh),(lw,rw)]:
        if a is not None and b is not None:
            cv2.line(img, ip(a), ip(b), MUTED, 2)
    for p in [rs,re,rw,lh,rh]:
        if p is not None: cv2.circle(img, ip(p), 4, MUTED, -1)

def draw_red_arm(img):
    """Red arm — REF_RED solid, rebuilt every frame."""
    cv2.line(img, ip(ls), ip(le), REF_RED, 4, cv2.LINE_AA)
    cv2.line(img, ip(le), ip(lw), REF_RED, 4, cv2.LINE_AA)
    for p in [ls, le, lw]:
        cv2.circle(img, ip(p), NODE_R, REF_RED, -1)

def draw_green_line(img):
    """
    Green glow line — core is REF_GREEN solid.
    Glow added as a semi-transparent wider line UNDERNEATH, then solid core on top.
    Both rebuilt every frame to ensure color consistency.
    """
    ip_ls, ip_lw, ip_ge = ip(ls), ip(lw), ip(green_elbow)

    # glow: wide dim line first
    ov = img.copy()
    cv2.line(ov, ip_ls, ip_lw, (8, 140, 30), 12, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.25, img, 0.75, 0, img)

    # solid core — REF_GREEN, always the same
    cv2.line(img, ip_ls, ip_lw, REF_GREEN, 4, cv2.LINE_AA)

    # nodes — all solid REF_GREEN
    cv2.circle(img, ip_ls, NODE_R, REF_GREEN, -1)
    cv2.circle(img, ip_lw, NODE_R, REF_GREEN, -1)
    cv2.circle(img, ip_ge, 5, REF_GREEN, -1)  # elbow node: small dot

# ── standard barb arrow ───────────────────────────────────────────────────────
def draw_barb(img, tip, fwd, width=14, length=ARROW_LENGTH, notch=0.38, color=REF_GREEN):
    perp     = np.array([-fwd[1], fwd[0]])
    tail_c   = tip - fwd * length
    notch_pt = tip - fwd * (length * (1-notch))
    pts      = np.array([ip(tip), ip(tail_c+perp*width),
                         ip(notch_pt), ip(tail_c-perp*width)], dtype=np.int32)
    cv2.fillPoly(img, [pts], color)
    cv2.polylines(img, [pts], True, color, 1, cv2.LINE_AA)

N_ARROWS  = 2
ARROW_GAP = 28

def draw_series(img, lead_tip):
    for i in range(N_ARROWS):
        tip = lead_tip - travel_dir * i * ARROW_GAP
        if np.dot(tip - travel_start, travel_dir) < -(ARROW_GAP*0.5): break
        if np.dot(tip - travel_end,   travel_dir) > 0: continue
        draw_barb(img, tip, travel_dir)

# ── force key colors into GIF palette every frame ────────────────────────────
def stamp_palette(img):
    """Stamp 3x3 pixels of REF_RED and REF_GREEN in bottom-right corner.
    Ensures GIF palette always includes these colors.
    Placed out of the way (bottom-right 10px margin)."""
    h, w = img.shape[:2]
    # REF_GREEN patch
    img[h-10:h-7, w-10:w-7] = REF_GREEN
    # REF_RED patch
    img[h-10:h-7, w-7:w-4]  = REF_RED

# ── animation ─────────────────────────────────────────────────────────────────
def smoothstep(t): return t*t*(3-2*t)

PUSH        = 16
PAUSE_END   = 2   # halved from 3 (was 3→2)
INVISIBLE   = 1   # halved from 2 (was 2→1)
PAUSE_START = 2   # halved from 5 (was 5→2)

def make_timeline():
    tl = []
    for i in range(PUSH):        tl.append((smoothstep(i/(PUSH-1)), True))
    for _ in range(PAUSE_END):   tl.append((1.0, True))
    for _ in range(INVISIBLE):   tl.append((0.0, False))
    for _ in range(PAUSE_START): tl.append((0.0, True))
    return tl

timeline  = make_timeline()
base_raw  = extract_frame(DISPLAY_FR)
assert base_raw is not None

gif_frames = []; durations = []

for (e, visible) in timeline:
    # rebuild every frame from base_raw — guarantees consistent colors
    img = base_raw.copy()
    draw_muted(img)
    draw_red_arm(img)
    draw_green_line(img)
    if visible:
        lead_tip = travel_start + e * travel_vec
        draw_series(img, lead_tip)
    stamp_palette(img)

    gif_frames.append(PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    if not visible: dur = 60
    elif e >= 1.0:  dur = 180
    elif e == 0.0:  dur = 120   # halved: was 240
    else:           dur = 55
    durations.append(dur)

gif_frames[0].save(str(OUT), save_all=True,
                   append_images=gif_frames[1:],
                   loop=0, duration=durations, optimize=False)
print(f"=> {OUT}")
print(f"cycle: {len(timeline)} frames / {sum(durations)}ms")
print(f"\nSTANDARD ARROW SPACING:")
print(f"  STANDOFF={STANDOFF}px from elbow to arrow TIP")
print(f"  Arrow body length={ARROW_LENGTH}px")
print(f"  Gap (arrow tail to elbow) = {STD_GAP_PX:.0f}px = {STD_GAP_RATIO:.2f}x arrow_length")
