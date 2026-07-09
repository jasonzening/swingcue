"""
CUE-CHICKENWING-001 — v10
精确色: RED BGR=(17,15,228), GREEN BGR=(12,220,48)
标准barb箭头: 4顶点内凹菱形 (concave quad, 带尾部V形凹口)
2个串联, 单向推进消失重现循环
"""

import json, cv2, numpy as np
from pathlib import Path
from PIL import Image as PILImage

ROOT  = Path("/home/jason/projects/swingcue-postest")
CACHE = ROOT / "engine/kp_cache/batch2/fo-wrong-4.json"
VID   = Path("/mnt/c/Users/jason/Zening/Swingcue/Video/fo-wrong-4.mp4")
OUT   = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/cw001_v10_final.gif")
OUT.parent.mkdir(parents=True, exist_ok=True)

DISPLAY_FR = 149
STANDOFF   = 70

# ── sampled colors from Jason's reference images ──────────────────────────────
REF_RED   = (17,  15, 228)   # BGR — RGB(228,15,17)  pure red
REF_GREEN = (12, 220,  48)   # BGR — RGB(48,220,12)  fluorescent green
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

print(f"RED   BGR={REF_RED}   RGB={REF_RED[::-1]}")
print(f"GREEN BGR={REF_GREEN} RGB={REF_GREEN[::-1]}")
print(f"red elbow:    {tuple(le.astype(int))}")
print(f"travel_start: {tuple(travel_start.astype(int))}  (outside elbow by {STANDOFF}px)")
print(f"travel_end:   {tuple(travel_end.astype(int))}  (green node)")
print(f"travel_len:   {np.linalg.norm(travel_vec):.1f}px")

def ip(a): return (int(round(a[0])), int(round(a[1])))

# ── standard barb arrow (4-vertex concave quad) ───────────────────────────────
def draw_barb(img, tip, fwd, width=14, length=22, notch=0.38, color=REF_GREEN):
    """
    Standard barb/dart arrow.
    tip    = leading point
    fwd    = unit travel direction
    width  = half-width at widest point
    length = total arrow length (tip to tail ends)
    notch  = notch depth as fraction of length (0.38 = 38% from tail)
    """
    perp = np.array([-fwd[1], fwd[0]])
    tail_center = tip - fwd * length           # midpoint of tail span
    notch_pt    = tip - fwd * (length * (1-notch))  # concave notch

    # 4 vertices: tip, top-tail, notch, bottom-tail
    p_tip  = tip
    p_top  = tail_center + perp * width
    p_notch= notch_pt
    p_bot  = tail_center - perp * width

    pts = np.array([ip(p_tip), ip(p_top), ip(p_notch), ip(p_bot)], dtype=np.int32)
    cv2.fillPoly(img, [pts], color)
    # crisp anti-aliased outline
    cv2.polylines(img, [pts], True, color, 1, cv2.LINE_AA)

N_ARROWS  = 2
ARROW_GAP = 28   # tip-to-tip spacing

def draw_series(img, lead_tip):
    for i in range(N_ARROWS):
        tip = lead_tip - travel_dir * i * ARROW_GAP
        # don't draw if behind start
        if np.dot(tip - travel_start, travel_dir) < -(ARROW_GAP * 0.5):
            break
        # don't draw if past end
        if np.dot(tip - travel_end, travel_dir) > 0:
            continue
        draw_barb(img, tip, travel_dir)

# ── green glow line ───────────────────────────────────────────────────────────
def draw_green_glow(img):
    ip_ls, ip_lw, ip_ge = ip(ls), ip(lw), ip(green_elbow)
    ov1 = img.copy()
    cv2.line(ov1, ip_ls, ip_lw, (8, 160, 35), 10, cv2.LINE_AA)
    cv2.addWeighted(ov1, 0.28, img, 0.72, 0, img)
    ov2 = img.copy()
    cv2.line(ov2, ip_ls, ip_lw, (10, 190, 40), 6, cv2.LINE_AA)
    cv2.addWeighted(ov2, 0.42, img, 0.58, 0, img)
    cv2.line(img, ip_ls, ip_lw, REF_GREEN, 3, cv2.LINE_AA)
    cv2.circle(img, ip_ls, NODE_R, REF_GREEN, -1)
    cv2.circle(img, ip_lw, NODE_R, REF_GREEN, -1)
    cv2.circle(img, ip_ge, 5, REF_GREEN, -1)

def draw_red_arm(img):
    cv2.line(img, ip(ls), ip(le), REF_RED, 4, cv2.LINE_AA)
    cv2.line(img, ip(le), ip(lw), REF_RED, 4, cv2.LINE_AA)
    for p in [ls, le, lw]:
        cv2.circle(img, ip(p), NODE_R, REF_RED, -1)

def draw_muted(img):
    for a,b in [(rs,re),(re,rw),(lh,rh),(ls,rs),(ls,lh),(rs,rh),(lw,rw)]:
        if a is not None and b is not None:
            cv2.line(img, ip(a), ip(b), MUTED, 2)
    for p in [rs,re,rw,lh,rh]:
        if p is not None: cv2.circle(img, ip(p), 4, MUTED, -1)

# bake base
base_raw = extract_frame(DISPLAY_FR)
assert base_raw is not None
base = base_raw.copy()
draw_muted(base)
draw_red_arm(base)
draw_green_glow(base)

# ── animation ─────────────────────────────────────────────────────────────────
def smoothstep(t): return t*t*(3-2*t)

PUSH=16; PAUSE_END=3; INVISIBLE=2; PAUSE_START=5

def make_timeline():
    tl = []
    for i in range(PUSH):        tl.append((smoothstep(i/(PUSH-1)), True))
    for _ in range(PAUSE_END):   tl.append((1.0, True))
    for _ in range(INVISIBLE):   tl.append((0.0, False))
    for _ in range(PAUSE_START): tl.append((0.0, True))
    return tl

timeline = make_timeline()
gif_frames = []; durations = []

for (e, visible) in timeline:
    img      = base.copy()
    lead_tip = travel_start + e * travel_vec
    if visible:
        draw_series(img, lead_tip)
    gif_frames.append(PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    if not visible: dur = 80
    elif e >= 1.0:  dur = 200
    elif e == 0.0:  dur = 240
    else:           dur = 55
    durations.append(dur)

gif_frames[0].save(str(OUT), save_all=True,
                   append_images=gif_frames[1:],
                   loop=0, duration=durations, optimize=False)
print(f"=> {OUT}")
print(f"cycle: {len(timeline)} frames / {sum(durations)}ms")
