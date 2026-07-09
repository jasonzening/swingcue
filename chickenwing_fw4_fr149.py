"""
CUE-CHICKENWING-001 — fr149 展示帧, 黄箭头改从肘外侧内戳
静态版先出, 确认后加动画
"""

import json, math, cv2, numpy as np
from pathlib import Path

ROOT  = Path("/home/jason/projects/swingcue-postest")
CACHE = ROOT / "engine/kp_cache/batch2/fo-wrong-4.json"
VID   = Path("/mnt/c/Users/jason/Zening/Swingcue/Video/fo-wrong-4.mp4")
OUT_S = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/cw001_fw4_fr149_static.jpg")
OUT_A = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/cw001_fw4_fr149_anim.gif")
OUT_S.parent.mkdir(parents=True, exist_ok=True)

DISPLAY_FR = 149

with open(CACHE) as f:
    frames = json.load(f)['frames']

def get_kp(fi):
    fr = frames[fi]
    if not fr['persons']: return {}
    return {k: (v['x'], v['y'], v['score'])
            for k, v in fr['persons'][0]['keypoints'].items()}

def pt(kp, name):
    return np.array(kp[name][:2]) if name in kp else None

def extract_frame(fi):
    cap = cv2.VideoCapture(str(VID))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

kp = get_kp(DISPLAY_FR)
ls = pt(kp, 'left_shoulder')
le = pt(kp, 'left_elbow')
lw = pt(kp, 'left_wrist')
rs = pt(kp, 'right_shoulder')
re = pt(kp, 'right_elbow')
rw = pt(kp, 'right_wrist')
lh = pt(kp, 'left_hip')
rh = pt(kp, 'right_hip')

print(f"fr{DISPLAY_FR}: ls={tuple(ls.astype(int))} le={tuple(le.astype(int))} lw={tuple(lw.astype(int))}")

# ── geometry: arrow from elbow OUTSIDE, pointing inward toward green line ────
# green line direction: ls → lw
sw_vec  = lw - ls
sw_norm = sw_vec / (np.linalg.norm(sw_vec) + 1e-9)

# perpendicular to green line (pointing away from body = "outside" direction)
# In image coords, perpendicular candidates: rotate sw_norm by ±90°
perp_a = np.array([-sw_norm[1],  sw_norm[0]])   # rotated +90
perp_b = np.array([ sw_norm[1], -sw_norm[0]])   # rotated -90

# projection of elbow onto green line
t      = np.dot(le - ls, sw_vec) / (np.dot(sw_vec, sw_vec) + 1e-9)
t      = float(np.clip(t, 0.05, 0.95))
proj   = ls + t * sw_vec          # closest point on green line to elbow

# the "outside" direction = from proj toward elbow
outward = le - proj
outward_norm = outward / (np.linalg.norm(outward) + 1e-9)

# arrow: start 40px outside elbow along outward_norm, end = proj (on green line)
ARROW_STANDOFF = 40    # how far outside the elbow the arrow starts
arrow_start = le + outward_norm * ARROW_STANDOFF
arrow_end   = proj     # points to green line

print(f"outward_norm={outward_norm}  arrow_start={tuple(arrow_start.astype(int))}  arrow_end={tuple(arrow_end.astype(int))}")
print(f"elbow offset from green line: {np.linalg.norm(outward):.1f}px")

RED    = (40,  60, 230)
GREEN  = (40, 200,  60)
YELLOW = ( 0, 210, 255)
MUTED  = (60,  60,  60)

def render_frame(base_img, arrow_alpha=1.0):
    """Render v3 indicator on a copy of base_img.
    arrow_alpha: 0.0~1.0 for animation pulsing
    """
    img = base_img.copy()

    def ip(a): return (int(round(a[0])), int(round(a[1])))

    # muted skeleton
    for a,b in [(rs,re),(re,rw),(lh,rh),(ls,rs),(ls,lh),(rs,rh),(lw,rw)]:
        if a is not None and b is not None:
            cv2.line(img, ip(a), ip(b), MUTED, 2)
    for p in [rs,re,rw,lh,rh]:
        if p is not None: cv2.circle(img, ip(p), 4, MUTED, -1)

    # red: shoulder → elbow → wrist
    cv2.line(img, ip(ls), ip(le), RED, 5)
    cv2.line(img, ip(le), ip(lw), RED, 5)
    for p, r in [(ls,8),(le,11),(lw,8)]:
        cv2.circle(img, ip(p), r, RED, -1)

    # green: shoulder → wrist direct (same endpoints)
    cv2.line(img, ip(ls), ip(lw), GREEN, 5)
    cv2.circle(img, ip(ls), 8, GREEN, -1)
    cv2.circle(img, ip(lw), 8, GREEN, -1)

    # yellow arrow: outside elbow → green line (inward poke)
    # animate: interpolate arrow_start from full-standoff to closer
    anim_start = le + outward_norm * (ARROW_STANDOFF * arrow_alpha)
    y_col = tuple(int(c * max(0.4, arrow_alpha)) for c in YELLOW)
    thickness = max(2, int(4 * arrow_alpha))
    cv2.arrowedLine(img, ip(anim_start), ip(arrow_end),
                    YELLOW, 4, tipLength=0.25)

    # small circle at arrow_start to mark "outside" position
    cv2.circle(img, ip(arrow_start), 6, YELLOW, 2)

    return img

# ── static output ─────────────────────────────────────────────────────────────
base = extract_frame(DISPLAY_FR)
assert base is not None

static = render_frame(base, arrow_alpha=1.0)
cv2.imwrite(str(OUT_S), static)
print(f"\nStatic => {OUT_S}")

# ── animated GIF: arrow pulses inward (standoff 40→0→40, loop) ───────────────
# Use cv2 + PIL for GIF
try:
    from PIL import Image as PILImage

    gif_frames = []
    N_FRAMES   = 20   # frames per cycle
    PAUSE      = 6    # hold at fully-in position

    for i in range(N_FRAMES + PAUSE):
        if i < N_FRAMES:
            # ease in: standoff goes from 40 down to 0
            t_norm = i / (N_FRAMES - 1)
            # ease-out curve: fast start, slow end
            ease   = 1.0 - t_norm**2
            alpha  = ease
        else:
            alpha = 0.0   # fully in, hold

        # interpolate arrow start position
        anim_start = le + outward_norm * (ARROW_STANDOFF * alpha)
        frame_img  = base.copy()

        def ip(a): return (int(round(a[0])), int(round(a[1])))

        # muted skeleton
        for a,b in [(rs,re),(re,rw),(lh,rh),(ls,rs),(ls,lh),(rs,rh),(lw,rw)]:
            if a is not None and b is not None:
                cv2.line(frame_img, ip(a), ip(b), MUTED, 2)

        # red arm
        cv2.line(frame_img, ip(ls), ip(le), RED, 5)
        cv2.line(frame_img, ip(le), ip(lw), RED, 5)
        for p,r in [(ls,8),(le,11),(lw,8)]:
            cv2.circle(frame_img, ip(p), r, RED, -1)

        # green line
        cv2.line(frame_img, ip(ls), ip(lw), GREEN, 5)
        cv2.circle(frame_img, ip(ls), 8, GREEN, -1)
        cv2.circle(frame_img, ip(lw), 8, GREEN, -1)

        # yellow arrow (animated position)
        cv2.arrowedLine(frame_img, ip(anim_start), ip(arrow_end),
                        YELLOW, 4, tipLength=0.28)
        # anchor dot at outer position
        cv2.circle(frame_img, ip(arrow_start), 7, YELLOW, 2)

        # convert BGR → RGB for PIL
        rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        gif_frames.append(PILImage.fromarray(rgb))

    # durations: 60ms per frame, 300ms hold at end
    durations = [60] * N_FRAMES + [300] * PAUSE
    gif_frames[0].save(
        str(OUT_A),
        save_all=True,
        append_images=gif_frames[1:],
        loop=0,
        duration=durations,
        optimize=False
    )
    print(f"Anim GIF => {OUT_A}")

except ImportError:
    print("PIL not available — skipping GIF, static only")
