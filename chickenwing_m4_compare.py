"""
CUE-CHICKENWING-001 — 补充对比图
user fr085 (M4=+45.8px) vs coach fr049 (M4=+28.7px)
清晰标注 M4 的度量方式：右肩-右腕轴线 + 右肘垂直偏距
"""

import json, math, cv2, numpy as np
from pathlib import Path

ROOT        = Path("/home/jason/projects/swingcue-postest")
COACH_CACHE = ROOT / "engine/kp_cache/ghost004/coach-fo.json"
USER_CACHE  = ROOT / "engine/kp_cache/batch2/fo-ok-1.json"
COACH_VID   = Path("/mnt/c/Users/jason/Zening/Swingcue/教学视频/coach-video/coach-fo.mp4")
USER_VID    = ROOT / "input/fo-ok-1.mp4"
OUT         = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/chickenwing_m4_compare.jpg")
OUT.parent.mkdir(parents=True, exist_ok=True)

# --- target frames ---
COACH_FR = 49   # impact-3, M4=+28.7px
USER_FR  = 85   # argmax M4=+45.8px

# --- helpers ---
def get_kp(frames, fi):
    fr = frames[fi]
    if not fr['persons']: return {}
    return {k: (v['x'], v['y'], v['score'])
            for k,v in fr['persons'][0]['keypoints'].items()}

def pt(kp, name):
    return np.array(kp[name][:2]) if name in kp else None

def extract_frame(vid, fi):
    cap = cv2.VideoCapture(str(vid))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

# --- load ---
with open(COACH_CACHE) as f: coach_frames = json.load(f)['frames']
with open(USER_CACHE)  as f: user_frames  = json.load(f)['frames']

coach_kp = get_kp(coach_frames, COACH_FR)
user_kp  = get_kp(user_frames,  USER_FR)

# --- render one panel ---
def render_panel(img, kp, label, m4_val, flare_color):
    """
    Draw:
      - full upper-body skeleton (muted)
      - RIGHT arm highlighted (trail arm)
      - Right shoulder → right wrist AXIS LINE (white dashed)
      - Right elbow PROJECTION point on that line
      - Perpendicular distance line (flare_color)
      - M4 measurement label
    """
    def ip(name):
        p = pt(kp, name)
        return None if p is None else (int(p[0]), int(p[1]))
    def np_(name):
        return pt(kp, name)

    ls = ip('left_shoulder');  le = ip('left_elbow');  lw = ip('left_wrist')
    rs = ip('right_shoulder'); re = ip('right_elbow'); rw = ip('right_wrist')
    lh = ip('left_hip');       rh = ip('right_hip')

    # muted skeleton: lead arm + hips
    MUTED = (120, 120, 120)
    for a,b in [(ls,le),(le,lw),(lh,rh),(ls,rs),(ls,lh),(rs,rh)]:
        if a and b: cv2.line(img, a, b, MUTED, 2)
    for p in [ls,le,lw,lh,rh]:
        if p: cv2.circle(img, p, 5, MUTED, -1)

    # highlighted trail arm (right)
    TRAIL = (60, 200, 255)
    for a,b in [(rs,re),(re,rw)]:
        if a and b: cv2.line(img, a, b, TRAIL, 3)
    for p,r in [(rs,7),(re,10),(rw,7)]:
        if p: cv2.circle(img, p, r, TRAIL, -1)

    # === M4 measurement visualization ===
    rs_np = np_('right_shoulder')
    re_np = np_('right_elbow')
    rw_np = np_('right_wrist')

    if rs_np is not None and re_np is not None and rw_np is not None:
        # axis line: right shoulder → right wrist (extended slightly)
        ab  = rw_np - rs_np
        ab_n = ab / (np.linalg.norm(ab) + 1e-9)
        # extend ±40px for visibility
        ax_start = rs_np - ab_n * 30
        ax_end   = rw_np + ab_n * 30

        # draw dashed axis line
        total_len = np.linalg.norm(ax_end - ax_start)
        dash = 12; num_dashes = int(total_len / dash / 2)
        for i in range(num_dashes + 1):
            t0 = (2*i*dash) / total_len
            t1 = min((2*i+1)*dash / total_len, 1.0)
            p0 = (ax_start + t0*(ax_end - ax_start)).astype(int)
            p1 = (ax_start + t1*(ax_end - ax_start)).astype(int)
            cv2.line(img, tuple(p0), tuple(p1), (255, 255, 255), 1)

        # projection of elbow onto axis
        t = np.dot(re_np - rs_np, ab) / (np.dot(ab, ab) + 1e-9)
        proj = rs_np + t * ab
        proj_pt = tuple(proj.astype(int))

        # perpendicular distance line (the M4 measure)
        re_pt = tuple(re_np.astype(int))
        cv2.line(img, proj_pt, re_pt, flare_color, 2)

        # small tick at projection point
        perp = np.array([-ab_n[1], ab_n[0]])  # perpendicular to axis
        tick_a = proj + perp * 8
        tick_b = proj - perp * 8
        cv2.line(img, tuple(tick_a.astype(int)), tuple(tick_b.astype(int)), (255,255,255), 2)

        # arrow from proj → elbow (show direction of flare)
        cv2.arrowedLine(img, proj_pt, re_pt, flare_color, 2, tipLength=0.25)

        # M4 label next to elbow
        mid = ((proj + re_np) / 2).astype(int)
        sign = "+" if m4_val > 0 else ""
        m4_txt = f"M4={sign}{m4_val:.1f}px"
        offset = (-120, -20) if re_np[0] > proj[0] else (10, -20)
        cv2.putText(img, m4_txt, (mid[0]+offset[0], mid[1]+offset[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, flare_color, 2)

        # axis label
        ax_mid = ((ax_start + ax_end)/2).astype(int)
        cv2.putText(img, "right shoulder-wrist axis", (ax_mid[0]-80, ax_mid[1]-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

    # panel label
    cv2.putText(img, label, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

    # interpretation text
    verdict = "ELBOW FLARE (chicken wing)" if m4_val > 0 else "ELBOW TUCKED (correct)"
    verdict_color = (0, 80, 255) if m4_val > 20 else (60, 255, 60)
    cv2.putText(img, verdict, (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, verdict_color, 2)

    return img

# extract frames
coach_img = extract_frame(COACH_VID, COACH_FR)
user_img  = extract_frame(USER_VID,  USER_FR)

if coach_img is None or user_img is None:
    print("ERROR: frame extraction failed")
    exit(1)

# compute actual M4 from kp for labeling
def calc_m4(kp):
    rs = pt(kp, 'right_shoulder')
    re = pt(kp, 'right_elbow')
    rw = pt(kp, 'right_wrist')
    if rs is None or re is None or rw is None: return 0.0
    ab = rw - rs; ap = re - rs
    cross = ab[0]*ap[1] - ab[1]*ap[0]
    return -cross / (np.linalg.norm(ab) + 1e-9)

coach_m4 = calc_m4(coach_kp)
user_m4  = calc_m4(user_kp)
print(f"coach fr{COACH_FR:03d} M4 = {coach_m4:.1f}px")
print(f"user  fr{USER_FR:03d}  M4 = {user_m4:.1f}px")

# render panels
render_panel(coach_img, coach_kp,
             f"COACH fr{COACH_FR:03d}  (downswing -3fr from impact)",
             coach_m4, (60, 255, 60))    # green = OK

render_panel(user_img, user_kp,
             f"USER  fr{USER_FR:03d}  (argmax M4 -- worst frame)",
             user_m4, (0, 80, 255))       # red-ish = bad

# match heights
h = max(coach_img.shape[0], user_img.shape[0])
def pad_h(img, H):
    dh = H - img.shape[0]
    if dh <= 0: return img
    return np.pad(img, ((0,dh),(0,0),(0,0)), constant_values=20)

ci = pad_h(coach_img, h)
ui = pad_h(user_img,  h)

# side-by-side with separator
sep  = np.full((h, 8, 3), 50, dtype=np.uint8)
combined = np.hstack([ci, sep, ui])

# bottom legend
legend_h = 180
legend = np.full((legend_h, combined.shape[1], 3), 15, dtype=np.uint8)
lines = [
    "M4 DEFINITION:  right elbow perpendicular distance from the right_shoulder → right_wrist axis",
    "  positive (+) = elbow flares OUTWARD  (away from body)  = chicken wing",
    "  negative (-) = elbow tucked INWARD   (toward body)     = correct pro form",
    "",
    f"  COACH fr{COACH_FR:03d}:  M4 = {coach_m4:+.1f}px   (slight flare at downswing entry, normal in transition)",
    f"  USER  fr{USER_FR:03d}:   M4 = {user_m4:+.1f}px  << argmax, maximum chicken wing",
    "",
    "  Jason: is the right elbow visibly flying out in user fr085?",
    "         What threshold should trigger the chicken-wing flag?  (candidate: M4 > 30px or > 35px)",
]
for i, line in enumerate(lines):
    color = (100, 220, 255) if i == 0 else (170, 170, 170)
    if "COACH" in line: color = (60, 220, 100)
    if "USER"  in line: color = (80, 140, 255)
    if "Jason" in line: color = (220, 200, 80)
    cv2.putText(legend, line, (20, 28+i*19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

final = np.vstack([combined, legend])
cv2.imwrite(str(OUT), final)
print(f"\n=> {OUT}")
