"""
CUE-CHICKENWING-001 — 肘部折叠方向判据
主判据: lead 肘相对 lead 肩的运动方向 (B2横向 + B3垂直)
正确 = 肘向下 + 收体前 (B3增大, B2减小或维持)
鸡翅膀 = 肘上抬 + 外飞 (B3减小/反转, B2持续增大)
"""

import json, math, cv2, numpy as np
from pathlib import Path

ROOT       = Path("/home/jason/projects/swingcue-postest")
USER_CACHE = ROOT / "engine/kp_cache/batch2/fo-ok-1.json"
USER_VID   = ROOT / "input/fo-ok-1.mp4"
COACH_CACHE = ROOT / "engine/kp_cache/ghost004/coach-fo.json"
COACH_VID  = Path("/mnt/c/Users/jason/Zening/Swingcue/教学视频/coach-video/coach-fo.mp4")
OUT        = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/chickenwing_direction.jpg")
OUT.parent.mkdir(parents=True, exist_ok=True)

USER_WIN  = list(range(88, 103))   # impact→follow
COACH_WIN = list(range(52, 68))

def get_kp(frames, fi):
    fr = frames[fi]
    if not fr['persons']: return {}
    return {k: (v['x'], v['y'], v['score'])
            for k, v in fr['persons'][0]['keypoints'].items()}

def pt(kp, name):
    return np.array(kp[name][:2]) if name in kp else None

def lead_elbow_rel(kp):
    """Returns (B2, B3) = lead elbow position relative to lead shoulder.
    B2: elbow_x - shoulder_x  (+ = elbow flew left/outward for right-handed golfer)
    B3: elbow_y - shoulder_y  (+ = elbow below shoulder = correct down-fold)
    """
    ls = pt(kp, 'left_shoulder')
    le = pt(kp, 'left_elbow')
    rs = pt(kp, 'right_shoulder')
    if ls is None or le is None:
        return None, None, None
    sw = np.linalg.norm(ls - rs) if rs is not None else 1.0
    b2 = le[0] - ls[0]          # lateral
    b3 = le[1] - ls[1]          # vertical (+ = down in image)
    # normalized by shoulder width
    b2n = b2 / sw
    b3n = b3 / sw
    return b2, b3, (b2n, b3n)

with open(USER_CACHE)  as f: user_frames  = json.load(f)['frames']
with open(COACH_CACHE) as f: coach_frames = json.load(f)['frames']

# ── compute trajectory ────────────────────────────────────────────────────────
print("=" * 72)
print("USER lead elbow trajectory (relative to lead shoulder)")
print("impact fr088 → follow-through")
print("=" * 72)
print(f"  {'fr':>4}  {'B2_lat':>8}  {'B3_vert':>8}  {'dB2':>7}  {'dB3':>7}  direction  verdict")

user_traj = []
for fi in USER_WIN:
    kp = get_kp(user_frames, fi)
    b2, b3, _ = lead_elbow_rel(kp)
    user_traj.append((fi, b2, b3))

prev_b2, prev_b3 = user_traj[0][1], user_traj[0][2]
for i, (fi, b2, b3) in enumerate(user_traj):
    if b2 is None: continue
    db2 = b2 - prev_b2 if i > 0 else 0.0
    db3 = b3 - prev_b3 if i > 0 else 0.0

    # direction classification per frame
    # elbow moving: right(db2<0)=tucking / left(db2>0)=flying
    #               down(db3>0)=correct  / up(db3<0)=bad
    lat_dir  = "OUT" if db2 > 2 else ("IN" if db2 < -2 else "~")
    vert_dir = "DOWN" if db3 > 2 else ("UP" if db3 < -2 else "~")
    direction = f"{vert_dir}+{lat_dir}"

    # verdict: chicken wing = moving UP and/or OUT simultaneously
    cw_score = 0
    if db3 < -3: cw_score += 1   # elbow rising
    if db2 > 3:  cw_score += 1   # elbow flying out
    verdict = "!! CW" if cw_score == 2 else ("? UP" if db3 < -3 else ("? OUT" if db2 > 3 else "ok"))

    note = ""
    if fi == 88: note = "  << impact"
    if fi == 97: note = "  << follow GT"
    print(f"  {fi:4d}  {b2:+8.1f}  {b3:+8.1f}  {db2:+7.1f}  {db3:+7.1f}  {direction:12s}  {verdict}{note}")
    prev_b2, prev_b3 = b2, b3

print()
print("=" * 72)
print("COACH lead elbow trajectory (reference only — not used as baseline)")
print("=" * 72)
print(f"  {'fr':>4}  {'B2_lat':>8}  {'B3_vert':>8}  {'dB2':>7}  {'dB3':>7}  direction")
coach_traj = []
for fi in COACH_WIN:
    kp = get_kp(coach_frames, fi)
    b2, b3, _ = lead_elbow_rel(kp)
    coach_traj.append((fi, b2, b3))

prev_b2, prev_b3 = coach_traj[0][1], coach_traj[0][2]
for i, (fi, b2, b3) in enumerate(coach_traj):
    if b2 is None: continue
    db2 = b2 - prev_b2 if i > 0 else 0.0
    db3 = b3 - prev_b3 if i > 0 else 0.0
    lat_dir  = "OUT" if db2 > 2 else ("IN" if db2 < -2 else "~")
    vert_dir = "DOWN" if db3 > 2 else ("UP" if db3 < -2 else "~")
    note = ""
    if fi == 52: note = "  << impact"
    if fi == 56: note = "  << follow start"
    print(f"  {fi:4d}  {b2:+8.1f}  {b3:+8.1f}  {db2:+7.1f}  {db3:+7.1f}  {vert_dir}+{lat_dir}{note}")
    prev_b2, prev_b3 = b2, b3

# ── visualization ─────────────────────────────────────────────────────────────
def extract_frame(vid, fi):
    cap = cv2.VideoCapture(str(vid))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

# pick 5 user frames spanning impact→follow: fr088,091,094,097,100
KEY_USER = [88, 91, 94, 97, 100]
TARGET_H = 620

panels = []
for fi in KEY_USER:
    img = extract_frame(USER_VID, fi)
    if img is None:
        img = np.full((TARGET_H, 360, 3), 30, dtype=np.uint8)
        panels.append(img)
        continue

    kp = get_kp(user_frames, fi)
    b2, b3, _ = lead_elbow_rel(kp)

    def ip(name):
        p = pt(kp, name)
        return None if p is None else (int(p[0]), int(p[1]))

    ls = ip('left_shoulder');  le = ip('left_elbow');  lw = ip('left_wrist')
    rs = ip('right_shoulder'); re = ip('right_elbow');  rw = ip('right_wrist')
    lh = ip('left_hip');       rh = ip('right_hip')

    # muted background skeleton
    MUTED = (80, 80, 80)
    for a,b_ in [(rs,re),(re,rw),(lh,rh),(ls,rs),(ls,lh),(rs,rh),(lw,rw)]:
        if a and b_: cv2.line(img, a, b_, MUTED, 2)

    # lead arm — bright
    LEAD = (0, 230, 255)
    for a,b_ in [(ls,le),(le,lw)]:
        if a and b_: cv2.line(img, a, b_, LEAD, 4)
    for p,r in [(ls,8),(le,13),(lw,7)]:
        if p: cv2.circle(img, p, r, LEAD, -1)

    # elbow direction arrow from shoulder position
    if ls is not None and le is not None and b2 is not None:
        # draw the position vector: from shoulder to elbow
        cv2.arrowedLine(img, ls, le, (255, 180, 0), 2, tipLength=0.15)

        # classify direction for THIS frame (absolute position, not delta)
        # correct = b3 large positive (elbow well below shoulder) + b2 moderate
        # chicken wing = b3 small or negative (elbow near shoulder height or above)
        down_ok   = b3 > 40     # elbow at least 40px below shoulder
        out_bad   = b2 > 60     # elbow >60px laterally past shoulder

        if b3 < 20 and b2 > 50:
            verdict_txt = "UP+OUT (CW)"
            verdict_col = (0, 60, 255)
        elif b3 < 20:
            verdict_txt = "ELBOW UP"
            verdict_col = (0, 130, 255)
        elif b2 > 60:
            verdict_txt = "ELBOW OUT"
            verdict_col = (0, 180, 255)
        else:
            verdict_txt = "DOWN+IN (ok)"
            verdict_col = (60, 220, 60)

        # B3 vertical guide
        guide = (ls[0], le[1])
        cv2.line(img, ls, (ls[0], le[1]), (200,200,50), 1)   # vertical
        cv2.line(img, (ls[0], le[1]), le, (200,200,50), 1)   # horizontal
        cv2.putText(img, f"B3={b3:+.0f}", (ls[0]+6, (ls[1]+le[1])//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,50), 1)
        cv2.putText(img, f"B2={b2:+.0f}", ((ls[0]+le[0])//2, le[1]+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,50), 1)

        # verdict box
        cv2.putText(img, verdict_txt, (12, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, verdict_col, 2)

    phase = "impact" if fi == 88 else ("follow GT" if fi == 97 else f"+{fi-88}fr")
    cv2.putText(img, f"USER fr{fi:03d}  {phase}", (12, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255,255,255), 2)

    scale = TARGET_H / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1]*scale), TARGET_H))
    panels.append(img)

sep = np.full((TARGET_H, 6, 3), 35, dtype=np.uint8)
combined = panels[0]
for p in panels[1:]:
    combined = np.hstack([combined, sep, p])

# bottom: trajectory plot as a simple 2D chart (B2 vs B3 over time)
CHART_H = 260
CHART_W = combined.shape[1]
chart = np.full((CHART_H, CHART_W, 3), 15, dtype=np.uint8)

# draw axes
CX, CY = 80, 200   # origin in chart (B2=0, B3=0 reference)
SCALE_B2 = 1.8     # px per unit
SCALE_B3 = 1.8

cv2.line(chart, (CX, 20), (CX, CHART_H-20), (80,80,80), 1)     # B3 axis (vertical)
cv2.line(chart, (20, CY), (CHART_W-20, CY), (80,80,80), 1)     # B2 axis (horizontal)
cv2.putText(chart, "B3 (elbow vertical relative to shoulder)", (CX+8, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150,150,150), 1)
cv2.putText(chart, "B2 ->", (CHART_W-80, CY-6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150,150,150), 1)
cv2.putText(chart, "DOWN=correct", (CX+8, CY+18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60,200,60), 1)
cv2.putText(chart, "UP=bad", (CX+8, CY-6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,80,255), 1)
cv2.putText(chart, "OUT=bad->", (CHART_W//2, CY-6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,80,255), 1)

# draw quadrant labels
cv2.putText(chart, "DOWN+IN", (CX-75, CY+30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60,180,60), 1)
cv2.putText(chart, "CORRECT", (CX-75, CY+46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60,180,60), 1)
cv2.putText(chart, "CW ZONE", (CHART_W//2+20, CY-30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,60,220), 1)

# plot user trajectory
user_pts = [(b2, b3) for (fi, b2, b3) in user_traj if b2 is not None]
for i, (b2, b3) in enumerate(user_pts):
    x = int(CX + b2 * SCALE_B2)
    y = int(CY - b3 * SCALE_B3)   # invert: larger B3 = lower in chart
    x = np.clip(x, 5, CHART_W-5)
    y = np.clip(y, 5, CHART_H-5)
    fi = USER_WIN[i]
    col = (0, 200, 255)
    cv2.circle(chart, (x, y), 5, col, -1)
    if fi in [88, 91, 94, 97, 100]:
        cv2.putText(chart, f"u{fi}", (x+4, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1)
    if i > 0:
        px2, pb3 = user_pts[i-1]
        px_ = int(np.clip(CX + px2*SCALE_B2, 5, CHART_W-5))
        py_ = int(np.clip(CY - pb3*SCALE_B3, 5, CHART_H-5))
        cv2.arrowedLine(chart, (px_, py_), (x, y), col, 1, tipLength=0.4)

# plot coach trajectory (muted)
coach_pts = [(b2, b3) for (fi, b2, b3) in coach_traj if b2 is not None]
for i, (b2, b3) in enumerate(coach_pts):
    x = int(CX + b2 * SCALE_B2)
    y = int(CY - b3 * SCALE_B3)
    x = np.clip(x, 5, CHART_W-5)
    y = np.clip(y, 5, CHART_H-5)
    cv2.circle(chart, (x, y), 3, (60,180,60), -1)
    if i > 0:
        pb2, pb3 = coach_pts[i-1]
        px_ = int(np.clip(CX + pb2*SCALE_B2, 5, CHART_W-5))
        py_ = int(np.clip(CY - pb3*SCALE_B3, 5, CHART_H-5))
        cv2.line(chart, (px_, py_), (x, y), (40,120,40), 1)

# legend
cv2.putText(chart, "Cyan=USER  Green=COACH(ref only)", (CHART_W//2-100, CHART_H-12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150,150,150), 1)

# text panel right of chart
INFO_X = CX + int(max(b2 for b2,_ in user_pts)*SCALE_B2) + 80
INFO_X = max(INFO_X, CHART_W//2 + 80)
info_lines = [
    "FOLD DIRECTION STANDARD (HackMotion/TPI):",
    "  correct = elbow folds DOWN + stays IN",
    "  chicken = elbow goes UP + OUT",
    "",
    "B3 = elbow_y - shoulder_y",
    "  LARGE+ = elbow well below shoulder (correct)",
    "  near 0 or - = elbow near/above shoulder (bad)",
    "",
    "B2 = elbow_x - shoulder_x",
    "  moderate = normal swing arc",
    "  large+ sustained = elbow flying out (bad)",
]
for i, line in enumerate(info_lines):
    col = (100,230,255) if i==0 else (155,155,155)
    if "correct" in line.lower(): col = (60,200,60)
    if "chicken" in line.lower() or "bad" in line.lower(): col = (80,130,255)
    if i < CHART_H // 17:
        cv2.putText(chart, line, (INFO_X, 22+i*17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

final = np.vstack([combined, chart])
cv2.imwrite(str(OUT), final)
print(f"\n=> {OUT}")
