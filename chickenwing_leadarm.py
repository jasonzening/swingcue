"""
CUE-CHICKENWING-001 — 正确判据重算
Lead arm (左臂) + impact→follow-through 窗口
A = 左肩-左肘-左腕角度 (伸展度)
B1 = 两肘间距 / 肩宽
B2 = 左肘到躯干侧线距离 (左肘相对左肩的横向偏移)
"""

import json, math, cv2, numpy as np
from pathlib import Path

ROOT        = Path("/home/jason/projects/swingcue-postest")
COACH_CACHE = ROOT / "engine/kp_cache/ghost004/coach-fo.json"
USER_CACHE  = ROOT / "engine/kp_cache/batch2/fo-ok-1.json"
COACH_VID   = Path("/mnt/c/Users/jason/Zening/Swingcue/教学视频/coach-video/coach-fo.mp4")
USER_VID    = ROOT / "input/fo-ok-1.mp4"
OUT         = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/chickenwing_leadarm.jpg")
OUT.parent.mkdir(parents=True, exist_ok=True)

# correct windows: impact → follow-through
# coach: impact=fr052, follow_through start=fr056, go to fr065
# user:  impact=fr088, follow fr097, go to fr100
COACH_WIN = list(range(52, 68))
USER_WIN  = list(range(88, 103))

def get_kp(frames, fi):
    fr = frames[fi]
    if not fr['persons']: return {}
    return {k: (v['x'], v['y'], v['score'])
            for k, v in fr['persons'][0]['keypoints'].items()}

def pt(kp, name):
    return np.array(kp[name][:2]) if name in kp else None

def angle_deg(a, b, c):
    ba = a - b; bc = c - b
    n = np.linalg.norm(ba) * np.linalg.norm(bc)
    if n < 1e-9: return 0.0
    return math.degrees(math.acos(np.clip(np.dot(ba, bc) / n, -1, 1)))

def compute_lead_metrics(kp):
    ls = pt(kp, 'left_shoulder')
    le = pt(kp, 'left_elbow')
    lw = pt(kp, 'left_wrist')
    rs = pt(kp, 'right_shoulder')
    re = pt(kp, 'right_elbow')

    res = {}

    # A: lead arm extension angle (left shoulder – left elbow – left wrist)
    # 180° = fully straight; chicken wing = angle drops (elbow bends/collapses)
    if ls is not None and le is not None and lw is not None:
        res['A_lead_arm_angle'] = angle_deg(ls, le, lw)

    # B1: inter-elbow distance / shoulder width (normalized)
    # correct = elbows close together; chicken wing = large gap
    if le is not None and re is not None and ls is not None and rs is not None:
        sw = np.linalg.norm(ls - rs)
        ew = np.linalg.norm(le - re)
        res['B1_elbow_gap_norm'] = ew / (sw + 1e-9)

    # B2: lead elbow lateral distance from left shoulder vertical
    # positive = elbow flew out LEFT of left shoulder (away from body)
    # negative = elbow is right of left shoulder (tucked toward body)
    if le is not None and ls is not None:
        res['B2_lead_elbow_vs_shoulder'] = le[0] - ls[0]  # + = elbow further left

    # B3: lead elbow vertical drop below left shoulder
    # chicken wing often collapses DOWN as well as out
    if le is not None and ls is not None:
        res['B3_lead_elbow_drop'] = le[1] - ls[1]  # + = elbow below shoulder (normal)

    return res

# load
with open(COACH_CACHE) as f: coach_frames = json.load(f)['frames']
with open(USER_CACHE)  as f: user_frames  = json.load(f)['frames']

# tables
print("=" * 78)
print("COACH  impact(fr052) → follow-through")
print("=" * 78)
print(f"{'fr':>5}  {'A_angle':>8}  {'B1_gap':>8}  {'B2_lat':>8}  {'B3_drop':>8}  note")
coach_table = []
for fi in COACH_WIN:
    kp = get_kp(coach_frames, fi)
    m  = compute_lead_metrics(kp)
    note = ""
    if fi == 52: note = "<< impact"
    if fi == 56: note = "<< follow start"
    print(f"  {fi:03d}  {m.get('A_lead_arm_angle',0):8.1f}  "
          f"{m.get('B1_elbow_gap_norm',0):8.3f}  "
          f"{m.get('B2_lead_elbow_vs_shoulder',0):+8.1f}  "
          f"{m.get('B3_lead_elbow_drop',0):+8.1f}  {note}")
    coach_table.append((fi, m))

print()
print("=" * 78)
print("USER   impact(fr088) → follow-through")
print("=" * 78)
print(f"{'fr':>5}  {'A_angle':>8}  {'B1_gap':>8}  {'B2_lat':>8}  {'B3_drop':>8}  note")
user_table = []
for fi in USER_WIN:
    kp = get_kp(user_frames, fi)
    m  = compute_lead_metrics(kp)
    note = ""
    if fi == 88: note = "<< impact"
    if fi == 97: note = "<< follow (Jason GT)"
    print(f"  {fi:03d}  {m.get('A_lead_arm_angle',0):8.1f}  "
          f"{m.get('B1_elbow_gap_norm',0):8.3f}  "
          f"{m.get('B2_lead_elbow_vs_shoulder',0):+8.1f}  "
          f"{m.get('B3_lead_elbow_drop',0):+8.1f}  {note}")
    user_table.append((fi, m))

# anchor comparison
print()
print("=" * 78)
print("ANCHOR SUMMARY")
print("=" * 78)
anchors = [
    ("COACH impact  fr052", coach_frames, 52),
    ("COACH follow  fr056", coach_frames, 56),
    ("COACH follow  fr060", coach_frames, 60),
    ("USER  impact  fr088", user_frames,  88),
    ("USER  follow  fr092", user_frames,  92),
    ("USER  follow  fr097", user_frames,  97),
]
for label, frames, fi in anchors:
    kp = get_kp(frames, fi)
    m  = compute_lead_metrics(kp)
    print(f"  {label}: A={m.get('A_lead_arm_angle',0):.1f}deg  "
          f"B1={m.get('B1_elbow_gap_norm',0):.3f}  "
          f"B2={m.get('B2_lead_elbow_vs_shoulder',0):+.1f}px")

# multi-frame skeleton image
def extract_frame(vid, fi):
    cap = cv2.VideoCapture(str(vid))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

# 6 key frames: coach impact/+4/+8  user impact/+4/+9
KEY_FRAMES = [
    ("COACH fr052\nimpact",         COACH_VID, coach_frames, 52, (60,220,100)),
    ("COACH fr056\nfollow start",   COACH_VID, coach_frames, 56, (60,220,100)),
    ("COACH fr060\nfollow +8fr",    COACH_VID, coach_frames, 60, (60,220,100)),
    ("USER fr088\nimpact",          USER_VID,  user_frames,  88, (80,140,255)),
    ("USER fr092\nfollow +4fr",     USER_VID,  user_frames,  92, (80,140,255)),
    ("USER fr097\nfollow (GT)",     USER_VID,  user_frames,  97, (80,140,255)),
]

TARGET_H = 580
panels = []

for label, vid, frames, fi, color in KEY_FRAMES:
    img = extract_frame(vid, fi)
    if img is None:
        img = np.full((TARGET_H, 380, 3), 30, dtype=np.uint8)
        cv2.putText(img, f"no frame {fi}", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,200), 2)
        panels.append(img)
        continue

    kp = get_kp(frames, fi)
    m  = compute_lead_metrics(kp)

    def ip(name):
        p = pt(kp, name)
        return None if p is None else (int(p[0]), int(p[1]))

    ls = ip('left_shoulder');  le = ip('left_elbow');  lw = ip('left_wrist')
    rs = ip('right_shoulder'); re = ip('right_elbow');  rw = ip('right_wrist')
    lh = ip('left_hip');       rh = ip('right_hip')

    # muted: trail arm + hips
    MUTED = (90, 90, 90)
    for a,b in [(rs,re),(re,rw),(lh,rh),(ls,rs),(ls,lh),(rs,rh)]:
        if a and b: cv2.line(img, a, b, MUTED, 2)
    for p in [rs,re,rw,lh,rh]:
        if p: cv2.circle(img, p, 5, MUTED, -1)

    # highlight lead arm (left) thick
    LEAD = (0, 230, 255)
    for a,b in [(ls,le),(le,lw)]:
        if a and b: cv2.line(img, a, b, LEAD, 4)
    for p,r in [(ls,8),(le,12),(lw,8)]:
        if p: cv2.circle(img, p, r, LEAD, -1)

    # A: draw angle arc at left elbow
    if ls is not None and le is not None and lw is not None:
        A = m.get('A_lead_arm_angle', 0)
        # arc annotation
        cv2.putText(img, f"A={A:.0f}deg", (le[0]+14, le[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,230,255), 2)

    # B2: lateral guide — vertical line through left shoulder
    if ls is not None and le is not None:
        guide_top = (ls[0], ls[1] - 60)
        guide_bot = (ls[0], ls[1] + 140)
        cv2.line(img, guide_top, guide_bot, (200,200,50), 1)
        # horizontal arrow: shoulder x → elbow x (at elbow y)
        proj = (ls[0], le[1])
        B2 = m.get('B2_lead_elbow_vs_shoulder', 0)
        b2_col = (0,80,255) if B2 > 20 else (60,220,60)
        cv2.arrowedLine(img, proj, le, b2_col, 2, tipLength=0.3)
        cv2.putText(img, f"B2={B2:+.0f}px", (min(proj[0],le[0])-10, le[1]-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, b2_col, 2)

    # B1: inter-elbow line
    if le is not None and re is not None:
        cv2.line(img, le, re, (180,100,255), 1)
        mid_e = ((le[0]+re[0])//2, (le[1]+re[1])//2)
        B1 = m.get('B1_elbow_gap_norm', 0)
        cv2.putText(img, f"B1={B1:.2f}", (mid_e[0]-25, mid_e[1]+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180,100,255), 1)

    # panel label
    for i, line in enumerate(label.split('\n')):
        cv2.putText(img, line, (12, 38+i*26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)

    # metrics bottom strip
    h_img = img.shape[0]
    metrics = [
        f"A  lead arm angle: {m.get('A_lead_arm_angle',0):.1f} deg",
        f"B1 elbow gap/sw:   {m.get('B1_elbow_gap_norm',0):.3f}",
        f"B2 l_elbow-l_sh:   {m.get('B2_lead_elbow_vs_shoulder',0):+.1f} px",
        f"B3 elbow drop:     {m.get('B3_lead_elbow_drop',0):+.1f} px",
    ]
    for i, txt in enumerate(metrics):
        cv2.putText(img, txt, (12, h_img - 88 + i*21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (190,190,190), 1)

    scale = TARGET_H / img.shape[0]
    w_new = int(img.shape[1] * scale)
    img = cv2.resize(img, (w_new, TARGET_H))
    panels.append(img)

sep = np.full((TARGET_H, 6, 3), 40, dtype=np.uint8)
combined = panels[0]
for p in panels[1:]:
    combined = np.hstack([combined, sep, p])

# legend
leg_h = 210
leg = np.full((leg_h, combined.shape[1], 3), 12, dtype=np.uint8)
lines = [
    "LEAD ARM (left arm) METRICS  —  impact → follow-through window",
    "",
    "  A   left shoulder - left elbow - left wrist angle  |  180=fully straight  |  chicken wing = drops sharply",
    "  B1  inter-elbow distance / shoulder width           |  correct = small gap  |  chicken wing = large gap",
    "  B2  left elbow x - left shoulder x (px)            |  + = elbow flew LEFT past shoulder = flying out",
    "  B3  left elbow y - left shoulder y (px)            |  large = elbow dropped low (collapse)",
    "",
    "  Cyan = lead arm (left)   |  Yellow guide = left shoulder vertical   |  Purple line = inter-elbow span",
    "  Green arrow = elbow tucked   |   Red arrow = elbow flaring out",
    "",
    "  Jason: in the USER follow-through frames — is the left elbow bending and flying out?",
    "         Which metric (A angle dropping / B1 gap growing / B2 flaring) matches what you see?",
]
for i, line in enumerate(lines):
    c = (100,230,255) if i == 0 else (155,155,155)
    if line.startswith("  A ") or line.startswith("  B"): c = (200,200,100)
    if "Jason" in line: c = (220,200,60)
    cv2.putText(leg, line, (18, 24+i*17), cv2.FONT_HERSHEY_SIMPLEX, 0.46, c, 1)

final = np.vstack([combined, leg])
cv2.imwrite(str(OUT), final)
print(f"\n=> {OUT}")
