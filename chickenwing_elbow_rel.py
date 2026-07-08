"""
CUE-CHICKENWING-001 — 双肘关系候选量 E1~E5
逐帧计算, 多帧骨架图, 给 Jason 筛选正确判据
"""

import json, math, cv2, numpy as np
from pathlib import Path

ROOT        = Path("/home/jason/projects/swingcue-postest")
COACH_CACHE = ROOT / "engine/kp_cache/ghost004/coach-fo.json"
USER_CACHE  = ROOT / "engine/kp_cache/batch2/fo-ok-1.json"
COACH_VID   = Path("/mnt/c/Users/jason/Zening/Swingcue/教学视频/coach-video/coach-fo.mp4")
USER_VID    = ROOT / "input/fo-ok-1.mp4"
OUT         = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/chickenwing_elbow_rel.jpg")
OUT.parent.mkdir(parents=True, exist_ok=True)

# downswing windows (phase_report + context)
COACH_WIN = list(range(46, 57))  # transition→impact
USER_WIN  = list(range(80, 93))  # user downswing area (impact=88)

def get_kp(frames, fi):
    fr = frames[fi]
    if not fr['persons']: return {}
    return {k: (v['x'], v['y'], v['score'])
            for k, v in fr['persons'][0]['keypoints'].items()}

def pt(kp, name):
    return np.array(kp[name][:2]) if name in kp else None

def angle_deg(a, b, c):
    ba = a - b; bc = c - b
    cos_ = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return math.degrees(math.acos(np.clip(cos_, -1, 1)))

# ─── candidate quantities ────────────────────────────────────────────────────
def compute_elbow_metrics(kp):
    ls = pt(kp, 'left_shoulder');  le = pt(kp, 'left_elbow')
    rs = pt(kp, 'right_shoulder'); re = pt(kp, 'right_elbow')
    lh = pt(kp, 'left_hip');       rh = pt(kp, 'right_hip')
    lw = pt(kp, 'left_wrist');     rw = pt(kp, 'right_wrist')

    res = {}

    # ── E1: inter-elbow distance / shoulder width
    # Chicken wing: elbows fly apart → large ratio
    if le is not None and re is not None and ls is not None and rs is not None:
        sw = np.linalg.norm(ls - rs)
        ew = np.linalg.norm(le - re)
        res['E1_elbow_dist_norm'] = ew / (sw + 1e-9)

    # ── E2: mean lateral outward offset of both elbows from torso midline
    # Torso midline = vertical line through mid-shoulder
    # Positive = elbow is OUTSIDE body (chicken wing)
    # Use signed distance: left elbow to left of center = good (negative),
    # right elbow to right of center = good (positive, but we want |offset| from body)
    if le is not None and re is not None and ls is not None and rs is not None:
        mid_x = (ls[0] + rs[0]) / 2
        # left elbow: chicken wing if it goes right of left shoulder
        l_flare = le[0] - ls[0]   # >0 means elbow crossed inward (toward center)
        # right elbow: chicken wing if it goes left of right shoulder
        r_flare = rs[0] - re[0]   # >0 means elbow crossed inward (toward center)
        # "both elbows tucked" = both positive; "chicken wing" = one/both negative
        res['E2_l_elbow_vs_shoulder_x'] = l_flare    # + = elbow inside shoulder line
        res['E2_r_elbow_vs_shoulder_x'] = r_flare    # + = elbow inside shoulder line
        res['E2_mean_tuck'] = (l_flare + r_flare) / 2  # higher = more tucked

    # ── E3: right elbow lateral distance from torso side boundary
    # Torso right boundary ≈ right shoulder x.
    # Positive = elbow is to the RIGHT of right shoulder (flying out = chicken wing)
    if re is not None and rs is not None:
        res['E3_r_elbow_vs_r_shoulder'] = re[0] - rs[0]  # + = elbow flew out right

    # ── E4: elbow-span / shoulder-span ratio (width comparison)
    # How much wider are the elbows than the shoulders?
    # Compact swing: elbow span ≤ shoulder span
    # Chicken wing: elbows wider than shoulders
    if le is not None and re is not None and ls is not None and rs is not None:
        shoulder_span = abs(ls[0] - rs[0])
        elbow_span    = abs(le[0] - re[0])
        res['E4_elbow_span_ratio'] = elbow_span / (shoulder_span + 1e-9)

    # ── E5: "elbow flare angle" — angle between shoulder-midline vertical
    #        and the line connecting mid-elbow to mid-shoulder
    # More outward tilt of the elbow pair = larger angle = chicken wing
    if le is not None and re is not None and ls is not None and rs is not None:
        mid_shoulder = (ls + rs) / 2
        mid_elbow    = (le + re) / 2
        # vector from mid-shoulder to mid-elbow
        v = mid_elbow - mid_shoulder
        # angle from vertical (downward = 0°)
        vert = np.array([0, 1])  # y increases downward in image
        cos_ = np.dot(v, vert) / (np.linalg.norm(v) * np.linalg.norm(vert) + 1e-9)
        res['E5_elbow_pair_drop_angle'] = math.degrees(math.acos(np.clip(cos_, -1, 1)))
        # lateral offset of mid-elbow from mid-shoulder (normalized by shoulder width)
        sw = np.linalg.norm(ls - rs)
        res['E5_elbow_center_lateral_norm'] = (mid_elbow[0] - mid_shoulder[0]) / (sw + 1e-9)

    return res

# ─── load ────────────────────────────────────────────────────────────────────
with open(COACH_CACHE) as f: coach_frames = json.load(f)['frames']
with open(USER_CACHE)  as f: user_frames  = json.load(f)['frames']

# ─── compute tables ──────────────────────────────────────────────────────────
print("\n" + "="*80)
print("COACH downswing window (fr046~056) — phase: transition→downswing→impact")
print("="*80)
coach_table = []
for fi in COACH_WIN:
    kp = get_kp(coach_frames, fi)
    m  = compute_elbow_metrics(kp)
    coach_table.append((fi, m))
    mark = " << Jason says slight CW" if fi in [49,52] else ""
    print(f"  fr{fi:03d}: E1={m.get('E1_elbow_dist_norm',0):.2f} "
          f"E2_tuck={m.get('E2_mean_tuck',0):+.1f}px "
          f"E3={m.get('E3_r_elbow_vs_r_shoulder',0):+.1f}px "
          f"E4={m.get('E4_elbow_span_ratio',0):.2f} "
          f"E5_lat={m.get('E5_elbow_center_lateral_norm',0):+.2f}{mark}")

print("\n" + "="*80)
print("USER downswing window (fr080~092) — impact=fr088")
print("="*80)
user_table = []
for fi in USER_WIN:
    kp = get_kp(user_frames, fi)
    m  = compute_elbow_metrics(kp)
    user_table.append((fi, m))
    mark = " << old M4 argmax (Jason: natural swing?)" if fi == 85 else ""
    mark = " << impact" if fi == 88 else mark
    print(f"  fr{fi:03d}: E1={m.get('E1_elbow_dist_norm',0):.2f} "
          f"E2_tuck={m.get('E2_mean_tuck',0):+.1f}px "
          f"E3={m.get('E3_r_elbow_vs_r_shoulder',0):+.1f}px "
          f"E4={m.get('E4_elbow_span_ratio',0):.2f} "
          f"E5_lat={m.get('E5_elbow_center_lateral_norm',0):+.2f}{mark}")

# ─── key anchor comparison ───────────────────────────────────────────────────
print("\n" + "="*80)
print("ANCHOR COMPARISON  (Jason visual anchors vs numbers)")
print("="*80)
METRICS = ['E1_elbow_dist_norm','E2_mean_tuck','E3_r_elbow_vs_r_shoulder',
           'E4_elbow_span_ratio','E5_elbow_center_lateral_norm']
anchors = [
    ("COACH fr049", coach_frames, 49, "Jason: slight CW"),
    ("COACH fr052", coach_frames, 52, "Jason: slight CW"),
    ("USER  fr085", user_frames,  85, "Jason: natural swing?"),
    ("USER  fr088", user_frames,  88, "impact anchor"),
]
for label, frames, fi, note in anchors:
    kp = get_kp(frames, fi)
    m  = compute_elbow_metrics(kp)
    print(f"\n  {label}  [{note}]")
    for k in METRICS:
        print(f"    {k}: {m.get(k, 'N/A')}")

# ─── multi-frame skeleton image ──────────────────────────────────────────────
def extract_frame(vid, fi):
    cap = cv2.VideoCapture(str(vid))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

# pick 4 key frames: coach fr049, coach fr052, user fr085, user fr088
KEY_FRAMES = [
    ("COACH fr049\nJason: slight CW", COACH_VID, coach_frames, 49, (60, 220, 100)),
    ("COACH fr052\nJason: slight CW", COACH_VID, coach_frames, 52, (60, 220, 100)),
    ("USER fr085\nM4 argmax (natural?)", USER_VID, user_frames, 85, (80, 140, 255)),
    ("USER fr088\nimpact anchor",       USER_VID, user_frames, 88, (80, 140, 255)),
]

TARGET_H = 640
panels = []

for label, vid, frames, fi, color in KEY_FRAMES:
    img = extract_frame(vid, fi)
    if img is None:
        img = np.full((TARGET_H, 400, 3), 30, dtype=np.uint8)
        cv2.putText(img, f"no frame {fi}", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,200), 2)
        panels.append(img)
        continue

    kp = get_kp(frames, fi)
    m  = compute_elbow_metrics(kp)

    def ip(name):
        p = pt(kp, name)
        return None if p is None else (int(p[0]), int(p[1]))

    ls = ip('left_shoulder');  le = ip('left_elbow');  lw = ip('left_wrist')
    rs = ip('right_shoulder'); re = ip('right_elbow');  rw = ip('right_wrist')
    lh = ip('left_hip');       rh = ip('right_hip')

    # skeleton
    MUTED = (100,100,100)
    for a,b in [(ls,le),(le,lw),(lh,rh),(ls,rs),(ls,lh),(rs,rh),(lw,rw)]:
        if a and b: cv2.line(img, a, b, MUTED, 2)

    # highlight both elbows BIG
    ELBOW_COL = (0, 220, 255)
    if le: cv2.circle(img, le, 14, ELBOW_COL, 3)
    if re: cv2.circle(img, re, 14, ELBOW_COL, 3)
    # connect elbows with a line → shows inter-elbow span
    if le and re:
        cv2.line(img, le, re, ELBOW_COL, 2)
        mid_e = ((le[0]+re[0])//2, (le[1]+re[1])//2)
        cv2.putText(img, f"E1={m.get('E1_elbow_dist_norm',0):.2f}",
                    (mid_e[0]-40, mid_e[1]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, ELBOW_COL, 2)

    # shoulder line
    if ls and rs:
        cv2.line(img, ls, rs, (180,180,180), 2)

    # E3: right elbow vs right shoulder vertical line
    if rs and re:
        # vertical guide through right shoulder
        rs_np = np.array(rs); re_np = np.array(re)
        guide_top = (rs[0], rs[1] - 80)
        guide_bot = (rs[0], rs[1] + 120)
        cv2.line(img, guide_top, guide_bot, (200,200,50), 1)
        # horizontal distance: re.x - rs.x
        proj = (rs[0], re[1])  # same y as elbow, x of shoulder
        e3 = m.get('E3_r_elbow_vs_r_shoulder', 0)
        e3_color = (0,80,255) if e3 > 0 else (60,220,60)
        cv2.arrowedLine(img, proj, re, e3_color, 2, tipLength=0.25)
        cv2.putText(img, f"E3={e3:+.0f}px",
                    (re[0]+8, re[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, e3_color, 2)

    # shoulder joints
    for p in [ls, rs]:
        if p: cv2.circle(img, p, 7, (180,180,180), -1)

    # label top
    for i, line in enumerate(label.split('\n')):
        cv2.putText(img, line, (14, 42+i*28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # metrics bottom
    metrics_txt = [
        f"E1 elbow_dist/sw: {m.get('E1_elbow_dist_norm',0):.2f}",
        f"E2 mean_tuck:     {m.get('E2_mean_tuck',0):+.1f}px",
        f"E3 r_elbow-r_sh:  {m.get('E3_r_elbow_vs_r_shoulder',0):+.1f}px",
        f"E4 elbow/sh span: {m.get('E4_elbow_span_ratio',0):.2f}",
        f"E5 lateral norm:  {m.get('E5_elbow_center_lateral_norm',0):+.2f}",
    ]
    h_img = img.shape[0]
    for i, txt in enumerate(metrics_txt):
        cv2.putText(img, txt, (14, h_img - 110 + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200,200,200), 1)

    # resize to target height
    scale = TARGET_H / img.shape[0]
    w_new = int(img.shape[1] * scale)
    img = cv2.resize(img, (w_new, TARGET_H))
    panels.append(img)

# combine
sep = np.full((TARGET_H, 6, 3), 40, dtype=np.uint8)
combined = panels[0]
for p in panels[1:]:
    combined = np.hstack([combined, sep, p])

# legend
leg_h = 220
leg = np.full((leg_h, combined.shape[1], 3), 12, dtype=np.uint8)
lines = [
    "DUAL-ELBOW CANDIDATE QUANTITIES",
    "",
    "  E1  inter-elbow dist / shoulder width       larger = elbows flying apart",
    "  E2  mean elbow tuck vs shoulder line (px)   LOWER = elbows tucked; HIGHER = elbows inside or neutral",
    "  E3  right elbow x - right shoulder x (px)   POSITIVE = r_elbow flew out past shoulder = chicken wing signal",
    "  E4  elbow horizontal span / shoulder span   > 1.0 = elbows wider than shoulders",
    "  E5  lateral shift of elbow pair center       nonzero = whole arm pair shifted sideways",
    "",
    "  Yellow circle = right elbow  |  Cyan line = inter-elbow span  |  Yellow vertical = r_shoulder guide",
    "  Red/blue arrow (E3) = r_elbow displacement from r_shoulder vertical",
    "",
    "  Jason: which quantity's ranking matches your visual judgment?",
    "         coach fr049/052 should score HIGHER (more CW) than user fr085 if Jason's eye is right.",
]
for i, line in enumerate(lines):
    color = (100, 230, 255) if i == 0 else (160,160,160)
    if "E1" in line[:4] or "E2" in line[:4] or "E3" in line[:4] or "E4" in line[:4] or "E5" in line[:4]:
        color = (200,200,100)
    if "Jason" in line: color = (220,200,60)
    cv2.putText(leg, line, (18, 26+i*16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)

final = np.vstack([combined, leg])
cv2.imwrite(str(OUT), final)
print(f"\n=> {OUT}")
