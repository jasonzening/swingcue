"""
CUE-CHICKENWING-001 Step 2 — 静态指示器原型
最严重帧: fr096  (B2=+55.1px 外飞, B3=-26.6px 肘高于肩)
触发帧:   fr090  (首次进入 UP+OUT 方向翻转)

两版设计:
  V1 = 箭头 + 理想肘位幽灵点 (极简, Law10)
  V2 = 箭头 + 理想肘位 + 当前→目标引导弧 (稍多信息)
"""

import json, math, cv2, numpy as np
from pathlib import Path

ROOT      = Path("/home/jason/projects/swingcue-postest")
USER_CACHE = ROOT / "engine/kp_cache/batch2/fo-ok-1.json"
USER_VID   = ROOT / "input/fo-ok-1.mp4"
OUT_V1    = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/cw001_indicator_v1.jpg")
OUT_V2    = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/cw001_indicator_v2.jpg")
for p in [OUT_V1, OUT_V2]: p.parent.mkdir(parents=True, exist_ok=True)

WORST_FR   = 96   # argmax severity (B2=+55.1, B3=-26.6)
TRIGGER_FR = 90   # first UP+OUT frame

with open(USER_CACHE) as f: user_frames = json.load(f)['frames']

def get_kp(frames, fi):
    fr = frames[fi]
    if not fr['persons']: return {}
    return {k: (v['x'], v['y'], v['score']) for k,v in fr['persons'][0]['keypoints'].items()}

def pt(kp, name):
    return np.array(kp[name][:2]) if name in kp else None

def ip_(arr):
    return (int(arr[0]), int(arr[1]))

def extract_frame(vid, fi):
    cap = cv2.VideoCapture(str(vid))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

# ── geometry: ideal elbow position ─────────────────────────────────────────
# Rule: "elbow should fold DOWN — point toward ground"
# Construct: take user's shoulder + wrist at worst frame,
#            place ideal elbow directly below shoulder at upper-arm length
# Upper arm length calibrated from impact fr088 (arm near straight = good ref)
kp88  = get_kp(user_frames, 88)
ls88  = pt(kp88, 'left_shoulder')
le88  = pt(kp88, 'left_elbow')
UA_LEN = np.linalg.norm(le88 - ls88) if ls88 is not None and le88 is not None else 90.0

kp_w  = get_kp(user_frames, WORST_FR)
ls_w  = pt(kp_w, 'left_shoulder')
le_w  = pt(kp_w, 'left_elbow')
lw_w  = pt(kp_w, 'left_wrist')
rs_w  = pt(kp_w, 'right_shoulder')
re_w  = pt(kp_w, 'right_elbow')

# Ideal elbow: directly below shoulder at upper-arm distance
# "elbow points down" = shoulder_pos + (0, +UA_LEN) in image coords
ideal_elbow = ls_w + np.array([0.0, UA_LEN])

# Computed B2/B3 at worst frame
B2 = le_w[0] - ls_w[0]
B3 = le_w[1] - ls_w[1]
print(f"WORST fr{WORST_FR}: B2={B2:+.1f}  B3={B3:+.1f}")
print(f"Upper arm length: {UA_LEN:.1f}px")
print(f"Ideal elbow: {ideal_elbow}")
print(f"Current elbow: {le_w}")
print(f"Displacement: {le_w - ideal_elbow}")

# ── draw helper functions ───────────────────────────────────────────────────
def draw_base_skeleton(img, kp, lead_col=(0,220,255), muted=(70,70,70)):
    """Muted full skeleton + highlighted lead arm."""
    def ip(name):
        p = pt(kp, name)
        return None if p is None else (int(p[0]), int(p[1]))
    ls=ip('left_shoulder'); le=ip('left_elbow'); lw=ip('left_wrist')
    rs=ip('right_shoulder');re=ip('right_elbow');rw=ip('right_wrist')
    lh=ip('left_hip');      rh=ip('right_hip')

    # muted background
    for a,b in [(rs,re),(re,rw),(lh,rh),(ls,rs),(ls,lh),(rs,rh),(lw,rw)]:
        if a and b: cv2.line(img, a, b, muted, 2)
    for p in [rs,re,rw,lh,rh]:
        if p: cv2.circle(img, p, 4, muted, -1)

    # lead arm
    for a,b in [(ls,le),(le,lw)]:
        if a and b: cv2.line(img, a, b, lead_col, 3)
    for p,r in [(ls,7),(lw,6)]:
        if p: cv2.circle(img, p, r, lead_col, -1)
    # lead elbow — bigger, error-highlight
    if le: cv2.circle(img, le, 11, (0,80,220), -1)
    if le: cv2.circle(img, le, 11, (40,140,255), 2)

def draw_arrow_down(img, elbow_pt, label_txt, col=(255,255,255)):
    """Arc arrow pointing DOWNWARD from elbow — 'bring it down'."""
    ex, ey = int(elbow_pt[0]), int(elbow_pt[1])
    # curved arc: draw as a series of points curving down-inward
    # from current elbow to ~45° below-left
    angle_start = -30   # degrees from vertical
    angle_end   = 100
    radius      = 55
    pts = []
    for a in range(angle_start, angle_end+1, 3):
        rad = math.radians(a)
        px  = ex + int(radius * math.sin(rad))
        py  = ey - int(radius * math.cos(rad))   # y up in image = decrease
        pts.append((px, py))
    # draw arc
    for i in range(len(pts)-1):
        cv2.line(img, pts[i], pts[i+1], col, 3)
    # arrowhead at end
    if len(pts) >= 2:
        cv2.arrowedLine(img, pts[-2], pts[-1], col, 3, tipLength=0.6)
    # label
    cv2.putText(img, label_txt, (ex + radius + 8, ey + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

def draw_ideal_ghost(img, ideal_pt, ls_pt, lw_pt, alpha=0.55):
    """Draw ghosted ideal elbow position as a glowing target circle + arm lines."""
    ix, iy = int(ideal_pt[0]), int(ideal_pt[1])
    # glow rings
    for r, a in [(22,40),(16,80),(11,140)]:
        overlay = img.copy()
        cv2.circle(overlay, (ix,iy), r, (0,255,120), -1)
        cv2.addWeighted(overlay, a/255, img, 1-a/255, 0, img)
    # solid circle border
    cv2.circle(img, (ix,iy), 11, (0,255,120), 2)
    # ghosted lead arm lines in ideal position (dashed)
    if ls_pt is not None:
        ls_ = ip_(ls_pt); lw_ = ip_(lw_pt) if lw_pt is not None else None
        # shoulder → ideal elbow (dashed)
        _draw_dashed(img, ls_, (ix,iy), (0,200,100), 2)
        # ideal elbow → wrist (dashed, keep wrist same position)
        if lw_: _draw_dashed(img, (ix,iy), lw_, (0,200,100), 2)

def _draw_dashed(img, p1, p2, col, thickness, dash=12):
    p1 = np.array(p1, float); p2 = np.array(p2, float)
    d  = np.linalg.norm(p2-p1)
    n  = max(int(d/dash/2), 1)
    for i in range(n+1):
        t0 = (2*i*dash)/d if d>0 else 0
        t1 = min((2*i+1)*dash/d, 1.0) if d>0 else 1
        a  = (p1 + t0*(p2-p1)).astype(int)
        b  = (p1 + t1*(p2-p1)).astype(int)
        cv2.line(img, tuple(a), tuple(b), col, thickness)

def draw_guide_line(img, from_pt, to_pt, col=(180,180,50)):
    """Thin dashed line: current elbow → ideal elbow."""
    _draw_dashed(img, ip_(from_pt), ip_(to_pt), col, 1)
    cv2.arrowedLine(img, ip_(from_pt), ip_(to_pt), col, 1, tipLength=0.15)

def add_header(img, title, sub, trigger, worst):
    cv2.putText(img, title, (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(img, sub,   (18, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (160,160,160), 1)
    cv2.putText(img, f"trigger fr{trigger}  |  worst fr{worst}",
                (18, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200,180,80), 1)

def add_metric_strip(img, b2, b3):
    h = img.shape[0]
    cv2.putText(img, f"B2(lateral)={b2:+.0f}px  B3(vertical)={b3:+.0f}px",
                (18, h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
    cv2.putText(img, "B3<0 = elbow ABOVE shoulder  |  B2>0 = elbow OUTSIDE shoulder",
                (18, h-34), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120,120,120), 1)

# ============================================================
#  V1:极简版 — 箭头 + 理想肘位幽灵点 (≤2 元素)
# ============================================================
img_v1 = extract_frame(USER_VID, WORST_FR)
assert img_v1 is not None

draw_base_skeleton(img_v1, kp_w)

# element 1: arc arrow at current (wrong) elbow → point DOWN
draw_arrow_down(img_v1, le_w, "tuck down", col=(255,80,60))

# element 2: ideal elbow ghost
draw_ideal_ghost(img_v1, ideal_elbow, ls_w, lw_w)

# small label on ideal
ix, iy = int(ideal_elbow[0]), int(ideal_elbow[1])
cv2.putText(img_v1, "elbow here", (ix-20, iy+28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,220,100), 1)

add_header(img_v1,
    f"V1 — fr{WORST_FR}  lead elbow UP+OUT  [chicken wing]",
    "red arc = fix direction  |  green ghost = where elbow should be",
    TRIGGER_FR, WORST_FR)
add_metric_strip(img_v1, B2, B3)

cv2.imwrite(str(OUT_V1), img_v1)
print(f"V1 => {OUT_V1}")

# ============================================================
#  V2: 稍多信息 — 箭头 + 理想肘位 + 当前→目标引导线
# ============================================================
img_v2 = extract_frame(USER_VID, WORST_FR)
assert img_v2 is not None

draw_base_skeleton(img_v2, kp_w)

# guide line: current elbow → ideal (thin, background)
draw_guide_line(img_v2, le_w, ideal_elbow)

# ideal ghost
draw_ideal_ghost(img_v2, ideal_elbow, ls_w, lw_w)
cv2.putText(img_v2, "correct", (ix-12, iy+30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,220,100), 1)

# arrow at wrong elbow
draw_arrow_down(img_v2, le_w, "bring down", col=(255,80,60))

# B2/B3 displacement callout
le_ix, le_iy = int(le_w[0]), int(le_w[1])
cv2.putText(img_v2, f"+{B2:.0f}px out", (le_ix+14, le_iy+6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,120,60), 1)
cv2.putText(img_v2, f"{B3:.0f}px high", (le_ix+14, le_iy+24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,120,60), 1)

add_header(img_v2,
    f"V2 — fr{WORST_FR}  lead elbow UP+OUT  [chicken wing]",
    "arrow=fix  |  green=target  |  dashed=displacement",
    TRIGGER_FR, WORST_FR)
add_metric_strip(img_v2, B2, B3)

cv2.imwrite(str(OUT_V2), img_v2)
print(f"V2 => {OUT_V2}")
