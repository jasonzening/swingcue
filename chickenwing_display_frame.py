"""
CUE-CHICKENWING-001 — 候选展示帧筛选
follow-through 窗口 fr088~102
同时计算: 肩线宽度(正面度) + 鸡翅膀可见性(B2/B3) + 运动模糊估计
"""

import json, cv2, numpy as np
from pathlib import Path

ROOT       = Path("/home/jason/projects/swingcue-postest")
USER_CACHE = ROOT / "engine/kp_cache/batch2/fo-ok-1.json"
USER_VID   = ROOT / "input/fo-ok-1.mp4"
OUT        = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/cw001_display_candidates.jpg")
OUT.parent.mkdir(parents=True, exist_ok=True)

WIN = list(range(88, 103))

with open(USER_CACHE) as f:
    user_frames = json.load(f)['frames']

def get_kp(fi):
    fr = user_frames[fi]
    if not fr['persons']: return {}
    return {k: (v['x'], v['y'], v['score'])
            for k, v in fr['persons'][0]['keypoints'].items()}

def pt(kp, name):
    return np.array(kp[name][:2]) if name in kp else None

def extract_frame(fi):
    cap = cv2.VideoCapture(str(USER_VID))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def blur_score(img):
    """Laplacian variance — higher = sharper."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def shoulder_width_px(kp):
    """Horizontal span of shoulders — larger = more face-on."""
    ls = pt(kp, 'left_shoulder')
    rs = pt(kp, 'right_shoulder')
    if ls is None or rs is None: return 0.0
    return abs(ls[0] - rs[0])

# ── score each frame ──────────────────────────────────────────────────────────
print(f"{'fr':>4}  {'B2':>7}  {'B3':>7}  {'sh_w':>6}  {'blur':>7}  cw_vis  facing  notes")
scores = []
for fi in WIN:
    kp  = get_kp(fi)
    img = extract_frame(fi)
    if img is None or not kp:
        continue

    ls  = pt(kp, 'left_shoulder')
    le  = pt(kp, 'left_elbow')
    lw  = pt(kp, 'left_wrist')
    if ls is None or le is None or lw is None:
        continue

    b2  = le[0] - ls[0]
    b3  = le[1] - ls[1]
    shw = shoulder_width_px(kp)
    blr = blur_score(img)

    # chicken wing visible: elbow clearly outside shoulder-wrist line
    ab  = lw - ls
    ap  = le - ls
    cross = ab[0]*ap[1] - ab[1]*ap[0]
    elbow_offset = cross / (np.linalg.norm(ab) + 1e-9)
    cw_vis = "YES" if elbow_offset > 12 else ("weak" if elbow_offset > 5 else "no")

    # facing: shoulder horizontal span > 60px = still reasonably face-on
    facing = "good" if shw > 70 else ("ok" if shw > 50 else "turned")

    note = ""
    if fi == 96: note = "<< worst(detect)"
    if fi == 88: note = "<< impact"
    if fi == 97: note = "<< follow GT"

    print(f"  {fi:3d}  {b2:+7.1f}  {b3:+7.1f}  {shw:6.1f}  {blr:7.1f}  {cw_vis:6s}  {facing:6s}  {note}")
    scores.append((fi, b2, b3, shw, blr, elbow_offset, cw_vis, facing))

# ── pick top candidates: cw_vis=YES + facing=good/ok, ranked by blur*shoulder ──
candidates = [(fi, b2, b3, shw, blr, eo)
              for fi, b2, b3, shw, blr, eo, cw, fc in scores
              if cw == "YES" and fc in ("good","ok")]
# sort: maximize blur sharpness first, then shoulder width
candidates.sort(key=lambda x: (x[4], x[3]), reverse=True)
print("\nTop candidates (cw visible + face-on, sorted by sharpness):")
for fi, b2, b3, shw, blr, eo in candidates[:5]:
    print(f"  fr{fi:03d}  elbow_offset={eo:.1f}px  shoulder_w={shw:.1f}px  blur={blr:.1f}")

# ── contact sheet: 5 candidate frames with v3 overlay ────────────────────────
TARGET_H = 560
show_frs = [fi for fi, *_ in candidates[:5]]
if 96 not in show_frs:
    show_frs.append(96)   # always include detect frame for comparison
show_frs = sorted(show_frs)[:6]

RED    = (40, 60, 230)
GREEN  = (40, 200, 60)
YELLOW = (0, 210, 255)
MUTED  = (60, 60, 60)

panels = []
for fi in show_frs:
    img = extract_frame(fi)
    kp  = get_kp(fi)
    if img is None or not kp:
        continue

    ls  = pt(kp, 'left_shoulder')
    le  = pt(kp, 'left_elbow')
    lw  = pt(kp, 'left_wrist')
    rs  = pt(kp, 'right_shoulder')
    re  = pt(kp, 'right_elbow')
    rw  = pt(kp, 'right_wrist')
    lh  = pt(kp, 'left_hip')
    rh  = pt(kp, 'right_hip')

    def ip(a): return (int(a[0]), int(a[1]))

    # muted skeleton
    for a,b in [(rs,re),(re,rw),(lh,rh),(ls,rs),(ls,lh),(rs,rh),(lw,rw)]:
        if a is not None and b is not None:
            cv2.line(img, ip(a), ip(b), MUTED, 2)

    if ls is not None and le is not None and lw is not None:
        # red: shoulder→elbow→wrist
        cv2.line(img, ip(ls), ip(le), RED, 5)
        cv2.line(img, ip(le), ip(lw), RED, 5)
        cv2.circle(img, ip(ls), 8, RED, -1)
        cv2.circle(img, ip(le), 9, RED, -1)
        cv2.circle(img, ip(lw), 8, RED, -1)

        # green: shoulder→wrist direct
        cv2.line(img, ip(ls), ip(lw), GREEN, 5)
        cv2.circle(img, ip(ls), 8, GREEN, -1)
        cv2.circle(img, ip(lw), 8, GREEN, -1)

        # yellow arrow: elbow → nearest point on green line
        sw_vec = lw - ls
        t = float(np.clip(np.dot(le - ls, sw_vec) / (np.dot(sw_vec, sw_vec)+1e-9), 0.05, 0.95))
        tgt = ls + t * sw_vec
        cv2.arrowedLine(img, ip(le), ip(tgt), YELLOW, 4, tipLength=0.25)

    # elbow offset label
    ab = lw - ls; ap = le - ls
    cross = ab[0]*ap[1] - ab[1]*ap[0]
    eo = cross / (np.linalg.norm(ab)+1e-9)
    shw = shoulder_width_px(kp)
    blr = blur_score(img)

    label = "DETECT" if fi == 96 else "candidate"
    label_col = (0, 140, 255) if fi == 96 else (200, 200, 200)
    cv2.putText(img, f"fr{fi:03d}  {label}", (14, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, label_col, 2)
    cv2.putText(img, f"elbow_offset={eo:.0f}px", (14, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180,180,180), 1)
    cv2.putText(img, f"sh_w={shw:.0f}px  blur={blr:.0f}", (14, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180,180,180), 1)

    scale = TARGET_H / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1]*scale), TARGET_H))
    panels.append(img)

sep = np.full((TARGET_H, 5, 3), 35, dtype=np.uint8)
combined = panels[0]
for p in panels[1:]:
    combined = np.hstack([combined, sep, p])

# caption bar
cap_h = 60
cap = np.full((cap_h, combined.shape[1], 3), 12, dtype=np.uint8)
cv2.putText(cap, "Candidates: red=current arm  green=shoulder-wrist line  yellow=fix direction  |  fr096=detect frame (reference)",
            (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (150,150,150), 1)
final = np.vstack([combined, cap])
cv2.imwrite(str(OUT), final)
print(f"\n=> {OUT}")
