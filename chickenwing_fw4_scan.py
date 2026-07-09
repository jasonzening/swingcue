"""
CUE-CHICKENWING-001 — fo-wrong-4 全序列扫描
1. 8相位检测
2. lead肘方向轨迹全程扫
3. 候选展示帧筛选
4. v3三色指示器渲染
"""

import json, math, cv2, numpy as np
from pathlib import Path

ROOT      = Path("/home/jason/projects/swingcue-postest")
CACHE     = ROOT / "engine/kp_cache/batch2/fo-wrong-4.json"
VID       = Path("/mnt/c/Users/jason/Zening/Swingcue/Video/fo-wrong-4.mp4")
OUT_SCAN  = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/cw001_fw4_scan.jpg")
OUT_CUE   = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/cw001_fw4_indicator.jpg")
for p in [OUT_SCAN, OUT_CUE]: p.parent.mkdir(parents=True, exist_ok=True)

with open(CACHE) as f:
    d = json.load(f)
frames = d['frames']
NF     = len(frames)
FPS    = d['stats']['source_fps']

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

# ── phase detection (wrist trajectory) ────────────────────────────────────────
wrist_y = []
for fi in range(NF):
    kp = get_kp(fi)
    lw = pt(kp, 'left_wrist')
    rw = pt(kp, 'right_wrist')
    wy = min(lw[1] if lw is not None else 9999,
             rw[1] if rw is not None else 9999)
    wrist_y.append(wy)

wrist_arr = np.array(wrist_y)

# address: first stable section before wrist starts moving up
# find first significant rise
dwy = np.diff(wrist_arr)
addr_end = 0
for i in range(10, NF-10):
    if wrist_arr[i] - wrist_arr[i-5] < -15:  # wrist going up = takeaway start
        addr_end = i
        break
if addr_end == 0: addr_end = 20

# top: wrist at minimum y (highest point)
search_start = addr_end + 5
search_end   = min(addr_end + 80, NF - 20)
top_fr = int(np.argmin(wrist_arr[search_start:search_end])) + search_start

# impact: wrist at maximum y (lowest, after top), before follow
impact_search_end = min(top_fr + 60, NF - 10)
impact_fr = int(np.argmax(wrist_arr[top_fr:impact_search_end])) + top_fr

# follow: after impact
follow_start = impact_fr + 1

phases = {
    'address':    (0, addr_end),
    'backswing':  (addr_end, top_fr - 2),
    'top':        (top_fr - 2, top_fr + 2),
    'downswing':  (top_fr + 2, impact_fr - 1),
    'impact':     (impact_fr - 1, impact_fr + 2),
    'follow':     (follow_start, NF - 1),
}

print(f"NF={NF}  FPS={FPS:.1f}")
print("Phase anchors:")
for ph, (s,e) in phases.items():
    print(f"  {ph:12s}: fr{s:03d}~fr{e:03d}")

# ── full scan ────────────────────────────────────────────────────────────────
def phase_of(fi):
    for ph, (s,e) in phases.items():
        if s <= fi <= e: return ph
    return 'unknown'

def blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

scan = []
prev_b2, prev_b3 = None, None
for fi in range(NF):
    kp = get_kp(fi)
    ls = pt(kp,'left_shoulder'); le = pt(kp,'left_elbow')
    lw = pt(kp,'left_wrist');    rs = pt(kp,'right_shoulder')
    if ls is None or le is None or lw is None:
        prev_b2, prev_b3 = None, None
        continue
    b2 = le[0] - ls[0]
    b3 = le[1] - ls[1]
    ab = lw - ls; ap = le - ls
    cross = ab[0]*ap[1] - ab[1]*ap[0]
    offset = cross / (np.linalg.norm(ab)+1e-9)
    shw = abs(ls[0] - rs[0]) if rs is not None else 0
    db2 = b2 - prev_b2 if prev_b2 is not None else 0
    db3 = b3 - prev_b3 if prev_b3 is not None else 0
    prev_b2, prev_b3 = b2, b3

    cw_dir = (db2 > 8 and db3 < -4)
    cw_abs = (offset > 14)
    facing = 'good' if shw > 70 else ('ok' if shw > 50 else 'turned')
    ph = phase_of(fi)

    scan.append({
        'fi': fi, 'b2': b2, 'b3': b3, 'offset': offset,
        'db2': db2, 'db3': db3, 'shw': shw,
        'cw_dir': cw_dir, 'cw_abs': cw_abs, 'facing': facing, 'phase': ph
    })

# print signal frames
print(f"\n{'fr':>4}  {'phase':12s}  {'B2':>7}  {'B3':>7}  {'offset':>7}  {'sh_w':>5}  {'dB2':>6}  {'dB3':>6}  cw  facing")
print('-'*80)
for s in scan:
    if s['cw_dir'] or s['cw_abs'] or s['fi'] % 20 == 0:
        cw_tag = 'CW' if (s['cw_dir'] or s['cw_abs']) else '  '
        print(f"  {s['fi']:3d}  {s['phase']:12s}  {s['b2']:+7.1f}  {s['b3']:+7.1f}  "
              f"{s['offset']:+7.1f}  {s['shw']:5.0f}  {s['db2']:+6.1f}  {s['db3']:+6.1f}  "
              f"{cw_tag}  {s['facing']}")

# ── find best display frames ──────────────────────────────────────────────────
candidates = [s for s in scan
              if (s['cw_dir'] or s['cw_abs'])
              and s['facing'] in ('good','ok')
              and s['offset'] > 10]

print(f"\nDisplay candidates (CW visible + facing good/ok + offset>10):")
for s in candidates:
    print(f"  fr{s['fi']:03d}  {s['phase']:12s}  offset={s['offset']:+.1f}px  "
          f"sh_w={s['shw']:.0f}px  facing={s['facing']}")

# pick best: max offset among good-facing frames
if candidates:
    best = max(candidates, key=lambda s: s['offset'] * (1.5 if s['facing']=='good' else 1.0))
    print(f"\nBEST display frame: fr{best['fi']:03d}  phase={best['phase']}  "
          f"offset={best['offset']:+.1f}px  facing={best['facing']}")
else:
    print("\nNo clean display frame found.")
    best = max(scan, key=lambda s: s['offset'])
    print(f"Fallback (max offset): fr{best['fi']:03d}  offset={best['offset']:+.1f}px")

DISPLAY_FR  = best['fi']
DETECT_FR   = max(scan, key=lambda s: s['offset'])['fi']
print(f"Detect frame (argmax offset): fr{DETECT_FR:03d}  offset={max(s['offset'] for s in scan):+.1f}px")

# ── contact sheet: scan overview ─────────────────────────────────────────────
# pick frames spanning the CW region for visual overview
cw_frames = [s['fi'] for s in scan if (s['cw_dir'] or s['cw_abs'])]
if cw_frames:
    cw_start, cw_end = cw_frames[0], cw_frames[-1]
    # sample 6 frames: phase start, CW start-2, CW start, display, detect, CW end
    overview_frs = sorted(set([
        max(0, cw_start - 3),
        cw_start,
        DISPLAY_FR,
        DETECT_FR,
        min(NF-1, cw_end),
    ]))[:6]
else:
    overview_frs = [0, NF//4, NF//2, DISPLAY_FR]

RED    = (40, 60, 230)
GREEN  = (40, 200, 60)
YELLOW = (0, 210, 255)
MUTED  = (60, 60, 60)

def draw_v3(img, kp, label, label_col, show_metrics=True):
    def ip(a): return (int(a[0]), int(a[1]))
    ls = pt(kp,'left_shoulder'); le = pt(kp,'left_elbow')
    lw = pt(kp,'left_wrist');    rs = pt(kp,'right_shoulder')
    re = pt(kp,'right_elbow');   rw = pt(kp,'right_wrist')
    lh = pt(kp,'left_hip');      rh = pt(kp,'right_hip')

    for a,b in [(rs,re),(re,rw),(lh,rh),(ls,rs),(ls,lh),(rs,rh),(lw,rw)]:
        if a is not None and b is not None:
            cv2.line(img, ip(a), ip(b), MUTED, 2)
    for p in [rs,re,rw,lh,rh]:
        if p is not None: cv2.circle(img, ip(p), 4, MUTED, -1)

    if ls is not None and le is not None and lw is not None:
        # red
        cv2.line(img, ip(ls), ip(le), RED, 5)
        cv2.line(img, ip(le), ip(lw), RED, 5)
        for p,r in [(ls,8),(le,10),(lw,8)]: cv2.circle(img, ip(p), r, RED, -1)
        # green
        cv2.line(img, ip(ls), ip(lw), GREEN, 5)
        cv2.circle(img, ip(ls), 8, GREEN, -1)
        cv2.circle(img, ip(lw), 8, GREEN, -1)
        # yellow arrow
        sw_vec = lw - ls
        t = float(np.clip(np.dot(le - ls, sw_vec)/(np.dot(sw_vec,sw_vec)+1e-9), 0.05, 0.95))
        tgt = ls + t * sw_vec
        cv2.arrowedLine(img, ip(le), ip(tgt), YELLOW, 4, tipLength=0.25)

        if show_metrics:
            b2 = le[0]-ls[0]; b3 = le[1]-ls[1]
            ab = lw-ls; ap = le-ls
            cross = ab[0]*ap[1]-ab[1]*ap[0]
            off = cross/(np.linalg.norm(ab)+1e-9)
            cv2.putText(img, f"offset={off:+.0f}px", (14, img.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

    cv2.putText(img, label, (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, label_col, 2)
    return img

TARGET_H = 560
panels = []
for fi in overview_frs:
    img = extract_frame(fi)
    kp  = get_kp(fi)
    if img is None: continue
    s_info = next((s for s in scan if s['fi']==fi), {})
    ph = s_info.get('phase','?')
    cw = 'CW' if (s_info.get('cw_dir') or s_info.get('cw_abs')) else ''
    marker = ' DISPLAY' if fi==DISPLAY_FR else (' DETECT' if fi==DETECT_FR else '')
    col = (0,210,255) if fi==DISPLAY_FR else ((0,80,200) if fi==DETECT_FR else (180,180,180))
    draw_v3(img, kp, f"fr{fi:03d} {ph} {cw}{marker}", col)
    scale = TARGET_H/img.shape[0]
    img = cv2.resize(img, (int(img.shape[1]*scale), TARGET_H))
    panels.append(img)

sep = np.full((TARGET_H, 5, 3), 35, dtype=np.uint8)
combined = panels[0]
for p in panels[1:]: combined = np.hstack([combined, sep, p])

cap_h = 50
cap = np.full((cap_h, combined.shape[1], 3), 12, dtype=np.uint8)
cv2.putText(cap,
    f"fo-wrong-4 scan  |  cyan=display fr  |  blue=detect fr  |  red/green/yellow=v3 indicator",
    (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (150,150,150), 1)
cv2.imwrite(str(OUT_SCAN), np.vstack([combined, cap]))
print(f"\nScan => {OUT_SCAN}")

# ── final indicator on display frame ─────────────────────────────────────────
img_cue = extract_frame(DISPLAY_FR)
kp_cue  = get_kp(DISPLAY_FR)
draw_v3(img_cue, kp_cue,
        f"fr{DISPLAY_FR:03d}  chicken wing  [display]",
        (0,210,255), show_metrics=True)
cv2.imwrite(str(OUT_CUE), img_cue)
print(f"Cue  => {OUT_CUE}")
