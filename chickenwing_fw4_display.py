"""
CUE-CHICKENWING-001 — fo-wrong-4 候选展示帧对比
针对两个不同时机的最佳帧出 v3 指示器
fr149: CW方向刚起, 正面好, 肘刚开始飞出(B2=+41px)
fr166: CW绝对值最强可见帧(offset=+26px), 身体稍转
"""

import json, cv2, numpy as np
from pathlib import Path

ROOT  = Path("/home/jason/projects/swingcue-postest")
CACHE = ROOT / "engine/kp_cache/batch2/fo-wrong-4.json"
VID   = Path("/mnt/c/Users/jason/Zening/Swingcue/Video/fo-wrong-4.mp4")
OUT   = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/cw001_fw4_display_compare.jpg")
OUT.parent.mkdir(parents=True, exist_ok=True)

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

RED    = (40,  60, 230)
GREEN  = (40, 200,  60)
YELLOW = ( 0, 210, 255)
MUTED  = (60,  60,  60)

def draw_v3(img, kp, title, title_col, info_lines):
    def ip(a): return (int(a[0]), int(a[1]))
    ls = pt(kp,'left_shoulder'); le = pt(kp,'left_elbow')
    lw = pt(kp,'left_wrist');    rs = pt(kp,'right_shoulder')
    re = pt(kp,'right_elbow');   rw = pt(kp,'right_wrist')
    lh = pt(kp,'left_hip');      rh = pt(kp,'right_hip')

    # muted skeleton
    for a,b in [(rs,re),(re,rw),(lh,rh),(ls,rs),(ls,lh),(rs,rh),(lw,rw)]:
        if a is not None and b is not None:
            cv2.line(img, ip(a), ip(b), MUTED, 2)
    for p in [rs,re,rw,lh,rh]:
        if p is not None: cv2.circle(img, ip(p), 4, MUTED, -1)

    if ls is not None and le is not None and lw is not None:
        # red: shoulder→elbow→wrist
        cv2.line(img, ip(ls), ip(le), RED, 5)
        cv2.line(img, ip(le), ip(lw), RED, 5)
        for p,r in [(ls,8),(le,11),(lw,8)]:
            cv2.circle(img, ip(p), r, RED, -1)
        # green: shoulder→wrist direct
        cv2.line(img, ip(ls), ip(lw), GREEN, 5)
        cv2.circle(img, ip(ls), 8, GREEN, -1)
        cv2.circle(img, ip(lw), 8, GREEN, -1)
        # yellow arrow: elbow → nearest point on green line
        sw = lw - ls
        t  = float(np.clip(np.dot(le-ls, sw)/(np.dot(sw,sw)+1e-9), 0.05, 0.95))
        tgt = ls + t * sw
        cv2.arrowedLine(img, ip(le), ip(tgt), YELLOW, 4, tipLength=0.25)

    # title
    cv2.putText(img, title, (14, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, title_col, 2)
    # info lines bottom
    h = img.shape[0]
    for i, line in enumerate(info_lines):
        col = (200,200,200) if i > 0 else (180,180,80)
        cv2.putText(img, line, (14, h - 80 + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1)
    return img

# ── fr149: CW direction onset, face-on ───────────────────────────────────────
img_149 = extract_frame(149)
kp_149  = get_kp(149)
draw_v3(img_149, kp_149,
    "fr149  backswing  CW starting (face-on)",
    (0, 210, 255),
    ["CW: direction onset  dB2=+18  dB3=-9",
     "B2=+41px (elbow 41px left of shoulder)",
     "shoulder_w=96px  facing=GOOD",
     "offset=-16px (elbow just inside line)"])

# ── fr163: peak offset, still somewhat visible ───────────────────────────────
img_163 = extract_frame(163)
kp_163  = get_kp(163)
draw_v3(img_163, kp_163,
    "fr163  backswing  CW peak offset",
    (100, 100, 255),
    ["CW: offset=+39px (elbow outside line)",
     "B2=+45px  B3=+21px",
     "shoulder_w=42px  facing=TURNED",
     "argmax offset — detect frame"])

# ── fr166: best absolute+facing trade-off ────────────────────────────────────
img_166 = extract_frame(166)
kp_166  = get_kp(166)
draw_v3(img_166, kp_166,
    "fr166  top  CW visible + ok facing",
    (80, 200, 80),
    ["CW: offset=+26px (elbow outside line)",
     "B2=+44px  B3=+2px",
     "shoulder_w=54px  facing=OK",
     "best abs-offset + not fully turned"])

# ── combine ───────────────────────────────────────────────────────────────────
TARGET_H = 640
def resize_h(img, H):
    s = H / img.shape[0]
    return cv2.resize(img, (int(img.shape[1]*s), H))

p149 = resize_h(img_149, TARGET_H)
p163 = resize_h(img_163, TARGET_H)
p166 = resize_h(img_166, TARGET_H)

sep = np.full((TARGET_H, 8, 3), 35, dtype=np.uint8)
combined = np.hstack([p149, sep, p163, sep, p166])

# caption
cap_h = 130
cap = np.full((cap_h, combined.shape[1], 3), 12, dtype=np.uint8)
lines = [
    "fo-wrong-4 — v3 chicken wing indicator: RED=current arm  GREEN=shoulder-wrist line  YELLOW=fix direction",
    "",
    "  fr149 (cyan):  CW direction just starting; body FULLY face-on (sh_w=96px)  |  elbow offset=-16px (kink subtle)",
    "  fr163 (blue):  CW at peak (offset=+39px, kink most visible); body mostly turned (sh_w=42px)",
    "  fr166 (green): top of swing; offset=+26px visible kink; body ok (sh_w=54px) — best compromise",
    "",
    "  Jason: which frame do you want as the display frame? (or all three are bad -> DTL machine angle?)",
]
for i, line in enumerate(lines):
    c = (100,230,255) if i==0 else (160,160,160)
    if "fr149" in line: c = (0,210,255)
    if "fr163" in line: c = (100,100,255)
    if "fr166" in line: c = (60,200,60)
    if "Jason" in line: c = (220,200,60)
    cv2.putText(cap, line, (16, 22+i*17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, c, 1)

final = np.vstack([combined, cap])
cv2.imwrite(str(OUT), final)
print(f"=> {OUT}")
