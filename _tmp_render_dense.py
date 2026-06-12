"""Render every frame in specified ranges with tush line + spine axis."""
import json, sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine.a_measurement.pose_pipeline import PosePipeline
from engine.b_phase.swing_phase import SwingPhaseEngine

PROJ = Path("/home/jason/projects/swingcue-postest")

C_TUSH  = (0, 220, 255)
C_SPINE = (255, 220, 0)
C_WHITE = (255, 255, 255)
C_BLACK = (0, 0, 0)
FONT    = cv2.FONT_HERSHEY_DUPLEX
LINE_W  = 3

def kp_pt(kps, name, thr=0.3):
    if name not in kps: return None
    k = kps[name]
    if k["score"] < thr: return None
    return (float(k["x"]), float(k["y"]))

def mid_pt(a, b):
    if a is None or b is None: return None
    return ((a[0]+b[0])/2, (a[1]+b[1])/2)

def draw_vline(img, x, color, label="", proxy=False):
    h = img.shape[0]
    cv2.line(img, (int(x),0), (int(x),h), color, LINE_W, cv2.LINE_AA)
    tag = label + (" PROXY" if proxy else "")
    if tag: cv2.putText(img, tag, (int(x)+4,40), FONT, 0.45, color, 1, cv2.LINE_AA)

def draw_spine(img, hip_mid, sh_mid, ext=0.20):
    dx = sh_mid[0]-hip_mid[0]; dy = sh_mid[1]-hip_mid[1]
    p1 = (int(hip_mid[0]-dx*ext), int(hip_mid[1]-dy*ext))
    p2 = (int(sh_mid[0]+dx*ext),  int(sh_mid[1]+dy*ext))
    cv2.line(img, p1, p2, C_SPINE, LINE_W, cv2.LINE_AA)
    for p in [(int(hip_mid[0]),int(hip_mid[1])), (int(sh_mid[0]),int(sh_mid[1]))]:
        cv2.circle(img, p, 5, C_SPINE, -1, cv2.LINE_AA)

def label_img(img, vid_id, fr, phase):
    text = f"{vid_id} fr{fr:03d} {phase}"
    (tw,th),_ = cv2.getTextSize(text, FONT, 0.52, 1)
    cv2.rectangle(img,(0,0),(tw+12,th+12),C_BLACK,-1)
    cv2.putText(img, text,(6,th+4),FONT,0.52,C_WHITE,1,cv2.LINE_AA)

def render_range(vid_stem, kp_cache_key, angle, fr_start, fr_end, out_dir):
    cache = PROJ / "engine/kp_cache" / f"{kp_cache_key}.json"
    if not cache.exists():
        # try normal_group subfolder
        cache = PROJ / "engine/kp_cache/normal_group" / f"{kp_cache_key}.json"
    with open(cache) as f:
        kp_json = json.load(f)

    vpath_candidates = [
        PROJ / "input" / f"{vid_stem}.mp4",
        PROJ / "input/normal_group" / f"{vid_stem}.mp4",
    ]
    vpath = next((p for p in vpath_candidates if p.exists()), None)
    if vpath is None:
        print(f"  ERROR: video not found for {vid_stem}")
        return 0

    pipe = PosePipeline(device="cpu")
    meas, fps = pipe.run_from_json(kp_json)

    eng = SwingPhaseEngine()
    ann, anchors = eng.run(meas, fps, angle=angle)
    phase_map = {a.frame_idx: a.phase for a in ann}

    # Address anchors for fixed lines
    addr_fr = anchors.address
    fd0 = kp_json["frames"][addr_fr]
    if not fd0["persons"]:
        print(f"  ERROR: no person at address fr{addr_fr}")
        return 0
    kps0 = fd0["persons"][0]["keypoints"]
    lh = kp_pt(kps0,"left_hip"); rh = kp_pt(kps0,"right_hip")
    ls = kp_pt(kps0,"left_shoulder"); rs = kp_pt(kps0,"right_shoulder")
    if not (lh and rh): return 0
    hip_mid = mid_pt(lh,rh)
    sh_mid  = mid_pt(ls,rs)
    tush_x  = hip_mid[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(kp_json["frames"])
    count = 0
    for fr in range(fr_start, min(fr_end+1, n)):
        cap = cv2.VideoCapture(str(vpath))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ret, img = cap.read(); cap.release()
        if not ret: continue

        draw_vline(img, tush_x, C_TUSH, "TUSH", proxy=True)
        if sh_mid: draw_spine(img, hip_mid, sh_mid)
        phase = phase_map.get(fr, "?")
        label_img(img, vid_stem[-12:], fr, phase)

        fname = f"fr{fr:03d}_{phase}.jpg"
        cv2.imwrite(str(out_dir/fname), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        count += 1

    return count

BASE = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/gt_lines/normal_group")

# test-dwontheline fr31-46
out1 = BASE / "test-dwontheline/dense_fr031-046"
n1 = render_range("test-dwontheline", "test-dwontheline", "down-the-line", 31, 46, out1)
print(f"test-dwontheline fr31-46: {n1} frames → {out1}")

# stodownload(53) fr208-221
out2 = BASE / "stodownload(53)/dense_fr208-221"
n2 = render_range("stodownload(53)", "stodownload(53)", "down-the-line", 208, 221, out2)
print(f"stodownload(53) fr208-221: {n2} frames → {out2}")
