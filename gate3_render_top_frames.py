#!/usr/bin/env python3
"""
gate3_render_top_frames.py
关卡3 — top帧渲染（供人工目视验证top定位是否合理）
"""
import sys, json, math, cv2, shutil
from pathlib import Path
import numpy as np

PROJ = Path("/home/jason/projects/swingcue-postest")
sys.path.insert(0, str(PROJ))

from engine.features.triline_geometry import _safe_pt, _lateral_tilt_deg, render_triline_frame
from gate3_no_false_positive import load_kp_json, find_top, tilt_from_kps

CLIPS = [
    ("fo-eet-1", "input/fo-eet-1.mp4", None),
    ("fo-eet-2", "input/fo-eet-2.mp4", None),
    ("fo-eet-3", "input/fo-eet-3.mp4", None),
    ("fo-ok-1",  "input/fo-ok-1.mp4",  None),
    ("fo-ok-2",  "input/fo-ok-2.mp4",  None),
]

OUT_DIR = PROJ / "output/gate3_no_fp"
OUT_WIN = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/gate3_no_fp")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_WIN.mkdir(parents=True, exist_ok=True)

RESULTS = json.load(open(OUT_DIR / "gate3_results.json"))["results"]
res_by_id = {r["clip_id"]: r for r in RESULTS}

for clip_id, rel_vid, _ in CLIPS:
    vid = PROJ / rel_vid
    r = res_by_id.get(clip_id, {})
    top_idx = r.get("top_idx")
    tilt    = r.get("tilt_deg")
    conf    = r.get("confidence", r.get("status", "?"))

    if top_idx is None:
        print(f"  {clip_id}: top_idx=None, skip render")
        continue

    cap = cv2.VideoCapture(str(vid))
    cap.set(cv2.CAP_PROP_POS_FRAMES, top_idx)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        print(f"  {clip_id}: frame read failed")
        continue

    # Get keypoints from kp_cache for this frame
    kp_json = load_kp_json(clip_id)
    frames  = kp_json.get("frames", []) if kp_json else []
    kps     = {}
    if top_idx < len(frames):
        p = frames[top_idx].get("persons", [])
        if p:
            kps = p[0].get("keypoints", {})

    tilt_s = f"{tilt:+.2f}" if tilt is not None else "N/A"
    label  = f"{clip_id}  fr{top_idx}  tilt={tilt_s}°  {conf}"
    feat   = {"shoulder_lateral_tilt": tilt or 0}
    ann    = render_triline_frame(bgr, kps, feat, label=label)

    out_path = OUT_DIR / f"gate3_{clip_id}_top.jpg"
    cv2.imwrite(str(out_path), ann, [cv2.IMWRITE_JPEG_QUALITY, 90])
    shutil.copy(out_path, OUT_WIN / out_path.name)
    print(f"  {clip_id}: top=fr{top_idx}  tilt={tilt_s}°  conf={conf}  → {out_path.name}")

print("done")
