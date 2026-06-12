import json, sys
import numpy as np
from scipy.signal import savgol_filter, find_peaks

with open("engine/kp_cache/Videos2026-06-09_201015_827.json") as f:
    d = json.load(f)
frames = d["frames"]
fps = d["stats"]["source_fps"]
n = len(frames)

ys = np.full(n, np.nan)
for fd in frames:
    fi = fd["frame"]
    if fd["persons"]:
        kps = fd["persons"][0]["keypoints"]
        lw, rw = kps["left_wrist"], kps["right_wrist"]
        lsc, rsc = lw["score"], rw["score"]
        if max(lsc, rsc) >= 0.4:
            w = lsc + rsc
            ys[fi] = (lw["y"]*lsc + rw["y"]*rsc) / w
idx = np.arange(n)
nans = np.isnan(ys)
ys[nans] = np.interp(idx[nans], idx[~nans], ys[~nans])

win = max(7, int(fps*200/1000)) | 1
ys_s = savgol_filter(ys, win, 3)

print(f"n={n} fps={fps:.1f}")
print(f"wrist-Y range: {ys_s.min():.0f} to {ys_s.max():.0f}  range={ys_s.max()-ys_s.min():.0f}")
print(f"GT impact frames: swing1=fr59, swing2~fr207, swing3~fr369")

# Try different min_dist values
for md_s in [1.5, 2.5, 3.0]:
    md = max(int(fps * md_s), 10)
    for prom in [20, 35, 50]:
        peaks, props = find_peaks(ys_s, prominence=prom, distance=md)
        valid = (peaks >= int(n*0.05)) & (peaks <= int(n*0.95))
        pk = peaks[valid]
        pr = props["prominences"][valid]
        print(f"  min_dist={md_s}s prom={prom}: count={len(pk)} peaks={pk.tolist()} proms={[round(x,1) for x in pr.tolist()]}")
