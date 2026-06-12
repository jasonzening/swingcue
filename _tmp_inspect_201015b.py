import json
import numpy as np
from scipy.signal import savgol_filter, find_peaks

with open("engine/kp_cache/Videos2026-06-09_201015_827.json") as f:
    d = json.load(f)
frames_d = d["frames"]
fps = d["stats"]["source_fps"]
n = len(frames_d)
ys = np.full(n, np.nan)
for fd in frames_d:
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

print("Testing different min_dist:")
for md_s in [3.5, 4.0, 4.5, 5.0]:
    md = int(fps * md_s)
    peaks, props = find_peaks(ys_s, prominence=20, distance=md)
    valid = (peaks >= int(n*0.05)) & (peaks <= int(n*0.95))
    pk = peaks[valid]
    pr = props["prominences"][valid]
    print(f"  min_dist={md_s}s: count={len(pk)} peaks={pk.tolist()} proms={[round(x,1) for x in pr.tolist()]}")

print()
# Approach 2: require preceding local minimum (backswing top) within 0.3-2s before each peak
# Check which peaks in the 1.5s list have a local min in the 30-90fr before them
peaks_all, _ = find_peaks(ys_s, prominence=20, distance=int(fps*1.5))
valid = (peaks_all >= int(n*0.05)) & (peaks_all <= int(n*0.95))
peaks_all = peaks_all[valid]

print("Peaks with preceding backswing-top check (local min 0.5-3s before):")
real_impacts = []
for pk in peaks_all:
    lo = max(0, pk - int(fps*3.0))
    hi = max(0, pk - int(fps*0.3))
    if hi <= lo:
        continue
    region = ys_s[lo:hi]
    # Find local minima in region
    lm, _ = find_peaks(-region, prominence=30)
    if len(lm) > 0:
        real_impacts.append(int(pk))
        print(f"  fr{pk}: has backswing top(s) at {[int(lo+l) for l in lm]} -> REAL IMPACT")
    else:
        print(f"  fr{pk}: no preceding local min -> PLATEAU (skip)")

print(f"\nFiltered swing count: {len(real_impacts)}, peaks: {real_impacts}")
