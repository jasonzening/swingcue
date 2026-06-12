import json
import numpy as np
from scipy.signal import savgol_filter, find_peaks

with open("engine/kp_cache/Videos2026-06-09_201015_827.json") as f:
    d = json.load(f)
frames_d = d["frames"]
fps = d["stats"]["source_fps"]
n = len(frames_d)

xs = np.full(n, np.nan); ys = np.full(n, np.nan)
for fd in frames_d:
    fi = fd["frame"]
    if fd["persons"]:
        kps = fd["persons"][0]["keypoints"]
        lw, rw = kps["left_wrist"], kps["right_wrist"]
        lsc, rsc = lw["score"], rw["score"]
        if max(lsc, rsc) >= 0.4:
            w = lsc + rsc
            xs[fi] = (lw["x"]*lsc + rw["x"]*rsc) / w
            ys[fi] = (lw["y"]*lsc + rw["y"]*rsc) / w

idx = np.arange(n)
for arr in (xs, ys):
    nans = np.isnan(arr)
    arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])

win = max(7, int(fps*200/1000)) | 1
xs_s = savgol_filter(xs, win, 3)
ys_s = savgol_filter(ys, win, 3)
dx = np.diff(xs_s, prepend=xs_s[0])
dy = np.diff(ys_s, prepend=ys_s[0])
spd = savgol_filter(np.sqrt(dx**2 + dy**2), win, 3)

candidate_peaks = [59, 154, 207, 269, 368]
print("Speed analysis at each candidate peak (window = ±10fr):")
print(f"{'Peak':>6}  {'spd_peak':>10}  {'max_spd_before':>16}  {'mean_spd_-20':>14}  {'verdict':>12}")
for pk in candidate_peaks:
    sp = spd[pk]
    win_lo = max(0, pk-20); win_hi = min(n, pk+1)
    max_spd_before = spd[win_lo:win_hi].max()
    mean_spd_before = spd[win_lo:win_hi].mean()
    verdict = "IMPACT" if max_spd_before > 8.0 else "PLATEAU"
    print(f"{pk:>6}  {sp:>10.2f}  {max_spd_before:>16.2f}  {mean_spd_before:>14.2f}  {verdict:>12}")

print()
print("Speed-based swing detection:")
# Find ALL peaks by 1.5s min_dist
peaks_all, props_all = find_peaks(ys_s, prominence=20, distance=int(fps*1.5))
valid = (peaks_all >= int(n*0.05)) & (peaks_all <= int(n*0.95))
peaks_all = peaks_all[valid]
proms_all = props_all["prominences"][valid]

# For each peak: check max speed in 20 frames before
spd_thr = 8.0  # px/frame — at 30fps this is moderate speed
real_impacts = []
for pk, prom in zip(peaks_all, proms_all):
    win_lo = max(0, pk - 20)
    max_spd = spd[win_lo:pk+1].max()
    real = max_spd >= spd_thr
    if real:
        real_impacts.append(int(pk))
    print(f"  fr{pk}: prom={prom:.0f}  max_spd_before={max_spd:.1f}  -> {'REAL' if real else 'PLATEAU'}")

print(f"\nFinal: swing_count={len(real_impacts)}, real_impacts={real_impacts}")
