import json
import numpy as np
from scipy.signal import savgol_filter

with open('/home/jason/projects/swingcue-postest/output/rtmpose/test-faceon_keypoints.json') as f:
    d = json.load(f)

n = len(d['frames'])
xs = np.full(n, np.nan)
ys = np.full(n, np.nan)

for fd in d['frames']:
    fi = fd['frame']
    if not fd['persons']:
        continue
    kps = fd['persons'][0]['keypoints']
    lw, rw = kps['left_wrist'], kps['right_wrist']
    lsc, rsc = lw['score'], rw['score']
    w = lsc + rsc
    xs[fi] = (lw['x']*lsc + rw['x']*rsc) / w
    ys[fi] = (lw['y']*lsc + rw['y']*rsc) / w

nans = np.isnan(xs)
idx = np.arange(n)
xs[nans] = np.interp(idx[nans], idx[~nans], xs[~nans])
ys[nans] = np.interp(idx[nans], idx[~nans], ys[~nans])

xs_s = savgol_filter(xs, 7, 3)
ys_s = savgol_filter(ys, 7, 3)
dx = np.diff(xs_s, prepend=xs_s[0])
dy = np.diff(ys_s, prepend=ys_s[0])
spd_s = savgol_filter(np.sqrt(dx**2+dy**2), 7, 3)

print(f"total frames: {n}, fps: {d['stats']['source_fps']:.1f}")
print()
print("Frame | mid_x  mid_y  | speed | notes")
for fi in range(n):
    note = ""
    if spd_s[fi] > 20: note += " *** HIGH SPEED"
    print(f"  {fi:3d} | {xs_s[fi]:6.1f} {ys_s[fi]:6.1f} | {spd_s[fi]:5.1f} {note}")
