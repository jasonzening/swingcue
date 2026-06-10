import json
import numpy as np
from scipy.signal import savgol_filter, find_peaks

with open('/home/jason/projects/swingcue-postest/output/rtmpose/test-dwontheline_keypoints.json') as f:
    d = json.load(f)

n = len(d['frames'])
ys = np.full(n, np.nan)
spd_raw = np.zeros(n)

for fd in d['frames']:
    fi = fd['frame']
    if not fd['persons']:
        continue
    kps = fd['persons'][0]['keypoints']
    lw = kps['left_wrist']
    rw = kps['right_wrist']
    lsc, rsc = lw['score'], rw['score']
    if max(lsc, rsc) < 0.4:
        continue
    w = lsc + rsc
    ys[fi] = (lw['y']*lsc + rw['y']*rsc) / w

# interpolate nans
nans = np.isnan(ys)
idx = np.arange(n)
ys[nans] = np.interp(idx[nans], idx[~nans], ys[~nans])

ys_s = savgol_filter(ys, 7, 3)
dy = np.diff(ys_s, prepend=ys_s[0])
spd_s = savgol_filter(np.abs(np.diff(ys_s, prepend=ys_s[0])), 7, 3)

print("Frame | ys_smooth | dy (vy) | speed | notes")
for i in range(n):
    note = ""
    if i == 31: note = " << TOP candidate"
    if i == 32: note = " << TOP candidate"
    if i == 55: note = " << second high point"
    if i == 56: note = " << second high point"
    print(f"{i:4d}  | {ys_s[i]:7.1f}  | {dy[i]:+7.2f}  | {spd_s[i]:5.2f} {note}")

# Find all local minima of ys_s (= wrist high points in image)
minima, _ = find_peaks(-ys_s, prominence=30)
print("\nAll local minima of ys (= peaks in wrist height):")
for m in minima:
    print(f"  frame {m}: ys_s={ys_s[m]:.1f}  spd={spd_s[m]:.2f}")
