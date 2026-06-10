import json, numpy as np
from scipy.signal import savgol_filter, find_peaks

with open('/home/jason/projects/swingcue-postest/output/rtmpose/test-dwontheline_keypoints.json') as f:
    d = json.load(f)
n = len(d['frames'])

xs = np.full(n, np.nan); ys = np.full(n, np.nan)
for fd in d['frames']:
    fi = fd['frame']
    if not fd['persons']: continue
    kps = fd['persons'][0]['keypoints']
    lw = kps['left_wrist']; rw = kps['right_wrist']
    xs[fi] = (lw['x']+rw['x'])/2
    ys[fi] = (lw['y']+rw['y'])/2

idx = np.arange(n)
for arr in (xs, ys):
    nans = np.isnan(arr)
    arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])

xs_s = savgol_filter(xs, 7, 3)
ys_s = savgol_filter(ys, 7, 3)

addr_x = xs_s[10]; addr_y = ys_s[10]
print(f"Address wrist: x={addr_x:.1f} y={addr_y:.1f}")
print()
print("Frames 30-60:")
print("  fr | wrist_x  wrist_y | dx_from_addr")
for fi in range(30, 61):
    dx = xs_s[fi] - addr_x
    marker = " <<< impact" if fi == 47 else ""
    print(f"  {fi:2d} | {xs_s[fi]:7.1f}  {ys_s[fi]:7.1f} | dx={dx:+.1f}{marker}")

down_x = xs_s[35:58]
pk, _ = find_peaks(down_x, prominence=3)
print(f"\nLocal X peaks in fr35-58: {[35+p for p in pk]}")
print(f"Global X max in fr35-58: fr{35+int(np.argmax(down_x))} x={xs_s[35+int(np.argmax(down_x))]:.1f}")
