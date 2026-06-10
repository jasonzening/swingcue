import json
import numpy as np
from scipy.signal import savgol_filter

with open('/home/jason/projects/swingcue-postest/output/rtmpose/test-dwontheline_keypoints.json') as f:
    d = json.load(f)

n = len(d['frames'])
xs = np.full(n, np.nan)
ys = np.full(n, np.nan)
raw = {}

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
    raw[fi] = (lsc, rsc)

nans = np.isnan(xs)
idx = np.arange(n)
xs[nans] = np.interp(idx[nans], idx[~nans], xs[~nans])
ys[nans] = np.interp(idx[nans], idx[~nans], ys[~nans])

xs_s = savgol_filter(xs, 7, 3)
ys_s = savgol_filter(ys, 7, 3)
dx = np.diff(xs_s, prepend=xs_s[0])
dy = np.diff(ys_s, prepend=ys_s[0])
spd_s = savgol_filter(np.sqrt(dx**2+dy**2), 7, 3)

# Address anchor (frame 10)
addr_x, addr_y = xs_s[10], ys_s[10]
print(f"Address anchor (frame 10): x={addr_x:.1f}  y={addr_y:.1f}")
print()

# Speed peak in downswing (frames 31-55)
peak_frame = 31 + int(np.argmax(spd_s[31:55]))
print(f"Speed peak: frame {peak_frame}  speed={spd_s[peak_frame]:.1f}px/fr")
print()

# After speed peak: Euclidean distance to address anchor
print("Frame | mid_x  mid_y  | dist_to_addr | speed | lsc  rsc")
for fi in range(peak_frame, min(peak_frame+25, n)):
    lsc, rsc = raw.get(fi, (0, 0))
    dist = np.sqrt((xs_s[fi]-addr_x)**2 + (ys_s[fi]-addr_y)**2)
    print(f"  {fi:3d} | {xs_s[fi]:6.1f} {ys_s[fi]:6.1f} | {dist:8.1f}px      | {spd_s[fi]:5.1f} | {lsc:.2f} {rsc:.2f}")
