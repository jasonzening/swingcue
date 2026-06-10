import json
import numpy as np
from scipy.signal import savgol_filter, find_peaks

with open('/home/jason/projects/swingcue-postest/output/rtmpose/test-faceon_keypoints.json') as f:
    d = json.load(f)
n = len(d['frames'])
xs = np.full(n, np.nan); ys = np.full(n, np.nan)
for fd in d['frames']:
    fi = fd['frame']
    if not fd['persons']: continue
    kps = fd['persons'][0]['keypoints']
    lw,rw = kps['left_wrist'],kps['right_wrist']
    lsc,rsc = lw['score'],rw['score']; w=lsc+rsc
    xs[fi]=(lw['x']*lsc+rw['x']*rsc)/w; ys[fi]=(lw['y']*lsc+rw['y']*rsc)/w
idx=np.arange(n); nans=np.isnan(xs)
xs[nans]=np.interp(idx[nans],idx[~nans],xs[~nans])
ys[nans]=np.interp(idx[nans],idx[~nans],ys[~nans])
xs_s=savgol_filter(xs,7,3); ys_s=savgol_filter(ys,7,3)
dx=np.diff(xs_s,prepend=xs_s[0]); dy=np.diff(ys_s,prepend=ys_s[0])
spd_s=savgol_filter(np.sqrt(dx**2+dy**2),7,3)

# address=60 → top search region = ys_s[60:104]
address=60; fps=28.0
top_end=int(n*0.75)   # 104
ys_region=ys_s[address:top_end]
print(f"Top search: frames {address}..{top_end-1}")
print(f"  ys in region: min={ys_region.min():.1f} at local idx={ys_region.argmin()}")
peaks_l,props=find_peaks(-ys_region, prominence=30, distance=int(fps*0.25))
print(f"  Peaks with prominence>=30: {peaks_l}  (abs frames: {peaks_l+address})")
print(f"  Prominence values: {props.get('prominences',[])}")

# The problem: y descends monotonically from 673→388 in [60:104]
# No local minimum inside the window → prominence=0
# Right side of window is open → no measured right-base
print(f"\n  ys_s at frame 97-103:")
for i in range(97,104):
    print(f"    fr{i}: y={ys_s[i]:.1f}  spd={spd_s[i]:.1f}")

# Finish analysis
print(f"\nFinish analysis after impact (fr111):")
static_thr=max(np.percentile(spd_s,20)*3.0, 1.0)
settle_thr=static_thr*1.5
settle_win=max(int(fps*0.35),4)
print(f"  static_thr={static_thr:.2f}  settle_thr={settle_thr:.2f}  settle_win={settle_win}")
print(f"  Frame range for settle loop: {111}..{n-settle_win-1}")
print(f"  Speed at last 10 frames:")
for i in range(129,139):
    print(f"    fr{i}: spd={spd_s[i]:.2f} {'<settle' if spd_s[i]<settle_thr else ''}")
