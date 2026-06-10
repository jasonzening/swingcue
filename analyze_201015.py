import json, numpy as np
from scipy.signal import savgol_filter, find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

with open('/home/jason/projects/swingcue-postest/engine/kp_cache/Videos2026-06-09_201015_827.json') as f:
    d = json.load(f)

n = len(d['frames']); fps = d['stats']['source_fps']
xs = np.full(n, np.nan); ys = np.full(n, np.nan)

for fd in d['frames']:
    fi = fd['frame']
    if not fd['persons']: continue
    kps = fd['persons'][0]['keypoints']
    lw = kps['left_wrist']; rw = kps['right_wrist']
    lsc = lw['score']; rsc = rw['score']
    if max(lsc,rsc) < 0.35: continue
    w = lsc + rsc
    xs[fi] = (lw['x']*lsc + rw['x']*rsc)/w
    ys[fi] = (lw['y']*lsc + rw['y']*rsc)/w

idx = np.arange(n)
for arr in (xs,ys):
    nans = np.isnan(arr)
    arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])

ys_s = savgol_filter(ys, 11, 3)

peaks, props = find_peaks(-ys_s, prominence=40, distance=int(fps*0.5))
print("Wrist Y local minima (swing tops):")
for p in peaks:
    print(f"  fr{p}: y={ys_s[p]:.0f}  prom={props['prominences'][list(peaks).index(p)]:.0f}")

valleys, vprops = find_peaks(ys_s, prominence=30, distance=int(fps*0.5))
print("Wrist Y local maxima (impact candidates):")
for v in valleys:
    print(f"  fr{v}: y={ys_s[v]:.0f}  prom={vprops['prominences'][list(valleys).index(v)]:.0f}")

# Plot
fig, axes = plt.subplots(2,1, figsize=(16,8), sharex=True)
frames = np.arange(n)
ax0 = axes[0]
ax0.plot(frames, ys, alpha=0.3, color='gray', label='raw y')
ax0.plot(frames, ys_s, color='blue', lw=1.5, label='smooth y')
ax0.invert_yaxis()
ax0.set_ylabel('Wrist Y (inverted)')
ax0.set_title('201015_827 — Wrist Y trajectory (inverted: up=higher in frame = backswing)')
for p in peaks:
    ax0.axvline(p, color='red', lw=1.5, ls='--')
    ax0.text(p+1, ax0.get_ylim()[1]*0.98, f'top fr{p}', color='red', fontsize=7)
for v in valleys:
    ax0.axvline(v, color='green', lw=1.5, ls='--')
    ax0.text(v+1, ax0.get_ylim()[0]*0.98, f'impact fr{v}', color='green', fontsize=7)
ax0.legend()

ax1 = axes[1]
ax1.plot(frames, xs, alpha=0.3, color='gray', label='raw x')
xs_s = savgol_filter(xs, 11, 3)
ax1.plot(frames, xs_s, color='purple', lw=1.5, label='smooth x')
ax1.set_ylabel('Wrist X')
ax1.set_xlabel('Frame')
ax1.set_title('Wrist X (rightward motion = toward target in DTL, or follow-through in face-on)')
ax1.legend()

plt.tight_layout()
out = '/mnt/c/Users/jason/Desktop/rtmpose_results/preview/gate1/201015_wrist_curve.png'
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
