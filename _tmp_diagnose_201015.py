"""Diagnose 201015 anchor detection step by step."""
import json, numpy as np, sys
sys.path.insert(0, ".")
from scipy.signal import savgol_filter, find_peaks
from engine.a_measurement.pose_pipeline import PosePipeline

d = json.load(open("engine/kp_cache/Videos2026-06-09_201015_827.json"))
pipe = PosePipeline(device="cpu")
meas, fps = pipe.run_from_json(d)
n = len(meas)

# Reproduce B-layer signals
xs = np.full(n, np.nan); ys = np.full(n, np.nan)
for m in meas:
    fi = m.frame_idx
    wm = m.wrist_mid()
    if wm: xs[fi], ys[fi] = wm
idx = np.arange(n)
for arr in (xs, ys):
    nans = np.isnan(arr)
    arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
win = max(7, int(fps*200/1000)) | 1
xs_s = savgol_filter(xs, win, 3)
ys_s = savgol_filter(ys, win, 3)
dx = np.diff(xs_s, prepend=xs_s[0])
dy = np.diff(ys_s, prepend=ys_s[0])
spd = savgol_filter(np.sqrt(dx**2+dy**2), win, 3)

# --- swing detection already limits to first_swing_end=133 ---
n_eff = 133
speed_thr_swing = max(float(np.percentile(spd[:n_eff], 60)), 25.0)
print(f"n_eff={n_eff}  speed_thr(p60)={speed_thr_swing:.1f} px/fr")

# --- address ---
static_thr = max(np.percentile(spd[:n_eff], 20) * 3.0, 1.0)
address = 2
for i in range(2, int(n_eff * 0.50)):
    if spd[i] < static_thr: address = i
    elif spd[i] > static_thr * 3 and i > address + 5: break
print(f"address=fr{address}  static_thr={static_thr:.2f}")

# --- top ---
top_end = int(n_eff * 0.82)  # = 109
ys_region = ys_s[address:top_end]
peaks_l, pp = find_peaks(-ys_region, prominence=30, distance=int(fps*0.25))
if len(peaks_l) == 0:
    top = address + int(np.argmin(ys_region))
    left_h  = ys_region[0]  - ys_region.min()
    right_h = ys_region[-1] - ys_region.min()
    top_prom = float(min(left_h, right_h))
    print(f"top=fr{top}  (FALLBACK, no prominent peak found in [{address},{top_end}])")
else:
    top = address + peaks_l[0]
    top_prom = float(pp["prominences"][0])
    print(f"top=fr{top}  prom={top_prom:.1f}  (from find_peaks)")

print(f"top_end searched up to fr{top_end}")
print()

# Show wrist-Y in the search region to understand the landscape
print("Wrist-Y landscape in [address, top_end]:")
for fr in range(address, top_end, 3):
    print(f"  fr{fr:3d}: ys={ys_s[fr]:.1f}  spd={spd[fr]:.1f}")

# --- impact search ---
print()
print("Impact search from top+2 onward:")
search_start = top + 2
search_end = min(n_eff-1, top + int(fps*4.5))
cap = ys_s[search_start:search_end+1]
peaks_imp, props_imp = find_peaks(cap, prominence=15, distance=max(int(fps*0.1), 2))
print(f"search window: [{search_start}, {search_end}]  peaks found at: {(peaks_imp + search_start).tolist()}")
for pi, pp_v in zip(peaks_imp, props_imp["prominences"]):
    fr_abs = int(search_start + pi)
    lo = max(0, fr_abs-20)
    ms = spd[lo:fr_abs+1].max()
    print(f"  fr{fr_abs}: prom={pp_v:.1f}  max_spd_before20fr={ms:.1f}")

# Show what speed filter would do
print()
print(f"Speed filter (thr={speed_thr_swing:.1f}): peaks with ms>=thr:")
for pi, pp_v in zip(peaks_imp, props_imp["prominences"]):
    fr_abs = int(search_start + pi)
    lo = max(0, fr_abs-20)
    ms = spd[lo:fr_abs+1].max()
    if ms >= speed_thr_swing:
        print(f"  fr{fr_abs}: PASS (ms={ms:.1f})")
    else:
        print(f"  fr{fr_abs}: FAIL (ms={ms:.1f}) -> would be rejected")

# Check fr59 specifically
print()
print("fr59 analysis:")
lo59 = max(0, 59-20)
ms59 = spd[lo59:60].max()
print(f"  ys_s[59]={ys_s[59]:.1f}  spd[59]={spd[59]:.1f}  max_spd_before20fr={ms59:.1f}")
print(f"  Is fr59 in [top+2, n_eff]? {search_start <= 59 <= n_eff-1}")
print(f"  Is fr59 in [address+2, n_eff]? {address+2 <= 59 <= n_eff-1}")

# Check ordering violation
print()
print("ORDERING CHECK ANALYSIS:")
impact_detected = 132
print(f"  address={address}  top={top}  impact(detected)={impact_detected}  finish=?")
print(f"  address < top: {address < top}  (fr{address} < fr{top})")
print(f"  top < impact:  {top < impact_detected}  (fr{top} < fr{impact_detected})")
print(f"  -> ordering check PASSES even though top({top}) > real_impact(59)")
print(f"  The bug: fr{top} is the FOLLOWTHROUGH high, not backswing top")
print(f"  fr132 is a rest plateau (slow), not an impact — speed filter would have caught this")
