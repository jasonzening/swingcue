#!/usr/bin/env python3
import cv2
import numpy as np
import math
from pathlib import Path

OUT = Path("/home/jason/projects/swingcue-postest/keyframes/preview")
OUT.mkdir(parents=True, exist_ok=True)
DTL_VIDEO = "/home/jason/projects/swingcue-postest/input/test-dwontheline.mp4"

cap = cv2.VideoCapture(DTL_VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, 47)
ret, frame = cap.read(); cap.release()
assert ret

img = frame.copy()
overlay = frame.copy()

# BGR conversions
C_ORANGE = (0x27, 0x9F, 0xEF)   # #EF9F27
C_GREEN  = (0x50, 0xAF, 0x4C)   # #4CAF50
C_RED    = (0x4A, 0x4B, 0xE2)   # #E24B4A
C_WHITE  = (255, 255, 255)

# ── Line 1: orange (397,479)→(448,558)→(430,655)  5px round ──────────────────
pts1 = [(397,479),(448,558),(430,655)]
for i in range(len(pts1)-1):
    cv2.line(overlay, pts1[i], pts1[i+1], C_ORANGE, 5, cv2.LINE_AA)
for p in pts1:
    cv2.circle(overlay, p, 3, C_ORANGE, -1, cv2.LINE_AA)

# ── Line 2: green (397,479)→(404,583)→(414,688)   5px round ──────────────────
pts2 = [(397,479),(404,583),(414,688)]
for i in range(len(pts2)-1):
    cv2.line(overlay, pts2[i], pts2[i+1], C_GREEN, 5, cv2.LINE_AA)
for p in pts2:
    cv2.circle(overlay, p, 3, C_GREEN, -1, cv2.LINE_AA)

# ── Line 3: red arc arrow (448,558)→(404,583)  3px ──────────────────────────
src = (448, 558)
dst = (404, 583)
mx  = (src[0]+dst[0])//2
my  = (src[1]+dst[1])//2
dx  = dst[0]-src[0]; dy = dst[1]-src[1]
n   = math.hypot(dx, dy) or 1
ctrl = (int(mx + (dy/n)*40), int(my - (dx/n)*40))

def bezier(p0, c, p2, steps=40):
    return [(int((1-t)**2*p0[0]+2*(1-t)*t*c[0]+t**2*p2[0]),
             int((1-t)**2*p0[1]+2*(1-t)*t*c[1]+t**2*p2[1]))
            for t in [i/steps for i in range(steps+1)]]

bpts = bezier(src, ctrl, dst)
for i in range(len(bpts)-1):
    cv2.line(overlay, bpts[i], bpts[i+1], C_RED, 3, cv2.LINE_AA)

# Arrowhead at dst
d2  = (bpts[-1][0]-bpts[-4][0], bpts[-1][1]-bpts[-4][1])
nm2 = math.hypot(*d2) or 1
ux, uy = d2[0]/nm2, d2[1]/nm2
head = 16
a = math.radians(25); ca, sa = math.cos(a), math.sin(a)
w1 = (int(dst[0]-head*(ux*ca-uy*sa)), int(dst[1]-head*(uy*ca+ux*sa)))
w2 = (int(dst[0]-head*(ux*ca+uy*sa)), int(dst[1]-head*(uy*ca-ux*sa)))
cv2.fillPoly(overlay, [np.array([dst,w1,w2],np.int32)], C_RED, cv2.LINE_AA)

# ── Blend 90% ────────────────────────────────────────────────────────────────
result = frame.copy()
cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, result)

# ── Label ────────────────────────────────────────────────────────────────────
font = cv2.FONT_HERSHEY_DUPLEX
text = "DEMO  --  (orange) vs  (green)"
cv2.putText(result, text, (14, 36), font, 0.60, (20,20,20), 3, cv2.LINE_AA)
cv2.putText(result, text, (14, 36), font, 0.60, C_WHITE,    1, cv2.LINE_AA)

out = OUT / "demo_overlay_fr47.png"
cv2.imwrite(str(out), result)
print(f"Saved: {out}")

import shutil
desk = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview")
desk.mkdir(parents=True, exist_ok=True)
shutil.copy(out, desk / out.name)
print(f"Desktop: {desk / out.name}")
