#!/usr/bin/env python3
"""
generate_final_impact_frames.py
Run ball+club detection for all 5 videos, output final impact frames
with ball position (red circle) and shaft (cyan line) annotated.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, "/home/jason/projects/swingcue-postest/keyframes")

INPUT = Path("/home/jason/projects/swingcue-postest/input")
DESK  = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/impact_recheck")
DESK.mkdir(exist_ok=True)

VIDEOS_INFO = [
    # (filename, angle, address_fr, search_windows [(start,end), ...])
    ("Videos2026-06-09_201015_827.mp4", "face-on",       19, [(40,90),(240,290)]),
    ("Videos2026-06-09_201039_231.mp4", "face-on",       80, [(180,230)]),
    ("Videos2026-06-09_201047_915.mp4", "face-on",      107, [(255,300)]),
    ("Videos2026-06-09_201054_561.mp4", "down-the-line",  90, [(135,165)]),
    ("Videos2026-06-09_201058_697.mp4", "down-the-line",  88, [(175,200)]),
]

PREV_IMPACT = {
    "Videos2026-06-09_201015_827.mp4":  59,
    "Videos2026-06-09_201039_231.mp4": 205,
    "Videos2026-06-09_201047_915.mp4": 277,
    "Videos2026-06-09_201054_561.mp4": 150,
    "Videos2026-06-09_201058_697.mp4": 185,
}


def get_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, f = cap.read()
    return f if ret else None


def find_ball(frame):
    """Find bright white ball in bottom 40% of frame."""
    h, w = frame.shape[:2]
    y0 = int(h * 0.55)
    crop = frame[y0:, :]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=20,
        param1=80, param2=18, minRadius=7, maxRadius=35
    )
    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        best = None; best_b = 0
        for (x, y, r) in circles:
            roi = gray[max(0,y-r):y+r, max(0,x-r):x+r]
            b = float(roi.mean()) if roi.size > 0 else 0
            if b > best_b:
                best_b = b; best = (int(x), y0+int(y), int(r))
        return best

    # Fallback: largest bright blob
    _, bright = cv2.threshold(gray, 195, 255, cv2.THRESH_BINARY)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [(c, cv2.contourArea(c)) for c in cnts if 80 < cv2.contourArea(c) < 3000]
    if valid:
        c = max(valid, key=lambda x: x[1])[0]
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
            r  = int(np.sqrt(cv2.contourArea(c)/np.pi))
            return (cx, y0+cy, r)
    return None


def find_shaft_lines(frame, prev_frame):
    """Return top HoughLine segments on motion edges."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_frame is not None:
        pg = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, pg)
        _, mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        inp = cv2.bitwise_and(gray, gray, mask=mask)
    else:
        inp = gray
    edges = cv2.Canny(cv2.GaussianBlur(inp,(3,3),0), 25, 90)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=35,
                             minLineLength=70, maxLineGap=25)
    if lines is None: return []
    result = [(l[0][0],l[0][1],l[0][2],l[0][3],
               np.hypot(l[0][2]-l[0][0],l[0][3]-l[0][1])) for l in lines]
    return sorted(result, key=lambda x: -x[4])[:3]


def scan_window(cap, ball_xy, window_start, window_end):
    """Find frame in window where shaft far end is closest to ball."""
    bx, by = ball_xy
    results = []
    prev = None
    for fi in range(window_start, window_end+1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret: break
        lines = find_shaft_lines(frame, prev)
        prev = frame.copy()
        if not lines:
            results.append((fi, 9999, None, None)); continue
        x1,y1,x2,y2,_ = lines[0]
        # Far end from hand-height (mid of frame)
        mid_h = frame.shape[0]//2
        head = (x2,y2) if abs(y1-mid_h) < abs(y2-mid_h) else (x1,y1)
        dist = float(np.hypot(head[0]-bx, head[1]-by))
        results.append((fi, dist, (x1,y1,x2,y2), head))
    if not results: return None, 9999, []
    best = min(results, key=lambda r: r[1])
    return best[0], best[1], results


def annotate_frame(frame, ball, shaft_line, head_pt, fi, label, prev_fi):
    out = frame.copy()
    font = cv2.FONT_HERSHEY_DUPLEX
    if ball:
        bx, by, br = ball
        cv2.circle(out, (bx,by), br+5, (0,0,220), 3, cv2.LINE_AA)
        cv2.circle(out, (bx,by), 3, (0,0,220), -1)
    if shaft_line:
        x1,y1,x2,y2 = shaft_line
        cv2.line(out, (x1,y1),(x2,y2), (0,220,220), 3, cv2.LINE_AA)
    if head_pt:
        cv2.circle(out, head_pt, 10, (220,60,220), -1, cv2.LINE_AA)
        cv2.circle(out, head_pt, 10, (255,255,255), 2, cv2.LINE_AA)
    # Banner
    banner = np.zeros((65, out.shape[1], 3), np.uint8); banner[:] = (20,20,20)
    cv2.putText(banner, f"IMPACT fr{fi}  [{label}]",
                (10,28), font, 0.75, (60,220,60), 2)
    cv2.putText(banner, f"prev=fr{prev_fi}   red=ball  cyan=shaft  magenta=head",
                (10,55), font, 0.50, (180,180,180), 1)
    return np.vstack([banner, out])


def main():
    print("Generating final impact frames with ball+club detection\n")

    for fname, angle, addr_fr, windows in VIDEOS_INFO:
        vpath = INPUT / fname
        stem  = vpath.stem[-14:]
        prev  = PREV_IMPACT[fname]
        print(f"\n{fname}  [{angle}]")

        cap = cv2.VideoCapture(str(vpath))

        # Find ball in address frame
        addr = get_frame(cap, addr_fr)
        ball = find_ball(addr)
        if ball:
            print(f"  Ball: ({ball[0]},{ball[1]}) r={ball[2]}")
        else:
            print(f"  Ball: NOT FOUND — using center-bottom estimate")
            h,w = addr.shape[:2]; ball = (w//2, int(h*0.85), 15)

        bx, by = ball[0], ball[1]

        # Scan all windows, pick overall best
        best_fi = prev; best_dist = 9999; best_line = None; best_head = None
        for ws, we in windows:
            fi, dist, results = scan_window(cap, (bx,by), ws, we)
            if dist < best_dist:
                best_dist = dist; best_fi = fi
                # Get line/head for best fi
                for r in results:
                    if r[0] == fi:
                        best_line = r[2]; best_head = r[3]; break

        print(f"  New impact: fr{best_fi}  dist={best_dist:.0f}px  (prev=fr{prev}  delta={best_fi-prev:+d})")

        # Annotate frame
        frame = get_frame(cap, best_fi)
        cap.release()
        if frame is None: continue

        annotated = annotate_frame(frame, ball, best_line, best_head,
                                   best_fi, angle, prev)
        out = DESK / f"{stem}_impact_FINAL_fr{best_fi}.jpg"
        cv2.imwrite(str(out), annotated, [cv2.IMWRITE_JPEG_QUALITY, 93])
        print(f"  Saved: {out.name}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
