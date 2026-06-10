#!/usr/bin/env python3
"""
ball_club_detector.py
Feasibility test: detect ball + club shaft using classical image methods.
Test video: Videos2026-06-09_201015_827.mp4 (face-on, impact detected at fr59 vs fr268)

Steps:
1. Ball detection: on address frame, find small white circle on grass
2. Club shaft: Hough line transform on downswing frames
3. Find frame where club head (far end of shaft) reaches ball position
4. Output annotated image showing ball, shaft, impact candidate
"""

import cv2
import numpy as np
from pathlib import Path

VID   = "/home/jason/projects/swingcue-postest/input/Videos2026-06-09_201015_827.mp4"
OUT   = Path("/home/jason/projects/swingcue-postest/keyframes/ball_club_debug")
DESK  = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/impact_recheck")
OUT.mkdir(exist_ok=True); DESK.mkdir(exist_ok=True)


def get_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, f = cap.read()
    return f if ret else None


# ── Step 1: Find ball in address frame ────────────────────────────────────────
def find_ball(frame, search_region=None):
    """
    Look for a small white/bright circle on grass.
    Uses HoughCircles on a preprocessed crop.
    Returns (x, y, r) or None.
    """
    h, w = frame.shape[:2]
    # Search bottom 40% of frame (ball is on ground)
    y0 = int(h * 0.60)
    crop = frame[y0:, :]

    # Convert to grayscale, enhance contrast
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Ball is bright white on green grass → threshold for bright spots
    _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Morphological clean
    kernel = np.ones((3,3), np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
    
    # HoughCircles: golf ball is ~10-30px radius at this resolution
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.5,
        minDist=20, param1=80, param2=20,
        minRadius=8, maxRadius=35
    )
    
    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        # Filter: only circles in bright regions
        best = None; best_brightness = 0
        for (x, y, r) in circles:
            cx, cy = int(x), int(y)
            roi = gray[max(0,cy-r):cy+r, max(0,cx-r):cx+r]
            if roi.size == 0: continue
            brightness = float(roi.mean())
            if brightness > best_brightness:
                best_brightness = brightness
                best = (cx, y0 + cy, r)
        return best
    
    # Fallback: find brightest blob in bottom region
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Filter by area (golf ball ~200-2000 sq px)
        valid = [(c, cv2.contourArea(c)) for c in contours if 100 < cv2.contourArea(c) < 3000]
        if valid:
            best_c = max(valid, key=lambda x: x[1])[0]
            M = cv2.moments(best_c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                r  = int(np.sqrt(cv2.contourArea(best_c) / np.pi))
                return (cx, y0 + cy, r)
    return None


# ── Step 2: Find club shaft via Hough lines ───────────────────────────────────
def find_club_shaft(frame, prev_frame=None):
    """
    Detect the club shaft as the dominant straight line in the frame.
    Uses frame difference to isolate moving elements (shaft moves fast).
    Returns list of (x1,y1,x2,y2) line segments, sorted by length.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if prev_frame is not None:
        # Motion mask: club moves, background static
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        _, motion = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        # Apply motion mask
        edges_input = cv2.bitwise_and(gray, gray, mask=motion)
    else:
        edges_input = gray

    # Edge detection
    blur = cv2.GaussianBlur(edges_input, (3, 3), 0)
    edges = cv2.Canny(blur, 30, 100)
    
    # Hough lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40,
                             minLineLength=80, maxLineGap=20)
    if lines is None:
        return []
    
    # Return lines sorted by length (longest = club shaft most likely)
    result = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        length = np.hypot(x2-x1, y2-y1)
        result.append((x1, y1, x2, y2, length))
    result.sort(key=lambda x: -x[4])
    return result[:5]  # top 5 longest lines


def shaft_far_end(x1, y1, x2, y2, hand_region):
    """
    Given a shaft line, return the endpoint farther from the hand region.
    hand_region = (cx, cy) of hand midpoint.
    """
    hx, hy = hand_region
    d1 = np.hypot(x1-hx, y1-hy)
    d2 = np.hypot(x2-hx, y2-hy)
    if d1 > d2:
        return (x1, y1)  # point 1 is farther = club head
    else:
        return (x2, y2)


# ── Step 3: Sweep frames to find club head → ball ─────────────────────────────
def find_true_impact(cap, ball_pos, hand_approx_frames, search_start, search_end):
    """
    Scan frames from search_start to search_end.
    For each frame, detect shaft + compute club head position.
    Find frame where club head is closest to ball_pos.
    Returns (best_frame_idx, min_dist, debug_info)
    """
    bx, by, _ = ball_pos
    
    results = []
    prev_frame = None
    
    for fi in range(search_start, search_end + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret: break
        
        lines = find_club_shaft(frame, prev_frame)
        prev_frame = frame.copy()
        
        if not lines:
            results.append((fi, float('inf'), None, None))
            continue
        
        # Use longest line as shaft candidate
        x1, y1, x2, y2, length = lines[0]
        
        # Hand approximate position (upper part of the line, since hands are above ball)
        # In face-on view, hands are roughly at mid-frame height during downswing
        mid_h = frame.shape[0] // 2
        # Point closer to mid-frame = hand end; farther end = club head
        hand_y = mid_h
        if abs(y1 - hand_y) < abs(y2 - hand_y):
            head_pt = (x2, y2)
        else:
            head_pt = (x1, y1)
        
        dist = float(np.hypot(head_pt[0] - bx, head_pt[1] - by))
        results.append((fi, dist, (x1,y1,x2,y2), head_pt))
    
    if not results:
        return None, None, None
    
    # Find minimum distance frame
    best = min(results, key=lambda r: r[1])
    return best[0], best[1], results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cap   = cv2.VideoCapture(VID)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total}fr @ {fps}fps")
    
    # ── 1. Find ball in address frame (fr19) ──────────────────────────────────
    addr_frame = get_frame(cap, 19)
    ball = find_ball(addr_frame)
    if ball:
        bx, by, br = ball
        print(f"Ball found at: ({bx}, {by})  radius={br}px")
    else:
        print("Ball NOT found by HoughCircles - using manual estimate")
        # From visual inspection of similar face-on videos: ball is near center-bottom
        # Will fallback to lowest wrist position heuristic
        bx, by, br = 360, 1050, 15
    
    # ── 2. Scan frames around old impact (fr59) for shaft detection ───────────
    print(f"\nScanning frames fr40-fr90 for shaft + club head...")
    best_fi, min_dist, all_results = find_true_impact(cap, (bx, by, br),
                                                       None, 40, 90)
    
    # Also scan a wider window to check fr268 hypothesis
    print(f"Scanning frames fr240-fr290 for shaft + club head...")
    best_fi2, min_dist2, all_results2 = find_true_impact(cap, (bx, by, br),
                                                          None, 240, 290)
    
    cap.release()
    
    print(f"\nBest impact candidate in fr40-90:   fr{best_fi}  dist={min_dist:.0f}px")
    print(f"Best impact candidate in fr240-290: fr{best_fi2}  dist={min_dist2:.0f}px")
    
    # ── 3. Build annotated comparison image ───────────────────────────────────
    cap = cv2.VideoCapture(VID)
    
    # Left: address frame with ball marker
    cap.set(cv2.CAP_PROP_POS_FRAMES, 19)
    _, addr = cap.read()
    cv2.circle(addr, (bx, by), br+4, (0, 0, 255), 3)
    cv2.circle(addr, (bx, by), 3, (0, 0, 255), -1)
    cv2.putText(addr, f"BALL ({bx},{by})", (bx-60, by-br-10),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,255), 2)
    
    # Middle: best candidate from first window
    cap.set(cv2.CAP_PROP_POS_FRAMES, best_fi)
    _, mid = cap.read()
    # Draw detected shaft
    if all_results:
        for fi, dist, line, head in all_results:
            if fi == best_fi and line:
                x1,y1,x2,y2 = line
                cv2.line(mid, (x1,y1), (x2,y2), (0,255,255), 3)
                if head:
                    cv2.circle(mid, head, 8, (255,0,255), -1)
                    cv2.circle(mid, (bx,by), br+4, (0,0,255), 2)
    cv2.putText(mid, f"CANDIDATE fr{best_fi} dist={min_dist:.0f}",
                (10, 50), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0,255,255), 2)
    
    # Right: best candidate from second window
    cap.set(cv2.CAP_PROP_POS_FRAMES, best_fi2)
    _, right = cap.read()
    if all_results2:
        for fi, dist, line, head in all_results2:
            if fi == best_fi2 and line:
                x1,y1,x2,y2 = line
                cv2.line(right, (x1,y1),(x2,y2), (0,255,255), 3)
                if head:
                    cv2.circle(right, head, 8, (255,0,255), -1)
                    cv2.circle(right, (bx,by), br+4, (0,0,255), 2)
    cv2.putText(right, f"CANDIDATE fr{best_fi2} dist={min_dist2:.0f}",
                (10, 50), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0,255,255), 2)
    
    cap.release()
    
    # Add labels
    for img, lbl in [(addr,"ADDRESS fr19 [ball=red]"),
                     (mid, f"WINDOW1 fr40-90 best=fr{best_fi}"),
                     (right,f"WINDOW2 fr240-290 best=fr{best_fi2}")]:
        banner = np.zeros((50, img.shape[1], 3), np.uint8)
        banner[:] = (20,20,20)
        cv2.putText(banner, lbl, (8,33), cv2.FONT_HERSHEY_DUPLEX, 0.65,
                    (200,230,200), 1)
        img[:] = np.vstack([banner, img[50:]])
    
    # Scale down to 480px wide for the composite
    def resize_w(img, w=480):
        h2 = int(img.shape[0] * w / img.shape[1])
        return cv2.resize(img, (w, h2))
    
    composite = np.hstack([resize_w(addr), resize_w(mid), resize_w(right)])
    
    out = DESK / "201015_ball_club_feasibility.png"
    cv2.imwrite(str(out), composite)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
