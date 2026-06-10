#!/usr/bin/env python3
"""
recheck_impact_faceon.py

For face-on videos: re-run RTMPose on the full video, then find true impact
as the frame where wrist Y is maximum (lowest in frame = closest to ball)
in the window around the original speed-peak detection.

For down-the-line: keep existing logic (anchor-return method).

Outputs: impact frame image with red vertical lines marking wrist x-positions,
so the club shaft position is visible.
"""

import cv2
import numpy as np
import sys, os
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks

sys.path.insert(0, "/home/jason/projects/swingcue-postest/keyframes")

INPUT = Path("/home/jason/projects/swingcue-postest/input")
DESK  = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/impact_recheck")
DESK.mkdir(parents=True, exist_ok=True)

VIDEOS = sorted(INPUT.glob("Videos2026-06-09*.mp4"))

# Known angles (from shoulder-span classifier)
KNOWN_ANGLE = {
    "Videos2026-06-09_201015_827.mp4": "face-on",
    "Videos2026-06-09_201039_231.mp4": "face-on",
    "Videos2026-06-09_201047_915.mp4": "face-on",
    "Videos2026-06-09_201054_561.mp4": "down-the-line",
    "Videos2026-06-09_201058_697.mp4": "down-the-line",
}

# Previous impact detections (to compare)
PREV_IMPACT = {
    "Videos2026-06-09_201015_827.mp4":  59,
    "Videos2026-06-09_201039_231.mp4": 205,
    "Videos2026-06-09_201047_915.mp4": 277,
    "Videos2026-06-09_201054_561.mp4": 150,
    "Videos2026-06-09_201058_697.mp4": 185,
}

JOINT_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]

os.environ["LD_LIBRARY_PATH"] = ":".join([
    "/home/jason/projects/swingcue-postest/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib",
    "/home/jason/projects/swingcue-postest/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib",
    "/home/jason/projects/swingcue-postest/.venv/lib/python3.12/site-packages/nvidia/cublas/lib",
    "/usr/lib/wsl/lib",
]) + ":" + os.environ.get("LD_LIBRARY_PATH", "")


def run_rtmpose(video_path):
    from rtmlib import Body
    body = Body(
        pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip',
        det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip',
        det_input_size=(640, 640), pose_input_size=(288, 384),
        mode='performance', backend='onnxruntime', device='cuda',
    )
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames_kps = []
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        kps, scores = body(frame)
        if kps is not None and len(kps) > 0:
            frames_kps.append({
                "frame": fi,
                "kps": kps[0].tolist(),
                "scores": scores[0].tolist(),
            })
        else:
            frames_kps.append({"frame": fi, "kps": None, "scores": None})
        fi += 1
    cap.release()
    return frames_kps, fps, fi


def wrist_track(frames_kps, n):
    """Extract smoothed wrist midpoint (x, y) arrays."""
    xs = np.full(n, np.nan)
    ys = np.full(n, np.nan)
    for fd in frames_kps:
        fi = fd["frame"]
        if fd["kps"] is None: continue
        kps = fd["kps"]; sc = fd["scores"]
        lw_sc, rw_sc = sc[9], sc[10]
        if max(lw_sc, rw_sc) < 0.35: continue
        w = lw_sc + rw_sc
        xs[fi] = (kps[9][0]*lw_sc + kps[10][0]*rw_sc) / w
        ys[fi] = (kps[9][1]*lw_sc + kps[10][1]*rw_sc) / w
    # interpolate NaN
    idx = np.arange(n)
    for arr in (xs, ys):
        nans = np.isnan(arr)
        if not nans.all():
            arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
    return xs, ys


def sg(arr, w=7, p=3):
    w2 = min(w, len(arr)-1); w2 = w2 if w2%2==1 else w2-1; w2 = max(w2, p+2)
    return savgol_filter(arr, w2, p)


def find_impact_faceon(xs, ys, fps, prev_impact):
    """
    Face-on: wrist Y maximum (closest to ball/ground) in a generous window
    around the speed peak.
    The speed peak identifies the downswing. Impact = where wrist gets
    lowest (Y largest in image coords) in a ±50 frame window.
    """
    n = len(ys)
    win = int(fps * 0.20)   # 200ms smoothing
    ys_s = sg(ys, win)
    dx = np.diff(sg(xs, win), prepend=sg(xs,win)[0])
    dy = np.diff(ys_s, prepend=ys_s[0])
    spd = savgol_filter(np.sqrt(dx**2+dy**2), win, 3)

    # Speed peak in downswing (search from 30% to 85% of video)
    search_start = int(n * 0.30)
    search_end   = int(n * 0.85)
    down_spd = spd[search_start:search_end]
    peaks, props = find_peaks(down_spd, height=np.percentile(spd, 40))
    if len(peaks) == 0:
        spd_peak = search_start + int(np.argmax(down_spd))
    else:
        spd_peak = search_start + peaks[np.argmax(props["peak_heights"])]

    # Search window: spd_peak to spd_peak + 60 frames (impact is AFTER peak)
    i_start = spd_peak
    i_end   = min(n - 1, spd_peak + 60)

    # Find frame where wrist Y is maximum (lowest = closest to ball)
    region_y = ys_s[i_start:i_end+1]
    impact = i_start + int(np.argmax(region_y))

    return impact, spd_peak, ys_s, spd


def find_impact_dtl(xs, ys, fps, frames_kps):
    """
    Down-the-line: use existing anchor-return method (already works).
    Returns previously detected value.
    """
    from detect_keyframes import SwingPhaseDetector

    n = len(ys)
    d_frames = []
    for fd in frames_kps:
        fi = fd["frame"]
        entry = {"frame": fi, "persons": []}
        if fd["kps"] is not None:
            kps = fd["kps"]; sc = fd["scores"]
            kp_dict = {name: {"x": kps[i][0], "y": kps[i][1], "score": sc[i]}
                       for i, name in enumerate(JOINT_NAMES)}
            entry["persons"] = [{"person_id": 0, "keypoints": kp_dict}]
        d_frames.append(entry)

    data = {"frames": d_frames, "stats": {"source_fps": fps}}
    det = SwingPhaseDetector()
    result = det.detect_from_dict(data)
    return result["keyframes"]["impact"]


def annotate_impact_frame(video_path, frame_idx, label, prev_idx, kps_at_frame):
    """Extract frame, draw red lines at wrist x positions, annotate."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read(); cap.release()
    if not ret: return None

    h, w = frame.shape[:2]
    img = frame.copy()
    font = cv2.FONT_HERSHEY_DUPLEX

    # Red vertical lines at left_wrist and right_wrist x positions
    if kps_at_frame and kps_at_frame["kps"] is not None:
        kps = kps_at_frame["kps"]
        sc  = kps_at_frame["scores"]
        for ji, name in [(9,"left_wrist"), (10,"right_wrist")]:
            if sc[ji] > 0.35:
                x = int(kps[ji][0])
                cv2.line(img, (x, 0), (x, h), (0, 0, 220), 2, cv2.LINE_AA)
        # Also draw wrist midpoint horizontal line
        lw_x, lw_y = kps[9][0], kps[9][1]
        rw_x, rw_y = kps[10][0], kps[10][1]
        if sc[9] > 0.35 and sc[10] > 0.35:
            mid_y = int((lw_y + rw_y) / 2)
            cv2.line(img, (0, mid_y), (w, mid_y), (0, 0, 220), 1, cv2.LINE_AA)

    # Header banner
    banner = np.zeros((70, w, 3), dtype=np.uint8)
    banner[:] = (20, 20, 20)
    cv2.putText(banner, f"IMPACT  fr{frame_idx}  [{label}]",
                (10, 28), font, 0.75, (60, 220, 60), 2, cv2.LINE_AA)
    cv2.putText(banner, f"(prev: fr{prev_idx}   red lines = wrist positions)",
                (10, 55), font, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    return np.vstack([banner, img])


def main():
    print("Recheck impact frames for face-on videos\n")

    for vpath in VIDEOS:
        name  = vpath.name
        angle = KNOWN_ANGLE[name]
        stem  = vpath.stem[-14:]
        prev  = PREV_IMPACT[name]

        print(f"\n{'='*55}")
        print(f"{name}  [{angle}]")
        print(f"  Running RTMPose...")

        frames_kps, fps, n = run_rtmpose(vpath)
        xs, ys = wrist_track(frames_kps, n)

        if angle == "face-on":
            impact, spd_peak, ys_s, spd = find_impact_faceon(xs, ys, fps, prev)
            print(f"  spd_peak={spd_peak}  new_impact={impact}  prev={prev}  delta={impact-prev:+d}")
        else:
            impact = find_impact_dtl(xs, ys, fps, frames_kps)
            print(f"  new_impact={impact}  prev={prev}  delta={impact-prev:+d}")

        # Get keypoints at impact frame for annotation
        kps_at = next((fd for fd in frames_kps if fd["frame"] == impact), None)

        annotated = annotate_impact_frame(vpath, impact, angle, prev, kps_at)
        if annotated is not None:
            out = DESK / f"{stem}_impact_NEW_fr{impact}.jpg"
            cv2.imwrite(str(out), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"  Saved: {out.name}")


if __name__ == "__main__":
    main()
