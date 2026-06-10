#!/usr/bin/env python3
"""
hands_forward_impact.py

Impact detection using wrist position geometry (no ball/club needed):

DTL (down-the-line):
  At impact, hands are MOST FORWARD (rightward toward target).
  → find local MAXIMUM of wrist-X in downswing region.
  Verified on confirmed-good fr47 of test-dwontheline (X peak at fr46/47).

Face-on:
  At impact, hands are at their LOWEST point (most toward ground).
  → find local MAXIMUM of wrist-Y (Y increases downward in image) in downswing.

Both: search in the downswing region identified by the speed detector.

Target direction:
  DTL + right-handed golfer: target = screen right (+X direction)
  Face-on + right-handed golfer: impact hand position is slightly LEFT 
    of address (camera-left = target side in face-on view)
    AND at maximum Y (wrist lowest).
"""

import cv2, numpy as np, sys, os, json
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks

sys.path.insert(0, "/home/jason/projects/swingcue-postest/keyframes")

INPUT = Path("/home/jason/projects/swingcue-postest/input")
DESK  = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/impact_recheck")
DESK.mkdir(exist_ok=True)

VIDEOS_INFO = [
    ("Videos2026-06-09_201015_827.mp4", "face-on",       19, 88),   # addr=19 prev_top≈46
    ("Videos2026-06-09_201039_231.mp4", "face-on",       80, 146),
    ("Videos2026-06-09_201047_915.mp4", "face-on",      107, 171),
    ("Videos2026-06-09_201054_561.mp4", "down-the-line",  90, 133),
    ("Videos2026-06-09_201058_697.mp4", "down-the-line",  88, 137),
]
PREV_IMPACT = {
    "Videos2026-06-09_201015_827.mp4":  59,
    "Videos2026-06-09_201039_231.mp4": 205,
    "Videos2026-06-09_201047_915.mp4": 277,
    "Videos2026-06-09_201054_561.mp4": 150,
    "Videos2026-06-09_201058_697.mp4": 185,
}

JOINT_NAMES = ["nose","left_eye","right_eye","left_ear","right_ear",
               "left_shoulder","right_shoulder","left_elbow","right_elbow",
               "left_wrist","right_wrist","left_hip","right_hip",
               "left_knee","right_knee","left_ankle","right_ankle"]

os.environ["LD_LIBRARY_PATH"] = ":".join([
    "/home/jason/projects/swingcue-postest/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib",
    "/home/jason/projects/swingcue-postest/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib",
    "/home/jason/projects/swingcue-postest/.venv/lib/python3.12/site-packages/nvidia/cublas/lib",
    "/usr/lib/wsl/lib",
]) + ":" + os.environ.get("LD_LIBRARY_PATH","")


def sg(arr, w=7, p=3):
    w2 = min(w, len(arr)-1); w2 = w2|1; w2 = max(w2, p+2)
    return savgol_filter(arr, w2, p)


def run_rtmpose(vpath):
    from rtmlib import Body
    body = Body(
        pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip',
        det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip',
        det_input_size=(640,640), pose_input_size=(288,384),
        mode='performance', backend='onnxruntime', device='cuda',
    )
    cap = cv2.VideoCapture(str(vpath))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    data = []
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        kps, sc = body(frame)
        fd = {"frame": fi, "persons": []}
        if kps is not None and len(kps) > 0:
            kp_dict = {name: {"x": float(kps[0][i][0]), "y": float(kps[0][i][1]),
                              "score": float(sc[0][i])}
                       for i, name in enumerate(JOINT_NAMES)}
            fd["persons"] = [{"person_id": 0, "keypoints": kp_dict}]
        data.append(fd)
        fi += 1
    cap.release()
    return data, fps, fi


def extract_wrist_track(data, n):
    xs = np.full(n, np.nan); ys = np.full(n, np.nan)
    for fd in data:
        fi = fd["frame"]
        if not fd["persons"]: continue
        kps = fd["persons"][0]["keypoints"]
        lw = kps["left_wrist"]; rw = kps["right_wrist"]
        sc = max(lw["score"], rw["score"])
        if sc < 0.35: continue
        w = lw["score"] + rw["score"]
        xs[fi] = (lw["x"]*lw["score"] + rw["x"]*rw["score"]) / w
        ys[fi] = (lw["y"]*lw["score"] + rw["y"]*rw["score"]) / w
    idx = np.arange(n)
    for arr in (xs, ys):
        nans = np.isnan(arr)
        if not nans.all():
            arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
    return xs, ys


def torso_height(data, frame_idx):
    fd = data[frame_idx]
    if not fd["persons"]: return 150.0
    kps = fd["persons"][0]["keypoints"]
    sh_y = (kps["left_shoulder"]["y"] + kps["right_shoulder"]["y"]) / 2
    hp_y = (kps["left_hip"]["y"]      + kps["right_hip"]["y"])      / 2
    sh_x = (kps["left_shoulder"]["x"] + kps["right_shoulder"]["x"]) / 2
    hp_x = (kps["left_hip"]["x"]      + kps["right_hip"]["x"])      / 2
    return float(np.hypot(sh_x-hp_x, sh_y-hp_y))


def find_impact_hands_forward(xs, ys, fps, angle, addr_frame, top_frame):
    """
    Core logic:
      DTL:     local MAX of wrist-X after top, in (top → top+80fr)
      Face-on: local MAX of wrist-Y after top, in (top → top+80fr)
      Both:    must be AFTER the top of backswing
    Returns (impact_frame, signal_used, value_at_impact, addr_value)
    """
    n = len(xs)
    win = max(7, int(fps * 0.20)) | 1
    xs_s = sg(xs, win); ys_s = sg(ys, win)

    addr_x = xs_s[addr_frame]; addr_y = ys_s[addr_frame]

    # Search window: top+2 frames to top+90 frames (3 seconds max)
    search_start = top_frame + 2
    search_end   = min(n - 1, top_frame + int(fps * 3.0))

    if angle == "down-the-line":
        # Impact = rightmost wrist X (hands most forward toward target)
        signal   = xs_s
        ref_val  = addr_x
        region   = signal[search_start:search_end]
        peaks, _ = find_peaks(region, prominence=10, distance=int(fps*0.1))
        if len(peaks) == 0:
            impact = search_start + int(np.argmax(region))
        else:
            impact = search_start + peaks[np.argmax(region[peaks])]
        return impact, "wrist_X", xs_s[impact], addr_x

    else:  # face-on
        # Impact = lowest wrist Y (hands at bottom of arc)
        signal   = ys_s
        ref_val  = addr_y
        region   = signal[search_start:search_end]
        peaks, _ = find_peaks(region, prominence=15, distance=int(fps*0.1))
        if len(peaks) == 0:
            impact = search_start + int(np.argmax(region))
        else:
            impact = search_start + peaks[np.argmax(region[peaks])]
        return impact, "wrist_Y", ys_s[impact], addr_y


def annotate_impact(vpath, impact_fi, angle, signal_name, val, addr_val,
                    prev_fi, addr_fi, data):
    cap = cv2.VideoCapture(str(vpath))
    cap.set(cv2.CAP_PROP_POS_FRAMES, impact_fi)
    ret, frame = cap.read(); cap.release()
    if not ret: return None

    out = frame.copy()
    font = cv2.FONT_HERSHEY_DUPLEX

    # Draw wrist midpoint
    fd = data[impact_fi]
    if fd["persons"]:
        kps = fd["persons"][0]["keypoints"]
        lw = kps["left_wrist"]; rw = kps["right_wrist"]
        mx = int((lw["x"]+rw["x"])/2); my = int((lw["y"]+rw["y"])/2)
        cv2.circle(out, (mx, my), 16, (0, 220, 220), 3, cv2.LINE_AA)
        cv2.circle(out, (mx, my), 4,  (0, 220, 220), -1, cv2.LINE_AA)

        # Also draw address wrist for comparison
        fd_addr = data[addr_fi]
        if fd_addr["persons"]:
            akps = fd_addr["persons"][0]["keypoints"]
            ax = int((akps["left_wrist"]["x"]+akps["right_wrist"]["x"])/2)
            ay = int((akps["left_wrist"]["y"]+akps["right_wrist"]["y"])/2)
            cv2.circle(out, (ax, ay), 12, (80, 80, 200), 2, cv2.LINE_AA)
            cv2.line(out, (ax, ay), (mx, my), (150, 150, 150), 1, cv2.LINE_AA)

    delta = val - addr_val
    banner = np.zeros((80, out.shape[1], 3), np.uint8); banner[:] = (20,20,20)
    cv2.putText(banner, f"IMPACT fr{impact_fi}  [{angle}]",
                (10, 28), font, 0.75, (60,220,60), 2)
    cv2.putText(banner,
                f"method: {signal_name} peak={val:.0f}  addr={addr_val:.0f}  d={delta:+.0f}  (prev fr{prev_fi})",
                (10, 58), font, 0.52, (180,180,180), 1)
    cv2.putText(banner,
                "cyan=impact wrist  blue-ring=address wrist",
                (10, 74), font, 0.44, (140,140,140), 1)
    return np.vstack([banner, out])


def main():
    print("Hands-forward impact detection\n")
    for fname, angle, addr_fr, top_fr in VIDEOS_INFO:
        vpath = INPUT / fname
        stem  = vpath.stem[-14:]
        prev  = PREV_IMPACT[fname]
        print(f"\n{'='*55}")
        print(f"{fname}  [{angle}]  addr=fr{addr_fr}  top≈fr{top_fr}")
        print("  Running RTMPose...")

        data, fps, n = run_rtmpose(vpath)
        xs, ys = extract_wrist_track(data, n)

        impact, sig, val, addr_val = find_impact_hands_forward(
            xs, ys, fps, angle, addr_fr, top_fr)

        th = torso_height(data, addr_fr)
        delta = val - addr_val
        print(f"  torso_h={th:.0f}px  addr_{sig}={addr_val:.0f}")
        print(f"  impact fr{impact}: {sig}={val:.0f}  delta={delta:+.0f}px ({delta/th:+.2f} torso)  prev=fr{prev}  shift={impact-prev:+d}")

        annotated = annotate_impact(vpath, impact, angle, sig, val, addr_val,
                                     prev, addr_fr, data)
        if annotated is not None:
            out = DESK / f"{stem}_impact_HANDS_fr{impact}.jpg"
            cv2.imwrite(str(out), annotated, [cv2.IMWRITE_JPEG_QUALITY, 93])
            print(f"  Saved: {out.name}")


if __name__ == "__main__":
    main()
