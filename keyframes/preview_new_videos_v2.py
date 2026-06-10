#!/usr/bin/env python3
"""
preview_new_videos_v2.py

Fixes:
1. Angle classification: use shoulder x-span relative to frame width.
   Face-on:      both shoulders visible, large horizontal gap (>8% of frame width)
   Down-the-line: one shoulder behind the other, small horizontal gap
   Uses rtmlib Body detector (already cached weights).

2. Keyframe detection: use SwingPhaseDetector on RTMPose keypoints
   (same detector used on old videos), output actual frame indices.
   Also output single full-res frame images for each phase.
"""

import cv2
import numpy as np
import json
import sys
import os
from pathlib import Path

# Add project root so we can import the detector
sys.path.insert(0, "/home/jason/projects/swingcue-postest")
sys.path.insert(0, "/home/jason/projects/swingcue-postest/keyframes")

INPUT = Path("/home/jason/projects/swingcue-postest/input")
OUT   = Path("/home/jason/projects/swingcue-postest/keyframes/new_video_preview")
DESK  = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/new_videos")
OUT.mkdir(parents=True, exist_ok=True)
DESK.mkdir(parents=True, exist_ok=True)

VIDEOS = sorted(INPUT.glob("Videos2026-06-09*.mp4"))
LABELS = ["address", "top", "impact", "finish"]
FONT   = cv2.FONT_HERSHEY_DUPLEX


# ── Angle classifier (shoulder-based) ─────────────────────────────────────────
def classify_angle(video_path: Path) -> str:
    """
    Sample a few mid-swing frames, run lightweight pose detection
    to get shoulder positions, compute horizontal span.
    Face-on:       |l_shoulder_x - r_shoulder_x| / frame_w  > 0.10
    Down-the-line: ratio < 0.10  (shoulders stacked front-back)
    """
    try:
        from rtmlib import Body
        body = Body(
            mode='lightweight',
            backend='onnxruntime',
            device='cpu',   # just angle detection, no need for GPU
        )
    except Exception:
        return "unknown"

    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ratios = []

    for frac in (0.20, 0.35, 0.50):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac))
        ret, frame = cap.read()
        if not ret:
            continue
        try:
            kps, scores = body(frame)
            if kps is None or len(kps) == 0:
                continue
            # COCO: index 5=left_shoulder, 6=right_shoulder
            p = kps[0]
            if scores[0][5] > 0.3 and scores[0][6] > 0.3:
                span = abs(float(p[5][0]) - float(p[6][0])) / W
                ratios.append(span)
        except Exception:
            continue

    cap.release()

    if not ratios:
        return "unknown"

    med = float(np.median(ratios))
    return "face-on" if med > 0.10 else "down-the-line"


# ── Keyframe detection ─────────────────────────────────────────────────────────
def detect_keyframes(video_path: Path):
    """
    Run RTMPose on the video, build a keypoints structure,
    then use SwingPhaseDetector to get address/top/impact/finish.
    Returns dict {"address": N, "top": N, "impact": N, "finish": N}
    and the raw kps data.
    """
    # Lazy import — GPU needed
    os.environ.setdefault("LD_LIBRARY_PATH",
        ":".join([
            "/home/jason/projects/swingcue-postest/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib",
            "/home/jason/projects/swingcue-postest/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib",
            "/home/jason/projects/swingcue-postest/.venv/lib/python3.12/site-packages/nvidia/cublas/lib",
            "/usr/lib/wsl/lib",
        ]) + ":" + os.environ.get("LD_LIBRARY_PATH", ""))

    from rtmlib import Body
    body = Body(
        pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip',
        det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip',
        det_input_size=(640, 640),
        pose_input_size=(288, 384),
        mode='performance',
        backend='onnxruntime',
        device='cuda',
    )

    JOINT_NAMES = [
        "nose","left_eye","right_eye","left_ear","right_ear",
        "left_shoulder","right_shoulder","left_elbow","right_elbow",
        "left_wrist","right_wrist","left_hip","right_hip",
        "left_knee","right_knee","left_ankle","right_ankle"
    ]

    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30

    frames_data = []
    fi = 0
    print(f"  RTMPose inference on {total} frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        kps, scores = body(frame)
        fd = {"frame": fi, "timestamp_ms": round(fi/fps*1000, 1), "persons": []}
        if kps is not None and len(kps) > 0:
            person = {"person_id": 0, "keypoints": {}}
            for ki, name in enumerate(JOINT_NAMES):
                person["keypoints"][name] = {
                    "x": round(float(kps[0][ki][0]), 2),
                    "y": round(float(kps[0][ki][1]), 2),
                    "score": round(float(scores[0][ki]), 4),
                }
            fd["persons"].append(person)
        frames_data.append(fd)
        fi += 1
    cap.release()

    # Build JSON structure for SwingPhaseDetector
    data = {
        "model": "RTMPose-x",
        "keypoint_format": "COCO-17",
        "stats": {"source_fps": fps, "total_frames": fi},
        "frames": frames_data,
    }

    from detect_keyframes import SwingPhaseDetector
    det    = SwingPhaseDetector()
    result = det.detect_from_dict(data)
    return result, data


# ── Frame extractor ────────────────────────────────────────────────────────────
def extract_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, f = cap.read()
    cap.release()
    return f if ret else None


def label_frame(frame, label, frame_idx, color=(40, 220, 55)):
    """Draw a small banner on the frame in-place (copy)."""
    out = frame.copy()
    banner = np.zeros((50, out.shape[1], 3), dtype=np.uint8)
    banner[:] = (20, 20, 20)
    cv2.putText(banner, f"{label.upper()}  fr{frame_idx}",
                (12, 33), FONT, 0.85, color, 2, cv2.LINE_AA)
    return np.vstack([banner, out])


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Processing {len(VIDEOS)} videos...\n")

    for vpath in VIDEOS:
        stem  = vpath.stem[-14:]   # short id
        print(f"\n{'='*60}")
        print(f"{vpath.name}")

        # 1. Angle
        print("  Classifying angle...")
        angle = classify_angle(vpath)
        print(f"  Angle: {angle}")

        # 2. Keyframes
        print("  Running RTMPose + SwingPhaseDetector...")
        result, _ = detect_keyframes(vpath)
        kf = result["keyframes"]
        cf = result["confidence"]
        fps = result["fps"]

        print(f"  Keyframes:")
        for phase in LABELS:
            fr = kf[phase]
            t  = fr / fps * 1000
            print(f"    {phase:10s}: fr{fr:4d}  ({t:.0f}ms)  conf={cf[phase]:.2f}")

        # 3. Extract + save individual full-res frames
        angle_color = (80, 210, 80) if "face" in angle else (80, 180, 255)
        for phase in LABELS:
            fr    = kf[phase]
            frame = extract_frame(vpath, fr)
            if frame is None:
                continue
            labeled = label_frame(frame, f"{phase} [{angle}]", fr, angle_color)
            out_path = DESK / f"{stem}_{phase}_fr{fr}.jpg"
            cv2.imwrite(str(out_path), labeled, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"    Saved: {out_path.name}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
