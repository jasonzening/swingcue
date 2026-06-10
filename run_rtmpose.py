#!/usr/bin/env python3
"""
RTMPose Golf Swing Pose Estimation
- Uses rtmlib (ONNX Runtime GPU backend)
- Inputs: input/test-faceon.mp4, input/test-dwontheline.mp4
- Outputs: annotated videos, keypoint JSON, nvidia-smi log, FPS report
"""

import os
import cv2
import json
import time
import subprocess
import numpy as np
from pathlib import Path
from tqdm import tqdm

WORK_DIR = Path(__file__).parent
INPUT_DIR = WORK_DIR / "input"
OUTPUT_DIR = WORK_DIR / "output" / "rtmpose"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEOS = [
    "test-faceon.mp4",
    "test-dwontheline.mp4",
]

# COCO 17-keypoint skeleton connections for drawing
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),   # head
    (5, 6),                             # shoulders
    (5, 7), (7, 9),                     # left arm
    (6, 8), (8, 10),                    # right arm
    (5, 11), (6, 12),                   # torso
    (11, 12),                           # hips
    (11, 13), (13, 15),                 # left leg
    (12, 14), (14, 16),                 # right leg
]

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

def record_nvidia_smi():
    """Capture GPU stats at this moment."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout
    except Exception as e:
        return f"nvidia-smi failed: {e}"

def draw_skeleton(frame, keypoints, scores, score_thr=0.3):
    """Draw keypoints and skeleton on frame."""
    h, w = frame.shape[:2]
    vis = frame.copy()

    # Draw skeleton connections
    for (i, j) in SKELETON:
        if scores[i] > score_thr and scores[j] > score_thr:
            pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
            pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
            cv2.line(vis, pt1, pt2, (0, 255, 0), 2)

    # Draw keypoints
    for idx, (kp, sc) in enumerate(zip(keypoints, scores)):
        if sc > score_thr:
            x, y = int(kp[0]), int(kp[1])
            # Color: head=yellow, upper body=cyan, lower=magenta
            if idx < 5:
                color = (0, 255, 255)
            elif idx < 11:
                color = (255, 255, 0)
            else:
                color = (255, 0, 255)
            cv2.circle(vis, (x, y), 4, color, -1)
            cv2.circle(vis, (x, y), 4, (255, 255, 255), 1)

    return vis

def process_video(video_name, body_estimator):
    """Run RTMPose on one video, return stats."""
    input_path = INPUT_DIR / video_name
    stem = Path(video_name).stem
    output_video = OUTPUT_DIR / f"{stem}_rtmpose.mp4"
    output_json = OUTPUT_DIR / f"{stem}_keypoints.json"

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_src = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n{'='*60}")
    print(f"Video: {video_name}")
    print(f"  {width}x{height} @ {fps_src:.1f}fps, {total_frames} frames")
    print(f"{'='*60}")

    # VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_video), fourcc, fps_src, (width, height))

    all_keypoints = []
    frame_times = []
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc=f"RTMPose {stem}", unit="fr")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        # rtmlib Body returns (keypoints, scores) for all detected persons
        keypoints, scores = body_estimator(frame)
        t1 = time.perf_counter()
        frame_times.append(t1 - t0)

        # Draw on frame
        vis_frame = frame.copy()
        if keypoints is not None and len(keypoints) > 0:
            for person_kps, person_scores in zip(keypoints, scores):
                vis_frame = draw_skeleton(vis_frame, person_kps, person_scores)

        # Overlay info
        avg_fps = 1.0 / np.mean(frame_times[-30:]) if frame_times else 0
        cv2.putText(vis_frame, f"RTMPose | Frame {frame_idx} | FPS: {avg_fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        cv2.putText(vis_frame, f"ONNX GPU | RTX 4060 Ti",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        writer.write(vis_frame)

        # Store keypoints
        frame_data = {
            "frame": frame_idx,
            "timestamp_ms": round(frame_idx / fps_src * 1000, 1),
            "persons": []
        }
        if keypoints is not None and len(keypoints) > 0:
            for pi, (pkps, psc) in enumerate(zip(keypoints, scores)):
                person_entry = {
                    "person_id": pi,
                    "keypoints": {}
                }
                for ki, (kname, (x, y), sc) in enumerate(zip(KEYPOINT_NAMES, pkps, psc)):
                    person_entry["keypoints"][kname] = {
                        "x": round(float(x), 2),
                        "y": round(float(y), 2),
                        "score": round(float(sc), 4)
                    }
                frame_data["persons"].append(person_entry)
        all_keypoints.append(frame_data)

        frame_idx += 1
        pbar.update(1)
        pbar.set_postfix({"fps": f"{avg_fps:.1f}"})

    pbar.close()
    cap.release()
    writer.release()

    # Compute stats
    total_time = sum(frame_times)
    avg_inference_fps = len(frame_times) / total_time if total_time > 0 else 0
    p50 = np.percentile(frame_times, 50) * 1000
    p95 = np.percentile(frame_times, 95) * 1000

    stats = {
        "video": video_name,
        "total_frames": frame_idx,
        "source_fps": fps_src,
        "avg_inference_fps": round(avg_inference_fps, 2),
        "inference_time_ms": {
            "mean": round(np.mean(frame_times) * 1000, 2),
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "min": round(np.min(frame_times) * 1000, 2),
            "max": round(np.max(frame_times) * 1000, 2),
        },
        "output_video": str(output_video),
        "output_json": str(output_json),
    }

    # Save keypoints JSON
    output = {
        "model": "RTMPose-m (rtmlib, ONNX GPU)",
        "keypoint_format": "COCO-17",
        "stats": stats,
        "frames": all_keypoints
    }
    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Inference FPS: {avg_inference_fps:.1f}")
    print(f"  Latency: mean={np.mean(frame_times)*1000:.1f}ms, p95={p95:.1f}ms")
    print(f"  Output video: {output_video}")
    print(f"  Output JSON:  {output_json}")

    return stats

def main():
    print("=" * 60)
    print("RTMPose Golf Swing Benchmark")
    print("=" * 60)

    # Record GPU state before inference
    print("\n[GPU State - Before Inference]")
    smi_before = record_nvidia_smi()
    print(smi_before)

    # Save nvidia-smi output
    smi_log_path = OUTPUT_DIR / "nvidia_smi.txt"

    # Initialize RTMPose Body estimator with GPU
    # mode='performance' = RTMPose-x + YOLOX-x (best accuracy)
    # mode='balanced'   = RTMPose-m + YOLOX-m (good balance)
    print("\nInitializing RTMPose (performance mode, ONNX GPU)...")
    print("  Det model: YOLOX-x (person detector)")
    print("  Pose model: RTMPose-x 384x288")
    print("  Downloading weights if not cached (~100MB)...")

    from rtmlib import Body

    body = Body(
        pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip',
        det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip',
        det_input_size=(640, 640),
        pose_input_size=(288, 384),
        mode='performance',
        backend='onnxruntime',
        device='cuda',  # GPU inference
    )
    print("  RTMPose initialized OK")

    # Record GPU state after model load
    print("\n[GPU State - After Model Load]")
    smi_after_load = record_nvidia_smi()
    print(smi_after_load)

    all_stats = []
    for video_name in VIDEOS:
        stats = process_video(video_name, body)
        all_stats.append(stats)

    # Record GPU state during/after inference
    print("\n[GPU State - After Inference]")
    smi_after = record_nvidia_smi()
    print(smi_after)

    # Save combined nvidia-smi log
    smi_content = f"""RTMPose GPU Usage Log
{'='*60}

[BEFORE INFERENCE]
{smi_before}

[AFTER MODEL LOAD]
{smi_after_load}

[AFTER INFERENCE]
{smi_after}
"""
    with open(smi_log_path, "w") as f:
        f.write(smi_content)

    # Summary report
    report_path = OUTPUT_DIR / "report.txt"
    report = f"""RTMPose Benchmark Report
{'='*60}
Model: RTMPose-x + YOLOX-x (performance mode)
Backend: ONNX Runtime {__import__('onnxruntime').__version__} / CUDA GPU
Device: RTX 4060 Ti 8GB

Results:
"""
    for s in all_stats:
        report += f"""
  Video: {s['video']}
    Frames: {s['total_frames']}
    Source FPS: {s['source_fps']:.1f}
    Inference FPS: {s['avg_inference_fps']:.1f}
    Latency (ms): mean={s['inference_time_ms']['mean']}, p50={s['inference_time_ms']['p50']}, p95={s['inference_time_ms']['p95']}
    Output video: {s['output_video']}
    Keypoint JSON: {s['output_json']}
"""

    print("\n" + report)
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nAll outputs in: {OUTPUT_DIR}")
    print(f"  nvidia-smi log: {smi_log_path}")
    print(f"  report:         {report_path}")
    print("\nDone!")

if __name__ == "__main__":
    main()
