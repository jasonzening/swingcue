#!/usr/bin/env python3
"""
ViTPose Golf Swing Pose Estimation
- Model: ViTPose-l-coco (COCO 17 keypoints, 2D)
- Backend: ONNX Runtime CUDA GPU
- Det: YOLOX-x (reuse from RTMPose run)
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
OUTPUT_DIR = WORK_DIR / "output" / "vitpose"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEOS = [
    "test-faceon.mp4",
    "test-dwontheline.mp4",
]

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

def record_nvidia_smi():
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        return result.stdout
    except Exception as e:
        return f"nvidia-smi failed: {e}"

def draw_skeleton(frame, keypoints, scores, score_thr=0.3, color_main=(0, 200, 255)):
    """Draw keypoints and skeleton. ViTPose uses cyan color to distinguish from RTMPose."""
    vis = frame.copy()
    for (i, j) in SKELETON:
        if scores[i] > score_thr and scores[j] > score_thr:
            pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
            pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
            cv2.line(vis, pt1, pt2, color_main, 2)
    for idx, (kp, sc) in enumerate(zip(keypoints, scores)):
        if sc > score_thr:
            x, y = int(kp[0]), int(kp[1])
            if idx < 5:
                color = (0, 255, 255)
            elif idx < 11:
                color = (255, 200, 0)
            else:
                color = (200, 0, 255)
            cv2.circle(vis, (x, y), 5, color, -1)
            cv2.circle(vis, (x, y), 5, (255, 255, 255), 1)
    return vis

def process_video(video_name, det_model, pose_model):
    """Run ViTPose on one video."""
    input_path = INPUT_DIR / video_name
    stem = Path(video_name).stem
    output_video = OUTPUT_DIR / f"{stem}_vitpose.mp4"
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

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_video), fourcc, fps_src, (width, height))

    all_keypoints = []
    frame_times = []
    det_times = []
    pose_times = []
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc=f"ViTPose {stem}", unit="fr")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        bboxes = det_model(frame)
        t1 = time.perf_counter()
        keypoints, scores = pose_model(frame, bboxes=bboxes)
        t2 = time.perf_counter()

        frame_times.append(t2 - t0)
        det_times.append(t1 - t0)
        pose_times.append(t2 - t1)

        vis_frame = frame.copy()
        if keypoints is not None and len(keypoints) > 0:
            for person_kps, person_scores in zip(keypoints, scores):
                vis_frame = draw_skeleton(vis_frame, person_kps, person_scores)

        avg_fps = 1.0 / np.mean(frame_times[-30:]) if frame_times else 0
        cv2.putText(vis_frame, f"ViTPose-L | Frame {frame_idx} | FPS: {avg_fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.putText(vis_frame, "ONNX GPU | RTX 4060 Ti",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        writer.write(vis_frame)

        frame_data = {
            "frame": frame_idx,
            "timestamp_ms": round(frame_idx / fps_src * 1000, 1),
            "persons": []
        }
        if keypoints is not None and len(keypoints) > 0:
            for pi, (pkps, psc) in enumerate(zip(keypoints, scores)):
                person_entry = {"person_id": pi, "keypoints": {}}
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

    total_time = sum(frame_times)
    avg_fps_val = len(frame_times) / total_time if total_time > 0 else 0
    p50 = np.percentile(frame_times, 50) * 1000
    p95 = np.percentile(frame_times, 95) * 1000

    stats = {
        "video": video_name,
        "total_frames": frame_idx,
        "source_fps": fps_src,
        "avg_inference_fps": round(avg_fps_val, 2),
        "inference_time_ms": {
            "mean": round(np.mean(frame_times) * 1000, 2),
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "det_mean": round(np.mean(det_times) * 1000, 2),
            "pose_mean": round(np.mean(pose_times) * 1000, 2),
        },
        "output_video": str(output_video),
        "output_json": str(output_json),
    }

    output = {
        "model": "ViTPose-L-coco (rtmlib, ONNX GPU)",
        "keypoint_format": "COCO-17",
        "stats": stats,
        "frames": all_keypoints
    }
    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Inference FPS: {avg_fps_val:.1f}")
    print(f"  Latency: mean={np.mean(frame_times)*1000:.1f}ms (det={np.mean(det_times)*1000:.1f}ms + pose={np.mean(pose_times)*1000:.1f}ms), p95={p95:.1f}ms")
    return stats

def main():
    print("=" * 60)
    print("ViTPose Golf Swing Benchmark")
    print("=" * 60)

    smi_before = record_nvidia_smi()
    print("\n[GPU State - Before]\n" + smi_before)
    smi_log_path = OUTPUT_DIR / "nvidia_smi.txt"

    print("\nInitializing ViTPose-L (ONNX GPU)...")
    print("  Det: YOLOX-x (cached from RTMPose run)")
    print("  Pose: ViTPose-L-coco (~347MB, downloading from HuggingFace)...")

    from rtmlib.tools.object_detection.yolox import YOLOX
    from rtmlib.tools.pose_estimation.vitpose import ViTPose

    # Reuse YOLOX-x detector (already cached)
    det = YOLOX(
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip',
        model_input_size=(640, 640),
        backend='onnxruntime',
        device='cuda',
    )

    # ViTPose-L on COCO 17 keypoints
    pose = ViTPose(
        'https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/coco/vitpose-l-coco.onnx',
        model_input_size=(192, 256),
        backend='onnxruntime',
        device='cuda',
    )
    print("  ViTPose-L initialized OK")

    smi_after_load = record_nvidia_smi()
    print("\n[GPU State - After Load]\n" + smi_after_load)

    all_stats = []
    for video_name in VIDEOS:
        stats = process_video(video_name, det, pose)
        all_stats.append(stats)

    smi_after = record_nvidia_smi()
    print("\n[GPU State - After Inference]\n" + smi_after)

    smi_content = f"""ViTPose GPU Usage Log
{'='*60}
[BEFORE]
{smi_before}
[AFTER MODEL LOAD]
{smi_after_load}
[AFTER INFERENCE]
{smi_after}
"""
    with open(smi_log_path, "w") as f:
        f.write(smi_content)

    report_path = OUTPUT_DIR / "report.txt"
    report = f"""ViTPose Benchmark Report
{'='*60}
Model: ViTPose-L-coco + YOLOX-x
Backend: ONNX Runtime / CUDA GPU
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
    Det latency: {s['inference_time_ms']['det_mean']}ms
    Pose latency: {s['inference_time_ms']['pose_mean']}ms
    Output video: {s['output_video']}
    Keypoint JSON: {s['output_json']}
"""

    print("\n" + report)
    with open(report_path, "w") as f:
        f.write(report)

    print(f"All outputs in: {OUTPUT_DIR}")
    print("Done!")

if __name__ == "__main__":
    main()
