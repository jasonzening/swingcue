#!/usr/bin/env python3
"""
RTMW3D Golf Swing Pose Estimation
- Model: RTMW3D-x (133 wholebody 3D keypoints)
- Backend: ONNX Runtime CUDA GPU
- Outputs: annotated 2D+3D overlay video, keypoint JSON, FPS report
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
OUTPUT_DIR = WORK_DIR / "output" / "rtmw3d"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEOS = [
    "test-faceon.mp4",
    "test-dwontheline.mp4",
]

# RTMW3D uses 133-keypoint wholebody format (body17 + hands + face)
# We focus on body 17 for golf analysis (indices 0-16)
BODY17_INDICES = list(range(17))
BODY17_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
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

def record_nvidia_smi():
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        return result.stdout
    except Exception as e:
        return f"nvidia-smi failed: {e}"

def draw_skeleton_3d(frame, keypoints_2d, keypoints_3d, scores, score_thr=0.3):
    """Draw 2D skeleton + Z-depth indicator on frame."""
    vis = frame.copy()

    # Normalize Z values for color coding (depth visualization)
    body_kps_3d = keypoints_3d[:17]
    z_vals = body_kps_3d[:, 2]
    z_valid = z_vals[scores[:17] > score_thr]
    if len(z_valid) > 0:
        z_min, z_max = z_valid.min(), z_valid.max()
        z_range = z_max - z_min if z_max > z_min else 1.0
    else:
        z_min, z_range = 0, 1.0

    # Draw skeleton connections using 2D projected coords
    for (i, j) in SKELETON:
        if scores[i] > score_thr and scores[j] > score_thr:
            pt1 = (int(keypoints_2d[i][0]), int(keypoints_2d[i][1]))
            pt2 = (int(keypoints_2d[j][0]), int(keypoints_2d[j][1]))
            cv2.line(vis, pt1, pt2, (0, 255, 0), 2)

    # Draw keypoints with Z-depth color coding
    for idx in BODY17_INDICES:
        if scores[idx] > score_thr:
            x, y = int(keypoints_2d[idx][0]), int(keypoints_2d[idx][1])
            z = keypoints_3d[idx][2]
            # Blue=near, Red=far
            t = (z - z_min) / z_range
            b = int(255 * (1 - t))
            r = int(255 * t)
            cv2.circle(vis, (x, y), 5, (b, 50, r), -1)
            cv2.circle(vis, (x, y), 5, (255, 255, 255), 1)

    return vis

def draw_3d_skeleton_panel(keypoints_3d, scores, panel_size=(300, 400), score_thr=0.3):
    """Draw a top-down 3D skeleton view as a side panel."""
    panel = np.zeros((panel_size[1], panel_size[0], 3), dtype=np.uint8)
    panel[:] = (20, 20, 40)  # dark background

    # Use XZ plane (top-down view) for golf swing analysis
    body_kps = keypoints_3d[:17]
    valid_mask = scores[:17] > score_thr

    if valid_mask.sum() < 3:
        cv2.putText(panel, "No detection", (10, panel_size[1]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        return panel

    # Normalize XZ to panel coords
    x_vals = body_kps[valid_mask, 0]
    z_vals = body_kps[valid_mask, 2]

    x_min, x_max = x_vals.min() - 0.2, x_vals.max() + 0.2
    z_min, z_max = z_vals.min() - 0.2, z_vals.max() + 0.2

    def to_panel(x, z):
        px = int((x - x_min) / (x_max - x_min + 1e-6) * (panel_size[0] - 40) + 20)
        pz = int((z - z_min) / (z_max - z_min + 1e-6) * (panel_size[1] - 60) + 30)
        return (px, pz)

    # Draw skeleton in XZ plane
    for (i, j) in SKELETON:
        if scores[i] > score_thr and scores[j] > score_thr:
            pt1 = to_panel(body_kps[i, 0], body_kps[i, 2])
            pt2 = to_panel(body_kps[j, 0], body_kps[j, 2])
            cv2.line(panel, pt1, pt2, (0, 200, 0), 2)

    for idx in BODY17_INDICES:
        if scores[idx] > score_thr:
            pt = to_panel(body_kps[idx, 0], body_kps[idx, 2])
            cv2.circle(panel, pt, 4, (0, 255, 255), -1)

    cv2.putText(panel, "Top-down (X-Z)", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    return panel

def process_video(video_name, wb3d_estimator):
    """Run RTMW3D on one video."""
    input_path = INPUT_DIR / video_name
    stem = Path(video_name).stem
    output_video = OUTPUT_DIR / f"{stem}_rtmw3d.mp4"
    output_json = OUTPUT_DIR / f"{stem}_keypoints3d.json"

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_src = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    panel_w = 300
    out_w = width + panel_w

    print(f"\n{'='*60}")
    print(f"Video: {video_name}")
    print(f"  {width}x{height} @ {fps_src:.1f}fps, {total_frames} frames")
    print(f"  Output: {out_w}x{height} (main + 3D top-down panel)")
    print(f"{'='*60}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_video), fourcc, fps_src, (out_w, height))

    all_keypoints = []
    frame_times = []
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc=f"RTMW3D {stem}", unit="fr")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        keypoints_3d, scores, keypoints_simcc, keypoints_2d = wb3d_estimator(frame)
        t1 = time.perf_counter()
        frame_times.append(t1 - t0)

        vis_frame = frame.copy()
        panel = np.zeros((height, panel_w, 3), dtype=np.uint8)
        panel[:] = (20, 20, 40)

        if keypoints_3d is not None and len(keypoints_3d) > 0:
            # Use first person (golfer)
            kps3d = keypoints_3d[0]   # (133, 3)
            kps2d = keypoints_2d[0]   # (133, 2)
            sc = scores[0]             # (133,)

            vis_frame = draw_skeleton_3d(frame, kps2d, kps3d, sc)

            # Draw 3D panel (top-down view)
            panel_content = draw_3d_skeleton_panel(kps3d, sc, panel_size=(panel_w, height))
            panel = panel_content

        avg_fps = 1.0 / np.mean(frame_times[-30:]) if frame_times else 0
        cv2.putText(vis_frame, f"RTMW3D | Frame {frame_idx} | FPS: {avg_fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        cv2.putText(vis_frame, "3D depth: Blue=near Red=far",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 1)

        # Combine main + panel
        combined = np.zeros((height, out_w, 3), dtype=np.uint8)
        combined[:, :width] = vis_frame
        combined[:, width:] = panel
        writer.write(combined)

        # Store keypoints
        frame_data = {
            "frame": frame_idx,
            "timestamp_ms": round(frame_idx / fps_src * 1000, 1),
            "persons": []
        }
        if keypoints_3d is not None and len(keypoints_3d) > 0:
            for pi in range(len(keypoints_3d)):
                kps3d = keypoints_3d[pi]
                kps2d = keypoints_2d[pi]
                sc = scores[pi]
                person_entry = {
                    "person_id": pi,
                    "body17_3d": {},
                    "body17_2d": {},
                }
                for ki, name in enumerate(BODY17_NAMES):
                    person_entry["body17_3d"][name] = {
                        "x": round(float(kps3d[ki][0]), 4),
                        "y": round(float(kps3d[ki][1]), 4),
                        "z": round(float(kps3d[ki][2]), 4),
                        "score": round(float(sc[ki]), 4),
                    }
                    person_entry["body17_2d"][name] = {
                        "x": round(float(kps2d[ki][0]), 2),
                        "y": round(float(kps2d[ki][1]), 2),
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
            "min": round(np.min(frame_times) * 1000, 2),
            "max": round(np.max(frame_times) * 1000, 2),
        },
        "output_video": str(output_video),
        "output_json": str(output_json),
    }

    output = {
        "model": "RTMW3D-x (rtmlib, ONNX GPU)",
        "keypoint_format": "133-wholebody (body17 stored)",
        "stats": stats,
        "frames": all_keypoints
    }
    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Inference FPS: {avg_fps_val:.1f}")
    print(f"  Latency: mean={np.mean(frame_times)*1000:.1f}ms, p95={p95:.1f}ms")
    print(f"  Output video: {output_video}")
    print(f"  Output JSON:  {output_json}")

    return stats

def main():
    print("=" * 60)
    print("RTMW3D Golf Swing Benchmark")
    print("=" * 60)

    smi_before = record_nvidia_smi()
    print("\n[GPU State - Before]\n" + smi_before)

    smi_log_path = OUTPUT_DIR / "nvidia_smi.txt"

    print("\nInitializing RTMW3D (balanced mode, ONNX GPU)...")
    print("  Det model: YOLOX-m")
    print("  Pose model: RTMW3D-x 384x288 (133 wholebody 3D keypoints)")
    print("  Downloading weights if not cached (RTMW3D-x ~200MB from HuggingFace)...")

    from rtmlib import Wholebody3d

    wb3d = Wholebody3d(
        mode='balanced',
        backend='onnxruntime',
        device='cuda',
    )
    print("  RTMW3D initialized OK")

    smi_after_load = record_nvidia_smi()
    print("\n[GPU State - After Model Load]\n" + smi_after_load)

    all_stats = []
    for video_name in VIDEOS:
        stats = process_video(video_name, wb3d)
        all_stats.append(stats)

    smi_after = record_nvidia_smi()
    print("\n[GPU State - After Inference]\n" + smi_after)

    smi_content = f"""RTMW3D GPU Usage Log
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

    report_path = OUTPUT_DIR / "report.txt"
    report = f"""RTMW3D Benchmark Report
{'='*60}
Model: RTMW3D-x + YOLOX-m (balanced mode)
Backend: ONNX Runtime / CUDA GPU
Device: RTX 4060 Ti 8GB
Keypoints: 133 wholebody 3D (body17 extracted for analysis)

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
    print("Done!")

if __name__ == "__main__":
    main()
