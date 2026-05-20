"""
mediapipe_pose.py — production-mirror runner.

Replicates python/analyzer.py + python/pose_timeline.py's per-frame
extraction so the benchmark baseline matches what the live pipeline
produces. If this runner's keypoints diverge from a stored
pose_timeline_2d row for the same video, the divergence is purely from
sampling/timestamp differences, not algorithmic drift.

Confidence: HIGH. Production has shipped on this exact code path since
PR-4 (commit a0bdaba). API is stable, model_complexity=1 is what
production uses.

CLI:
    python -m benchmark.runners.mediapipe_pose <video_path> <video_id> [--out-dir output]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp

from ..runner import (
    COCO_NAMES,
    FrameKeypoints,
    RunResult,
    Runner,
    empty_keypoints,
    make_kp,
)

# MediaPipe Pose 33-point index → COCO 17 name. Mirrors
# python/pose_timeline.py MEDIAPIPE_TO_COCO_IDX exactly.
MEDIAPIPE_TO_COCO_IDX: dict[str, int] = {
    "nose":          0,
    "left_eye":      2,  "right_eye":      5,
    "left_ear":      7,  "right_ear":      8,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow":    13, "right_elbow":    14,
    "left_wrist":    15, "right_wrist":    16,
    "left_hip":      23, "right_hip":      24,
    "left_knee":     25, "right_knee":     26,
    "left_ankle":    27, "right_ankle":    28,
}


class MediaPipePoseRunner(Runner):
    name = "mediapipe_pose"

    def __init__(self, model_complexity: int = 1):
        # Production uses model_complexity=1 (Full). Pass 2 for Heavy if
        # you want to compare model_complexity within the same library.
        self.model_complexity = model_complexity
        self.pose = None

    def setup(self) -> None:
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def run(
        self,
        video_path: Path,
        video_id: str,
        sample_fps: float = 10.0,
    ) -> RunResult:
        assert self.pose is not None, "call setup() first"
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open {video_path}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = n_total / src_fps if src_fps > 0 else 0.0
        # Stride that approximates sample_fps.
        stride = max(1, round(src_fps / sample_fps))

        frames: list[FrameKeypoints] = []
        idx = 0
        t0 = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.pose.process(rgb)
                kps = empty_keypoints()
                if result.pose_landmarks:
                    for name in COCO_NAMES:
                        lm = result.pose_landmarks.landmark[
                            MEDIAPIPE_TO_COCO_IDX[name]
                        ]
                        kps[name] = make_kp(
                            lm.x, lm.y, lm.visibility, width, height,
                        )
                frames.append(FrameKeypoints(
                    ts=idx / src_fps,
                    frame_idx=idx,
                    keypoints=kps,
                ))
            idx += 1
        elapsed = time.perf_counter() - t0
        cap.release()

        return RunResult(
            video_id=video_id,
            runner=self.name,
            video_width=width,
            video_height=height,
            fps_native=src_fps,
            fps_sampled=src_fps / stride,
            duration_sec=duration,
            frames=frames,
            notes=[
                f"model_complexity={self.model_complexity}",
                f"smooth_landmarks=True",
                f"min_detection_confidence=0.5  min_tracking_confidence=0.5",
                f"native_frames={n_total}  sampled={len(frames)}  "
                f"stride={stride}  elapsed={elapsed:.1f}s",
            ],
        )

    def teardown(self) -> None:
        if self.pose is not None:
            self.pose.close()
            self.pose = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video",    type=Path)
    ap.add_argument("video_id")
    ap.add_argument("--out-dir", type=Path, default=Path("output"))
    ap.add_argument("--sample-fps", type=float, default=10.0)
    ap.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    args = ap.parse_args()

    r = MediaPipePoseRunner(model_complexity=args.model_complexity)
    r.setup()
    try:
        res = r.run(args.video, args.video_id, sample_fps=args.sample_fps)
    finally:
        r.teardown()
    out = args.out_dir / r.name / args.video_id / "keypoints.json"
    res.save(out)
    print(f"[{r.name}] {args.video_id}: {len(res.frames)} frames → {out}")


if __name__ == "__main__":
    sys.exit(main())
