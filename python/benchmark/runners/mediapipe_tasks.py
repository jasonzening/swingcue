"""
mediapipe_tasks.py — MediaPipe Tasks API (PoseLandmarker, heavy model).

The Tasks API is the newer (post-0.10.x) MediaPipe entry-point. It uses
the same BlazePose family of models as `mp.solutions.pose` but exposes
a different runtime with stricter typing, explicit model files (`.task`
bundles), and a more flexible threading model. Crucially, the
`pose_landmarker_heavy.task` model is *roughly* equivalent in accuracy
to `mp.solutions.pose model_complexity=2`, which production has never
tried — so this is the most promising "free upgrade" candidate.

Confidence: MEDIUM. The Tasks API has shifted naming a few times across
MediaPipe 0.10.x patch releases. Specific risk hot-spots are flagged
inline with `# TODO(jason)` comments. Most likely failure modes:

  1. Model file path — must be downloaded once (see download in setup()).
     If the URL changes, edit MODEL_URL.
  2. `detect_for_video(mp_image, timestamp_ms)` is the documented entry
     in VIDEO running mode; if MediaPipe rev moves to `process_video()`
     or async, the call inside _run_frame will throw a clear error.
  3. `result.pose_landmarks` is `list[list[NormalizedLandmark]]` —
     OUTER list is one entry per detected pose. Single-person golf
     swings should always yield exactly one entry; if zero we treat the
     frame as null (matches mediapipe_pose behaviour).

CLI:
    python -m benchmark.runners.mediapipe_tasks <video> <video_id>
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp

# Tasks API imports. These are the "current" names as of MediaPipe
# 0.10.14. If MediaPipe is upgraded past that, these may shift.
# TODO(jason): if ImportError fires here, check release notes for renames.
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from ..runner import (
    COCO_NAMES,
    FrameKeypoints,
    RunResult,
    Runner,
    empty_keypoints,
    make_kp,
)

# Heavy model from Google's mediapipe-models bucket. The "float16" build
# is the standard one Google publishes; the "int8" quantised variant
# exists for mobile and is NOT recommended here (accuracy degraded).
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)
MODEL_CACHE = Path.home() / ".cache" / "swingcue-benchmark" / "pose_landmarker_heavy.task"

# Same 33-point → COCO 17 mapping as mediapipe_pose — Tasks API outputs
# the same BlazePose 33 landmark schema, so the indices are identical.
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


def _ensure_model() -> Path:
    if MODEL_CACHE.exists():
        return MODEL_CACHE
    MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"[mediapipe_tasks] downloading model → {MODEL_CACHE}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_CACHE)
    sz = MODEL_CACHE.stat().st_size / 1024 / 1024
    print(f"[mediapipe_tasks] model downloaded ({sz:.1f} MB)")
    return MODEL_CACHE


class MediaPipeTasksRunner(Runner):
    name = "mediapipe_tasks"

    def __init__(self):
        self.landmarker = None

    def setup(self) -> None:
        model_path = _ensure_model()
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=str(model_path),
            ),
            # TODO(jason): VIDEO mode requires monotonically-increasing
            # timestamps via detect_for_video(). If you'd rather use
            # IMAGE mode (no temporal state) swap to:
            #   running_mode=mp_vision.RunningMode.IMAGE
            # and call self.landmarker.detect(mp_image) instead.
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            # TODO(jason): output_segmentation_masks defaults to False;
            # leave it that way for benchmark speed. Set True only if
            # you want to inspect mask quality separately.
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def run(
        self,
        video_path: Path,
        video_id: str,
        sample_fps: float = 10.0,
    ) -> RunResult:
        assert self.landmarker is not None, "call setup() first"
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open {video_path}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = n_total / src_fps if src_fps > 0 else 0.0
        stride = max(1, round(src_fps / sample_fps))

        frames: list[FrameKeypoints] = []
        idx = 0
        t0 = time.perf_counter()
        last_ts_ms = -1  # VIDEO mode requires strictly-increasing timestamps
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # TODO(jason): mp.Image expects SRGB (3-channel RGB uint8).
                # If it complains about format mismatch, try
                # mp.ImageFormat.SRGB_4444 or fall back to
                # mp.solutions API in mediapipe_pose.py.
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = int((idx / src_fps) * 1000)
                # Ensure strictly increasing.
                if ts_ms <= last_ts_ms:
                    ts_ms = last_ts_ms + 1
                last_ts_ms = ts_ms
                result = self.landmarker.detect_for_video(mp_image, ts_ms)
                kps = empty_keypoints()
                # `result.pose_landmarks` is list[list[NormalizedLandmark]]
                # — outer index = pose number (we asked for num_poses=1).
                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    pose0 = result.pose_landmarks[0]
                    for name in COCO_NAMES:
                        lm = pose0[MEDIAPIPE_TO_COCO_IDX[name]]
                        # TODO(jason): NormalizedLandmark exposes both
                        # `.visibility` and `.presence`. Production
                        # (mp.solutions) uses visibility; Tasks API
                        # docs are ambiguous on which gives better signal.
                        # Using visibility for apples-to-apples; flip to
                        # presence if Phase 1B numbers look off.
                        conf = float(lm.visibility) if hasattr(lm, "visibility") else 0.5
                        kps[name] = make_kp(lm.x, lm.y, conf, width, height)
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
                "model=pose_landmarker_heavy.task (float16)",
                "running_mode=VIDEO num_poses=1",
                "min_pose_detection_confidence=0.5  min_pose_presence_confidence=0.5",
                "min_tracking_confidence=0.5",
                f"native_frames={n_total}  sampled={len(frames)}  "
                f"stride={stride}  elapsed={elapsed:.1f}s",
                "kp_conf source: lm.visibility (see TODO inline)",
            ],
        )

    def teardown(self) -> None:
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video",    type=Path)
    ap.add_argument("video_id")
    ap.add_argument("--out-dir", type=Path, default=Path("output"))
    ap.add_argument("--sample-fps", type=float, default=10.0)
    args = ap.parse_args()

    r = MediaPipeTasksRunner()
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
