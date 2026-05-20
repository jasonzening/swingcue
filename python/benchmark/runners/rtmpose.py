"""
rtmpose.py — RTMPose runner via rtmlib lightweight wrapper.

RTMPose (OpenMMLab, Apache-2.0) is a SOTA real-time top-down 2D pose
estimator. Through rtmlib it's a pure ONNXRuntime path — no mmcv /
mmpose / torch — so the install footprint is small and matches the
swingcue PR-3 YOLO11m ONNX pattern. RTMPose's paper benchmark claims
90+ FPS on Intel i7-11700 CPU; on a laptop CPU we expect realtime
comfortably for 7s × 30fps swing clips.

Confidence: MEDIUM-HIGH. rtmlib is the RTMPose team's officially-
recommended thin wrapper. The Body class bundles a YOLOX-s human
detector + RTMPose-m head together and outputs 17 COCO keypoints in
the standard order, so no remap is needed.

Hot-spots flagged inline with # TODO(jason):

  1. First setup() will download two ONNX files (~50 MB total) to
     ~/.cache/rtmlib/. Behind a strict proxy this may need RTMLIB_CACHE
     env var or manual placement.
  2. rtmlib reports invalid keypoints as (-1, -1) with conf 0. We
     normalise these to [None, None, conf] to match the make_kp
     convention.
  3. Body() takes a mode string — "balanced" picks RTMPose-m at
     256x192 input. Swap to "performance" (rtmpose-x 384x288) if Phase
     1B numbers look ceiling-limited.

CLI:
    python -m benchmark.runners.rtmpose <video> <video_id>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

# rtmlib import deferred until setup() runs, so --help works without
# the ML deps installed. TODO(jason): first call to Body() downloads
# RTMPose + YOLOX ONNX (~50 MB total) to ~/.cache/rtmlib/. Set
# RTMLIB_CACHE if you want them elsewhere.

from ..runner import (
    COCO_NAMES,
    FrameKeypoints,
    RunResult,
    Runner,
    empty_keypoints,
    make_kp,
)

# rtmlib Body in "balanced" mode = YOLOX-s detector + RTMPose-m head,
# RTMPose-m 256x192 trained on body7 (COCO + AIC + ...). Direct 17 COCO
# keypoint output in standard order, no remap needed.
RTMLIB_MODE = "balanced"


class RTMPoseRunner(Runner):
    name = "rtmpose"

    def __init__(self, mode: str = RTMLIB_MODE):
        self.mode = mode
        self.body = None

    def setup(self) -> None:
        # TODO(jason): if rtmlib import errors with `ModuleNotFoundError:
        # onnxruntime`, you forgot to `pip install onnxruntime` (it's a
        # peer dep, not a hard dep of rtmlib in some versions).
        from rtmlib import Body
        print(f"[rtmpose] loading rtmlib Body(mode={self.mode!r}, backend=onnxruntime, device=cpu)")
        self.body = Body(
            mode=self.mode,
            backend="onnxruntime",
            device="cpu",
        )

    def run(
        self,
        video_path: Path,
        video_id: str,
        sample_fps: float = 10.0,
    ) -> RunResult:
        assert self.body is not None, "call setup() first"
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
        extra_notes: list[str] = []
        miss_count = 0
        err_count = 0
        idx = 0
        t0 = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                kps = empty_keypoints()
                try:
                    # rtmlib accepts BGR np arrays (same as cv2.imread).
                    # Returns (keypoints, scores) where keypoints.shape
                    # is (n_persons, 17, 2) in pixel coords and
                    # scores.shape is (n_persons, 17). Invalid keypoints
                    # come back as (-1, -1) with score 0.
                    keypoints, scores = self.body(frame)
                except Exception as e:
                    err_count += 1
                    extra_notes.append(f"frame_idx={idx}: inference error: {e!r}")
                    keypoints, scores = [], []

                if len(keypoints) == 0:
                    # detector / pose head saw nothing — leave 17 nulls.
                    miss_count += 1
                    extra_notes.append(f"frame_idx={idx}: detector miss")
                else:
                    # Golf swing is single-person — take the first detection.
                    # TODO(jason): if rtmlib returns multiple persons (camera
                    # caught a passerby), [0] picks whichever YOLOX scored
                    # highest. Usually that's the golfer; verify visually
                    # via overlay.mp4 if a video has crowd.
                    kp0 = keypoints[0]
                    sc0 = scores[0]
                    for i, name in enumerate(COCO_NAMES):
                        x_px = float(kp0[i][0])
                        y_px = float(kp0[i][1])
                        conf = float(sc0[i])
                        if x_px < 0 or y_px < 0:
                            # rtmlib invalid keypoint sentinel — match the
                            # [None, None, conf] convention from runner.py.
                            kps[name] = [None, None, round(conf, 3)]
                        else:
                            # Already in native pixel space → img_w=img_h=1.
                            kps[name] = make_kp(x_px, y_px, conf, 1, 1)

                frames.append(FrameKeypoints(
                    ts=idx / src_fps,
                    frame_idx=idx,
                    keypoints=kps,
                ))
            idx += 1
        elapsed = time.perf_counter() - t0
        cap.release()

        notes = [
            f"rtmlib Body(mode={self.mode!r})  backend=onnxruntime  device=cpu",
            "outputs COCO 17 in native pixel space — no remap",
            f"native_frames={n_total}  sampled={len(frames)}  "
            f"stride={stride}  elapsed={elapsed:.1f}s",
            f"detector_misses={miss_count}  inference_errors={err_count}",
        ]
        notes.extend(extra_notes)

        return RunResult(
            video_id=video_id,
            runner=self.name,
            video_width=width,
            video_height=height,
            fps_native=src_fps,
            fps_sampled=src_fps / stride,
            duration_sec=duration,
            frames=frames,
            notes=notes,
        )

    def teardown(self) -> None:
        # rtmlib holds ONNX sessions internally; drop the ref and let GC
        # close them. No explicit .close().
        self.body = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video",    type=Path)
    ap.add_argument("video_id")
    ap.add_argument("--out-dir", type=Path, default=Path("output"))
    ap.add_argument("--sample-fps", type=float, default=10.0)
    ap.add_argument("--mode", default=RTMLIB_MODE,
                    choices=["lightweight", "balanced", "performance"],
                    help="rtmlib Body mode (default: balanced)")
    args = ap.parse_args()

    r = RTMPoseRunner(mode=args.mode)
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
