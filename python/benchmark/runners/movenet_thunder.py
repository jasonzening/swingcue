"""
movenet_thunder.py — Google MoveNet Thunder (TF Hub).

MoveNet is the third-party candidate. Thunder is the larger/slower
sibling of Lightning, with ~10% better accuracy on swing-like motion.
The single-person variant is used here — exactly what golf swings need.

Confidence: MEDIUM. Two hot-spots flagged:

  1. tensorflow / tensorflow_hub install is heavy (~500MB combined).
     The benchmark req file pins versions that worked on Py 3.11 as of
     2026-04. If TF rev breaks the API, see TODO in setup().
  2. The 17 keypoints come out in COCO order ALREADY (no 33→17 remap),
     but the layout is [y, x, conf] with NORMALISED y,x in [0,1]
     relative to the LETTERBOXED 256×256 input — not the original
     frame. We un-pad to native px below; TODO marks the spot to
     re-check if outputs look offset.

Pre-existing TF setup tip: if you have an existing miniconda/venv with
TF installed, point this runner at it. Don't try to mix TF and
MediaPipe wheels in the same env unless you know they're compatible.

CLI:
    python -m benchmark.runners.movenet_thunder <video> <video_id>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# tensorflow / tensorflow_hub imports deferred until setup() runs, so
# `python -m benchmark.runners.movenet_thunder --help` works without TF.
# TODO(jason): if the TF import errors at setup() time with a CUDA /
# cuDNN message, you don't need GPU — TF will fall back to CPU
# automatically once you `pip uninstall nvidia-*` from the venv, or use
# `tensorflow-cpu` package instead of `tensorflow`.

from ..runner import (
    COCO_NAMES,
    FrameKeypoints,
    RunResult,
    Runner,
    empty_keypoints,
    make_kp,
)

# MoveNet outputs 17 keypoints in this exact order (per TF Hub docs):
#   0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
#   5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
#   9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
#   13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle
# This matches COCO_NAMES by construction.
MOVENET_HUB_URL = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
INPUT_SIZE = 256  # Thunder model expects 256x256 (Lightning is 192x192)


def _letterbox(rgb: np.ndarray, target: int = INPUT_SIZE) -> tuple[np.ndarray, float, int, int]:
    """
    Resize while preserving aspect ratio and pad to `target × target`.
    Returns the padded image and (scale, pad_x, pad_y) so the caller can
    undo the transform when interpreting model output.
    """
    h, w = rgb.shape[:2]
    scale = target / max(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padded = np.zeros((target, target, 3), dtype=rgb.dtype)
    pad_y = (target - new_h) // 2
    pad_x = (target - new_w) // 2
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return padded, scale, pad_x, pad_y


class MoveNetThunderRunner(Runner):
    name = "movenet_thunder"

    def __init__(self):
        self.movenet = None

    def setup(self) -> None:
        # TODO(jason): if tensorflow_hub.load() complains about TF
        # version, the supported pairing for the v4 Thunder model is:
        #   tensorflow==2.16.x  tensorflow_hub==0.16.1
        # That's what requirements_benchmark.txt pins.
        import tensorflow as tf
        import tensorflow_hub as hub
        print(f"[movenet_thunder] tf={tf.__version__} loading model …")
        model = hub.load(MOVENET_HUB_URL)
        # `signatures['serving_default']` is the call entry; v4 of
        # this model exposes input `input` and output `output_0`.
        self.movenet = model.signatures["serving_default"]
        self._tf = tf

    def run(
        self,
        video_path: Path,
        video_id: str,
        sample_fps: float = 10.0,
    ) -> RunResult:
        assert self.movenet is not None, "call setup() first"
        tf = self._tf

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
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                padded, scale, pad_x, pad_y = _letterbox(rgb)
                # MoveNet wants int32 [1, 256, 256, 3] — quirky but documented.
                tensor = tf.cast(
                    tf.expand_dims(tf.constant(padded), axis=0),
                    dtype=tf.int32,
                )
                outputs = self.movenet(input=tensor)
                # Shape: [1, 1, 17, 3] → squeeze to [17, 3].
                # TODO(jason): if the output key is renamed in a future
                # TF Hub version (e.g. 'output_0' → 'keypoints'), check
                # `list(outputs.keys())` and update this line.
                arr = outputs["output_0"].numpy().reshape(17, 3)
                kps = empty_keypoints()
                for i, name in enumerate(COCO_NAMES):
                    y_norm, x_norm, conf = arr[i]
                    # y_norm, x_norm are normalised to the 256×256
                    # padded input. Un-letterbox: subtract pad in
                    # normalised input space, divide by (new_size /
                    # target), then * native dims.
                    y_px_pad = y_norm * INPUT_SIZE
                    x_px_pad = x_norm * INPUT_SIZE
                    # back to original image space
                    y_orig = (y_px_pad - pad_y) / scale
                    x_orig = (x_px_pad - pad_x) / scale
                    # Renormalise to 0..1 for make_kp(width, height).
                    kps[name] = make_kp(
                        x_orig / width  if width  > 0 else None,
                        y_orig / height if height > 0 else None,
                        float(conf),
                        width, height,
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
                f"hub_url={MOVENET_HUB_URL}",
                f"input_size={INPUT_SIZE} (letterbox-padded)",
                f"native_frames={n_total}  sampled={len(frames)}  "
                f"stride={stride}  elapsed={elapsed:.1f}s",
                "MoveNet outputs [y, x, conf] normalised to padded input;"
                " un-padded back to native px above (see TODO inline)",
            ],
        )

    def teardown(self) -> None:
        # tf hub Models hold a graph; let GC handle it. No explicit close().
        self.movenet = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video",    type=Path)
    ap.add_argument("video_id")
    ap.add_argument("--out-dir", type=Path, default=Path("output"))
    ap.add_argument("--sample-fps", type=float, default=10.0)
    args = ap.parse_args()

    r = MoveNetThunderRunner()
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
