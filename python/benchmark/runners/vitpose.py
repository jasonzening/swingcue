"""
vitpose.py — ViTPose runner via rtmlib (two-stage: YOLOX detector +
ViTPose pose head).

ViTPose (ViTAE / Wang et al., Apache-2.0) is a transformer pose
estimator. ViTPose-B (86 M params) reaches 78.0+ AP on COCO val —
above the BlazePose / MoveNet family. We pay for that with 5-10× CPU
latency vs RTMPose, so this runner is the "accuracy ceiling" probe;
RTMPose is the "realtime CPU" probe. Both ship as ONNX so we benchmark
them through the same rtmlib pattern.

rtmlib's Body class doesn't bundle ViTPose, so we wire YOLOX-s ourselves
as the detector and feed its bboxes to rtmlib's RTMPose class loaded
with a ViTPose ONNX checkpoint. ViTPose's ONNX I/O is bbox-cropped
input → 17 (or 36, depending on training set) keypoint heatmaps, which
matches RTMPose's ONNX I/O contract, so the rtmlib RTMPose wrapper
loads ViTPose weights cleanly.

Confidence: MEDIUM. Hot-spots flagged inline with # TODO(jason):

  1. ONNX checkpoint URL — primary is the 17-kp **COCO human** export
     from JunkyByte/easy_ViTPose. If that URL 404s, try the alternates
     listed below; the apt36k (animal, 36-kp) export is a LAST-RESORT
     fallback and tagged as such in notes.
  2. rtmlib's `RTMPose` class is what we use to wrap the ViTPose ONNX
     because both follow the same bbox→heatmap contract. If rtmlib
     gains a dedicated `ViTPose` class, swap to it.
  3. ViTPose-B is ~340 MB. If runtime is unacceptable (>20 min/video
     on Jason's CPU), switch MODEL_URL to a ViTPose-S checkpoint
     (~60 MB) — see TODO inline.

CLI:
    python -m benchmark.runners.vitpose <video> <video_id>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from ..runner import (
    COCO_NAMES,
    FrameKeypoints,
    RunResult,
    Runner,
    empty_keypoints,
    make_kp,
)

# Primary: 17-kp **COCO human** export of ViTPose-B from
# JunkyByte/easy_ViTPose. This is the human-trained, COCO-ordered
# checkpoint we want — same keypoint order as RTMPose / MoveNet /
# mediapipe_pose, so no remap is needed downstream.
#
# TODO(jason): if HuggingFace 404s this exact path, try in this order:
#   1. https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/coco_25/vitpose-b-coco_25.onnx
#      (25-kp COCO superset; we'd take the first 17 — same canonical order)
#   2. https://huggingface.co/Pukei-Pukei/ViTPose-ONNX/resolve/main/vitpose-b-coco.onnx
#   3. https://huggingface.co/onnx-community/vitpose-base-simple/resolve/main/onnx/model.onnx
#      (note: transformers.js layout — pre/post-processing may differ)
# LAST RESORT: apt36k animal-trained 36-kp checkpoint. If that's what
# loads, the notes field flags "animal-trained" so Phase 1B reading
# can discount it accordingly.
MODEL_URL = (
    "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/"
    "onnx/coco/vitpose-b-coco.onnx"
)
# TODO(jason): drop to ViTPose-S if -B too slow (>20 min/video on CPU):
#   "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/coco/vitpose-s-coco.onnx"

# YOLOX-s human detector — same one rtmlib's Body class uses
# internally for RTMPose. Stable URL on the OpenMMLab CDN.
DETECTOR_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "onnx_sdk/yolox_s_8xb8-300e_humanart-3ef259a7.zip"
)

# ViTPose-B input resolution. The checkpoint head expects 256×192 just
# like RTMPose-m. If the chosen checkpoint reports a different input
# size at load time, rtmlib will raise — adjust here.
VITPOSE_INPUT_SIZE = (192, 256)  # (width, height) per rtmlib convention


class ViTPoseRunner(Runner):
    name = "vitpose"

    def __init__(
        self,
        pose_model_url: str = MODEL_URL,
        det_model_url: str = DETECTOR_URL,
    ):
        self.pose_model_url = pose_model_url
        self.det_model_url = det_model_url
        self.det = None
        self.pose = None
        self._pose_class_name = "uninitialised"
        self._is_animal_fallback = "apt36k" in pose_model_url.lower()

    def setup(self) -> None:
        # Prefer a native ViTPose class if rtmlib exposes one; fall back
        # to the generic RTMPose wrapper (same bbox-cropped contract).
        # Either way we log which class actually loaded so Phase 1B can
        # tell from the run log + notes field whether the runner used
        # ViTPose-specific preprocessing or the RTMPose default path.
        import rtmlib
        from rtmlib import YOLOX
        if hasattr(rtmlib, "ViTPose"):
            PoseClass = rtmlib.ViTPose
            self._pose_class_name = "rtmlib.ViTPose"
        else:
            # TODO(jason): if rtmlib's RTMPose constructor rejects this
            # URL (signature mismatch with ViTPose's ONNX), the fallback
            # is to use onnxruntime.InferenceSession() directly and
            # re-implement the TopdownAffine preprocessing — keep an eye
            # on https://github.com/Tau-J/rtmlib for a native ViTPose class.
            PoseClass = rtmlib.RTMPose
            self._pose_class_name = "rtmlib.RTMPose (generic; no ViTPose class in this rtmlib version)"

        print(f"[vitpose] loading YOLOX detector   → {self.det_model_url}")
        self.det = YOLOX(
            onnx_model=self.det_model_url,
            model_input_size=(640, 640),
            backend="onnxruntime",
            device="cpu",
        )
        print(f"[vitpose] loading pose head via {self._pose_class_name}")
        print(f"[vitpose]   model → {self.pose_model_url}")
        self.pose = PoseClass(
            onnx_model=self.pose_model_url,
            model_input_size=VITPOSE_INPUT_SIZE,
            backend="onnxruntime",
            device="cpu",
        )

    def run(
        self,
        video_path: Path,
        video_id: str,
        sample_fps: float = 10.0,
    ) -> RunResult:
        assert self.det is not None and self.pose is not None, "call setup() first"
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
                bboxes = []
                try:
                    # rtmlib's YOLOX takes a BGR frame and returns
                    # bboxes in shape (n_persons, 4) — xyxy pixel coords.
                    bboxes = self.det(frame)
                except Exception as e:
                    err_count += 1
                    extra_notes.append(f"frame_idx={idx}: detector error: {e!r}")
                    bboxes = []

                if len(bboxes) == 0:
                    miss_count += 1
                    extra_notes.append(f"frame_idx={idx}: detector miss")
                else:
                    # Single-person golf swing → keep the first bbox.
                    # TODO(jason): if a crowd creeps into view, [0] picks
                    # the highest-conf YOLOX detection. Should be the
                    # golfer; verify visually via overlay.mp4.
                    try:
                        keypoints, scores = self.pose(frame, bboxes[:1])
                    except Exception as e:
                        err_count += 1
                        extra_notes.append(
                            f"frame_idx={idx}: pose inference error: {e!r}"
                        )
                        keypoints, scores = [], []

                    if len(keypoints) == 0:
                        miss_count += 1
                        extra_notes.append(
                            f"frame_idx={idx}: pose head returned empty"
                        )
                    else:
                        kp0 = keypoints[0]
                        sc0 = scores[0]
                        # ViTPose COCO checkpoint outputs 17 kp in
                        # standard COCO order — same as rtmlib RTMPose.
                        # An animal-trained checkpoint will have 36
                        # outputs; we take the first 17 and tag the
                        # notes field so Phase 1B knows the result is
                        # not apples-to-apples.
                        # TODO(jason): if your checkpoint isn't 17 or 36
                        # (e.g. 25-kp coco_25 fallback), the first 17 in
                        # those exports still align with COCO order.
                        n_kp = min(len(kp0), 17)
                        for i in range(n_kp):
                            name = COCO_NAMES[i]
                            x_px = float(kp0[i][0])
                            y_px = float(kp0[i][1])
                            conf = float(sc0[i])
                            if x_px < 0 or y_px < 0:
                                kps[name] = [None, None, round(conf, 3)]
                            else:
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
            f"detector={self.det_model_url}",
            f"pose_model={self.pose_model_url}",
            f"pose_class={self._pose_class_name}",
            f"input_size={VITPOSE_INPUT_SIZE}  backend=onnxruntime  device=cpu",
            f"native_frames={n_total}  sampled={len(frames)}  "
            f"stride={stride}  elapsed={elapsed:.1f}s",
            f"detector_misses={miss_count}  inference_errors={err_count}",
        ]
        if self._is_animal_fallback:
            notes.append(
                "WARNING: animal-trained apt36k checkpoint in use — "
                "keypoint geometry may be unfair to human pose; treat "
                "Phase 1B numbers from this runner with caution"
            )
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
        self.det = None
        self.pose = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video",    type=Path)
    ap.add_argument("video_id")
    ap.add_argument("--out-dir", type=Path, default=Path("output"))
    ap.add_argument("--sample-fps", type=float, default=10.0)
    ap.add_argument("--pose-model-url", default=MODEL_URL,
                    help="ViTPose ONNX URL (override for ViTPose-S, etc.)")
    ap.add_argument("--det-model-url", default=DETECTOR_URL,
                    help="YOLOX detector ONNX URL")
    args = ap.parse_args()

    r = ViTPoseRunner(
        pose_model_url=args.pose_model_url,
        det_model_url=args.det_model_url,
    )
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
