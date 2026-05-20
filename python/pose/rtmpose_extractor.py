"""
rtmpose_extractor.py — production RTMPose 17-COCO per-frame extractor.

PR-6.1a deliverable. Mirrors the contract of
pose_timeline.extract_coco_subset_from_mediapipe so analyzer.py can swap
extractors behind the POSE_RUNNER_OVERRIDE env flag with zero
downstream-pipeline changes.

What this DOES emit (PR-6.1a scope):
  - 17 standard COCO keypoint names in canonical order
  - Pixel-space [x_px | None, y_px | None, conf 0-1] triples
  - Below-MIN_VISIBILITY keypoints normalised to [None, None, conf]
  - Invalid rtmlib sentinels (negative coords) normalised the same way

What this DOES NOT emit yet (deferred to PR-6.1b):
  - head_crown — RTMPose 17 COCO has no mouth landmarks, so the existing
    mediapipe-mouth+ear geometry doesn't port. New ear+nose derivation
    needs empirical scale_factor tuning (PR-6.1_SPEC_v2 §4).

Confidence: MEDIUM-HIGH. Same rtmlib Body(mode="balanced", backend=
onnxruntime, device=cpu) pattern proven in the Phase 1B smoke
(comparison_b3fea3f0-….mp4, 47/47 frames, agree@30px = 93.1% vs the
mediapipe_pose baseline). Hot-spots flagged inline with # TODO(jason).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Canonical COCO 17 keypoint order — must match pose_timeline.COCO_NAMES
# tuple exactly so downstream pipeline code reads the same names whether
# the source is MediaPipe or RTMPose.
COCO_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",      "right_eye",
    "left_ear",      "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow",    "right_elbow",
    "left_wrist",    "right_wrist",
    "left_hip",      "right_hip",
    "left_knee",     "right_knee",
    "left_ankle",    "right_ankle",
)

# Below-visibility threshold matches pose_timeline.MIN_VISIBILITY so the
# rtmpose path produces the same coord-null pattern as MediaPipe.
MIN_VISIBILITY: float = 0.3

# rtmlib Body mode. "balanced" = YOLOX-m detector + RTMPose-m head at
# 256x192 input. Picked from PR-6.0 Phase 1B verdict (visual rank #1).
# TODO(jason): try mode="performance" (rtmpose-x 384x288) if visual
# review surfaces remaining drift on extremities.
_RTMLIB_MODE = "balanced"

# Module-level singleton. rtmlib loads YOLOX-m (~90 MB) + RTMPose-m
# (~50 MB) ONNX sessions; loading on every request would burn ~2s of
# wall clock per analyze call. FastAPI keeps the process alive between
# requests so this caches naturally.
_body_instance = None


def _get_body():
    """Lazy-load rtmlib Body singleton on first call."""
    global _body_instance
    if _body_instance is None:
        # Import deferred so module import doesn't pull rtmlib for the
        # mediapipe-default code path.
        from rtmlib import Body
        logger.info(
            "[rtmpose_extractor] loading rtmlib Body("
            f"mode={_RTMLIB_MODE!r}, backend=onnxruntime, device=cpu)"
        )
        # TODO(jason): if production Dockerfile RTMLIB_OFFLINE=1 hasn't
        # populated /opt/rtmlib-cache, this call will attempt to
        # download model weights at runtime and fail in air-gapped envs.
        # PR-6.1_SPEC_v2 §6 covers the Dockerfile prefetch.
        _body_instance = Body(
            mode=_RTMLIB_MODE,
            backend="onnxruntime",
            device="cpu",
        )
        logger.info("[rtmpose_extractor] rtmlib Body loaded")
    return _body_instance


def extract_coco_subset_from_rtmpose(
    frame_bgr: Any,
    video_width: int,
    video_height: int,
) -> Optional[dict[str, list[Any]]]:
    """
    Run RTMPose on one BGR frame and return the COCO 17 keypoint dict in
    video native pixel coordinates.

    Args:
        frame_bgr:     cv2 BGR frame (np.ndarray, HxWx3 uint8). Same
                       format `cv2.VideoCapture.read()` produces, no
                       colour-space conversion required.
        video_width:   native pixel width (for pass-through into the
                       keypoint triples — rtmlib already returns px).
        video_height:  native pixel height (same).

    Returns:
        {name: [x_px | None, y_px | None, confidence]} for all 17 COCO
        names, OR None if rtmlib's bundled YOLOX detector finds no
        person in the frame. Per-keypoint below-MIN_VISIBILITY entries
        are nulled the same way the mediapipe extractor does.

        head_crown is intentionally absent from this output — deferred
        to PR-6.1b empirical sweep.
    """
    body = _get_body()
    # rtmlib accepts BGR np arrays directly (same as cv2.imread output).
    # Returns (keypoints, scores):
    #   keypoints.shape == (n_persons, 17, 2)  pixel coords
    #   scores.shape    == (n_persons, 17)     confidence 0-1
    # Invalid keypoints come back as (-1, -1) with score 0.
    try:
        keypoints, scores = body(frame_bgr)
    except Exception as e:
        logger.warning(
            f"[rtmpose_extractor] inference failed on one frame: {e!r}"
        )
        return None

    if len(keypoints) == 0:
        # Detector miss — no person found. Caller treats this as a
        # null frame (matches mediapipe extractor's pose_landmarks=None).
        return None

    # Golf swing is single-person. YOLOX-m sorts detections by score so
    # [0] is the highest-confidence person — almost always the golfer.
    # TODO(jason): if a passerby crosses the camera, [0] may briefly
    # snap to them. Visual review of overlay.mp4 is the safety net.
    kp0 = keypoints[0]
    sc0 = scores[0]

    out: dict[str, list[Any]] = {}
    for i, name in enumerate(COCO_NAMES):
        x_px = float(kp0[i][0])
        y_px = float(kp0[i][1])
        conf = round(float(sc0[i]), 3)
        # rtmlib's invalid-keypoint sentinel is (-1, -1) with score 0.
        # Match the mediapipe extractor's [None, None, conf] convention
        # so the downstream pipeline doesn't have to special-case rtmpose.
        if x_px < 0 or y_px < 0 or conf < MIN_VISIBILITY:
            out[name] = [None, None, conf]
        else:
            out[name] = [round(x_px, 1), round(y_px, 1), conf]

    # NOTE: no head_crown derivation. PR-6.1b will add the ear+nose
    # geometric derivation after the empirical scale_factor sweep.
    # Frontend gracefully degrades when head_crown is absent (see
    # src/components/SkeletonOverlay.tsx — it iterates COCO_KEYPOINT_NAMES,
    # head_crown isn't in that list, so the rtmpose path simply doesn't
    # render the crown dot. No visual breakage.

    return out
