"""
scene_detect.py — PR-8c.4 cheap scene-cut detector.

OpenCV HSV histogram BHATTACHARYYA distance. SwingCue MVP only
supports a single continuous camera clip of one golf swing. Multi-
scene compilations (TikTok-style cuts, edited highlight reels) are
rejected at preprocessing so we never burn Modal $ on WHAM inference
for content WHAM's SLAM module can't model anyway.

Cheap implementation (per PR-8c.4 R4):
  - No PySceneDetect (would add 30-40MB to the Docker image + its own
    ffmpeg deps). OpenCV is already pulled in by mediapipe at runtime.
  - Sample every N frames, compute HSV histogram, BHATTACHARYYA-compare
    adjacent samples. Any pair > threshold → scene cut.
  - 1+ scene cut anywhere in the clip → MVP reject.

Threshold 0.5 is the spec-locked default. Golf single-clip footage
has natural lighting/swing-motion variance so an over-strict
threshold would false-positive. Validation set (spec R4):
  a7a5f936  — multi-scene → expect cuts >= 1   (REJECT)
  b32e0f21  — single clip → expect cuts == 0   (PASS)
  5bbcfbc8  — single clip → expect cuts == 0   (PASS)
"""
from __future__ import annotations

import logging
from typing import Final

logger = logging.getLogger(__name__)

# Spec R4 defaults.
SAMPLE_EVERY_N_FRAMES: Final[int] = 10
BHATTACHARYYA_CUT_THRESHOLD: Final[float] = 0.5


def detect_scene_cuts(
    video_path: str,
    sample_every_n_frames: int = SAMPLE_EVERY_N_FRAMES,
    threshold: float = BHATTACHARYYA_CUT_THRESHOLD,
) -> int:
    """
    Returns count of scene cuts in `video_path`.

      >= 0  → number of cuts detected (0 = single scene; reject if >= 1)
      -1    → could not read video (treat as suspicious upstream)

    Lazy-imports cv2 + numpy so this module is cheap to import in
    contexts that may not need scene detection (e.g., unit tests
    that mock at a higher level).
    """
    try:
        import cv2  # noqa: WPS433  (runtime import is intentional)
    except ImportError:
        logger.error("[scene_detect] cv2 not available — cannot detect scene cuts")
        return -1

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"[scene_detect] could not open {video_path}")
        return -1

    prev_hist = None
    cuts = 0
    fi = 0
    max_dist = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % sample_every_n_frames == 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # 8 bins per channel — coarse enough that lighting/pose
            # variation doesn't trip the threshold, fine enough to
            # catch actual scene changes (different background or
            # different subject).
            hist = cv2.calcHist(
                [hsv], [0, 1, 2], None, [8, 8, 8],
                [0, 180, 0, 256, 0, 256],
            )
            cv2.normalize(hist, hist)
            if prev_hist is not None:
                dist = float(cv2.compareHist(
                    prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA,
                ))
                if dist > max_dist:
                    max_dist = dist
                if dist > threshold:
                    cuts += 1
                    logger.info(
                        f"[scene_detect] frame={fi} dist={dist:.3f} "
                        f"> threshold={threshold} → cut #{cuts}"
                    )
            prev_hist = hist
        fi += 1
    cap.release()

    logger.info(
        f"[scene_detect] {video_path}: scanned {fi} frames "
        f"({fi // sample_every_n_frames} samples), max_dist={max_dist:.3f}, "
        f"cuts={cuts}"
    )
    return cuts
