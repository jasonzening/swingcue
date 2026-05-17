"""
pose_timeline.py — 17-COCO frame-level pose timeline + data-quality pipeline.

PR-4 data foundation. See docs/decisions/PR-4_DESIGN.md for the full story.

Pipeline order (composed by main.py / future callers):

    raw_coco_frames
      → detect_outliers_and_reject       (kp jumps > 10% of video width)
      → smooth_ema                        (alpha=0.4 EMA per kp trajectory)
      → gap_fill_linear                   (interp null runs <= 5 frames)
      → apply_yolo_anchor_correction      (when YOLO 5-phase data available)
      → validate_timeline                 (gate before write)

Coordinate convention: video native pixels throughout. Confidence values
preserved through the pipeline; coordinate `None` values flow as
[None, None, conf] triples representing rejected keypoints.

Logging: all messages prefixed `[pose_timeline]` per project convention.
"""

from __future__ import annotations
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# COCO 17-keypoint names in canonical order. Frontend src/lib/skeleton/coco.ts
# mirrors this list and order — keep in sync.
COCO_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow",    "right_elbow",
    "left_wrist",    "right_wrist",
    "left_hip",      "right_hip",
    "left_knee",     "right_knee",
    "left_ankle",    "right_ankle",
)

# MediaPipe pose 33-point indices for the 17 COCO subset.
# See docs/decisions/PR-4_DESIGN.md §B for the full mapping table.
MEDIAPIPE_TO_COCO_IDX: dict[str, int] = {
    "nose": 0,
    "left_eye": 2,  "right_eye": 5,
    "left_ear": 7,  "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow":    13, "right_elbow":    14,
    "left_wrist":    15, "right_wrist":    16,
    "left_hip":      23, "right_hip":      24,
    "left_knee":     25, "right_knee":     26,
    "left_ankle":    27, "right_ankle":    28,
}

# Per-keypoint visibility threshold below which the coordinate pair is
# rejected as [None, None, conf]. Matches MediaPipe's recommended default.
MIN_VISIBILITY: float = 0.3


# ---------------------------------------------------------------------------
# 1. Extractor — MediaPipe 33-point → COCO 17 subset (per frame)
# ---------------------------------------------------------------------------

def extract_coco_subset_from_mediapipe(
    mp_pose_landmarks: Any,
    video_width: int,
    video_height: int,
) -> dict[str, list[Any]]:
    """
    Convert MediaPipe's 33-point pose_landmarks.landmark list into the
    COCO 17-keypoint subset, in video native pixel coordinates.

    Returns:
        {name: [x_px | None, y_px | None, confidence]} for all 17 names.
        Coordinates are None when MediaPipe visibility < MIN_VISIBILITY;
        confidence is still recorded in the third slot for diagnostics.
    """
    out: dict[str, list[Any]] = {}
    for name in COCO_NAMES:
        idx = MEDIAPIPE_TO_COCO_IDX[name]
        lm = mp_pose_landmarks[idx]
        conf = round(float(lm.visibility), 3)
        if conf < MIN_VISIBILITY:
            out[name] = [None, None, conf]
        else:
            out[name] = [
                round(float(lm.x) * video_width, 1),
                round(float(lm.y) * video_height, 1),
                conf,
            ]
    return out


# ---------------------------------------------------------------------------
# 2. Outlier rejection — single-frame jumps > 10% of video width
# ---------------------------------------------------------------------------

def detect_outliers_and_reject(
    timeline: dict,
    max_pixel_jump_factor: float = 0.10,
) -> dict:
    """
    Reject keypoints whose per-frame displacement exceeds
    max_pixel_jump_factor × video_width pixels (with a 100 px floor).
    Replaces the rejected entry with [None, None, 0.0]. Each keypoint
    is treated independently.

    Mutates timeline in place and returns it.
    """
    frames = timeline["frames"]
    if len(frames) < 2:
        return timeline
    threshold = max(100.0, timeline["video_width"] * max_pixel_jump_factor)
    threshold_sq = threshold * threshold

    rejected_count = 0
    for name in COCO_NAMES:
        last_valid: Optional[tuple[float, float]] = None
        for f in frames:
            kp = f["keypoints"][name]
            x, y, _ = kp
            if x is None or y is None:
                continue
            if last_valid is None:
                last_valid = (x, y)
                continue
            dx = x - last_valid[0]
            dy = y - last_valid[1]
            if (dx * dx + dy * dy) > threshold_sq:
                f["keypoints"][name] = [None, None, 0.0]
                rejected_count += 1
            else:
                last_valid = (x, y)

    logger.info(
        f"[pose_timeline] detect_outliers_and_reject: "
        f"rejected={rejected_count} threshold={threshold:.0f}px"
    )
    return timeline


# ---------------------------------------------------------------------------
# 3. EMA smoothing — per-keypoint trajectory, null-aware
# ---------------------------------------------------------------------------

def smooth_ema(timeline: dict, alpha: float = 0.4) -> dict:
    """
    Exponential moving average per keypoint trajectory.

    Null frames are skipped (smoothing does not fill in gaps — that is
    gap_fill_linear's responsibility). The EMA state resets when a null
    is encountered so blending never spans a gap.

    Confidence values pass through unchanged. Mutates and returns timeline.

    alpha — initial value 0.4 per PR-4 design. Implementation-time
    tunable; lower = smoother + more lag, higher = more reactive.
    """
    frames = timeline["frames"]
    for name in COCO_NAMES:
        prev: Optional[tuple[float, float]] = None
        for f in frames:
            kp = f["keypoints"][name]
            x, y, _ = kp
            if x is None or y is None:
                prev = None
                continue
            if prev is None:
                prev = (x, y)
                continue
            new_x = prev[0] + alpha * (x - prev[0])
            new_y = prev[1] + alpha * (y - prev[1])
            kp[0] = round(new_x, 1)
            kp[1] = round(new_y, 1)
            prev = (new_x, new_y)
    logger.info(f"[pose_timeline] smooth_ema: alpha={alpha}")
    return timeline


# ---------------------------------------------------------------------------
# 4. Gap fill — linear interpolate short null runs
# ---------------------------------------------------------------------------

def gap_fill_linear(timeline: dict, max_gap: int = 5) -> dict:
    """
    Linearly interpolate null runs of length <= max_gap that are bracketed
    by valid frames on both sides. Marks affected frames with
    interpolated=true. Confidence of filled keypoints set to 0.5 to flag
    them as synthesised.

    Frames outside any bracketed run (i.e., leading / trailing nulls or
    long gaps > max_gap) are left as-is.

    Mutates and returns timeline.
    """
    frames = timeline["frames"]
    filled_count = 0
    for name in COCO_NAMES:
        i = 0
        while i < len(frames):
            kp = frames[i]["keypoints"][name]
            if kp[0] is not None:
                i += 1
                continue
            # Found start of a null run; scan to its end.
            run_start = i
            while (i < len(frames)
                   and frames[i]["keypoints"][name][0] is None):
                i += 1
            run_end = i  # exclusive
            run_len = run_end - run_start
            # Need valid frames on both sides + run_len <= max_gap
            if (run_start == 0 or run_end == len(frames)
                    or run_len > max_gap):
                continue
            a = frames[run_start - 1]["keypoints"][name]
            b = frames[run_end]["keypoints"][name]
            if a[0] is None or b[0] is None:
                continue
            for j in range(run_start, run_end):
                t = (j - run_start + 1) / (run_len + 1)
                fx = a[0] + (b[0] - a[0]) * t
                fy = a[1] + (b[1] - a[1]) * t
                frames[j]["keypoints"][name] = [
                    round(fx, 1), round(fy, 1), 0.5,
                ]
                frames[j]["interpolated"] = True
                filled_count += 1
    logger.info(
        f"[pose_timeline] gap_fill_linear: filled={filled_count} "
        f"max_gap={max_gap}"
    )
    return timeline


# ---------------------------------------------------------------------------
# 5. Validation — gate before writing to swing_videos.pose_timeline_2d
# ---------------------------------------------------------------------------

def validate_timeline(
    timeline: dict,
    min_valid_frame_ratio: float = 0.5,
    min_kp_per_valid_frame: int = 8,
) -> bool:
    """
    Returns True only if the timeline has enough usable frames to render.

    Rule: at least min_valid_frame_ratio of frames must have at least
    min_kp_per_valid_frame keypoints with non-null coordinates.

    A False return signals main.py to write NULL into the column;
    frontend then treats it as "skeleton overlay unavailable" and the
    toggle is disabled.
    """
    frames = timeline.get("frames", [])
    if not frames:
        logger.warning("[pose_timeline] validate_timeline: zero frames")
        return False
    valid_count = 0
    for f in frames:
        n_valid = sum(
            1 for kp in f["keypoints"].values()
            if kp[0] is not None
        )
        if n_valid >= min_kp_per_valid_frame:
            valid_count += 1
    ratio = valid_count / len(frames)
    ok = ratio >= min_valid_frame_ratio
    logger.info(
        f"[pose_timeline] validate_timeline: "
        f"valid={valid_count}/{len(frames)} ratio={ratio:.2f} "
        f"min_ratio={min_valid_frame_ratio} → {ok}"
    )
    return ok
