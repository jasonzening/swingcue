"""
pose_timeline.py — 17-COCO frame-level pose timeline + data-quality pipeline.

PR-4 data foundation. See docs/decisions/PR-4_DESIGN.md for the full story.

Pipeline order (composed by main.py / future callers):

    raw_coco_frames
      → detect_outliers_and_reject       (kp jumps > 10% of video width)
      → bidirectional_ema                 (PR-5.9: forward+backward, zero phase)
      → gap_fill_linear                   (interp null runs <= 5 frames)
      → apply_yolo_anchor_correction      (when YOLO 5-phase data available)
      → validate_timeline                 (gate before write)

PR-5.9 also snapshots raw_keypoints into each frame in main.py BEFORE
the pipeline runs, so the post-pipeline `keypoints` field can be
compared against the untouched `raw_keypoints` field by the frontend
debug overlay.

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

# PR-5.9 Task 7: head_crown derivation.
# Tunable Python module constant. crown sits roughly factor × (ear→mouth)
# distance above the ear midline. First-cut 0.45 calibrated on b3fea3f0
# face-on setup pose. Adjustable via Python redeploy.
HEAD_CROWN_FACTOR: float = 0.45

# Internal-only MediaPipe 33 indices for landmarks NOT in the 17 output
# keypoints, but required as inputs to derived points (e.g., head_crown).
# Kept separate from MEDIAPIPE_TO_COCO_IDX so the canonical output map
# stays at 17 entries.
_MP_MOUTH_LEFT_IDX: int = 9
_MP_MOUTH_RIGHT_IDX: int = 10


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
    COCO 17-keypoint subset, in video native pixel coordinates. Also
    emits the PR-5.9-derived `head_crown` point (computed from internal
    mouth + ear landmarks; mouth itself is not exported).

    Returns:
        {name: [x_px | None, y_px | None, confidence]} for all 17 COCO
        names + `head_crown`. Coordinates are None when MediaPipe
        visibility < MIN_VISIBILITY; confidence is still recorded in
        the third slot for diagnostics.
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
    # PR-5.9 Task 7: derive head_crown from ear midline + mouth midline.
    out["head_crown"] = derive_head_crown(
        mp_pose_landmarks, video_width, video_height,
    )
    return out


def derive_head_crown(
    mp_pose_landmarks: Any,
    video_width: int,
    video_height: int,
) -> list[Any]:
    """
    PR-5.9 Task 7: skull-crown approximation from MediaPipe ear + mouth
    landmarks. MediaPipe Pose does not output skull crown directly.

    crown = ear_mid + (ear_mid - mouth_mid) * HEAD_CROWN_FACTOR
    conf  = min(left_ear, right_ear, mouth_left, mouth_right) * 0.8

    Returns [x_px, y_px, confidence] in video native pixels. Returns
    [None, None, conf] when any of the four input landmarks has
    visibility below MIN_VISIBILITY (mirrors the existing per-keypoint
    null contract).
    """
    left_ear   = mp_pose_landmarks[MEDIAPIPE_TO_COCO_IDX["left_ear"]]
    right_ear  = mp_pose_landmarks[MEDIAPIPE_TO_COCO_IDX["right_ear"]]
    mouth_l    = mp_pose_landmarks[_MP_MOUTH_LEFT_IDX]
    mouth_r    = mp_pose_landmarks[_MP_MOUTH_RIGHT_IDX]
    conf = round(
        min(left_ear.visibility, right_ear.visibility,
            mouth_l.visibility,  mouth_r.visibility) * 0.8,
        3,
    )
    if (left_ear.visibility  < MIN_VISIBILITY
            or right_ear.visibility < MIN_VISIBILITY
            or mouth_l.visibility   < MIN_VISIBILITY
            or mouth_r.visibility   < MIN_VISIBILITY):
        return [None, None, conf]
    ear_mid_x   = (left_ear.x + right_ear.x) / 2
    ear_mid_y   = (left_ear.y + right_ear.y) / 2
    mouth_mid_x = (mouth_l.x  + mouth_r.x)  / 2
    mouth_mid_y = (mouth_l.y  + mouth_r.y)  / 2
    crown_x = ear_mid_x + (ear_mid_x - mouth_mid_x) * HEAD_CROWN_FACTOR
    crown_y = ear_mid_y + (ear_mid_y - mouth_mid_y) * HEAD_CROWN_FACTOR
    return [
        round(float(crown_x) * video_width,  1),
        round(float(crown_y) * video_height, 1),
        conf,
    ]


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
    DEPRECATED (PR-5.9 Task 2). Causal EMA — left in place for any
    historical caller / test. The pipeline now invokes
    `bidirectional_ema` instead. See that function for the replacement.

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


# Names processed by PR-5.9 bidirectional_ema. Includes the derived
# head_crown so it shares the same smoothing treatment as the COCO 17
# (otherwise it would jitter per-frame while everything else is smooth).
# Outlier rejection + gap fill remain COCO-only; those are independent
# concerns and head_crown's nulls come from its inputs' visibility, not
# from per-frame motion outliers.
_SMOOTHABLE_NAMES: tuple[str, ...] = COCO_NAMES + ("head_crown",)


def _bidirectional_ema_1d(values: list[Optional[float]], alpha: float) -> list[Optional[float]]:
    """
    PR-5.9 Task 2: forward-backward EMA pass.
    Null-aware: a None value resets both passes (so blending never spans
    a gap). The output at each position is the average of the forward
    and backward smoothed estimates — zero phase delay.
    """
    n = len(values)
    if n == 0:
        return []
    # Forward (causal)
    forward: list[Optional[float]] = [None] * n
    prev_f: Optional[float] = None
    for i in range(n):
        v = values[i]
        if v is None:
            forward[i] = None
            prev_f = None
            continue
        if prev_f is None:
            forward[i] = v
            prev_f = v
        else:
            new_v = prev_f + alpha * (v - prev_f)
            forward[i] = new_v
            prev_f = new_v
    # Backward (anti-causal)
    backward: list[Optional[float]] = [None] * n
    prev_b: Optional[float] = None
    for i in range(n - 1, -1, -1):
        v = values[i]
        if v is None:
            backward[i] = None
            prev_b = None
            continue
        if prev_b is None:
            backward[i] = v
            prev_b = v
        else:
            new_v = prev_b + alpha * (v - prev_b)
            backward[i] = new_v
            prev_b = new_v
    # Average — when both passes have a value at i, return the mean.
    # When only one side has a value (the other is None due to a null
    # somewhere ahead/behind), fall back to the side that does. This
    # preserves the original null-skip behaviour at gap boundaries.
    out: list[Optional[float]] = [None] * n
    for i in range(n):
        f, b = forward[i], backward[i]
        if f is None and b is None:
            out[i] = None
        elif f is None:
            out[i] = b
        elif b is None:
            out[i] = f
        else:
            out[i] = (f + b) / 2
    return out


def bidirectional_ema(timeline: dict, alpha: float = 0.4) -> dict:
    """
    PR-5.9 Task 2: replaces the causal `smooth_ema` with a forward-
    backward double pass. Zero phase delay — appropriate for offline
    analysis where the entire timeline is available before write.

    Applied per-keypoint, per-coordinate (x and y independently).
    Confidence values pass through unchanged. Null-aware per the
    _bidirectional_ema_1d contract. Iterates `_SMOOTHABLE_NAMES` which
    includes the PR-5.9 head_crown derivation.

    Mutates and returns timeline.
    """
    frames = timeline["frames"]
    if not frames:
        logger.info(f"[pose_timeline] bidirectional_ema: zero frames, no-op")
        return timeline
    for name in _SMOOTHABLE_NAMES:
        xs: list[Optional[float]] = []
        ys: list[Optional[float]] = []
        for f in frames:
            kp = f["keypoints"].get(name)
            if kp is None:
                xs.append(None)
                ys.append(None)
                continue
            xs.append(kp[0])
            ys.append(kp[1])
        smoothed_x = _bidirectional_ema_1d(xs, alpha)
        smoothed_y = _bidirectional_ema_1d(ys, alpha)
        for i, f in enumerate(frames):
            kp = f["keypoints"].get(name)
            if kp is None:
                continue
            if smoothed_x[i] is not None:
                kp[0] = round(smoothed_x[i], 1)
            if smoothed_y[i] is not None:
                kp[1] = round(smoothed_y[i], 1)
    logger.info(
        f"[pose_timeline] bidirectional_ema: alpha={alpha} "
        f"names={len(_SMOOTHABLE_NAMES)}"
    )
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
# 4b. Build envelope — wrap raw frames into the v1 JSON shape
# ---------------------------------------------------------------------------

def build_timeline_from_raw_coco_frames(
    raw_frames: list[dict],
    video_width: int,
    video_height: int,
    sample_fps: float,
) -> dict:
    """
    Wrap a list of per-frame coco dicts into the v1 pose_timeline_2d JSON
    envelope. Caller is expected to pass the output through the data-
    quality helpers above + apply_yolo_anchor_correction before writing.
    """
    return {
        "version": 1,
        "fps_sampled": int(round(sample_fps)),
        "video_width": video_width,
        "video_height": video_height,
        "keypoint_source": "mediapipe_pose_v1_5",
        "yolo_anchor_correction": {"applied": False},
        "frames": raw_frames,
    }


# ---------------------------------------------------------------------------
# 4c. YOLO anchor correction — per-keypoint per-segment linear lerp
# ---------------------------------------------------------------------------

def apply_yolo_anchor_correction(
    timeline: dict,
    yolo_keypoints_per_phase: dict[str, list[list[float]]],
    phase_markers: dict[str, float],
    min_conf: float = 0.3,
) -> dict:
    """
    Use the 5-phase YOLO COCO-17 keypoints (from PR-3) as ground-truth
    anchors and linearly interpolate the MP→YOLO offset across the rest
    of the timeline.

    Args:
        timeline:                  envelope from build_timeline_from_raw_coco_frames
                                   (already passed through outlier / smooth / gap).
        yolo_keypoints_per_phase:  {phase_name: [17 × [x, y, conf]]}
                                   — collected from yolo_summary.results in main.py.
        phase_markers:             {f"{phase}Time": float_seconds} from detect_phases.
        min_conf:                  drop anchor offsets for kp pairs where
                                   either MP or YOLO conf < min_conf.

    Mutates timeline.frames in place; updates timeline.yolo_anchor_correction
    metadata. Returns the timeline.

    Algorithm:
      1. For each phase in ['setup','top','transition','impact','finish']:
           a. Find MediaPipe frame closest to phase_markers[f"{phase}Time"].
           b. For each of the 17 keypoints, compute offset = yolo_kp - mp_kp.
           c. If either side has conf<min_conf, mark that (phase, kp) None.
      2. For each timeline frame:
           a. If ts < first anchor: apply first anchor's offsets.
           b. If ts > last anchor: apply last anchor's offsets.
           c. Otherwise: linear lerp between bracketing anchors per kp.
           d. None anchors are skipped (use neighbouring anchor's offset).
    """
    if not yolo_keypoints_per_phase:
        timeline["yolo_anchor_correction"] = {"applied": False}
        logger.info("[pose_timeline] apply_yolo_anchor_correction: no YOLO data — skipped")
        return timeline

    phase_names_order = ("setup", "top", "transition", "impact", "finish")
    frames = timeline["frames"]
    if not frames:
        timeline["yolo_anchor_correction"] = {"applied": False}
        return timeline

    # ── Step 1: build anchors ───────────────────────────────────────────
    anchor_ts: list[float] = []
    anchor_offsets: list[dict[str, Optional[tuple[float, float]]]] = []
    applied_phases: list[str] = []

    for phase in phase_names_order:
        if phase not in yolo_keypoints_per_phase:
            continue
        time_key = f"{phase}Time"
        phase_ts = phase_markers.get(time_key)
        if phase_ts is None:
            continue
        yolo_kps = yolo_keypoints_per_phase[phase]
        if not isinstance(yolo_kps, list) or len(yolo_kps) < 17:
            continue

        # Find MediaPipe frame closest to phase_ts
        best_idx = min(
            range(len(frames)),
            key=lambda i: abs(frames[i]["ts"] - phase_ts),
        )
        mp_frame = frames[best_idx]

        per_kp_offset: dict[str, Optional[tuple[float, float]]] = {}
        for kp_idx, name in enumerate(COCO_NAMES):
            mp_kp = mp_frame["keypoints"][name]
            yolo_kp = yolo_kps[kp_idx]
            if (mp_kp[0] is None
                    or not isinstance(yolo_kp, (list, tuple)) or len(yolo_kp) < 3
                    or yolo_kp[2] < min_conf or mp_kp[2] < min_conf):
                per_kp_offset[name] = None
                continue
            per_kp_offset[name] = (
                float(yolo_kp[0]) - mp_kp[0],
                float(yolo_kp[1]) - mp_kp[1],
            )

        anchor_ts.append(float(phase_ts))
        anchor_offsets.append(per_kp_offset)
        applied_phases.append(phase)

    if not anchor_ts:
        timeline["yolo_anchor_correction"] = {"applied": False}
        logger.info("[pose_timeline] apply_yolo_anchor_correction: no usable anchors")
        return timeline

    # ── Step 2: apply per-frame interpolated offsets ───────────────────
    for f in frames:
        ts = f["ts"]
        # Locate bracket
        if ts <= anchor_ts[0]:
            seg_a, seg_b = anchor_offsets[0], anchor_offsets[0]
            t_lerp = 0.0
        elif ts >= anchor_ts[-1]:
            seg_a, seg_b = anchor_offsets[-1], anchor_offsets[-1]
            t_lerp = 0.0
        else:
            seg_a = seg_b = anchor_offsets[0]
            t_lerp = 0.0
            for i in range(len(anchor_ts) - 1):
                if anchor_ts[i] <= ts <= anchor_ts[i + 1]:
                    seg_a = anchor_offsets[i]
                    seg_b = anchor_offsets[i + 1]
                    span = anchor_ts[i + 1] - anchor_ts[i]
                    t_lerp = (ts - anchor_ts[i]) / span if span > 0 else 0.0
                    break

        for name in COCO_NAMES:
            kp = f["keypoints"][name]
            if kp[0] is None:
                continue
            off_a = seg_a.get(name)
            off_b = seg_b.get(name)
            if off_a is None and off_b is None:
                continue
            if off_a is None:
                off = off_b
            elif off_b is None:
                off = off_a
            else:
                off = (
                    off_a[0] + (off_b[0] - off_a[0]) * t_lerp,
                    off_a[1] + (off_b[1] - off_a[1]) * t_lerp,
                )
            kp[0] = round(kp[0] + off[0], 1)
            kp[1] = round(kp[1] + off[1], 1)

    timeline["yolo_anchor_correction"] = {
        "applied": True,
        "anchor_phases": applied_phases,
        "method": "linear_per_segment",
    }
    logger.info(
        f"[pose_timeline] apply_yolo_anchor_correction: applied "
        f"phases={applied_phases}"
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
