"""
analyzer.py — SwingCue 视频分析核心

功能：
1. 下载视频（Supabase Storage signed URL）
2. 提取 VideoMetadata（真实 duration, fps, width, height）
3. MediaPipe Pose 逐帧分析
4. 输出 KeypointTimeline（归一化 0-1 坐标）

Level2 升级：Point2D 新增 z 字段（MediaPipe 深度，负值=靠近镜头）
用于前端精确计算肩部旋转盘遮挡关系
"""

import os
import cv2
import tempfile
import httpx
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# MediaPipe Pose landmark indices (subset we care about)
LM = {
    'NOSE': 0,
    'LEFT_SHOULDER': 11,
    'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13,
    'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15,
    'RIGHT_WRIST': 16,
    'LEFT_HIP': 23,
    'RIGHT_HIP': 24,
    'LEFT_KNEE': 25,
    'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27,
    'RIGHT_ANKLE': 28,
}

@dataclass
class Point2D:
    x: float          # normalized 0-1 (relative to video width)
    y: float          # normalized 0-1 (relative to video height)
    z: float          # MediaPipe depth (negative = closer to camera)
    confidence: float

@dataclass
class BodyLandmarks:
    head: Optional[Point2D] = None
    leftShoulder: Optional[Point2D] = None
    rightShoulder: Optional[Point2D] = None
    leftElbow: Optional[Point2D] = None
    rightElbow: Optional[Point2D] = None
    leftWrist: Optional[Point2D] = None
    rightWrist: Optional[Point2D] = None
    leftHip: Optional[Point2D] = None
    rightHip: Optional[Point2D] = None
    leftKnee: Optional[Point2D] = None
    rightKnee: Optional[Point2D] = None
    leftAnkle: Optional[Point2D] = None
    rightAnkle: Optional[Point2D] = None

@dataclass
class KeypointFrame:
    time: float  # seconds
    landmarks: BodyLandmarks

    def to_dict(self) -> Dict[str, Any]:
        result = {'time': self.time, 'landmarks': {}}
        lm = self.landmarks
        for attr in ['head', 'leftShoulder', 'rightShoulder',
                     'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist',
                     'leftHip', 'rightHip', 'leftKnee', 'rightKnee',
                     'leftAnkle', 'rightAnkle']:
            pt = getattr(lm, attr)
            if pt is not None:
                result['landmarks'][attr] = {
                    'x': pt.x, 'y': pt.y, 'z': pt.z,
                    'confidence': pt.confidence
                }
        return result

@dataclass
class VideoMetadata:
    durationSec: float
    fps: float
    width: int
    height: int

def download_video(url: str, timeout: int = 60) -> str:
    """Download video to a temp file, return path."""
    suffix = '.mp4'
    if '.mov' in url.lower(): suffix = '.mov'
    if '.avi' in url.lower(): suffix = '.avi'

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        tmp_path = f.name

    logger.info(f"Downloading video to {tmp_path}")
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        with client.stream('GET', url) as resp:
            resp.raise_for_status()
            with open(tmp_path, 'wb') as f:
                for chunk in resp.iter_bytes(chunk_size=8192):
                    f.write(chunk)

    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    logger.info(f"Downloaded {size_mb:.1f} MB")
    return tmp_path

def get_video_metadata(cap: cv2.VideoCapture) -> VideoMetadata:
    """Extract real video metadata from OpenCV capture."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0.0
    return VideoMetadata(
        durationSec=round(duration, 3),
        fps=round(fps, 2),
        width=width,
        height=height,
    )

def extract_landmarks(result, conf_threshold: float = 0.3) -> Optional[BodyLandmarks]:
    """Convert MediaPipe result to our BodyLandmarks structure."""
    if not result.pose_landmarks:
        return None

    lms = result.pose_landmarks.landmark

    def pt(idx: int) -> Optional[Point2D]:
        lm = lms[idx]
        if lm.visibility < conf_threshold:
            return None
        return Point2D(
            x=round(float(lm.x), 4),
            y=round(float(lm.y), 4),
            z=round(float(lm.z), 4),  # Level2: 深度信息
            confidence=round(float(lm.visibility), 3),
        )

    return BodyLandmarks(
        head=pt(LM['NOSE']),
        leftShoulder=pt(LM['LEFT_SHOULDER']),
        rightShoulder=pt(LM['RIGHT_SHOULDER']),
        leftElbow=pt(LM['LEFT_ELBOW']),
        rightElbow=pt(LM['RIGHT_ELBOW']),
        leftWrist=pt(LM['LEFT_WRIST']),
        rightWrist=pt(LM['RIGHT_WRIST']),
        leftHip=pt(LM['LEFT_HIP']),
        rightHip=pt(LM['RIGHT_HIP']),
        leftKnee=pt(LM['LEFT_KNEE']),
        rightKnee=pt(LM['RIGHT_KNEE']),
        leftAnkle=pt(LM['LEFT_ANKLE']),
        rightAnkle=pt(LM['RIGHT_ANKLE']),
    )

def moving_average(arr: np.ndarray, window: int = 3) -> np.ndarray:
    """Simple moving average smoothing."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')

_SAMPLE_FPS_CAP = 60.0
_SAMPLE_FPS_FALLBACK_NATIVE = 30.0

def _analyze_video_mediapipe(
    video_path: str, sample_fps: Optional[float] = None,
) -> tuple[VideoMetadata, List[KeypointFrame], list[dict], float]:
    """
    MediaPipe extractor path (production default — predates PR-6.1).
    Samples the video at sample_fps. When sample_fps is None (PR-5.9
    default), uses min(native_fps, 60). When explicitly provided, the
    value is clamped to [1, 60]. Runs MediaPipe Pose on each sampled
    frame.

    Returns:
        VideoMetadata,
        List[KeypointFrame]    — existing BodyLandmarks (NOSE + 12 joints),
        list[dict]             — PR-4 raw COCO 17 frames {ts, frame_idx,
                                  interpolated:False, keypoints:{name:[x,y,c]}}
                                  for input to pose_timeline.py pipeline,
        float                  — PR-5.9: effective sample_fps actually used
                                  (so the caller can write it to the
                                  pose_timeline_2d envelope).
    """
    # PR-4: COCO 17 extractor — defined here (rather than imported) so the
    # callers that don't care about pose_timeline still don't pay an import.
    from pose_timeline import extract_coco_subset_from_mediapipe

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    metadata = get_video_metadata(cap)
    logger.info(f"Video: {metadata.width}x{metadata.height} @ {metadata.fps}fps, {metadata.durationSec:.2f}s")

    video_fps = metadata.fps or _SAMPLE_FPS_FALLBACK_NATIVE
    # PR-5.9 Task 1: native-fps sampling by default, capped at 60.
    if sample_fps is None:
        effective_sample_fps = min(video_fps, _SAMPLE_FPS_CAP)
    else:
        # Explicit override (e.g., test harness). Clamp to [1, 60].
        effective_sample_fps = max(1.0, min(float(sample_fps), _SAMPLE_FPS_CAP))
    logger.info(
        f"[analyze_video] sample_fps: requested={sample_fps} → "
        f"effective={effective_sample_fps} (native={video_fps})"
    )
    frame_interval = max(1, int(round(video_fps / effective_sample_fps)))

    pose_config = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # 0=lite, 1=full, 2=heavy
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    keypoint_frames: List[KeypointFrame] = []
    raw_coco_frames: list[dict] = []
    frame_idx = 0

    with pose_config as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                time_sec = frame_idx / video_fps

                # MediaPipe expects RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                landmarks = extract_landmarks(result)

                if landmarks:
                    keypoint_frames.append(KeypointFrame(
                        time=round(time_sec, 3),
                        landmarks=landmarks,
                    ))

                # PR-4: capture full 17-COCO subset for pose_timeline pipeline.
                # Independent of the BodyLandmarks/conf_threshold path above
                # so partial-confidence frames still enter the timeline (the
                # COCO extractor nulls out individual low-vis keypoints).
                if result.pose_landmarks:
                    raw_coco_frames.append({
                        "ts": round(time_sec, 3),
                        "frame_idx": frame_idx,
                        "interpolated": False,
                        "keypoints": extract_coco_subset_from_mediapipe(
                            result.pose_landmarks.landmark,
                            metadata.width,
                            metadata.height,
                        ),
                    })

            frame_idx += 1

    cap.release()
    logger.info(
        f"Extracted {len(keypoint_frames)} keypoint frames; "
        f"{len(raw_coco_frames)} raw COCO frames"
    )
    return metadata, keypoint_frames, raw_coco_frames, effective_sample_fps


# ---------------------------------------------------------------------------
# PR-6.1a: RTMPose extractor path. Behind POSE_RUNNER_OVERRIDE env flag.
# Ships dormant by default; opt-in via POSE_RUNNER_OVERRIDE=rtmpose.
# See docs/files/PR-6.1_SPEC_v2.md §13 (env-flagged rollout).
# ---------------------------------------------------------------------------

# COCO 17 names used to build BodyLandmarks from the rtmpose extractor's
# pixel dict. Mirrors the same 13 named slots in BodyLandmarks but sourced
# from COCO names instead of MediaPipe 33-point indices.
_BODY_LANDMARK_FROM_COCO: tuple[tuple[str, str], ...] = (
    ("head",          "nose"),
    ("leftShoulder",  "left_shoulder"),
    ("rightShoulder", "right_shoulder"),
    ("leftElbow",     "left_elbow"),
    ("rightElbow",    "right_elbow"),
    ("leftWrist",     "left_wrist"),
    ("rightWrist",    "right_wrist"),
    ("leftHip",       "left_hip"),
    ("rightHip",      "right_hip"),
    ("leftKnee",      "left_knee"),
    ("rightKnee",     "right_knee"),
    ("leftAnkle",     "left_ankle"),
    ("rightAnkle",    "right_ankle"),
)


def _coco_pixel_dict_to_body_landmarks(
    coco: dict[str, list],
    video_width: int,
    video_height: int,
) -> Optional[BodyLandmarks]:
    """
    Convert the 17-COCO pixel-space dict (from rtmpose_extractor) into the
    same BodyLandmarks shape the MediaPipe path emits.

    Coords come back from the extractor in native pixels; BodyLandmarks
    holds normalized 0-1 floats — divide by width/height. RTMPose does
    not predict depth; z is forced to 0.0.

    Returns None when ALL 13 mapped keypoints are null. Otherwise returns
    a BodyLandmarks whose individual fields may still be None for nulled
    keypoints (matches MediaPipe extract_landmarks semantics).
    """
    if not coco:
        return None

    def pt(coco_name: str) -> Optional[Point2D]:
        kp = coco.get(coco_name)
        if kp is None or kp[0] is None or kp[1] is None:
            return None
        x_px, y_px, conf = kp
        return Point2D(
            x=round(float(x_px) / video_width,  4) if video_width  else 0.0,
            y=round(float(y_px) / video_height, 4) if video_height else 0.0,
            # rtmpose does not predict depth; coaching code that reads
            # Point2D.z (e.g. shoulder rotation disc occlusion) silently
            # gets 0.0 for now. Disc visualisation already gracefully
            # handles flat-z input — verified in PR-6.0 Phase 1B smoke.
            z=0.0,
            confidence=round(float(conf), 3),
        )

    bl = BodyLandmarks(
        head=pt("nose"),
        leftShoulder=pt("left_shoulder"),
        rightShoulder=pt("right_shoulder"),
        leftElbow=pt("left_elbow"),
        rightElbow=pt("right_elbow"),
        leftWrist=pt("left_wrist"),
        rightWrist=pt("right_wrist"),
        leftHip=pt("left_hip"),
        rightHip=pt("right_hip"),
        leftKnee=pt("left_knee"),
        rightKnee=pt("right_knee"),
        leftAnkle=pt("left_ankle"),
        rightAnkle=pt("right_ankle"),
    )
    # All-null guard mirrors the MediaPipe path's "no pose_landmarks → None"
    # behavior. If every keypoint dropped below visibility, treat the frame
    # as null so callers skip appending a KeypointFrame.
    has_any = any(getattr(bl, attr) is not None for attr, _ in _BODY_LANDMARK_FROM_COCO)
    return bl if has_any else None


def _analyze_video_rtmpose(
    video_path: str, sample_fps: Optional[float] = None,
) -> tuple[VideoMetadata, List[KeypointFrame], list[dict], float]:
    """
    RTMPose extractor path — same return shape as _analyze_video_mediapipe.

    Sampling logic mirrors the MediaPipe path exactly so a side-by-side
    comparison varies only the extractor, not the timing of sampled
    frames. head_crown is intentionally NOT derived here (PR-6.1a scope
    decision; PR-6.1b adds an ear+nose derivation after empirical sweep).
    """
    # Deferred import keeps the mediapipe-default code path free of any
    # rtmlib transitive cost when POSE_RUNNER_OVERRIDE is unset.
    from pose.rtmpose_extractor import extract_coco_subset_from_rtmpose

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    metadata = get_video_metadata(cap)
    logger.info(
        f"[rtmpose] Video: {metadata.width}x{metadata.height} @ "
        f"{metadata.fps}fps, {metadata.durationSec:.2f}s"
    )

    video_fps = metadata.fps or _SAMPLE_FPS_FALLBACK_NATIVE
    # Same sampling logic as the MediaPipe path. PR-6.1d will replace
    # this fixed-stride sampling with adaptive (low fps in static
    # phases, high fps in active swing window); for 6.1a we keep parity.
    if sample_fps is None:
        effective_sample_fps = min(video_fps, _SAMPLE_FPS_CAP)
    else:
        effective_sample_fps = max(1.0, min(float(sample_fps), _SAMPLE_FPS_CAP))
    logger.info(
        f"[analyze_video_rtmpose] sample_fps: requested={sample_fps} → "
        f"effective={effective_sample_fps} (native={video_fps})"
    )
    frame_interval = max(1, int(round(video_fps / effective_sample_fps)))

    keypoint_frames: List[KeypointFrame] = []
    raw_coco_frames: list[dict] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            time_sec = frame_idx / video_fps

            # rtmpose extractor takes BGR directly — no colour conversion.
            coco_dict = extract_coco_subset_from_rtmpose(
                frame, metadata.width, metadata.height,
            )

            if coco_dict is not None:
                landmarks = _coco_pixel_dict_to_body_landmarks(
                    coco_dict, metadata.width, metadata.height,
                )
                if landmarks is not None:
                    keypoint_frames.append(KeypointFrame(
                        time=round(time_sec, 3),
                        landmarks=landmarks,
                    ))
                # Raw COCO frame mirrors the mediapipe path's envelope so
                # pose_timeline.py downstream pipeline doesn't need to
                # special-case the source. head_crown is intentionally
                # absent — 6.1b adds it.
                raw_coco_frames.append({
                    "ts": round(time_sec, 3),
                    "frame_idx": frame_idx,
                    "interpolated": False,
                    "keypoints": coco_dict,
                })

        frame_idx += 1

    cap.release()
    logger.info(
        f"[rtmpose] Extracted {len(keypoint_frames)} keypoint frames; "
        f"{len(raw_coco_frames)} raw COCO frames"
    )
    return metadata, keypoint_frames, raw_coco_frames, effective_sample_fps


# Env var name documented in PR-6.1_SPEC_v2.md §9 + §13. Default value
# 'mediapipe' is explicit so the production code path is identical to
# pre-PR-6.1 behavior unless this flag is set on the deploy.
_POSE_RUNNER_ENV: str = "POSE_RUNNER_OVERRIDE"


def analyze_video(
    video_path: str, sample_fps: Optional[float] = None,
) -> tuple[VideoMetadata, List[KeypointFrame], list[dict], float]:
    """
    PR-6.1a env-flag dispatcher.

    Reads POSE_RUNNER_OVERRIDE at call time (not import time) so a
    Railway env var flip + redeploy switches the extractor without code
    changes. Unrecognised values fall back to the mediapipe default with
    a warning — never a hard error.

    Supported values:
      - "mediapipe" (default if unset): legacy MediaPipe Pose extractor
      - "rtmpose":                     PR-6.1 RTMPose extractor

    Both paths return the identical tuple shape.
    """
    runner = os.environ.get(_POSE_RUNNER_ENV, "mediapipe").strip().lower()
    if runner == "rtmpose":
        logger.info(
            f"[analyze_video] dispatch: POSE_RUNNER_OVERRIDE={runner} "
            f"→ _analyze_video_rtmpose"
        )
        return _analyze_video_rtmpose(video_path, sample_fps)
    if runner not in ("", "mediapipe"):
        logger.warning(
            f"[analyze_video] unknown POSE_RUNNER_OVERRIDE={runner!r}; "
            f"falling back to mediapipe"
        )
    return _analyze_video_mediapipe(video_path, sample_fps)
