"""
analyzer.py — SwingCue 视频分析核心

功能：
1. 下载视频（Supabase Storage signed URL）
2. 提取 VideoMetadata（真实 duration, fps, width, height）
3. MediaPipe Pose 逐帧分析
4. 输出 KeypointTimeline（归一化 0-1 坐标）
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
    'NOSE':            0,
    'LEFT_SHOULDER':   11,
    'RIGHT_SHOULDER':  12,
    'LEFT_ELBOW':      13,
    'RIGHT_ELBOW':     14,
    'LEFT_WRIST':      15,
    'RIGHT_WRIST':     16,
    'LEFT_HIP':        23,
    'RIGHT_HIP':       24,
    'LEFT_KNEE':       25,
    'RIGHT_KNEE':      26,
    'LEFT_ANKLE':      27,
    'RIGHT_ANKLE':     28,
}


@dataclass
class Point2D:
    x: float        # normalized 0–1 (relative to video width)
    y: float        # normalized 0–1 (relative to video height)
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
    time: float          # seconds
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
                result['landmarks'][attr] = {'x': pt.x, 'y': pt.y, 'confidence': pt.confidence}
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
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
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


def analyze_video(video_path: str, sample_fps: float = 4.0) -> tuple[VideoMetadata, List[KeypointFrame]]:
    """
    Main analysis function.
    Samples the video at sample_fps (default 4 frames/sec),
    runs MediaPipe Pose on each frame,
    returns VideoMetadata + List[KeypointFrame].
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    metadata = get_video_metadata(cap)
    logger.info(f"Video: {metadata.width}x{metadata.height} @ {metadata.fps}fps, {metadata.durationSec:.2f}s")

    video_fps = metadata.fps or 30.0
    frame_interval = max(1, int(video_fps / sample_fps))

    pose_config = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,           # 0=lite, 1=full, 2=heavy
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    keypoint_frames: List[KeypointFrame] = []
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

            frame_idx += 1

    cap.release()
    logger.info(f"Extracted {len(keypoint_frames)} keypoint frames")
    return metadata, keypoint_frames
