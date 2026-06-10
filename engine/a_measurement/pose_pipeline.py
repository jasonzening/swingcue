"""
engine/a_measurement/pose_pipeline.py
A Layer — Pose/Club Measurement Engine

Input:  video path
Output: list of FrameMeasurement objects (one per frame)
        each carries: frame_idx, keypoints dict, confidences, bone_lengths,
                      measurement_quality (ok/degraded/bad)

Stability rules:
  - Confidence threshold: kp_score < 0.35 → that joint is "missing"
  - Bone-length sentinel: if a bone changes >20% vs rolling median → flag frame degraded
  - No-detection frame: quality = "bad", all joints None
"""

from __future__ import annotations
import os, sys
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
import numpy as np

# Add parent to path so engine is importable from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

JOINT_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]

BONE_PAIRS = [
    ("left_shoulder","left_elbow"),
    ("left_elbow","left_wrist"),
    ("right_shoulder","right_elbow"),
    ("right_elbow","right_wrist"),
    ("left_shoulder","left_hip"),
    ("right_shoulder","right_hip"),
    ("left_hip","left_knee"),
    ("left_knee","left_ankle"),
    ("right_hip","right_knee"),
    ("right_knee","right_ankle"),
]

CONF_THRESHOLD = 0.35
BONE_CHANGE_LIMIT = 0.20   # 20% change from rolling median = sentinel

@dataclass
class FrameMeasurement:
    frame_idx: int
    keypoints: Dict[str, Optional[Tuple[float, float]]]   # name → (x, y) or None
    confidences: Dict[str, float]                          # name → score
    bone_lengths: Dict[str, float]                         # pair_key → px
    measurement_quality: str                               # "ok" / "degraded" / "bad"
    fps: float = 30.0

    def kp(self, name: str) -> Optional[Tuple[float, float]]:
        """Convenience: get (x,y) or None."""
        return self.keypoints.get(name)

    def kp_conf(self, name: str) -> float:
        return self.confidences.get(name, 0.0)

    def wrist_mid(self) -> Optional[Tuple[float, float]]:
        lw = self.kp("left_wrist"); rw = self.kp("right_wrist")
        lsc = self.kp_conf("left_wrist"); rsc = self.kp_conf("right_wrist")
        if lw is None and rw is None:
            return None
        if lw is None: return rw
        if rw is None: return lw
        w = lsc + rsc + 1e-9
        return ((lw[0]*lsc + rw[0]*rsc)/w, (lw[1]*lsc + rw[1]*rsc)/w)

    def shoulder_mid(self) -> Optional[Tuple[float, float]]:
        ls = self.kp("left_shoulder"); rs = self.kp("right_shoulder")
        if ls is None and rs is None: return None
        if ls is None: return rs
        if rs is None: return ls
        return ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)

    def hip_mid(self) -> Optional[Tuple[float, float]]:
        lh = self.kp("left_hip"); rh = self.kp("right_hip")
        if lh is None and rh is None: return None
        if lh is None: return rh
        if rh is None: return lh
        return ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)

    def torso_height(self) -> float:
        sm = self.shoulder_mid(); hm = self.hip_mid()
        if sm is None or hm is None: return 0.0
        return float(np.hypot(sm[0]-hm[0], sm[1]-hm[1]))


class PosePipeline:
    """
    Wraps RTMPose inference + measurement quality sentinel.
    Call run(video_path) → List[FrameMeasurement]
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._body = None
        self._setup_gpu_env()

    def _setup_gpu_env(self):
        venv = "/home/jason/projects/swingcue-postest/.venv/lib/python3.12/site-packages"
        cuda_dirs = [
            f"{venv}/nvidia/cuda_runtime/lib",
            f"{venv}/nvidia/cudnn/lib",
            f"{venv}/nvidia/cublas/lib",
            "/usr/lib/wsl/lib",
        ]
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        if cuda_dirs[0] not in existing:
            os.environ["LD_LIBRARY_PATH"] = ":".join(cuda_dirs) + ":" + existing

    def _get_body(self):
        if self._body is None:
            from rtmlib import Body
            self._body = Body(
                pose="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip",
                det="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",
                det_input_size=(640, 640), pose_input_size=(288, 384),
                mode="performance", backend="onnxruntime", device=self.device,
            )
        return self._body

    def run(self, video_path: str, verbose: bool = True) -> Tuple[List[FrameMeasurement], float]:
        """
        Returns (measurements, fps).
        """
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        body = self._get_body()

        raw_frames = []
        fi = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            kps_arr, sc_arr = body(frame)
            raw_frames.append((fi, kps_arr, sc_arr))
            fi += 1
        cap.release()

        if verbose:
            print(f"  A-layer: {fi} frames processed @ {fps:.0f}fps")

        return self._build_measurements(raw_frames, fps), fps

    def run_from_json(self, kp_json: dict) -> Tuple[List[FrameMeasurement], float]:
        """Build measurements from saved RTMPose JSON (avoids re-inference)."""
        fps = kp_json["stats"].get("source_fps", 30.0)
        raw_frames = []
        for fd in kp_json["frames"]:
            fi = fd["frame"]
            if not fd["persons"]:
                raw_frames.append((fi, None, None))
                continue
            kp_dict = fd["persons"][0]["keypoints"]
            kps = [[kp_dict[n]["x"], kp_dict[n]["y"]] for n in JOINT_NAMES]
            sc  = [kp_dict[n]["score"] for n in JOINT_NAMES]
            raw_frames.append((fi, [np.array(kps)], [np.array(sc)]))
        return self._build_measurements(raw_frames, fps), fps

    def _build_measurements(self, raw_frames, fps) -> List[FrameMeasurement]:
        n = len(raw_frames)
        measurements = []

        # First pass: build all measurements
        for fi, kps_arr, sc_arr in raw_frames:
            if kps_arr is None or len(kps_arr) == 0:
                m = FrameMeasurement(
                    frame_idx=fi, fps=fps,
                    keypoints={name: None for name in JOINT_NAMES},
                    confidences={name: 0.0 for name in JOINT_NAMES},
                    bone_lengths={},
                    measurement_quality="bad",
                )
            else:
                kps = kps_arr[0]; sc = sc_arr[0]
                kp_dict = {}; conf_dict = {}
                for i, name in enumerate(JOINT_NAMES):
                    score = float(sc[i])
                    conf_dict[name] = score
                    kp_dict[name] = (float(kps[i][0]), float(kps[i][1])) if score >= CONF_THRESHOLD else None

                bone_dict = {}
                for (a, b) in BONE_PAIRS:
                    pa = kp_dict.get(a); pb = kp_dict.get(b)
                    if pa and pb:
                        bone_dict[f"{a}_{b}"] = float(np.hypot(pa[0]-pb[0], pa[1]-pb[1]))

                quality = "bad" if all(v is None for v in kp_dict.values()) else "ok"
                m = FrameMeasurement(
                    frame_idx=fi, fps=fps,
                    keypoints=kp_dict, confidences=conf_dict,
                    bone_lengths=bone_dict, measurement_quality=quality,
                )
            measurements.append(m)

        # Second pass: bone-length sentinel (rolling median ±20%)
        bone_history: Dict[str, List[float]] = {}
        for m in measurements:
            if m.measurement_quality == "bad": continue
            for key, val in m.bone_lengths.items():
                bone_history.setdefault(key, []).append(val)

        bone_medians: Dict[str, float] = {k: float(np.median(v)) for k, v in bone_history.items()}

        for m in measurements:
            if m.measurement_quality == "bad": continue
            for key, val in m.bone_lengths.items():
                median = bone_medians.get(key, val)
                if median > 0 and abs(val - median) / median > BONE_CHANGE_LIMIT:
                    m.measurement_quality = "degraded"
                    break

        return measurements
