"""
runner.py — base Runner contract + shared COCO 17 schema.

All benchmark runners (mediapipe_pose, mediapipe_tasks, movenet_thunder)
produce the same `RunResult` shape so overlay.py / compare.py / metrics.py
can iterate over any runner output uniformly.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

# COCO 17 keypoint canonical order. Mirrors python/pose_timeline.py
# COCO_NAMES so production code and benchmark output line up byte-for-byte
# (allowing future direct diffs against stored pose_timeline_2d). Keep in
# sync — this is intentionally duplicated so the benchmark dir has zero
# imports from production code.
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
COCO_INDEX: dict[str, int] = {n: i for i, n in enumerate(COCO_NAMES)}

# Below-threshold keypoints are stored as [None, None, conf] (matches the
# production convention in extract_coco_subset_from_mediapipe).
MIN_VISIBILITY: float = 0.3


@dataclass
class FrameKeypoints:
    """One sampled frame from a video, with COCO 17 keypoints in native px."""
    ts: float
    frame_idx: int
    # Map of COCO name → [x_px | None, y_px | None, conf 0-1].
    keypoints: dict[str, list[Any]]


@dataclass
class RunResult:
    """Full pose timeline for one (video, runner) pair."""
    video_id: str
    runner: str
    video_width: int
    video_height: int
    fps_native: float          # source video fps
    fps_sampled: float          # what the runner actually sampled at
    duration_sec: float
    frames: list[FrameKeypoints] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "video_id":     self.video_id,
            "runner":       self.runner,
            "video_width":  self.video_width,
            "video_height": self.video_height,
            "fps_native":   self.fps_native,
            "fps_sampled":  self.fps_sampled,
            "duration_sec": self.duration_sec,
            "frames":       [asdict(f) for f in self.frames],
            "notes":        self.notes,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2))


class Runner(ABC):
    """
    Contract:
      - name        — unique identifier, used as output subdirectory
      - setup()     — load model / warmup; may download files
      - run(video)  — process one video file, return a RunResult
      - teardown()  — release resources
    """

    name: str = ""

    @abstractmethod
    def setup(self) -> None: ...

    @abstractmethod
    def run(
        self,
        video_path: Path,
        video_id: str,
        sample_fps: float = 10.0,
    ) -> RunResult: ...

    def teardown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers shared by runners
# ---------------------------------------------------------------------------

def make_kp(
    x: Optional[float],
    y: Optional[float],
    conf: float,
    img_w: int,
    img_h: int,
) -> list[Any]:
    """
    Build a [x_px | None, y_px | None, conf] triple.

    `x` and `y` are normalised 0-1 (MediaPipe convention). Producers that
    already work in pixel space should pass img_w=img_h=1 (no rescaling).
    Below MIN_VISIBILITY → coords are nulled but conf is preserved.
    """
    c = round(float(conf), 3)
    if x is None or y is None or c < MIN_VISIBILITY:
        return [None, None, c]
    return [round(float(x) * img_w, 1), round(float(y) * img_h, 1), c]


def empty_keypoints() -> dict[str, list[Any]]:
    """A keypoints dict with all 17 entries set to [None, None, 0.0]."""
    return {name: [None, None, 0.0] for name in COCO_NAMES}
