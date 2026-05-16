"""
yolo — YOLO11-pose (Ultralytics) integration toolkit.

Public surface used by main.py / orchestrator.py:

    from yolo.orchestrator import yolo_for_all_phases

The submodules are imported here for testing convenience but should not be
reached from outside the package in production code.

License note: Ultralytics YOLO is AGPL-3.0. See
docs/decisions/POSE_MODEL_LICENSE.md for the replacement plan
(RTMPose / YOLO-NAS / MoveNet) prior to commercial deployment.
"""

from yolo.inference import MODEL_NAME, infer_pose
from yolo.keypoints import (
    COCO_KP,
    LEFT_HIP,
    LEFT_SHOULDER,
    MIN_CONFIDENCE,
    RIGHT_HIP,
    RIGHT_SHOULDER,
)
from yolo.supabase_writer import write_yolo_phase

__all__ = [
    # constants
    "COCO_KP",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_HIP",
    "RIGHT_HIP",
    "MIN_CONFIDENCE",
    "MODEL_NAME",
    # inference
    "infer_pose",
    # writer
    "write_yolo_phase",
]
