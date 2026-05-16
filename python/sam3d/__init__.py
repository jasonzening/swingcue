"""
sam3d — SAM 3D Body (fal-ai/sam-3/3d-body) integration toolkit.

Public surface used by main.py / orchestrator.py:

    from sam3d.orchestrator import pose3d_for_all_phases

Internal modules are exposed for testing but should not be imported directly
from outside this package in production code.
"""

from sam3d.fal_client_wrap import call_fal
from sam3d.frame_extract import extract_frame
from sam3d.keypoints import (
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
)
from sam3d.storage import ensure_bucket, upload_phase_frame
from sam3d.supabase_writer import write_pose_phase, write_pose_phase_failed

__all__ = [
    # keypoint constants (re-exported for caller convenience)
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_HIP",
    "RIGHT_HIP",
    # frame extraction
    "extract_frame",
    # storage
    "ensure_bucket",
    "upload_phase_frame",
    # fal
    "call_fal",
    # supabase writer
    "write_pose_phase",
    "write_pose_phase_failed",
]
