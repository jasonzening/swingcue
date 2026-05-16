"""
COCO 17 keypoint indices for YOLO11-pose.

COCO 17 keypoints are anatomical surface landmarks (annotated on real
photographs), so:
  - LEFT_SHOULDER / RIGHT_SHOULDER = real acromion
  - LEFT_HIP / RIGHT_HIP            = real hip joint

vs SAM 3D Body's 70 MHR keypoints which are mesh-internal anchors
(kp 7,8 land on chest, kp 9,10 land on abdomen).
"""

# COCO 17 order (matches every COCO-trained pose model — YOLO, MoveNet,
# RTMPose, ViTPose, etc.)
COCO_KP = {
    "NOSE": 0,
    "LEFT_EYE": 1,
    "RIGHT_EYE": 2,
    "LEFT_EAR": 3,
    "RIGHT_EAR": 4,
    "LEFT_SHOULDER": 5,    # acromion
    "RIGHT_SHOULDER": 6,
    "LEFT_ELBOW": 7,
    "RIGHT_ELBOW": 8,
    "LEFT_WRIST": 9,
    "RIGHT_WRIST": 10,
    "LEFT_HIP": 11,        # hip joint
    "RIGHT_HIP": 12,
    "LEFT_KNEE": 13,
    "RIGHT_KNEE": 14,
    "LEFT_ANKLE": 15,
    "RIGHT_ANKLE": 16,
}

# Convenience module-level constants for the four anchors used by the
# frontend disc builder.
LEFT_SHOULDER = COCO_KP["LEFT_SHOULDER"]
RIGHT_SHOULDER = COCO_KP["RIGHT_SHOULDER"]
LEFT_HIP = COCO_KP["LEFT_HIP"]
RIGHT_HIP = COCO_KP["RIGHT_HIP"]

# Per-keypoint confidence threshold below which the frontend should fall
# back to SAM anchors (or skip the disc). Mirrors the frontend constant.
MIN_CONFIDENCE: float = 0.3
