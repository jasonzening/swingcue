"""
SAM 3D Body MHR keypoint indices.

Verified 2026-05-16 via:
  scripts/fal_test.py + scripts/fal_inspect.py on test_finish.png

The 70-keypoint array is organized as:
  0-6:   head + face cluster (7 points)
  7-20:  body joints (shoulders, hips, knees, ankles, feet) (14 points)
  21-41: hand_1 (21 points)
  42-62: hand_2 (21 points)
  63-69: extra body detail — clavicle/deltoid/sternum/throat (7 points)

Total: 7 + 14 + 21 + 21 + 7 = 70.

Left/Right disambiguation in finish frame uses 3D Z-coordinate:
  Z+ = toward camera (anatomical front in finish pose)
  Z- = away from camera (anatomical back)
"""

# Body anchors (primary)
LEFT_SHOULDER = 7   # acromion, target-side (3D Z=+0.041 in finish test)
RIGHT_SHOULDER = 8  # acromion, trail-side  (3D Z=-0.238 in finish test)
LEFT_HIP = 9
RIGHT_HIP = 10
LEFT_KNEE = 11
RIGHT_KNEE = 12
LEFT_ANKLE = 13
RIGHT_ANKLE = 14

# Foot detail
LEFT_TOE = 15
LEFT_TOE_OUTER = 16
LEFT_HEEL = 17
RIGHT_TOE = 18
RIGHT_TOE_OUTER = 19
RIGHT_HEEL = 20

# Wrists (assumed = hand cluster origins; validate with setup-frame test later)
LEFT_WRIST = 21
RIGHT_WRIST = 42

# Extra detail (use when needed)
LEFT_DELTOID = 63
RIGHT_DELTOID = 64
LEFT_CLAVICLE = 65
RIGHT_CLAVICLE = 66
NECK = 67
STERNUM = 68
THROAT = 69
