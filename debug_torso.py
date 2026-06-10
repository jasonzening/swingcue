import json
import numpy as np
from scipy.signal import savgol_filter

with open('/home/jason/projects/swingcue-postest/output/rtmpose/test-dwontheline_keypoints.json') as f:
    d = json.load(f)

# Get torso height at address frame (frame 10) for sanity threshold
fd = d['frames'][10]
kps = fd['persons'][0]['keypoints']
sh_y = (kps['left_shoulder']['y'] + kps['right_shoulder']['y']) / 2
hip_y = (kps['left_hip']['y'] + kps['right_hip']['y']) / 2
torso_h = abs(hip_y - sh_y)
print(f"Frame 10 shoulder_y={sh_y:.1f}  hip_y={hip_y:.1f}  torso_h={torso_h:.1f}px")
print(f"30% of torso = {torso_h*0.3:.1f}px  (y-tolerance for impact)")
