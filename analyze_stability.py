import json
import numpy as np

combos = [
    ('RTMPose', 'faceon',      'output/rtmpose/test-faceon_keypoints.json'),
    ('RTMPose', 'downtheline', 'output/rtmpose/test-dwontheline_keypoints.json'),
    ('ViTPose', 'faceon',      'output/vitpose/test-faceon_keypoints.json'),
    ('ViTPose', 'downtheline', 'output/vitpose/test-dwontheline_keypoints.json'),
]

key_joints = ['left_shoulder','right_shoulder','left_elbow','right_elbow',
              'left_wrist','right_wrist','left_hip','right_hip']

for model, video, json_path in combos:
    with open(f'/home/jason/projects/swingcue-postest/{json_path}') as f:
        d = json.load(f)

    prev = {}
    drifts = {j: [] for j in key_joints}
    detect_rate = 0
    total = len(d['frames'])

    for frame_data in d['frames']:
        if not frame_data['persons']:
            continue
        detect_rate += 1
        kps = frame_data['persons'][0]['keypoints']
        for jname in key_joints:
            if jname in kps and kps[jname]['score'] > 0.3:
                x, y = kps[jname]['x'], kps[jname]['y']
                if jname in prev:
                    dx = x - prev[jname][0]
                    dy = y - prev[jname][1]
                    drifts[jname].append(np.sqrt(dx*dx + dy*dy))
                prev[jname] = (x, y)

    print(f'\n{model} {video}:')
    print(f'  Detection rate: {detect_rate}/{total} frames ({100*detect_rate/total:.0f}%)')
    for j in key_joints:
        if drifts[j]:
            arr = np.array(drifts[j])
            print(f'  {j:20s}: mean_drift={arr.mean():.1f}px  p95={np.percentile(arr,95):.1f}px  max={arr.max():.1f}px')
