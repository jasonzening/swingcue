import json

with open('/home/jason/projects/swingcue-postest/output/rtmpose/test-dwontheline_keypoints.json') as f:
    d = json.load(f)

print('total frames:', len(d['frames']))
print('fps:', d['stats']['source_fps'])
print()
print('Frame | LW_x   LW_y  LW_sc | RW_x   RW_y  RW_sc | mid_y')
for fd in d['frames']:
    fi = fd['frame']
    if not fd['persons']:
        print(f'{fi:5d} | NO DETECTION')
        continue
    kps = fd['persons'][0]['keypoints']
    lw = kps['left_wrist']
    rw = kps['right_wrist']
    mid_y = (lw['y'] + rw['y']) / 2
    print(f"{fi:5d} | {lw['x']:6.1f} {lw['y']:6.1f} {lw['score']:.2f} | {rw['x']:6.1f} {rw['y']:6.1f} {rw['score']:.2f} | {mid_y:6.1f}")
