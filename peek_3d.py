import json
with open('/home/jason/projects/swingcue-postest/output/rtmw3d/test-dwontheline_keypoints3d.json') as f:
    d = json.load(f)
fd = d['frames'][47]
p = fd['persons'][0]
print('body17_3d:')
for k,v in p['body17_3d'].items():
    print(f"  {k:18s}: x={v['x']:.1f} y={v['y']:.1f} z={v['z']:.3f} sc={v['score']:.2f}")
print()
print('body17_2d:')
for k,v in p['body17_2d'].items():
    print(f"  {k:18s}: x={v['x']:.1f} y={v['y']:.1f}")
