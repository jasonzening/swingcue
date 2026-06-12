import json, sys
sys.path.insert(0, ".")

d = json.load(open("engine/kp_cache/Videos2026-06-09_201054_561.json"))
fr88 = d["frames"][88]
kp = fr88["persons"][0]["keypoints"]
print("Keys:", list(kp.keys()))
print("\nfps:", d["stats"]["source_fps"])
print("n_frames:", len(d["frames"]))

# Check 201054 video dimensions
import cv2
cap = cv2.VideoCapture("input/Videos2026-06-09_201054_561.mp4")
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
print(f"\n201054: {w}x{h}  fps={fps}  frames={total}")

cap2 = cv2.VideoCapture("input/Videos2026-06-09_201039_231.mp4")
w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap2.release()
print(f"201039: {w2}x{h2}")

# Check what face kp are available
d2 = json.load(open("engine/kp_cache/Videos2026-06-09_201039_231.json"))
fr80 = d2["frames"][80]
kp2 = fr80["persons"][0]["keypoints"]
print("\n201039 kp keys:", list(kp2.keys()))
# Check head/nose/ear keys
for k in kp2:
    if any(x in k for x in ["nose","ear","eye","head"]):
        v = kp2[k]
        print(f"  {k}: x={v['x']:.1f} y={v['y']:.1f} s={v['score']:.2f}")
