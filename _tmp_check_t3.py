import sys; sys.path.insert(0, ".")
from engine.orientation.resolver import _trail_to_handedness, _opposite

print("trail=right ->", _trail_to_handedness("right"))
print("trail=left  ->", _trail_to_handedness("left"))

ft = "right"; ftrail = _opposite(ft); h = _trail_to_handedness(ftrail)
print(f"T3: f_target=right, f_trail={ftrail}, handedness={h}")

# Check what the resolver actually does in T3 path
from engine.orientation.resolver import OrientationResolver

class FakeMeas:
    def __init__(self, fi, wx, hx=150):
        self.frame_idx = fi
        self._wx = wx; self._hx = hx
        self.measurement_quality = "ok"
        self.confidences = {}; self.bone_lengths = {}; self.keypoints = {}
    def wrist_mid(self): return (self._wx, 300.0)
    def hip_mid(self): return (self._hx, 500.0)
    def shoulder_mid(self): return None
    def torso_height(self): return 200.0

# T3: addr=top=0, impact=15, imp+5=20
meas = [FakeMeas(i, {0:200, 15:210, 20:370}.get(i, 200)) for i in range(25)]
res = OrientationResolver().resolve(meas, "face-on", 0, 0, 15)
print(f"\nT3 result: handedness={res.handedness} target={res.target_side} trail={res.trail_side} conf={res.confidence}")
print(f"method: {res.method}")
