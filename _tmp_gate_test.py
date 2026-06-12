"""Register the 5 pre-Layer0 videos as PASS, then verify hard gate."""
import sys; sys.path.insert(0, ".")
from engine.layer0.perception_gate import PerceptionGate

gate = PerceptionGate()

# The 5 original videos were pre-screened by Jason (ground truth session)
for stem, angle in [
    ("Videos2026-06-09_201015_827", "face-on"),
    ("Videos2026-06-09_201039_231", "face-on"),
    ("Videos2026-06-09_201047_915", "face-on"),
    ("Videos2026-06-09_201054_561", "down-the-line"),
    ("Videos2026-06-09_201058_697", "down-the-line"),
]:
    r = gate.ingest(stem, {
        "frames": [],
        "verdict": "PASS",
        "angle":   angle,
        "reason":  "Pre-Layer0 video; manually reviewed by Jason+Claude in Gate-1 sessions; accepted as ground truth baseline.",
    })
    print(f"{stem[-10:]}: {r.verdict}")

# Verify hard gate blocks REJECT
try:
    gate.assert_pass("stodownload(53)")
    print("ERROR: should have raised")
except RuntimeError as e:
    print(f"Gate blocked stodownload(53): OK — {str(e)[:80]}")

# Verify hard gate passes known-good video
try:
    gate.assert_pass("Videos2026-06-09_201054_561")
    print("Gate passed 201054: OK")
except RuntimeError as e:
    print(f"ERROR: {e}")

# Verify REJECT also blocks run_pipeline import-time check
print("\nAll records:")
for f in sorted(gate.records_dir.glob("*.json") if hasattr(gate, 'records_dir') else
                __import__('pathlib').Path("engine/layer0/records").glob("*.json")):
    import json
    d = json.load(open(f))
    print(f"  {f.stem:40s}: {d['verdict']}")
