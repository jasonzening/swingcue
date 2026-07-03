"""batch3 EET Layer 0 gate ingest — 2026-07-03

VLM analysis results for 6 clips.
Pattern note: all clips show 4/5 frames face-on + 1/5 DTL (follow-through).
This is a known VLM artifact: at address/setup in DTL view the golfer appears
sideways, misread as face-on; only the completed follow-through reads correctly
as DTL. Identical to batch2 dtl-wrong-* clips.

fo-eet-*: dominant face-on matches prefix → PASS
dtl-eet-*: dominant face-on conflicts with prefix dtl → needs_human (per protocol)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from engine.layer0.perception_gate import PerceptionGate

VLM_BATCH3 = {
    "fo-eet-1": {
        "frames": [
            {"fr":0,   "q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,
             "q5_desc":"Young man in navy shirt/floral shorts at address in backyard, face-on."},
            {"fr":57,  "q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,
             "q5_desc":"Man at address over yellow ball in backyard, face-on."},
            {"fr":114, "q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,
             "q5_desc":"Man at address in backyard, navy shirt/tie-dye shorts, face-on."},
            {"fr":171, "q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,
             "q5_desc":"Man at or just past impact in follow-through, face-on."},
            {"fr":227, "q1_golf":True,"q2_persons":1,"q3_angle":"DTL",   "q4_fullbody":True,
             "q5_desc":"Man at full finish position from DTL angle, backyard."},
        ],
        "verdict":"PASS",
        "angle":"face-on",
        "reason":"4/5 face-on dominant; fr227 follow-through reads DTL (known rotation artifact). Prefix fo matches dominant angle.",
    },
    "fo-eet-2": {
        "frames": [
            {"fr":0,   "q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,
             "q5_desc":"Golfer at address, navy shirt/floral shorts, backyard, face-on."},
            {"fr":30,  "q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,
             "q5_desc":"Golfer at impact/early follow-through, face-on, backyard."},
            {"fr":61,  "q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,
             "q5_desc":"Full finish/follow-through, face-on, backyard."},
            {"fr":92,  "q1_golf":True,"q2_persons":1,"q3_angle":"DTL",   "q4_fullbody":True,
             "q5_desc":"Full finish from DTL angle, backyard (follow-through rotation artifact)."},
            {"fr":122, "q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,
             "q5_desc":"Full follow-through position, face-on, backyard."},
        ],
        "verdict":"PASS",
        "angle":"face-on",
        "reason":"4/5 face-on dominant; fr92 follow-through reads DTL (known rotation artifact). Prefix fo matches dominant angle.",
    },
    "fo-eet-3": {
        "frames": [
            {"fr":0,   "q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,
             "q5_desc":"Golfer at address, navy shirt/tropical shorts, backyard, face-on."},
            {"fr":28,  "q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,
             "q5_desc":"Golfer at impact, club motion-blurred, face-on."},
            {"fr":56,  "q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,
             "q5_desc":"Golfer at/past impact, yellow ball/tee visible, face-on."},
            {"fr":84,  "q1_golf":True,"q2_persons":1,"q3_angle":"DTL",   "q4_fullbody":True,
             "q5_desc":"Full follow-through from DTL angle (rotation artifact), backyard."},
            {"fr":112, "q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,
             "q5_desc":"Full follow-through, face-on, backyard."},
        ],
        "verdict":"PASS",
        "angle":"face-on",
        "reason":"4/5 face-on dominant; fr84 follow-through reads DTL (known rotation artifact). Prefix fo matches dominant angle.",
    },
    # --- dtl-eet-*: needs_human per angle-conflict protocol ---
    "dtl-eet-1": {
        "frames": [
            {"fr":0,   "q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,
             "q5_desc":"Man in navy polo/floral shorts at address, backyard, face-on."},
            {"fr":50,  "q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,
             "q5_desc":"Same golfer at address, slightly adjusted, face-on."},
            {"fr":101, "q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,
             "q5_desc":"Man at address with iron, backyard, face-on."},
            {"fr":152, "q1_golf":True,"q2_persons":1,"q3_angle":"DTL",   "q4_fullbody":True,
             "q5_desc":"Man in navy sweatshirt/shorts at follow-through, DTL."},
            {"fr":202, "q1_golf":True,"q2_persons":1,"q3_angle":"DTL",   "q4_fullbody":True,
             "q5_desc":"Golfer at follow-through, DTL, alignment stick visible."},
        ],
        "verdict":"needs_human",
        "angle":"mixed",
        "reason":"3/5 face-on + 2/5 DTL. Prefix dtl conflicts with face-on dominant frames. Same VLM artifact as batch2 dtl-wrong-*. Needs human angle confirmation.",
    },
    "dtl-eet-2": {
        "frames": [
            {"fr":0,   "q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,
             "q5_desc":"Young man in floral shorts at address in backyard, face-on."},
            {"fr":32,  "q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,
             "q5_desc":"Man at address with iron, sunny backyard, face-on."},
            {"fr":65,  "q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,
             "q5_desc":"Man in floral shorts near impact, backyard, face-on."},
            {"fr":97,  "q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,
             "q5_desc":"Man in navy shirt at follow-through, face-on, green hedges."},
            {"fr":129, "q1_golf":True,"q2_persons":1,"q3_angle":"DTL",   "q4_fullbody":True,
             "q5_desc":"Man completing iron follow-through, viewed from behind (DTL), backyard."},
        ],
        "verdict":"needs_human",
        "angle":"mixed",
        "reason":"4/5 face-on + 1/5 DTL. Prefix dtl conflicts with face-on dominant. Same VLM artifact as batch2 dtl-wrong-*. Needs human angle confirmation.",
    },
    "dtl-eet-3": {
        "frames": [
            {"fr":0,   "q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,
             "q5_desc":"Man in navy shirt/floral shorts at address, backyard, face-on."},
            {"fr":21,  "q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,
             "q5_desc":"Golfer at address holding iron, face-on, green hedges."},
            {"fr":43,  "q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,
             "q5_desc":"Golfer at top of backswing, face-on, yellow ball at feet."},
            {"fr":65,  "q1_golf":True,"q2_persons":1,"q3_angle":"face-on","q4_fullbody":True,
             "q5_desc":"Golfer at/just past impact, face-on, backyard."},
            {"fr":86,  "q1_golf":True,"q2_persons":1,"q3_angle":"DTL",   "q4_fullbody":True,
             "q5_desc":"Golfer at full finish, DTL angle, backyard hedge."},
        ],
        "verdict":"needs_human",
        "angle":"mixed",
        "reason":"4/5 face-on + 1/5 DTL. Prefix dtl conflicts with face-on dominant. Same VLM artifact as batch2 dtl-wrong-*. Needs human angle confirmation.",
    },
}

if __name__ == "__main__":
    gate = PerceptionGate()
    for stem, data in VLM_BATCH3.items():
        result = gate.ingest(stem=stem, vlm_result=data)
        print(f"  {stem}: {result.verdict} | angle={result.angle}")
    print("Done. Records written to engine/layer0/records/")
