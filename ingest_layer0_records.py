#!/usr/bin/env python3
"""
ingest_layer0_records.py
========================
Ingest pre-computed VLM verdicts from the parallel analysis run into
the Layer 0 gate records directory.

VLM results from the analysis task are hardcoded here (no re-running the VLM).
Cross-references shoulder-ratio from screening JSON where available.

Outputs:
  engine/layer0/records/<stem>.json  — one per video
  pipeline_output/normal_group_screening_v2.json  — updated with VLM verdicts
"""

import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine.layer0.perception_gate import PerceptionGate

PROJ = Path(__file__).resolve().parent

# ── VLM results (from parallel analysis, 2026-06-10) ─────────────────────────
# Exactly 60 frames analyzed (5 per video × 12 videos)

VLM_RESULTS = {
    "stodownload(20)": {
        "frames": [
            {"fr": 0,   "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Minimalist duplex apartment real estate — luxury bathroom with freestanding oval bathtub."},
            {"fr": 98,  "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Spa-like bathroom with freestanding oval tub through a doorway."},
            {"fr": 196, "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Top-down view of modern interior staircase with warm LED strip lighting."},
            {"fr": 295, "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Modern home upper landing with glass stair railing and warmly lit wooden staircase."},
            {"fr": 393, "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Minimalist bedroom with white duvet, patterned pillows, floating oak nightstand."},
        ],
        "verdict": "REJECT",
        "angle": "other",
        "reason": "No golf content. Chinese interior design/real estate video showcasing a minimalist duplex apartment.",
    },
    "stodownload(24)": {
        "frames": [
            {"fr": 0,   "q1_golf": False, "q2_persons": 1, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Man in dark gray suit examines open corner cabinet in kitchen showroom."},
            {"fr": 94,  "q1_golf": False, "q2_persons": 1, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Man interacts with rotating carousel storage in luxury kitchen showroom."},
            {"fr": 188, "q1_golf": False, "q2_persons": 1, "q3_angle": "other", "q4_fullbody": True,  "q5_desc": "Woman in yellow blazer reaches toward open burgundy storage cabinet in kitchen showroom."},
            {"fr": 282, "q1_golf": False, "q2_persons": 1, "q3_angle": "other", "q4_fullbody": True,  "q5_desc": "Person in tan blazer opens tall pantry cabinet in modern kitchen showroom."},
            {"fr": 376, "q1_golf": False, "q2_persons": 1, "q3_angle": "other", "q4_fullbody": True,  "q5_desc": "Woman demonstrates pull-out pantry storage in burgundy-and-wood kitchen showroom."},
        ],
        "verdict": "REJECT",
        "angle": "other",
        "reason": "No golf content. Kitchen furniture/cabinetry showroom promotional video.",
    },
    "stodownload(28)": {
        "frames": [
            {"fr": 0,   "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Luxury minimalist bathroom in sloped-roof loft with freestanding oval bathtub and skylights."},
            {"fr": 63,  "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Contemporary bathroom with white vessel sink, backlit arched mirror, herringbone wood wall."},
            {"fr": 127, "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Modern luxury attic bathroom with round freestanding tub beneath skylight."},
            {"fr": 190, "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Stunning modern luxury attic bathroom with round tub beneath large Velux skylight."},
            {"fr": 254, "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Bright minimalist attic loft living room with white boucle sectional sofa and skylights."},
        ],
        "verdict": "REJECT",
        "angle": "other",
        "reason": "No golf content. Interior design showcase of a sloped-roof loft home (bathrooms, living spaces).",
    },
    "stodownload(32)": {
        "frames": [
            {"fr": 0,   "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Empty modern room with dark cave-stone accent wall, herringbone wood floor, Chinese text overlay."},
            {"fr": 58,  "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Modern empty room with faux cave-stone feature wall, chevron floor, black lighting fixtures."},
            {"fr": 116, "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Monochromatic apartment with cave-stone accent wall, white kitchen cabinetry, grey sofa."},
            {"fr": 174, "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Dramatically lit cave-stone accent wall bathed in teal LED lighting with Chinese text."},
            {"fr": 232, "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Monochromatic apartment with natural stone accent wall, white cabinetry, herringbone floor."},
        ],
        "verdict": "REJECT",
        "angle": "other",
        "reason": "No golf content. Chinese interior design concept video — black-and-white cave stone (洞石风) residential design.",
    },
    "stodownload(43)": {
        "frames": [
            {"fr": 0,   "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Hand demonstrates rose gold pull-out kitchen faucet producing waterfall water stream over dark granite sink."},
            {"fr": 61,  "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Hand holds detached pull-out wand from rose gold smart kitchen faucet, water streaming into sink."},
            {"fr": 123, "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Luxury rose gold kitchen faucet with digital display over dark granite sink, pull-out spray wand in use."},
            {"fr": 185, "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Dimly lit modern luxury kitchen with dark wood cabinetry and light countertops, no people."},
            {"fr": 247, "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Sleek contemporary open-plan kitchen with large island, warm wood accents, bar stools."},
        ],
        "verdict": "REJECT",
        "angle": "other",
        "reason": "No golf content. Kitchen product promo — rose gold smart pull-out faucet with waterfall stream mode.",
    },
    "stodownload(45)": {
        "frames": [
            {"fr": 1,   "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Hand reveals hidden staircase under sliding wooden kitchen island in luxurious villa."},
            {"fr": 79,  "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Sage-green kitchen island in modern open-plan villa with Chinese text about hidden door feature."},
            {"fr": 158, "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Wooden kitchen island top slid open on rails, revealing glowing orange-lit hidden staircase below."},
            {"fr": 237, "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Hidden staircase under sliding table, hand/orange sleeve visible accessing secret underground passage."},
            {"fr": 316, "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Top-down view of narrow traditional wooden staircase with dual handrails leading down."},
        ],
        "verdict": "REJECT",
        "angle": "other",
        "reason": "No golf content. Chinese villa promotional video — secret hidden door/staircase under sliding kitchen island.",
    },
    "stodownload(49)": {
        "frames": [
            {"fr": 0,   "q1_golf": False, "q2_persons": 1, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Hand and foot partially visible pointing toward hidden-slot faucet built into curved stone basin."},
            {"fr": 89,  "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Ground-level photo of large circular stone/concrete object on stone-tiled floor."},
            {"fr": 178, "q1_golf": False, "q2_persons": 1, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Partially visible hand holds pull-out spray head from rose-gold kitchen faucet as water streams into dark undermount sink."},
            {"fr": 267, "q1_golf": False, "q2_persons": 1, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Dimly lit kitchen sink with tall gooseneck faucet and partially visible human hand gripping sink edge."},
            {"fr": 356, "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Heavily blurred image through wet/dirty glass surface with indistinct background."},
        ],
        "verdict": "REJECT",
        "angle": "other",
        "reason": "No golf content. Kitchen/bathroom faucet product video with only partial hand visible in some frames.",
    },
    "stodownload(53)": {
        "frames": [
            {"fr": 0,   "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "3D architectural rendering of luxurious marble bathroom with Chinese installation-height measurements."},
            {"fr": 51,  "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Luxury CGI bathroom render with gold-accented marble walls and Chinese fixture-height annotations."},
            {"fr": 102, "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Luxury marble bathroom interior design render featuring gold fixtures and Chinese measurement guidelines."},
            {"fr": 153, "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Luxury marble bathroom render with gold heated towel rail and annotated Chinese height measurements."},
            {"fr": 204, "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Luxury marble bathroom render with gold fixtures and Chinese-language height annotations with golden reference line."},
        ],
        "verdict": "REJECT",
        "angle": "other",
        "reason": "No golf content. Chinese bathroom interior design tutorial — 3D renders with fixture installation height annotations. No real human, no golf swing.",
    },
    "stodownload(60)": {
        "frames": [
            {"fr": 0,   "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Chinese lighting advertisement — open-plan kitchen/dining at 4000K neutral color temperature."},
            {"fr": 51,  "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Smart lighting promo — modern kitchen/dining bathed in 2500K warm golden-amber light."},
            {"fr": 103, "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Chinese lighting ad displaying modern interior at 3500K with warm-to-cool UI overlay."},
            {"fr": 154, "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Smart lighting ad — minimalist open-plan dining/living at 4000K with overlaid scale graphic."},
            {"fr": 206, "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Stylish modern kitchen/dining with Chinese overlay graphics demonstrating 5500K cool lighting."},
        ],
        "verdict": "REJECT",
        "angle": "other",
        "reason": "No golf content. Chinese smart-lighting promotional video showing interior rooms at varying color temperatures (2500K–6000K).",
    },
    "stodownload(64)": {
        "frames": [
            {"fr": 0,   "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Sleek minimalist kitchen with Bosch appliances, marble island, dining table and circular pendant light."},
            {"fr": 71,  "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Modern kitchen island with built-in induction cooktop photographed against flat-panel white cabinetry."},
            {"fr": 142, "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Minimalist kitchen with dark stone backsplash, handle-less white cabinetry, dried branch near sink."},
            {"fr": 213, "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Bright minimalist dining room with rounded light-wood table overlooking snowy winter landscape."},
            {"fr": 284, "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Dark stone kitchen island with built-in outlets and pop-up cooktop, light wood dining table with white chairs."},
        ],
        "verdict": "REJECT",
        "angle": "other",
        "reason": "No golf content. High-end kitchen and dining room interior design photography. No people present.",
    },
    "stodownload(91)": {
        "frames": [
            {"fr": 0,   "q1_golf": False, "q2_persons": 1, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Woman in gray blazer demonstrates laterally-sliding marble kitchen island in modern showroom."},
            {"fr": 80,  "q1_golf": False, "q2_persons": 1, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Woman in beige blazer stands beside sliding marble kitchen island in sleek modern kitchen showroom."},
            {"fr": 160, "q1_golf": False, "q2_persons": 1, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Woman browses luxury kitchen showroom featuring marble island with retractable faucet."},
            {"fr": 240, "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Large marble counter/wall panel with gold LED accent lighting in luxury showroom or hotel lobby."},
            {"fr": 320, "q1_golf": False, "q2_persons": 0, "q3_angle": "other", "q4_fullbody": False, "q5_desc": "Organized double-layer kitchen drawer with warm orange wooden insert holding sorted silverware."},
        ],
        "verdict": "REJECT",
        "angle": "other",
        "reason": "No golf content. Kitchen showroom product demonstration with female presenter. No golf activity.",
    },
    "test-dwontheline": {
        "frames": [
            {"fr": 0,  "q1_golf": True,  "q2_persons": 1, "q3_angle": "face-on", "q4_fullbody": True,  "q5_desc": "Man in dark polo and khaki pants at golf address position in suburban backyard, gripping iron club."},
            {"fr": 24, "q1_golf": True,  "q2_persons": 1, "q3_angle": "face-on", "q4_fullbody": True,  "q5_desc": "Man in casual clothing performs golf iron swing at impact/follow-through in sunny suburban backyard."},
            {"fr": 48, "q1_golf": True,  "q2_persons": 1, "q3_angle": "DTL",     "q4_fullbody": True,  "q5_desc": "Man in gray t-shirt performs golf swing from behind (DTL) in suburban backyard with pine tree."},
            {"fr": 72, "q1_golf": True,  "q2_persons": 1, "q3_angle": "DTL",     "q4_fullbody": True,  "q5_desc": "Golfer in branded golf shirt completes full follow-through from DTL in residential backyard."},
            {"fr": 96, "q1_golf": True,  "q2_persons": 1, "q3_angle": "DTL",     "q4_fullbody": True,  "q5_desc": "Golfer in ResinGear branded shirt completes iron follow-through from DTL camera position."},
        ],
        "verdict": "needs_human",
        "angle": "inconsistent",
        "reason": "All 5 frames: real golfer, 1 person, full body visible (hard criteria PASS). BUT camera angle inconsistent: fr0/fr24 = face-on; fr48/fr72/fr96 = DTL. Video likely contains two swings from different angles. Needs human to split into single-angle segments before pipeline ingestion.",
    },
}

# ── Load shoulder-ratio data from screening JSON ──────────────────────────────

def load_sh_ratios():
    screen_path = PROJ / "pipeline_output/normal_group_screening.json"
    if not screen_path.exists():
        return {}
    with open(screen_path) as f:
        data = json.load(f)
    return {
        Path(r["file"]).stem: (r.get("sh_ratio"), r.get("angle"))
        for r in data
    }


def main():
    gate = PerceptionGate()
    sh_map = load_sh_ratios()

    print("Ingesting Layer 0 records...")
    results_summary = []

    for stem, vlm in VLM_RESULTS.items():
        sh_ratio, sh_angle = sh_map.get(stem, (None, None))
        result = gate.ingest(stem, vlm, sh_ratio=sh_ratio, sh_angle=sh_angle)
        print(f"  {stem:30s}  {result.verdict:14s}  angle={result.angle}"
              + (" ⚠ angle_conflict" if result.angle_conflict else ""))
        results_summary.append(result)

    # Update screening JSON with VLM verdicts
    screen_path = PROJ / "pipeline_output/normal_group_screening.json"
    if screen_path.exists():
        with open(screen_path) as f:
            screening = json.load(f)
        for row in screening:
            stem_key = Path(row["file"]).stem
            rec = gate.load(stem_key)
            if rec:
                row["layer0_verdict"] = rec.verdict
                row["layer0_angle"]   = rec.angle
                row["layer0_reason"]  = rec.reason[:120]
        v2_path = PROJ / "pipeline_output/normal_group_screening_v2.json"
        with open(v2_path, "w") as f:
            json.dump(screening, f, indent=2)
        print(f"\nScreening v2 saved: {v2_path}")

    print(f"\nRecords written to: {PROJ}/engine/layer0/records/")
    print(f"Total records: {len(list((PROJ / 'engine/layer0/records').glob('*.json')))}")

    # Print summary table
    print("\n" + gate.summary_table([s for s in VLM_RESULTS]))

    # Update PROGRESS.log
    prog = PROJ / "PROGRESS.log"
    import datetime
    with open(prog, "a") as f:
        ts = datetime.datetime.now().isoformat()
        pass_count = sum(1 for r in results_summary if r.verdict == "PASS")
        reject_count = sum(1 for r in results_summary if r.verdict == "REJECT")
        nh_count = sum(1 for r in results_summary if r.verdict == "needs_human")
        f.write(f"{ts}  Layer 0 gate: {pass_count} PASS, {reject_count} REJECT, "
                f"{nh_count} needs_human\n")


if __name__ == "__main__":
    main()
