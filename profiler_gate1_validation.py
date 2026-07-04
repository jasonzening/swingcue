#!/usr/bin/env python3
"""
profiler_gate1_validation.py
Checkpoint 1 validation: camera_view detection on 10 known-GT clips.

Ground truth:
  face-on: 201015, 201039, 201047, fo-eet-1, fo-eet-2, fo-eet-3
  DTL:     201054, 201058, dtl-eet-2, dtl-eet-3
  (dtl-eet-1 cache missing — skipped)

Outputs:
  stdout: per-clip table with 3 evidences + decision + correct?
  output/profiler/gate1_camera_view_validation.json
"""

import sys, json, pathlib

PROJ = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ))

from engine.profiler.camera_view import detect_camera_view, CameraViewResult

CACHE = PROJ / "engine" / "kp_cache"
OUT   = PROJ / "output" / "profiler"
OUT.mkdir(parents=True, exist_ok=True)

TEST_CLIPS = [
    (CACHE / "Videos2026-06-09_201015_827.json", "face_on",  "201015"),
    (CACHE / "Videos2026-06-09_201039_231.json", "face_on",  "201039"),
    (CACHE / "Videos2026-06-09_201047_915.json", "face_on",  "201047"),
    (CACHE / "Videos2026-06-09_201054_561.json", "dtl",      "201054"),
    (CACHE / "Videos2026-06-09_201058_697.json", "dtl",      "201058"),
    (CACHE / "batch3" / "dtl-eet-2.json",        "dtl",      "dtl-eet-2"),
    (CACHE / "batch3" / "dtl-eet-3.json",        "dtl",      "dtl-eet-3"),
    (CACHE / "batch3" / "fo-eet-1.json",         "face_on",  "fo-eet-1"),
    (CACHE / "batch3" / "fo-eet-2.json",         "face_on",  "fo-eet-2"),
    (CACHE / "batch3" / "fo-eet-3.json",         "face_on",  "fo-eet-3"),
]

HEADER = (
    f"{'label':12s} {'GT':8s} {'sh_lat_ratio':>12s}  "
    f"{'vote_sh':>7s} {'sh_asym':>7s} {'vote_asym':>9s}  "
    f"{'view':8s} {'conf':>6s} {'needs_h':>7s} {'ok?':>4s}"
)
print(HEADER)
print("-" * len(HEADER))

rows = []
correct_count = 0
uncertain_count = 0

for path, gt, label in TEST_CLIPS:
    if not path.exists():
        print(f"  {label:12s} {gt:8s}  [MISSING cache file — skipped]")
        continue

    kp_json = json.load(open(path))
    res: CameraViewResult = detect_camera_view(kp_json)
    ev = res.evidence

    predicted = res.camera_view  # face_on / dtl / uncertain
    if predicted == gt:
        mark = "✓"; correct_count += 1
    elif predicted == "uncertain":
        mark = "?"; uncertain_count += 1
    else:
        mark = "✗"

    sh_lat_s  = f"{ev.sh_lat_ratio:.4f}" if ev.sh_lat_ratio is not None else "  N/A"
    sh_asym_s = f"{ev.sh_asym:.4f}"      if ev.sh_asym      is not None else "  N/A"

    print(
        f"{label:12s} {gt:8s} {sh_lat_s:>12s}  "
        f"{ev.vote_sh_lat:>7d} {sh_asym_s:>7s} {ev.vote_sh_asym:>9d}  "
        f"{predicted:8s} {res.confidence:>6.3f} {str(res.needs_human):>7s} {mark:>4s}"
    )

    rows.append({
        "label":         label,
        "gt":            gt,
        "predicted":     predicted,
        "confidence":    res.confidence,
        "correct":       mark,
        "needs_human":   res.needs_human,
        "note":          res.note,
        "evidence": {
            "sh_lat_ratio":  ev.sh_lat_ratio,
            "sh_asym":       ev.sh_asym,
            "face_occ":      ev.face_occ,
            "vote_sh_lat":   ev.vote_sh_lat,
            "vote_sh_asym":  ev.vote_sh_asym,
            "vote_combined": ev.vote_combined,
            "valid_frames":  ev.valid_frames,
            "window_frames": ev.window_frames,
        }
    })

n_total  = len(rows)
n_correct = correct_count
n_uncertain = uncertain_count
n_wrong   = n_total - n_correct - n_uncertain

print("-" * len(HEADER))
print(f"\nSummary: {n_total} clips | correct={n_correct} | uncertain={n_uncertain} | wrong={n_wrong}")
print(f"Accuracy (excl uncertain): {n_correct}/{n_total-n_uncertain} = "
      f"{n_correct/(max(n_total-n_uncertain,1)):.0%}")
print(f"Overall accuracy:          {n_correct}/{n_total} = {n_correct/max(n_total,1):.0%}")

# Save JSON
report = {
    "module":        "camera_view v0.1",
    "checkpoint":    1,
    "n_total":       n_total,
    "n_correct":     n_correct,
    "n_uncertain":   n_uncertain,
    "n_wrong":       n_wrong,
    "accuracy_all":  round(n_correct / max(n_total, 1), 4),
    "clips":         rows,
}
out_path = OUT / "gate1_camera_view_validation.json"
out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(f"\nReport saved: {out_path}")
