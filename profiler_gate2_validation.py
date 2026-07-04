#!/usr/bin/env python3
"""
profiler_gate2_validation.py
Checkpoint 2: Full VideoProfile (identity card) for all 11 test clips.

Ground truth for 11 clips:
  201015: face-on, single, full-swing, right-handed
  201039: face-on, single, full-swing, right-handed
  201047: face-on, single, full-swing, right-handed
  201054: DTL,     single, full-swing, right-handed
  201058: DTL,     single, full-swing, right-handed
  dtl-eet-1: DTL,  single, full-swing, right-handed
  dtl-eet-2: DTL,  single, full-swing, right-handed
  dtl-eet-3: DTL,  single, full-swing, right-handed
  fo-eet-1:  face-on, single, full-swing, right-handed
  fo-eet-2:  face-on, single, full-swing, right-handed
  fo-eet-3:  face-on, single, full-swing, right-handed

Video dimensions for old clips (1920x1080 assumed — actual values from file).
Batch3 clips: check from video files.

Outputs:
  stdout: per-clip summary table
  output/profiler/gate2_identity_cards.json  — 11 identity cards
  output/profiler/gate2_accuracy_report.json — field-by-field accuracy
"""

import sys, json, pathlib
import cv2

PROJ = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ))

from engine.profiler.video_profiler import VideoProfiler

CACHE  = PROJ / "engine" / "kp_cache"
L0_DIR = PROJ / "engine" / "layer0" / "records"
OUT    = PROJ / "output" / "profiler"
OUT.mkdir(parents=True, exist_ok=True)

VIDEO_DIR_OLD = pathlib.Path("/mnt/c/Users/jason/Zening/Swingcue/Videos2026-06-09")
VIDEO_DIR_EET = pathlib.Path("/mnt/c/Users/jason/Zening/Swingcue/Video")

GT = {
    "201015":   {"camera_view": "face_on", "layout": "single", "swing_type": "full_swing", "handedness": "right"},
    "201039":   {"camera_view": "face_on", "layout": "single", "swing_type": "full_swing", "handedness": "right"},
    "201047":   {"camera_view": "face_on", "layout": "single", "swing_type": "full_swing", "handedness": "right"},
    "201054":   {"camera_view": "dtl",     "layout": "single", "swing_type": "full_swing", "handedness": "right"},
    "201058":   {"camera_view": "dtl",     "layout": "single", "swing_type": "full_swing", "handedness": "right"},
    "dtl-eet-1":{"camera_view": "dtl",     "layout": "single", "swing_type": "full_swing", "handedness": "right"},
    "dtl-eet-2":{"camera_view": "dtl",     "layout": "single", "swing_type": "full_swing", "handedness": "right"},
    "dtl-eet-3":{"camera_view": "dtl",     "layout": "single", "swing_type": "full_swing", "handedness": "right"},
    "fo-eet-1": {"camera_view": "face_on", "layout": "single", "swing_type": "full_swing", "handedness": "right"},
    "fo-eet-2": {"camera_view": "face_on", "layout": "single", "swing_type": "full_swing", "handedness": "right"},
    "fo-eet-3": {"camera_view": "face_on", "layout": "single", "swing_type": "full_swing", "handedness": "right"},
}

CLIPS = [
    # (label, kp_cache_path, video_path, layer0_stem)
    ("201015",    CACHE/"Videos2026-06-09_201015_827.json", VIDEO_DIR_OLD/"Videos2026-06-09_201015_827.mp4", "Videos2026-06-09_201015_827"),
    ("201039",    CACHE/"Videos2026-06-09_201039_231.json", VIDEO_DIR_OLD/"Videos2026-06-09_201039_231.mp4", "Videos2026-06-09_201039_231"),
    ("201047",    CACHE/"Videos2026-06-09_201047_915.json", VIDEO_DIR_OLD/"Videos2026-06-09_201047_915.mp4", "Videos2026-06-09_201047_915"),
    ("201054",    CACHE/"Videos2026-06-09_201054_561.json", VIDEO_DIR_OLD/"Videos2026-06-09_201054_561.mp4", "Videos2026-06-09_201054_561"),
    ("201058",    CACHE/"Videos2026-06-09_201058_697.json", VIDEO_DIR_OLD/"Videos2026-06-09_201058_697.mp4", "Videos2026-06-09_201058_697"),
    ("dtl-eet-1", CACHE/"batch3/dtl-eet-1.json", VIDEO_DIR_EET/"dtl-eet-1.mp4", "dtl-eet-1"),
    ("dtl-eet-2", CACHE/"batch3/dtl-eet-2.json", VIDEO_DIR_EET/"dtl-eet-2.mp4", "dtl-eet-2"),
    ("dtl-eet-3", CACHE/"batch3/dtl-eet-3.json", VIDEO_DIR_EET/"dtl-eet-3.mp4", "dtl-eet-3"),
    ("fo-eet-1",  CACHE/"batch3/fo-eet-1.json",  VIDEO_DIR_EET/"fo-eet-1.mp4",  "fo-eet-1"),
    ("fo-eet-2",  CACHE/"batch3/fo-eet-2.json",  VIDEO_DIR_EET/"fo-eet-2.mp4",  "fo-eet-2"),
    ("fo-eet-3",  CACHE/"batch3/fo-eet-3.json",  VIDEO_DIR_EET/"fo-eet-3.mp4",  "fo-eet-3"),
]

FIELDS = ["camera_view", "layout", "swing_type", "handedness"]

def get_video_dims(video_path):
    """Get (width, height) from video file."""
    if not pathlib.Path(video_path).exists():
        return None, None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h

def load_layer0(stem):
    """Load existing layer0 record if available."""
    for ext in ["", ".json"]:
        p = L0_DIR / f"{stem}{ext}"
        if p.exists():
            return json.load(open(p))
    # Try without ext
    for f in L0_DIR.iterdir():
        if f.stem == stem:
            return json.load(open(f))
    return None

profiler = VideoProfiler()
cards = []
field_results = {f: {"correct": 0, "wrong": 0, "uncertain": 0, "needs_human": 0} for f in FIELDS}

print(f"\n{'='*90}")
print(f"Gate 2 — Full VideoProfile validation on 11 test clips")
print(f"{'='*90}")

for label, kp_path, video_path, l0_stem in CLIPS:
    if not pathlib.Path(kp_path).exists():
        print(f"\n[SKIP] {label}: kp_cache missing {kp_path}")
        continue

    kp_json = json.load(open(kp_path))
    w, h    = get_video_dims(video_path)
    l0_rec  = load_layer0(l0_stem)

    profile = profiler.profile_from_kp_json(
        video_id      = label,
        kp_json       = kp_json,
        video_width   = w,
        video_height  = h,
        layer0_record = l0_rec,
        split_hint    = "single",   # all 11 test clips are single-panel
    )

    card = profile.to_dict()
    cards.append(card)
    gt = GT[label]

    # ── Print header per clip ────────────────────────────────────────────────
    print(f"\n┌─ {label:12s}  {w or '?'}x{h or '?'}  frames={len(kp_json['frames'])}")
    print(f"│  is_golf_swing={profile.is_golf_swing}  persons={profile.persons}")

    marks = {}
    for f in FIELDS:
        predicted = getattr(profile, f)
        expected  = gt[f]
        conf_val  = profile.confidence.get(f, 0)
        nh        = f in profile.needs_human

        if predicted == expected:
            m = "✓"; field_results[f]["correct"] += 1
        elif predicted in ("unknown", "uncertain") or nh:
            m = "?"; field_results[f]["uncertain"] += 1
        else:
            m = "✗"; field_results[f]["wrong"] += 1
        if nh:
            field_results[f]["needs_human"] += 1
        marks[f] = m
        print(f"│  {f:14s}: pred={predicted:12s} exp={expected:12s} "
              f"conf={conf_val:.2f} nh={nh} {m}")

    print(f"│  camera_profile: center=({profile.camera_profile['subject_center_x']}, "
          f"{profile.camera_profile['subject_center_y']})  "
          f"h_ratio={profile.camera_profile['subject_height_ratio']}  "
          f"cam_h={profile.camera_profile['camera_height']}")
    print(f"│  needs_human: {profile.needs_human}")
    print(f"│  notes: {profile.notes}")
    print(f"└─ {' '.join(f'{f[0].upper()}:{marks[f]}' for f in FIELDS)}")

# ── Summary ───────────────────────────────────────────────────────────────────
n = len(cards)
print(f"\n{'='*70}")
print(f"FIELD ACCURACY SUMMARY (n={n})")
print(f"{'Field':18s} {'✓':>5s} {'✗':>5s} {'?':>5s}  {'acc':>6s}  {'acc+nh':>8s}")
print("-"*60)
for f in FIELDS:
    r = field_results[f]
    c, w2, u = r["correct"], r["wrong"], r["uncertain"]
    acc = f"{c}/{c+w2}" if (c+w2) > 0 else "n/a"
    acc_all = f"{c}/{n}"
    print(f"{f:18s} {c:>5d} {w2:>5d} {u:>5d}  {acc:>6s}  {acc_all:>8s}")

# ── Save outputs ──────────────────────────────────────────────────────────────
cards_out = OUT / "gate2_identity_cards.json"
cards_out.write_text(json.dumps(cards, ensure_ascii=False, indent=2))

report = {
    "module":       "VideoProfiler v0.1",
    "checkpoint":   2,
    "n_clips":      n,
    "field_results": field_results,
    "cards":        cards,
}
(OUT / "gate2_accuracy_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

print(f"\nOutputs:")
print(f"  {cards_out}")
print(f"  {OUT / 'gate2_accuracy_report.json'}")
