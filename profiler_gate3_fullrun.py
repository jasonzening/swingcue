#!/usr/bin/env python3
"""
profiler_gate3_fullrun.py
Checkpoint 3: Full library body check — 137 clips from 教学视频/

Strategy:
  - Sampled RTMPose: extract only 24 strategic frames per clip (not all frames)
    → ~12s per clip vs ~800s full extraction → feasible in ~30 min
  - Swing type: frame-diff motion proxy on raw video (no RTMPose needed)
  - Layout: from inventory VLM tag (pre-validated, reliable)
  - For split-screen: profile left half and right half separately
  - camera_view: geometric from sampled kps
  - handedness: from sampled kp wrist trajectory
  - camera_profile: from sampled kps

Output:
  output/video_profile_full.json — all identity cards
  output/video_profile_full.md  — summary table + statistics

Notes:
  - model loaded ONCE, clips processed sequentially
  - progress printed every 10 clips
  - on any clip error: store error card, continue
"""

import sys, json, time, math
from pathlib import Path
import cv2, numpy as np
from scipy.signal import find_peaks, savgol_filter

PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ))

from engine.profiler.camera_view import detect_camera_view
from engine.profiler.camera_profile import compute_camera_profile
from engine.profiler.video_profiler import _probe_handedness, NEEDS_HUMAN_CONF_THR

# ── Config ──────────────────────────────────────────────────────────────────
VIDEO_DIR = Path("/mnt/c/Users/jason/Zening/Swingcue/教学视频")
OUT_DIR   = PROJ / "output"
INV_JSON  = PROJ / "output" / "inventory_2026-07-03.json"

N_SAMPLE_FRAMES = 24   # RTMPose frames to extract per clip (per panel)
KP_THR          = 0.25
SWING_THR       = 12.0  # motion proxy peak threshold (pixels/frame avg)

# ── Model loader (singleton) ─────────────────────────────────────────────────
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from engine.a_measurement.pose_pipeline import PosePipeline
        print("  [MODEL] Loading RTMPose...", flush=True)
        _pipeline = PosePipeline(device="cuda")
        print("  [MODEL] Ready.", flush=True)
    return _pipeline

# ── Sampled kp extraction ────────────────────────────────────────────────────

def extract_sampled_kp(video_path: Path, n_samples: int = N_SAMPLE_FRAMES,
                       x_crop: tuple = None) -> dict:
    """
    Extract RTMPose keypoints from n_samples strategic frames.
    x_crop: (x0, x1) pixel columns to crop (for split-screen halves).
    Returns kp_json dict (same structure as full extraction).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"frames": [], "stats": {"source_fps": 30, "total_frames": 0}}
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if n_frames < 4:
        cap.release()
        return {"frames": [], "stats": {"source_fps": fps, "total_frames": n_frames}}

    # Strategic sampling: weight toward first 30% (address window) + uniform rest
    early_n = min(12, n_samples // 2)
    rest_n  = n_samples - early_n
    early_end = max(int(n_frames * 0.35), early_n + 1)
    early_idxs = sorted(set(int(n_frames * 0.02 + (early_end - n_frames*0.02) * i / max(early_n-1,1))
                            for i in range(early_n)))
    rest_idxs  = sorted(set(int(early_end + (n_frames - early_end) * i / max(rest_n-1,1))
                            for i in range(rest_n)))
    sample_idxs = sorted(set(early_idxs + rest_idxs))
    sample_idxs = [min(i, n_frames-1) for i in sample_idxs]

    # Read selected frames
    frames_bgr = {}
    for fi in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret: continue
        if x_crop is not None:
            x0, x1 = x_crop
            frame = frame[:, x0:x1]
        frames_bgr[fi] = frame
    cap.release()

    if not frames_bgr:
        return {"frames": [], "stats": {"source_fps": fps, "total_frames": n_frames}}

    # Run RTMPose on batch of frames (using _get_body() direct call)
    pipeline = get_pipeline()
    kp_frames = []
    for fi in sorted(frames_bgr.keys()):
        frame = frames_bgr[fi]
        kps_list = _infer_single_frame(pipeline, frame)
        if kps_list:
            kp_frames.append({"frame_idx": fi, "persons": [{"keypoints": kps_list[0]}]})
        else:
            kp_frames.append({"frame_idx": fi, "persons": []})

    return {
        "frames": kp_frames,
        "stats":  {"source_fps": fps, "total_frames": n_frames},
        "sampled": True,
        "n_sampled": len(kp_frames),
    }


def _infer_single_frame(pipeline, frame_bgr):
    """
    Run RTMPose on a single BGR frame. Returns list of kp dicts.
    Uses pipeline._get_body() → rtmlib Body model directly.
    """
    from engine.a_measurement.pose_pipeline import JOINT_NAMES
    try:
        body = pipeline._get_body()
        kps_arr, scores_arr = body(frame_bgr)
        if kps_arr is None or len(kps_arr) == 0:
            return []
        kps    = kps_arr[0]
        scores = scores_arr[0]
        result = {}
        for i, name in enumerate(JOINT_NAMES):
            if i < len(kps):
                result[name] = {
                    "x":     float(kps[i][0]),
                    "y":     float(kps[i][1]),
                    "score": float(scores[i]) if i < len(scores) else 0.0,
                }
        return [result]
    except Exception as e:
        return []


# ── Motion proxy (swing_type) without RTMPose ─────────────────────────────────

def detect_swing_type_video(video_path: Path, x_crop: tuple = None) -> dict:
    """
    Compute swing type from frame-diff motion proxy on raw video.
    x_crop: (x0, x1) for split-screen.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"swing_type": "unknown", "confidence": 0.1, "peak_motion_px": None}

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    y0 = int(h * 0.30)  # skip top 30% (UI/text overlays)

    ret, prev = cap.read()
    if not ret:
        cap.release()
        return {"swing_type": "unknown", "confidence": 0.1, "peak_motion_px": None}

    if x_crop: prev = prev[:, x_crop[0]:x_crop[1]]
    prev_gray = cv2.cvtColor(prev[y0:], cv2.COLOR_BGR2GRAY).astype(np.float32)
    motion = [0.0]

    step = max(1, n // 150)  # sample at most 150 frames for speed
    fi = 1
    while True:
        if fi % step != 0:
            fi += 1
            if not cap.grab(): break
            continue
        ret, frame = cap.read()
        if not ret: break
        if x_crop: frame = frame[:, x_crop[0]:x_crop[1]]
        gray = cv2.cvtColor(frame[y0:], cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.mean(np.abs(gray - prev_gray))
        motion.append(diff)
        prev_gray = gray
        fi += 1
    cap.release()

    if len(motion) < 4:
        return {"swing_type": "unknown", "confidence": 0.2, "peak_motion_px": 0.0}

    # Smooth
    wl = min(5, len(motion)-1 if len(motion)%2==0 else len(motion))
    if wl >= 3:
        try:
            smoothed = savgol_filter(motion, min(wl, 5), 2)
        except Exception:
            smoothed = np.array(motion)
    else:
        smoothed = np.array(motion)

    peak = float(np.max(smoothed))

    if peak >= SWING_THR * 2.0:
        st = "full_swing"; conf = 0.90
    elif peak >= SWING_THR:
        st = "full_swing"; conf = 0.75
    elif peak >= SWING_THR * 0.5:
        st = "mixed"; conf = 0.60
    else:
        st = "static_demo"; conf = 0.88

    return {"swing_type": st, "confidence": round(conf, 3), "peak_motion_px": round(peak, 2)}


# ── Auto split detection ──────────────────────────────────────────────────────

def get_split_x(video_path: Path, is_split: bool) -> tuple:
    """
    Returns (left_crop, right_crop) where each is (x0, x1) pixel range.
    For single clips: returns (None, None).
    For split: reads first non-black frame and finds vertical split seam.
    Falls back to 50/50 if seam not found.
    """
    if not is_split:
        return None, None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        # Assume 50/50
        return (0, 540), (540, 1080)  # guessed

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * 0.5))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        half = w // 2
        return (0, half), (half, w)

    # Look for vertical seam: column with lowest average gradient
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Compute vertical column profile: variance (low = seam)
    col_var = np.var(gray, axis=0).astype(float)
    # Search in middle 30% of width
    lo = int(w * 0.35); hi = int(w * 0.65)
    mid_var = col_var[lo:hi]
    seam_rel = int(np.argmin(mid_var))
    seam = lo + seam_rel

    # Use 5% margin from seam
    left_crop  = (0,    seam)
    right_crop = (seam, w)
    return left_crop, right_crop


# ── Build full VideoProfile dict ──────────────────────────────────────────────

def profile_clip(video_id: str, video_path: Path, inv_rec: dict,
                 x_crop=None, parent_id=None, side=None) -> dict:
    """
    Profile a single video (or split half).
    Returns identity card dict.
    """
    w_full = inv_rec.get("w", None)
    h_full = inv_rec.get("h", None)
    # Adjust width for cropped half
    if x_crop is not None and w_full is not None:
        w_use = x_crop[1] - x_crop[0]
    else:
        w_use = w_full

    needs_human = []
    conf = {}

    # ── 1. is_golf_swing + persons (from inventory VLM) ──────────────────────
    is_golf_swing = True  # all clips in 教学视频 are golf teaching, treat as True
    persons       = inv_rec.get("persons", 1)

    # ── 2. Layout (from inventory, high confidence) ───────────────────────────
    inv_split = inv_rec.get("split", "single")
    if side is not None:
        layout = "split_screen"
        conf["layout"] = 0.90
    elif inv_split == "split-screen":
        layout = "split_screen"
        conf["layout"] = 0.85
    else:
        layout = "single"
        conf["layout"] = 0.85

    # ── 3. Swing type (motion proxy on raw video) ─────────────────────────────
    swing_res = detect_swing_type_video(video_path, x_crop=x_crop)
    swing_type = swing_res["swing_type"]
    conf["swing_type"] = swing_res["confidence"]
    if conf["swing_type"] < NEEDS_HUMAN_CONF_THR:
        needs_human.append("swing_type")

    # ── 4-6. Geometric fields (need RTMPose on sample frames) ─────────────────
    kp_json = extract_sampled_kp(video_path, x_crop=x_crop)
    has_kp = len(kp_json.get("frames", [])) >= 4

    if has_kp:
        # camera_view
        cam_res = detect_camera_view(kp_json)
        camera_view = cam_res.camera_view
        conf["camera_view"] = cam_res.confidence
        if camera_view == "uncertain" or cam_res.needs_human:
            needs_human.append("camera_view")

        # handedness
        handedness, hand_conf, _ = _probe_handedness(kp_json, camera_view=camera_view)
        conf["handedness"] = hand_conf
        if handedness == "unknown" or hand_conf < NEEDS_HUMAN_CONF_THR:
            needs_human.append("handedness")

        # camera_profile
        cp = compute_camera_profile(kp_json, video_width=w_use, video_height=h_full)
        camera_profile = {
            "subject_center_x":    cp["subject_center_x"],
            "subject_center_y":    cp["subject_center_y"],
            "subject_height_ratio": cp["subject_height_ratio"],
            "camera_height":       cp["camera_height"],
        }
        conf["camera_profile"] = cp["confidence"]
        if cp["confidence"] < NEEDS_HUMAN_CONF_THR:
            needs_human.append("camera_profile")

    else:
        # No valid kp → fall back to inventory for camera_view
        inv_angle = inv_rec.get("angle", "unsure")
        angle_map = {"face-on": "face_on", "DTL": "dtl", "unsure": "uncertain", "other": "other"}
        camera_view  = angle_map.get(inv_angle, "uncertain")
        conf["camera_view"] = 0.50   # VLM-only, lower confidence
        needs_human.append("camera_view")
        handedness = "unknown"; conf["handedness"] = 0.20
        needs_human.append("handedness")
        camera_profile = {"subject_center_x": None, "subject_center_y": None,
                          "subject_height_ratio": None, "camera_height": "unknown"}
        conf["camera_profile"] = 0.10
        needs_human.append("camera_profile")

    conf = {k: round(v, 3) for k, v in conf.items()}

    # ── Notes ─────────────────────────────────────────────────────────────────
    view_str  = camera_view.replace("_", "-")
    lay_str   = layout.replace("_", "-")
    swing_str = swing_type.replace("_", " ")
    notes = f"{view_str} {lay_str} {swing_str} ({handedness}-handed)"
    if side:
        notes += f" [{side}]"

    card = {
        "video_id":      video_id,
        "parent_video":  parent_id,
        "side":          side,        # "left" | "right" | None
        "source_file":   inv_rec.get("rel", ""),
        "is_golf_swing": is_golf_swing,
        "persons":       persons,
        "layout":        layout,
        "camera_view":   camera_view,
        "swing_type":    swing_type,
        "handedness":    handedness,
        "camera_profile": camera_profile,
        "confidence":    conf,
        "needs_human":   sorted(set(needs_human)),
        "notes":         notes,
    }
    return card


# ── Main batch loop ───────────────────────────────────────────────────────────

def main():
    inv    = json.load(open(INV_JSON))
    records = inv["records"]

    # Build video path map: rel → full path
    # All videos are in VIDEO_DIR (flat, plus dtl-1/ subdir)
    all_mp4 = list(VIDEO_DIR.rglob("*.mp4"))
    path_by_stem = {p.name: p for p in all_mp4}
    path_by_stem_noext = {p.stem: p for p in all_mp4}

    # Preload model once
    get_pipeline()

    cards       = []
    total       = len(records)
    t_start     = time.time()
    errors      = []

    print(f"\n{'='*70}")
    print(f"Gate 3 — Full library body check: {total} clips")
    print(f"{'='*70}\n")

    for i, rec in enumerate(records):
        rel = rec.get("rel", "")
        video_path = path_by_stem.get(rel) or path_by_stem.get(Path(rel).name)
        if video_path is None:
            # Try dtl-1 subdir
            candidate = VIDEO_DIR / "dtl-1" / rel
            if candidate.exists():
                video_path = candidate
        if video_path is None:
            card = {
                "video_id": f"clip_{i:03d}", "parent_video": None, "side": None,
                "source_file": rel, "is_golf_swing": None, "persons": None,
                "layout": "unknown", "camera_view": "uncertain",
                "swing_type": "unknown", "handedness": "unknown",
                "camera_profile": {}, "confidence": {},
                "needs_human": ["all"], "notes": f"ERROR: video file not found: {rel}",
            }
            errors.append(rel)
            cards.append(card)
            print(f"  [{i+1:3d}/{total}] MISSING: {rel[:60]}")
            continue

        vid_id   = f"clip_{i:03d}"
        is_split = rec.get("split", "single") == "split-screen"

        try:
            if is_split:
                # Split first, profile each half
                left_crop, right_crop = get_split_x(video_path, is_split=True)
                card_l = profile_clip(f"{vid_id}_L", video_path, rec,
                                      x_crop=left_crop, parent_id=vid_id, side="left")
                card_r = profile_clip(f"{vid_id}_R", video_path, rec,
                                      x_crop=right_crop, parent_id=vid_id, side="right")
                # Also add a "parent" summary card
                parent_card = {
                    "video_id": vid_id, "parent_video": None, "side": None,
                    "source_file": rel, "is_golf_swing": True,
                    "persons": rec.get("persons",1),
                    "layout": "split_screen",
                    "camera_view":  card_l["camera_view"],   # left panel primary
                    "swing_type":   card_l["swing_type"],
                    "handedness":   card_l["handedness"],
                    "camera_profile": card_l["camera_profile"],
                    "confidence":   {"layout": 0.90, "camera_view": card_l["confidence"].get("camera_view", 0.5)},
                    "needs_human":  [],
                    "notes":        f"split-screen parent  L:[{card_l['camera_view']}]  R:[{card_r['camera_view']}]",
                    "children":     [f"{vid_id}_L", f"{vid_id}_R"],
                }
                cards.extend([parent_card, card_l, card_r])
            else:
                card = profile_clip(vid_id, video_path, rec)
                cards.append(card)

        except Exception as e:
            import traceback
            err_str = str(e)[:120]
            card = {
                "video_id": vid_id, "parent_video": None, "side": None,
                "source_file": rel, "is_golf_swing": None, "persons": None,
                "layout": "unknown", "camera_view": "uncertain",
                "swing_type": "unknown", "handedness": "unknown",
                "camera_profile": {}, "confidence": {},
                "needs_human": ["all"], "notes": f"ERROR: {err_str}",
            }
            errors.append(f"{rel}: {err_str}")
            cards.append(card)
            print(f"  [{i+1:3d}/{total}] ERROR on {rel[:40]}: {err_str[:60]}")
            traceback.print_exc()
            continue

        # Progress every 10
        if (i+1) % 10 == 0 or (i+1) == total:
            elapsed = time.time() - t_start
            rate    = (i+1) / elapsed if elapsed > 0 else 0
            eta     = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:3d}/{total}] {rel[:35]:<35s}  "
                  f"elapsed={elapsed/60:.1f}min  rate={rate:.1f}clip/s  "
                  f"ETA={eta/60:.1f}min", flush=True)
        else:
            view_s = cards[-1]["camera_view"][:3] if cards else "?"
            st_s   = cards[-1]["swing_type"][:4] if cards else "?"
            print(f"  [{i+1:3d}/{total}] {rel[:45]:<45s}  {view_s} {st_s}", flush=True)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out_json = OUT_DIR / "video_profile_full.json"
    out_json.write_text(json.dumps(cards, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out_json}  ({len(cards)} cards)")

    # ── Build stats ───────────────────────────────────────────────────────────
    _write_summary(cards, records, errors, out_json)
    elapsed_total = time.time() - t_start
    print(f"\nTotal time: {elapsed_total/60:.1f} min")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e}")


def _write_summary(cards, records, errors, out_json):
    from collections import Counter

    # Only count leaf cards (non-parent, i.e. no "children" key or children=[])
    leaf_cards = [c for c in cards if not c.get("children")]

    # Distributions
    cam_dist   = Counter(c["camera_view"] for c in leaf_cards)
    lay_dist   = Counter(c["layout"] for c in leaf_cards)
    swing_dist = Counter(c["swing_type"] for c in leaf_cards)
    hand_dist  = Counter(c["handedness"] for c in leaf_cards)

    # Key stats for 破局点3
    face_full   = [c for c in leaf_cards
                   if c["camera_view"]=="face_on" and c["swing_type"]=="full_swing"]
    single_face_full = [c for c in face_full if c["layout"]=="single" and c["side"] is None]
    split_face_full  = [c for c in face_full if c["side"] in ("left","right")]

    # Paired analysis: find split-screen clips where BOTH halves are face_on + full_swing
    from itertools import groupby
    split_parents = {}
    for c in leaf_cards:
        if c.get("parent_video"):
            pid = c["parent_video"]
            if pid not in split_parents: split_parents[pid] = []
            split_parents[pid].append(c)

    paired_face_full = 0
    paired_ids = []
    for pid, children in split_parents.items():
        if len(children) == 2:
            l = next((c for c in children if c["side"]=="left"),  None)
            r = next((c for c in children if c["side"]=="right"), None)
            if l and r:
                if (l["camera_view"]=="face_on" and l["swing_type"]=="full_swing" and
                    r["camera_view"]=="face_on" and r["swing_type"]=="full_swing"):
                    paired_face_full += 1
                    paired_ids.append(pid)

    # Write markdown
    md_lines = [
        "# Video Profile Full Library Report",
        f"\nGenerated from {len(records)} source clips → {len(leaf_cards)} profiled panels",
        f"(split-screen clips counted as 2 panels each)",
        "",
        "## Camera View Distribution",
        f"| view | count |", f"|---|---|",
    ]
    for k, v in sorted(cam_dist.items()):
        md_lines.append(f"| {k} | {v} |")

    md_lines += [
        "", "## Layout Distribution",
        f"| layout | count |", f"|---|---|",
    ]
    for k, v in sorted(lay_dist.items()):
        md_lines.append(f"| {k} | {v} |")

    md_lines += [
        "", "## Swing Type Distribution",
        f"| swing_type | count |", f"|---|---|",
    ]
    for k, v in sorted(swing_dist.items()):
        md_lines.append(f"| {k} | {v} |")

    md_lines += [
        "", "## Handedness Distribution",
        f"| handedness | count |", f"|---|---|",
    ]
    for k, v in sorted(hand_dist.items()):
        md_lines.append(f"| {k} | {v} |")

    md_lines += [
        "",
        "## 破局点3 关键统计 — face_on + full_swing 可分析素材",
        f"",
        f"- **单镜头 face_on + full_swing**: {len(single_face_full)} 段",
        f"- **分屏拆出 face_on + full_swing 半屏**: {len(split_face_full)} 个",
        f"- **同视频左右均为 face_on + full_swing (可配对)**: {paired_face_full} 对",
        f"  配对 video_id: {', '.join(paired_ids[:20]) if paired_ids else '(none)'}",
        "",
        "## Per-Clip Detail Table",
        "| idx | source_file | camera_view | layout | swing_type | handedness | needs_human |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in leaf_cards:
        nh = ",".join(c.get("needs_human",[]))
        src = c.get("source_file","")[:40]
        side_s = f"[{c['side']}] " if c.get("side") else ""
        md_lines.append(
            f"| {c['video_id']} | {side_s}{src} "
            f"| {c['camera_view']} | {c['layout']} "
            f"| {c['swing_type']} | {c['handedness']} | {nh} |"
        )

    if errors:
        md_lines += ["", f"## Errors ({len(errors)})", ""]
        for e in errors:
            md_lines.append(f"- {e}")

    out_md = out_json.parent / "video_profile_full.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved: {out_md}")

    # Print key stats to stdout
    print(f"\n{'='*60}")
    print(f"SUMMARY ({len(leaf_cards)} leaf panels from {len(records)} clips)")
    print(f"Camera view: {dict(cam_dist)}")
    print(f"Layout:      {dict(lay_dist)}")
    print(f"Swing type:  {dict(swing_dist)}")
    print(f"Handedness:  {dict(hand_dist)}")
    print(f"\n[破局点3]")
    print(f"  Single face_on+full_swing:        {len(single_face_full)}")
    print(f"  Split half face_on+full_swing:    {len(split_face_full)}")
    print(f"  Paired face_on+full_swing:        {paired_face_full} pairs")


if __name__ == "__main__":
    main()
