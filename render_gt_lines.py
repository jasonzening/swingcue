#!/usr/bin/env python3
"""
render_gt_lines.py
==================
Render FAULT_VISUAL_STANDARDS v0.2 reference lines onto video keyframes
for human GT annotation.

Rules:
- Lines are anchored at the ADDRESS frame (P1) and held fixed throughout
  except the face-on forearm line (per-frame tracking).
- NO diagnostic labels or fault conclusions on any frame.
- Output to Desktop/rtmpose_results/preview/gt_lines/<videoID>/

Line colors (BGR):
  Tush Line (yellow)        = (0, 220, 255)
  Spine axis (cyan)         = (255, 220, 0)
  Head vertical (magenta)   = (200, 0, 200)
  Head horizontal (orange)  = (0, 140, 255)
  Forearm chain (green)     = (0, 200, 60)

Run:
  python render_gt_lines.py
"""

import json, math, os, sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine.a_measurement.pose_pipeline import PosePipeline
from engine.b_phase.swing_phase import SwingPhaseEngine, PHASE_NAMES

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJ    = Path("/home/jason/projects/swingcue-postest")
INPUT   = PROJ / "input"
KP_DIR  = PROJ / "engine/kp_cache"
DESK    = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/gt_lines")
DESK.mkdir(parents=True, exist_ok=True)

VENV_PY = str(PROJ / ".venv/bin/python3")

# ── Colors (BGR) ──────────────────────────────────────────────────────────────
C_TUSH     = (0,   220, 255)   # yellow
C_SPINE    = (255, 220,   0)   # cyan
C_HEAD_V   = (200,   0, 200)   # magenta
C_HEAD_H   = (0,   140, 255)   # orange
C_FOREARM  = (0,   200,  60)   # green
C_LABEL    = (255, 255, 255)   # white text
C_LABEL_BG = (0,   0,   0)     # black bg
LINE_W = 3
FONT   = cv2.FONT_HERSHEY_DUPLEX

# ── Video config ──────────────────────────────────────────────────────────────
VIDEOS = {
    "Videos2026-06-09_201015_827": {"angle": "face-on",        "gt_impact": 59},
    "Videos2026-06-09_201039_231": {"angle": "face-on",        "gt_impact": 208},
    "Videos2026-06-09_201047_915": {"angle": "face-on",        "gt_impact": 282},
    "Videos2026-06-09_201054_561": {"angle": "down-the-line",  "gt_impact": 150},
    "Videos2026-06-09_201058_697": {"angle": "down-the-line",  "gt_impact": 186},
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def kp_pt(kps: dict, name: str, score_thr: float = 0.3) -> Optional[tuple]:
    """Return (x, y) int tuple or None if below threshold."""
    if name not in kps:
        return None
    k = kps[name]
    if k["score"] < score_thr:
        return None
    return (int(round(k["x"])), int(round(k["y"])))


def mid(a: tuple, b: tuple) -> tuple:
    return (int((a[0]+b[0])//2), int((a[1]+b[1])//2))


def angle_3pt(a, b, c) -> float:
    """Angle at vertex b, in degrees."""
    v1 = (a[0]-b[0], a[1]-b[1])
    v2 = (c[0]-b[0], c[1]-b[1])
    l1 = math.hypot(*v1); l2 = math.hypot(*v2)
    if l1 < 1 or l2 < 1:
        return 0.0
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    cos = max(-1.0, min(1.0, dot / (l1*l2)))
    return math.degrees(math.acos(cos))


def label_frame(img: np.ndarray, vid_id: str, frame_idx: int,
                phase: str, extra: str = "") -> None:
    """Top-left corner: VideoID / fr_idx / phase."""
    text = f"{vid_id[-6:]} fr{frame_idx:03d} {phase}"
    if extra:
        text += f" {extra}"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
    cv2.rectangle(img, (0, 0), (tw+12, th+12), C_LABEL_BG, -1)
    cv2.putText(img, text, (6, th+4), FONT, 0.55, C_LABEL, 1, cv2.LINE_AA)


def draw_vline(img: np.ndarray, x: int, color: tuple,
               label: str = "", proxy: bool = False) -> None:
    h = img.shape[0]
    cv2.line(img, (x, 0), (x, h), color, LINE_W, cv2.LINE_AA)
    if label:
        tag = label + (" PROXY" if proxy else "")
        cv2.putText(img, tag, (x+4, 40), FONT, 0.45, color, 1, cv2.LINE_AA)


def draw_hline(img: np.ndarray, y: int, color: tuple, label: str = "") -> None:
    w = img.shape[1]
    cv2.line(img, (0, y), (w, y), color, LINE_W, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (8, y-6), FONT, 0.45, color, 1, cv2.LINE_AA)


def draw_spine_line(img: np.ndarray, hip_mid: tuple, sh_mid: tuple,
                    extend: float = 0.20) -> None:
    """Draw hip→shoulder extended by extend fraction on both ends."""
    dx = sh_mid[0] - hip_mid[0]; dy = sh_mid[1] - hip_mid[1]
    p1 = (int(hip_mid[0] - dx*extend), int(hip_mid[1] - dy*extend))
    p2 = (int(sh_mid[0]  + dx*extend), int(sh_mid[1]  + dy*extend))
    cv2.line(img, p1, p2, C_SPINE, LINE_W, cv2.LINE_AA)
    # Dots at hip and shoulder
    cv2.circle(img, hip_mid, 5, C_SPINE, -1, cv2.LINE_AA)
    cv2.circle(img, sh_mid,  5, C_SPINE, -1, cv2.LINE_AA)


def draw_forearm_chain(img: np.ndarray, shoulder: tuple, elbow: tuple,
                       wrist: tuple) -> None:
    """Draw shoulder→elbow→wrist with elbow angle annotation."""
    cv2.line(img, shoulder, elbow,  C_FOREARM, LINE_W, cv2.LINE_AA)
    cv2.line(img, elbow,    wrist,  C_FOREARM, LINE_W, cv2.LINE_AA)
    for pt in (shoulder, elbow, wrist):
        cv2.circle(img, pt, 5, C_FOREARM, -1, cv2.LINE_AA)
    ang = angle_3pt(shoulder, elbow, wrist)
    cv2.putText(img, f"{ang:.0f}deg", (elbow[0]+8, elbow[1]-8),
                FONT, 0.50, C_FOREARM, 1, cv2.LINE_AA)


def get_frame(cap_path: str, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(cap_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, f = cap.read(); cap.release()
    if ret:
        return f
    return np.zeros((1280, 720, 3), np.uint8)


def save_jpg(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])


def load_kp_and_anchors(vid_stem: str, angle: str):
    """Load cached keypoints and run B-layer to get anchors."""
    kp_path = KP_DIR / f"{vid_stem}.json"
    with open(kp_path) as f:
        kp_json = json.load(f)
    pipe = PosePipeline(device="cpu")
    meas, fps = pipe.run_from_json(kp_json)
    eng = SwingPhaseEngine()
    annotations, anchors = eng.run(meas, fps, angle=angle)
    phase_map = {a.frame_idx: a.phase for a in annotations}
    return kp_json, meas, fps, anchors, phase_map

# ── Address-frame anchor extraction ──────────────────────────────────────────

def get_dtl_anchors(kp_json: dict, addr_fr: int) -> dict:
    """
    For DTL: Tush Line x + spine line.
    Tush Line x: from the keypoint data, take hip_mid_x as PROXY
    (SAM2 not available — label accordingly).
    'Ball side' = wrist_mid_x > hip_mid_x ? right : left.
    Tush line = hip posterior edge. With only skeleton data, we use
    hip_mid_x as proxy and mark PROXY on the image.
    """
    frame = kp_json["frames"][addr_fr]
    if not frame["persons"]:
        return {}
    kps = frame["persons"][0]["keypoints"]

    lh = kp_pt(kps, "left_hip");  rh = kp_pt(kps, "right_hip")
    ls = kp_pt(kps, "left_shoulder"); rs = kp_pt(kps, "right_shoulder")
    lw = kp_pt(kps, "left_wrist");    rw = kp_pt(kps, "right_wrist")

    if not (lh and rh):
        return {}

    hip_mid = mid(lh, rh)
    sh_mid  = mid(ls, rs) if (ls and rs) else None

    # Wrist mid for ball-side determination
    if lw and rw:
        wrist_x = (lw[0] + rw[0]) // 2
    elif lw:
        wrist_x = lw[0]
    elif rw:
        wrist_x = rw[0]
    else:
        wrist_x = hip_mid[0] + 1  # fallback: arbitrary

    # Ball side = wrist side. In DTL, the golfer faces one direction.
    # "Posterior edge" (tush line) = opposite side from ball.
    # Since we only have skeleton: proxy = hip_mid_x.
    # The tush line x is approximately hip_mid_x (the posterior hip center).
    tush_x = hip_mid[0]

    return {
        "tush_x":  tush_x,
        "tush_proxy": True,
        "hip_mid": hip_mid,
        "sh_mid":  sh_mid,
    }


def get_faceon_anchors(kp_json: dict, addr_fr: int) -> dict:
    """
    For face-on: head vertical line (x=ear/eye midpoint),
    head horizontal line (y=highest head point at address).
    """
    frame = kp_json["frames"][addr_fr]
    if not frame["persons"]:
        return {}
    kps = frame["persons"][0]["keypoints"]

    le = kp_pt(kps, "left_ear");   re = kp_pt(kps, "right_ear")
    leye = kp_pt(kps, "left_eye"); reye = kp_pt(kps, "right_eye")
    nose = kp_pt(kps, "nose")

    # Head center x: average of available head landmarks
    head_xs = [p[0] for p in [le, re, leye, reye, nose] if p]
    head_ys = [p[1] for p in [le, re, leye, reye, nose] if p]

    if not head_xs:
        return {}

    head_v_x = int(sum(head_xs) / len(head_xs))  # vertical line x
    head_h_y = min(head_ys)                        # horizontal line y = highest (min y)

    return {
        "head_v_x": head_v_x,
        "head_h_y": head_h_y,
    }


def get_forearm_kps(kp_json: dict, fr: int, is_right_handed: bool = True):
    """
    Return (lead_shoulder, lead_elbow, lead_wrist) for the lead arm.
    For right-handed golfer: lead = LEFT arm.
    Returns None if any key point is missing.
    """
    frame = kp_json["frames"][fr]
    if not frame["persons"]:
        return None
    kps = frame["persons"][0]["keypoints"]
    if is_right_handed:
        sh = kp_pt(kps, "left_shoulder", score_thr=0.3)
        el = kp_pt(kps, "left_elbow",    score_thr=0.3)
        wr = kp_pt(kps, "left_wrist",    score_thr=0.3)
    else:
        sh = kp_pt(kps, "right_shoulder", score_thr=0.3)
        el = kp_pt(kps, "right_elbow",    score_thr=0.3)
        wr = kp_pt(kps, "right_wrist",    score_thr=0.3)
    if sh and el and wr:
        return sh, el, wr
    return None

# ── Phase window helpers ──────────────────────────────────────────────────────

def phase_label_at(phase_map: dict, fr: int) -> str:
    return phase_map.get(fr, "?")


def frames_in_phases(phase_map: dict, target_phases, stride: int = 2,
                     forced: list = None):
    """
    Return sorted list of frame indices:
    - every `stride` frames where phase is in target_phases
    - plus any `forced` frames (anchor keyframes)
    """
    pool = [fr for fr, p in phase_map.items() if p in target_phases]
    selected = set(pool[::stride])
    if forced:
        for f in forced:
            if f in phase_map:
                selected.add(f)
    return sorted(selected)


# ── Per-video renderers ───────────────────────────────────────────────────────

def render_dtl(vid_stem: str, cfg: dict, out_base: Path):
    print(f"\n  === DTL: {vid_stem[-6:]} ===")
    vid_path = str(INPUT / f"{vid_stem}.mp4")
    vid_id   = vid_stem[-6:]
    out_dir  = out_base / vid_id
    out_dir.mkdir(parents=True, exist_ok=True)

    kp_json, meas, fps, anchors, phase_map = load_kp_and_anchors(
        vid_stem, cfg["angle"])

    addr_fr   = anchors.address
    impact_fr = anchors.impact

    dtl_anc = get_dtl_anchors(kp_json, addr_fr)
    if not dtl_anc:
        print(f"    WARNING: no DTL anchors for fr{addr_fr}")
        return 0

    tush_x  = dtl_anc["tush_x"]
    proxy   = dtl_anc["tush_proxy"]
    hip_mid = dtl_anc["hip_mid"]
    sh_mid  = dtl_anc["sh_mid"]

    # ── Window: P5 → impact + 5 ──────────────────────────────────────────────
    window_phases = {"transition", "downswing", "impact", "follow_through"}
    # P5 = transition start; we want from transition to impact+5
    forced_frames = [addr_fr, anchors.top, impact_fr]
    candidates = frames_in_phases(phase_map, window_phases, stride=2,
                                   forced=forced_frames)
    # Trim to impact+5
    candidates = [f for f in candidates if f <= impact_fr + 5]
    # Also include address frame
    candidates = sorted(set(candidates) | {addr_fr})

    count = 0
    for fr in candidates:
        raw = get_frame(vid_path, fr)
        img = raw.copy()
        phase = phase_label_at(phase_map, fr)

        # Tush Line
        draw_vline(img, tush_x, C_TUSH, "TUSH", proxy=proxy)

        # Spine axis (fixed from address)
        if sh_mid:
            draw_spine_line(img, hip_mid, sh_mid)

        # Label
        label_frame(img, vid_id, fr, phase)
        fname = f"fr{fr:03d}_{phase}.jpg"
        save_jpg(img, out_dir / fname)
        count += 1

    # Address overview (all lines on address frame)
    addr_img = get_frame(vid_path, addr_fr).copy()
    draw_vline(addr_img, tush_x, C_TUSH, "TUSH", proxy=proxy)
    if sh_mid:
        draw_spine_line(addr_img, hip_mid, sh_mid)
    label_frame(addr_img, vid_id, addr_fr, "address", "ADDRESS_OVERVIEW")
    save_jpg(addr_img, out_dir / f"fr{addr_fr:03d}_ADDRESS_OVERVIEW.jpg")
    count += 1

    print(f"    {count} frames -> {out_dir}")
    return count


def render_faceon(vid_stem: str, cfg: dict, out_base: Path):
    print(f"\n  === Face-on: {vid_stem[-6:]} ===")
    vid_path = str(INPUT / f"{vid_stem}.mp4")
    vid_id   = vid_stem[-6:]

    kp_json, meas, fps, anchors, phase_map = load_kp_and_anchors(
        vid_stem, cfg["angle"])

    addr_fr   = anchors.address
    impact_fr = anchors.impact

    fo_anc = get_faceon_anchors(kp_json, addr_fr)
    if not fo_anc:
        print(f"    WARNING: no face-on anchors for fr{addr_fr}")
        return 0

    head_v_x = fo_anc["head_v_x"]
    head_h_y = fo_anc["head_h_y"]

    total_count = 0

    # ── Sub-folder 1: backswing/ (P3-P4, head vertical) ──────────────────────
    bs_dir = out_base / vid_id / "backswing"
    bs_phases = {"takeaway", "backswing", "top"}
    bs_frames = frames_in_phases(phase_map, bs_phases, stride=2,
                                  forced=[addr_fr, anchors.top])
    for fr in bs_frames:
        raw = get_frame(vid_path, fr)
        img = raw.copy()
        phase = phase_label_at(phase_map, fr)
        draw_vline(img, head_v_x, C_HEAD_V, "HEAD-V")
        label_frame(img, vid_id, fr, phase)
        save_jpg(img, bs_dir / f"fr{fr:03d}_{phase}.jpg")
        total_count += 1
    print(f"    backswing/: {len(bs_frames)} frames")

    # ── Sub-folder 2: downswing/ (P5→impact, head horizontal) ────────────────
    ds_dir = out_base / vid_id / "downswing"
    ds_phases = {"transition", "downswing", "impact"}
    ds_frames = frames_in_phases(phase_map, ds_phases, stride=2,
                                  forced=[addr_fr, impact_fr])
    for fr in ds_frames:
        raw = get_frame(vid_path, fr)
        img = raw.copy()
        phase = phase_label_at(phase_map, fr)
        draw_hline(img, head_h_y, C_HEAD_H, "HEAD-H")
        label_frame(img, vid_id, fr, phase)
        save_jpg(img, ds_dir / f"fr{fr:03d}_{phase}.jpg")
        total_count += 1
    print(f"    downswing/: {len(ds_frames)} frames")

    # ── Sub-folder 3: followthrough/ (impact→+8, forearm chain) ──────────────
    ft_dir = out_base / vid_id / "followthrough"
    ft_end = impact_fr + 8
    ft_frames = sorted(set(
        [f for f in range(impact_fr, min(ft_end+1, len(phase_map)), 2)]
        + [impact_fr]
    ))
    for fr in ft_frames:
        if fr >= len(kp_json["frames"]):
            continue
        raw = get_frame(vid_path, fr)
        img = raw.copy()
        phase = phase_label_at(phase_map, fr)
        # Per-frame forearm chain (tracking)
        chain = get_forearm_kps(kp_json, fr)
        if chain:
            draw_forearm_chain(img, *chain)
        label_frame(img, vid_id, fr, phase)
        save_jpg(img, ft_dir / f"fr{fr:03d}_{phase}.jpg")
        total_count += 1
    print(f"    followthrough/: {len(ft_frames)} frames")

    # ── Address overview ──────────────────────────────────────────────────────
    ov_dir = out_base / vid_id
    ov_img = get_frame(vid_path, addr_fr).copy()
    draw_vline(ov_img, head_v_x, C_HEAD_V, "HEAD-V")
    draw_hline(ov_img, head_h_y, C_HEAD_H, "HEAD-H")
    # forearm at address
    chain = get_forearm_kps(kp_json, addr_fr)
    if chain:
        draw_forearm_chain(ov_img, *chain)
    label_frame(ov_img, vid_id, addr_fr, "address", "ADDRESS_OVERVIEW")
    save_jpg(ov_img, ov_dir / f"fr{addr_fr:03d}_ADDRESS_OVERVIEW.jpg")
    total_count += 1

    print(f"    TOTAL: {total_count} frames -> {out_base / vid_id}")
    return total_count


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import datetime
    print(f"render_gt_lines.py  started {datetime.datetime.now().isoformat()}")

    grand_total = 0
    summary = {}

    for vid_stem, cfg in VIDEOS.items():
        angle = cfg["angle"]
        if angle == "down-the-line":
            n = render_dtl(vid_stem, cfg, DESK)
        else:
            n = render_faceon(vid_stem, cfg, DESK)
        summary[vid_stem[-6:]] = n
        grand_total += n

    print("\n" + "="*55)
    print("SUMMARY")
    print("="*55)
    for vid_id, cnt in summary.items():
        print(f"  {vid_id}: {cnt} frames")
    print(f"  TOTAL: {grand_total}")
    print(f"  Output: {DESK}")
    print("\nNo diagnostic labels were applied. Lines only.")
    print("Awaiting human GT annotation.")

    # Update NEEDS_HUMAN.md
    needs = Path("/home/jason/projects/swingcue-postest/NEEDS_HUMAN.md")
    existing = needs.read_text() if needs.exists() else ""
    note = (
        "\n\n## GT Line Rendering — gt_lines/ (2026-06-10)\n\n"
        f"Rendered {grand_total} annotated frames to:\n"
        "  Desktop/rtmpose_results/preview/gt_lines/\n\n"
        "Per-video sub-folders:\n"
        "  DTL (201054, 201058): Tush Line (yellow) + Spine axis (cyan)\n"
        "    -> Window: P5 transition through impact+5\n"
        "  Face-on (201015, 201039, 201047):\n"
        "    backswing/   : Head vertical line (magenta) — RP check\n"
        "    downswing/   : Head horizontal line (orange) — LoP check\n"
        "    followthrough/: Lead forearm chain (green) + elbow angle — CW check\n\n"
        "**NOT RENDERED** (deferred):\n"
        "  - Shaft plane line (Over-the-Top check): requires ball/club detection pipeline\n"
        "    which is not yet built. Will be added when club detection flow is complete.\n\n"
        "Human action: inspect frames in gt_lines/ and confirm or correct anchor frames.\n"
    )
    needs.write_text(existing + note)

    # Update PROGRESS.log
    prog = Path("/home/jason/projects/swingcue-postest/PROGRESS.log")
    with open(prog, "a") as f:
        ts = datetime.datetime.now().isoformat()
        f.write(f"{ts}  GT line rendering complete: {grand_total} frames in gt_lines/\n")


if __name__ == "__main__":
    main()
