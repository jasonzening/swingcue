"""
render_overlay_compare.py — side-by-side overlay video.

Left panel:  RAW WHAM skeleton (pinhole-projected from joint_centers_3d.json).
Right panel: PR-7a CORRECTED skeleton + coaching_anchors_2d markers
             (read directly from <video>_<view>_corrected.json — already
             projected by the orchestrator).

Output: docs/PR-7a_OFFLINE_OUTPUT/<video_id>_<view>_overlay_compare.mp4

Color convention (matches python/pilot/runners/_overlay.py):
  left_*  : cyan/blue
  right_* : orange/yellow
  midline : grey
Coaching anchors render as larger filled magenta circles, distinguishable
from raw skeleton dots.

CLI:
    .venv-benchmark/Scripts/python.exe \\
        python/motion_correction/scripts/render_overlay_compare.py \\
        --video-id <uuid> --view face_on|down_the_line
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

# Mirror of _overlay.py — keep render parity.
SKELETON_EDGES: tuple[tuple[str, str], ...] = (
    ("pelvis", "spine1"),
    ("spine1", "neck"),
    ("neck", "head"),
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("pelvis", "left_hip"),
    ("pelvis", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)

COLOR_LEFT    = (255, 200,  60)   # BGR cyan/blue
COLOR_RIGHT   = ( 60, 200, 255)   # BGR orange
COLOR_MID     = (180, 180, 180)   # BGR grey
COLOR_EDGE    = (153, 153, 153)
COLOR_ANCHOR  = (255,   0, 255)   # BGR magenta — coaching anchors
COLOR_TITLE_BG_RAW       = ( 40,  40, 140)   # deep red — raw side
COLOR_TITLE_BG_CORRECTED = ( 40, 140,  40)   # deep green — corrected side
KP_RADIUS = 6
EDGE_THICKNESS = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Per-side visual anchors (left/right shoulders, left/right hips, neck):
# larger filled magenta circle, distinct from disc centers.
ANCHOR_PER_SIDE_RADIUS = 12
# Midpoint disc centers (shoulder_disc_center, hip_ring_center):
# smaller hollow magenta ring with 2 px stroke.
ANCHOR_DISC_RADIUS     = 8
ANCHOR_DISC_STROKE     = 2

PER_SIDE_ANCHOR_NAMES: frozenset[str] = frozenset({
    "left_shoulder_visual",
    "right_shoulder_visual",
    "left_hip_visual",
    "right_hip_visual",
    "neck_visual",
})
DISC_ANCHOR_NAMES: frozenset[str] = frozenset({
    "shoulder_disc_center",
    "hip_ring_center",
})


def _dot_color(name: str) -> tuple[int, int, int]:
    if "left" in name:
        return COLOR_LEFT
    if "right" in name:
        return COLOR_RIGHT
    return COLOR_MID


def _project_raw(joints_3d: dict, fx: float, fy: float, cx: float, cy: float) -> dict:
    """Pinhole project raw WHAM 3D joints to pixel coords (mirrors _overlay.py)."""
    out = {}
    for name, xyz in joints_3d.items():
        if xyz is None or len(xyz) != 3:
            continue
        X, Y, Z = xyz
        if Z is None or Z <= 0:
            continue
        u = fx * X / Z + cx
        v = fy * Y / Z + cy
        out[name] = (int(round(u)), int(round(v)))
    return out


def _to_int_xy(uv: list | None) -> tuple[int, int] | None:
    if uv is None or len(uv) != 2:
        return None
    return (int(round(uv[0])), int(round(uv[1])))


def _draw_skeleton(frame: np.ndarray, pts: dict[str, tuple[int, int]]) -> None:
    for a, b in SKELETON_EDGES:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], COLOR_EDGE, EDGE_THICKNESS, cv2.LINE_AA)
    for name, (u, v) in pts.items():
        color = _dot_color(name)
        cv2.circle(frame, (u, v), KP_RADIUS, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (u, v), KP_RADIUS + 1, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_coaching_anchors(
    frame: np.ndarray, anchors: dict[str, list | None],
) -> None:
    """
    Distinguish per-side visuals from midpoint disc centers:
      - Per-side anchors (left/right shoulders, left/right hips, neck):
        large FILLED magenta circle (r=12), black 1-px outline for
        contrast against the video frame.
      - Disc centers (shoulder_disc_center, hip_ring_center):
        smaller HOLLOW magenta ring (r=8, 2-px stroke).
    """
    for name, uv in anchors.items():
        pt = _to_int_xy(uv)
        if pt is None:
            continue
        if name in PER_SIDE_ANCHOR_NAMES:
            cv2.circle(frame, pt, ANCHOR_PER_SIDE_RADIUS, COLOR_ANCHOR,
                       -1, cv2.LINE_AA)
            cv2.circle(frame, pt, ANCHOR_PER_SIDE_RADIUS + 1, (0, 0, 0),
                       1, cv2.LINE_AA)
        elif name in DISC_ANCHOR_NAMES:
            cv2.circle(frame, pt, ANCHOR_DISC_RADIUS, COLOR_ANCHOR,
                       ANCHOR_DISC_STROKE, cv2.LINE_AA)
        # Unknown anchor names are silently skipped — keeps the renderer
        # tolerant of plugin namespace additions.


def _title_bar(
    frame: np.ndarray, text: str, bg_color: tuple[int, int, int],
) -> None:
    H, W = frame.shape[:2]
    bar_h = 38
    cv2.rectangle(frame, (0, 0), (W, bar_h), bg_color, -1)
    cv2.putText(
        frame, text, (12, 27), FONT, 0.75, (255, 255, 255), 2, cv2.LINE_AA,
    )


def _phase_label(frame: np.ndarray, text: str) -> None:
    """Bottom-left phase label."""
    H, W = frame.shape[:2]
    cv2.rectangle(frame, (8, H - 36), (8 + 11 * len(text) + 8, H - 8),
                  (0, 0, 0), -1)
    cv2.putText(frame, text, (12, H - 16), FONT, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)


def render_compare(
    raw_json_path: Path,
    corrected_json_path: Path,
    video_path: Path,
    out_path: Path,
) -> None:
    raw = json.loads(raw_json_path.read_text())
    cor = json.loads(corrected_json_path.read_text())

    raw_frames = raw["frames"]
    cor_frames = cor["frames"]
    if not raw_frames:
        sys.exit(f"no frames in {raw_json_path}")
    if not cor_frames:
        sys.exit(f"no frames in {corrected_json_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"could not open {video_path}")
    fps_native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_native = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # WHAM default intrinsics (matches _overlay.py + projection.default_intrinsics).
    fx = fy = float(max(W, H))
    cx, cy = W / 2.0, H / 2.0
    print(f"[compare] {W}x{H} @ {fps_native:.2f}fps  raw={len(raw_frames)}  "
          f"cor={len(cor_frames)}  native={n_native}")

    raw_by_idx = {int(f["frame_idx"]): f for f in raw_frames}
    cor_by_idx = {int(f["frame_idx"]): f for f in cor_frames}

    out_W = W * 2
    out_H = H
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_native, (out_W, out_H))
    if not writer.isOpened():
        sys.exit(f"could not open writer for {out_path}")

    last_raw_pts = {}
    last_cor_pts = {}
    last_anchors = {}
    last_phase = "?"
    written = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Left panel: raw WHAM.
        rf = raw_by_idx.get(written)
        if rf is not None:
            last_raw_pts = _project_raw(rf["joint_centers_3d"], fx, fy, cx, cy)
        left = frame.copy()
        _draw_skeleton(left, last_raw_pts)
        _title_bar(left, "RAW WHAM (uncorrected)", COLOR_TITLE_BG_RAW)
        _phase_label(left, f"frame {written}/{n_native - 1}  ({len(last_raw_pts)} kp)")

        # Right panel: corrected — keypoints already projected by orchestrator.
        cf = cor_by_idx.get(written)
        if cf is not None:
            kp2d = cf.get("keypoints_2d_projected", {})
            last_cor_pts = {
                name: _to_int_xy(uv) for name, uv in kp2d.items()
                if _to_int_xy(uv) is not None
            }
            last_anchors = cf.get("coaching_anchors_2d", {})
            last_phase = cf.get("phase", "?")
        right = frame.copy()
        _draw_skeleton(right, last_cor_pts)
        _draw_coaching_anchors(right, last_anchors)
        _title_bar(right, "PR-7a CORRECTED  (magenta: filled=per-side / hollow=disc-center)",
                   COLOR_TITLE_BG_CORRECTED)
        _phase_label(right, f"phase: {last_phase}   frame {written}/{n_native - 1}")

        out_frame = np.hstack([left, right])
        writer.write(out_frame)
        written += 1

    cap.release()
    writer.release()
    sz_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[compare] wrote {written} frames → {out_path}  ({sz_mb:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--view", required=True, choices=("face_on", "down_the_line"))
    ap.add_argument("--video-path", type=Path, default=None)
    ap.add_argument("--raw-path", type=Path, default=None)
    ap.add_argument("--corrected-path", type=Path, default=None)
    ap.add_argument("--out-path", type=Path, default=None)
    args = ap.parse_args()

    raw_path = args.raw_path or (
        REPO_ROOT / "python" / "pilot" / "output" / "wham" / args.video_id
        / "joint_centers_3d.json"
    )
    # corrected JSON is keyed by short prefix (b3fea3f0_*) not full UUID.
    short = args.video_id.split("-")[0]
    cor_path = args.corrected_path or (
        REPO_ROOT / "docs" / "PR-7a_OFFLINE_OUTPUT"
        / f"{short}_{args.view}_corrected.json"
    )
    video_path = args.video_path or (
        REPO_ROOT / "python" / "benchmark" / "test_videos"
        / f"{args.video_id}.mp4"
    )
    out_path = args.out_path or (
        REPO_ROOT / "docs" / "PR-7a_OFFLINE_OUTPUT"
        / f"{short}_{args.view}_overlay_compare.mp4"
    )

    for p, name in ((raw_path, "raw JSON"),
                    (cor_path, "corrected JSON"),
                    (video_path, "source video")):
        if not p.exists():
            sys.exit(f"[compare] missing {name}: {p}")

    render_compare(raw_path, cor_path, video_path, out_path)


if __name__ == "__main__":
    sys.exit(main())
