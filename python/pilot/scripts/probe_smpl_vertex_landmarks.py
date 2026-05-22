"""
probe_smpl_vertex_landmarks.py (PR-7a.5 — fresh) — measure SMPL vertex
sampling vs current PR-7a fitted-offset anchors against Jason's GT
labels on b3fea3f0 face_on.

Supersedes yesterday's stub which couldn't run because verts were
unavailable locally. With PR-7a.5 wham_runner.py patch + Modal cycle
saving smpl_params.npz, this script now has real data to compare.

Inputs (all read-only):
  python/pilot/output/wham/b3fea3f0-*/smpl_params.npz
    └─ from PR-7a.5-patched wham_runner with --save-smpl-params
  docs/PR-7a4_PROBE/smpl_landmark_indices.json
    └─ vertex indices derived from SMPL T-pose (yesterday)
  docs/PR-7a_OFFLINE_OUTPUT/b3fea3f0_face_on_corrected.json
    └─ current PR-7a corrected anchors
  docs/PR-7_GROUND_TRUTH/golf/b3fea3f0_{setup,impact,finish}_face_on.json
    └─ Jason's red dots
  python/benchmark/test_videos/b3fea3f0-*.mp4
    └─ source video for background frame

Outputs:
  docs/PR-7a5_PROBE/b3fea3f0_face_on_{setup,impact,finish}_compare.png
  docs/PR-7a5_PROBE/distance_to_gt.csv
  stdout: PROBE PASS / FAIL acceptance summary

PASS gate (per PR-7a.5 spec):
  - acromion vertex closer to GT than PR-7a anchor on >= 2 of 3 frames
  - knee/ankle/foot vertices visibly on lateral anatomy (visual, PNG)
  - no L/R confusion across frames
  - no projection blowouts (all 2D coords within image bounds)
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))
from motion_correction.engine.projection import (  # noqa: E402
    default_intrinsics, project_xyz_to_uv,
)

# ── Paths ───────────────────────────────────────────────────────────
VIDEO_ID = "b3fea3f0-e248-44d7-a923-0bb43172b5bf"
SHORT    = VIDEO_ID.split("-")[0]
VIEW     = "face_on"
PROBE_DIR = REPO_ROOT / "docs" / "PR-7a5_PROBE"
INDICES   = REPO_ROOT / "docs" / "PR-7a4_PROBE" / "smpl_landmark_indices.json"
WHAM_DIR  = REPO_ROOT / "python" / "pilot" / "output" / "wham" / VIDEO_ID
COR_JSON  = (REPO_ROOT / "docs" / "PR-7a_OFFLINE_OUTPUT"
             / f"{SHORT}_{VIEW}_corrected.json")
GT_DIR    = REPO_ROOT / "docs" / "PR-7_GROUND_TRUTH" / "golf"
VIDEO_MP4 = REPO_ROOT / "python" / "benchmark" / "test_videos" / f"{VIDEO_ID}.mp4"

# ── Landmark→GT/anchor mapping ──────────────────────────────────────
LANDMARK_MAP: dict[str, tuple[str | None, str | None]] = {
    "acromion_left":              ("left_shoulder",  "left_shoulder_visual"),
    "acromion_right":             ("right_shoulder", "right_shoulder_visual"),
    "greater_trochanter_left":    ("left_hip",       "left_hip_visual"),
    "greater_trochanter_right":   ("right_hip",      "right_hip_visual"),
    "throat_midpoint":            ("neck_center",    "neck_visual"),
    # NEW landmarks (no GT, no PR-7a anchor — visual sanity only).
    "head_crown":                 (None, None),
    "c7":                         (None, None),
    "lateral_epicondyle_left":    (None, None),
    "lateral_epicondyle_right":   (None, None),
    "lateral_malleolus_left":     (None, None),
    "lateral_malleolus_right":    (None, None),
}

# Render colors (BGR).
COLOR_PR7A_OLD = (255,   0, 255)   # magenta — old PR-7a corrected
COLOR_VERTEX   = (255, 255,   0)   # cyan — new SMPL vertex
COLOR_GT       = (  0,   0, 255)   # red — Jason GT


def dist(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def extract_video_frame(path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"can't read frame {frame_idx} from {path}")
    return frame


def render_one_frame(
    *, frame_idx: int, phase: str,
    verts_3d: np.ndarray,
    landmarks: dict,
    pr7a_anchors: dict,
    gt_labels: dict,
    intrinsics: dict,
    out_png: Path,
) -> list[dict]:
    img = extract_video_frame(VIDEO_MP4, frame_idx)
    H, W = img.shape[:2]
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    rows: list[dict] = []

    for vname, (gt_key, anchor_key) in LANDMARK_MAP.items():
        info = landmarks.get(vname)
        if info is None:
            continue
        vidx = int(info["index"])
        v3 = verts_3d[vidx].tolist()
        new_uv = project_xyz_to_uv(v3, fx, fy, cx, cy)
        in_bounds = (
            new_uv is not None
            and 0 <= new_uv[0] < W and 0 <= new_uv[1] < H
        )
        old_uv = pr7a_anchors.get(anchor_key) if anchor_key else None
        gt_xy = None
        if gt_key:
            g = gt_labels.get(gt_key)
            if g and "x" in g and "y" in g:
                gt_xy = (float(g["x"]), float(g["y"]))

        old_d = dist(old_uv, gt_xy) if (old_uv and gt_xy) else None
        new_d = dist(new_uv, gt_xy) if (new_uv and gt_xy) else None
        improvement = (
            old_d - new_d if (old_d is not None and new_d is not None) else None
        )

        # ── Draw ──
        if new_uv and in_bounds:
            x, y = int(round(new_uv[0])), int(round(new_uv[1]))
            cv2.rectangle(img, (x - 7, y - 7), (x + 7, y + 7),
                          COLOR_VERTEX, 2, cv2.LINE_AA)
            # Tiny label for new landmarks (no GT).
            if gt_key is None:
                short_label = vname.replace("lateral_", "").replace("_", "")[:6]
                cv2.putText(img, short_label, (x + 9, y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, short_label, (x + 9, y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_VERTEX, 1, cv2.LINE_AA)
        if old_uv:
            ox, oy = int(round(old_uv[0])), int(round(old_uv[1]))
            cv2.circle(img, (ox, oy), 7, COLOR_PR7A_OLD, -1, cv2.LINE_AA)
        if gt_xy:
            gx, gy = int(round(gt_xy[0])), int(round(gt_xy[1]))
            cv2.circle(img, (gx, gy), 6, COLOR_GT, -1, cv2.LINE_AA)
            cv2.circle(img, (gx, gy), 7, (0, 0, 0), 1, cv2.LINE_AA)
        if new_uv and in_bounds and gt_xy:
            cv2.line(img, (int(new_uv[0]), int(new_uv[1])),
                     (int(gt_xy[0]), int(gt_xy[1])),
                     COLOR_VERTEX, 1, cv2.LINE_AA)
        if old_uv and gt_xy:
            cv2.line(img, (int(old_uv[0]), int(old_uv[1])),
                     (int(gt_xy[0]), int(gt_xy[1])),
                     COLOR_PR7A_OLD, 1, cv2.LINE_AA)

        rows.append({
            "frame_idx":     frame_idx,
            "phase":         phase,
            "landmark":      vname,
            "vertex_index":  vidx,
            "in_bounds":     bool(in_bounds),
            "gt_key":        gt_key,
            "gt_2d_x":       gt_xy[0] if gt_xy else None,
            "gt_2d_y":       gt_xy[1] if gt_xy else None,
            "old_anchor_key": anchor_key,
            "old_anchor_x":  old_uv[0] if old_uv else None,
            "old_anchor_y":  old_uv[1] if old_uv else None,
            "new_vertex_x":  new_uv[0] if new_uv else None,
            "new_vertex_y":  new_uv[1] if new_uv else None,
            "old_dist_px":   old_d,
            "new_dist_px":   new_d,
            "improvement_px": improvement,
        })

    # Title + legend
    cv2.rectangle(img, (0, 0), (W, 42), (40, 40, 40), -1)
    cv2.putText(img, f"PR-7a.5 probe — {VIEW} / {phase} / frame {frame_idx}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)
    legend = "magenta=PR-7a anchor  cyan square=SMPL vertex  red=GT"
    cv2.putText(img, legend, (W - 480, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), img)
    return rows


def main() -> None:
    # Load .npz
    npz_path = WHAM_DIR / "smpl_params.npz"
    if not npz_path.exists():
        sys.exit(f"[probe] missing {npz_path}")
    npz = np.load(npz_path)
    verts = npz["verts"]
    print(f"[probe] loaded verts shape={verts.shape} dtype={verts.dtype} "
          f"file={npz_path.stat().st_size / 1024 / 1024:.1f} MB")

    raw_json = json.loads((WHAM_DIR / "joint_centers_3d.json").read_text())
    schema_ver = raw_json.get("_wham_runner_version", "(missing)")
    print(f"[probe] _wham_runner_version = {schema_ver}")

    frame_ids = [int(f["frame_idx"]) for f in raw_json["frames"]]
    fi_to_row = {fi: i for i, fi in enumerate(frame_ids)}

    landmarks = json.loads(INDICES.read_text())
    pr7a = json.loads(COR_JSON.read_text())
    pr7a_by_frame = {int(f["frame_idx"]): f for f in pr7a["frames"]}
    W = int(raw_json["video_width"])
    H = int(raw_json["video_height"])
    intr = default_intrinsics(W, H)

    csv_rows: list[dict] = []
    for phase in ("setup", "impact", "finish"):
        gt = json.loads((GT_DIR / f"{SHORT}_{phase}_{VIEW}.json").read_text())
        fi = int(gt["frame_idx"])
        if fi not in fi_to_row:
            print(f"[probe] frame {fi} not in WHAM output — skip")
            continue
        verts_3d = verts[fi_to_row[fi]]
        cor_frame = pr7a_by_frame.get(fi, {})
        pr7a_anchors = cor_frame.get("coaching_anchors_2d", {})
        out_png = PROBE_DIR / f"{SHORT}_{VIEW}_{phase}_compare.png"
        print(f"[probe] rendering {out_png.name} (frame {fi})")
        csv_rows.extend(render_one_frame(
            frame_idx=fi, phase=phase,
            verts_3d=verts_3d, landmarks=landmarks,
            pr7a_anchors=pr7a_anchors,
            gt_labels=gt["labels"],
            intrinsics=intr,
            out_png=out_png,
        ))

    csv_path = PROBE_DIR / "distance_to_gt.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        if csv_rows:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
    print(f"[probe] wrote {csv_path}")

    # ── Acceptance ──
    print()
    print("=" * 78)
    print("PR-7a.5 PROBE ACCEPTANCE")
    print("=" * 78)
    by_lm: dict[str, list[dict]] = {}
    for r in csv_rows:
        if r["old_dist_px"] is not None and r["new_dist_px"] is not None:
            by_lm.setdefault(r["landmark"], []).append(r)
    print(f"\n{'landmark':<28s} {'n':>3s}  {'old_mean':>9s}  {'new_mean':>9s}  "
          f"{'beats':>8s}")
    print("-" * 78)
    acromion_beats = {"acromion_left": 0, "acromion_right": 0}
    acromion_n = {"acromion_left": 0, "acromion_right": 0}
    for name, rows in sorted(by_lm.items()):
        olds = [r["old_dist_px"] for r in rows]
        news = [r["new_dist_px"] for r in rows]
        old_mean = sum(olds) / len(olds)
        new_mean = sum(news) / len(news)
        beats = sum(1 for r in rows if r["new_dist_px"] < r["old_dist_px"])
        if name in acromion_beats:
            acromion_beats[name] = beats
            acromion_n[name] = len(rows)
        print(f"  {name:<28s} {len(rows):>3d}  {old_mean:>9.2f}  {new_mean:>9.2f}  "
              f"{beats}/{len(rows)}")

    out_of_bounds = sum(1 for r in csv_rows if not r["in_bounds"])

    # L/R consistency check.
    left_x: dict[int, float] = {}
    right_x: dict[int, float] = {}
    for r in csv_rows:
        if r["landmark"] == "acromion_left" and r["new_vertex_x"] is not None:
            left_x[r["frame_idx"]] = r["new_vertex_x"]
        if r["landmark"] == "acromion_right" and r["new_vertex_x"] is not None:
            right_x[r["frame_idx"]] = r["new_vertex_x"]
    print()
    print("L/R x-coord per frame (acromion vertices):")
    for fi in sorted(set(left_x) | set(right_x)):
        lx = left_x.get(fi, None)
        rx = right_x.get(fi, None)
        ord_str = ""
        if lx is not None and rx is not None:
            ord_str = f"  L<R? {lx < rx}"
        print(f"  frame {fi:>3d}  L={lx}  R={rx}{ord_str}")

    print()
    acromion_l_pass = acromion_beats.get("acromion_left", 0) >= 2
    acromion_r_pass = acromion_beats.get("acromion_right", 0) >= 2
    bounds_pass = out_of_bounds == 0
    print(f"acromion_left:  {acromion_beats.get('acromion_left',0)}/"
          f"{acromion_n.get('acromion_left',0)} new<old "
          f"(gate >=2) → {'PASS' if acromion_l_pass else 'FAIL'}")
    print(f"acromion_right: {acromion_beats.get('acromion_right',0)}/"
          f"{acromion_n.get('acromion_right',0)} new<old "
          f"(gate >=2) → {'PASS' if acromion_r_pass else 'FAIL'}")
    print(f"projection bounds: {out_of_bounds} out-of-bounds → "
          f"{'PASS' if bounds_pass else 'FAIL'}")
    print()
    overall = acromion_l_pass and acromion_r_pass and bounds_pass
    print(f"FINAL: {'PROBE PASS (acromion + bounds)' if overall else 'PROBE FAIL'}")
    print(f"       knee/ankle/foot landmarks: REVIEW PNGs visually")
    print()
    for phase in ("setup", "impact", "finish"):
        p = PROBE_DIR / f"{SHORT}_{VIEW}_{phase}_compare.png"
        if p.exists():
            print(f"  → {p}")


if __name__ == "__main__":
    main()
