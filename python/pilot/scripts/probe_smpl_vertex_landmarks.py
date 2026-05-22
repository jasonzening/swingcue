"""
probe_smpl_vertex_landmarks.py — local-only rendering + measurement
script for the PR-7a.4 SMPL vertex sampling probe.

Inputs (all read-only):
  - docs/PR-7a4_PROBE/smpl_landmark_indices.json
        (output of probe_derive_smpl_landmarks.py)
  - python/pilot/output/wham/<video_id>/verts_at_frames.json
        (output of probe_extract_verts_modal.py)
  - docs/PR-7a_OFFLINE_OUTPUT/<short_id>_<view>_corrected.json
        (current PR-7a corrected timeline)
  - docs/PR-7_GROUND_TRUTH/golf/<short_id>_<phase>_<view>.json
        (Jason's GT labels)
  - python/benchmark/test_videos/<video_id>.mp4
        (source video for frame extraction)

Outputs (probe-only — never touches motion_correction):
  - docs/PR-7a4_PROBE/<short_id>_<view>_<phase>_compare.png
        (one per frame: source + OLD anchors + NEW vertex samples + GT dots)
  - docs/PR-7a4_PROBE/distance_to_gt.csv

PASS/FAIL acceptance reported to stdout at end.
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
from motion_correction.engine.projection import (
    default_intrinsics, project_xyz_to_uv,
)

PROBE_DIR = REPO_ROOT / "docs" / "PR-7a4_PROBE"
GT_DIR    = REPO_ROOT / "docs" / "PR-7_GROUND_TRUTH" / "golf"
COR_DIR   = REPO_ROOT / "docs" / "PR-7a_OFFLINE_OUTPUT"
WHAM_DIR  = REPO_ROOT / "python" / "pilot" / "output" / "wham"
VIDEOS    = REPO_ROOT / "python" / "benchmark" / "test_videos"


# Mapping from GT label key → SMPL landmark name + corresponding PR-7a
# coaching_anchors_2d key (when present).
GT_TO_VERTEX = {
    "left_shoulder":  "acromion_left",
    "right_shoulder": "acromion_right",
    "left_hip":       "greater_trochanter_left",
    "right_hip":      "greater_trochanter_right",
    "neck_center":    "throat_midpoint",
}
GT_TO_PR7A_ANCHOR = {
    "left_shoulder":  "left_shoulder_visual",
    "right_shoulder": "right_shoulder_visual",
    "left_hip":       "left_hip_visual",
    "right_hip":      "right_hip_visual",
    "neck_center":    "neck_visual",
}

# Render config.
COLOR_PR7A_ANCHOR = (255,   0, 255)   # magenta — OLD PR-7a corrected
COLOR_VERTEX      = (255, 255,   0)   # cyan — NEW SMPL vertex sample
COLOR_GT          = (  0,   0, 255)   # red — Jason's GT label
COLOR_RAW_DOT     = (180, 180, 180)   # grey — raw WHAM 2D projection (faint)


def dist(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def extract_video_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"can't read frame {frame_idx} from {video_path}")
    return frame


def render_one_frame(
    *,
    video_path: Path,
    frame_idx: int,
    phase: str,
    view: str,
    verts_3d: np.ndarray,         # (6890, 3)
    landmark_indices: dict[str, dict],
    pr7a_anchors: dict[str, list | None],
    gt_labels: dict[str, dict],
    intrinsics: dict[str, float],
    out_png: Path,
) -> list[dict]:
    """
    Render one comparison frame and return per-landmark rows for the CSV.
    """
    img = extract_video_frame(video_path, frame_idx)
    H, W = img.shape[:2]
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]

    rows: list[dict] = []
    for gt_key, vertex_name in GT_TO_VERTEX.items():
        gt = gt_labels.get(gt_key)
        if gt is None:
            continue
        gt_xy = (float(gt["x"]), float(gt["y"]))

        # NEW: project the candidate SMPL vertex.
        info = landmark_indices.get(vertex_name)
        new_xy = None
        if info is not None:
            vert_idx = int(info["index"])
            v3 = verts_3d[vert_idx].tolist()
            new_xy = project_xyz_to_uv(v3, fx, fy, cx, cy)

        # OLD: PR-7a corrected coaching anchor at this frame.
        old_xy = pr7a_anchors.get(GT_TO_PR7A_ANCHOR.get(gt_key, ""))

        old_d = dist(old_xy, gt_xy) if old_xy else None
        new_d = dist(new_xy, gt_xy) if new_xy else None

        # Draw lines from each candidate to GT for visual distance.
        if new_xy is not None:
            cv2.line(img, (int(round(new_xy[0])), int(round(new_xy[1]))),
                     (int(round(gt_xy[0])), int(round(gt_xy[1]))),
                     COLOR_VERTEX, 1, cv2.LINE_AA)
            cv2.circle(img, (int(round(new_xy[0])), int(round(new_xy[1]))),
                       8, COLOR_VERTEX, 2, cv2.LINE_AA)
        if old_xy is not None:
            cv2.line(img, (int(round(old_xy[0])), int(round(old_xy[1]))),
                     (int(round(gt_xy[0])), int(round(gt_xy[1]))),
                     COLOR_PR7A_ANCHOR, 1, cv2.LINE_AA)
            cv2.circle(img, (int(round(old_xy[0])), int(round(old_xy[1]))),
                       7, COLOR_PR7A_ANCHOR, -1, cv2.LINE_AA)
        cv2.circle(img, (int(round(gt_xy[0])), int(round(gt_xy[1]))),
                   6, COLOR_GT, -1, cv2.LINE_AA)
        cv2.circle(img, (int(round(gt_xy[0])), int(round(gt_xy[1]))),
                   7, (0, 0, 0), 1, cv2.LINE_AA)

        rows.append({
            "frame_idx":   frame_idx,
            "phase":       phase,
            "landmark":    gt_key,
            "vertex_name": vertex_name,
            "vertex_index": info["index"] if info else None,
            "gt_2d_x":     gt_xy[0], "gt_2d_y": gt_xy[1],
            "old_anchor_x": old_xy[0] if old_xy else None,
            "old_anchor_y": old_xy[1] if old_xy else None,
            "new_vertex_x": new_xy[0] if new_xy else None,
            "new_vertex_y": new_xy[1] if new_xy else None,
            "old_dist_px": old_d,
            "new_dist_px": new_d,
            "improvement_px": (
                old_d - new_d if (old_d is not None and new_d is not None) else None
            ),
        })

    # Additionally project the lateral knee + lateral ankle landmarks
    # (no GT to compare against, but visual sanity).
    for extra_name in ("lateral_epicondyle_left", "lateral_epicondyle_right",
                        "lateral_malleolus_left", "lateral_malleolus_right"):
        info = landmark_indices.get(extra_name)
        if info is None:
            continue
        v3 = verts_3d[int(info["index"])].tolist()
        uv = project_xyz_to_uv(v3, fx, fy, cx, cy)
        if uv is not None:
            cv2.circle(img, (int(round(uv[0])), int(round(uv[1]))),
                       7, COLOR_VERTEX, 2, cv2.LINE_AA)
            cv2.putText(img, extra_name.split("_", 2)[2][:1] + "_" + extra_name.split("_")[0][0],
                        (int(round(uv[0])) + 10, int(round(uv[1])) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, extra_name.split("_", 2)[2][:1] + "_" + extra_name.split("_")[0][0],
                        (int(round(uv[0])) + 10, int(round(uv[1])) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_VERTEX, 1, cv2.LINE_AA)

    # Header bar with legend.
    cv2.rectangle(img, (0, 0), (W, 42), (40, 40, 40), -1)
    title = f"{view} / {phase} / frame {frame_idx}"
    cv2.putText(img, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "magenta=PR-7a anchor  cyan=SMPL vertex  red=GT",
                (W - 480, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), img)
    return rows


def main() -> None:
    landmarks = json.loads((PROBE_DIR / "smpl_landmark_indices.json").read_text())
    # b3fea3f0 (face_on) — 5 GT-labeled frames.
    video_id = "b3fea3f0-e248-44d7-a923-0bb43172b5bf"
    short = video_id.split("-")[0]
    view = "face_on"
    verts_sidecar = WHAM_DIR / video_id / "verts_at_frames.json"
    if not verts_sidecar.exists():
        sys.exit(f"[probe] verts sidecar missing: {verts_sidecar}\n"
                  "Run probe_extract_verts_modal.py first.")
    verts_data = json.loads(verts_sidecar.read_text())

    pr7a = json.loads(
        (COR_DIR / f"{short}_{view}_corrected.json").read_text()
    )
    pr7a_by_frame = {int(f["frame_idx"]): f for f in pr7a["frames"]}

    W = int(verts_data["video_width"])
    H = int(verts_data["video_height"])
    intr = default_intrinsics(W, H)

    video_path = VIDEOS / f"{video_id}.mp4"
    if not video_path.exists():
        sys.exit(f"[probe] missing video: {video_path}")

    csv_rows: list[dict] = []
    for fi_str, payload in verts_data["verts_at_frames"].items():
        frame_idx = int(fi_str)
        verts_3d = np.asarray(payload["verts"], dtype=np.float32)
        # Find matching GT label file.
        gt_files = list(GT_DIR.glob(f"{short}_*_{view}.json"))
        gt_for_this_frame = None
        for p in gt_files:
            d = json.loads(p.read_text())
            if int(d["frame_idx"]) == frame_idx:
                gt_for_this_frame = d
                break
        if gt_for_this_frame is None:
            print(f"[probe] no GT label for frame {frame_idx} — skipping")
            continue
        phase = gt_for_this_frame["phase"]
        pr7a_frame = pr7a_by_frame.get(frame_idx, {})
        pr7a_anchors = pr7a_frame.get("coaching_anchors_2d", {})

        out_png = PROBE_DIR / f"{short}_{view}_{phase}_frame{frame_idx}_compare.png"
        print(f"[probe] rendering {out_png.name}")
        rows = render_one_frame(
            video_path=video_path,
            frame_idx=frame_idx,
            phase=phase,
            view=view,
            verts_3d=verts_3d,
            landmark_indices=landmarks,
            pr7a_anchors=pr7a_anchors,
            gt_labels=gt_for_this_frame["labels"],
            intrinsics=intr,
            out_png=out_png,
        )
        csv_rows.extend(rows)

    # Write CSV.
    csv_path = PROBE_DIR / "distance_to_gt.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        if csv_rows:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
    print(f"[probe] wrote {csv_path}")
    print()

    # ── Acceptance summary ──────────────────────────────────────────
    print("=" * 78)
    print("PROBE ACCEPTANCE — aggregate per landmark class")
    print("=" * 78)
    print(f"{'landmark':<18s} {'n':>3s}  {'old_mean':>9s}  {'new_mean':>9s}  "
          f"{'beats_old':>10s}  {'verdict':>10s}")
    print("-" * 78)
    by_landmark: dict[str, list[dict]] = {}
    for r in csv_rows:
        by_landmark.setdefault(r["landmark"], []).append(r)
    pass_count = 0
    total_count = 0
    for name, rows in sorted(by_landmark.items()):
        olds = [r["old_dist_px"] for r in rows
                if r["old_dist_px"] is not None and r["new_dist_px"] is not None]
        news = [r["new_dist_px"] for r in rows
                if r["old_dist_px"] is not None and r["new_dist_px"] is not None]
        if not olds:
            continue
        old_mean = sum(olds) / len(olds)
        new_mean = sum(news) / len(news)
        beats = sum(1 for r in rows
                    if r["old_dist_px"] is not None
                    and r["new_dist_px"] is not None
                    and r["new_dist_px"] < r["old_dist_px"])
        # PASS criterion per spec: new closer than old on >= 3 of 4 frames
        # (for shoulders). Other landmarks: report only.
        verdict = ""
        if name in ("left_shoulder", "right_shoulder"):
            total_count += 1
            n = len(olds)
            need = (n * 3 + 3) // 4   # ceil(0.75 * n)
            if beats >= need:
                verdict = "PASS"
                pass_count += 1
            else:
                verdict = "FAIL"
        print(f"  {name:<18s} {len(olds):>3d}  {old_mean:>9.2f}  {new_mean:>9.2f}  "
              f"{beats}/{len(olds):>4s}  {verdict:>10s}")

    print()
    print(f"Shoulder PASS: {pass_count}/{total_count} sides "
          f"(both = strong PASS; 1 of 2 = mixed; 0 = FAIL)")


if __name__ == "__main__":
    main()
