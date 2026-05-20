"""
metrics.py — single-runner and cross-runner metrics.

Single-runner: frame coverage, mean confidence per kp, presence rate.
Cross-runner: per-kp Euclidean distance frame-by-frame, mean + max
              disagreement, "agreement %" within a px threshold.

Output is a `<runner>.metrics.json` for single-runner numbers and a
`comparison_<video_id>.metrics.json` for cross-runner numbers.

CLI:
    # single-runner stats:
    python -m benchmark.metrics single output/<runner>/<video_id>/keypoints.json

    # cross-runner comparison (baseline must come first):
    python -m benchmark.metrics compare \\
        output/mediapipe_pose/<video_id>/keypoints.json \\
        output/mediapipe_tasks/<video_id>/keypoints.json \\
        output/movenet_thunder/<video_id>/keypoints.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from .runner import COCO_NAMES

# Threshold for "agrees with baseline" (Euclidean px). 30 px on a
# 720x1280 swing video is roughly 4% of the short axis — about a
# half-disc-radius at setup. Tunable; report uses this as one column.
AGREEMENT_PX_THRESHOLD: float = 30.0


def _coverage(frames: list[dict]) -> dict[str, Any]:
    """Per-kp visibility rate + overall frame coverage."""
    total = len(frames)
    if total == 0:
        return {"frames": 0}
    by_kp: dict[str, int] = {n: 0 for n in COCO_NAMES}
    any_kp = 0
    conf_sum: dict[str, float] = {n: 0.0 for n in COCO_NAMES}
    conf_cnt: dict[str, int]  = {n: 0 for n in COCO_NAMES}
    for f in frames:
        has_any = False
        for name, kp in f["keypoints"].items():
            if kp[0] is not None:
                by_kp[name] += 1
                has_any = True
            conf_sum[name] += float(kp[2])
            conf_cnt[name] += 1
        if has_any:
            any_kp += 1
    return {
        "frames":             total,
        "frames_with_any_kp": any_kp,
        "frame_coverage":     round(any_kp / total, 3),
        "kp_visibility":      {n: round(by_kp[n] / total, 3) for n in COCO_NAMES},
        "kp_mean_conf":       {
            n: round(conf_sum[n] / max(1, conf_cnt[n]), 3) for n in COCO_NAMES
        },
    }


def single_runner_metrics(run_json: Path) -> dict:
    data = json.loads(run_json.read_text())
    return {
        "video_id":     data["video_id"],
        "runner":       data["runner"],
        "video_width":  data["video_width"],
        "video_height": data["video_height"],
        "fps_native":   data["fps_native"],
        "fps_sampled":  data["fps_sampled"],
        "duration_sec": data["duration_sec"],
        "stats":        _coverage(data["frames"]),
    }


def _align_frames(baseline: list[dict], other: list[dict]) -> list[tuple[dict, dict]]:
    """
    Pair frames by nearest ts. Assumes both lists are sorted ascending.
    Drops other-frames whose nearest baseline mate is more than 0.1s away
    (avoids spurious comparisons when sample rates diverge).
    """
    if not baseline or not other:
        return []
    pairs: list[tuple[dict, dict]] = []
    b_idx = 0
    for o in other:
        # Advance baseline cursor to the closest ts to o["ts"].
        while (b_idx + 1 < len(baseline)
               and abs(baseline[b_idx + 1]["ts"] - o["ts"])
               <  abs(baseline[b_idx]["ts"] - o["ts"])):
            b_idx += 1
        if abs(baseline[b_idx]["ts"] - o["ts"]) > 0.1:
            continue
        pairs.append((baseline[b_idx], o))
    return pairs


def cross_runner_metrics(
    baseline_path: Path,
    other_paths: list[Path],
) -> dict:
    base = json.loads(baseline_path.read_text())
    out: dict[str, Any] = {
        "video_id": base["video_id"],
        "baseline": base["runner"],
        "comparisons": {},
    }
    for op in other_paths:
        other = json.loads(op.read_text())
        pairs = _align_frames(base["frames"], other["frames"])
        per_kp_dists: dict[str, list[float]] = {n: [] for n in COCO_NAMES}
        agreements: dict[str, int] = {n: 0 for n in COCO_NAMES}
        valid_counts: dict[str, int] = {n: 0 for n in COCO_NAMES}
        for b_frame, o_frame in pairs:
            for name in COCO_NAMES:
                b_kp = b_frame["keypoints"][name]
                o_kp = o_frame["keypoints"][name]
                if b_kp[0] is None or o_kp[0] is None:
                    continue
                dx = b_kp[0] - o_kp[0]
                dy = b_kp[1] - o_kp[1]
                d = math.sqrt(dx * dx + dy * dy)
                per_kp_dists[name].append(d)
                valid_counts[name] += 1
                if d <= AGREEMENT_PX_THRESHOLD:
                    agreements[name] += 1
        # Aggregate.
        out["comparisons"][other["runner"]] = {
            "paired_frames": len(pairs),
            "per_kp": {
                name: {
                    "n":                     valid_counts[name],
                    "mean_px":               (round(sum(per_kp_dists[name]) / valid_counts[name], 1)
                                              if valid_counts[name] else None),
                    "max_px":                (round(max(per_kp_dists[name]), 1)
                                              if per_kp_dists[name] else None),
                    "agreement_pct_at_30px": (round(agreements[name] / valid_counts[name], 3)
                                              if valid_counts[name] else None),
                }
                for name in COCO_NAMES
            },
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("single")
    s.add_argument("run_json", type=Path)
    s.add_argument("--out", type=Path, default=None,
                   help="write to <run_json>.metrics.json by default")
    c = sub.add_parser("compare")
    c.add_argument("baseline", type=Path)
    c.add_argument("others",   type=Path, nargs="+")
    c.add_argument("--out",    type=Path, default=None)
    args = ap.parse_args()

    if args.cmd == "single":
        result = single_runner_metrics(args.run_json)
        out = args.out or args.run_json.with_suffix(".metrics.json")
        out.write_text(json.dumps(result, indent=2))
        print(f"[metrics] wrote {out}")
    else:
        result = cross_runner_metrics(args.baseline, args.others)
        out = args.out or (args.baseline.parent.parent.parent
                           / f"comparison_{result['video_id']}.metrics.json")
        out.write_text(json.dumps(result, indent=2))
        print(f"[metrics] wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
