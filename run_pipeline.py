#!/usr/bin/env python3
"""
run_pipeline.py — A→B→C→D→E→F end-to-end pipeline  (first closed loop)

Input:  one video file (DTL assumed for C/D/E/F; face-on stops after B)
Output: diagnosis JSON + one-liner + feature curve plot

Usage:
    bash run_with_gpu.sh run_pipeline.py <video.mp4> [--angle down-the-line|face-on|auto]
                                          [--plot] [--out DIR]

GT anchors can be supplied for calibration comparison (never used as GT by code):
    --gt_impact FRAME

FIRST CLOSED-LOOP ACCEPTANCE CRITERIA:
  - Normal DTL swings (201054, 201058) must output  root_cause="none"
  - No false alarms on clean swings
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.a_measurement.pose_pipeline import PosePipeline
from engine.b_phase.swing_phase import SwingPhaseEngine, PHASE_NAMES
from engine.c_features.feature_extractor import FeatureExtractor
from engine.layer0.perception_gate import PerceptionGate
from src.judgment.rules import (
    bone_length_sentinel,
    r1_loss_of_posture,
    r2_hip_toward_ball,
)
from src.judgment.root_cause import RootCauseEngine
from src.judgment.output import CoachingOutput

KP_CACHE = Path("engine/kp_cache")
KP_CACHE.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_or_run_rtmpose(video_path: str, verbose: bool = False):
    vname = Path(video_path).name
    cache = KP_CACHE / f"{Path(video_path).stem}.json"
    pipeline = PosePipeline(device="cuda")
    if cache.exists():
        print(f"  A-layer: loading cache {cache.name}")
        import json as _j
        with open(cache) as f:
            kp_json = _j.load(f)
        return pipeline.run_from_json(kp_json)
    else:
        print(f"  A-layer: running RTMPose on {vname}...")
        t0 = time.time()
        meas, fps = pipeline.run(video_path, verbose=verbose)
        print(f"  A-layer: done in {time.time()-t0:.1f}s")
        _save_cache(meas, fps, vname, cache, pipeline)
        return meas, fps


def _save_cache(measurements, fps, vname, cache_path, pipeline):
    from engine.a_measurement.pose_pipeline import JOINT_NAMES
    frames = []
    for m in measurements:
        persons = []
        if m.measurement_quality != "bad":
            kps = {}
            for name in JOINT_NAMES:
                pt = m.keypoints.get(name)
                sc = m.confidences.get(name, 0.0)
                kps[name] = {"x": float(pt[0]) if pt else 0.0,
                             "y": float(pt[1]) if pt else 0.0,
                             "score": sc}
            persons = [{"person_id": 0, "keypoints": kps}]
        frames.append({"frame": m.frame_idx, "persons": persons})
    data = {"model": "RTMPose-x", "keypoint_format": "COCO-17",
            "stats": {"source_fps": fps, "video": vname}, "frames": frames}
    with open(cache_path, "w") as f:
        json.dump(data, f)
    print(f"  Cached: {cache_path.name}")


def phase_summary(annotations) -> dict:
    s = {}
    for a in annotations:
        p = a.phase
        if p not in s:
            s[p] = [a.frame_idx, a.frame_idx, 0]
        else:
            s[p][1] = a.frame_idx
        s[p][2] += 1
    return {p: tuple(v) for p, v in s.items()}


# ── main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    video_path: str,
    angle: str = "auto",
    gt_impact: Optional[int] = None,
    out_dir: str = "pipeline_output",
    do_plot: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Run A→B→C→D→E→F on a single video.

    Returns a results dict with all layer outputs.
    """
    vpath  = Path(video_path)
    stem   = vpath.stem
    out    = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Pipeline: {stem}  [{angle}]")

    # ── A layer ───────────────────────────────────────────────────────────────
    measurements, fps = load_or_run_rtmpose(video_path, verbose=verbose)
    n = len(measurements)
    print(f"  A-layer: {n} frames @ {fps:.1f} fps")

    # ── B layer ───────────────────────────────────────────────────────────────
    engine_b = SwingPhaseEngine()
    annotations, anchors = engine_b.run(measurements, fps, angle=angle)
    psummary = phase_summary(annotations)
    sc = anchors.swing_count; fse = anchors.first_swing_end
    print(f"  B-layer: swing_count={sc}  first_swing_end=fr{fse}")
    print(f"  Anchors: addr=fr{anchors.address}  top=fr{anchors.top}(tc={anchors.top_conf:.2f})"
          f"  impact=fr{anchors.impact}(ic={anchors.impact_conf:.2f})  finish=fr{anchors.finish}")

    # GT deviation report (informational only — GT from human annotation)
    if gt_impact is not None:
        err = anchors.impact - gt_impact
        ok  = abs(err) <= 2
        print(f"  GT impact=fr{gt_impact}  detected=fr{anchors.impact}  "
              f"error={err:+d}fr  [{'PASS' if ok else 'FAIL (>±2fr)'}]")

    # ── C layer ───────────────────────────────────────────────────────────────
    extractor = FeatureExtractor()
    feat = extractor.extract(measurements, fps, angle, anchors.address)
    unreliable_ratio = float(np.mean(feat.unreliable))
    print(f"  C-layer: torso_h={feat.torso_h:.0f}px  "
          f"addr_spine={feat.meta['addr_spine_angle']:.1f}deg  "
          f"unreliable_frames={unreliable_ratio:.1%}")

    # ── Phase labels for D-layer ──────────────────────────────────────────────
    phase_labels = [a.phase for a in annotations]

    # ── Bone-length sentinel → unreliable mask ────────────────────────────────
    # FeatureExtractor already computed unreliable; pass it to rules
    # Also compute bone_length_ratios dict for the official sentinel call
    bone_length_ratios: dict[str, np.ndarray] = {}
    # Only use lower-body bones for sentinel — arm bones project differently during swing
    bone_keys = ["left_hip_left_knee", "right_hip_right_knee"]
    for bk in bone_keys:
        lengths = np.array([m.bone_lengths.get(bk, 0.0) for m in measurements])
        med = float(np.median(lengths[lengths > 0])) if np.any(lengths > 0) else 1.0
        if med > 0:
            bone_length_ratios[bk] = lengths / med
    unreliable_mask = bone_length_sentinel(bone_length_ratios)

    # ── D layer ───────────────────────────────────────────────────────────────
    faults = []
    if angle == "down-the-line":
        r1 = r1_loss_of_posture(
            feat.spine_delta, phase_labels,
            joint_confidences=feat.joint_conf,
            unreliable_mask=unreliable_mask if len(unreliable_mask) == n else None,
        )
        r2 = r2_hip_toward_ball(
            feat.hip_disp, phase_labels,
            joint_confidences=feat.joint_conf,
            unreliable_mask=unreliable_mask if len(unreliable_mask) == n else None,
        )
        if r1: faults.append(r1); print(f"  D-layer R1: {r1.fault_type} {r1.severity} conf={r1.confidence:.3f}")
        if r2: faults.append(r2); print(f"  D-layer R2: {r2.fault_type} {r2.severity} conf={r2.confidence:.3f}")
        if not faults: print("  D-layer: no faults detected")
    else:
        print("  D-layer: face-on — C/D/E/F features skipped (DTL only in first loop)")

    # ── E layer ───────────────────────────────────────────────────────────────
    engine_e = RootCauseEngine()
    rc = engine_e.analyze(faults)
    print(f"  E-layer: root_cause={rc.root_cause}  certainty={rc.certainty}")

    # ── F layer ───────────────────────────────────────────────────────────────
    coaching = CoachingOutput()
    out_f = coaching.generate(rc, unreliable_frame_ratio=unreliable_ratio)
    print(f"  F-layer: {out_f.one_liner}")

    # ── Save diagnosis JSON ───────────────────────────────────────────────────
    diag = {
        "video": stem,
        "angle": angle,
        "fps":   fps,
        "n_frames": n,
        "b_layer": {
            "swing_count":      sc,
            "first_swing_end":  fse,
            "address":          anchors.address,
            "top":              anchors.top,
            "top_conf":         anchors.top_conf,
            "impact":           anchors.impact,
            "impact_conf":      anchors.impact_conf,
            "finish":           anchors.finish,
        },
        "gt_impact":       gt_impact,
        "gt_error_frames": (anchors.impact - gt_impact) if gt_impact is not None else None,
        "c_layer": {
            "torso_h":          feat.torso_h,
            "addr_spine_deg":   feat.meta["addr_spine_angle"],
            "unreliable_ratio": round(unreliable_ratio, 4),
        },
        "d_layer_faults":  [f.to_dict() for f in faults],
        **out_f.diagnosis_json,
    }
    json_path = out / f"{stem}_diagnosis.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(diag, jf, indent=2, ensure_ascii=False)
    print(f"  Saved: {json_path}")

    # ── Feature curve plot ────────────────────────────────────────────────────
    if do_plot and angle == "down-the-line":
        plot_path = out / f"{stem}_features.png"
        _make_feature_plot(
            stem, fps, n, feat, phase_labels, anchors,
            faults, rc, str(plot_path)
        )
        # Copy to Windows desktop
        desk = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/pipeline")
        desk.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(str(plot_path), str(desk / plot_path.name))
        shutil.copy(str(json_path), str(desk / json_path.name))
        print(f"  Copied plot+JSON to desktop: {desk}")

    return diag


# ── feature plot ──────────────────────────────────────────────────────────────

PHASE_COLORS_MPL = {
    "address": "#888888", "takeaway": "#c8960a", "backswing": "#c86428",
    "top": "#3232dc", "transition": "#b432b4", "downswing": "#32b4dc",
    "impact": "#32dc32", "follow_through": "#64c864",
}


def _make_feature_plot(stem, fps, n, feat, phase_labels, anchors,
                       faults, rc, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  (matplotlib not available, skipping plot)")
        return

    frames = np.arange(n)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"Feature Curves — {stem}  [{anchors.swing_count} swing(s)]\n"
        f"E-layer: root_cause={rc.root_cause}  certainty={rc.certainty}",
        fontsize=12, fontweight="bold"
    )

    # ── Panel 1: spine_delta ─────────────────────────────────────────────────
    ax0 = axes[0]
    ax0.plot(frames, feat.spine_delta, color="#2c3e50", lw=1.5, label="spine_delta (deg)")
    ax0.axhline(8.0,  color="#e74c3c", lw=1, ls="--", alpha=0.7, label="R1 threshold (8 deg)")
    ax0.axhline(12.0, color="#c0392b", lw=1, ls=":",  alpha=0.7, label="R1 mild_max (12 deg)")
    ax0.axhline(0.0,  color="#7f8c8d", lw=0.5, alpha=0.5)
    _shade_phases(ax0, phase_labels, n)
    _draw_anchors(ax0, anchors)
    ax0.set_ylabel("Spine delta (deg)", fontsize=10)
    ax0.set_title("R1: Spine Forward Tilt Change from Address", fontsize=9)
    ax0.legend(loc="upper left", fontsize=8)

    # ── Panel 2: hip_disp ────────────────────────────────────────────────────
    ax1 = axes[1]
    ax1.plot(frames, feat.hip_disp, color="#8e44ad", lw=1.5, label="hip_fwd_disp (frac torso_h)")
    ax1.axhline(0.05, color="#e74c3c", lw=1, ls="--", alpha=0.7, label="R2 threshold (0.05)")
    ax1.axhline(0.09, color="#c0392b", lw=1, ls=":",  alpha=0.7, label="R2 mild_max (0.09)")
    ax1.axhline(0.0,  color="#7f8c8d", lw=0.5, alpha=0.5)
    _shade_phases(ax1, phase_labels, n)
    _draw_anchors(ax1, anchors)
    ax1.set_ylabel("Hip fwd disp (frac)", fontsize=10)
    ax1.set_title("R2: Hip Forward Displacement (normalised by torso height)", fontsize=9)
    ax1.legend(loc="upper left", fontsize=8)

    # ── Panel 3: joint confidence ────────────────────────────────────────────
    ax2 = axes[2]
    ax2.plot(frames, feat.joint_conf, color="#27ae60", lw=1.2, label="joint_conf")
    ax2.fill_between(frames, feat.unreliable.astype(float) * 0.3,
                     color="#e74c3c", alpha=0.25, label="unreliable (sentinel)")
    ax2.axhline(0.4, color="#e67e22", lw=1, ls="--", alpha=0.7, label="conf threshold (0.4)")
    _shade_phases(ax2, phase_labels, n)
    _draw_anchors(ax2, anchors)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("Joint confidence", fontsize=10)
    ax2.set_xlabel("Frame", fontsize=10)
    ax2.set_title("Joint Quality (shoulder + hip mean confidence)", fontsize=9)
    ax2.legend(loc="upper left", fontsize=8)

    # Fault onset markers
    for fault in faults:
        onset = fault.onset_frame
        if onset is not None:
            for ax in axes:
                ax.axvline(onset, color="#e74c3c", lw=2, ls="--",
                           alpha=0.8, label=f"fault onset {fault.fault_type}")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {out_path}")


def _shade_phases(ax, phase_labels, n):
    """Background phase-color shading."""
    import matplotlib.pyplot as plt
    prev_phase = None; start = 0
    for i, p in enumerate(phase_labels):
        if p != prev_phase:
            if prev_phase is not None:
                c = PHASE_COLORS_MPL.get(prev_phase, "#cccccc")
                ax.axvspan(start, i, alpha=0.08, color=c, zorder=0)
            prev_phase = p; start = i
    if prev_phase is not None:
        ax.axvspan(start, n, alpha=0.08, color=PHASE_COLORS_MPL.get(prev_phase, "#cccccc"))


def _draw_anchors(ax, anchors):
    ANCHOR_COLORS = {"A": "#888888", "T": "#6464ff", "I": "#20dd20", "F": "#b432b4"}
    for label, fr in [("A", anchors.address), ("T", anchors.top),
                       ("I", anchors.impact),  ("F", anchors.finish)]:
        ax.axvline(fr, color=ANCHOR_COLORS[label], lw=1.5, ls="--", alpha=0.9)
        ymin, ymax = ax.get_ylim()
        ax.text(fr+0.5, ymin + (ymax-ymin)*0.03, label,
                color=ANCHOR_COLORS[label], fontsize=8, fontweight="bold")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="SwingCue A→B→C→D→E→F pipeline")
    ap.add_argument("video", help="Video file path")
    ap.add_argument("--angle", default="auto",
                    choices=["auto","face-on","down-the-line"])
    ap.add_argument("--gt_impact", type=int, default=None,
                    help="Human-annotated GT impact frame (informational only)")
    ap.add_argument("--out", default="pipeline_output")
    ap.add_argument("--no_plot", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    result = run_pipeline(
        video_path=args.video,
        angle=args.angle,
        gt_impact=args.gt_impact,
        out_dir=args.out,
        do_plot=not args.no_plot,
        verbose=args.verbose,
    )

    print("\n=== DIAGNOSIS ===")
    print(json.dumps({
        "root_cause":  result.get("root_cause"),
        "certainty":   result.get("certainty"),
        "one_liner":   result.get("one_liner"),
        "b_layer":     result.get("b_layer"),
        "gt_error_fr": result.get("gt_error_frames"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
