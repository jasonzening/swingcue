"""
ghost004_step1_rtmpose.py  —  GHOST-004 Step 1/2
RTMPose extraction + B-layer 8-phase detection for coach videos.

Runs in main .venv (rtmlib available).
Outputs:
  engine/kp_cache/ghost004/coach-fo.json
  engine/kp_cache/ghost004/coach-dtl.json
  output/ghost004/phase_report_step1.json   (anchors + confidence per video)
  output/ghost004/iou_conf_fo.npy  (per-frame confidence array)
  output/ghost004/iou_conf_dtl.npy

Usage:
  cd /home/jason/projects/swingcue-postest
  source .venv/bin/activate
  python ghost004_step1_rtmpose.py
"""
import os, sys, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

VIDEOS = {
    "fo":  Path("/mnt/c/Users/jason/Zening/Swingcue/教学视频/coach-video/coach-fo.mp4"),
    "dtl": Path("/mnt/c/Users/jason/Zening/Swingcue/教学视频/coach-video/coach-dtl.mp4"),
}
ANGLES = {"fo": "face-on", "dtl": "down-the-line"}

KP_CACHE_DIR = Path("engine/kp_cache/ghost004")
OUT_DIR      = Path("output/ghost004")
KP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

JOINT_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]
CONF_THRESHOLD = 0.35

# ── RTMPose inference ─────────────────────────────────────────────────────────
from engine.a_measurement.pose_pipeline import PosePipeline, FrameMeasurement
from engine.b_phase.swing_phase import SwingPhaseEngine, PHASE_NAMES

def run_rtmpose(video_path: Path, cache_path: Path) -> tuple:
    """Run RTMPose or load cache. Returns (measurements, fps)."""
    pipeline = PosePipeline(device="cuda")

    if cache_path.exists():
        print(f"  cache hit: {cache_path.name}")
        with open(cache_path) as f:
            kp_json = json.load(f)
        meas, fps = pipeline.run_from_json(kp_json)
        return meas, fps

    print(f"  RTMPose inference: {video_path.name}...")
    t0 = time.time()
    meas, fps = pipeline.run(str(video_path))
    print(f"  done in {time.time()-t0:.1f}s  NF={len(meas)}  fps={fps:.2f}")

    # Save kp_cache (same format as batch2)
    frames_out = []
    for m in meas:
        persons = []
        if m.measurement_quality != "bad":
            kps = {}
            for name in JOINT_NAMES:
                pt = m.keypoints.get(name)
                sc = m.confidences.get(name, 0.0)
                kps[name] = {
                    "x": float(pt[0]) if pt else 0.0,
                    "y": float(pt[1]) if pt else 0.0,
                    "score": float(sc)
                }
            persons = [{"person_id": 0, "keypoints": kps}]
        frames_out.append({"frame": m.frame_idx, "persons": persons})

    data = {
        "model": "RTMPose-x",
        "keypoint_format": "COCO-17",
        "stats": {"source_fps": fps, "video": video_path.name},
        "frames": frames_out
    }
    with open(cache_path, "w") as f:
        json.dump(data, f)
    print(f"  saved: {cache_path}")
    return meas, fps


def compute_per_frame_conf(meas: list) -> np.ndarray:
    """Mean keypoint confidence per frame for key joints."""
    KEY_JOINTS = ["left_shoulder","right_shoulder","left_elbow","right_elbow",
                  "left_wrist","right_wrist","left_hip","right_hip",
                  "left_knee","right_knee"]
    arr = []
    for m in meas:
        scores = [m.confidences.get(j, 0.0) for j in KEY_JOINTS]
        arr.append(float(np.mean(scores)))
    return np.array(arr)


def find_bad_frames(meas: list, conf_arr: np.ndarray, threshold=0.50) -> list:
    """Return frames with quality=bad or mean_conf < threshold."""
    bad = []
    for i, m in enumerate(meas):
        reason = None
        if m.measurement_quality == "bad":
            reason = "no_detection"
        elif conf_arr[i] < threshold:
            reason = f"low_conf={conf_arr[i]:.2f}"
        if reason:
            bad.append({"frame": m.frame_idx, "reason": reason})
    return bad


# ── main ──────────────────────────────────────────────────────────────────────
phase_engine = SwingPhaseEngine()
report = {}

for key, video_path in VIDEOS.items():
    print(f"\n{'='*60}")
    print(f"Processing {key.upper()}: {video_path.name}")
    print(f"{'='*60}")

    cache_path = KP_CACHE_DIR / f"coach-{key}.json"
    meas, fps = run_rtmpose(video_path, cache_path)
    NF = len(meas)
    print(f"  NF={NF}  fps={fps:.2f}")

    # Per-frame confidence
    conf_arr = compute_per_frame_conf(meas)
    np.save(OUT_DIR / f"kp_conf_{key}.npy", conf_arr)
    print(f"  conf: mean={conf_arr.mean():.3f}  min={conf_arr.min():.3f}  "
          f"P5={np.percentile(conf_arr,5):.3f}")

    # Bad frame detection
    bad_frames = find_bad_frames(meas, conf_arr)
    print(f"  bad frames: {len(bad_frames)}")
    for b in bad_frames[:10]:
        print(f"    fr{b['frame']:03d}  {b['reason']}")

    # B-layer 8-phase detection
    angle = ANGLES[key]
    print(f"\n  B-layer ({angle})...")
    try:
        annotations, anchors = phase_engine.run(meas, fps, angle=angle)
        print(f"  anchors: address=fr{anchors.address}  top=fr{anchors.top}  "
              f"impact=fr{anchors.impact}  finish=fr{anchors.finish}")
        print(f"  impact_conf={anchors.impact_conf:.3f}  top_conf={anchors.top_conf:.3f}  "
              f"swing_count={anchors.swing_count}")

        # Phase timeline
        phase_map: dict = {}
        for a in annotations:
            p = a.phase
            if p not in phase_map:
                phase_map[p] = {"start": a.frame_idx, "end": a.frame_idx, "count": 0}
            else:
                phase_map[p]["end"] = a.frame_idx
            phase_map[p]["count"] += 1

        print(f"\n  8-phase timeline:")
        for ph in PHASE_NAMES:
            if ph in phase_map:
                pm = phase_map[ph]
                print(f"    {ph:20s} fr{pm['start']:03d}–fr{pm['end']:03d}  ({pm['count']} frames)")
            else:
                print(f"    {ph:20s} —")

        report[key] = {
            "video": video_path.name,
            "nf": NF,
            "fps": fps,
            "angle": angle,
            "conf_mean": float(conf_arr.mean()),
            "conf_min":  float(conf_arr.min()),
            "conf_p5":   float(np.percentile(conf_arr, 5)),
            "bad_frames": bad_frames,
            "anchors": {
                "address": anchors.address,
                "top":     anchors.top,
                "impact":  anchors.impact,
                "finish":  anchors.finish,
                "impact_conf": float(anchors.impact_conf),
                "top_conf":    float(anchors.top_conf),
                "swing_count": anchors.swing_count,
            },
            "phase_map": phase_map,
        }

    except Exception as e:
        print(f"  B-layer ERROR: {e}")
        report[key] = {
            "video": video_path.name,
            "nf": NF,
            "fps": fps,
            "angle": angle,
            "conf_mean": float(conf_arr.mean()),
            "bad_frames": bad_frames,
            "b_layer_error": str(e),
        }

# ── cross-video phase alignment check ─────────────────────────────────────────
print(f"\n{'='*60}")
print("CROSS-VIDEO PHASE ALIGNMENT CHECK")
print(f"{'='*60}")

def phase_normalized(phase_map, nf):
    """Return normalized [0,1] midpoint of each phase."""
    normed = {}
    for ph, v in phase_map.items():
        mid = (v["start"] + v["end"]) / 2.0
        normed[ph] = mid / nf
    return normed

if "fo" in report and "anchors" in report["fo"] and "dtl" in report and "anchors" in report["dtl"]:
    fo_anch  = report["fo"]["anchors"]
    dtl_anch = report["dtl"]["anchors"]
    fo_nf    = report["fo"]["nf"]
    dtl_nf   = report["dtl"]["nf"]

    print(f"  {'phase':15s}  {'FO norm':>9s}  {'DTL norm':>9s}  {'delta':>8s}")
    for ph in ["address","top","impact","finish"]:
        fo_key  = {"address":"address","top":"top","impact":"impact","finish":"finish"}[ph]
        fo_fr   = fo_anch[fo_key]
        dtl_fr  = dtl_anch[fo_key]
        fo_n    = fo_fr / fo_nf
        dtl_n   = dtl_fr / dtl_nf
        delta   = abs(fo_n - dtl_n)
        flag    = " ⚠ MISALIGN" if delta > 0.10 else " ✓"
        print(f"  {ph:15s}  {fo_n:9.3f}  {dtl_n:9.3f}  {delta:8.3f}{flag}")

    report["cross_alignment"] = {
        ph: {
            "fo_norm":  report["fo"]["anchors"][ph] / fo_nf if ph in report["fo"]["anchors"] else None,
            "dtl_norm": report["dtl"]["anchors"][ph] / dtl_nf if ph in report["dtl"]["anchors"] else None,
        } for ph in ["address","top","impact","finish"]
    }

# ── save report ───────────────────────────────────────────────────────────────
report_path = OUT_DIR / "phase_report_step1.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nSaved: {report_path}")
print("Step 1 complete. Run ghost004_step2_mhr.py next (sam3d_venv).")
