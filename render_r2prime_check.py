#!/usr/bin/env python3
"""
render_r2prime_check.py
Renders 12 frames (6 peak + 6 address) with SAM2 mask overlay for R2' visual inspection.

Each frame gets:
  - Semi-transparent cyan mask overlay (alpha=0.35)
  - Hip band horizontal lines (yellow, hip_mid_y ± 0.12*torso_h)
  - Detected rear-edge point (large red dot)
  - Address rear-edge vertical reference line (orange dashed look)
  - Hip_mid keypoint (blue dot)
  - Frame info banner
  - mask_quality score

Outputs: Desktop/rtmpose_results/preview/batch2/r2prime_check/
"""
import sys, json, math
from pathlib import Path
import numpy as np
import cv2
import torch

sys.path.insert(0, "/home/jason/projects/swingcue-postest")

from engine.a_measurement.pose_pipeline import PosePipeline
from engine.b_phase.swing_phase import SwingPhaseEngine
from engine.orientation.resolver import OrientationResolver
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

PROJ   = Path("/home/jason/projects/swingcue-postest")
DEST   = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/batch2/r2prime_check")
DEST.mkdir(parents=True, exist_ok=True)

SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_CKPT = str(PROJ / "models/sam2/sam2.1_hiera_tiny.pt")
BAND_FRAC = 0.12
FONT = cv2.FONT_HERSHEY_DUPLEX

VIDEOS = [
    ("dtl-ok-1",    84,
     PROJ/"engine/kp_cache/batch2/dtl-ok-1.json",
     PROJ/"input/dtl-ok-1.mp4"),
    ("dtl-ok-2",    109,
     PROJ/"engine/kp_cache/batch2/dtl-ok-2.json",
     PROJ/"input/dtl-ok-2.mp4"),
    ("dtl-wrong-1", 98,
     PROJ/"engine/kp_cache/batch2/dtl-wrong-1.json",
     PROJ/"input/dtl-wrong-1.mp4"),
    ("dtl-wrong-2", 85,
     PROJ/"engine/kp_cache/batch2/dtl-wrong-2.json",
     PROJ/"input/dtl-wrong-2.mp4"),
    ("201058",      181,
     PROJ/"engine/kp_cache/Videos2026-06-09_201058_697.json",
     PROJ/"input/Videos2026-06-09_201058_697.mp4"),
    ("201054",      153,
     PROJ/"engine/kp_cache/Videos2026-06-09_201054_561.json",
     PROJ/"input/Videos2026-06-09_201054_561.mp4"),
]

# Colors BGR
C_MASK_OVERLAY = (255, 200, 0)    # cyan-yellow mask tint
C_BAND_LO      = (0,   220, 255)  # yellow hip band line
C_BAND_HI      = (0,   220, 255)
C_REAR_EDGE    = (0,   50,  255)  # red dot = detected rear edge
C_ADDR_REF     = (0,   165, 255)  # orange vertical = address rear-edge ref
C_HIP_MID      = (255, 180, 0)    # blue dot = hip_mid
C_WHITE        = (255, 255, 255)
C_BLACK        = (0,   0,   0)


def get_frame(vpath, idx):
    cap = cv2.VideoCapture(str(vpath))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, f = cap.read(); cap.release()
    return f if ret else np.zeros((1280, 720, 3), np.uint8)


def kp_valid(k, thr=0.3):
    return k["score"] >= thr and k["x"] > 0 and k["y"] > 0


def hip_mid_from_kp(kps, thr=0.3):
    lh = kps.get("left_hip",  {}); rh = kps.get("right_hip", {})
    lv = kp_valid(lh, thr);        rv = kp_valid(rh, thr)
    if lv and rv:
        return ((lh["x"]+rh["x"])/2, (lh["y"]+rh["y"])/2)
    if lv: return (lh["x"], lh["y"])
    if rv: return (rh["x"], rh["y"])
    return None


def torso_h_from_kp(kps, thr=0.3):
    lh = kps.get("left_hip",{}); rh = kps.get("right_hip",{})
    ls = kps.get("left_shoulder",{}); rs = kps.get("right_shoulder",{})
    hip_y  = [(lh["y"] if kp_valid(lh,thr) else None),
              (rh["y"] if kp_valid(rh,thr) else None)]
    sh_y   = [(ls["y"] if kp_valid(ls,thr) else None),
              (rs["y"] if kp_valid(rs,thr) else None)]
    hip_y  = [v for v in hip_y if v]; sh_y = [v for v in sh_y if v]
    if hip_y and sh_y:
        return sum(hip_y)/len(hip_y) - sum(sh_y)/len(sh_y)
    return 200.0


def run_sam2_frame(predictor, frame_bgr, hip_pt):
    """Run SAM2 on one BGR frame, return (mask_bool, score)."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(frame_rgb)
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[hip_pt[0], hip_pt[1]]]),
        point_labels=np.array([1]),
        multimask_output=False,
    )
    return masks[0].astype(bool), float(scores[0])


def find_rear_edge(mask, band_y_lo, band_y_hi, rear_is_left):
    h, w = mask.shape
    r0 = max(0, int(band_y_lo)); r1 = min(h-1, int(band_y_hi)+1)
    band = mask[r0:r1, :]
    col_occ = np.any(band, axis=0)
    occupied = np.where(col_occ)[0]
    if len(occupied) == 0:
        return None
    return float(occupied.min() if rear_is_left else occupied.max())


def render_frame(frame_bgr, mask, hip_pt, torso_h, band_y_lo, band_y_hi,
                 rear_x, addr_rear_x, ball_side, fr_idx, stem, phase,
                 mask_quality, hip_rear_pct=None):
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()

    # --- 1. Semi-transparent mask overlay ---
    if mask is not None:
        overlay = out.copy()
        overlay[mask] = (
            (overlay[mask].astype(np.float32) * 0.65 +
             np.array(C_MASK_OVERLAY, dtype=np.float32) * 0.35)
            .clip(0, 255).astype(np.uint8)
        )
        # Mask contour
        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
        out = overlay

    # --- 2. Hip band horizontal lines ---
    for y, label in [(int(band_y_lo), "band-lo"), (int(band_y_hi), "band-hi")]:
        if 0 <= y < h:
            cv2.line(out, (0, y), (w, y), C_BAND_LO, 2, cv2.LINE_AA)
    # Center line (dashed look via segments)
    cy = int((band_y_lo + band_y_hi) / 2)
    for x0 in range(0, w, 20):
        cv2.line(out, (x0, cy), (min(x0+10, w), cy), (0, 180, 200), 1)

    # --- 3. Address rear-edge reference vertical line (orange) ---
    if addr_rear_x is not None:
        ax = int(addr_rear_x)
        if 0 <= ax < w:
            for y0 in range(0, h, 18):
                cv2.line(out, (ax, y0), (ax, min(y0+10, h)), C_ADDR_REF, 2)
            # Label
            cv2.putText(out, "ADDR_REAR", (ax+4, int(band_y_lo)-8),
                        FONT, 0.40, C_ADDR_REF, 1, cv2.LINE_AA)

    # --- 4. Detected rear-edge point (large red circle) ---
    if rear_x is not None:
        rx = int(rear_x)
        ry = int((band_y_lo + band_y_hi) / 2)
        if 0 <= rx < w:
            cv2.circle(out, (rx, ry), 12, C_REAR_EDGE, -1, cv2.LINE_AA)
            cv2.circle(out, (rx, ry), 14, C_WHITE, 2, cv2.LINE_AA)
            cv2.putText(out, f"rear x={rx}", (rx+16, ry+6),
                        FONT, 0.40, C_REAR_EDGE, 1, cv2.LINE_AA)

    # --- 5. Hip_mid point (blue dot) ---
    if hip_pt is not None:
        hx, hy = int(hip_pt[0]), int(hip_pt[1])
        cv2.circle(out, (hx, hy), 7, C_HIP_MID, -1, cv2.LINE_AA)
        cv2.putText(out, "hip_mid", (hx+8, hy-4), FONT, 0.38, C_HIP_MID, 1)

    # --- 6. Info banner ---
    disp_str = f"  hip_rear={hip_rear_pct:+.1f}%" if hip_rear_pct is not None else ""
    banner = (f"{stem}  fr{fr_idx:04d}  phase={phase}  "
              f"ball={ball_side}  mq={mask_quality:.3f}{disp_str}")
    (tw, th), _ = cv2.getTextSize(banner, FONT, 0.55, 1)
    cv2.rectangle(out, (0, h-th-16), (tw+16, h), C_BLACK, -1)
    cv2.putText(out, banner, (8, h-8), FONT, 0.55, C_WHITE, 1, cv2.LINE_AA)

    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building SAM2 predictor on {device}...")
    model = build_sam2(SAM2_CFG, SAM2_CKPT, device=device)
    predictor = SAM2ImagePredictor(model)
    print("SAM2 ready.")

    saved = []
    quality_report = {}  # stem -> {fr: quality}

    for stem, peak_fr, kp_path, video_path in VIDEOS:
        print(f"\n{'='*50}")
        print(f"{stem}  peak=fr{peak_fr}")

        with open(kp_path) as f:
            kp_json = json.load(f)
        pipeline = PosePipeline(device="cpu")
        measurements, fps = pipeline.run_from_json(kp_json)
        n = len(measurements)

        engine = SwingPhaseEngine()
        annotations, anchors = engine.run(measurements, fps, angle="down-the-line")
        phase_labels = [a.phase for a in annotations]
        phase_map    = {a.frame_idx: a.phase for a in annotations}
        addr_fr = anchors.address

        # Orientation
        resolver = OrientationResolver()
        ori = resolver.resolve(
            measurements=measurements, angle="down-the-line",
            address_frame=addr_fr, top_frame=anchors.top,
            impact_frame=anchors.impact,
        )
        ball_side = ori.ball_side if ori.ball_side else "right"
        rear_is_left = (ball_side == "right")

        # Address frame biomechanics from kp_json
        fd_addr = kp_json["frames"][addr_fr]
        kps_addr = fd_addr["persons"][0]["keypoints"] if fd_addr["persons"] else {}
        addr_hip_pt = hip_mid_from_kp(kps_addr)
        torso_h = torso_h_from_kp(kps_addr)
        if torso_h <= 0: torso_h = measurements[addr_fr].torso_height() or 200.0

        band_cy   = addr_hip_pt[1] if addr_hip_pt else 400.0
        band_y_lo = band_cy - BAND_FRAC * torso_h
        band_y_hi = band_cy + BAND_FRAC * torso_h

        # Run SAM2 on address frame to get addr_rear_x
        addr_frame_bgr = get_frame(str(video_path), addr_fr)
        addr_mask, addr_mq = run_sam2_frame(predictor, addr_frame_bgr,
                                            addr_hip_pt if addr_hip_pt else (360, 400))
        addr_rear_x = find_rear_edge(addr_mask, band_y_lo, band_y_hi, rear_is_left)
        print(f"  addr=fr{addr_fr}  ball_side={ball_side}  torso_h={torso_h:.0f}  "
              f"band=[{band_y_lo:.0f},{band_y_hi:.0f}]  addr_rear_x={addr_rear_x}")

        quality_report[stem] = {}

        for label, fr_idx in [("addr", addr_fr), ("peak", peak_fr)]:
            if fr_idx >= n:
                print(f"  {label} fr{fr_idx} out of range (n={n}), skip")
                continue

            frame_bgr = (addr_frame_bgr if fr_idx == addr_fr
                         else get_frame(str(video_path), fr_idx))

            fd = kp_json["frames"][fr_idx]
            kps = fd["persons"][0]["keypoints"] if fd["persons"] else {}
            hip_pt = hip_mid_from_kp(kps)
            if hip_pt is None:
                hip_pt = addr_hip_pt  # fallback to address hip

            mask, mq = run_sam2_frame(predictor, frame_bgr, hip_pt)
            quality_report[stem][fr_idx] = round(mq, 4)
            rear_x = find_rear_edge(mask, band_y_lo, band_y_hi, rear_is_left)

            # hip_rear_pct for display
            hip_rear_pct = None
            if rear_x is not None and addr_rear_x is not None:
                toward_sign = 1 if ball_side == "right" else -1
                hip_rear_pct = (rear_x - addr_rear_x) * toward_sign / max(torso_h, 1) * 100

            phase = phase_map.get(fr_idx, "?")
            rendered = render_frame(
                frame_bgr, mask, hip_pt, torso_h, band_y_lo, band_y_hi,
                rear_x, addr_rear_x, ball_side, fr_idx, stem, phase,
                mq, hip_rear_pct
            )

            fname = f"{stem}_{label}_fr{fr_idx:04d}_mq{mq:.3f}.jpg"
            out_path = DEST / fname
            cv2.imwrite(str(out_path), rendered, [cv2.IMWRITE_JPEG_QUALITY, 93])
            saved.append(out_path)
            print(f"  [{label}] fr{fr_idx}  rear_x={rear_x}  hip_rear={hip_rear_pct}  "
                  f"mq={mq:.3f}  → {fname}")

    # Print results
    print("\n" + "="*70)
    print("mask_quality per frame:")
    for stem, qd in quality_report.items():
        for fr, mq in sorted(qd.items()):
            print(f"  {stem:15s} fr{fr:4d}  mq={mq:.4f}")

    print("\n=== Windows paths ===")
    for p in saved:
        win = str(p).replace("/mnt/c/", "C:\\").replace("/", "\\")
        print(f"  {win}")

    return quality_report, saved


if __name__ == "__main__":
    quality_report, saved = main()
    print(f"\nTotal: {len(saved)} files saved to {DEST}")
