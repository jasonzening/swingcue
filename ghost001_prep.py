"""
ghost001_prep.py  — GHOST-001 T1 素材准备

1. 提取 fo-ok-1 address 帧 (fr0) → address_ref.jpg
2. 切割 P1(fr0)→P4(fr107) 降采样到 ≤72 帧 → drive_segment.mp4
3. 打印相位关键帧信息
"""
import cv2, json, pathlib, math, sys, shutil
import numpy as np

ROOT   = pathlib.Path('/home/jason/projects/swingcue-postest')
VIDEO  = pathlib.Path('/mnt/c/Users/jason/Zening/Swingcue/Video/fo-ok-1.mp4')
OUT    = ROOT / 'output/ghost001'
OUT.mkdir(parents=True, exist_ok=True)

# ── Phase boundaries (from B-layer 8-phase system) ────────────────────────────
# B层正式相位: address/takeaway/backswing/top/transition/downswing/impact/follow_through
# fo-ok-1: 112 frames, 30fps
ADDR_FR     = 0    # address
TOP_FR      = 97   # top (RTMPose wrist_y min, Jason目视确认 fr185→normalized fr97)
IMPACT_FR   = 104  # impact (approximate, wrist returning near hip)
FOLLOW_FR   = 107  # follow_through (last stable frame)

DRIVE_START = ADDR_FR
DRIVE_END   = FOLLOW_FR   # inclusive
TARGET_FRAMES = 72

total_span = DRIVE_END - DRIVE_START + 1  # 108 frames
print(f"Drive span: fr{DRIVE_START}(address)→fr{DRIVE_END}(follow_through) = {total_span} frames")

# downsample to TARGET_FRAMES
indices = [int(DRIVE_START + i * (total_span - 1) / (TARGET_FRAMES - 1))
           for i in range(TARGET_FRAMES)]
indices = sorted(set(indices))
print(f"After downsample: {len(indices)} frames, step~={total_span/TARGET_FRAMES:.2f}")

# Map original frame indices to B-layer phase labels
phase_map = {ADDR_FR: 'address', TOP_FR: 'top', IMPACT_FR: 'impact', FOLLOW_FR: 'follow_through'}

cap = cv2.VideoCapture(str(VIDEO))
orig_fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Source: {W}x{H} @ {orig_fps:.1f}fps")

# ── 1. Extract address frame (P1, fr0) as reference image ─────────────────────
cap.set(cv2.CAP_PROP_POS_FRAMES, P1_FR)
ok, addr_frame = cap.read()
assert ok, f"Cannot read fr{P1_FR}"
ref_path = OUT / 'address_ref.jpg'
cv2.imwrite(str(ref_path), addr_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"Ref image → {ref_path}  ({W}x{H})")

# ── 2. Write drive segment (downsampled to ≤72 frames) ────────────────────────
drive_fps = 15.0   # MimicMotion output target fps
drive_path = OUT / 'drive_segment.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_vw = cv2.VideoWriter(str(drive_path), fourcc, drive_fps, (W, H))

phase_sampled = {}   # downsampled idx → original fr

for ds_idx, orig_fr in enumerate(indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES, orig_fr)
    ok, fr = cap.read()
    if not ok:
        print(f"  [warn] cannot read fr{orig_fr}")
        continue
    out_vw.write(fr)
    # find closest phase label
    for ph_fr, ph_name in phase_map.items():
        if abs(orig_fr - ph_fr) <= max(1, int(total_span / TARGET_FRAMES)):
            phase_sampled[ds_idx] = (ph_name, orig_fr)

out_vw.release()
cap.release()
print(f"Drive segment → {drive_path}  ({len(indices)} frames @ {drive_fps}fps)")

# ── 3. Phase frame info ───────────────────────────────────────────────────────
print(f"\nPhase key frames (original → downsampled idx):")
for ph_fr, ph_name in sorted(phase_map.items()):
    # find closest downsampled idx
    ds_idx = min(range(len(indices)), key=lambda i: abs(indices[i] - ph_fr))
    orig_fr_actual = indices[ds_idx]
    print(f"  {ph_name}: orig fr{ph_fr} → ds_idx={ds_idx} (actual orig fr{orig_fr_actual})")

# ── 4. Copy ref to Windows Desktop for inspection ─────────────────────────────
desk = pathlib.Path('/mnt/c/Users/jason/Desktop/rtmpose_results/preview/ghost001')
desk.mkdir(parents=True, exist_ok=True)
shutil.copy2(ref_path, desk / ref_path.name)
shutil.copy2(drive_path, desk / drive_path.name)
print(f"\nCopied to Windows: {desk}")
print("Done ✓")
