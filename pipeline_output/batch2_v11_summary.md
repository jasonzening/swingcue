# Batch2 Gate v1.1 + Full Pipeline Results

**Date**: 2026-06-11  
**GT iron rule**: no fault labels. ok/wrong are filename tokens only.

## Gate v1.1 — Address-Only VLM + 3-Vote

Fix: v1.0 used full-video frames including follow-through, causing VLM to
misclassify DTL as face-on (body turns away from camera at follow-through).
v1.1 uses only the low-motion address-region frames (3 frames from first 20%
of video). All 7 previously-blocked videos now PASS.

| Stem | Prev v1.0 | v1.1 Verdict | Angle |
|---|---|---|---|
| dtl-ok-1    | needs_human | **PASS** | DTL |
| dtl-ok-2    | needs_human | **PASS** | DTL |
| dtl-wrong-1 | needs_human | **PASS** | DTL |
| dtl-wrong-2 | needs_human | **PASS** | DTL |
| dtl-wrong-3 | needs_human | **PASS** | DTL |
| fo-wrong-3  | needs_human | **PASS** | face-on |
| fo-wrong-4  | needs_human | **PASS** | face-on |

## DTL Pipeline Results

| Stem | sc | addr | top | impact | ic | hip_win(fr/%) | spine_win(fr/°) | Diagnosis (verbatim) |
|---|---|---|---|---|---|---|---|---|
| dtl-ok-1 | 1 | 28 | 62 | 82 | 0.943 | fr76/17.6% | fr65/-7.72° | none/none |
| dtl-ok-2 | 2 | 60 | 90 | 105 | 0.839 | fr103/16.3% | fr94/-10.89° | none/none |
| dtl-wrong-1 | 1 | 46 | 74 | 93 | 0.929 | fr81/16.9% | fr78/-6.54° | none/none |
| dtl-wrong-2 | 1 | 47 | 82 | 98 | 0.914 | fr92/28.4% | fr85/-12.93° | none/none |
| dtl-wrong-3 | 1 | 8 | 21 | 388 | 0.929 | fr375/17.4% | fr375/-11.67° | none/none |

## DTL Group Distribution (no labels — ok/wrong are filenames)

**dtl-ok group** (n=2)
  hip_win_pct: +17.6%, +16.3%
  → min=+16.3%  max=+17.6%  mean=+17.0%
  sp_win_deg: -7.72°, -10.89°
  → min=-10.89°  max=-7.72°  mean=-9.30°

**dtl-wrong group** (n=3)
  hip_win_pct: +16.9%, +28.4%, +17.4%
  → min=+16.9%  max=+28.4%  mean=+20.9%
  sp_win_deg: -6.54°, -12.93°, -11.67°
  → min=-12.93°  max=-6.54°  mean=-10.38°

## Face-On Measurements (fo-wrong-3, fo-wrong-4)

| Stem | addr | top | impact | head_lat+(fr/%) | head_lat-(fr/%) | head_vert(fr/%) | elbow_min(CW/fr/deg) |
|---|---|---|---|---|---|---|---|
| fo-wrong-3 | 85 | 130 | 145 | fr85/+0.0% | fr140/-76.7% | fr140/+90.3% | fr145-fr147/fr147/162.5deg |
| fo-wrong-4 | 85 | 126 | 147 | fr87/+0.1% | fr131/-36.6% | fr129/+13.8% | fr147-fr150/fr149/155.7deg |

## fo-ok-2 Anomaly Report

fr75 is a confirmed single-frame keypoint jump:
  head_x: 269.4 → 215.8 → 265.1 (jump then immediate return)
  head_y: 541.6 → 433.6 → 543.3 (jump then immediate return)
  dx shift: -44.7% torso_h in one frame (physically impossible)

Previous reported peaks (head_lat=-70.9%, head_vert=+72.5%) were fr75 outliers.
Corrected: head_lat_neg ~ -21.6% at fr80 (smooth trend); head_vert_peak negative (head moves down).
NEEDS_HUMAN.md updated: sentinel gap noted.