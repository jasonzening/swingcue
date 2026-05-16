# Pose Model License Decision

**Status:** Active (MVP)
**Date:** 2026-05-16
**Applies to:** `python/yolo/` package (PR-3 onward)

## Decision

Use **Ultralytics YOLO11m-pose** for production pose inference during the
SwingCue MVP, accepting that the model and its weights are licensed under
**AGPL-3.0**. Replace before any commercial / monetised deployment.

## Why this matters

Ultralytics YOLOv8+ (including YOLO11) is AGPL-3.0. The licence covers:

- The Python `ultralytics` package source.
- The pre-trained weight files (e.g. `yolo11m-pose.pt`).

AGPL is **stronger** than GPL — it covers SaaS use. If SwingCue exposes a
network endpoint (`/analyze`) whose response depends on AGPL-licensed
inference, AGPL-3.0 §13 obliges us to offer the **complete corresponding
source** of the SaaS itself to any user who requests it. For a closed-source
commercial product this is a non-starter.

During MVP this is acceptable because:

1. The service is invitation-only / pre-revenue.
2. No third-party user has yet exercised AGPL §13 rights.
3. Speed-to-validation on the SAM-vs-YOLO disc accuracy comparison is
   strictly more valuable than licence cleanliness right now.

## Replacement path (priority order)

| Option | License | Notes |
|---|---|---|
| 1. **RTMPose** (MMPose family) | Apache-2.0 | Comparable or better COCO AP than YOLO11m. Install pipeline is heavier (`mmcv`, `mmengine`). 17-kp COCO output is drop-in identical. Best accuracy-per-license. |
| 2. **YOLO-NAS-Pose** (Deci AI) | Apache-2.0 | Architecture search by Deci; native Apache. Cleanest swap from a code-shape standpoint (`super_gradients` package). Slightly lower COCO AP than YOLO11m. |
| 3. **MoveNet Thunder** (TF Hub) | Apache-2.0 | CPU-fastest of the three; accuracy noticeably below RTMPose / YOLO-NAS on small / occluded subjects. Acceptable for full-body golf swing. |

Migration scope estimate (any of the three): ~6 commits, parallel to PR-3 —
swap `python/yolo/inference.py` model loader + tweak keypoint indexing if
the chosen model uses non-COCO output. Database schema and frontend
unaffected (still COCO 17-kp).

## Trigger conditions

Replace **before** the first of:

- First externally-billed user is invoiced.
- Public marketing of the analysis pipeline as a paid feature.
- Legal review for fundraising / acquisition due diligence.
- Receipt of an AGPL §13 request from any user of the service.

## Operational consequence today

- Ultralytics is a runtime dependency in `python/Dockerfile`.
- `yolo11m-pose.pt` is baked into the production image.
- No external user has been granted access to the source tree of the
  inference service.

When the replacement happens, this ADR moves to status `Superseded` and a
new ADR records the chosen replacement model and any accuracy delta
observed on the SwingCue test set.
