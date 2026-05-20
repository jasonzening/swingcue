"""
main.py — SwingCue 分析服务 (FastAPI)

启动策略：
- 服务立即启动，不预加载任何 ML 模型
- /health 端点立即可用（不依赖 MediaPipe）
- MediaPipe 只在真正分析视频时才懒加载
"""

# PR-3 Option C: runtime is back to numpy 1.x + opencv-python-headless
# 4.10.0.84 + mediapipe 0.10.14 (proven pre-PR-3 production state),
# plus onnxruntime. Startup-verify subprocess uses a fresh interpreter
# so module caching cannot mask a broken binary ABI. Surfaces the exact
# package versions in Railway runtime logs at every container boot.
import subprocess
import sys

try:
    _v = subprocess.check_output(
        [
            sys.executable, "-c",
            "import numpy, cv2, mediapipe, onnxruntime; "
            "assert numpy.__version__.startswith('1.'), "
            "f'expected numpy 1.x, got {numpy.__version__}'; "
            "print(f'numpy={numpy.__version__} cv2={cv2.__version__} "
            "mediapipe={mediapipe.__version__} "
            "onnxruntime={onnxruntime.__version__}')",
        ],
        stderr=subprocess.STDOUT,
    ).decode().strip()
    print(f"[startup-verify] {_v}", flush=True)
except subprocess.CalledProcessError as _e:
    _out = _e.output.decode(errors="replace") if _e.output else "(no captured output)"
    print(f"[startup-verify-FAIL] exit={_e.returncode}", flush=True)
    print(f"[startup-verify-FAIL] subprocess stdout+stderr:\n{_out}", flush=True)
except Exception as _e:
    print(f"[startup-verify-FAIL] {_e!r}", flush=True)

# pip freeze dump (relevant packages only) — proves what's installed in
# the live container, independent of any module's __version__ attribute.
try:
    _f = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"],
        stderr=subprocess.STDOUT,
    ).decode()
    _relevant = [
        line.strip() for line in _f.split("\n")
        if any(p in line.lower() for p in ["numpy", "opencv", "mediapipe", "onnxruntime", "torch", "ultralytics"])
    ]
    for line in _relevant:
        print(f"[runtime-pkg] {line}", flush=True)
except Exception as _e:
    print(f"[runtime-pkg-FAIL] {_e!r}", flush=True)

import asyncio
import logging
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SwingCue Analysis Service",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    video_url: str
    view_type: str = "face_on"
    club_type: str = "unknown"
    # PR-5.9 Task 1: sample_fps is now None-by-default. When unset, the
    # analyzer uses min(video_native_fps, 60) — see analyze_video. Any
    # explicit value here is still respected (clamped to [1, 60]) for
    # callers that want a fixed rate (e.g., testing harness).
    sample_fps: Optional[float] = None
    # PR-2B: required for pose_3d_phases service-role writes. The Next.js
    # API route enforces auth + ownership before calling us, so we trust
    # these values here.
    video_id: Optional[str] = None
    user_id: Optional[str] = None


@app.get("/health")
def health():
    """Health check — always returns immediately, no ML deps."""
    return {"status": "ok", "service": "swingcue-analyzer", "version": "1.0.0"}


@app.get("/")
def root():
    return {"service": "SwingCue Analyzer", "status": "running", "endpoints": ["/health", "/analyze"]}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """
    Analyze a golf swing video.
    Lazily imports MediaPipe only on first call.
    """
    tmp_path = None
    try:
        logger.info(f"Starting analysis: {req.video_url[:60]}...")

        # Lazy import — only loads when needed
        from analyzer import download_video, analyze_video
        from phase_detector import detect_phases, score_issue_from_keypoints

        # 1. Download
        tmp_path = download_video(req.video_url)

        # 2. Analyze
        # PR-5.9 Task 1: analyze_video now returns the effective sample_fps
        # it actually used (native fps capped at 60 by default; explicit
        # req.sample_fps clamped to [1, 60] when provided).
        metadata, keypoint_frames, raw_coco_frames, effective_sample_fps = analyze_video(
            tmp_path, sample_fps=req.sample_fps,
        )

        # 3. Phase detection
        phases = detect_phases(keypoint_frames, metadata.durationSec)

        # 4a. Frame extraction — once per phase, shared between SAM + YOLO.
        png_bytes_per_phase: dict[str, bytes] = {}
        phase_timestamps: dict[str, float] = {}
        if req.video_id and req.user_id:
            from sam3d.frame_extract import extract_frame
            phase_pairs = [
                ("setup", "setupTime"),
                ("top", "topTime"),
                ("transition", "transitionTime"),
                ("impact", "impactTime"),
                ("finish", "finishTime"),
            ]
            phase_timestamps = {
                name: float(phases.get(time_key) or 0.0)
                for name, time_key in phase_pairs
            }
            extracted = await asyncio.gather(
                *[extract_frame(tmp_path, ts) for ts in phase_timestamps.values()],
                return_exceptions=True,
            )
            for name, result in zip(phase_timestamps.keys(), extracted):
                if isinstance(result, BaseException):
                    logger.error(
                        f"[frame] phase {name} extract failed: {result!r}"
                    )
                    continue
                png_bytes_per_phase[name] = result
            logger.info(
                f"[frame] extracted {len(png_bytes_per_phase)}/5 phase frames"
            )

        # 4b. SAM 3D Body + YOLO11-pose — parallel fire on shared PNG bytes.
        # Both inference paths are non-fatal: MediaPipe response below is
        # still useful even if both fail.
        pose3d_summary = None
        yolo_summary = None
        if png_bytes_per_phase and req.video_id and req.user_id:
            from sam3d.orchestrator import pose3d_for_all_phases
            from yolo.orchestrator import yolo_for_all_phases
            sam_result, yolo_result = await asyncio.gather(
                pose3d_for_all_phases(
                    png_bytes_per_phase=png_bytes_per_phase,
                    phase_timestamps=phase_timestamps,
                    video_id=req.video_id,
                    user_id=req.user_id,
                    image_width=metadata.width,
                    image_height=metadata.height,
                    fps=metadata.fps,
                ),
                yolo_for_all_phases(
                    png_bytes_per_phase=png_bytes_per_phase,
                    phase_timestamps=phase_timestamps,
                    video_id=req.video_id,
                    user_id=req.user_id,
                    image_width=metadata.width,
                    image_height=metadata.height,
                    fps=metadata.fps,
                ),
                return_exceptions=True,
            )
            if isinstance(sam_result, BaseException):
                logger.error(f"pose3d step raised: {sam_result!r}", exc_info=sam_result)
                pose3d_summary = {"completed": 0, "failed": 5, "error": str(sam_result)}
            else:
                pose3d_summary = sam_result
            if isinstance(yolo_result, BaseException):
                logger.error(f"yolo step raised: {yolo_result!r}", exc_info=yolo_result)
                yolo_summary = {"completed": 0, "failed": 5, "error": str(yolo_result)}
            else:
                yolo_summary = yolo_result
        elif not req.video_id or not req.user_id:
            logger.warning(
                "video_id/user_id missing in request — skipping pose3d+yolo"
            )

        # 4c. PR-4: pose_timeline_2d pipeline.
        # Independent of pose3d / yolo paths above; runs on the
        # extractor's raw_coco_frames. YOLO 5-phase keypoints (if
        # available) are used as anchor corrections — but ONLY for the
        # MediaPipe path. RTMPose is already a top-down detector with
        # its own bbox-cropped pose head, so the MP→YOLO offset
        # correction designed for MediaPipe drift doesn't transfer and
        # may add jitter. PR-6.1_SPEC_v2 §10 Q6 + audit-stage decision.
        # Non-fatal on failure.
        # PR-6.1a: read POSE_RUNNER_OVERRIDE here too so we can label the
        # envelope correctly and skip YOLO anchor for the rtmpose path.
        # analyzer.py reads the same env at the same call boundary.
        _pose_runner = os.environ.get(
            "POSE_RUNNER_OVERRIDE", "mediapipe",
        ).strip().lower()
        _is_rtmpose = _pose_runner == "rtmpose"
        _keypoint_source = "rtmpose_v1" if _is_rtmpose else "mediapipe_pose_v1_5"
        pose_timeline_2d: Optional[dict] = None
        try:
            if raw_coco_frames:
                from pose_timeline import (
                    apply_yolo_anchor_correction,
                    bidirectional_ema,
                    build_timeline_from_raw_coco_frames,
                    detect_outliers_and_reject,
                    gap_fill_linear,
                    validate_timeline,
                )
                tl = build_timeline_from_raw_coco_frames(
                    raw_frames=raw_coco_frames,
                    video_width=metadata.width,
                    video_height=metadata.height,
                    sample_fps=effective_sample_fps,
                    keypoint_source=_keypoint_source,
                )
                # PR-5.9 Task 4: snapshot raw extract per frame BEFORE
                # outlier/smooth/gap mutate `keypoints`. The debug
                # overlay (?debug=pose, frontend commit 5) compares
                # raw_keypoints vs the post-pipeline `keypoints` to
                # visualize smoothing's effect. deepcopy is necessary —
                # outlier/smooth/gap mutate the inner lists in place.
                import copy as _copy
                for f in tl["frames"]:
                    f["raw_keypoints"] = _copy.deepcopy(f["keypoints"])
                tl = detect_outliers_and_reject(tl)
                # PR-5.9 Task 2: bidirectional (forward+backward) EMA —
                # zero phase delay vs the prior causal smooth_ema.
                tl = bidirectional_ema(tl, alpha=0.4)
                tl = gap_fill_linear(tl)
                if _is_rtmpose:
                    # PR-6.1a: skip YOLO anchor correction for the rtmpose
                    # path; see envelope comment above for rationale.
                    # PR-6.1c may re-enable after visual review.
                    logger.info(
                        "[pose_timeline] rtmpose path: skipping "
                        "apply_yolo_anchor_correction (PR-6.1a)"
                    )
                else:
                    # Collect YOLO keypoints per phase from the orchestrator
                    # summary (PR-4 patched yolo/orchestrator.py to include
                    # keypoints_2d on completed-status entries).
                    yolo_kps_per_phase: dict[str, list] = {}
                    if isinstance(yolo_summary, dict):
                        for r in yolo_summary.get("results", []) or []:
                            if (isinstance(r, dict)
                                    and r.get("status") == "completed"
                                    and r.get("keypoints_2d") is not None):
                                yolo_kps_per_phase[r["phase"]] = r["keypoints_2d"]
                    if yolo_kps_per_phase:
                        tl = apply_yolo_anchor_correction(
                            tl, yolo_kps_per_phase, phases,
                        )
                if validate_timeline(tl):
                    pose_timeline_2d = tl
                else:
                    logger.warning(
                        "[pose_timeline] validation failed; writing NULL"
                    )
        except Exception as e:
            logger.error(
                f"[pose_timeline] pipeline failed (non-fatal): {e}",
                exc_info=True,
            )

        # 5. Issue detection
        issue_result = score_issue_from_keypoints(keypoint_frames)

        # 6. Serialize
        kp_timeline = [f.to_dict() for f in keypoint_frames]

        logger.info(
            f"Done: {len(kp_timeline)} frames, "
            f"{metadata.durationSec:.2f}s, "
            f"issue={issue_result['issue']}, "
            f"pose3d={pose3d_summary}, "
            f"yolo={yolo_summary}, "
            f"pose_timeline_2d={'OK' if pose_timeline_2d else 'NULL'}"
        )

        return {
            "status": "success",
            "videoMetadata": {
                "durationSec": metadata.durationSec,
                "fps": metadata.fps,
                "width": metadata.width,
                "height": metadata.height,
            },
            "phaseMarkers": phases,
            "keypointTimeline": kp_timeline,
            "issueDetection": issue_result,
            "pose3dSummary": pose3d_summary,
            "yoloSummary": yolo_summary,
            "poseTimeline2d": pose_timeline_2d,
        }

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
