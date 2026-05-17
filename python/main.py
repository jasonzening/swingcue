"""
main.py — SwingCue 分析服务 (FastAPI)

启动策略：
- 服务立即启动，不预加载任何 ML 模型
- /health 端点立即可用（不依赖 MediaPipe）
- MediaPipe 只在真正分析视频时才懒加载
"""

# PR-3 hotfix v4: log runtime package versions BEFORE any heavy imports.
# Helps diagnose any future numpy/cv2/mediapipe ABI mismatch by surfacing
# the exact versions the live container sees at startup, not at build time.
# Uses a subprocess to get a fresh interpreter so module caching cannot mask
# a broken binary ABI.
import subprocess
import sys

try:
    _v = subprocess.check_output(
        [
            sys.executable, "-c",
            "import numpy, cv2, mediapipe; "
            "print(f'numpy={numpy.__version__} cv2={cv2.__version__} mediapipe={mediapipe.__version__}')",
        ],
        stderr=subprocess.STDOUT,
    ).decode().strip()
    print(f"[startup-verify] {_v}", flush=True)
except subprocess.CalledProcessError as _e:
    # CalledProcessError.__str__ doesn't show captured output — fetch it
    # explicitly so we see the real Python traceback from the subprocess.
    _out = _e.output.decode(errors="replace") if _e.output else "(no captured output)"
    print(f"[startup-verify-FAIL] exit={_e.returncode}", flush=True)
    print(f"[startup-verify-FAIL] subprocess stdout+stderr:\n{_out}", flush=True)
except Exception as _e:
    print(f"[startup-verify-FAIL] {_e!r}", flush=True)

# Also dump runtime package versions via pip freeze so we know what's
# actually installed in the live container.
try:
    _f = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"],
        stderr=subprocess.STDOUT,
    ).decode()
    _relevant = [
        line.strip() for line in _f.split("\n")
        if any(p in line.lower() for p in ["numpy", "opencv", "mediapipe", "ultralytics", "torch"])
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
    sample_fps: float = 4.0
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
        metadata, keypoint_frames = analyze_video(tmp_path, sample_fps=req.sample_fps)

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

        # 5. Issue detection
        issue_result = score_issue_from_keypoints(keypoint_frames)

        # 6. Serialize
        kp_timeline = [f.to_dict() for f in keypoint_frames]

        logger.info(
            f"Done: {len(kp_timeline)} frames, "
            f"{metadata.durationSec:.2f}s, "
            f"issue={issue_result['issue']}, "
            f"pose3d={pose3d_summary}, "
            f"yolo={yolo_summary}"
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
