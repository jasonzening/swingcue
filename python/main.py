"""
main.py — SwingCue 分析服务 (FastAPI)

启动策略：
- 服务立即启动，不预加载任何 ML 模型
- /health 端点立即可用（不依赖 MediaPipe）
- MediaPipe 只在真正分析视频时才懒加载
"""

import os
import logging
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

        # 4. Issue detection
        issue_result = score_issue_from_keypoints(keypoint_frames)

        # 5. Serialize
        kp_timeline = [f.to_dict() for f in keypoint_frames]

        logger.info(
            f"Done: {len(kp_timeline)} frames, "
            f"{metadata.durationSec:.2f}s, "
            f"issue={issue_result['issue']}"
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
