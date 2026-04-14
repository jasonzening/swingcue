"""
main.py — SwingCue Python 分析服务

FastAPI service that:
1. Receives video URL
2. Downloads video
3. Runs MediaPipe Pose
4. Detects swing phases
5. Returns structured analysis data
"""

import os
import logging
import tempfile
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

from analyzer import download_video, analyze_video, VideoMetadata
from phase_detector import detect_phases, score_issue_from_keypoints

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SwingCue Analysis Service",
    description="Real-time golf swing analysis using MediaPipe Pose",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Vercel and any frontend
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    video_url: str              # Supabase Storage signed URL
    view_type: str = "face_on"
    club_type: str = "unknown"
    sample_fps: float = 4.0    # frames per second to sample


class AnalyzeResponse(BaseModel):
    status: str
    videoMetadata: dict
    phaseMarkers: dict
    keypointTimeline: list      # List of KeypointFrame dicts
    issueDetection: dict        # Detected issue + confidence
    error: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "service": "swingcue-analyzer"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """
    Main analysis endpoint.
    Downloads video, runs MediaPipe, returns structured data.
    """
    tmp_path = None
    try:
        # 1. Download video
        logger.info(f"Starting analysis for video: {req.video_url[:60]}...")
        tmp_path = download_video(req.video_url)

        # 2. Run MediaPipe analysis
        metadata, keypoint_frames = analyze_video(tmp_path, sample_fps=req.sample_fps)

        # 3. Detect phases
        phases = detect_phases(keypoint_frames, metadata.durationSec)

        # 4. Issue scoring from keypoints
        issue_result = score_issue_from_keypoints(keypoint_frames)

        # 5. Serialize keypoint frames
        kp_timeline = [f.to_dict() for f in keypoint_frames]

        logger.info(
            f"Analysis complete: {len(kp_timeline)} frames, "
            f"duration={metadata.durationSec:.2f}s, "
            f"issue={issue_result['issue']} (conf={issue_result['confidence']:.2f})"
        )

        return AnalyzeResponse(
            status="success",
            videoMetadata={
                "durationSec": metadata.durationSec,
                "fps":         metadata.fps,
                "width":       metadata.width,
                "height":      metadata.height,
            },
            phaseMarkers=phases,
            keypointTimeline=kp_timeline,
            issueDetection=issue_result,
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
