/**
 * analyze/[id]/route.ts
 *
 * 主分析 API。执行顺序：
 * 1. 获取视频 signed URL
 * 2. 调用 Python 分析服务（真实 MediaPipe 数据）
 * 3. 如果 Python 服务不可用，降级到 stub 数据
 * 4. 应用 Golf Intelligence Layer 确定主问题
 * 5. 生成 overlay timeline
 * 6. 写入 Supabase swing_analysis
 */

import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@/lib/supabase/server';
import { generateOverlayTimeline } from '@/lib/overlay/templates';
import { getIssueResult, ISSUE_PRIORITY, ISSUE_DEFINITIONS } from '@/lib/golf/issues';
import type {
  MainIssueType, PhaseMarkers, VideoMetadata,
  KeypointTimeline, KeypointFrame
} from '@/types/analysis';

const PYTHON_ANALYZER_URL = process.env.PYTHON_ANALYZER_URL ?? '';

/* ── Stub data fallback ── */
const STUBS: Array<{
  issue_type: MainIssueType;
  duration_estimate: number;
}> = [
  { issue_type: 'early_extension',  duration_estimate: 2.8 },
  { issue_type: 'steep_downswing',  duration_estimate: 3.1 },
  { issue_type: 'head_movement',    duration_estimate: 2.5 },
  { issue_type: 'weight_shift_issue', duration_estimate: 3.0 },
  { issue_type: 'hand_path_issue',  duration_estimate: 2.7 },
];

function pickStub(videoId: string) {
  let hash = 0;
  for (const ch of videoId.replace(/-/g, '')) hash = (hash * 31 + parseInt(ch, 16)) >>> 0;
  return STUBS[hash % STUBS.length];
}

function proportionalPhases(dur: number): PhaseMarkers {
  return {
    setupTime:      round3(dur * 0.02),
    topTime:        round3(dur * 0.50),
    transitionTime: round3(dur * 0.62),
    impactTime:     round3(dur * 0.75),
    finishTime:     round3(dur * 0.90),
  };
}

function round3(n: number) { return Math.round(n * 1000) / 1000; }

/* ── Call Python analysis service ── */
async function callPythonAnalyzer(videoUrl: string): Promise<{
  videoMetadata: VideoMetadata;
  phaseMarkers: PhaseMarkers;
  keypointTimeline: KeypointFrame[];
  detectedIssue: MainIssueType;
  confidence: number;
} | null> {
  if (!PYTHON_ANALYZER_URL) {
    console.log('[analyze] PYTHON_ANALYZER_URL not set — using stub');
    return null;
  }

  try {
    const res = await fetch(`${PYTHON_ANALYZER_URL}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        video_url: videoUrl,
        view_type: 'face_on',
        sample_fps: 4.0,
      }),
      signal: AbortSignal.timeout(30_000), // 30 sec timeout
    });

    if (!res.ok) {
      console.error(`[analyze] Python service returned ${res.status}`);
      return null;
    }

    const data = await res.json();
    if (data.status !== 'success') {
      console.error('[analyze] Python service error:', data.error);
      return null;
    }

    // Map Python response to our TypeScript types
    const videoMeta: VideoMetadata = {
      durationSec: data.videoMetadata.durationSec,
      fps:         data.videoMetadata.fps,
      width:       data.videoMetadata.width,
      height:      data.videoMetadata.height,
    };

    const phases: PhaseMarkers = {
      setupTime:      data.phaseMarkers.setupTime,
      topTime:        data.phaseMarkers.topTime,
      transitionTime: data.phaseMarkers.transitionTime,
      impactTime:     data.phaseMarkers.impactTime,
      finishTime:     data.phaseMarkers.finishTime,
    };

    // Determine best issue: use Python's detection if confidence > 0.5,
    // otherwise fall back to ranking system
    let detectedIssue: MainIssueType = data.issueDetection?.issue ?? 'early_extension';
    let confidence: number = data.issueDetection?.confidence ?? 0.5;

    // Validate issue type
    if (!ISSUE_DEFINITIONS[detectedIssue as MainIssueType]) {
      detectedIssue = 'early_extension';
      confidence = 0.4;
    }

    return {
      videoMetadata:    videoMeta,
      phaseMarkers:     phases,
      keypointTimeline: data.keypointTimeline ?? [],
      detectedIssue:    detectedIssue as MainIssueType,
      confidence,
    };

  } catch (err) {
    console.error('[analyze] Python service call failed:', err);
    return null;
  }
}

/* ── Main API handler ── */
export async function POST(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id: videoId } = await params;
  const supabase = await createClient();

  // Auth
  const { data: { user }, error: authErr } = await supabase.auth.getUser();
  if (authErr || !user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  // Get video record
  const { data: video } = await supabase
    .from('swing_videos')
    .select('id, status, storage_path, user_id')
    .eq('id', videoId)
    .eq('user_id', user.id)
    .single();

  if (!video) return NextResponse.json({ error: 'Not found' }, { status: 404 });

  // Already done
  if (video.status === 'completed') {
    const { data: existing } = await supabase
      .from('swing_analysis').select('*').eq('video_id', videoId).single();
    return NextResponse.json({ status: 'completed', analysis: existing });
  }

  if (video.status === 'processing') {
    return NextResponse.json({ status: 'processing' }, { status: 202 });
  }

  // Mark processing
  await supabase.from('swing_videos').update({
    status: 'processing',
    processing_started_at: new Date().toISOString(),
  }).eq('id', videoId);

  let videoMetadata: VideoMetadata;
  let phaseMarkers: PhaseMarkers;
  let keypointTimeline: KeypointFrame[] = [];
  let detectedIssue: MainIssueType;
  let issueConfidence: number;
  let dataSource = 'stub';

  /* ── Try Python service ── */
  if (PYTHON_ANALYZER_URL && video.storage_path) {
    // Get a signed URL for the Python service to download
    const { data: signedData } = await supabase.storage
      .from('swing-videos')
      .createSignedUrl(video.storage_path, 300); // 5 min expiry for analysis

    if (signedData?.signedUrl) {
      const pythonResult = await callPythonAnalyzer(signedData.signedUrl);

      if (pythonResult) {
        videoMetadata    = pythonResult.videoMetadata;
        phaseMarkers     = pythonResult.phaseMarkers;
        keypointTimeline = pythonResult.keypointTimeline;
        detectedIssue    = pythonResult.detectedIssue;
        issueConfidence  = pythonResult.confidence;
        dataSource       = 'mediapipe';
        console.log(`[analyze] Using real MediaPipe data: ${detectedIssue} (conf=${issueConfidence.toFixed(2)})`);
      }
    }
  }

  /* ── Fallback to stub if Python unavailable ── */
  if (dataSource === 'stub') {
    const stub = pickStub(videoId);
    const dur  = stub.duration_estimate;
    videoMetadata = { durationSec: dur, fps: 30, width: 640, height: 360 };
    phaseMarkers  = proportionalPhases(dur);
    detectedIssue = stub.issue_type;
    issueConfidence = 0.5;

    // Simulate processing time only for stub
    await new Promise(r => setTimeout(r, 3000));
    console.log(`[analyze] Stub data used: ${detectedIssue}`);
  }

  /* ── Golf Intelligence Layer ── */
  const issueResult = getIssueResult(detectedIssue!, issueConfidence!);

  /* ── Generate overlay timeline ── */
  const overlayTimeline = generateOverlayTimeline({
    phaseMarkers:   phaseMarkers!,
    videoMetadata:  videoMetadata!,
    issue:          detectedIssue!,
    keypointTimeline: keypointTimeline.length > 0
      ? ({ frames: keypointTimeline } as KeypointTimeline)
      : undefined,
  });

  /* ── Write to Supabase ── */
  try {
    const { data: analysis, error: insertErr } = await supabase
      .from('swing_analysis')
      .insert({
        video_id:    videoId,
        issue_type:  issueResult.issue,
        summary_text: issueResult.summary,
        cue_text:    issueResult.cue,
        drill_text:  issueResult.drill,
        score:       Math.round(50 + issueResult.confidence * 30 + (1 - ISSUE_PRIORITY[issueResult.issue]) * 20),
        severity:    issueResult.severity,
        overlay_timeline_json: overlayTimeline,
        phase_markers_json:   phaseMarkers!,
        video_metadata_json:  { ...videoMetadata!, dataSource },
        // Store keypoint timeline if we have real data
        ...(keypointTimeline.length > 0 ? {
          keypoint_timeline_json: { frames: keypointTimeline },
        } : {}),
      })
      .select()
      .single();

    if (insertErr) throw new Error(insertErr.message);

    await supabase.from('swing_videos').update({
      status: 'completed',
      processing_completed_at: new Date().toISOString(),
    }).eq('id', videoId);

    return NextResponse.json({ status: 'completed', analysis, dataSource });

  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    await supabase.from('swing_videos').update({
      status: 'failed',
      error_code: 'ANALYSIS_ERROR',
      error_message: msg,
      processing_completed_at: new Date().toISOString(),
    }).eq('id', videoId);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
