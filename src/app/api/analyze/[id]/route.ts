import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@/lib/supabase/server';
import { generateOverlayTimeline } from '@/lib/overlay/templates';
import type { MainIssueType, PhaseMarkers, VideoMetadata } from '@/types/analysis';

interface StubData {
  issue_type: MainIssueType;
  cue_text: string;
  drill_text: string;
  score: number;
  severity: 'low' | 'medium' | 'high';
  duration_estimate: number;  // seconds
}

const STUBS: StubData[] = [
  {
    issue_type: 'early_extension',
    cue_text: 'Stay in your posture through impact.',
    drill_text: 'Wall Drill: backside against wall, swing without losing contact.',
    score: 72, severity: 'medium', duration_estimate: 2.8,
  },
  {
    issue_type: 'steep_downswing',
    cue_text: 'Drop your trail elbow into your hip.',
    drill_text: 'Headcover Drill: keep it tucked through impact.',
    score: 65, severity: 'medium', duration_estimate: 3.1,
  },
  {
    issue_type: 'head_movement',
    cue_text: "Head stays still — pinned to the wall behind you.",
    drill_text: "Friend holds your cap during backswing. 15 slow reps.",
    score: 78, severity: 'low', duration_estimate: 2.5,
  },
  {
    issue_type: 'weight_shift_issue',
    cue_text: 'Lead knee drives toward the target.',
    drill_text: 'Step Drill: step trail foot toward lead on downswing.',
    score: 68, severity: 'high', duration_estimate: 3.0,
  },
  {
    issue_type: 'hand_path_issue',
    cue_text: 'Keep hands close to the body in transition.',
    drill_text: 'Towel Drill: pinch towel between trail arm and chest.',
    score: 70, severity: 'medium', duration_estimate: 2.7,
  },
];

function pickStub(videoId: string): StubData {
  let hash = 0;
  for (const ch of videoId.replace(/-/g, '')) hash = (hash * 31 + parseInt(ch, 16)) >>> 0;
  return STUBS[hash % STUBS.length];
}

export async function POST(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id: videoId } = await params;
  const supabase = await createClient();

  const { data: { user }, error: authErr } = await supabase.auth.getUser();
  if (authErr || !user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const { data: video } = await supabase
    .from('swing_videos').select('id, status').eq('id', videoId).eq('user_id', user.id).single();
  if (!video) return NextResponse.json({ error: 'Not found' }, { status: 404 });

  if (video.status === 'completed') {
    const { data: existing } = await supabase.from('swing_analysis').select('*').eq('video_id', videoId).single();
    return NextResponse.json({ status: 'completed', analysis: existing });
  }
  if (video.status === 'processing') return NextResponse.json({ status: 'processing' }, { status: 202 });

  // Mark processing
  await supabase.from('swing_videos').update({
    status: 'processing',
    processing_started_at: new Date().toISOString(),
  }).eq('id', videoId);

  // Simulate analysis delay
  await new Promise(r => setTimeout(r, 3500));

  const stub = pickStub(videoId);
  const dur = stub.duration_estimate;

  // Phase markers in seconds
  const phaseMarkers: PhaseMarkers = {
    setupTime:      0,
    topTime:        dur * 0.50,
    transitionTime: dur * 0.62,
    impactTime:     dur * 0.75,
    finishTime:     dur * 0.92,
  };

  // Video metadata
  const videoMetadata: VideoMetadata = {
    durationSec: dur, fps: 30, width: 640, height: 360,
  };

  // Generate structured overlay timeline
  const overlayTimeline = generateOverlayTimeline({
    phaseMarkers,
    videoMetadata,
    issue: stub.issue_type,
  });

  try {
    const { data: analysis, error: insertErr } = await supabase
      .from('swing_analysis')
      .insert({
        video_id:            videoId,
        issue_type:          stub.issue_type,
        summary_text:        `Your main issue is: ${stub.issue_type.replace(/_/g, ' ')}.`,
        cue_text:            stub.cue_text,
        drill_text:          stub.drill_text,
        score:               stub.score,
        severity:            stub.severity,
        overlay_timeline_json: overlayTimeline,
        phase_markers_json:    phaseMarkers,
        video_metadata_json:   videoMetadata,
      })
      .select()
      .single();

    if (insertErr) throw new Error(insertErr.message);

    await supabase.from('swing_videos').update({
      status: 'completed',
      processing_completed_at: new Date().toISOString(),
    }).eq('id', videoId);

    return NextResponse.json({ status: 'completed', analysis });
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    await supabase.from('swing_videos').update({
      status: 'failed', error_code: 'ANALYSIS_ERROR', error_message: msg,
      processing_completed_at: new Date().toISOString(),
    }).eq('id', videoId);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
