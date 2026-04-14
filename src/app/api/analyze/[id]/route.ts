import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@/lib/supabase/server';
import { generateOverlayTimeline } from '@/lib/timeline/generateOverlayTimeline';
import type { SwingIssueType, SwingSeverity } from '@/lib/types/swing';

interface StubData {
  issue_type: SwingIssueType;
  summary_text: string;
  cue_text: string;
  drill_text: string;
  score: number;
  severity: SwingSeverity;
}

const STUBS: StubData[] = [
  {
    issue_type: 'early_extension',
    summary_text: 'Your hips are firing early and pulling you off the ball before impact. This breaks your posture angle and causes thin shots and loss of distance.',
    cue_text: 'Stay in your posture through impact.',
    drill_text: 'Wall Drill: Stand with your backside lightly against a wall. Make slow practice swings without losing contact through impact. 5 reps before every session.',
    score: 72, severity: 'medium',
  },
  {
    issue_type: 'steep_downswing',
    summary_text: 'Your club is coming over the top on the downswing — too steep and outside-in. This causes pulls, weak fades, and chunks.',
    cue_text: 'Drop your trail elbow into your hip on the downswing.',
    drill_text: 'Headcover Drill: Tuck a headcover under your trail arm. Keep it there through impact without dropping it. 10 reps on the range.',
    score: 65, severity: 'medium',
  },
  {
    issue_type: 'head_movement',
    summary_text: 'Your head is drifting laterally during the backswing, making consistent ball contact very difficult.',
    cue_text: "Keep your head still — feel like it's pinned to the wall behind you.",
    drill_text: 'Statue Drill: Have a friend lightly hold your cap during your backswing. Repeat 15 times slowly. The resistance trains stillness into muscle memory.',
    score: 78, severity: 'low',
  },
  {
    issue_type: 'weight_shift_issue',
    summary_text: "Your weight is staying on your back foot through impact — a reverse pivot. It steepens your attack angle and kills your distance.",
    cue_text: 'Feel your lead knee drive toward the target to start your downswing.',
    drill_text: 'Step Drill: At the top of your backswing, step your trail foot toward your lead foot as you start down. 20 reps, slow and deliberate.',
    score: 68, severity: 'high',
  },
  {
    issue_type: 'hand_path_issue',
    summary_text: 'Your hands are looping away from your body on the downswing — a casting motion that robs you of lag and consistency.',
    cue_text: 'Keep your hands close to your body on the downswing.',
    drill_text: 'Towel Drill: Tuck a rolled towel between your upper right arm and chest. Make half-swings keeping it pinched. When it drops, you disconnected. 15 reps.',
    score: 70, severity: 'medium',
  },
];

function pickStub(videoId: string): StubData {
  let hash = 0;
  for (const ch of videoId.replace(/-/g, '')) {
    hash = (hash * 31 + parseInt(ch, 16)) >>> 0;
  }
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

  const { data: video, error: videoErr } = await supabase
    .from('swing_videos').select('id, status, user_id').eq('id', videoId).eq('user_id', user.id).single();
  if (videoErr || !video) return NextResponse.json({ error: 'Video not found' }, { status: 404 });

  if (video.status === 'completed') {
    const { data: existing } = await supabase.from('swing_analysis').select('*').eq('video_id', videoId).single();
    return NextResponse.json({ status: 'completed', analysis: existing });
  }
  if (video.status === 'processing') {
    return NextResponse.json({ status: 'processing' }, { status: 202 });
  }

  // Start processing
  await supabase.from('swing_videos').update({
    status: 'processing',
    processing_started_at: new Date().toISOString(),
  }).eq('id', videoId);

  // Simulate AI analysis delay
  await new Promise(r => setTimeout(r, 3500));

  const stub = pickStub(videoId);

  // Generate the full overlay timeline + phase markers
  const overlayTimeline = generateOverlayTimeline(stub.issue_type as Parameters<typeof generateOverlayTimeline>[0]);

  try {
    const { data: analysis, error: insertErr } = await supabase
      .from('swing_analysis')
      .insert({
        video_id: videoId,
        issue_type: stub.issue_type,
        summary_text: stub.summary_text,
        cue_text: stub.cue_text,
        drill_text: stub.drill_text,
        score: stub.score,
        severity: stub.severity,
        // NEW: store timeline data for the interactive player
        overlay_timeline_json: overlayTimeline,
        phase_markers_json: overlayTimeline.phases,
        video_metadata_json: {
          fps: 30,
          aspect_ratio: 1.777,
          note: 'stub — real metadata from video processing in Phase 2',
        },
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
      status: 'failed',
      error_code: 'ANALYSIS_STUB_ERROR',
      error_message: msg,
      processing_completed_at: new Date().toISOString(),
    }).eq('id', videoId);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
