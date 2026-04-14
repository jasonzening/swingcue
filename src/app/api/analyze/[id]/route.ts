import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@/lib/supabase/server';
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
    summary_text:
      'Your hips are firing early and pulling you off the ball before impact. This is the most common power leak for amateur golfers — it causes thin shots, loss of distance, and inconsistency.',
    cue_text: 'Stay in your posture through impact.',
    drill_text:
      'Wall Drill: Stand with your backside lightly touching a wall at address. Make slow practice swings without losing contact with the wall through impact. 5 reps before every session.',
    score: 72,
    severity: 'medium',
  },
  {
    issue_type: 'steep_downswing',
    summary_text:
      'Your club is coming down too steep, creating an over-the-top path. This causes pulls, weak fades, and chunks — and it is robbing you of significant distance.',
    cue_text: 'Drop your trail elbow into your hip on the downswing.',
    drill_text:
      'Headcover Drill: Tuck a headcover under your trail arm at address. Make your downswing without dropping it. When you can do 10 reps cleanly, your downswing plane is fixed.',
    score: 65,
    severity: 'medium',
  },
  {
    issue_type: 'head_movement',
    summary_text:
      'Your head is drifting laterally during the backswing. This lateral sway makes it very hard to return the club to the ball consistently — a small drift creates a large miss.',
    cue_text: "Keep your head still — feel like it's pinned to the wall behind you.",
    drill_text:
      'Statue Drill: Ask a friend to lightly hold your cap with two fingers while you make your backswing. Repeat 15 times slowly. The resistance will train stillness into muscle memory.',
    score: 78,
    severity: 'low',
  },
  {
    issue_type: 'weight_shift_issue',
    summary_text:
      "Your weight is staying on your back foot through impact — this is called a reverse pivot. It steepens your attack angle and kills your distance. You're likely hitting fat or thin shots.",
    cue_text: 'Feel your lead knee drive toward the target to start your downswing.',
    drill_text:
      'Step Drill: At the top of your backswing, step your trail foot toward your lead foot as you start down. This forces proper weight transfer. 20 reps, slow and deliberate.',
    score: 68,
    severity: 'high',
  },
  {
    issue_type: 'hand_path_issue',
    summary_text:
      'Your hands are looping away from your body at the start of the downswing. This creates an inconsistent impact position — sometimes early release, sometimes blocked shots.',
    cue_text: 'Keep your hands close to your body on the downswing.',
    drill_text:
      'Towel Drill: Tuck a rolled towel between your upper right arm and chest. Make half-swings, keeping the towel pinched. When the towel drops, you have disconnected. 15 reps per session.',
    score: 70,
    severity: 'medium',
  },
];

function pickStub(videoId: string): StubData {
  // Deterministic: same video always returns the same stub
  let hash = 0;
  for (const ch of videoId.replace(/-/g, '')) {
    hash = (hash * 31 + parseInt(ch, 16)) >>> 0;
  }
  return STUBS[hash % STUBS.length];
}

export async function POST(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id: videoId } = await params;
  const supabase = await createClient();

  // 1. Auth check
  const {
    data: { user },
    error: authErr,
  } = await supabase.auth.getUser();

  if (authErr || !user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  // 2. Verify video ownership
  const { data: video, error: videoErr } = await supabase
    .from('swing_videos')
    .select('id, status, user_id')
    .eq('id', videoId)
    .eq('user_id', user.id)
    .single();

  if (videoErr || !video) {
    return NextResponse.json({ error: 'Video not found' }, { status: 404 });
  }

  // 3. Already completed — return existing
  if (video.status === 'completed') {
    const { data: existing } = await supabase
      .from('swing_analysis')
      .select('*')
      .eq('video_id', videoId)
      .single();
    return NextResponse.json({ status: 'completed', analysis: existing });
  }

  // 4. Still processing — don't double-run
  if (video.status === 'processing') {
    return NextResponse.json({ status: 'processing' }, { status: 202 });
  }

  // 5. Start processing
  await supabase
    .from('swing_videos')
    .update({
      status: 'processing',
      processing_started_at: new Date().toISOString(),
    })
    .eq('id', videoId);

  // 6. Simulate AI processing (deterministic stub)
  await new Promise(resolve => setTimeout(resolve, 3500));

  const stub = pickStub(videoId);

  try {
    // 7. Write analysis row
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
      })
      .select()
      .single();

    if (insertErr) throw new Error(insertErr.message);

    // 8. Mark completed
    await supabase
      .from('swing_videos')
      .update({
        status: 'completed',
        processing_completed_at: new Date().toISOString(),
      })
      .eq('id', videoId);

    return NextResponse.json({ status: 'completed', analysis });
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : 'Unknown error';

    await supabase
      .from('swing_videos')
      .update({
        status: 'failed',
        error_code: 'ANALYSIS_STUB_ERROR',
        error_message: msg,
        processing_completed_at: new Date().toISOString(),
      })
      .eq('id', videoId);

    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
