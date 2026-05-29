import { NextRequest, NextResponse } from 'next/server';
import { requireAdmin } from '@/lib/auth';
import { createServiceClient } from '@/lib/supabase/admin';

/**
 * GET /api/admin/annotations/:videoId/export
 *
 * Returns the current annotator's manual_gt annotations for the video
 * in the standalone HTML tool's JSON format, so fixtures/pr-7a/
 * annotations/<video_id>.json stays drop-in compatible.
 *
 * Schema (schema_version 'pr7a-v1'):
 *   {
 *     video_id, video_filename, video_dimensions: { width, height },
 *     fps, annotated_at, schema_version,
 *     annotations: [{ frame_idx, phase, arm, visibility_note,
 *                     points: { shoulder_cap, elbow_center, wrist_center } }]
 *   }
 *
 * Each point is either { x, y } or null (whole point missing).
 */
export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ videoId: string }> },
) {
  const auth = await requireAdmin();
  if ('response' in auth) return auth.response;

  const { videoId } = await params;
  const admin = createServiceClient();

  const [videoRes, whamRes, anaRes, annRes] = await Promise.all([
    admin
      .from('swing_videos')
      .select('id, original_filename')
      .eq('id', videoId)
      .maybeSingle(),
    admin
      .from('wham_video_meta')
      .select('image_width, image_height, processed_fps')
      .eq('video_id', videoId)
      .maybeSingle(),
    admin
      .from('swing_analysis')
      .select('video_metadata_json')
      .eq('video_id', videoId)
      .order('id', { ascending: false })
      .limit(1)
      .maybeSingle(),
    admin
      .from('golf_landmark_annotations')
      .select('*')
      .eq('video_id', videoId)
      .eq('annotator_id', auth.user.id)
      .eq('task_type', 'manual_gt')
      .order('frame_idx', { ascending: true })
      .order('arm', { ascending: true }),
  ]);

  if (videoRes.error || !videoRes.data) {
    return NextResponse.json({ error: 'video_not_found' }, { status: 404 });
  }
  if (annRes.error) {
    return NextResponse.json(
      { error: 'annotation_query_failed', detail: annRes.error.message },
      { status: 500 },
    );
  }

  type VmJson = { width?: number; height?: number; fps?: number };
  const vm = (anaRes.data?.video_metadata_json as VmJson | null | undefined) ?? null;
  const wham = whamRes.data;

  const width = Number(wham?.image_width ?? vm?.width ?? 0);
  const height = Number(wham?.image_height ?? vm?.height ?? 0);
  const fps = Number(wham?.processed_fps ?? vm?.fps ?? 30);

  const rows = annRes.data ?? [];

  type AnnRow = {
    frame_idx: number;
    phase: string;
    arm: string;
    visibility: string;
    annotated_at: string;
    shoulder_x: number | null; shoulder_y: number | null;
    elbow_x: number | null;    elbow_y: number | null;
    wrist_x: number | null;    wrist_y: number | null;
  };

  type Pt = { x: number; y: number } | null;
  const toPoint = (x: number | null, y: number | null): Pt =>
    x !== null && y !== null ? { x, y } : null;

  const annotations = (rows as unknown as AnnRow[]).map(r => ({
    frame_idx: r.frame_idx,
    phase: r.phase,
    arm: r.arm,
    visibility_note: r.visibility,
    points: {
      shoulder_cap: toPoint(r.shoulder_x, r.shoulder_y),
      elbow_center: toPoint(r.elbow_x, r.elbow_y),
      wrist_center: toPoint(r.wrist_x, r.wrist_y),
    },
  }));

  const latestAnnotatedAt = (rows as unknown as AnnRow[])
    .map(r => r.annotated_at)
    .filter((s): s is string => typeof s === 'string')
    .sort()
    .pop() ?? null;

  const payload = {
    video_id: videoId,
    video_filename: videoRes.data.original_filename as string | null,
    video_dimensions: { width, height },
    fps,
    annotated_at: latestAnnotatedAt,
    schema_version: 'pr7a-v1',
    annotations,
  };

  const shortId = videoId.slice(0, 8);
  return new NextResponse(JSON.stringify(payload, null, 2), {
    status: 200,
    headers: {
      'Content-Type': 'application/json',
      'Content-Disposition': `attachment; filename="annotations-${shortId}.json"`,
    },
  });
}
