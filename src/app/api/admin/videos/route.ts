import { NextResponse } from 'next/server';
import { requireAdmin } from '@/lib/auth';
import { createServiceClient } from '@/lib/supabase/admin';
import type { VideoListEntry } from '@/lib/types/annotation';

/**
 * GET /api/admin/videos
 *
 * Returns up to 100 completed videos, sorted WHAM-ready first then
 * created_at desc. Each entry carries the WHAM frame-count badge data
 * and the count of manual_gt annotations the CURRENT admin has saved
 * on that video.
 *
 * Three queries server-side, hand-stitched — PostgREST nested counts
 * are fiddly enough that explicit composition is clearer.
 */
export async function GET() {
  const auth = await requireAdmin();
  if ('response' in auth) return auth.response;

  const admin = createServiceClient();

  const { data: videos, error: videosErr } = await admin
    .from('swing_videos')
    .select('id, original_filename, view_type, created_at')
    .eq('status', 'completed')
    .order('created_at', { ascending: false })
    .limit(100);

  if (videosErr) {
    return NextResponse.json(
      { error: 'query_failed', detail: videosErr.message },
      { status: 500 },
    );
  }
  if (!videos || videos.length === 0) {
    return NextResponse.json({ videos: [] as VideoListEntry[] });
  }

  const ids = videos.map(v => v.id as string);

  const [whamRes, annRes] = await Promise.all([
    admin
      .from('wham_video_meta')
      .select('video_id, frame_count, processed_fps')
      .in('video_id', ids),
    admin
      .from('golf_landmark_annotations')
      .select('video_id')
      .eq('annotator_id', auth.user.id)
      .eq('task_type', 'manual_gt')
      .in('video_id', ids),
  ]);

  if (whamRes.error) {
    return NextResponse.json(
      { error: 'wham_query_failed', detail: whamRes.error.message },
      { status: 500 },
    );
  }
  if (annRes.error) {
    return NextResponse.json(
      { error: 'annotation_query_failed', detail: annRes.error.message },
      { status: 500 },
    );
  }

  const whamByVid = new Map<string, { frame_count: number; processed_fps: number }>();
  for (const row of whamRes.data ?? []) {
    whamByVid.set(row.video_id as string, {
      frame_count: Number(row.frame_count ?? 0),
      processed_fps: Number(row.processed_fps ?? 0),
    });
  }

  const annCountByVid = new Map<string, number>();
  for (const row of annRes.data ?? []) {
    const vid = row.video_id as string;
    annCountByVid.set(vid, (annCountByVid.get(vid) ?? 0) + 1);
  }

  const stitched: VideoListEntry[] = videos.map(v => {
    const wham = whamByVid.get(v.id as string) ?? null;
    return {
      id: v.id as string,
      original_filename: (v.original_filename as string | null) ?? null,
      view_type: (v.view_type as string) ?? 'face_on',
      created_at: v.created_at as string,
      hasWham: wham !== null,
      whamMeta: wham,
      annotationCount: annCountByVid.get(v.id as string) ?? 0,
    };
  });

  // WHAM-ready first; within each group, preserve the created_at desc
  // order from the swing_videos query (sort is stable in V8).
  stitched.sort((a, b) => {
    if (a.hasWham !== b.hasWham) return a.hasWham ? -1 : 1;
    return 0;
  });

  return NextResponse.json({ videos: stitched });
}
