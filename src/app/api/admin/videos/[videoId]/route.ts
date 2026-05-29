import { NextRequest, NextResponse } from 'next/server';
import { requireAdmin } from '@/lib/auth';
import { createServiceClient } from '@/lib/supabase/admin';
import type { VideoMetaForAnnotation } from '@/lib/types/annotation';

/**
 * GET /api/admin/videos/:videoId
 *
 * Returns VideoMetaForAnnotation: dimensions, fps, frame count, phase
 * markers, and a 1-hour signed playback URL.
 *
 * Dimension precedence: wham_video_meta first (more accurate — it's the
 * canonical analysis dimensions), then swing_analysis.video_metadata_json
 * as fallback for pre-WHAM videos.
 */
export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ videoId: string }> },
) {
  const auth = await requireAdmin();
  if ('response' in auth) return auth.response;

  const { videoId } = await params;
  const admin = createServiceClient();

  const { data: video, error: videoErr } = await admin
    .from('swing_videos')
    .select('id, original_filename, storage_path, view_type')
    .eq('id', videoId)
    .maybeSingle();

  if (videoErr) {
    return NextResponse.json(
      { error: 'query_failed', detail: videoErr.message },
      { status: 500 },
    );
  }
  if (!video) {
    return NextResponse.json({ error: 'not_found' }, { status: 404 });
  }

  const storagePath = (video.storage_path as string | null) ?? '';
  if (!storagePath) {
    return NextResponse.json(
      { error: 'no_storage_path', detail: 'video row has empty storage_path' },
      { status: 409 },
    );
  }

  const [whamRes, anaRes, signedRes] = await Promise.all([
    admin
      .from('wham_video_meta')
      .select('image_width, image_height, processed_fps, frame_count')
      .eq('video_id', videoId)
      .maybeSingle(),
    admin
      .from('swing_analysis')
      .select('video_metadata_json, phase_markers_json')
      .eq('video_id', videoId)
      // swing_analysis.id is uuid (lexicographic, not temporal). Order
      // by created_at so videos with multiple analyze runs return the
      // most recent analysis row — matches the fix applied to the
      // export route in commit a59f493.
      .order('created_at', { ascending: false })
      .limit(1)
      .maybeSingle(),
    admin.storage
      .from('swing-videos')
      .createSignedUrl(storagePath, 3600),
  ]);

  if (signedRes.error || !signedRes.data?.signedUrl) {
    return NextResponse.json(
      { error: 'signed_url_failed', detail: signedRes.error?.message ?? 'unknown' },
      { status: 500 },
    );
  }

  const wham = whamRes.data;

  type VmJson = {
    width?: number;
    height?: number;
    fps?: number;
    durationSec?: number;
  };
  const vm = (anaRes.data?.video_metadata_json as VmJson | null | undefined) ?? null;

  type PmJson = {
    setupTime?: number | null;
    topTime?: number | null;
    impactTime?: number | null;
    finishTime?: number | null;
    transitionTime?: number | null;
  };
  const pm = (anaRes.data?.phase_markers_json as PmJson | null | undefined) ?? null;

  const width = Number(wham?.image_width ?? vm?.width ?? 0);
  const height = Number(wham?.image_height ?? vm?.height ?? 0);
  const fps = Number(wham?.processed_fps ?? vm?.fps ?? 30);
  const durationSec = Number(vm?.durationSec ?? 0);
  const frameCount = Number(
    wham?.frame_count ?? (durationSec > 0 ? Math.round(durationSec * fps) : 0),
  );

  const meta: VideoMetaForAnnotation = {
    videoId,
    filename: (video.original_filename as string | null) ?? null,
    width,
    height,
    fps,
    durationSec,
    frameCount,
    hasWhamData: wham !== null,
    phaseMarkers: {
      setupTime: pm?.setupTime ?? null,
      topTime: pm?.topTime ?? null,
      impactTime: pm?.impactTime ?? null,
      finishTime: pm?.finishTime ?? null,
      transitionTime: pm?.transitionTime ?? null,
    },
    signedVideoUrl: signedRes.data.signedUrl,
  };

  return NextResponse.json(meta);
}
