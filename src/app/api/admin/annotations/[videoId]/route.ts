import { NextRequest, NextResponse } from 'next/server';
import { requireAdmin } from '@/lib/auth';
import { createServiceClient } from '@/lib/supabase/admin';
import type { AnnotationTaskType } from '@/lib/types/annotation';

const VALID_TASK_TYPES: AnnotationTaskType[] = [
  'manual_gt', 'correction_review', 'active_learning',
];

function parseTaskType(raw: string | null): AnnotationTaskType {
  if (raw && (VALID_TASK_TYPES as string[]).includes(raw)) {
    return raw as AnnotationTaskType;
  }
  return 'manual_gt';
}

/**
 * GET /api/admin/annotations/:videoId?taskType=manual_gt
 *
 * Lists the current annotator's saved annotations for a single video,
 * filtered by task_type (default 'manual_gt'). Ordered by frame_idx then
 * arm so the workbench can match them positionally against the
 * generated 8-task list.
 */
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ videoId: string }> },
) {
  const auth = await requireAdmin();
  if ('response' in auth) return auth.response;

  const { videoId } = await params;
  const taskType = parseTaskType(req.nextUrl.searchParams.get('taskType'));

  const admin = createServiceClient();
  const { data, error } = await admin
    .from('golf_landmark_annotations')
    .select('*')
    .eq('video_id', videoId)
    .eq('annotator_id', auth.user.id)
    .eq('task_type', taskType)
    .order('frame_idx', { ascending: true })
    .order('arm', { ascending: true });

  if (error) {
    return NextResponse.json(
      { error: 'query_failed', detail: error.message },
      { status: 500 },
    );
  }

  return NextResponse.json({ annotations: data ?? [] });
}
