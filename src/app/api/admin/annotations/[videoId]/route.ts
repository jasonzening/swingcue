import { NextRequest, NextResponse } from 'next/server';
import { requireAdmin } from '@/lib/auth';
import { createServiceClient } from '@/lib/supabase/admin';
import type { AnnotationTaskType } from '@/lib/types/annotation';

/**
 * PR-7A.2: whitelist extended with the four v2 task_types AND the
 * sentinel 'all' (fetch every v2 row regardless of task_type). The
 * workbench uses 'all' so its task-completion checks can see arm + hip
 * + head + leg rows in one request — fixes the PR-7A.1 hip-resume gap
 * where the workbench fetched only manual_gt and so hip_pair tasks
 * never registered as completed across sessions.
 */
const VALID_TASK_TYPES: AnnotationTaskType[] = [
  'manual_gt',
  'manual_gt_hip_pair',
  'manual_gt_head_set',
  'manual_gt_leg',
  'correction_review',
  'active_learning',
];

type TaskTypeFilter = AnnotationTaskType | 'all';

function parseTaskType(raw: string | null): TaskTypeFilter {
  if (raw === 'all') return 'all';
  if (raw && (VALID_TASK_TYPES as string[]).includes(raw)) {
    return raw as AnnotationTaskType;
  }
  return 'manual_gt';
}

/**
 * GET /api/admin/annotations/:videoId?taskType=manual_gt
 *                              ?taskType=all   (PR-7A.2 — every v2 row)
 *
 * Lists the current annotator's saved v2 anatomical-spec annotations
 * for a single video, filtered by task_type.
 *
 * PR-7A.2: filter on source_app_version LIKE 'swingcue-annotate-2.0%'
 * so progress counters never count v1 :deprecated-v1 rows. Architect
 * caught false "10/10 ARM done" reports on videos that had v1 data and
 * zero v2 data — that filter is the cure. Existing 998e1930 +
 * 51ca9428 v2 rows continue to register as completed because their
 * source_app_version starts with the matched prefix.
 *
 * Ordered by frame_idx then arm so the workbench can match them
 * positionally against the generated 30-task list.
 */
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ videoId: string }> },
) {
  try {
    const auth = await requireAdmin();
    if ('response' in auth) return auth.response;

    const { videoId } = await params;
    const taskType = parseTaskType(req.nextUrl.searchParams.get('taskType'));

    const admin = createServiceClient();
    let query = admin
      .from('golf_landmark_annotations')
      .select('*')
      .eq('video_id', videoId)
      .eq('annotator_id', auth.user.id)
      .like('source_app_version', 'swingcue-annotate-2.0%');

    if (taskType !== 'all') {
      query = query.eq('task_type', taskType);
    }

    const { data, error } = await query
      .order('frame_idx', { ascending: true })
      .order('arm', { ascending: true });

    if (error) {
      return NextResponse.json(
        { error: 'query_failed', detail: error.message },
        { status: 500 },
      );
    }

    return NextResponse.json({ annotations: data ?? [] });
  } catch (err) {
    // Currently the route can blow up before reaching the supabase
    // error-handling block (auth helper throw, env var missing,
    // service-role client construction failure, etc.) and the runtime
    // serializes the resulting 500 as Vercel's default opaque error
    // page — no body, no stack, nothing the workbench can show the
    // user. Wrap and surface enough to RCA from a single fetch.
    console.error('[admin/annotations GET] uncaught:', err);
    return NextResponse.json(
      {
        error: 'uncaught_exception',
        message: err instanceof Error ? err.message : String(err),
        name: err instanceof Error ? err.name : undefined,
        stack: err instanceof Error
          ? err.stack?.split('\n').slice(0, 5).join(' | ')
          : undefined,
      },
      { status: 500 },
    );
  }
}
