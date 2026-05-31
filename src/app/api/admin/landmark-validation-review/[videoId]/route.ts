import { NextRequest, NextResponse } from 'next/server';
import { requireAdmin } from '@/lib/auth';
import { createServiceClient } from '@/lib/supabase/admin';

/**
 * GET /api/admin/landmark-validation-review/:videoId
 *
 * Returns all landmark_validation_review rows for this video saved by
 * the current admin. Ordered by phase in the swing's natural sequence
 * (setup → takeaway → top → transition → impact), NOT alphabetical, so
 * the client can paint progress without re-sorting.
 *
 * Wrapped in try/catch (mirrors hotfix 169a7aa). 5 expected rows max
 * per (video, reviewer); no pagination needed.
 */

const PHASE_ORDER: readonly string[] = [
  'setup', 'takeaway', 'top', 'transition', 'impact',
];

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ videoId: string }> },
) {
  try {
    const auth = await requireAdmin();
    if ('response' in auth) return auth.response;

    const { videoId } = await params;

    const admin = createServiceClient();
    const { data, error } = await admin
      .from('landmark_validation_review')
      .select('*')
      .eq('video_id', videoId)
      .eq('reviewer_id', auth.user.id);

    if (error) {
      return NextResponse.json(
        { error: 'query_failed', detail: error.message, pg_code: error.code },
        { status: 500 },
      );
    }

    // Sort by phase enum order (Postgres can't .order() by a custom
    // enum cheaply; cheaper to sort client-side over ≤5 rows).
    type Row = { phase: string } & Record<string, unknown>;
    const rows = (data ?? []) as Row[];
    const phaseRank = (p: string) => {
      const i = PHASE_ORDER.indexOf(p);
      return i === -1 ? PHASE_ORDER.length : i;
    };
    rows.sort((a, b) => phaseRank(a.phase) - phaseRank(b.phase));

    return NextResponse.json({ reviews: rows });
  } catch (err) {
    console.error('[admin/landmark-validation-review GET] uncaught:', err);
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
