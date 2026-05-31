import { NextRequest, NextResponse } from 'next/server';
import { requireAdmin } from '@/lib/auth';
import { createServiceClient } from '@/lib/supabase/admin';

/**
 * POST /api/admin/landmark-validation-review
 *
 * Upsert one (video_id, reviewer_id, phase) verdict. Reviewer_id is
 * derived from the session — never trusted from the body. On a
 * duplicate (existing row for the same video+reviewer+phase) the row
 * is UPDATED in place; this is the intended flow for re-reviewing.
 *
 * Body shape:
 *   {
 *     video_id: uuid,
 *     phase: 'setup' | 'takeaway' | 'top' | 'transition' | 'impact',
 *     verdict: 'correct' | 'incorrect' | 'unsure',
 *     notes?: string,
 *     compared_sources?: string[]  // defaults to ['wham','mediapipe']
 *   }
 *
 * source_app_version is pinned server-side to 'swingcue-review-1.0'.
 * The DB CHECK on (phase) + (verdict) enforces the enums; this route
 * pre-validates to give the client a clean 400 instead of a 23514.
 *
 * Wrapped in try/catch (mirrors hotfix 169a7aa) so an uncaught throw
 * surfaces { error, message, name, stack[0..5] } instead of Vercel's
 * opaque default 500 page.
 */

const PHASES = ['setup', 'takeaway', 'top', 'transition', 'impact'] as const;
type ReviewPhase = typeof PHASES[number];

const VERDICTS = ['correct', 'incorrect', 'unsure'] as const;
type Verdict = typeof VERDICTS[number];

const JOINT_KEYS = [
  'lead_shoulder', 'lead_elbow', 'lead_wrist',
  'trail_shoulder', 'trail_elbow', 'trail_wrist',
  'lead_hip', 'trail_hip',
] as const;
type JointKey = typeof JOINT_KEYS[number];

type CalibratedKeypoints = Partial<Record<JointKey, { x: number; y: number }>>;

function isOneOf<T extends string>(v: unknown, allowed: readonly T[]): v is T {
  return typeof v === 'string' && (allowed as readonly string[]).includes(v);
}

/**
 * Validate the optional calibrated_keypoints body field.
 * Returns:
 *   { ok: true, value: CalibratedKeypoints | null }
 *     - null means client explicitly sent null / omitted ("no calibration")
 *     - object means a (possibly partial) per-joint calibration map
 *   { ok: false, error: string }
 *     - shape mismatch — caller returns 400
 */
function parseCalibratedKeypoints(
  raw: unknown,
): { ok: true; value: CalibratedKeypoints | null } | { ok: false; error: string } {
  if (raw === null || raw === undefined) return { ok: true, value: null };
  if (typeof raw !== 'object' || Array.isArray(raw)) {
    return { ok: false, error: 'calibrated_keypoints must be an object or null' };
  }
  const out: CalibratedKeypoints = {};
  for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
    if (!isOneOf<JointKey>(k, JOINT_KEYS)) {
      return { ok: false, error: `unknown joint key '${k}' in calibrated_keypoints` };
    }
    if (v === null || v === undefined) continue;
    if (typeof v !== 'object' || Array.isArray(v)) {
      return { ok: false, error: `calibrated_keypoints.${k} must be { x, y }` };
    }
    const obj = v as Record<string, unknown>;
    if (typeof obj.x !== 'number' || !Number.isFinite(obj.x) ||
        typeof obj.y !== 'number' || !Number.isFinite(obj.y)) {
      return { ok: false, error: `calibrated_keypoints.${k} must have numeric x and y` };
    }
    out[k] = { x: obj.x, y: obj.y };
  }
  return { ok: true, value: out };
}

interface ReviewBody {
  video_id?: unknown;
  phase?: unknown;
  verdict?: unknown;
  notes?: unknown;
  compared_sources?: unknown;
  calibrated_keypoints?: unknown;
}

export async function POST(req: NextRequest) {
  try {
    const auth = await requireAdmin();
    if ('response' in auth) return auth.response;

    let body: ReviewBody;
    try {
      body = (await req.json()) as ReviewBody;
    } catch {
      return NextResponse.json(
        { error: 'invalid_json', detail: 'body could not be parsed as JSON' },
        { status: 400 },
      );
    }

    if (typeof body.video_id !== 'string' || body.video_id.length === 0) {
      return NextResponse.json({ error: 'invalid_video_id' }, { status: 400 });
    }
    if (!isOneOf<ReviewPhase>(body.phase, PHASES)) {
      return NextResponse.json({ error: 'invalid_phase' }, { status: 400 });
    }
    if (!isOneOf<Verdict>(body.verdict, VERDICTS)) {
      return NextResponse.json({ error: 'invalid_verdict' }, { status: 400 });
    }

    const notes: string | null =
      typeof body.notes === 'string' ? body.notes : null;

    let comparedSources: string[] = ['wham', 'mediapipe'];
    if (Array.isArray(body.compared_sources)) {
      const cs = body.compared_sources.filter((s): s is string => typeof s === 'string');
      if (cs.length > 0) comparedSources = cs;
    }

    // PR-7A.1 Phase 3: drag-to-correct per-joint pixel positions.
    // null means GT was untouched (verdict='correct' inferred client-side).
    // object means at least one joint dragged; client sends the full
    // 8-joint set so downstream tooling can compute deltas without
    // having to look up the original GT.
    const calibParsed = parseCalibratedKeypoints(body.calibrated_keypoints);
    if (!calibParsed.ok) {
      return NextResponse.json({ error: 'invalid_calibrated_keypoints', detail: calibParsed.error }, { status: 400 });
    }
    const calibrated_keypoints = calibParsed.value;

    const admin = createServiceClient();
    const { data, error } = await admin
      .from('landmark_validation_review')
      .upsert(
        {
          video_id: body.video_id,
          reviewer_id: auth.user.id,
          phase: body.phase,
          verdict: body.verdict,
          notes,
          compared_sources: comparedSources,
          calibrated_keypoints,
          source_app_version: 'swingcue-review-1.0',
          reviewed_at: new Date().toISOString(),
        },
        { onConflict: 'video_id,reviewer_id,phase' },
      )
      .select()
      .single();

    if (error) {
      return NextResponse.json(
        {
          error: 'upsert_failed',
          detail: error.message,
          pg_code: error.code,
        },
        { status: error.code === '23514' ? 400 : 500 },
      );
    }

    return NextResponse.json({ saved: data });
  } catch (err) {
    console.error('[admin/landmark-validation-review POST] uncaught:', err);
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
