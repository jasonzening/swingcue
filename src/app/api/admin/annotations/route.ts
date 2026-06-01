import { NextRequest, NextResponse } from 'next/server';
import { requireAdmin } from '@/lib/auth';
import { createServiceClient } from '@/lib/supabase/admin';
import type {
  AnnotationArm,
  AnnotationRecord,
  AnnotationTaskType,
  AnnotationVisibility,
  Handedness,
} from '@/lib/types/annotation';
import { ANNOTATION_PHASES } from '@/lib/types/annotation';

// Runtime phase whitelist — sourced from ANNOTATION_PHASES in
// lib/types/annotation so it can never drift from the AnnotationPhase
// union. The Tier-2 expansion (takeaway, post_impact) flowed through
// here automatically once the const tuple was extended.
const PHASES = ANNOTATION_PHASES;
const TASK_TYPES: AnnotationTaskType[] = [
  'manual_gt',
  'manual_gt_hip_pair',
  'manual_gt_head_set',
  'manual_gt_leg',
  'correction_review',
  'active_learning',
];
const ARMS: AnnotationArm[] = ['lead', 'trail'];
const VISIBILITIES: AnnotationVisibility[] = ['clear', 'occluded', 'uncertain'];
const HANDEDNESSES: Handedness[] = ['right', 'left'];

function isOneOf<T extends string>(value: unknown, allowed: readonly T[]): value is T {
  return typeof value === 'string' && (allowed as readonly string[]).includes(value);
}

function asFiniteNumberOrNull(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  return null;
}

type InsertOrUpdateOk = {
  data: Record<string, unknown>;
  error?: undefined;
};
type InsertOrUpdateErr = {
  data?: undefined;
  error: string;
  detail: string;
  pg_code: string | undefined;
  status: number;
};

/**
 * PR-7A.2: insert-or-update-on-23505 helper for partial-unique-index
 * upserts. PostgREST's `onConflict` column inference cannot target a
 * partial unique index (no way to pass the index predicate), so we
 * INSERT first; on unique_violation (23505) we UPDATE in place using
 * the composite-key columns the caller passes. Both paths return the
 * row as-saved so the workbench can echo it back into local state.
 */
async function insertOrUpdateOnConflict(
  row: Record<string, unknown>,
  keyCols: Record<string, string | number | null>,
): Promise<InsertOrUpdateOk | InsertOrUpdateErr> {
  const admin = createServiceClient();
  const ins = await admin
    .from('golf_landmark_annotations')
    .insert(row)
    .select()
    .single();

  if (!ins.error) {
    return { data: ins.data as Record<string, unknown> };
  }

  // 23514 = check_violation: row failed the cross-cluster CHECK
  // constraint. Surface verbatim — DB is source of truth.
  if (ins.error.code !== '23505') {
    return {
      error: 'insert_failed',
      detail: ins.error.message,
      pg_code: ins.error.code,
      status: ins.error.code === '23514' ? 400 : 500,
    };
  }

  // unique_violation → UPDATE in place. eq() each key column so the
  // UPDATE targets exactly the row the partial index protected.
  let upd = admin
    .from('golf_landmark_annotations')
    .update(row);
  for (const [col, val] of Object.entries(keyCols)) {
    upd = upd.eq(col, val);
  }
  const updated = await upd.select().single();

  if (updated.error) {
    return {
      error: 'update_after_unique_violation_failed',
      detail: updated.error.message,
      pg_code: updated.error.code,
      status: updated.error.code === '23514' ? 400 : 500,
    };
  }
  return { data: updated.data as Record<string, unknown> };
}

/**
 * POST /api/admin/annotations
 *
 * Two-mode upsert/insert based on task_type discriminator:
 *
 *   - task_type === 'manual_gt' (or correction_review / active_learning)
 *     → arm-coord record. arm REQUIRED, shoulder/elbow/wrist on the
 *       visibility='clear' contract, hip columns MUST be null.
 *       UPSERT on (video_id, frame_idx, arm, annotator_id, task_type) —
 *       partial unique index uq_gla_arm_task. Re-saving overwrites.
 *
 *   - task_type === 'manual_gt_hip_pair'
 *     → hip-pair record. arm MUST be null, all 4 hip coords REQUIRED,
 *       arm columns MUST be null, visibility pinned to 'clear' (hip
 *       is an estimated internal joint center, not directly visible).
 *       INSERT — partial unique index uq_gla_hip_pair triggers a
 *       23505 unique_violation on dup; surfaced as 409 with the
 *       'hip_pair_already_exists' code so the client can route to
 *       UPDATE flow instead of POST.
 *
 * Both branches surface PG error codes + messages back to the client
 * (don't swallow) — DB CHECK constraints are the source of truth for
 * row coherence; API-level validation is for fast feedback.
 */
export async function POST(req: NextRequest) {
  const auth = await requireAdmin();
  if ('response' in auth) return auth.response;

  let body: Partial<AnnotationRecord>;
  try {
    body = (await req.json()) as Partial<AnnotationRecord>;
  } catch {
    return NextResponse.json(
      { error: 'invalid_json', detail: 'body could not be parsed as JSON' },
      { status: 400 },
    );
  }

  // ── Common (both task kinds) ──────────────────────────────────────
  if (typeof body.video_id !== 'string' || body.video_id.length === 0) {
    return NextResponse.json({ error: 'invalid_video_id' }, { status: 400 });
  }
  if (typeof body.frame_idx !== 'number' || !Number.isInteger(body.frame_idx) || body.frame_idx < 0) {
    return NextResponse.json({ error: 'invalid_frame_idx' }, { status: 400 });
  }
  if (!isOneOf(body.phase, PHASES)) {
    return NextResponse.json({ error: 'invalid_phase' }, { status: 400 });
  }
  if (!isOneOf(body.task_type, TASK_TYPES)) {
    return NextResponse.json({ error: 'invalid_task_type' }, { status: 400 });
  }
  if (!isOneOf(body.visibility, VISIBILITIES)) {
    return NextResponse.json({ error: 'invalid_visibility' }, { status: 400 });
  }
  if (!isOneOf(body.handedness, HANDEDNESSES)) {
    return NextResponse.json({ error: 'invalid_handedness' }, { status: 400 });
  }
  if (typeof body.source_app_version !== 'string' || body.source_app_version.length === 0) {
    return NextResponse.json({ error: 'invalid_source_app_version' }, { status: 400 });
  }

  const admin = createServiceClient();

  // ── Hip-pair branch ───────────────────────────────────────────────
  if (body.task_type === 'manual_gt_hip_pair') {
    // arm must be null (not undefined, not a string)
    if (body.arm !== null && body.arm !== undefined) {
      return NextResponse.json(
        { error: 'hip_pair_requires_arm_null', task_type: body.task_type },
        { status: 400 },
      );
    }

    const lead_hip_x  = asFiniteNumberOrNull(body.lead_hip_x);
    const lead_hip_y  = asFiniteNumberOrNull(body.lead_hip_y);
    const trail_hip_x = asFiniteNumberOrNull(body.trail_hip_x);
    const trail_hip_y = asFiniteNumberOrNull(body.trail_hip_y);

    if (lead_hip_x === null || lead_hip_y === null || trail_hip_x === null || trail_hip_y === null) {
      return NextResponse.json(
        { error: 'hip_pair_requires_all_four_coords', task_type: body.task_type },
        { status: 400 },
      );
    }

    // Reject any arm-coord columns — arm coords on a hip row violate
    // the row-coherence CHECK constraint anyway, but surface this as a
    // semantic 400 not a DB 23514.
    if (asFiniteNumberOrNull(body.shoulder_x) !== null || asFiniteNumberOrNull(body.shoulder_y) !== null ||
        asFiniteNumberOrNull(body.elbow_x)    !== null || asFiniteNumberOrNull(body.elbow_y)    !== null ||
        asFiniteNumberOrNull(body.wrist_x)    !== null || asFiniteNumberOrNull(body.wrist_y)    !== null) {
      return NextResponse.json(
        { error: 'hip_pair_rejects_arm_coords', task_type: body.task_type },
        { status: 400 },
      );
    }

    // Hip is an estimated internal joint center — there's no "occluded"
    // or "uncertain" semantic for it. Workbench enforces 'clear'; we
    // re-validate so curl/postman requests can't sneak past.
    if (body.visibility !== 'clear') {
      return NextResponse.json(
        { error: 'hip_pair_visibility_must_be_clear', task_type: body.task_type },
        { status: 400 },
      );
    }

    const hipRow = {
      video_id: body.video_id,
      annotator_id: auth.user.id,
      frame_idx: body.frame_idx,
      phase: body.phase,
      task_type: 'manual_gt_hip_pair' as const,
      arm: null,
      visibility: 'clear' as const,
      lead_hip_x, lead_hip_y,
      trail_hip_x, trail_hip_y,
      handedness: body.handedness,
      source_app_version: body.source_app_version,
    };

    const { data, error } = await admin
      .from('golf_landmark_annotations')
      .insert(hipRow)
      .select()
      .single();

    if (error) {
      // 23505 = PostgreSQL unique_violation. Hits when uq_gla_hip_pair
      // catches a duplicate (video, frame, annotator, task_type=hip_pair).
      // 409 with a specific code so the client can switch to UPDATE flow.
      if (error.code === '23505') {
        return NextResponse.json(
          {
            error: 'hip_pair_already_exists',
            message:
              'A hip annotation already exists for this video+phase. ' +
              'Use UPDATE instead of POST to revise.',
            detail: error.message,
            task_type: body.task_type,
          },
          { status: 409 },
        );
      }
      // 23514 = check_violation — row coherence CHECK rejected. Surface
      // the DB message verbatim so the client sees exactly which CHECK
      // failed (DB is source of truth per spec).
      return NextResponse.json(
        {
          error: 'insert_failed',
          detail: error.message,
          pg_code: error.code,
          task_type: body.task_type,
        },
        { status: error.code === '23514' ? 400 : 500 },
      );
    }

    return NextResponse.json({ saved: data });
  }

  // ── Head-set branch (PR-7A.2) ─────────────────────────────────────
  // head_crown + chin both required. arm MUST be null. visibility
  // pinned to 'clear' (these are bone-surface landmarks always visible
  // at full-body camera coverage; if either is off-frame the workbench
  // skips entirely rather than emitting a partial row).
  // Conflict key: (video, frame, annotator) per uq_gla_head_set partial
  // unique index → UPSERT so re-submits replace in place.
  if (body.task_type === 'manual_gt_head_set') {
    if (body.arm !== null && body.arm !== undefined) {
      return NextResponse.json(
        { error: 'head_set_requires_arm_null', task_type: body.task_type },
        { status: 400 },
      );
    }

    const head_crown_x = asFiniteNumberOrNull(body.head_crown_x);
    const head_crown_y = asFiniteNumberOrNull(body.head_crown_y);
    const chin_x       = asFiniteNumberOrNull(body.chin_x);
    const chin_y       = asFiniteNumberOrNull(body.chin_y);

    if (head_crown_x === null || head_crown_y === null || chin_x === null || chin_y === null) {
      return NextResponse.json(
        { error: 'head_set_requires_both_points', task_type: body.task_type },
        { status: 400 },
      );
    }

    if (asFiniteNumberOrNull(body.shoulder_x) !== null || asFiniteNumberOrNull(body.shoulder_y) !== null ||
        asFiniteNumberOrNull(body.elbow_x)    !== null || asFiniteNumberOrNull(body.elbow_y)    !== null ||
        asFiniteNumberOrNull(body.wrist_x)    !== null || asFiniteNumberOrNull(body.wrist_y)    !== null ||
        asFiniteNumberOrNull(body.lead_hip_x) !== null || asFiniteNumberOrNull(body.lead_hip_y) !== null ||
        asFiniteNumberOrNull(body.trail_hip_x) !== null || asFiniteNumberOrNull(body.trail_hip_y) !== null ||
        asFiniteNumberOrNull(body.knee_x)     !== null || asFiniteNumberOrNull(body.knee_y)     !== null ||
        asFiniteNumberOrNull(body.ankle_x)    !== null || asFiniteNumberOrNull(body.ankle_y)    !== null) {
      return NextResponse.json(
        { error: 'head_set_rejects_other_cluster_coords', task_type: body.task_type },
        { status: 400 },
      );
    }

    if (body.visibility !== 'clear') {
      return NextResponse.json(
        { error: 'head_set_visibility_must_be_clear', task_type: body.task_type },
        { status: 400 },
      );
    }

    const headRow = {
      video_id: body.video_id,
      annotator_id: auth.user.id,
      frame_idx: body.frame_idx,
      phase: body.phase,
      task_type: 'manual_gt_head_set' as const,
      arm: null,
      visibility: 'clear' as const,
      head_crown_x, head_crown_y,
      chin_x, chin_y,
      handedness: body.handedness,
      source_app_version: body.source_app_version,
    };

    // uq_gla_head_set is a partial unique index on
    //   (video_id, frame_idx, annotator_id) WHERE task_type = 'manual_gt_head_set'
    // PostgREST upsert via onConflict can't infer a partial index
    // arbiter without an index predicate — so try INSERT first, on
    // 23505 (unique_violation) do an UPDATE in-place by the same
    // composite key. Matches the spec contract: "on conflict UPDATE
    // the typed columns".
    const headOut = await insertOrUpdateOnConflict(headRow, {
      video_id: headRow.video_id,
      frame_idx: headRow.frame_idx,
      annotator_id: headRow.annotator_id,
      task_type: 'manual_gt_head_set',
    });
    if (headOut.error) {
      return NextResponse.json(
        {
          error: headOut.error,
          detail: headOut.detail,
          pg_code: headOut.pg_code,
          task_type: body.task_type,
        },
        { status: headOut.status },
      );
    }
    return NextResponse.json({ saved: headOut.data });
  }

  // ── Leg branch (PR-7A.2) ──────────────────────────────────────────
  // knee + ankle both required, arm = lead/trail discriminator (same
  // shape as legacy arm task). Conflict key includes arm — one row per
  // (video, frame, annotator, arm) per uq_gla_leg.
  if (body.task_type === 'manual_gt_leg') {
    if (!isOneOf(body.arm, ARMS)) {
      return NextResponse.json(
        { error: 'leg_requires_arm', task_type: body.task_type },
        { status: 400 },
      );
    }

    const knee_x  = asFiniteNumberOrNull(body.knee_x);
    const knee_y  = asFiniteNumberOrNull(body.knee_y);
    const ankle_x = asFiniteNumberOrNull(body.ankle_x);
    const ankle_y = asFiniteNumberOrNull(body.ankle_y);

    if (knee_x === null || knee_y === null || ankle_x === null || ankle_y === null) {
      return NextResponse.json(
        { error: 'leg_requires_both_points', task_type: body.task_type },
        { status: 400 },
      );
    }

    if (asFiniteNumberOrNull(body.shoulder_x) !== null || asFiniteNumberOrNull(body.shoulder_y) !== null ||
        asFiniteNumberOrNull(body.elbow_x)    !== null || asFiniteNumberOrNull(body.elbow_y)    !== null ||
        asFiniteNumberOrNull(body.wrist_x)    !== null || asFiniteNumberOrNull(body.wrist_y)    !== null ||
        asFiniteNumberOrNull(body.lead_hip_x) !== null || asFiniteNumberOrNull(body.lead_hip_y) !== null ||
        asFiniteNumberOrNull(body.trail_hip_x) !== null || asFiniteNumberOrNull(body.trail_hip_y) !== null ||
        asFiniteNumberOrNull(body.head_crown_x) !== null || asFiniteNumberOrNull(body.head_crown_y) !== null ||
        asFiniteNumberOrNull(body.chin_x)     !== null || asFiniteNumberOrNull(body.chin_y)     !== null) {
      return NextResponse.json(
        { error: 'leg_rejects_other_cluster_coords', task_type: body.task_type },
        { status: 400 },
      );
    }

    if (body.visibility !== 'clear') {
      return NextResponse.json(
        { error: 'leg_visibility_must_be_clear', task_type: body.task_type },
        { status: 400 },
      );
    }

    const legRow = {
      video_id: body.video_id,
      annotator_id: auth.user.id,
      frame_idx: body.frame_idx,
      phase: body.phase,
      task_type: 'manual_gt_leg' as const,
      arm: body.arm,
      visibility: 'clear' as const,
      knee_x, knee_y,
      ankle_x, ankle_y,
      handedness: body.handedness,
      source_app_version: body.source_app_version,
    };

    // uq_gla_leg is partial on (video_id, frame_idx, annotator_id, arm)
    // WHERE task_type = 'manual_gt_leg' AND arm IS NOT NULL — same
    // PostgREST limitation as head_set, so insert-then-update-on-23505.
    const legOut = await insertOrUpdateOnConflict(legRow, {
      video_id: legRow.video_id,
      frame_idx: legRow.frame_idx,
      annotator_id: legRow.annotator_id,
      arm: legRow.arm,
      task_type: 'manual_gt_leg',
    });
    if (legOut.error) {
      return NextResponse.json(
        {
          error: legOut.error,
          detail: legOut.detail,
          pg_code: legOut.pg_code,
          task_type: body.task_type,
        },
        { status: legOut.status },
      );
    }
    return NextResponse.json({ saved: legOut.data });
  }

  // ── Arm branch (task_type in {manual_gt, correction_review, active_learning}) ──
  if (!isOneOf(body.arm, ARMS)) {
    return NextResponse.json({ error: 'invalid_arm', task_type: body.task_type }, { status: 400 });
  }

  const shoulder_x = asFiniteNumberOrNull(body.shoulder_x);
  const shoulder_y = asFiniteNumberOrNull(body.shoulder_y);
  const elbow_x    = asFiniteNumberOrNull(body.elbow_x);
  const elbow_y    = asFiniteNumberOrNull(body.elbow_y);
  const wrist_x    = asFiniteNumberOrNull(body.wrist_x);
  const wrist_y    = asFiniteNumberOrNull(body.wrist_y);

  // Reject any hip-coord columns on an arm row (DB CHECK enforces too).
  if (asFiniteNumberOrNull(body.lead_hip_x)  !== null || asFiniteNumberOrNull(body.lead_hip_y)  !== null ||
      asFiniteNumberOrNull(body.trail_hip_x) !== null || asFiniteNumberOrNull(body.trail_hip_y) !== null) {
    return NextResponse.json(
      { error: 'arm_task_rejects_hip_coords', task_type: body.task_type },
      { status: 400 },
    );
  }

  // visibility === 'clear' is the strong-GT contract — annotator
  // confidently saw all three keypoints. Anything missing here is a
  // client-side bug, not user intent. Reject hard so the bug surfaces.
  if (body.visibility === 'clear') {
    const missing =
      shoulder_x === null || shoulder_y === null ||
      elbow_x    === null || elbow_y    === null ||
      wrist_x    === null || wrist_y    === null;
    if (missing) {
      return NextResponse.json(
        { error: 'clear_visibility_requires_all_points', task_type: body.task_type },
        { status: 400 },
      );
    }
  }

  const armRow = {
    video_id: body.video_id,
    annotator_id: auth.user.id,
    frame_idx: body.frame_idx,
    phase: body.phase,
    task_type: body.task_type,
    arm: body.arm,
    visibility: body.visibility,
    shoulder_x, shoulder_y,
    elbow_x,    elbow_y,
    wrist_x,    wrist_y,
    lead_hip_x: null,  lead_hip_y: null,
    trail_hip_x: null, trail_hip_y: null,
    handedness: body.handedness,
    source_app_version: body.source_app_version,
  };

  const { data, error } = await admin
    .from('golf_landmark_annotations')
    .upsert(armRow, { onConflict: 'video_id,frame_idx,arm,annotator_id,task_type' })
    .select()
    .single();

  if (error) {
    // 23514 = check_violation; surface as 400 with the DB message so
    // the client can see exactly which row-coherence rule failed.
    return NextResponse.json(
      {
        error: 'upsert_failed',
        detail: error.message,
        pg_code: error.code,
        task_type: body.task_type,
      },
      { status: error.code === '23514' ? 400 : 500 },
    );
  }

  return NextResponse.json({ saved: data });
}
