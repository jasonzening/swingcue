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
  'manual_gt', 'correction_review', 'active_learning',
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

/**
 * POST /api/admin/annotations
 *
 * Upserts an AnnotationRecord. Server fills annotator_id (from session)
 * and annotated_at (server clock). Unique key:
 *   (video_id, frame_idx, arm, annotator_id, task_type)
 * so re-clicking the same task overwrites the prior save instead of
 * creating duplicates.
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
  if (!isOneOf(body.arm, ARMS)) {
    return NextResponse.json({ error: 'invalid_arm' }, { status: 400 });
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

  const shoulder_x = asFiniteNumberOrNull(body.shoulder_x);
  const shoulder_y = asFiniteNumberOrNull(body.shoulder_y);
  const elbow_x    = asFiniteNumberOrNull(body.elbow_x);
  const elbow_y    = asFiniteNumberOrNull(body.elbow_y);
  const wrist_x    = asFiniteNumberOrNull(body.wrist_x);
  const wrist_y    = asFiniteNumberOrNull(body.wrist_y);

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
        { error: 'clear_visibility_requires_all_points' },
        { status: 400 },
      );
    }
  }

  const row = {
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
    handedness: body.handedness,
    source_app_version: body.source_app_version,
  };

  const admin = createServiceClient();
  const { data, error } = await admin
    .from('golf_landmark_annotations')
    .upsert(row, { onConflict: 'video_id,frame_idx,arm,annotator_id,task_type' })
    .select()
    .single();

  if (error) {
    return NextResponse.json(
      { error: 'upsert_failed', detail: error.message },
      { status: 500 },
    );
  }

  return NextResponse.json({ saved: data });
}
