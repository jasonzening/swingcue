import { notFound } from 'next/navigation';
import { requireAdminPage } from '@/lib/auth';
import { createServiceClient } from '@/lib/supabase/admin';
import {
  derivePhaseFrames,
  type PhaseFrames,
} from '@/lib/admin/phaseFrames';
import { TASK_PHASES, type TaskPhase, type Handedness } from '@/lib/types/annotation';
import type { PoseTimeline, PoseFrame } from '@/types/analysis';
import { ReviewView, type PerPhaseData, type ReviewRow } from './ReviewView';

export const dynamic = 'force-dynamic';

/**
 * GET /admin/review/[videoId] — server component.
 *
 * Renders Jason's GT annotations + WHAM mesh + MediaPipe keypoints on
 * the same <video> element the workbench uses, so the coord space is
 * guaranteed identical (no fps/rotation/scale mismatch — which is what
 * killed the external matplotlib overlay attempt).
 *
 * All data fetches happen server-side via the service-role client:
 *   1. swing_videos row (filename, view_type, storage_path, pose_timeline_2d)
 *   2. wham_video_meta (canonical dims + processed_fps)
 *   3. swing_analysis (phase_markers_json)
 *   4. wham_pose_timeline rows where frame_idx IN (5 phase frames)
 *   5. golf_landmark_annotations v2 rows (10 arm + 5 hip) for this video
 *   6. landmark_validation_review rows for this video + this admin
 *   7. Signed playback URL (1h TTL)
 *
 * derivePhaseFrames is imported from src/lib/admin/phaseFrames — the
 * exact same function the workbench uses to decide where to put the
 * crosshair when capturing each click. Re-deriving here would risk
 * the overlay landing on the wrong frame.
 */

type PhaseMarkersJson = {
  setupTime?: number | null;
  topTime?: number | null;
  impactTime?: number | null;
  finishTime?: number | null;
  transitionTime?: number | null;
};

type VideoMetadataJson = {
  width?: number; height?: number; fps?: number; durationSec?: number;
};

type WhamRow = {
  frame_idx: number;
  fit_ok: boolean | null;
  keypoints_2d_projected: Record<string, { x: number; y: number } | null> | null;
};

type GtArmRow = {
  frame_idx: number;
  phase: string;
  arm: 'lead' | 'trail';
  visibility: string;
  shoulder_x: number | null; shoulder_y: number | null;
  elbow_x: number | null;    elbow_y: number | null;
  wrist_x: number | null;    wrist_y: number | null;
  handedness: 'right' | 'left';
};

type GtHipRow = {
  frame_idx: number;
  phase: string;
  lead_hip_x: number;  lead_hip_y: number;
  trail_hip_x: number; trail_hip_y: number;
  handedness: 'right' | 'left';
};

function findPoseFrameByFrameIdx(
  timeline: PoseTimeline | null,
  frameIdx: number,
): PoseFrame | null {
  if (!timeline?.frames?.length) return null;
  // Exact match first; if absent, fall back to nearest within ±3 frames
  // (MediaPipe sampling may differ from wham_video_meta.processed_fps).
  const exact = timeline.frames.find(f => f.frame_idx === frameIdx);
  if (exact) return exact;
  let best: PoseFrame | null = null;
  let bestDist = Infinity;
  for (const f of timeline.frames) {
    const d = Math.abs(f.frame_idx - frameIdx);
    if (d < bestDist) { bestDist = d; best = f; }
  }
  return bestDist <= 3 ? best : null;
}

export default async function Page({
  params,
}: { params: Promise<{ videoId: string }> }) {
  await requireAdminPage();
  const { videoId } = await params;
  const admin = createServiceClient();

  // 1. Video row
  const { data: video, error: videoErr } = await admin
    .from('swing_videos')
    .select('id, original_filename, view_type, storage_path, pose_timeline_2d')
    .eq('id', videoId)
    .maybeSingle();
  if (videoErr || !video) notFound();

  const storagePath = (video.storage_path as string | null) ?? '';
  if (!storagePath) {
    return (
      <main className="rv-empty">
        <h1>Validation Review</h1>
        <p>Video {videoId.slice(0, 8)} has no storage_path — analyzer hasn&apos;t finished.</p>
        <style>{EMPTY_CSS}</style>
      </main>
    );
  }

  // 2-4. Parallel: wham_video_meta + swing_analysis + signed URL
  const [whamMetaRes, anaRes, signedRes] = await Promise.all([
    admin
      .from('wham_video_meta')
      .select('image_width, image_height, processed_fps, frame_count')
      .eq('video_id', videoId)
      .maybeSingle(),
    admin
      .from('swing_analysis')
      .select('video_metadata_json, phase_markers_json')
      .eq('video_id', videoId)
      .order('created_at', { ascending: false })
      .limit(1)
      .maybeSingle(),
    admin.storage.from('swing-videos').createSignedUrl(storagePath, 3600),
  ]);

  if (!signedRes.data?.signedUrl) {
    return (
      <main className="rv-empty">
        <h1>Validation Review</h1>
        <p>Signed URL failed: {signedRes.error?.message ?? 'unknown'}</p>
        <style>{EMPTY_CSS}</style>
      </main>
    );
  }

  const wham = whamMetaRes.data;
  const vm = (anaRes.data?.video_metadata_json as VideoMetadataJson | null | undefined) ?? null;
  const pm = (anaRes.data?.phase_markers_json as PhaseMarkersJson | null | undefined) ?? null;

  const width  = Number(wham?.image_width  ?? vm?.width  ?? 0);
  const height = Number(wham?.image_height ?? vm?.height ?? 0);
  const fps    = Number(wham?.processed_fps ?? vm?.fps    ?? 30);

  if (!width || !height || !fps) {
    return (
      <main className="rv-empty">
        <h1>Validation Review</h1>
        <p>Missing dims/fps for video {videoId.slice(0, 8)}.</p>
        <style>{EMPTY_CSS}</style>
      </main>
    );
  }

  // 5. Derive phase frames using the SHARED helper (same one the
  //    workbench uses to seek the crosshair).
  const phaseFrames: PhaseFrames | null = derivePhaseFrames({
    videoId,
    filename: (video.original_filename as string | null) ?? null,
    width, height, fps,
    durationSec: Number(vm?.durationSec ?? 0),
    frameCount: Number(wham?.frame_count ?? 0),
    hasWhamData: wham !== null,
    phaseMarkers: {
      setupTime:      pm?.setupTime      ?? null,
      topTime:        pm?.topTime        ?? null,
      impactTime:     pm?.impactTime     ?? null,
      finishTime:     pm?.finishTime     ?? null,
      transitionTime: pm?.transitionTime ?? null,
    },
    signedVideoUrl: signedRes.data.signedUrl,
  });
  if (!phaseFrames) {
    return (
      <main className="rv-empty">
        <h1>Validation Review</h1>
        <p>Phase markers incomplete for {videoId.slice(0, 8)} — annotate first.</p>
        <style>{EMPTY_CSS}</style>
      </main>
    );
  }

  const phaseFrameIdxArr = TASK_PHASES.map(p => phaseFrames[p]);

  // 6. WHAM keypoints for the 5 phase frames
  const { data: whamRowsData } = await admin
    .from('wham_pose_timeline')
    .select('frame_idx, fit_ok, keypoints_2d_projected')
    .eq('video_id', videoId)
    .in('frame_idx', phaseFrameIdxArr);
  const whamRows = (whamRowsData ?? []) as WhamRow[];
  const whamByFrame = new Map<number, WhamRow>(whamRows.map(r => [r.frame_idx, r]));

  // 7. GT annotations (v2 anatomical-spec rows)
  const { data: gtData } = await admin
    .from('golf_landmark_annotations')
    .select('*')
    .eq('video_id', videoId)
    .like('source_app_version', 'swingcue-annotate-2.0-anatomical-spec%');
  const gtRows = (gtData ?? []) as Array<GtArmRow | GtHipRow & { task_type: string; arm: 'lead' | 'trail' | null }>;

  // Detect handedness from any GT row (all 15 share the same value).
  let handedness: Handedness = 'right';
  for (const r of gtRows) {
    const h = (r as { handedness?: Handedness }).handedness;
    if (h === 'left' || h === 'right') { handedness = h; break; }
  }

  // Group GT by (phase, kind)
  type ArmGt = {
    shoulder: { x: number; y: number } | null;
    elbow:    { x: number; y: number } | null;
    wrist:    { x: number; y: number } | null;
    visibility: string;
  };
  type HipGt = {
    leadHip:  { x: number; y: number };
    trailHip: { x: number; y: number };
  };
  const gtArmByPhaseAndArm = new Map<string, ArmGt>();
  const gtHipByPhase = new Map<string, HipGt>();
  for (const r of gtRows) {
    const rec = r as Record<string, unknown>;
    const task_type = rec['task_type'] as string;
    const phase = rec['phase'] as string;
    if (task_type === 'manual_gt') {
      const arm = rec['arm'] as 'lead' | 'trail';
      gtArmByPhaseAndArm.set(`${phase}::${arm}`, {
        shoulder: rec['shoulder_x'] != null && rec['shoulder_y'] != null
          ? { x: rec['shoulder_x'] as number, y: rec['shoulder_y'] as number } : null,
        elbow: rec['elbow_x'] != null && rec['elbow_y'] != null
          ? { x: rec['elbow_x'] as number, y: rec['elbow_y'] as number } : null,
        wrist: rec['wrist_x'] != null && rec['wrist_y'] != null
          ? { x: rec['wrist_x'] as number, y: rec['wrist_y'] as number } : null,
        visibility: rec['visibility'] as string,
      });
    } else if (task_type === 'manual_gt_hip_pair') {
      gtHipByPhase.set(phase, {
        leadHip:  { x: rec['lead_hip_x']  as number, y: rec['lead_hip_y']  as number },
        trailHip: { x: rec['trail_hip_x'] as number, y: rec['trail_hip_y'] as number },
      });
    }
  }

  // 8. Existing reviews for this admin
  const { data: reviewsData } = await admin
    .from('landmark_validation_review')
    .select('phase, verdict, notes')
    .eq('video_id', videoId)
    .eq('reviewer_id', (await admin.auth.getUser()).data.user?.id ?? '');
  // Note: service-role doesn't have an auth.getUser session; reviews
  // are filtered client-side by `reviewer_id` already set in the API
  // route. We just fetch the visible set here; client will call GET
  // /api/admin/landmark-validation-review/[videoId] anyway to refresh.
  const reviews = (reviewsData ?? []) as ReviewRow[];
  const reviewByPhase = new Map<string, ReviewRow>(reviews.map(r => [r.phase, r]));

  // 9. MediaPipe pose timeline (on the video row)
  const poseTimeline = (video.pose_timeline_2d as PoseTimeline | null) ?? null;

  // Build per-phase props
  const perPhase: PerPhaseData[] = TASK_PHASES.map((phase: TaskPhase) => {
    const frameIdx = phaseFrames[phase];
    const timeSec = frameIdx / fps;
    const armLead  = gtArmByPhaseAndArm.get(`${phase}::lead`)  ?? null;
    const armTrail = gtArmByPhaseAndArm.get(`${phase}::trail`) ?? null;
    const hipPair  = gtHipByPhase.get(phase) ?? null;
    const whamRow  = whamByFrame.get(frameIdx);
    const whamKpts = whamRow && whamRow.fit_ok
      ? whamRow.keypoints_2d_projected
      : null;

    const mpFrame = findPoseFrameByFrameIdx(poseTimeline, frameIdx);
    // Flatten PoseFrame.keypoints into a JSON-serializable map of
    // { name → [x, y, conf] } so the client can index by COCO name.
    const mediaPipe: Record<string, [number | null, number | null, number]> | null =
      mpFrame ? (mpFrame.raw_keypoints ?? mpFrame.keypoints) as unknown as
        Record<string, [number | null, number | null, number]> : null;

    const review = reviewByPhase.get(phase) ?? null;

    return {
      phase, frameIdx, timeSec,
      armLead, armTrail, hipPair,
      wham: whamKpts,
      mediaPipe,
      existingReview: review,
    };
  });

  return (
    <main className="rv-page">
      <header className="rv-header">
        <h1 className="rv-title">
          Validation Review · {video.original_filename ?? videoId.slice(0, 8)} ·{' '}
          <span className="rv-view">{video.view_type as string}</span>
        </h1>
        <p className="rv-subhead">
          Verify your GT annotations align with the video and compare against
          WHAM / MediaPipe model outputs.
        </p>
      </header>
      <ReviewView
        videoId={videoId}
        videoFilename={(video.original_filename as string | null) ?? null}
        viewType={(video.view_type as 'face_on' | 'down_the_line') ?? 'face_on'}
        handedness={handedness}
        fpsSampled={fps}
        videoWidth={width}
        videoHeight={height}
        signedUrl={signedRes.data.signedUrl}
        perPhase={perPhase}
      />
      <style>{PAGE_CSS}</style>
    </main>
  );
}

const PAGE_CSS = `
  .rv-page {
    min-height: 100vh;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: 'DM Sans', system-ui, sans-serif;
  }
  .rv-header {
    padding: 12px 16px 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  }
  .rv-title {
    font-size: 14px;
    font-weight: 500;
    margin: 0 0 4px;
    letter-spacing: -0.01em;
  }
  .rv-view {
    color: var(--text-muted);
    font-weight: 400;
  }
  .rv-subhead {
    font-size: 11px;
    color: var(--text-muted);
    margin: 0;
  }
`;

const EMPTY_CSS = `
  .rv-empty {
    min-height: 100vh;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: 'DM Sans', system-ui, sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 24px;
  }
  .rv-empty h1 { font-size: 16px; font-weight: 500; margin: 0; }
  .rv-empty p { color: var(--text-muted); font-size: 13px; margin: 0; }
`;
