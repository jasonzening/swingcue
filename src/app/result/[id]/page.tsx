'use client';

/**
 * Result page — 满屏 Interactive Swing Player
 *
 * 渲染策略：
 * 1. 优先使用 keypoint_timeline_json 生成 dense overlay（每帧一个 overlay → 连续追踪）
 * 2. 如果没有 keypoint 数据，fallback 到存储的 overlay_timeline_json（5帧快照）
 */

import { useState, useEffect, useMemo } from 'react';
import { useRouter, useParams, useSearchParams } from 'next/navigation';
import { createClient } from '@/lib/supabase/client';
import { SwingPlayer } from '@/components/SwingPlayer';
import { generateDenseOverlayTimeline } from '@/lib/overlay/templates';
import { generateSparsePhaseOverlayTimeline } from '@/lib/overlay/sparsePhaseOverlay';
import { fetchPoseRows } from '@/lib/sam3d/poseFetch';
import type { MainIssueType, PhaseMarkers, VideoMetadata, OverlayTimeline, KeypointFrame, PoseTimeline } from '@/types/analysis';
import { ISSUE_LABELS } from '@/types/analysis';
// PR-5.8A: render-time coaching-anchor expansion (URL-tunable).
import { readExpandFactorsFromURL } from '@/lib/skeleton/coachingAnchors';
// PR-8d.1: WHAM bone-center skeleton type (the result page assembles
// the timeline object from wham_video_meta + wham_pose_timeline rows
// and passes it down to SwingPlayer).
import type { WhamPoseTimelineForOverlay } from '@/components/WhamSkeletonOverlay';

// ────────────────────────────────────────────────────────────────────
// PR-8d.0: wham_status state machine.
//
// Frontend reads swing_analysis.video_metadata_json.wham_status set by
// PR-8c.1 Railway BackgroundTask + PR-8c.3/8c.4 stage-aware failure
// writers. Five UI branches:
//
//   absent                 — legacy row (no wham_status key). Render
//                            current placeholder UI unchanged.
//   processing             — WHAM in flight. Show 3D-build screen with
//                            ETA + polling. Per R2: never frontend-
//                            mark failed; backend is sole writer.
//   ready                  — WHAM finished successfully. Current full
//                            result UI.
//   failed_preprocessing   — PR-8c.4 reject (duration < 3s, multi-scene).
//                            Show wham_error_message verbatim (these
//                            are user-friendly strings by construction).
//   failed_other           — Any other failed stage (dispatch / download /
//                            slam_init / inference / postprocess / timeout
//                            / unknown / stage absent). Show GENERIC
//                            "Analysis failed. Please retry." NEVER
//                            expose wham_error_message (may contain
//                            full Python tracebacks per R6).
// ────────────────────────────────────────────────────────────────────
type WhamUiState =
  | { kind: 'absent' }
  | { kind: 'processing'; startedAt: number; expectedSeconds: number }
  | { kind: 'ready' }
  | { kind: 'failed_preprocessing'; userMessage: string }
  | { kind: 'failed_other'; stage?: string };

function classifyWhamState(vmj: unknown): WhamUiState {
  if (!vmj || typeof vmj !== 'object') return { kind: 'absent' };
  const o = vmj as Record<string, unknown>;
  // R4: legacy = wham_status KEY MISSING (not null). Use `in` operator.
  if (!('wham_status' in o)) return { kind: 'absent' };
  const ws = o.wham_status;
  if (ws === 'ready') return { kind: 'ready' };
  if (ws === 'processing') {
    const startedAtStr = typeof o.wham_started_at === 'string' ? o.wham_started_at : null;
    const startedAt = startedAtStr ? new Date(startedAtStr).getTime() : Date.now();
    const expectedSeconds = typeof o.wham_expected_completion_seconds === 'number'
      ? o.wham_expected_completion_seconds
      : 60;
    return { kind: 'processing', startedAt, expectedSeconds };
  }
  if (ws === 'failed') {
    const stage = typeof o.wham_failure_stage === 'string' ? o.wham_failure_stage : undefined;
    if (stage === 'preprocessing') {
      // PR-8c.4 preprocessing messages are user-friendly by construction
      // ("Video too short for analysis (2.6s). Please upload at least
      // 3 seconds..."). Safe to show verbatim per R6.
      const msg = typeof o.wham_error_message === 'string'
        ? o.wham_error_message
        : 'Your video could not be analyzed. Please re-upload.';
      return { kind: 'failed_preprocessing', userMessage: msg };
    }
    return { kind: 'failed_other', stage };
  }
  // Unknown wham_status value — treat as legacy (don't override UI).
  return { kind: 'absent' };
}

// PR-8d.0 R6: short non-PII hash for failed_other support reference.
// Trivial hash; not crypto. Just a 6-char tag the user can read out so
// support can grep wham_error_message in logs.
function shortHash(input: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  // PR-8d.2 part 2 Q4: kebab-split 3+3 for readability — `a7f3-k2`.
  const hex = h.toString(16).padStart(8, '0').slice(0, 6);
  return `${hex.slice(0, 3)}-${hex.slice(3)}`;
}

export default function ResultPage() {
  const router = useRouter();
  const params = useParams();
  const searchParams = useSearchParams();
  const videoId = params.id as string;

  // PR-5.8A: parse ?shoulderExpand=...&hipExpand=... once per mount.
  // Falls back to defaults (0.40 / 0.25) when missing or out-of-range.
  const expandFactors = useMemo(
    () => readExpandFactorsFromURL(new URLSearchParams(searchParams.toString())),
    [searchParams],
  );

  // PR-5.9 Task 5: ?debug=pose enables the raw-vs-final dot overlay in
  // SkeletonOverlay. Hidden behind URL param — no production UI change.
  const debugMode = searchParams.get('debug') === 'pose' ? 'pose' : undefined;

  // PR-7c-frontend-v10: side-by-side tuning layout. When ?tune=anchors
  // active AND viewport >= 1100px, expand the .page max-width from
  // 430px → 820px so the SwingPlayer's .sp-row flex layout (video +
  // panel side-by-side) actually has room. Below 1100px, panel
  // falls back to absolute overlay and .page stays 430px.
  const isTuneMode = searchParams.get('tune') === 'anchors';
  const [isWideViewport, setIsWideViewport] = useState(false);
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const mq = window.matchMedia('(min-width: 1100px)');
    setIsWideViewport(mq.matches);
    const handler = (e: MediaQueryListEvent) => setIsWideViewport(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);
  const tuneSideBySide = isTuneMode && isWideViewport;

  const [state, setState] = useState<'loading' | 'ready' | 'error'>('loading');
  const [videoUrl, setVideoUrl] = useState('');
  const [issue, setIssue] = useState<MainIssueType>('early_extension');
  const [cue, setCue] = useState('');
  const [phases, setPhases] = useState<PhaseMarkers>({
    setupTime: 0, topTime: 0.5, transitionTime: 0.65, impactTime: 0.75, finishTime: 0.9,
  });
  const [meta, setMeta] = useState<VideoMetadata>({ durationSec: 3, fps: 30, width: 640, height: 360 });
  const [overlayTimeline, setOverlayTimeline] = useState<OverlayTimeline | null>(null);
  const [dataSource, setDataSource] = useState<string>('unknown');
  // PR-4: 17-COCO frame-level timeline (null when video predates PR-4
  // or the analyzer's validate_timeline gate rejected the data).
  const [poseTimeline, setPoseTimeline] = useState<PoseTimeline | null>(null);

  // PR-8d.0: wham_status branch state (independent of the result-page
  // load state machine above). Set from swing_analysis.video_metadata_json
  // after fetch + polled at 2s exp backoff (cap 8s) while in 'processing'.
  const [whamUiState, setWhamUiState] = useState<WhamUiState>({ kind: 'absent' });

  // PR-8d.1: trusted WHAM bone-center skeleton assembled from
  // wham_video_meta + wham_pose_timeline rows. Fetched lazily when
  // whamUiState.kind === 'ready'.
  //   undefined → not yet attempted to fetch
  //   null       → fetched but rows missing (R5 preparing fallback)
  //   object     → fetched and present (pass to SwingPlayer)
  const [whamTimeline, setWhamTimeline] = useState<
    WhamPoseTimelineForOverlay | null | undefined
  >(undefined);

  useEffect(() => {
    async function load() {
      const supabase = createClient();
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) { router.replace('/sign-in'); return; }

      const { data: vid } = await supabase
        .from('swing_videos')
        .select('*')
        .eq('id', videoId)
        .eq('user_id', user.id)
        .single();

      if (!vid || vid.status !== 'completed') { setState('error'); return; }

      const { data: signed } = await supabase.storage
        .from('swing-videos')
        .createSignedUrl(vid.storage_path, 3600);
      if (signed?.signedUrl) setVideoUrl(signed.signedUrl);

      // PR-4: hydrate the 17-COCO pose timeline if present on the row.
      // Inline cast pattern matches the rest of this file (no canonical
      // SwingVideoRow type yet — see PR-4_DESIGN.md §D).
      const pt = (vid as { pose_timeline_2d?: PoseTimeline | null }).pose_timeline_2d ?? null;
      if (pt && Array.isArray(pt.frames) && pt.frames.length > 0) {
        setPoseTimeline(pt);
      }

      const { data: ana } = await supabase
        .from('swing_analysis')
        .select('*')
        .eq('video_id', videoId)
        .single();

      if (!ana) { setState('error'); return; }

      const issueType = (ana.issue_type as MainIssueType) ?? 'early_extension';
      setIssue(issueType);
      setCue(ana.cue_text ?? '');

      // PR-5.9 Task 1: read native fps/width/height from video_metadata_json
      // (already populated by Python analyzer at upload time, per
      // docs/PR-5.9_AUDIT.md §7). Fallbacks preserve the previous
      // hardcoded values when the column is missing or partial.
      const vmJson = ana.video_metadata_json as {
        durationSec?: number;
        fps?: number;
        width?: number;
        height?: number;
        dataSource?: string;
      } | null;
      const dur = vmJson?.durationSec ?? 3;
      const source = vmJson?.dataSource ?? 'stub';
      setDataSource(source);

      // PR-8d.0: classify wham state from video_metadata_json.
      // Updates trigger the polling effect below if we land on 'processing'.
      setWhamUiState(classifyWhamState(ana.video_metadata_json));

      const pm: PhaseMarkers = (ana.phase_markers_json as PhaseMarkers | null) ?? {
        setupTime: 0,
        topTime: dur * 0.50,
        transitionTime: dur * 0.62,
        impactTime: dur * 0.75,
        finishTime: dur * 0.92,
      };
      setPhases(pm);

      const vm: VideoMetadata = {
        durationSec: dur,
        fps:    vmJson?.fps    ?? 30,
        width:  vmJson?.width  ?? 640,
        height: vmJson?.height ?? 360,
      };
      setMeta(vm);

      const viewType = (vid.view_type as 'face_on' | 'down_the_line') ?? 'face_on';

      // ── PATH 0: pose_3d_phases → sparse 5-frame timeline (YOLO ▶ SAM) ──
      //   Preferred path for any video analyzed after PR-2B. RLS scopes
      //   the query to this user; failure / empty / RLS-reject all collapse
      //   to [] so the cascade below proceeds.
      //   The per-row disc builder prefers YOLO11-pose anchors over SAM
      //   materialised columns; the badge reflects whichever source has
      //   data on any row.
      const poseRows = await fetchPoseRows(videoId);
      const hasYolo = poseRows.some(
        r => r.yolo_keypoints_2d !== null &&
             r.image_width > 0 && r.image_height > 0,
      );
      const hasSam = poseRows.some(
        r => r.fal_status === 'completed' &&
             r.image_width > 0 && r.image_height > 0,
      );
      if (hasYolo || hasSam) {
        const sparseOlt = generateSparsePhaseOverlayTimeline({
          poseRows,
          phaseMarkers: pm,
        });
        if (sparseOlt.frames.length > 0) {
          setOverlayTimeline(sparseOlt);
          setDataSource(hasYolo ? 'yolo' : 'sam3d');
          setState('ready');
          return;
        }
      }

      // ── PATH A: keypoint 数据存在 → dense overlay（连续追踪）──
      const kpJson = ana.keypoint_timeline_json as { frames?: KeypointFrame[] } | null;
      const kpFrames = kpJson?.frames;

      if (kpFrames && kpFrames.length >= 3) {
        const denseOlt = generateDenseOverlayTimeline({
          keypointFrames: kpFrames,
          phaseMarkers: pm,
          issue: issueType,
          viewType,
          duration: dur,
        });
        if (denseOlt.frames.length > 0) {
          setOverlayTimeline(denseOlt);
          setState('ready');
          return;
        }
      }

      // ── PATH B: fallback → 存储的5帧快照 ──
      const storedOverlay = ana.overlay_timeline_json as OverlayTimeline | null;
      if (!storedOverlay?.frames?.length) { setState('error'); return; }
      setOverlayTimeline(storedOverlay);
      setState('ready');
    }
    load();
  }, [videoId, router]);

  // PR-8d.1: when wham_status flips to 'ready', fetch the per-frame
  // skeleton from wham_video_meta + wham_pose_timeline. Result page
  // then passes whamTimeline into SwingPlayer; SwingPlayer renders
  // WhamSkeletonOverlay instead of MediaPipe-derived overlays.
  useEffect(() => {
    if (whamUiState.kind !== 'ready') return;
    let cancelled = false;
    (async () => {
      const supabase = createClient();
      const [metaRes, rowsRes] = await Promise.all([
        supabase.from('wham_video_meta')
          .select('image_width,image_height,processed_fps')
          .eq('video_id', videoId).maybeSingle(),
        supabase.from('wham_pose_timeline')
          .select('frame_idx,frame_timestamp_ms,fit_ok,keypoints_2d_projected')
          .eq('video_id', videoId)
          .order('frame_idx', { ascending: true })
          .limit(1000),
      ]);
      if (cancelled) return;
      const meta = metaRes.data;
      const rows = rowsRes.data;
      if (!meta || !rows || rows.length === 0) {
        // R5: ready but data missing — do NOT fall back to MediaPipe.
        console.warn('[result] wham_status=ready but data missing', {
          metaPresent: !!meta,
          rowCount: rows?.length ?? 0,
          metaErr: metaRes.error,
          rowsErr: rowsRes.error,
        });
        setWhamTimeline(null);
        return;
      }
      setWhamTimeline({
        image_width: meta.image_width,
        image_height: meta.image_height,
        processed_fps: meta.processed_fps,
        frames: rows.map((r) => ({
          frame_idx:              r.frame_idx as number,
          frame_timestamp_ms:     (r.frame_timestamp_ms as number | null) ?? null,
          fit_ok:                 (r.fit_ok as boolean | null) ?? false,
          keypoints_2d_projected: r.keypoints_2d_projected as
            Record<string, { x: number; y: number } | null> | null,
        })),
      });
    })();
    return () => { cancelled = true; };
  }, [whamUiState.kind, videoId]);

  // PR-8d.0 R1: poll swing_analysis.video_metadata_json every 2s → 4s →
  // 8s (cap) while wham_status='processing'. Stops on terminal state
  // (ready / failed_* / absent) OR unmount. Never polls legacy rows
  // (whamUiState.kind === 'absent' from initial load = no key).
  useEffect(() => {
    if (whamUiState.kind !== 'processing') return;
    const supabase = createClient();
    let cancelled = false;
    let timeout: ReturnType<typeof setTimeout> | null = null;
    let delay = 2000;

    const poll = async () => {
      try {
        const { data } = await supabase
          .from('swing_analysis')
          .select('video_metadata_json')
          .eq('video_id', videoId)
          .single();
        if (cancelled) return;
        const next = classifyWhamState(data?.video_metadata_json);
        setWhamUiState(next);
        if (next.kind !== 'processing') {
          // Terminal — cleanup will fire via deps change.
          return;
        }
      } catch (err) {
        console.error('[result/wham-poll] error:', err);
        // fall through to re-schedule with the same backoff.
      }
      delay = Math.min(delay * 2, 8000);
      if (!cancelled) timeout = setTimeout(poll, delay);
    };

    timeout = setTimeout(poll, delay);
    return () => {
      cancelled = true;
      if (timeout) clearTimeout(timeout);
    };
  }, [whamUiState.kind, videoId]);

  if (state === 'loading') {
    return (
      <div className="page-center">
        <div className="spinner" />
        <p className="load-txt">Loading your swing…</p>
        <style>{css}</style>
      </div>
    );
  }

  if (state === 'error' || !overlayTimeline) {
    return (
      <div className="page-center">
        <p className="err-txt">Result not found or still processing.</p>
        <button className="btn-back" onClick={() => router.push('/upload')}>← Back to upload</button>
        <style>{css}</style>
      </div>
    );
  }

  // ── PR-8d.0: wham_status branch (overrides full result UI) ────────
  if (whamUiState.kind === 'processing') {
    return (
      <ProcessingScreen
        startedAt={whamUiState.startedAt}
        expectedSeconds={whamUiState.expectedSeconds}
        onBack={() => router.push('/history')}
      />
    );
  }
  if (whamUiState.kind === 'failed_preprocessing') {
    // PR-8d.2 part 2 Q9: verbose userMessage preserved verbatim
    // (e.g. "Video too short (2.6s) — needs at least 3 seconds").
    // PR-8d.2 part 2 2A: + 3-bullet help row so the user knows
    // what "Try again" should actually look like.
    return (
      <FailedScreen
        title="Couldn't analyze this video"
        message={whamUiState.userMessage}
        helpHeader="What to try:"
        helpBullets={[
          'Record a longer clip (4-8 seconds works best)',
          'Use a single continuous take',
          'Avoid scene cuts or zoom changes',
        ]}
        onRetry={() => router.push('/upload')}
        onBack={() => router.push('/history')}
      />
    );
  }
  if (whamUiState.kind === 'failed_other') {
    // R6: NEVER expose wham_error_message — may contain Python traceback.
    // Show generic message; surface only a stable short hash so support
    // can grep logs for this specific failure.
    //
    // PR-8d.2 part 2 Q3/Q4: simplified to honest copy. Earlier "we're
    // already looking into it" + "this isn't your video" + "send the
    // error reference" lines were [PROMISE] — implied monitoring +
    // support channel that don't exist today. Reference hash kept
    // standalone, no instruction text. Future support-flow PR can
    // layer in the instruction when a real channel exists.
    const ref = shortHash(`${videoId}|${whamUiState.stage ?? 'unknown'}`);
    return (
      <FailedScreen
        title="Analysis failed"
        message="Something went wrong. Please try uploading again."
        onRetry={() => router.push('/upload')}
        onBack={() => router.push('/history')}
        supportRef={ref}
      />
    );
  }

  // PR-8d.1 R5: wham_status='ready' but the timeline rows are missing
  // (rare race / Modal write incomplete). Do NOT silently fall back to
  // MediaPipe — show a soft "preparing" message and let the user refresh.
  if (whamUiState.kind === 'ready' && whamTimeline === null) {
    return (
      <div className="page-center wham-screen">
        <button className="btn-corner-back" onClick={() => router.push('/history')}>← History</button>
        <div className="spinner" />
        <h2 className="wham-title">Analysis data is still preparing</h2>
        <p className="wham-detail">Please refresh in a moment.</p>
        <button className="wham-retry-btn" onClick={() => window.location.reload()}>Refresh</button>
        <style>{css}</style>
      </div>
    );
  }
  // ready but fetch in flight — show a lightweight spinner instead of
  // briefly flashing the MediaPipe placeholder UI.
  if (whamUiState.kind === 'ready' && whamTimeline === undefined) {
    return (
      <div className="page-center">
        <div className="spinner" />
        <p className="load-txt">Loading 3D analysis…</p>
        <style>{css}</style>
      </div>
    );
  }
  // kind === 'ready' (with whamTimeline present) OR 'absent'
  // → fall through to existing UI; SwingPlayer gets whamPoseTimeline
  //   when present and renders the WHAM skeleton path. Otherwise the
  //   legacy placeholder UI shows.

  const issueLabel = ISSUE_LABELS[issue] ?? issue;

  return (
    <div className={`page ${tuneSideBySide ? 'page-tune-wide' : ''}`}>
      <header className="hdr">
        <button className="btn-hdr-back" onClick={() => router.push('/history')}>←</button>
        <span className="hdr-logo">SwingCue</span>
        <button className="btn-new" onClick={() => router.push('/upload')}>+ New</button>
      </header>

      {videoUrl ? (
        <SwingPlayer
          videoId={videoId}
          videoUrl={videoUrl}
          timeline={overlayTimeline}
          phases={phases}
          duration={meta.durationSec}
          dataSource={dataSource}
          poseTimeline={poseTimeline}
          shoulderExpand={expandFactors.shoulder}
          hipExpand={expandFactors.hip}
          debugMode={debugMode}
          // PR-8d.1: pass the WHAM timeline when ready + present. When
          // SwingPlayer sees this prop it renders WhamSkeletonOverlay
          // instead of MediaPipe-derived overlays.
          whamPoseTimeline={
            whamUiState.kind === 'ready' && whamTimeline ? whamTimeline : null
          }
        />
      ) : (
        <div className="no-vid"><p>Video loading…</p></div>
      )}

      {/* PR-8d.2 part 1: surface the known WHAM body-width limitation
          (PR-8h.0 audit closure path 1) directly under the skeleton on
          the ready branch. Small + subtle so it informs without
          interrupting. Other branches keep their own UI — never shown
          on processing / failed / absent. Aligned with the coaching-bar
          gate below so only one of {disclaimer, cue} occupies this
          vertical slot at any time. */}
      {whamUiState.kind === 'ready' && whamTimeline && (
        <p className="wham-disclaimer" role="note">
          Body alignment is approximate — used as a coaching anchor.
        </p>
      )}

      {/* PR-8d.1 R4: hide the static "Head Movement / Keep your head
          centered" placeholder coaching cue when WHAM trusted analysis
          is rendering. Leaving the cue while showing real WHAM skeleton
          would falsely suggest the cue is WHAM-derived insight; it's
          actually a template carried over from the MediaPipe era.
          Real WHAM-derived coaching is PR-8d.2+ territory. */}
      {whamUiState.kind !== 'ready' && (
        <div className="coaching-bar">
          <div className="issue-row">
            <span className="issue-dot">⚡</span>
            <span className="issue-text">{issueLabel}</span>
          </div>
          <div className="cue-row">
            <span className="cue-quote">&ldquo;{cue}&rdquo;</span>
          </div>
        </div>
      )}

      <style>{css}</style>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────
// PR-8d.0 screens — processing + failed (preprocessing | other).
// ────────────────────────────────────────────────────────────────────

function ProcessingScreen({
  startedAt,
  expectedSeconds,
  onBack,
}: {
  startedAt: number;
  expectedSeconds: number;
  onBack: () => void;
}) {
  // R2 ETA logic — re-render every 1s to update the elapsed counter.
  // Frontend NEVER marks the row failed; we just adjust the message.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const tick = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(tick);
  }, []);
  const elapsedSec = Math.max(0, Math.floor((now - startedAt) / 1000));
  const expiredAt = expectedSeconds + 30;
  const isOver = elapsedSec > expiredAt;
  const isVeryOver = elapsedSec > 300;

  let headline = 'Building your 3D analysis';
  let detail = `Usually takes about ${expectedSeconds} seconds.`;
  if (isVeryOver) {
    headline = 'Analysis is taking too long';
    // PR-8d.2 part 2 Q2: honest [SHIPPING NOW] rewrite. Previous copy
    // "We're looking into it. You can retry from the upload page."
    // implied real-time monitoring that doesn't exist yet.
    detail = 'Still processing — this is taking longer than expected. You can wait or try a different upload.';
  } else if (isOver) {
    headline = 'Still analyzing…';
    detail = 'Taking longer than expected — hang tight.';
  }

  // PR-8d.2 part 2 R3 + Q8: client-side stage bucketization by
  // elapsed / expectedSeconds fraction. Real wham_status flips
  // override this animation at the polling layer above. Once
  // fraction >= 0.95, label freezes on `Finalizing` indefinitely
  // (Q8 Option A) — never "Almost done", never "Step 6". Past
  // 300s elapsed, hide the stage label entirely so the screen is
  // just headline + detail + counter.
  const fraction = expectedSeconds > 0 ? elapsedSec / expectedSeconds : 0;
  const stageLabel: string | null = isVeryOver
    ? null
    : fraction >= 0.95 ? 'Step 5 of 5 · Finalizing'
    : fraction >= 0.50 ? 'Step 4 of 5 · Building 3D'
    : fraction >= 0.15 ? 'Step 3 of 5 · Detecting pose'
    : fraction >= 0.05 ? 'Step 2 of 5 · Preparing'
    : 'Step 1 of 5 · Uploaded';

  return (
    <div className="page-center wham-screen">
      <button className="btn-corner-back" onClick={onBack}>← History</button>
      <div className="wham-anim">
        <div className="wham-anim-ring" />
        <div className="wham-anim-core" />
      </div>
      <h2 className="wham-title">{headline}</h2>
      {stageLabel && <p className="wham-stage">{stageLabel}</p>}
      <p className="wham-detail">{detail}</p>
      <p className="wham-elapsed">
        Elapsed: <span className="mono">{elapsedSec}s</span>
        {!isOver && expectedSeconds > 0 && (
          <> · target <span className="mono">{expectedSeconds}s</span></>
        )}
      </p>
      {/* PR-8d.2 part 2 2A: pre-disclaimer footer — sets the
          "approximate" expectation BEFORE the skeleton overlay
          loads so the ready-state disclaimer (PR-8d.2 Part 1)
          isn't a surprise. */}
      <p className="wham-pre-disclaimer">
        Body alignment in the analysis will be approximate — a
        coaching anchor, not a precise measurement.
      </p>
      <style>{css}</style>
    </div>
  );
}

function FailedScreen({
  title,
  message,
  helpHeader,
  helpBullets,
  onRetry,
  onBack,
  supportRef,
}: {
  title: string;
  message: string;
  helpHeader?: string;
  helpBullets?: readonly string[];
  onRetry: () => void;
  onBack: () => void;
  supportRef?: string;
}) {
  return (
    <div className="page-center wham-screen">
      <button className="btn-corner-back" onClick={onBack}>← History</button>
      <div className="wham-fail-icon">!</div>
      <h2 className="wham-title">{title}</h2>
      <p className="wham-fail-message">{message}</p>
      {/* PR-8d.2 part 2 2A: optional help row. Used by
          failed_preprocessing to spell out what "Try again"
          should look like. NOT used by failed_system per
          Q3/Q4 simplification. */}
      {helpBullets && helpBullets.length > 0 && (
        <div className="wham-help-row">
          {helpHeader && <p className="wham-help-header">{helpHeader}</p>}
          <ul className="wham-help-bullets">
            {helpBullets.map((b, i) => <li key={i}>{b}</li>)}
          </ul>
        </div>
      )}
      <button className="wham-retry-btn" onClick={onRetry}>Try again →</button>
      {supportRef && (
        <p className="wham-support-ref">
          Error reference: <span className="mono">{supportRef}</span>
        </p>
      )}
      <style>{css}</style>
    </div>
  );
}

const css = `
*, *::before, *::after { box-sizing:border-box; margin:0; padding:0; -webkit-tap-highlight-color:transparent; }
body { background:#050805; }
.page { min-height:100dvh; background:#050805; font-family:'DM Sans',system-ui,sans-serif; max-width:430px; margin:0 auto; display:flex; flex-direction:column; color:#f0f0ee; }
/* PR-7c-frontend-v10: side-by-side tuning layout. Expand the 430px
   mobile column to 820px (= 430 video + 16 gap + 360 panel + breathing
   room) so SwingPlayer's .sp-row flex actually has room. Class is
   only applied when ?tune=anchors AND viewport >= 1100px. */
.page.page-tune-wide { max-width:820px; }
.page-center { min-height:100dvh; background:#050805; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:16px; padding:40px; font-family:'DM Sans',system-ui; }
.hdr { display:flex; align-items:center; justify-content:space-between; padding:10px 14px; background:#050805; border-bottom:1px solid rgba(255,255,255,0.05); flex-shrink:0; }
.hdr-logo { font-size:16px; font-weight:800; color:#a8f040; letter-spacing:-0.3px; }
.btn-hdr-back { font-size:18px; color:#4a5a44; background:none; border:none; cursor:pointer; padding:4px 8px; font-family:inherit; }
.btn-new { font-size:12px; font-weight:700; color:#a8f040; background:rgba(168,240,64,0.10); border:1px solid rgba(168,240,64,0.25); padding:6px 14px; border-radius:100px; cursor:pointer; font-family:inherit; }
.no-vid { background:#0a100a; padding:60px 24px; text-align:center; color:#3a4a35; font-size:14px; }
.coaching-bar { padding:14px 18px 20px; display:flex; flex-direction:column; gap:8px; border-top:1px solid rgba(255,255,255,0.05); background:#050805; }
.issue-row { display:flex; align-items:center; gap:8px; }
.issue-dot { font-size:16px; flex-shrink:0; }
.issue-text { font-size:17px; font-weight:800; color:#a8f040; letter-spacing:-0.4px; line-height:1.1; }
.cue-row { padding-left:24px; }
.cue-quote { font-size:14px; font-style:italic; font-weight:600; color:#7a8a72; line-height:1.4; }
.wham-disclaimer { margin:10px 18px 14px; padding:0; font-size:12px; color:#5a6a55; text-align:center; line-height:1.4; font-weight:400; }
.spinner { width:32px; height:32px; border:3px solid rgba(168,240,64,0.15); border-top-color:#a8f040; border-radius:50%; animation:spin 0.8s linear infinite; }
@keyframes spin { to { transform:rotate(360deg); } }
.load-txt, .err-txt { font-size:14px; color:#3a4a35; font-family:'DM Sans',system-ui; }
.btn-back { font-size:14px; font-weight:700; color:#a8f040; background:none; border:none; cursor:pointer; font-family:'DM Sans',system-ui; }

/* PR-8d.0 wham-screen states (processing + failed) */
.wham-screen { gap:18px; padding:24px; max-width:430px; }
.btn-corner-back { position:absolute; top:14px; left:14px; font-size:14px; font-weight:600; color:#7a8a72; background:none; border:none; cursor:pointer; font-family:inherit; padding:6px 10px; }
.wham-anim { position:relative; width:96px; height:96px; display:flex; align-items:center; justify-content:center; margin-bottom:4px; }
.wham-anim-ring { position:absolute; inset:0; border-radius:50%; background:rgba(168,240,64,0.08); animation:wham-ring 1.8s ease-in-out infinite; }
.wham-anim-core { width:40px; height:40px; border-radius:50%; background:#a8f040; animation:wham-core 1.8s ease-in-out infinite; }
@keyframes wham-ring { 0%,100% { transform:scale(1); opacity:0.85; } 50% { transform:scale(1.22); opacity:0.30; } }
@keyframes wham-core { 0%,100% { transform:scale(1); } 50% { transform:scale(0.78); } }
.wham-title { font-size:20px; font-weight:800; color:#f0f0ee; letter-spacing:-0.3px; text-align:center; padding:0 12px; }
.wham-detail { font-size:14px; color:#7a8a72; text-align:center; line-height:1.5; max-width:320px; padding:0 8px; }
.wham-elapsed { font-size:12px; color:#3a4a35; font-family:'DM Sans',system-ui; margin-top:4px; }
/* PR-8d.2 part 2 2A — processing stage hint + pre-disclaimer footer. */
.wham-stage { font-size:13px; color:#7a8a72; text-align:center; font-weight:600; letter-spacing:0.2px; margin-top:-6px; margin-bottom:-2px; }
.wham-pre-disclaimer { font-size:11px; color:#5a6a55; text-align:center; line-height:1.5; max-width:320px; padding:14px 16px 0; margin:8px 0 0; border-top:1px solid rgba(255,255,255,0.05); }
/* PR-8d.2 part 2 2A — failed_preprocessing "What to try" help row. */
.wham-help-row { width:100%; max-width:340px; padding:0 12px; }
.wham-help-header { font-size:13px; font-weight:700; color:#a8f040; margin:0 0 8px; text-align:left; }
.wham-help-bullets { list-style:none; padding:0; margin:0; }
.wham-help-bullets li { font-size:13px; color:#7a8a72; line-height:1.5; padding-left:16px; position:relative; margin-bottom:6px; text-align:left; }
.wham-help-bullets li::before { content:"•"; position:absolute; left:2px; top:0; color:#a8f040; font-weight:700; }
.mono { font-family:ui-monospace, SFMono-Regular, "Menlo", monospace; color:#a8f040; }
.wham-fail-icon { width:56px; height:56px; border-radius:50%; background:rgba(240,96,64,0.12); border:1.5px solid rgba(240,96,64,0.4); color:#f06040; font-size:30px; font-weight:800; display:flex; align-items:center; justify-content:center; }
.wham-fail-message { font-size:14px; color:#c0c0bb; line-height:1.55; text-align:center; max-width:340px; padding:0 12px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:14px 16px; }
.wham-retry-btn { margin-top:4px; background:#a8f040; color:#080c08; font-family:inherit; font-size:15px; font-weight:800; height:48px; padding:0 24px; border-radius:100px; border:none; cursor:pointer; box-shadow:0 0 20px rgba(168,240,64,0.18); }
.wham-retry-btn:active { transform:scale(0.97); }
.wham-support-ref { font-size:11px; color:#3a4a35; margin-top:6px; }
`;
