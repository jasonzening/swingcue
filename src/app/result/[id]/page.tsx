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

      const vmJson = ana.video_metadata_json as { durationSec?: number; dataSource?: string } | null;
      const dur = vmJson?.durationSec ?? 3;
      const source = vmJson?.dataSource ?? 'stub';
      setDataSource(source);

      const pm: PhaseMarkers = (ana.phase_markers_json as PhaseMarkers | null) ?? {
        setupTime: 0,
        topTime: dur * 0.50,
        transitionTime: dur * 0.62,
        impactTime: dur * 0.75,
        finishTime: dur * 0.92,
      };
      setPhases(pm);

      const vm: VideoMetadata = { durationSec: dur, fps: 30, width: 640, height: 360 };
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

  const issueLabel = ISSUE_LABELS[issue] ?? issue;

  return (
    <div className="page">
      <header className="hdr">
        <button className="btn-hdr-back" onClick={() => router.push('/history')}>←</button>
        <span className="hdr-logo">SwingCue</span>
        <button className="btn-new" onClick={() => router.push('/upload')}>+ New</button>
      </header>

      {videoUrl ? (
        <SwingPlayer
          videoUrl={videoUrl}
          timeline={overlayTimeline}
          phases={phases}
          duration={meta.durationSec}
          dataSource={dataSource}
          poseTimeline={poseTimeline}
          shoulderExpand={expandFactors.shoulder}
          hipExpand={expandFactors.hip}
        />
      ) : (
        <div className="no-vid"><p>Video loading…</p></div>
      )}

      <div className="coaching-bar">
        <div className="issue-row">
          <span className="issue-dot">⚡</span>
          <span className="issue-text">{issueLabel}</span>
        </div>
        <div className="cue-row">
          <span className="cue-quote">&ldquo;{cue}&rdquo;</span>
        </div>
      </div>

      <style>{css}</style>
    </div>
  );
}

const css = `
*, *::before, *::after { box-sizing:border-box; margin:0; padding:0; -webkit-tap-highlight-color:transparent; }
body { background:#050805; }
.page { min-height:100dvh; background:#050805; font-family:'DM Sans',system-ui,sans-serif; max-width:430px; margin:0 auto; display:flex; flex-direction:column; color:#f0f0ee; }
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
.spinner { width:32px; height:32px; border:3px solid rgba(168,240,64,0.15); border-top-color:#a8f040; border-radius:50%; animation:spin 0.8s linear infinite; }
@keyframes spin { to { transform:rotate(360deg); } }
.load-txt, .err-txt { font-size:14px; color:#3a4a35; font-family:'DM Sans',system-ui; }
.btn-back { font-size:14px; font-weight:700; color:#a8f040; background:none; border:none; cursor:pointer; font-family:'DM Sans',system-ui; }
`;
