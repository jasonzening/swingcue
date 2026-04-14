'use client';

/**
 * Result page — 满屏 Interactive Swing Player
 *
 * 布局原则：
 *   第一层：大播放器（满屏宽）
 *   第二层：控制条 + 阶段按钮（内嵌在播放器组件里）
 *   第三层：1 句主问题 + 1 句 cue（紧凑，不占空间）
 *   不要长文，不要评分卡片，不要大段解释
 */

import { useState, useEffect } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { createClient } from '@/lib/supabase/client';
import { SwingPlayer } from '@/components/SwingPlayer';
import { generateOverlayTimeline } from '@/lib/overlay/templates';
import type { MainIssueType, PhaseMarkers, VideoMetadata } from '@/types/analysis';
import { ISSUE_LABELS } from '@/types/analysis';

export default function ResultPage() {
  const router = useRouter();
  const params = useParams();
  const videoId = params.id as string;

  const [state,    setState]    = useState<'loading' | 'ready' | 'error'>('loading');
  const [videoUrl, setVideoUrl] = useState('');
  const [issue,    setIssue]    = useState<MainIssueType>('early_extension');
  const [cue,      setCue]      = useState('');
  const [phases,   setPhases]   = useState<PhaseMarkers>({
    setupTime: 0, topTime: 0.5, transitionTime: 0.65, impactTime: 0.75, finishTime: 0.9,
  });
  const [meta,     setMeta]     = useState<VideoMetadata>({ durationSec: 3, fps: 30, width: 640, height: 360 });
  const [filename, setFilename] = useState('');
  const [overlayTimeline, setOverlayTimeline] = useState<ReturnType<typeof generateOverlayTimeline> | null>(null);

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
      setFilename(vid.original_filename ?? '');

      const { data: signed } = await supabase.storage
        .from('swing-videos')
        .createSignedUrl(vid.storage_path, 3600);
      if (signed?.signedUrl) setVideoUrl(signed.signedUrl);

      const { data: ana } = await supabase
        .from('swing_analysis')
        .select('*')
        .eq('video_id', videoId)
        .single();

      if (!ana) { setState('error'); return; }

      const issueType = (ana.issue_type as MainIssueType) ?? 'early_extension';
      setIssue(issueType);
      setCue(ana.cue_text ?? '');

      // Build phase markers from stored data or estimate from duration
      const dur = (ana.video_metadata_json as {durationSec?: number})?.durationSec ?? 3;
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
        fps: 30,
        width: 640,
        height: 360,
      };
      setMeta(vm);

      // Generate overlay timeline using the template system
      const olt = generateOverlayTimeline({ phaseMarkers: pm, videoMetadata: vm, issue: issueType });
      setOverlayTimeline(olt);

      setState('ready');
    }
    load();
  }, [videoId, router]);

  /* ── Loading ── */
  if (state === 'loading') {
    return (
      <div className="page-center">
        <div className="spinner" />
        <p className="load-txt">Loading your swing…</p>
        <style>{css}</style>
      </div>
    );
  }

  /* ── Error ── */
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
      {/* ══════════════════════════════════════════
          HEADER — minimal, doesn't steal space
      ══════════════════════════════════════════ */}
      <header className="hdr">
        <button className="btn-hdr-back" onClick={() => router.push('/history')}>←</button>
        <span className="hdr-logo">SwingCue</span>
        <button className="btn-new" onClick={() => router.push('/upload')}>+ New</button>
      </header>

      {/* ══════════════════════════════════════════
          INTERACTIVE SWING PLAYER — 满屏宽，页面主角
      ══════════════════════════════════════════ */}
      {videoUrl ? (
        <SwingPlayer
          videoUrl={videoUrl}
          timeline={overlayTimeline}
          phases={phases}
          duration={meta.durationSec}
        />
      ) : (
        <div className="no-vid">
          <p>Video loading…</p>
        </div>
      )}

      {/* ══════════════════════════════════════════
          ISSUE + CUE — only 2 lines, compact
      ══════════════════════════════════════════ */}
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

  /* Header */
  .hdr { display:flex; align-items:center; justify-content:space-between; padding:10px 14px; background:#050805; border-bottom:1px solid rgba(255,255,255,0.05); flex-shrink:0; }
  .hdr-logo { font-size:16px; font-weight:800; color:#a8f040; letter-spacing:-0.3px; }
  .btn-hdr-back { font-size:18px; color:#4a5a44; background:none; border:none; cursor:pointer; padding:4px 8px; font-family:inherit; }
  .btn-new { font-size:12px; font-weight:700; color:#a8f040; background:rgba(168,240,64,0.10); border:1px solid rgba(168,240,64,0.25); padding:6px 14px; border-radius:100px; cursor:pointer; font-family:inherit; }

  /* No video state */
  .no-vid { background:#0a100a; padding:60px 24px; text-align:center; color:#3a4a35; font-size:14px; font-family:'DM Sans',system-ui; }

  /* Coaching bar — 2 compact lines */
  .coaching-bar {
    padding: 14px 18px 20px;
    display: flex; flex-direction: column; gap: 8px;
    border-top: 1px solid rgba(255,255,255,0.05);
    background: #050805;
  }
  .issue-row { display:flex; align-items:center; gap:8px; }
  .issue-dot { font-size:16px; flex-shrink:0; }
  .issue-text { font-size:17px; font-weight:800; color:#a8f040; letter-spacing:-0.4px; line-height:1.1; }
  .cue-row { padding-left:24px; }
  .cue-quote { font-size:14px; font-style:italic; font-weight:600; color:#7a8a72; line-height:1.4; }

  /* Loading / error */
  .spinner { width:32px; height:32px; border:3px solid rgba(168,240,64,0.15); border-top-color:#a8f040; border-radius:50%; animation:spin 0.8s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .load-txt, .err-txt { font-size:14px; color:#3a4a35; font-family:'DM Sans',system-ui; }
  .btn-back { font-size:14px; font-weight:700; color:#a8f040; background:none; border:none; cursor:pointer; font-family:'DM Sans',system-ui; }
`;
