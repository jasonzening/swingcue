'use client';

import { useState, useEffect } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { createClient } from '@/lib/supabase/client';
import { SwingPlayer } from '@/components/SwingPlayer';
import type { SwingIssueType } from '@/lib/types/swing';
import type { OverlayTimeline } from '@/lib/timeline/generateOverlayTimeline';

const ISSUE_LABELS: Record<SwingIssueType, string> = {
  early_extension: 'Early Extension',
  steep_downswing: 'Steep Downswing',
  head_movement: 'Head Movement',
  weight_shift_issue: 'Reverse Pivot',
  hand_path_issue: 'Hand Path Issue',
  steep_backswing_plane: 'Steep Backswing Plane',
};

const SEV = {
  low:    { label: 'Low', color: '#60d040', bg: 'rgba(96,208,64,0.1)', border: 'rgba(96,208,64,0.2)' },
  medium: { label: 'Medium', color: '#f0c040', bg: 'rgba(240,192,64,0.1)', border: 'rgba(240,192,64,0.2)' },
  high:   { label: 'High', color: '#f06040', bg: 'rgba(240,96,64,0.1)', border: 'rgba(240,96,64,0.2)' },
};

export default function ResultPage() {
  const router = useRouter();
  const params = useParams();
  const videoId = params.id as string;

  const [state, setState] = useState<'loading' | 'ready' | 'error'>('loading');
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<Record<string, unknown> | null>(null);
  const [videoMeta, setVideoMeta] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    async function load() {
      const supabase = createClient();
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) { router.replace('/sign-in'); return; }

      // Load video record
      const { data: vid } = await supabase
        .from('swing_videos').select('*').eq('id', videoId).eq('user_id', user.id).single();
      if (!vid || vid.status !== 'completed') { setState('error'); return; }
      setVideoMeta(vid);

      // Get signed URL for video playback (1 hour expiry)
      const { data: signed } = await supabase.storage
        .from('swing-videos')
        .createSignedUrl(vid.storage_path, 3600);
      if (signed?.signedUrl) setVideoUrl(signed.signedUrl);

      // Load analysis
      const { data: ana } = await supabase
        .from('swing_analysis').select('*').eq('video_id', videoId).single();
      if (!ana) { setState('error'); return; }
      setAnalysis(ana);
      setState('ready');
    }
    load();
  }, [videoId, router]);

  if (state === 'loading') return (
    <div className="page center">
      <div className="spinner" />
      <p className="load-text">Loading your analysis…</p>
      <style>{css}</style>
    </div>
  );

  if (state === 'error' || !analysis) return (
    <div className="page center">
      <p className="err-text">Result not found.</p>
      <button className="btn-cta" onClick={() => router.push('/upload')}>Back to Upload</button>
      <style>{css}</style>
    </div>
  );

  const issueType = analysis.issue_type as SwingIssueType;
  const issueLabel = ISSUE_LABELS[issueType] ?? String(issueType);
  const sev = SEV[(analysis.severity as keyof typeof SEV) ?? 'medium'];
  const timeline = analysis.overlay_timeline_json as OverlayTimeline | null;
  const viewLabel = (videoMeta?.view_type === 'face_on') ? 'Face-On' : 'Down the Line';

  return (
    <div className="page">
      <header className="header">
        <a href="/" className="logo">SwingCue</a>
        <div className="header-right">
          <a href="/history" className="hist-link">History</a>
        </div>
      </header>

      {/* ══════════════════════════════════════
          INTERACTIVE SWING PLAYER — core product
          ══════════════════════════════════════ */}
      {videoUrl && timeline ? (
        <SwingPlayer
          videoUrl={videoUrl}
          timeline={timeline}
          
        />
      ) : (
        /* Fallback if no video URL or timeline */
        <div className="no-video">
          <p className="no-video-text">
            {!videoUrl ? '⚠ Video not accessible — signed URL may have expired' : '⚠ No overlay data'}
          </p>
        </div>
      )}

      {/* ══════════════════════════════════════
          ANALYSIS PANEL
          ══════════════════════════════════════ */}
      <main className="analysis-panel">

        {/* Score */}
        <div className="score-row">
          <div className="score-left">
            <span className="score-num">{analysis.score as number}</span>
            <span className="score-max">/100</span>
          </div>
          <div className="score-right">
            <span className="sev-badge" style={{ color: sev.color, background: sev.bg, border: `1px solid ${sev.border}` }}>
              ⚠ {sev.label} severity
            </span>
            <span className="view-badge">{viewLabel}</span>
          </div>
        </div>

        {/* Main Issue */}
        <section className="card">
          <p className="card-eyebrow">🎯 &nbsp;MAIN ISSUE</p>
          <h2 className="issue-name">{issueLabel}</h2>
          <p className="issue-body">{analysis.summary_text as string}</p>
        </section>

        <div className="divider" />

        {/* Cue */}
        <section className="card">
          <p className="card-eyebrow">💬 &nbsp;YOUR CUE</p>
          <p className="cue-quote">&ldquo;{analysis.cue_text as string}&rdquo;</p>
          <p className="cue-hint">Say this before each swing at the range.</p>
        </section>

        <div className="divider" />

        {/* Drill */}
        <section className="card">
          <p className="card-eyebrow">🏌️ &nbsp;TODAY&apos;S DRILL</p>
          <p className="drill-text">{analysis.drill_text as string}</p>
        </section>

        {/* Meta */}
        <div className="meta-row">
          <span>📹 {videoMeta?.original_filename as string}</span>
          <span>{new Date(videoMeta?.created_at as string).toLocaleDateString()}</span>
        </div>

        {/* Actions */}
        <div className="actions">
          <button className="btn-cta" onClick={() => router.push('/upload')}>
            Analyze Another Swing →
          </button>
          <a href="/history" className="btn-secondary">View History</a>
        </div>

      </main>
      <style>{css}</style>
    </div>
  );
}

const css = `
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
  body { background: #080c08; }
  .page { min-height: 100vh; background: #080c08; font-family: 'DM Sans', system-ui, sans-serif; max-width: 430px; margin: 0 auto; color: #f0f0ee; }
  .page.center { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 18px; padding: 40px; }

  .header { display: flex; justify-content: space-between; align-items: center; padding: 14px 20px; border-bottom: 1px solid rgba(255,255,255,0.05); position: sticky; top: 0; background: rgba(8,12,8,0.95); backdrop-filter: blur(16px); z-index: 50; }
  .logo { font-size: 18px; font-weight: 800; color: #a8f040; letter-spacing: -0.3px; text-decoration: none; }
  .header-right { display: flex; gap: 14px; }
  .hist-link { font-size: 13px; font-weight: 600; color: #4a5a44; text-decoration: none; }

  .no-video { background: #0a100a; padding: 24px; }
  .no-video-text { font-size: 13px; color: #3a4a35; text-align: center; }

  .analysis-panel { padding: 0 0 52px; }

  .score-row { display: flex; justify-content: space-between; align-items: center; padding: 18px 20px; border-bottom: 1px solid rgba(255,255,255,0.05); }
  .score-left { display: flex; align-items: baseline; gap: 3px; }
  .score-num { font-size: 52px; font-weight: 800; color: #a8f040; letter-spacing: -2px; line-height: 1; }
  .score-max { font-size: 20px; font-weight: 600; color: #3a4a35; }
  .score-right { display: flex; flex-direction: column; align-items: flex-end; gap: 7px; }
  .sev-badge { font-size: 11px; font-weight: 700; padding: 5px 11px; border-radius: 100px; }
  .view-badge { font-size: 11px; color: #3a4a35; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07); padding: 4px 11px; border-radius: 100px; }

  .card { padding: 20px; display: flex; flex-direction: column; gap: 8px; }
  .card-eyebrow { font-size: 10px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #2a3a25; }
  .issue-name { font-size: 28px; font-weight: 800; color: #a8f040; letter-spacing: -0.7px; line-height: 1.1; }
  .issue-body { font-size: 14px; color: #5a6a54; line-height: 1.7; }
  .cue-quote { font-size: 19px; font-weight: 700; font-style: italic; color: #d8e8d0; line-height: 1.35; }
  .cue-hint { font-size: 12px; color: #2a3a25; font-style: italic; }
  .drill-text { font-size: 14px; color: #5a6a54; line-height: 1.7; }
  .divider { height: 1px; background: rgba(255,255,255,0.05); margin: 0 20px; }

  .meta-row { display: flex; justify-content: space-between; font-size: 11px; color: #1a2818; padding: 14px 20px; border-top: 1px solid rgba(255,255,255,0.04); margin-top: 6px; }
  .actions { padding: 0 20px; display: flex; flex-direction: column; gap: 12px; }
  .btn-cta { background: #a8f040; color: #080c08; font-family: inherit; font-size: 16px; font-weight: 800; height: 56px; border-radius: 100px; border: none; cursor: pointer; width: 100%; -webkit-appearance: none; transition: transform 0.12s; }
  .btn-cta:active { transform: scale(0.97); }
  .btn-secondary { display: flex; align-items: center; justify-content: center; background: rgba(255,255,255,0.04); color: #4a5a44; font-size: 15px; font-weight: 600; height: 48px; border-radius: 100px; text-decoration: none; border: 1px solid rgba(255,255,255,0.08); }

  .spinner { width: 36px; height: 36px; border: 3px solid rgba(168,240,64,0.15); border-top-color: #a8f040; border-radius: 50%; animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .load-text, .err-text { font-size: 15px; color: #4a5a44; }
`;
