'use client';

import { useState, useEffect } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { createClient } from '@/lib/supabase/client';
import { SwingOverlay } from '@/lib/overlays/SwingOverlay';
import type { SwingAnalysisRow, SwingVideoRow, SwingIssueType } from '@/lib/types/swing';

const ISSUE_LABELS: Record<SwingIssueType, string> = {
  early_extension: 'Early Extension',
  steep_downswing: 'Steep Downswing',
  head_movement: 'Head Movement',
  weight_shift_issue: 'Reverse Pivot',
  hand_path_issue: 'Hand Path Issue',
  steep_backswing_plane: 'Steep Backswing Plane',
};

const ISSUE_WHAT: Record<SwingIssueType, string> = {
  early_extension: 'Your hips are thrusting toward the ball before impact, breaking your posture.',
  steep_downswing: 'Your club is coming over the top — the path is too steep and outside-in.',
  head_movement: 'Your head is drifting sideways during the swing, making consistent contact hard.',
  weight_shift_issue: 'Your weight is staying on the back foot through impact — a reverse pivot.',
  hand_path_issue: 'Your hands are looping away from your body on the downswing.',
  steep_backswing_plane: 'The club is going too vertical on the way back, making a shallow downswing harder.',
};

const SEV = {
  low: { label: 'Low', color: '#60d040', bg: 'rgba(96,208,64,0.1)', border: 'rgba(96,208,64,0.2)' },
  medium: { label: 'Medium', color: '#f0c040', bg: 'rgba(240,192,64,0.1)', border: 'rgba(240,192,64,0.2)' },
  high: { label: 'High', color: '#f06040', bg: 'rgba(240,96,64,0.1)', border: 'rgba(240,96,64,0.2)' },
};

export default function ResultPage() {
  const router = useRouter();
  const params = useParams();
  const videoId = params.id as string;
  const [state, setState] = useState<'loading' | 'ready' | 'error'>('loading');
  const [video, setVideo] = useState<SwingVideoRow | null>(null);
  const [analysis, setAnalysis] = useState<SwingAnalysisRow | null>(null);

  useEffect(() => {
    async function load() {
      const supabase = createClient();
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) { router.replace('/sign-in'); return; }

      const { data: vid } = await supabase
        .from('swing_videos').select('*').eq('id', videoId).eq('user_id', user.id).single();
      if (!vid || vid.status !== 'completed') { setState('error'); return; }
      setVideo(vid as SwingVideoRow);

      const { data: ana } = await supabase
        .from('swing_analysis').select('*').eq('video_id', videoId).single();
      if (!ana) { setState('error'); return; }
      setAnalysis(ana as SwingAnalysisRow);
      setState('ready');
    }
    load();
  }, [videoId, router]);

  if (state === 'loading') return (
    <div className="page center"><div className="spinner" /><style>{css}</style></div>
  );
  if (state === 'error') return (
    <div className="page center">
      <p className="err-msg">Result not found or still processing.</p>
      <button className="btn-cta" onClick={() => router.push('/upload')}>Back to Upload</button>
      <style>{css}</style>
    </div>
  );
  if (!analysis || !video) return null;

  const issueLabel = ISSUE_LABELS[analysis.issue_type] ?? analysis.issue_type;
  const issueWhat = ISSUE_WHAT[analysis.issue_type] ?? '';
  const sev = SEV[(analysis.severity ?? 'medium') as keyof typeof SEV];
  const viewLabel = video.view_type === 'face_on' ? 'Face-On' : 'Down the Line';

  return (
    <div className="page">
      <header className="header">
        <a href="/" className="logo">SwingCue</a>
        <div className="header-right">
          <a href="/history" className="hist-link">History</a>
        </div>
      </header>

      <main className="main">

        {/* ── VISUAL OVERLAY — the product's core ── */}
        <section className="overlay-section">
          <div className="overlay-label-row">
            <span className="overlay-label red-label">🔴 Current</span>
            <span className="overlay-issue-name">{issueLabel}</span>
            <span className="overlay-label green-label">🟢 Target</span>
          </div>
          <div className="overlay-frame">
            <SwingOverlay issueType={analysis.issue_type} width={390} height={380} />
          </div>
          <p className="overlay-caption">
            Visual correction — red shows your current motion, green shows the target
          </p>
        </section>

        {/* ── SCORE ── */}
        <section className="score-section">
          <div className="score-left">
            <span className="score-num">{analysis.score}</span>
            <span className="score-max">/100</span>
          </div>
          <div className="score-right">
            <span className="sev-badge" style={{ color: sev.color, background: sev.bg, border: `1px solid ${sev.border}` }}>
              ⚠ {sev.label} severity
            </span>
            <span className="view-badge">{viewLabel}</span>
          </div>
        </section>

        {/* ── MAIN ISSUE ── */}
        <section className="card issue-card">
          <p className="card-eyebrow">🎯 &nbsp;MAIN ISSUE</p>
          <h2 className="issue-name">{issueLabel}</h2>
          <p className="issue-what">{issueWhat}</p>
          <p className="issue-body">{analysis.summary_text}</p>
        </section>

        <div className="divider" />

        {/* ── CUE ── */}
        <section className="card">
          <p className="card-eyebrow">💬 &nbsp;YOUR CUE</p>
          <p className="cue-quote">&ldquo;{analysis.cue_text}&rdquo;</p>
          <p className="cue-hint">Say this to yourself before each swing on the range.</p>
        </section>

        <div className="divider" />

        {/* ── DRILL ── */}
        <section className="card">
          <p className="card-eyebrow">🏌️ &nbsp;TODAY&apos;S DRILL</p>
          <p className="drill-text">{analysis.drill_text}</p>
        </section>

        {/* ── META ── */}
        <div className="meta-row">
          <span>📹 {video.original_filename}</span>
          <span>{new Date(video.created_at).toLocaleDateString()}</span>
        </div>

        {/* ── ACTIONS ── */}
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
  .err-msg { font-size: 15px; color: #4a5a44; }

  .header { display: flex; justify-content: space-between; align-items: center; padding: 14px 20px; border-bottom: 1px solid rgba(255,255,255,0.05); position: sticky; top: 0; background: rgba(8,12,8,0.95); backdrop-filter: blur(16px); z-index: 50; }
  .logo { font-size: 18px; font-weight: 800; color: #a8f040; letter-spacing: -0.3px; text-decoration: none; }
  .header-right { display: flex; align-items: center; gap: 14px; }
  .hist-link { font-size: 13px; font-weight: 600; color: #4a5a44; text-decoration: none; }

  .main { padding: 0 0 52px; display: flex; flex-direction: column; }

  /* Overlay */
  .overlay-section { background: #0a120a; }
  .overlay-label-row { display: flex; justify-content: space-between; align-items: center; padding: 12px 20px 6px; }
  .overlay-label { font-size: 11px; font-weight: 700; letter-spacing: 0.04em; }
  .red-label { color: #ff6060; }
  .green-label { color: #60e060; }
  .overlay-issue-name { font-size: 13px; font-weight: 800; color: #f0f0ee; }
  .overlay-frame { width: 100%; }
  .overlay-frame svg { width: 100%; height: auto; }
  .overlay-caption { font-size: 11px; color: #2a3a25; text-align: center; padding: 8px 16px 14px; }

  /* Score */
  .score-section { display: flex; justify-content: space-between; align-items: center; padding: 18px 20px; border-bottom: 1px solid rgba(255,255,255,0.05); }
  .score-left { display: flex; align-items: baseline; gap: 3px; }
  .score-num { font-size: 52px; font-weight: 800; color: #a8f040; letter-spacing: -2px; line-height: 1; }
  .score-max { font-size: 20px; font-weight: 600; color: #3a4a35; }
  .score-right { display: flex; flex-direction: column; align-items: flex-end; gap: 7px; }
  .sev-badge { font-size: 11px; font-weight: 700; padding: 5px 11px; border-radius: 100px; }
  .view-badge { font-size: 11px; color: #3a4a35; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07); padding: 4px 11px; border-radius: 100px; }

  /* Cards */
  .card { padding: 20px; display: flex; flex-direction: column; gap: 8px; }
  .card-eyebrow { font-size: 10px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #2a3a25; }
  .issue-name { font-size: 28px; font-weight: 800; color: #a8f040; letter-spacing: -0.7px; line-height: 1.1; }
  .issue-what { font-size: 14px; color: #7a8a72; line-height: 1.55; }
  .issue-body { font-size: 13px; color: #4a5a44; line-height: 1.65; }
  .cue-quote { font-size: 19px; font-weight: 700; font-style: italic; color: #d8e8d0; line-height: 1.35; letter-spacing: -0.2px; }
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
`;
