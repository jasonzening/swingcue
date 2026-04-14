'use client';

import { useState, useEffect } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { createClient } from '@/lib/supabase/client';
import type { SwingAnalysisRow, SwingVideoRow, SwingIssueType } from '@/lib/types/swing';

const ISSUE_LABELS: Record<SwingIssueType, string> = {
  early_extension: 'Early Extension',
  steep_downswing: 'Steep Downswing',
  head_movement: 'Head Movement',
  weight_shift_issue: 'Reverse Pivot',
  hand_path_issue: 'Hand Path Issue',
  steep_backswing_plane: 'Steep Backswing Plane',
};

const SEVERITY_CONFIG = {
  low: { label: 'Low severity', color: '#60c840', bg: 'rgba(96,200,64,0.1)', border: 'rgba(96,200,64,0.2)' },
  medium: { label: 'Medium severity', color: '#f0c040', bg: 'rgba(240,192,64,0.1)', border: 'rgba(240,192,64,0.2)' },
  high: { label: 'High severity', color: '#f06040', bg: 'rgba(240,96,64,0.1)', border: 'rgba(240,96,64,0.2)' },
};

type LoadState = 'loading' | 'ready' | 'not-found' | 'error';

export default function ResultPage() {
  const router = useRouter();
  const params = useParams();
  const videoId = params.id as string;

  const [loadState, setLoadState] = useState<LoadState>('loading');
  const [video, setVideo] = useState<SwingVideoRow | null>(null);
  const [analysis, setAnalysis] = useState<SwingAnalysisRow | null>(null);

  useEffect(() => {
    async function load() {
      const supabase = createClient();

      const { data: { user } } = await supabase.auth.getUser();
      if (!user) { router.replace('/sign-in'); return; }

      const { data: vid, error: vidErr } = await supabase
        .from('swing_videos')
        .select('*')
        .eq('id', videoId)
        .eq('user_id', user.id)
        .single();

      if (vidErr || !vid) { setLoadState('not-found'); return; }
      setVideo(vid as SwingVideoRow);

      if (vid.status !== 'completed') { setLoadState('not-found'); return; }

      const { data: ana, error: anaErr } = await supabase
        .from('swing_analysis')
        .select('*')
        .eq('video_id', videoId)
        .single();

      if (anaErr || !ana) { setLoadState('error'); return; }
      setAnalysis(ana as SwingAnalysisRow);
      setLoadState('ready');
    }
    load();
  }, [videoId, router]);

  const handleSignOut = async () => {
    const supabase = createClient();
    await supabase.auth.signOut();
    router.replace('/sign-in');
  };

  if (loadState === 'loading') {
    return (
      <div className="page center">
        <div className="spinner" />
        <p className="loading-text">Loading your result…</p>
        <style>{css}</style>
      </div>
    );
  }

  if (loadState === 'not-found' || loadState === 'error') {
    return (
      <div className="page center">
        <span className="status-icon">⚠️</span>
        <p className="loading-text">
          {loadState === 'not-found' ? 'Result not found.' : 'Failed to load result.'}
        </p>
        <button className="btn-cta" onClick={() => router.push('/upload')}>
          Try another upload
        </button>
        <style>{css}</style>
      </div>
    );
  }

  if (!analysis || !video) return null;

  const issueLabel = ISSUE_LABELS[analysis.issue_type] ?? analysis.issue_type;
  const sev = SEVERITY_CONFIG[analysis.severity ?? 'medium'];
  const viewLabel = video.view_type === 'face_on' ? 'Face-On' : 'Down the Line';

  return (
    <div className="page">
      {/* Header */}
      <header className="header">
        <span className="logo">SwingCue</span>
        <button className="signout-btn" onClick={handleSignOut}>Sign out</button>
      </header>

      <main className="main">
        {/* Score row */}
        <div className="score-row">
          <div className="score-block">
            <span className="score-num">{analysis.score}</span>
            <span className="score-denom">/100</span>
          </div>
          <div className="meta-pills">
            <span
              className="sev-pill"
              style={{ color: sev.color, background: sev.bg, border: `1px solid ${sev.border}` }}
            >
              {sev.label}
            </span>
            <span className="view-pill">{viewLabel}</span>
          </div>
        </div>

        {/* Issue */}
        <div className="result-card issue-card">
          <p className="block-label">🎯 &nbsp;MAIN ISSUE</p>
          <h2 className="issue-title">{issueLabel}</h2>
          <p className="issue-body">{analysis.summary_text}</p>
        </div>

        <div className="divider" />

        {/* Cue */}
        <div className="result-card">
          <p className="block-label">💬 &nbsp;YOUR CUE</p>
          <p className="cue-quote">
            &ldquo;{analysis.cue_text}&rdquo;
          </p>
          <p className="cue-hint">
            Say this to yourself before each swing on the range.
          </p>
        </div>

        <div className="divider" />

        {/* Drill */}
        <div className="result-card">
          <p className="block-label">🏌️ &nbsp;TODAY&apos;S DRILL</p>
          <p className="drill-body">{analysis.drill_text}</p>
        </div>

        {/* Metadata */}
        <div className="meta-row">
          <span>📹 {video.original_filename}</span>
          <span>{new Date(video.created_at).toLocaleDateString()}</span>
        </div>

        {/* Bottom CTA */}
        <button className="btn-cta" onClick={() => router.push('/upload')}>
          Analyze Another Swing →
        </button>
      </main>

      <style>{css}</style>
    </div>
  );
}

const css = `
  * { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
  body { background: #080c08; }

  .page {
    min-height: 100vh;
    background: #080c08;
    font-family: 'DM Sans', system-ui, sans-serif;
    max-width: 430px;
    margin: 0 auto;
    color: #f0f0ee;
  }
  .page.center {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 16px;
    padding: 40px 24px;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 20px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    position: sticky;
    top: 0;
    background: rgba(8,12,8,0.95);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    z-index: 50;
  }
  .logo { font-size: 18px; font-weight: 800; color: #a8f040; letter-spacing: -0.3px; }
  .signout-btn {
    font-size: 13px; font-weight: 600; color: #3a4a35;
    background: none; border: none; cursor: pointer; font-family: inherit;
  }

  .main {
    padding: 24px 20px 52px;
    display: flex;
    flex-direction: column;
    gap: 0;
  }

  /* Score */
  .score-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 20px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 20px;
  }
  .score-block { display: flex; align-items: baseline; gap: 2px; }
  .score-num { font-size: 48px; font-weight: 800; color: #a8f040; letter-spacing: -2px; line-height: 1; }
  .score-denom { font-size: 20px; font-weight: 600; color: #4a5a44; }
  .meta-pills { display: flex; flex-direction: column; align-items: flex-end; gap: 6px; }
  .sev-pill {
    font-size: 11px; font-weight: 700;
    padding: 4px 10px; border-radius: 100px;
  }
  .view-pill {
    font-size: 11px; font-weight: 600;
    color: #4a5a44;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    padding: 4px 10px; border-radius: 100px;
  }

  /* Cards */
  .result-card {
    padding: 20px 0;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .issue-card { padding-top: 4px; }
  .block-label {
    font-size: 10px; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #3a4a35;
  }
  .issue-title {
    font-size: 28px; font-weight: 800;
    color: #a8f040; letter-spacing: -0.7px;
    line-height: 1.1;
  }
  .issue-body {
    font-size: 14px; color: #5a6a54; line-height: 1.65;
  }
  .cue-quote {
    font-size: 20px; font-weight: 700; font-style: italic;
    color: #d8e8d0; line-height: 1.35; letter-spacing: -0.3px;
  }
  .cue-hint {
    font-size: 12px; color: #3a4a35; font-style: italic;
  }
  .drill-body {
    font-size: 14px; color: #5a6a54; line-height: 1.7;
  }

  .divider { height: 1px; background: rgba(255,255,255,0.05); }

  /* Metadata */
  .meta-row {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: #2a3a25;
    padding: 16px 0;
    border-top: 1px solid rgba(255,255,255,0.04);
    margin-top: 4px;
  }

  /* CTA */
  .btn-cta {
    background: #a8f040;
    color: #080c08;
    font-family: inherit;
    font-size: 16px;
    font-weight: 800;
    height: 56px;
    border-radius: 100px;
    border: none;
    cursor: pointer;
    width: 100%;
    margin-top: 8px;
    -webkit-appearance: none;
    transition: transform 0.12s;
  }
  .btn-cta:active { transform: scale(0.97); }

  /* Loading */
  .spinner {
    width: 36px; height: 36px;
    border: 3px solid rgba(168,240,64,0.15);
    border-top-color: #a8f040;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loading-text { font-size: 15px; color: #4a5a44; }
  .status-icon { font-size: 40px; }
`;
