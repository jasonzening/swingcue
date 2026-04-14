'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { createClient } from '@/lib/supabase/client';
import type { SwingIssueType } from '@/lib/types/swing';

const ISSUE_LABELS: Record<SwingIssueType, string> = {
  early_extension: 'Early Extension',
  steep_downswing: 'Steep Downswing',
  head_movement: 'Head Movement',
  weight_shift_issue: 'Reverse Pivot',
  hand_path_issue: 'Hand Path Issue',
  steep_backswing_plane: 'Steep Backswing Plane',
};

const ISSUE_ICONS: Record<SwingIssueType, string> = {
  early_extension: '🏃',
  steep_downswing: '📉',
  head_movement: '👁',
  weight_shift_issue: '⚖️',
  hand_path_issue: '✋',
  steep_backswing_plane: '📐',
};

const SEV_COLOR: Record<string, string> = {
  low: '#60d040',
  medium: '#f0c040',
  high: '#f06040',
};

interface HistoryItem {
  id: string;
  original_filename: string;
  view_type: string;
  status: string;
  created_at: string;
  analysis?: {
    issue_type: SwingIssueType;
    score: number;
    severity: string;
    cue_text: string;
  } | null;
}

export default function HistoryPage() {
  const router = useRouter();
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [empty, setEmpty] = useState(false);

  useEffect(() => {
    async function load() {
      const supabase = createClient();
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) { router.replace('/sign-in'); return; }

      const { data: videos } = await supabase
        .from('swing_videos')
        .select('id, original_filename, view_type, status, created_at')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false })
        .limit(50);

      if (!videos || videos.length === 0) { setEmpty(true); setLoading(false); return; }

      // Fetch analyses for completed videos
      const completedIds = videos.filter(v => v.status === 'completed').map(v => v.id);
      let analysisMap: Record<string, HistoryItem['analysis']> = {};

      if (completedIds.length > 0) {
        const { data: analyses } = await supabase
          .from('swing_analysis')
          .select('video_id, issue_type, score, severity, cue_text')
          .in('video_id', completedIds);

        if (analyses) {
          analysisMap = Object.fromEntries(analyses.map(a => [a.video_id, a]));
        }
      }

      const enriched: HistoryItem[] = videos.map(v => ({
        ...v,
        analysis: analysisMap[v.id] ?? null,
      }));

      setItems(enriched);
      setLoading(false);
    }
    load();
  }, [router]);

  const handleSignOut = async () => {
    const supabase = createClient();
    await supabase.auth.signOut();
    router.replace('/sign-in');
  };

  return (
    <div className="page">
      <header className="header">
        <a href="/" className="logo">SwingCue</a>
        <div className="header-right">
          <a href="/upload" className="upload-link">+ New swing</a>
          <button className="signout-btn" onClick={handleSignOut}>Sign out</button>
        </div>
      </header>

      <main className="main">
        <div className="title-row">
          <h1 className="h1">Your swings</h1>
        </div>

        {loading && (
          <div className="center-fill">
            <div className="spinner" />
          </div>
        )}

        {!loading && empty && (
          <div className="center-fill">
            <span className="empty-icon">🏌️</span>
            <p className="empty-title">No swings yet</p>
            <p className="empty-sub">Upload your first swing video to get started</p>
            <a href="/upload" className="btn-cta">Upload Your Swing →</a>
          </div>
        )}

        {!loading && !empty && (
          <div className="list">
            {items.map(item => (
              <div
                key={item.id}
                className={`item ${item.status === 'completed' ? 'clickable' : ''}`}
                onClick={() => item.status === 'completed' && router.push(`/result/${item.id}`)}
              >
                {/* Left: icon */}
                <div className="item-icon">
                  {item.analysis
                    ? (ISSUE_ICONS[item.analysis.issue_type] ?? '🏌️')
                    : item.status === 'failed' ? '⚠️' : '⏳'}
                </div>

                {/* Middle: info */}
                <div className="item-body">
                  {item.analysis ? (
                    <>
                      <p className="item-issue">
                        {ISSUE_LABELS[item.analysis.issue_type] ?? item.analysis.issue_type}
                      </p>
                      <p className="item-cue">&ldquo;{item.analysis.cue_text.slice(0, 50)}&rdquo;</p>
                    </>
                  ) : (
                    <p className="item-issue">
                      {item.status === 'failed' ? 'Analysis failed' :
                       item.status === 'processing' ? 'Processing…' :
                       'Pending analysis'}
                    </p>
                  )}
                  <div className="item-meta">
                    <span>{item.view_type === 'face_on' ? 'Face-On' : 'Down the Line'}</span>
                    <span>·</span>
                    <span>{new Date(item.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}</span>
                  </div>
                </div>

                {/* Right: score or status */}
                <div className="item-right">
                  {item.analysis ? (
                    <>
                      <span className="item-score">{item.analysis.score}</span>
                      <span
                        className="item-sev"
                        style={{ color: SEV_COLOR[item.analysis.severity] ?? '#f0c040' }}
                      >
                        {item.analysis.severity}
                      </span>
                    </>
                  ) : (
                    <span className="item-status">{item.status}</span>
                  )}
                  {item.status === 'completed' && <span className="chevron">›</span>}
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
      <style>{css}</style>
    </div>
  );
}

const css = `
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
  body { background: #080c08; }
  .page { min-height: 100vh; background: #080c08; font-family: 'DM Sans', system-ui, sans-serif; max-width: 430px; margin: 0 auto; color: #f0f0ee; display: flex; flex-direction: column; }

  .header { display: flex; justify-content: space-between; align-items: center; padding: 14px 20px; border-bottom: 1px solid rgba(255,255,255,0.05); position: sticky; top: 0; background: rgba(8,12,8,0.95); backdrop-filter: blur(16px); z-index: 50; }
  .logo { font-size: 18px; font-weight: 800; color: #a8f040; letter-spacing: -0.3px; text-decoration: none; }
  .header-right { display: flex; align-items: center; gap: 14px; }
  .upload-link { font-size: 13px; font-weight: 700; color: #a8f040; text-decoration: none; background: rgba(168,240,64,0.1); border: 1px solid rgba(168,240,64,0.2); padding: 6px 14px; border-radius: 100px; }
  .signout-btn { font-size: 13px; font-weight: 600; color: #3a4a35; background: none; border: none; cursor: pointer; font-family: inherit; }

  .main { flex: 1; display: flex; flex-direction: column; padding-bottom: 52px; }
  .title-row { padding: 24px 20px 16px; }
  .h1 { font-size: 24px; font-weight: 800; color: #f0f0ee; letter-spacing: -0.5px; }

  .center-fill { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 14px; padding: 48px 24px; text-align: center; }
  .empty-icon { font-size: 52px; }
  .empty-title { font-size: 20px; font-weight: 800; color: #f0f0ee; letter-spacing: -0.4px; }
  .empty-sub { font-size: 14px; color: #3a4a35; line-height: 1.55; }
  .btn-cta { background: #a8f040; color: #080c08; font-family: inherit; font-size: 16px; font-weight: 800; height: 52px; border-radius: 100px; border: none; cursor: pointer; padding: 0 28px; text-decoration: none; display: flex; align-items: center; }

  .list { display: flex; flex-direction: column; gap: 0; }
  .item { display: flex; align-items: center; gap: 14px; padding: 16px 20px; border-bottom: 1px solid rgba(255,255,255,0.04); transition: background 0.12s; }
  .item.clickable { cursor: pointer; }
  .item.clickable:active { background: rgba(255,255,255,0.03); }

  .item-icon { font-size: 26px; flex-shrink: 0; width: 36px; text-align: center; }
  .item-body { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 4px; }
  .item-issue { font-size: 15px; font-weight: 700; color: #e0e8d8; }
  .item-cue { font-size: 12px; color: #3a4a35; font-style: italic; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .item-meta { display: flex; gap: 6px; font-size: 11px; color: #2a3a25; }

  .item-right { display: flex; flex-direction: column; align-items: flex-end; gap: 4px; flex-shrink: 0; }
  .item-score { font-size: 22px; font-weight: 800; color: #a8f040; letter-spacing: -0.5px; line-height: 1; }
  .item-sev { font-size: 10px; font-weight: 700; text-transform: capitalize; }
  .item-status { font-size: 11px; color: #3a4a35; text-transform: capitalize; }
  .chevron { font-size: 20px; color: #2a3a25; line-height: 1; margin-top: 4px; }

  .spinner { width: 32px; height: 32px; border: 3px solid rgba(168,240,64,0.15); border-top-color: #a8f040; border-radius: 50%; animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
`;
