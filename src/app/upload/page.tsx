'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { createClient } from '@/lib/supabase/client';
import type { SwingViewType } from '@/lib/types/swing';

type Stage = 'idle' | 'uploading' | 'analyzing' | 'done' | 'error';
type ClubType = 'iron' | 'driver' | 'unknown';
type SourceType = 'recorded' | 'uploaded';

const VIEW_OPTIONS: { value: SwingViewType; label: string; hint: string }[] = [
  { value: 'face_on', label: 'Face-On', hint: 'Camera in front' },
  { value: 'down_the_line', label: 'Down the Line', hint: 'Camera behind' },
];

const CLUB_OPTIONS: { value: ClubType; label: string; icon: string }[] = [
  { value: 'iron', label: 'Iron', icon: '⛳' },
  { value: 'driver', label: 'Driver', icon: '🏌️' },
  { value: 'unknown', label: 'Not sure', icon: '❓' },
];

const STAGE_STEPS = [
  { key: 'uploading', label: 'Uploading video…', icon: '⬆️' },
  { key: 'analyzing', label: 'Analyzing your swing…', icon: '🔍' },
];

export default function UploadPage() {
  const router = useRouter();
  const [checking, setChecking] = useState(true);
  const [stage, setStage] = useState<Stage>('idle');
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [viewType, setViewType] = useState<SwingViewType>('face_on');
  const [clubType, setClubType] = useState<ClubType>('iron');
  const [uploadPct, setUploadPct] = useState(0);
  const [errorMsg, setErrorMsg] = useState('');
  const [videoId, setVideoId] = useState('');
  const [analysisStep, setAnalysisStep] = useState(0);
  const recordRef = useRef<HTMLInputElement>(null);
  const uploadRef = useRef<HTMLInputElement>(null);
  const sourceTypeRef = useRef<SourceType>('uploaded');

  useEffect(() => {
    const supabase = createClient();
    supabase.auth.getUser().then(({ data: { user } }) => {
      if (!user) router.replace('/sign-in');
      else setChecking(false);
    });
  }, [router]);

  // Clean up preview URL
  useEffect(() => {
    return () => { if (previewUrl) URL.revokeObjectURL(previewUrl); };
  }, [previewUrl]);

  const handleFile = (f: File, source: SourceType) => {
    if (!f.type.startsWith('video/')) {
      setErrorMsg('Please select a video file (MP4, MOV, AVI, etc.)');
      return;
    }
    if (f.size > 500 * 1024 * 1024) {
      setErrorMsg('Video must be under 500 MB');
      return;
    }
    setErrorMsg('');
    setFile(f);
    sourceTypeRef.current = source;
    const url = URL.createObjectURL(f);
    setPreviewUrl(url);
    setStage('idle');
  };

  const handleSignOut = useCallback(async () => {
    const supabase = createClient();
    await supabase.auth.signOut();
    router.replace('/sign-in');
  }, [router]);

  const reset = () => {
    setStage('idle');
    setFile(null);
    setPreviewUrl(null);
    setUploadPct(0);
    setErrorMsg('');
    setVideoId('');
    setAnalysisStep(0);
    if (recordRef.current) recordRef.current.value = '';
    if (uploadRef.current) uploadRef.current.value = '';
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setErrorMsg('');
    try {
      const supabase = createClient();
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) { router.replace('/sign-in'); return; }

      // Stage 1: Upload
      setStage('uploading');
      setUploadPct(10);

      // Create DB record
      const { data: videoRow, error: insertErr } = await supabase
        .from('swing_videos')
        .insert({
          user_id: user.id,
          storage_path: '',
          original_filename: file.name,
          file_size_bytes: file.size,
          view_type: viewType,
          status: 'uploaded',
          source_type: sourceTypeRef.current,
          club_type: clubType,
        })
        .select('id')
        .single();

      if (insertErr || !videoRow) throw new Error(insertErr?.message ?? 'Failed to create record');
      const vid = videoRow.id as string;
      setVideoId(vid);
      setUploadPct(30);

      // Upload to storage
      const storagePath = `${user.id}/${vid}/${file.name}`;
      const { error: uploadErr } = await supabase.storage
        .from('swing-videos')
        .upload(storagePath, file, { upsert: false });

      if (uploadErr) throw new Error(`Upload failed: ${uploadErr.message}`);
      setUploadPct(80);

      // Update storage path
      await supabase.from('swing_videos')
        .update({ storage_path: storagePath })
        .eq('id', vid);

      setUploadPct(100);

      // Stage 2: Analyze
      setStage('analyzing');
      setAnalysisStep(0);

      // Simulate analysis steps for UX
      const stepInterval = setInterval(() => {
        setAnalysisStep(prev => Math.min(prev + 1, 3));
      }, 900);

      const res = await fetch(`/api/analyze/${vid}`, { method: 'POST' });
      clearInterval(stepInterval);

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error ?? 'Analysis failed');
      }

      setStage('done');
      setTimeout(() => router.push(`/result/${vid}`), 700);

    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Something went wrong. Please try again.';
      setErrorMsg(msg);
      setStage('error');
    }
  };

  if (checking) return (
    <div className="page center"><div className="spinner" /><style>{css}</style></div>
  );

  return (
    <div className="page">
      {/* Hidden file inputs */}
      <input ref={recordRef} type="file" accept="video/*" capture="environment"
        style={{ display: 'none' }} onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f, 'recorded'); }} />
      <input ref={uploadRef} type="file" accept="video/*"
        style={{ display: 'none' }} onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f, 'uploaded'); }} />

      <header className="header">
        <a href="/" className="logo">SwingCue</a>
        <div className="header-right">
          <a href="/history" className="hist-link">History</a>
          <button className="signout-btn" onClick={handleSignOut}>Sign out</button>
        </div>
      </header>

      <main className="main">

        {/* ── IDLE STATE ── */}
        {(stage === 'idle' || stage === 'error') && (
          <>
            {!file ? (
              /* ── No file selected: show two entry points ── */
              <div className="entry-screen">
                <h1 className="h1">Upload your swing</h1>
                <p className="sub">
                  Film your swing. See it in red.<br />Fix it in green.
                </p>

                <div className="two-buttons">
                  <button className="entry-btn record-btn" onClick={() => recordRef.current?.click()}>
                    <span className="entry-icon">📹</span>
                    <span className="entry-label">Record Swing</span>
                    <span className="entry-hint">Open camera</span>
                  </button>
                  <button className="entry-btn upload-btn" onClick={() => uploadRef.current?.click()}>
                    <span className="entry-icon">📁</span>
                    <span className="entry-label">Choose Video</span>
                    <span className="entry-hint">From library</span>
                  </button>
                </div>

                <div className="tips">
                  <p className="tip">💡 Film from the front (face-on) or from behind (down the line)</p>
                  <p className="tip">📱 Landscape mode works best · 10–30 seconds is ideal</p>
                </div>
              </div>

            ) : (
              /* ── File selected: show config + preview ── */
              <div className="config-screen">
                {/* Video preview */}
                <div className="preview-wrap">
                  <video className="preview-video" src={previewUrl ?? ''} controls playsInline muted />
                  <button className="change-btn" onClick={reset}>✕ Change</button>
                </div>

                {/* View type */}
                <div className="section">
                  <p className="section-label">Camera angle</p>
                  <div className="toggle-row">
                    {VIEW_OPTIONS.map(opt => (
                      <button key={opt.value}
                        className={`toggle-btn ${viewType === opt.value ? 'active' : ''}`}
                        onClick={() => setViewType(opt.value)}>
                        <span className="tb-label">{opt.label}</span>
                        <span className="tb-hint">{opt.hint}</span>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Club type */}
                <div className="section">
                  <p className="section-label">Club</p>
                  <div className="club-row">
                    {CLUB_OPTIONS.map(opt => (
                      <button key={opt.value}
                        className={`club-btn ${clubType === opt.value ? 'active' : ''}`}
                        onClick={() => setClubType(opt.value)}>
                        <span>{opt.icon}</span>
                        <span>{opt.label}</span>
                      </button>
                    ))}
                  </div>
                </div>

                {errorMsg && <p className="error-msg">{errorMsg}</p>}

                <button className="btn-analyze" onClick={handleAnalyze}>
                  Analyze My Swing →
                </button>
                <p className="notice">Your video is private and only visible to you</p>
              </div>
            )}
          </>
        )}

        {/* ── UPLOADING STATE ── */}
        {stage === 'uploading' && (
          <div className="status-screen">
            <div className="status-icon-lg">⬆️</div>
            <h2 className="status-title">Uploading your video</h2>
            <div className="progress-track">
              <div className="progress-bar" style={{ width: `${uploadPct}%` }} />
            </div>
            <p className="progress-pct">{uploadPct}%</p>
            <p className="status-note">Keep this screen open</p>
          </div>
        )}

        {/* ── ANALYZING STATE ── */}
        {stage === 'analyzing' && (
          <div className="status-screen">
            <div className="pulse-ring">
              <div className="pulse-core" />
            </div>
            <h2 className="status-title">Analyzing your swing</h2>

            <div className="steps-list">
              {[
                'Processing video frames',
                'Detecting body positions',
                'Identifying swing faults',
                'Generating visual correction',
              ].map((step, i) => (
                <div key={step} className={`step-item ${i < analysisStep ? 'done' : i === analysisStep ? 'active' : 'pending'}`}>
                  <span className="step-dot">
                    {i < analysisStep ? '✓' : i === analysisStep ? '⟳' : '○'}
                  </span>
                  <span>{step}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── DONE STATE ── */}
        {stage === 'done' && (
          <div className="status-screen center">
            <div className="done-circle">✅</div>
            <h2 className="status-title">Analysis complete!</h2>
            <p className="status-sub">Taking you to your result…</p>
          </div>
        )}

        {/* ── ERROR STATE ── */}
        {stage === 'error' && file && (
          <div className="status-screen center">
            <div className="status-icon-lg">⚠️</div>
            <h2 className="status-title">Something went wrong</h2>
            <p className="error-detail">{errorMsg}</p>
            <button className="btn-analyze" onClick={reset} style={{ marginTop: 8 }}>
              Try Again
            </button>
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

  .page {
    min-height: 100vh; background: #080c08;
    font-family: 'DM Sans', system-ui, sans-serif;
    max-width: 430px; margin: 0 auto;
    display: flex; flex-direction: column;
    color: #f0f0ee;
  }
  .page.center { align-items: center; justify-content: center; }

  .header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 14px 20px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    position: sticky; top: 0;
    background: rgba(8,12,8,0.95);
    backdrop-filter: blur(16px); z-index: 50;
  }
  .logo { font-size: 18px; font-weight: 800; color: #a8f040; letter-spacing: -0.3px; text-decoration: none; }
  .header-right { display: flex; align-items: center; gap: 14px; }
  .hist-link { font-size: 13px; font-weight: 600; color: #4a5a44; text-decoration: none; }
  .signout-btn { font-size: 13px; font-weight: 600; color: #3a4a35; background: none; border: none; cursor: pointer; font-family: inherit; }

  .main { flex: 1; padding: 0 0 52px; display: flex; flex-direction: column; }

  /* ── ENTRY SCREEN ── */
  .entry-screen { padding: 28px 20px; display: flex; flex-direction: column; gap: 24px; }
  .h1 { font-size: 26px; font-weight: 800; color: #f0f0ee; letter-spacing: -0.6px; }
  .sub { font-size: 15px; color: #4a5a44; line-height: 1.6; }

  .two-buttons { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .entry-btn {
    display: flex; flex-direction: column; align-items: center; gap: 6px;
    padding: 22px 12px; border-radius: 18px; border: 1.5px solid;
    cursor: pointer; font-family: inherit;
    transition: transform 0.12s, opacity 0.15s;
  }
  .entry-btn:active { transform: scale(0.96); }
  .record-btn {
    background: rgba(168,240,64,0.06);
    border-color: rgba(168,240,64,0.3);
  }
  .upload-btn {
    background: rgba(255,255,255,0.03);
    border-color: rgba(255,255,255,0.1);
  }
  .entry-icon { font-size: 28px; }
  .entry-label { font-size: 15px; font-weight: 700; color: #f0f0ee; }
  .entry-hint { font-size: 12px; color: #4a5a44; }

  .tips { display: flex; flex-direction: column; gap: 8px; }
  .tip { font-size: 12px; color: #2a3a25; line-height: 1.5; }

  /* ── CONFIG SCREEN ── */
  .config-screen { padding: 20px; display: flex; flex-direction: column; gap: 20px; }

  .preview-wrap { position: relative; }
  .preview-video {
    width: 100%; border-radius: 14px; background: #000;
    max-height: 220px; object-fit: cover;
    border: 1px solid rgba(255,255,255,0.08);
  }
  .change-btn {
    position: absolute; top: 8px; right: 8px;
    background: rgba(0,0,0,0.7); color: #f0f0ee;
    font-size: 12px; font-weight: 700; padding: 5px 10px;
    border-radius: 100px; border: 1px solid rgba(255,255,255,0.2);
    cursor: pointer; font-family: inherit;
  }

  .section { display: flex; flex-direction: column; gap: 10px; }
  .section-label { font-size: 13px; font-weight: 600; color: #5a6a54; }

  .toggle-row { display: flex; gap: 10px; }
  .toggle-btn {
    flex: 1; background: rgba(255,255,255,0.03);
    border: 1.5px solid rgba(255,255,255,0.08); border-radius: 14px;
    padding: 14px 10px; cursor: pointer; font-family: inherit;
    display: flex; flex-direction: column; gap: 4px; transition: all 0.15s;
  }
  .toggle-btn.active { border-color: #a8f040; background: rgba(168,240,64,0.08); }
  .tb-label { font-size: 14px; font-weight: 700; color: #e0e8d8; }
  .toggle-btn.active .tb-label { color: #a8f040; }
  .tb-hint { font-size: 11px; color: #3a4a35; }

  .club-row { display: flex; gap: 8px; }
  .club-btn {
    flex: 1; padding: 12px 6px;
    border: 1.5px solid rgba(255,255,255,0.08);
    border-radius: 12px; background: rgba(255,255,255,0.03);
    cursor: pointer; font-family: inherit; font-size: 13px;
    color: #5a6a54; display: flex; flex-direction: column;
    align-items: center; gap: 4px; transition: all 0.15s;
  }
  .club-btn.active { border-color: #a8f040; color: #a8f040; background: rgba(168,240,64,0.08); }

  .error-msg {
    font-size: 13px; color: #f06040;
    background: rgba(240,96,64,0.08); border: 1px solid rgba(240,96,64,0.15);
    border-radius: 10px; padding: 12px 14px; line-height: 1.5;
  }

  .btn-analyze {
    background: #a8f040; color: #080c08;
    font-family: inherit; font-size: 16px; font-weight: 800;
    height: 56px; border-radius: 100px; border: none;
    cursor: pointer; width: 100%;
    box-shadow: 0 0 28px rgba(168,240,64,0.22);
    transition: transform 0.12s; -webkit-appearance: none;
  }
  .btn-analyze:active { transform: scale(0.97); }
  .notice { font-size: 12px; color: #2a3a25; text-align: center; }

  /* ── STATUS SCREENS ── */
  .status-screen {
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 18px; padding: 40px 24px; min-height: 70vh;
    text-align: center;
  }
  .status-screen.center { justify-content: center; }
  .status-icon-lg { font-size: 52px; }
  .status-title { font-size: 22px; font-weight: 800; color: #f0f0ee; letter-spacing: -0.4px; }
  .status-sub { font-size: 14px; color: #4a5a44; }
  .status-note { font-size: 12px; color: #2a3a25; }

  .progress-track {
    width: 100%; max-width: 280px; height: 6px;
    background: rgba(255,255,255,0.06); border-radius: 3px; overflow: hidden;
  }
  .progress-bar { height: 100%; background: #a8f040; border-radius: 3px; transition: width 0.4s ease; }
  .progress-pct { font-size: 13px; color: #4a5a44; font-weight: 600; }

  .pulse-ring {
    width: 80px; height: 80px; border-radius: 50%;
    background: rgba(168,240,64,0.07);
    display: flex; align-items: center; justify-content: center;
    animation: ring 1.8s ease-in-out infinite;
  }
  .pulse-core { width: 36px; height: 36px; border-radius: 50%; background: #a8f040; animation: core 1.8s ease-in-out infinite; }
  @keyframes ring { 0%,100% { transform: scale(1); opacity: 0.8; } 50% { transform: scale(1.18); opacity: 0.3; } }
  @keyframes core { 0%,100% { transform: scale(1); } 50% { transform: scale(0.82); } }

  .steps-list { display: flex; flex-direction: column; gap: 10px; width: 100%; max-width: 260px; text-align: left; }
  .step-item { display: flex; align-items: center; gap: 10px; font-size: 13px; padding: 9px 12px; border-radius: 8px; }
  .step-item.done { color: #a8f040; background: rgba(168,240,64,0.07); }
  .step-item.active { color: #f0f0ee; background: rgba(255,255,255,0.04); }
  .step-item.pending { color: #2a3a25; }
  .step-dot { font-size: 14px; width: 18px; text-align: center; }

  .done-circle { font-size: 60px; }
  .error-detail {
    font-size: 13px; color: #f06040;
    background: rgba(240,96,64,0.08); border: 1px solid rgba(240,96,64,0.15);
    border-radius: 10px; padding: 12px 16px; max-width: 320px; line-height: 1.5;
  }

  .spinner {
    width: 36px; height: 36px;
    border: 3px solid rgba(168,240,64,0.15); border-top-color: #a8f040;
    border-radius: 50%; animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
`;
