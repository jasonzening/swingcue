'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { createClient } from '@/lib/supabase/client';
import type { SwingViewType } from '@/lib/types/swing';

type Stage = 'idle' | 'uploading' | 'analyzing' | 'done' | 'error';

const VIEW_OPTIONS: { value: SwingViewType; label: string; hint: string }[] = [
  { value: 'face_on', label: 'Face-On', hint: 'Camera in front of you' },
  { value: 'down_the_line', label: 'Down the Line', hint: 'Camera behind you' },
];

export default function UploadPage() {
  const router = useRouter();
  const [userEmail, setUserEmail] = useState('');
  const [checking, setChecking] = useState(true);
  const [stage, setStage] = useState<Stage>('idle');
  const [file, setFile] = useState<File | null>(null);
  const [viewType, setViewType] = useState<SwingViewType>('face_on');
  const [uploadPct, setUploadPct] = useState(0);
  const [videoId, setVideoId] = useState('');
  const [errorMsg, setErrorMsg] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auth guard
  useEffect(() => {
    const supabase = createClient();
    supabase.auth.getUser().then(({ data: { user } }) => {
      if (!user) { router.replace('/sign-in'); return; }
      setUserEmail(user.email ?? '');
      setChecking(false);
    });
  }, [router]);

  const handleSignOut = useCallback(async () => {
    const supabase = createClient();
    await supabase.auth.signOut();
    router.replace('/sign-in');
  }, [router]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    setErrorMsg('');
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setErrorMsg('');

    try {
      const supabase = createClient();
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) { router.replace('/sign-in'); return; }

      // 1. Create video record
      setStage('uploading');
      setUploadPct(5);

      const { data: videoRow, error: insertErr } = await supabase
        .from('swing_videos')
        .insert({
          user_id: user.id,
          storage_path: '', // filled after upload
          original_filename: file.name,
          file_size_bytes: file.size,
          view_type: viewType,
          status: 'uploaded',
        })
        .select('id')
        .single();

      if (insertErr || !videoRow) throw new Error(insertErr?.message ?? 'Failed to create record');
      const vid = videoRow.id as string;
      setVideoId(vid);
      setUploadPct(15);

      // 2. Upload to Supabase Storage
      const storagePath = `${user.id}/${vid}/${file.name}`;
      const { error: uploadErr } = await supabase.storage
        .from('swing-videos')
        .upload(storagePath, file, { upsert: false });

      if (uploadErr) throw new Error(uploadErr.message);
      setUploadPct(80);

      // 3. Update storage_path on the record
      await supabase
        .from('swing_videos')
        .update({ storage_path: storagePath })
        .eq('id', vid);

      setUploadPct(100);

      // 4. Trigger analysis
      setStage('analyzing');

      const res = await fetch(`/api/analyze/${vid}`, { method: 'POST' });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error ?? 'Analysis failed');
      }

      // 5. Done — navigate to result
      setStage('done');
      setTimeout(() => router.push(`/result/${vid}`), 800);

    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Something went wrong. Please try again.';
      setErrorMsg(msg);
      setStage('error');
    }
  };

  const reset = () => {
    setStage('idle');
    setFile(null);
    setUploadPct(0);
    setVideoId('');
    setErrorMsg('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  if (checking) {
    return (
      <div className="page">
        <div className="spinner" />
        <style>{pageStyle}</style>
      </div>
    );
  }

  return (
    <div className="page">
      {/* Header */}
      <header className="header">
        <span className="logo">SwingCue</span>
        <button className="signout-btn" onClick={handleSignOut}>Sign out</button>
      </header>

      <main className="main">
        {/* IDLE — upload form */}
        {stage === 'idle' && (
          <>
            <div className="headline-block">
              <h1 className="h1">Upload your swing</h1>
              <p className="sub">Film your swing from the range, upload it here, and get your fix in seconds.</p>
            </div>

            {/* File picker */}
            <div
              className={`dropzone ${file ? 'has-file' : ''}`}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                capture="environment"
                onChange={handleFileChange}
                style={{ display: 'none' }}
              />
              {file ? (
                <>
                  <span className="dz-icon">🎬</span>
                  <span className="dz-name">{file.name}</span>
                  <span className="dz-size">{(file.size / 1024 / 1024).toFixed(1)} MB</span>
                </>
              ) : (
                <>
                  <span className="dz-icon">📱</span>
                  <span className="dz-label">Tap to select video</span>
                  <span className="dz-hint">MP4, MOV, AVI · max 100 MB</span>
                </>
              )}
            </div>

            {/* View type */}
            <div className="view-section">
              <p className="view-label">Camera angle</p>
              <div className="view-toggle">
                {VIEW_OPTIONS.map(opt => (
                  <button
                    key={opt.value}
                    className={`view-btn ${viewType === opt.value ? 'active' : ''}`}
                    onClick={() => setViewType(opt.value)}
                  >
                    <span className="vb-label">{opt.label}</span>
                    <span className="vb-hint">{opt.hint}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* CTA */}
            <button
              className="btn-analyze"
              disabled={!file}
              onClick={handleAnalyze}
            >
              {file ? 'Analyze My Swing →' : 'Select a video first'}
            </button>

            <p className="notice">Your video is private and only visible to you.</p>
          </>
        )}

        {/* UPLOADING */}
        {stage === 'uploading' && (
          <div className="status-block">
            <div className="status-icon">⬆️</div>
            <h2 className="status-title">Uploading your video</h2>
            <p className="status-sub">{uploadPct}% complete</p>
            <div className="progress-track">
              <div className="progress-bar" style={{ width: `${uploadPct}%` }} />
            </div>
            <p className="status-note">Keep this screen open</p>
          </div>
        )}

        {/* ANALYZING */}
        {stage === 'analyzing' && (
          <div className="status-block">
            <div className="pulse-ring">
              <div className="pulse-dot" />
            </div>
            <h2 className="status-title">Analyzing your swing</h2>
            <p className="status-sub">Finding your biggest flaw…</p>
            <div className="analysis-steps">
              <div className="step done-step">✓ Video uploaded</div>
              <div className="step active-step">⟳ Frame-by-frame review</div>
              <div className="step pending-step">○ Generating your fix</div>
            </div>
          </div>
        )}

        {/* DONE */}
        {stage === 'done' && (
          <div className="status-block">
            <div className="done-icon">✅</div>
            <h2 className="status-title">Analysis complete!</h2>
            <p className="status-sub">Taking you to your result…</p>
          </div>
        )}

        {/* ERROR */}
        {stage === 'error' && (
          <div className="status-block">
            <div className="status-icon">⚠️</div>
            <h2 className="status-title">Something went wrong</h2>
            <p className="error-detail">{errorMsg}</p>
            <button className="btn-analyze" onClick={reset}>Try Again</button>
          </div>
        )}
      </main>

      <style>{pageStyle}</style>
    </div>
  );
}

const pageStyle = `
  * { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
  body { background: #080c08; }

  .page {
    min-height: 100vh;
    background: #080c08;
    font-family: 'DM Sans', system-ui, sans-serif;
    max-width: 430px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
  }

  /* Header */
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
    padding: 6px 0;
  }

  /* Main */
  .main {
    flex: 1;
    padding: 28px 20px 48px;
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  /* Idle state */
  .headline-block { display: flex; flex-direction: column; gap: 10px; }
  .h1 { font-size: 28px; font-weight: 800; color: #f0f0ee; letter-spacing: -0.7px; }
  .sub { font-size: 15px; color: #4a5a44; line-height: 1.55; }

  /* Dropzone */
  .dropzone {
    border: 2px dashed rgba(168,240,64,0.2);
    border-radius: 18px;
    padding: 32px 20px;
    text-align: center;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    background: rgba(168,240,64,0.03);
    transition: border-color 0.15s, background 0.15s;
    min-height: 140px;
    justify-content: center;
  }
  .dropzone:active, .dropzone.has-file {
    border-color: rgba(168,240,64,0.5);
    background: rgba(168,240,64,0.06);
  }
  .dz-icon { font-size: 32px; margin-bottom: 4px; }
  .dz-label { font-size: 16px; font-weight: 700; color: #a8f040; }
  .dz-hint { font-size: 12px; color: #3a4a35; }
  .dz-name { font-size: 14px; font-weight: 600; color: #e0e8d8; }
  .dz-size { font-size: 12px; color: #5a6a54; }

  /* View type toggle */
  .view-section { display: flex; flex-direction: column; gap: 10px; }
  .view-label { font-size: 13px; font-weight: 600; color: #5a6a54; }
  .view-toggle { display: flex; gap: 10px; }
  .view-btn {
    flex: 1;
    background: rgba(255,255,255,0.03);
    border: 1.5px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 14px 12px;
    cursor: pointer;
    font-family: inherit;
    display: flex;
    flex-direction: column;
    gap: 4px;
    transition: all 0.15s;
  }
  .view-btn.active {
    border-color: #a8f040;
    background: rgba(168,240,64,0.08);
  }
  .vb-label { font-size: 14px; font-weight: 700; color: #e0e8d8; }
  .view-btn.active .vb-label { color: #a8f040; }
  .vb-hint { font-size: 11px; color: #3a4a35; }

  /* CTA */
  .btn-analyze {
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
    -webkit-appearance: none;
    transition: transform 0.12s, opacity 0.15s;
  }
  .btn-analyze:disabled { opacity: 0.3; cursor: not-allowed; }
  .btn-analyze:active:not(:disabled) { transform: scale(0.97); }
  .notice { font-size: 12px; color: #2a3a25; text-align: center; }

  /* Status states */
  .status-block {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 16px;
    min-height: 60vh;
    text-align: center;
  }
  .status-icon { font-size: 48px; }
  .done-icon { font-size: 56px; }
  .status-title { font-size: 24px; font-weight: 800; color: #f0f0ee; letter-spacing: -0.5px; }
  .status-sub { font-size: 15px; color: #4a5a44; }
  .status-note { font-size: 12px; color: #2a3a25; }
  .error-detail {
    font-size: 13px;
    color: #f06040;
    background: rgba(240,96,64,0.08);
    border: 1px solid rgba(240,96,64,0.15);
    border-radius: 10px;
    padding: 12px 16px;
    max-width: 320px;
    line-height: 1.5;
  }

  /* Progress bar */
  .progress-track {
    width: 100%;
    max-width: 280px;
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 3px;
    overflow: hidden;
  }
  .progress-bar {
    height: 100%;
    background: #a8f040;
    border-radius: 3px;
    transition: width 0.3s ease;
  }

  /* Pulse ring */
  .pulse-ring {
    width: 72px; height: 72px;
    border-radius: 50%;
    background: rgba(168,240,64,0.08);
    display: flex;
    align-items: center;
    justify-content: center;
    animation: ring-pulse 1.8s ease-in-out infinite;
  }
  .pulse-dot {
    width: 32px; height: 32px;
    border-radius: 50%;
    background: #a8f040;
    animation: dot-pulse 1.8s ease-in-out infinite;
  }
  @keyframes ring-pulse {
    0%, 100% { transform: scale(1); opacity: 0.8; }
    50% { transform: scale(1.15); opacity: 0.4; }
  }
  @keyframes dot-pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(0.85); }
  }

  /* Analysis steps */
  .analysis-steps {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-top: 8px;
    width: 100%;
    max-width: 240px;
    text-align: left;
  }
  .step { font-size: 13px; padding: 8px 12px; border-radius: 8px; }
  .done-step { color: #a8f040; background: rgba(168,240,64,0.08); }
  .active-step { color: #f0f0ee; background: rgba(255,255,255,0.05); }
  .pending-step { color: #2a3a25; }

  /* Spinner (loading state) */
  .spinner {
    width: 36px; height: 36px;
    border: 3px solid rgba(168,240,64,0.15);
    border-top-color: #a8f040;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: auto;
    margin-top: 40vh;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
`;
