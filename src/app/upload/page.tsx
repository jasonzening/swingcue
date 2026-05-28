'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { createClient } from '@/lib/supabase/client';

// PR-8i.0 upload-flow refactor — auto-start on file select; bake
// camera_angle='face_on' (DTL no longer supported in the WHAM
// pipeline); defer club selection to PR-8i.1 (collected on the
// processing-wait screen, not pre-flight). No camera-angle picker
// + no club picker = no config screen between "file picked" and
// "upload running". File pick is the commit point.

type Stage = 'idle' | 'uploading' | 'analyzing' | 'done' | 'error';
type SourceType = 'recorded' | 'uploaded';

export default function UploadPage() {
  const router = useRouter();
  const [checking, setChecking] = useState(true);
  const [stage, setStage] = useState<Stage>('idle');
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [uploadPct, setUploadPct] = useState(0);
  const [errorMsg, setErrorMsg] = useState('');
  const [videoId, setVideoId] = useState('');
  const [analysisStep, setAnalysisStep] = useState(0);
  const recordRef = useRef<HTMLInputElement>(null);
  const uploadRef = useRef<HTMLInputElement>(null);
  const sourceTypeRef = useRef<SourceType>('uploaded');
  // Hold the picked File on a ref instead of state so handleAnalyze can
  // be invoked immediately from handleFile without waiting for a state
  // batch + render cycle (PR-8i.0 auto-start).
  const fileRef = useRef<File | null>(null);

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
    fileRef.current = f;
    sourceTypeRef.current = source;
    const url = URL.createObjectURL(f);
    setPreviewUrl(url);
    // PR-8i.0 auto-start: skip the config-screen review step. Upload
    // begins immediately on file pick. handleAnalyze reads from fileRef
    // so no state-batch race.
    void handleAnalyze(f);
  };

  const handleSignOut = useCallback(async () => {
    const supabase = createClient();
    await supabase.auth.signOut();
    router.replace('/sign-in');
  }, [router]);

  const reset = () => {
    setStage('idle');
    fileRef.current = null;
    setPreviewUrl(null);
    setUploadPct(0);
    setErrorMsg('');
    setVideoId('');
    setAnalysisStep(0);
    if (recordRef.current) recordRef.current.value = '';
    if (uploadRef.current) uploadRef.current.value = '';
  };

  const handleAnalyze = async (fileOverride?: File) => {
    // PR-8i.0: accept the picked file as a param so auto-start
    // doesn't have to wait for state propagation.
    const file = fileOverride ?? fileRef.current;
    if (!file) return;
    setErrorMsg('');

    // PR-8c.5 Bug 2 fix: capture vid in outer scope so the generic
    // catch can mark swing_videos.status='failed' if anything between
    // INSERT and successful storage upload throws. Without this the
    // row sits at status='uploaded' with storage_path='' = ghost row.
    let vid: string | null = null;
    const supabase = createClient();

    const markUploadFailed = async (
      videoId: string,
      errorCode: string,
      errorMessage: string,
    ) => {
      try {
        await supabase.from('swing_videos').update({
          status:        'failed',
          error_code:    errorCode,
          error_message: errorMessage.slice(0, 2000),
          processing_completed_at: new Date().toISOString(),
        }).eq('id', videoId);
      } catch (patchErr) {
        // Don't blow up the UI flow; the original error still surfaces.
        console.error('[upload] markUploadFailed PATCH error:', patchErr);
      }
    };

    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) { router.replace('/sign-in'); return; }

      // Stage 1: Upload
      setStage('uploading');
      setUploadPct(10);

      // Create DB record.
      // PR-8i.0: view_type baked to 'face_on' (DTL not supported by
      // the WHAM pipeline). club_type baked to 'unknown'; PR-8i.1
      // will collect the real value on the processing-wait screen
      // and PATCH it before the user lands on /result.
      const { data: videoRow, error: insertErr } = await supabase
        .from('swing_videos')
        .insert({
          user_id: user.id,
          storage_path: '',
          original_filename: file.name,
          file_size_bytes: file.size,
          view_type: 'face_on',
          status: 'uploaded',
          source_type: sourceTypeRef.current,
          club_type: 'unknown',
        })
        .select('id')
        .single();

      if (insertErr || !videoRow) throw new Error(insertErr?.message ?? 'Failed to create record');
      vid = videoRow.id as string;
      setVideoId(vid);
      setUploadPct(30);

      // PR-8c.5 Bug 1 fix: deterministic ASCII-only storage key.
      // Previously used `${user.id}/${vid}/${file.name}` which broke
      // for non-ASCII filenames (CJK, accented chars). Supabase Storage
      // rejected those with "Invalid key" — international users 100%
      // affected. Fix: ignore client-side filename for the storage path;
      // keep file extension only (sanitized to a known-safe whitelist
      // so playback still gets a correct Content-Type). The full
      // original filename is preserved separately in
      // swing_videos.original_filename for history/download UX.
      const rawExt = (file.name.split('.').pop() ?? '').toLowerCase().replace(/[^a-z0-9]/g, '');
      const safeExt = ['mp4', 'mov', 'webm', 'avi', 'm4v', 'mkv'].includes(rawExt) ? rawExt : 'mp4';
      const storagePath = `${user.id}/${vid}/${vid}.${safeExt}`;
      const { error: uploadErr } = await supabase.storage
        .from('swing-videos')
        .upload(storagePath, file, {
          upsert: false,
          // Ensure correct Content-Type even when the storage key extension
          // was forced to 'mp4' for a non-standard input.
          contentType: file.type || `video/${safeExt}`,
        });

      if (uploadErr) {
        // PR-8c.5 Bug 2 fix: PATCH swing_videos to 'failed' BEFORE
        // throwing so the row carries the failure context. Classify
        // the error: Supabase "Invalid key" was the historical Bug 1
        // symptom; should never fire post-fix but kept for defense.
        const msg = uploadErr.message;
        const errorCode = /invalid key/i.test(msg)
          ? 'invalid_storage_key'
          : /network|fetch|connection/i.test(msg)
            ? 'storage_network_error'
            : 'storage_upload_failed';
        await markUploadFailed(vid, errorCode, msg);
        throw new Error(`Upload failed: ${msg}`);
      }
      setUploadPct(80);

      // Update storage path. If THIS PATCH fails the blob is orphaned
      // in storage + the row points at empty path; still mark the row
      // 'failed' so the user can retry cleanly (the orphan blob is
      // bounded by user storage quota — not a hard correctness issue
      // here; a sweeper job would be a separate PR).
      const { error: pathUpdateErr } = await supabase.from('swing_videos')
        .update({ storage_path: storagePath })
        .eq('id', vid);
      if (pathUpdateErr) {
        await markUploadFailed(vid, 'storage_path_update_failed', pathUpdateErr.message);
        throw new Error(`DB update failed: ${pathUpdateErr.message}`);
      }

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
        // Note: /api/analyze (PR-8c.2) already writes swing_videos.status
        // and swing_analysis on backend failure; we just surface the
        // error to the user and skip our own markUploadFailed write.
        throw new Error(body.error_message ?? body.error ?? 'Analysis failed');
      }

      setStage('done');
      setTimeout(() => router.push(`/result/${vid}`), 700);

    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Something went wrong. Please try again.';
      // PR-8c.5 Bug 2 (defense in depth): if we got past the INSERT
      // but markUploadFailed wasn't already invoked above, ensure the
      // row carries an error so it doesn't sit as 'uploaded' ghost.
      // This catches unexpected throws (auth, network mid-stage, etc.).
      if (vid) {
        // Best-effort: only PATCH if the row isn't already in a terminal
        // failed state (some paths above already marked it). PostgREST
        // doesn't expose "WHERE status != 'failed'" cheaply from the
        // client; we just over-write — repeated PATCH with the same
        // failed fields is idempotent.
        await markUploadFailed(vid, 'upload_pipeline_error', msg);
      }
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

        {/* ── IDLE STATE (entry screen — auto-start on file pick). ──
            PR-8i.0: the previous "file picked → config screen with
            camera-angle + club pickers → click Analyze" middle step
            is removed. handleFile → handleAnalyze fires in one
            stroke, so there's no on-screen state between "no file"
            and "uploading". Error state has its own screen below
            (with a Try Again CTA that calls reset()). */}
        {stage === 'idle' && (
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
              <p className="tip">💡 Film face-on (camera in front of you)</p>
              <p className="tip">📱 Landscape mode works best · 4–10 seconds is ideal</p>
            </div>
          </div>
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

        {/* ── ERROR STATE ──
            PR-8i.0: no longer guarded on `file` (state removed —
            errors can fire after fileRef was cleared by a partial
            reset). The errorMsg + Try Again CTA + reset() flow back
            to the entry screen are the recovery path. */}
        {stage === 'error' && (
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
