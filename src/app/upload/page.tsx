'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { createClient } from '@/lib/supabase/client';
import type { SupabaseClient } from '@supabase/supabase-js';

// PR-8i.0 upload-flow refactor — auto-start on file select; bake
// camera_angle='face_on' (DTL no longer supported in the WHAM
// pipeline); defer club selection to PR-8i.1 (collected on the
// processing-wait screen, not pre-flight). No camera-angle picker
// + no club picker = no config screen between "file picked" and
// "upload running". File pick is the commit point.
//
// PR-8j-hotfix Phase A+B (2026-05-28) — fake "Analyzing your swing"
// progress UI deleted entirely (was leaked PR-8d.2-era dead code
// claiming WHAM steps were complete before WHAM had even started).
// Flow now: file pick → INSERT swing_videos → router.push immediately
// → storage upload + /api/analyze run in background after redirect.
// The /upload page is on-screen only for the brief INSERT round-trip
// (~300ms) and shows a minimal centered "Uploading…" spinner during
// that window. All real progress feedback is on the result page
// (ProcessingScreen handles the wham_status='processing' phase).

type Stage = 'idle' | 'uploading' | 'error';
type SourceType = 'recorded' | 'uploaded';

export default function UploadPage() {
  const router = useRouter();
  const [checking, setChecking] = useState(true);
  const [stage, setStage] = useState<Stage>('idle');
  const [errorMsg, setErrorMsg] = useState('');
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

  const handleSignOut = useCallback(async () => {
    const supabase = createClient();
    await supabase.auth.signOut();
    router.replace('/sign-in');
  }, [router]);

  const reset = () => {
    setStage('idle');
    fileRef.current = null;
    setErrorMsg('');
    if (recordRef.current) recordRef.current.value = '';
    if (uploadRef.current) uploadRef.current.value = '';
  };

  // PR-8j-hotfix Phase B: foreground does only auth + INSERT, then
  // redirects in the same tick. Storage upload + /api/analyze run in
  // a fire-and-forget background promise so the user lands on
  // /result/{vid} within ~1s of file pick. The browser keeps the
  // background promise alive across the client-side route change
  // (same SPA context); if the user closes the tab mid-upload, the
  // row stays at status='uploaded' and a future sweeper job (out of
  // scope here) can clean it up.
  const handleAnalyze = async (fileOverride?: File) => {
    const file = fileOverride ?? fileRef.current;
    if (!file) return;
    setErrorMsg('');
    setStage('uploading');

    const supabase = createClient();
    let vid: string;
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) { router.replace('/sign-in'); return; }

      // INSERT only — the row carries enough state for the result
      // page to render a Loading view until /api/analyze creates the
      // swing_analysis row + flips wham_status to 'processing'.
      // PR-8i.0: view_type baked to 'face_on'; club_type baked to
      // 'unknown' and PATCHed later by PR-8i.1's in-process picker.
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
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Something went wrong. Please try again.';
      setErrorMsg(msg);
      setStage('error');
      return;
    }

    // Redirect in the same tick. Background promise continues uploading.
    router.push(`/result/${vid}`);
    void backgroundUploadAndAnalyze(supabase, vid, file);
  };

  const handleFile = (f: File, source: SourceType) => {
    if (!f.type.startsWith('video/')) {
      setErrorMsg('Please select a video file (MP4, MOV, AVI, etc.)');
      setStage('error');
      return;
    }
    if (f.size > 500 * 1024 * 1024) {
      setErrorMsg('Video must be under 500 MB');
      setStage('error');
      return;
    }
    setErrorMsg('');
    fileRef.current = f;
    sourceTypeRef.current = source;
    // PR-8i.0 auto-start: skip the config-screen review step. Upload
    // begins immediately on file pick. handleAnalyze reads from fileRef
    // so no state-batch race.
    void handleAnalyze(f);
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

        {/* ── UPLOADING STATE ──
            PR-8j-hotfix Phase A: minimal centered white spinner ONLY.
            On-screen only during the INSERT round-trip (~300ms before
            router.push fires). NO step list, NO progress bar, NO fake
            "Analysis complete" card. Real progress UX lives on the
            result page (ProcessingScreen). */}
        {stage === 'uploading' && (
          <div className="uploading-screen">
            <div className="spinner" />
            <p className="uploading-text">Uploading…</p>
          </div>
        )}

        {/* ── ERROR STATE ──
            Fires when INSERT itself throws (auth lost, RLS reject,
            network error before any row exists). Recovery is reset()
            back to the entry screen. */}
        {stage === 'error' && (
          <div className="uploading-screen">
            <div className="status-icon-lg">⚠️</div>
            <h2 className="status-title">Something went wrong</h2>
            <p className="error-detail">{errorMsg}</p>
            <button className="try-again-btn" onClick={reset}>
              Try Again
            </button>
          </div>
        )}

      </main>
      <style>{css}</style>
    </div>
  );
}

// PR-8j-hotfix Phase B: fire-and-forget background pipeline. Runs
// AFTER router.push, so the user is already on /result/{vid} by the
// time this starts the storage upload. Errors are persisted to the
// swing_videos row (status='failed' + error_code/message); the
// result page reads that row and renders FailedScreen, so no UI
// surface in /upload is needed for background failures. Local
// console.error captures the throw context for debugging when the
// user reports a failure that didn't reach the DB.
const backgroundUploadAndAnalyze = async (
  supabase: SupabaseClient,
  vid: string,
  file: File,
) => {
  const markUploadFailed = async (errorCode: string, errorMessage: string) => {
    try {
      await supabase.from('swing_videos').update({
        status: 'failed',
        error_code: errorCode,
        error_message: errorMessage.slice(0, 2000),
        processing_completed_at: new Date().toISOString(),
      }).eq('id', vid);
    } catch (patchErr) {
      console.error('[upload/bg] markUploadFailed PATCH error:', patchErr);
    }
  };

  try {
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) return;

    // PR-8c.5 Bug 1 fix preserved — deterministic ASCII-only storage
    // key; the original filename lives on swing_videos.original_filename.
    const rawExt = (file.name.split('.').pop() ?? '').toLowerCase().replace(/[^a-z0-9]/g, '');
    const safeExt = ['mp4', 'mov', 'webm', 'avi', 'm4v', 'mkv'].includes(rawExt) ? rawExt : 'mp4';
    const storagePath = `${user.id}/${vid}/${vid}.${safeExt}`;

    const { error: uploadErr } = await supabase.storage
      .from('swing-videos')
      .upload(storagePath, file, {
        upsert: false,
        contentType: file.type || `video/${safeExt}`,
      });

    if (uploadErr) {
      const msg = uploadErr.message;
      const errorCode = /invalid key/i.test(msg)
        ? 'invalid_storage_key'
        : /network|fetch|connection/i.test(msg)
          ? 'storage_network_error'
          : 'storage_upload_failed';
      await markUploadFailed(errorCode, msg);
      return;
    }

    const { error: pathUpdateErr } = await supabase.from('swing_videos')
      .update({ storage_path: storagePath })
      .eq('id', vid);
    if (pathUpdateErr) {
      await markUploadFailed('storage_path_update_failed', pathUpdateErr.message);
      return;
    }

    const res = await fetch(`/api/analyze/${vid}`, { method: 'POST' });
    if (!res.ok) {
      // /api/analyze (PR-8c.2) writes its own failure to swing_videos.
      // markUploadFailed here is defensive — covers the case where the
      // request never reached the route handler (network mid-flight,
      // 502 from the platform, etc.) and the backend never got a
      // chance to set the failure state itself.
      const body = await res.json().catch(() => ({}));
      const msg = body.error_message ?? body.error ?? `Analysis HTTP ${res.status}`;
      await markUploadFailed('analyze_api_failed', String(msg));
      return;
    }
    // Success — swing_analysis row + wham_status='processing' now
    // present on the row. Result page polling picks up the new state.
  } catch (err) {
    console.error('[upload/bg] background pipeline threw:', err);
    await markUploadFailed(
      'upload_pipeline_error',
      err instanceof Error ? err.message : 'unknown',
    );
  }
};

const css = `
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
  body { background: var(--bg-primary); }

  .page {
    min-height: 100vh; background: var(--bg-primary);
    font-family: 'DM Sans', system-ui, sans-serif;
    max-width: 430px; margin: 0 auto;
    display: flex; flex-direction: column;
    color: var(--text-primary);
  }
  .page.center { align-items: center; justify-content: center; }

  .header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 14px 20px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    position: sticky; top: 0;
    background: rgba(5,8,10,0.95);
    backdrop-filter: blur(16px); z-index: 50;
  }
  .logo { font-size: 18px; font-weight: 800; color: var(--text-primary); letter-spacing: -0.3px; text-decoration: none; }
  .header-right { display: flex; align-items: center; gap: 14px; }
  .hist-link { font-size: 13px; font-weight: 600; color: var(--text-muted); text-decoration: none; }
  .signout-btn { font-size: 13px; font-weight: 600; color: var(--text-muted); background: none; border: none; cursor: pointer; font-family: inherit; }

  .main { flex: 1; padding: 0 0 52px; display: flex; flex-direction: column; }

  /* ── ENTRY SCREEN ── */
  .entry-screen { padding: 28px 20px; display: flex; flex-direction: column; gap: 24px; }
  .h1 { font-size: 26px; font-weight: 800; color: var(--text-primary); letter-spacing: -0.6px; }
  .sub { font-size: 15px; color: var(--text-muted); line-height: 1.6; }

  .two-buttons { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .entry-btn {
    display: flex; flex-direction: column; align-items: center; gap: 6px;
    padding: 22px 12px; border-radius: 18px; border: 1.5px solid;
    cursor: pointer; font-family: inherit;
    transition: transform 0.12s, opacity 0.15s;
  }
  .entry-btn:active { transform: scale(0.96); }
  .record-btn {
    background: rgba(255, 255, 255, 0.06);
    border-color: rgba(255, 255, 255, 0.3);
  }
  .upload-btn {
    background: rgba(255,255,255,0.03);
    border-color: rgba(255,255,255,0.1);
  }
  .entry-icon { font-size: 28px; }
  .entry-label { font-size: 15px; font-weight: 700; color: var(--text-primary); }
  .entry-hint { font-size: 12px; color: var(--text-muted); }

  .tips { display: flex; flex-direction: column; gap: 8px; }
  .tip { font-size: 12px; color: var(--text-muted); line-height: 1.5; }

  /* ── UPLOADING / ERROR STATE ──
     Single layout for both. Centered column, large vertical breathing
     room. Minimal — the visible affordance is just the spinner + a
     two-word label. Everything else lives on the result page. */
  .uploading-screen {
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 18px; padding: 40px 24px; min-height: 70vh;
    text-align: center;
  }
  .uploading-text { font-size: 14px; font-weight: 600; color: var(--text-muted); letter-spacing: 0.2px; }

  /* ── ERROR-only chrome ── */
  .status-icon-lg { font-size: 52px; }
  .status-title { font-size: 22px; font-weight: 800; color: var(--text-primary); letter-spacing: -0.4px; }
  .error-detail {
    font-size: 13px; color: #f06040;
    background: rgba(240,96,64,0.08); border: 1px solid rgba(240,96,64,0.15);
    border-radius: 10px; padding: 12px 16px; max-width: 320px; line-height: 1.5;
  }
  .try-again-btn {
    background: var(--text-primary); color: #080c08;
    font-family: inherit; font-size: 16px; font-weight: 800;
    height: 52px; border-radius: 100px; border: none;
    cursor: pointer; padding: 0 28px; margin-top: 8px;
    transition: transform 0.12s; -webkit-appearance: none;
  }
  .try-again-btn:active { transform: scale(0.97); }

  /* ── SPINNER ──
     White ring + white top-arc. Used for both the auth-check on mount
     and the post-INSERT uploading flash. */
  .spinner {
    width: 36px; height: 36px;
    border: 3px solid rgba(255, 255, 255, 0.15); border-top-color: var(--text-primary);
    border-radius: 50%; animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
`;
