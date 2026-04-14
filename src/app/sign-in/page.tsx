'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { createClient } from '@/lib/supabase/client';

export default function SignInPage() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [resetSent, setResetSent] = useState(false);
  const [showReset, setShowReset] = useState(false);

  async function handleSignIn(e: React.FormEvent) {
    e.preventDefault();
    if (!email || !password) return;
    setError('');
    setLoading(true);
    const supabase = createClient();
    const { error: authError } = await supabase.auth.signInWithPassword({ email: email.trim(), password });
    if (authError) {
      // Show exact error for debugging
      setError(authError.message || 'Sign in failed. Check your email and password.');
      setLoading(false);
    } else {
      router.push('/upload');
    }
  }

  async function handleReset(e: React.FormEvent) {
    e.preventDefault();
    if (!email) { setError('Enter your email first.'); return; }
    setLoading(true);
    const supabase = createClient();
    await supabase.auth.resetPasswordForEmail(email.trim(), {
      redirectTo: `${window.location.origin}/update-password`,
    });
    setResetSent(true);
    setLoading(false);
  }

  return (
    <div className="page">
      <div className="card">
        <span className="logo">SwingCue</span>
        <h1 className="title">Sign in</h1>
        <p className="sub">Internal testing access</p>

        {resetSent ? (
          <div className="reset-sent">
            <p>✅ Password reset email sent to <strong>{email}</strong></p>
            <p>Check your inbox and follow the link.</p>
            <button className="link-btn" onClick={() => { setResetSent(false); setShowReset(false); }}>
              ← Back to sign in
            </button>
          </div>
        ) : (
          <form onSubmit={showReset ? handleReset : handleSignIn} className="form">
            <div className="field">
              <label className="label">Email</label>
              <input
                type="email"
                className="input"
                placeholder="you@example.com"
                value={email}
                onChange={e => setEmail(e.target.value)}
                required
                autoComplete="email"
                autoCapitalize="none"
                autoCorrect="off"
                spellCheck={false}
              />
            </div>

            {!showReset && (
              <div className="field">
                <label className="label">Password</label>
                <input
                  type="password"
                  className="input"
                  placeholder="••••••••"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  required
                  autoComplete="current-password"
                />
              </div>
            )}

            {error && (
              <div className="error-box">
                <p>{error}</p>
              </div>
            )}

            <button type="submit" className="btn" disabled={loading}>
              {loading ? (showReset ? 'Sending…' : 'Signing in…') : (showReset ? 'Send Reset Email →' : 'Sign in →')}
            </button>

            <button
              type="button"
              className="link-btn"
              onClick={() => { setShowReset(!showReset); setError(''); }}
            >
              {showReset ? '← Back to sign in' : 'Forgot password?'}
            </button>
          </form>
        )}

        <a href="/" className="back-link">← Back to homepage</a>
      </div>

      <style>{css}</style>
    </div>
  );
}

const css = `
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #080c08; }
  .page {
    min-height: 100vh; background: #080c08;
    display: flex; align-items: center; justify-content: center;
    padding: 24px 20px;
    font-family: 'DM Sans', system-ui, sans-serif;
  }
  .card {
    width: 100%; max-width: 390px;
    background: #0c130b; border: 1px solid rgba(168,240,64,0.12);
    border-radius: 24px; padding: 36px 28px 32px;
    display: flex; flex-direction: column; gap: 8px;
  }
  .logo { font-size: 18px; font-weight: 800; color: #a8f040; letter-spacing: -0.3px; margin-bottom: 8px; }
  .title { font-size: 26px; font-weight: 800; color: #f0f0ee; letter-spacing: -0.6px; }
  .sub { font-size: 13px; color: #3a4a35; margin-bottom: 16px; }

  .form { display: flex; flex-direction: column; gap: 16px; margin-top: 8px; }
  .field { display: flex; flex-direction: column; gap: 6px; }
  .label { font-size: 13px; font-weight: 600; color: #7a8a72; }
  .input {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px; padding: 14px 16px; font-size: 16px; color: #f0f0ee;
    font-family: inherit; width: 100%; outline: none; -webkit-appearance: none;
    transition: border-color 0.15s;
  }
  .input:focus { border-color: rgba(168,240,64,0.5); }
  .input::placeholder { color: #2a3a25; }

  .error-box {
    background: rgba(240,96,64,0.10); border: 1px solid rgba(240,96,64,0.20);
    border-radius: 10px; padding: 12px 14px;
  }
  .error-box p { font-size: 13px; color: #f06040; line-height: 1.5; }

  .btn {
    background: #a8f040; color: #080c08;
    font-family: inherit; font-size: 16px; font-weight: 800;
    height: 52px; border-radius: 100px; border: none; cursor: pointer; width: 100%;
    transition: opacity 0.15s, transform 0.12s; -webkit-appearance: none;
  }
  .btn:disabled { opacity: 0.6; }
  .btn:active:not(:disabled) { transform: scale(0.97); }

  .link-btn {
    background: none; border: none; cursor: pointer; font-family: inherit;
    font-size: 13px; color: #4a5a44; text-align: center; padding: 4px 0;
  }
  .link-btn:hover { color: #a8f040; }

  .reset-sent {
    background: rgba(168,240,64,0.06); border: 1px solid rgba(168,240,64,0.15);
    border-radius: 12px; padding: 20px; display: flex; flex-direction: column;
    gap: 10px; margin-top: 8px;
  }
  .reset-sent p { font-size: 14px; color: #8a9a82; line-height: 1.5; }
  .reset-sent strong { color: #a8f040; }

  .back-link { font-size: 13px; color: #2a3a25; text-decoration: none; text-align: center; margin-top: 8px; }
`;
