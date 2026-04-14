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

  async function handleSignIn(e: React.FormEvent) {
    e.preventDefault();
    setError('');
    setLoading(true);
    const supabase = createClient();
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) {
      setError('Wrong email or password. Try again.');
      setLoading(false);
    } else {
      router.push('/upload');
    }
  }

  return (
    <div className="page">
      <div className="card">
        <span className="logo">SwingCue</span>
        <h1 className="title">Sign in</h1>
        <p className="sub">Internal testing access</p>

        <form onSubmit={handleSignIn} className="form">
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
            />
          </div>

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

          {error && <p className="error">{error}</p>}

          <button type="submit" className="btn" disabled={loading}>
            {loading ? 'Signing in…' : 'Sign in →'}
          </button>
        </form>

        <a href="/" className="back">← Back to homepage</a>
      </div>

      <style>{`
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #080c08; }
        .page {
          min-height: 100vh;
          background: #080c08;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 24px 20px;
          font-family: 'DM Sans', system-ui, sans-serif;
        }
        .card {
          width: 100%;
          max-width: 390px;
          background: #0c130b;
          border: 1px solid rgba(168,240,64,0.12);
          border-radius: 24px;
          padding: 36px 28px 32px;
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .logo {
          font-size: 18px;
          font-weight: 800;
          color: #a8f040;
          letter-spacing: -0.3px;
          margin-bottom: 8px;
        }
        .title {
          font-size: 26px;
          font-weight: 800;
          color: #f0f0ee;
          letter-spacing: -0.6px;
        }
        .sub {
          font-size: 13px;
          color: #3a4a35;
          margin-bottom: 16px;
        }
        .form {
          display: flex;
          flex-direction: column;
          gap: 16px;
          margin-top: 8px;
        }
        .field { display: flex; flex-direction: column; gap: 6px; }
        .label {
          font-size: 13px;
          font-weight: 600;
          color: #7a8a72;
        }
        .input {
          background: rgba(255,255,255,0.04);
          border: 1px solid rgba(255,255,255,0.1);
          border-radius: 12px;
          padding: 14px 16px;
          font-size: 16px;
          color: #f0f0ee;
          font-family: inherit;
          width: 100%;
          outline: none;
          -webkit-appearance: none;
          transition: border-color 0.15s;
        }
        .input:focus { border-color: rgba(168,240,64,0.5); }
        .input::placeholder { color: #2a3a25; }
        .error {
          font-size: 13px;
          color: #f06040;
          background: rgba(240,96,64,0.1);
          border: 1px solid rgba(240,96,64,0.2);
          border-radius: 8px;
          padding: 10px 14px;
        }
        .btn {
          background: #a8f040;
          color: #080c08;
          font-family: inherit;
          font-size: 16px;
          font-weight: 800;
          height: 52px;
          border-radius: 100px;
          border: none;
          cursor: pointer;
          width: 100%;
          margin-top: 4px;
          transition: opacity 0.15s, transform 0.12s;
          -webkit-appearance: none;
        }
        .btn:disabled { opacity: 0.6; }
        .btn:active:not(:disabled) { transform: scale(0.97); }
        .back {
          font-size: 13px;
          color: #3a4a35;
          text-decoration: none;
          text-align: center;
          margin-top: 8px;
        }
      `}</style>
    </div>
  );
}
