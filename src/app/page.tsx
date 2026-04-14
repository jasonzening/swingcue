export default function Home() {
  return (
    <main className="swingcue-home">

      {/* ── NAV ── */}
      <nav className="nav">
        <span className="logo">SwingCue</span>
        <a href="#waitlist" className="btn-nav">Get Early Access</a>
      </nav>

      {/* ── HERO ── */}
      <section className="hero">
        <div className="hero-badge">AI Golf Coach · In Your Pocket</div>
        <h1 className="hero-title">
          Fix your swing.<br />
          <span className="accent">Not your score.</span>
        </h1>
        <p className="hero-sub">
          Upload your swing video. See exactly what&apos;s wrong.
          Get one clear fix — no jargon, no guesswork.
        </p>
        <a href="#waitlist" className="btn-primary">
          Analyse My Swing →
        </a>
        <p className="hero-note">Free to try · No app download needed</p>
      </section>

      {/* ── DEMO PHONE MOCKUP ── */}
      <section className="mockup-wrap">
        <div className="phone">
          <div className="phone-screen">
            <div className="result-card">
              <div className="result-label">MAIN ISSUE</div>
              <div className="result-issue">Early Extension</div>
              <div className="result-divider" />
              <div className="result-label">FIX CUE</div>
              <div className="result-cue">&ldquo;Stay in your posture through impact&rdquo;</div>
              <div className="result-divider" />
              <div className="result-label">DRILL</div>
              <div className="result-drill">Wall drill — 5 reps before every round</div>
              <div className="score-row">
                <div className="score-pill">Score 72 / 100</div>
                <div className="severity-pill">Medium</div>
              </div>
            </div>
          </div>
        </div>
        <div className="phone-glow" />
      </section>

      {/* ── HOW IT WORKS ── */}
      <section className="steps-section">
        <h2 className="section-title">Three steps.<br />One clear answer.</h2>
        <ol className="steps">
          <li className="step">
            <span className="step-num">01</span>
            <div>
              <strong>Film your swing</strong>
              <p>Face-on or down-the-line — your phone camera is all you need.</p>
            </div>
          </li>
          <li className="step">
            <span className="step-num">02</span>
            <div>
              <strong>Upload &amp; analyse</strong>
              <p>Our AI reviews your video frame by frame in seconds.</p>
            </div>
          </li>
          <li className="step">
            <span className="step-num">03</span>
            <div>
              <strong>See your fix</strong>
              <p>One issue. One cue. One drill. Nothing else to overwhelm you.</p>
            </div>
          </li>
        </ol>
      </section>

      {/* ── ONE-ONE-ONE ── */}
      <section className="ooo-section">
        <div className="ooo-grid">
          <div className="ooo-card">
            <div className="ooo-icon">🎯</div>
            <div>
              <div className="ooo-label">One Issue</div>
              <div className="ooo-desc">The single biggest thing holding you back</div>
            </div>
          </div>
          <div className="ooo-card">
            <div className="ooo-icon">💬</div>
            <div>
              <div className="ooo-label">One Cue</div>
              <div className="ooo-desc">A plain-English phrase to carry to the range</div>
            </div>
          </div>
          <div className="ooo-card">
            <div className="ooo-icon">🏌️</div>
            <div>
              <div className="ooo-label">One Drill</div>
              <div className="ooo-desc">The fastest path from knowing to improving</div>
            </div>
          </div>
        </div>
      </section>

      {/* ── WAITLIST ── */}
      <section className="waitlist-section" id="waitlist">
        <h2 className="waitlist-title">Ready to stop guessing?</h2>
        <p className="waitlist-sub">
          Join early access — be the first to upload your swing.
        </p>
        <a href="mailto:support@swingcue.ai?subject=SwingCue Early Access" className="btn-primary">
          Request Early Access
        </a>
        <p className="waitlist-privacy">No spam · support@swingcue.ai</p>
      </section>

      {/* ── FOOTER ── */}
      <footer className="footer">
        <span className="logo small">SwingCue</span>
        <p>Questions? <a href="mailto:support@swingcue.ai">support@swingcue.ai</a></p>
        <p className="footer-copy">© 2025 SwingCue. All rights reserved.</p>
      </footer>

      <style>{`
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        .swingcue-home {
          min-height: 100vh;
          background: #080c08;
          color: #f0f0ee;
          font-family: 'DM Sans', system-ui, sans-serif;
          max-width: 430px;
          margin: 0 auto;
          position: relative;
          overflow-x: hidden;
        }

        .nav {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 18px 20px;
          position: sticky;
          top: 0;
          background: rgba(8,12,8,0.92);
          backdrop-filter: blur(12px);
          z-index: 50;
          border-bottom: 1px solid rgba(255,255,255,0.06);
        }
        .logo { font-size: 20px; font-weight: 700; letter-spacing: -0.5px; color: #a8f040; }
        .logo.small { font-size: 16px; }
        .btn-nav {
          font-size: 13px; font-weight: 600; color: #a8f040;
          text-decoration: none; border: 1px solid rgba(168,240,64,0.35);
          padding: 7px 16px; border-radius: 100px; transition: background 0.2s;
        }
        .btn-nav:hover { background: rgba(168,240,64,0.1); }

        .hero {
          padding: 48px 24px 40px;
          display: flex; flex-direction: column; align-items: flex-start; gap: 16px;
          position: relative;
        }
        .hero::before {
          content: ''; position: absolute; top: -60px; right: -60px;
          width: 280px; height: 280px;
          background: radial-gradient(circle, rgba(168,240,64,0.12) 0%, transparent 70%);
          pointer-events: none;
        }
        .hero-badge {
          font-size: 11px; font-weight: 600; letter-spacing: 0.08em;
          text-transform: uppercase; color: #a8f040;
          background: rgba(168,240,64,0.1); border: 1px solid rgba(168,240,64,0.25);
          padding: 5px 12px; border-radius: 100px;
        }
        .hero-title {
          font-size: 40px; font-weight: 800; line-height: 1.1;
          letter-spacing: -1.5px; color: #f0f0ee;
        }
        .accent { color: #a8f040; }
        .hero-sub { font-size: 16px; line-height: 1.6; color: #8a9a80; max-width: 320px; }
        .btn-primary {
          display: inline-flex; align-items: center; justify-content: center;
          background: #a8f040; color: #080c08;
          font-size: 15px; font-weight: 700;
          padding: 14px 28px; border-radius: 100px; border: none;
          cursor: pointer; text-decoration: none;
          transition: transform 0.15s, box-shadow 0.15s;
          box-shadow: 0 0 24px rgba(168,240,64,0.3); width: 100%;
        }
        .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 0 36px rgba(168,240,64,0.45); }
        .hero-note { font-size: 12px; color: #5a6a52; text-align: center; width: 100%; }

        .mockup-wrap {
          position: relative; display: flex; justify-content: center;
          padding: 16px 0 40px;
        }
        .phone-glow {
          position: absolute; bottom: 0; left: 50%; transform: translateX(-50%);
          width: 200px; height: 100px;
          background: radial-gradient(ellipse, rgba(168,240,64,0.2) 0%, transparent 70%);
          pointer-events: none;
        }
        .phone {
          width: 260px; background: #111711; border-radius: 32px;
          border: 1px solid rgba(255,255,255,0.1);
          box-shadow: 0 24px 80px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.08);
          padding: 20px 16px; position: relative;
        }
        .phone::before {
          content: ''; display: block; width: 60px; height: 5px;
          background: rgba(255,255,255,0.12); border-radius: 3px; margin: 0 auto 16px;
        }
        .phone-screen {
          background: #0d110d; border-radius: 20px; padding: 16px;
          border: 1px solid rgba(255,255,255,0.06);
        }
        .result-card { display: flex; flex-direction: column; gap: 10px; }
        .result-label { font-size: 9px; font-weight: 700; letter-spacing: 0.1em; color: #4a5a42; }
        .result-issue { font-size: 20px; font-weight: 800; color: #a8f040; letter-spacing: -0.5px; }
        .result-cue { font-size: 12px; color: #c8d8c0; font-style: italic; line-height: 1.4; }
        .result-drill { font-size: 12px; color: #c8d8c0; line-height: 1.4; }
        .result-divider { height: 1px; background: rgba(255,255,255,0.06); }
        .score-row { display: flex; gap: 6px; margin-top: 4px; }
        .score-pill, .severity-pill {
          font-size: 10px; font-weight: 600; padding: 4px 10px; border-radius: 100px;
        }
        .score-pill { background: rgba(168,240,64,0.15); color: #a8f040; border: 1px solid rgba(168,240,64,0.25); }
        .severity-pill { background: rgba(255,200,64,0.12); color: #ffc840; border: 1px solid rgba(255,200,64,0.25); }

        .steps-section { padding: 48px 24px; border-top: 1px solid rgba(255,255,255,0.06); }
        .section-title {
          font-size: 28px; font-weight: 800; line-height: 1.15;
          letter-spacing: -0.8px; margin-bottom: 32px; color: #f0f0ee;
        }
        .steps { list-style: none; display: flex; flex-direction: column; gap: 28px; }
        .step { display: flex; gap: 18px; align-items: flex-start; }
        .step-num {
          font-size: 12px; font-weight: 800; color: #a8f040;
          background: rgba(168,240,64,0.1); border: 1px solid rgba(168,240,64,0.2);
          border-radius: 8px; padding: 6px 10px; flex-shrink: 0; letter-spacing: 0.05em;
        }
        .step strong { display: block; font-size: 16px; font-weight: 700; color: #f0f0ee; margin-bottom: 4px; }
        .step p { font-size: 14px; color: #6a7a62; line-height: 1.5; }

        .ooo-section { padding: 0 24px 48px; }
        .ooo-grid { display: flex; flex-direction: column; gap: 12px; }
        .ooo-card {
          background: #0f140f; border: 1px solid rgba(255,255,255,0.07);
          border-radius: 16px; padding: 20px; display: flex; align-items: center; gap: 16px;
        }
        .ooo-icon { font-size: 24px; flex-shrink: 0; }
        .ooo-label { font-size: 15px; font-weight: 700; color: #f0f0ee; margin-bottom: 2px; }
        .ooo-desc { font-size: 13px; color: #5a6a52; line-height: 1.4; }

        .waitlist-section {
          margin: 0 16px 48px;
          background: linear-gradient(135deg, #111a0a 0%, #0d150a 100%);
          border: 1px solid rgba(168,240,64,0.15); border-radius: 24px;
          padding: 36px 24px; text-align: center;
          display: flex; flex-direction: column; gap: 12px; align-items: center;
        }
        .waitlist-title { font-size: 26px; font-weight: 800; letter-spacing: -0.7px; color: #f0f0ee; }
        .waitlist-sub { font-size: 14px; color: #6a7a62; line-height: 1.5; }
        .waitlist-privacy { font-size: 11px; color: #3a4a32; }

        .footer {
          padding: 32px 24px 48px; border-top: 1px solid rgba(255,255,255,0.06);
          display: flex; flex-direction: column; gap: 8px;
          align-items: center; text-align: center;
        }
        .footer p { font-size: 13px; color: #4a5a42; }
        .footer a { color: #a8f040; text-decoration: none; }
        .footer-copy { font-size: 11px; color: #2a3a22; }
      `}</style>
    </main>
  );
}
