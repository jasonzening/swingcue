const MAILTO = "mailto:support@swingcue.ai?subject=SwingCue%20Early%20Access";

export default function Home() {
  return (
    <>
      <div style={{ background: "#080c08", minHeight: "100vh" }}>
        <main className="sc">

          {/* ── NAV ── */}
          <nav className="nav">
            <span className="logo">SwingCue</span>
            <a href={MAILTO} className="btn-nav">Get Early Access</a>
          </nav>

          {/* ── HERO ── */}
          <section className="hero">
            {/* Badge: signals live product, not vaporware */}
            <div className="badge">
              <span className="dot" />
              Early access open
            </div>

            <h1 className="h1">
              See exactly what&apos;s wrong<br />
              with your swing.
            </h1>

            <p className="sub">
              Upload a short video. Get the <strong>#1 thing</strong> holding
              you back — in plain English, in under a minute.
            </p>

            {/* Direct mailto — eliminates 2-tap friction */}
            <a href={MAILTO} className="btn-cta">
              Get My Swing Analyzed →
            </a>

            <p className="note">Free · Works in your browser · No download needed</p>
          </section>

          {/* ── MOCKUP — with context label ── */}
          <section className="mockup-section">
            <p className="mockup-label">Here&apos;s what your result looks like</p>
            <div className="phone">
              <div className="notch" />
              <div className="screen">
                <p className="tag">MAIN ISSUE DETECTED</p>
                <p className="issue">Early Extension</p>
                <hr className="divider" />
                <p className="tag">YOUR FIX CUE</p>
                <p className="cue">&ldquo;Stay in your posture through impact&rdquo;</p>
                <hr className="divider" />
                <p className="tag">TODAY&apos;S DRILL</p>
                <p className="drill">Wall drill — 5 slow reps before you play</p>
                <div className="pills">
                  <span className="pill gpill">Score 72 / 100</span>
                  <span className="pill ypill">Medium severity</span>
                </div>
              </div>
            </div>
            <div className="glow" />
          </section>

          {/* ── HOW IT WORKS ── */}
          <section className="steps-section">
            <p className="section-eyebrow">How it works</p>
            <h2 className="h2">Three steps.<br />One clear answer.</h2>
            <ol className="ol">
              <li className="li">
                <span className="num">01</span>
                <div>
                  <strong>Film your swing</strong>
                  <p>Face-on or down-the-line. Your phone camera is all you need — no tripod, no special setup.</p>
                </div>
              </li>
              <li className="li">
                <span className="num">02</span>
                <div>
                  <strong>Upload the video</strong>
                  <p>Our AI reviews your swing frame by frame and finds the biggest problem in seconds.</p>
                </div>
              </li>
              <li className="li">
                <span className="num">03</span>
                <div>
                  <strong>Get your fix</strong>
                  <p>One issue. One cue. One drill. Clear enough to use on the range tomorrow.</p>
                </div>
              </li>
            </ol>
          </section>

          {/* ── ONE-ONE-ONE ── */}
          <section className="cards-section">
            <div className="card">
              <span className="icon">🎯</span>
              <div>
                <p className="card-title">One issue at a time</p>
                <p className="card-sub">The single biggest thing holding your swing back — not a list of everything wrong</p>
              </div>
            </div>
            <div className="card">
              <span className="icon">💬</span>
              <div>
                <p className="card-title">One plain-English cue</p>
                <p className="card-sub">A short phrase you can actually remember and feel during your swing</p>
              </div>
            </div>
            <div className="card">
              <span className="icon">🏌️</span>
              <div>
                <p className="card-title">One targeted drill</p>
                <p className="card-sub">The fastest path from understanding the problem to fixing it for real</p>
              </div>
            </div>
          </section>

          {/* ── BOTTOM CTA ── */}
          <section className="cta-section" id="cta">
            <p className="cta-eyebrow">Early access · Limited spots</p>
            <h2 className="cta-h2">Stop guessing.<br />Start improving.</h2>
            <p className="cta-sub">
              Join now and be first in line when we open the full upload experience.
            </p>
            <a href={MAILTO} className="btn-cta">
              Get My Swing Analyzed →
            </a>
            <p className="note">Reach us at support@swingcue.ai</p>
          </section>

          {/* ── FOOTER ── */}
          <footer className="footer">
            <span className="logo">SwingCue</span>
            <p className="footer-p">
              Questions?{" "}
              <a href="mailto:support@swingcue.ai" className="flink">
                support@swingcue.ai
              </a>
            </p>
            <p className="copy">© {new Date().getFullYear()} SwingCue · All rights reserved.</p>
          </footer>

        </main>
      </div>

      <style>{`
        /* ── BASE ── */
        *, *::before, *::after { box-sizing: border-box; }
        * { -webkit-tap-highlight-color: transparent; }

        .sc {
          max-width: 430px;
          margin: 0 auto;
          background: #080c08;
          color: #f0f0ee;
          font-family: 'DM Sans', system-ui, sans-serif;
          overflow-x: hidden;
          min-height: 100vh;
        }

        /* ── NAV ── */
        .nav {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 14px 20px;
          position: sticky;
          top: 0;
          background: rgba(8,12,8,0.94);
          backdrop-filter: blur(20px);
          -webkit-backdrop-filter: blur(20px);
          z-index: 100;
          border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .logo {
          font-size: 18px;
          font-weight: 800;
          letter-spacing: -0.4px;
          color: #a8f040;
        }
        .btn-nav {
          font-size: 13px;
          font-weight: 700;
          color: #080c08;
          background: #a8f040;
          text-decoration: none;
          padding: 8px 16px;
          border-radius: 100px;
          transition: opacity 0.15s;
          white-space: nowrap;
          min-height: 36px;
          display: flex;
          align-items: center;
        }
        .btn-nav:active { opacity: 0.8; }

        /* ── HERO ── */
        .hero {
          padding: 28px 22px 32px;
          display: flex;
          flex-direction: column;
          gap: 18px;
          position: relative;
        }
        .hero::after {
          content: '';
          position: absolute;
          top: 0; right: -20px;
          width: 260px; height: 260px;
          background: radial-gradient(circle, rgba(168,240,64,0.08) 0%, transparent 70%);
          pointer-events: none;
          z-index: 0;
        }
        .hero > * { position: relative; z-index: 1; }

        /* Pulsing "live" badge */
        .badge {
          display: inline-flex;
          align-items: center;
          gap: 7px;
          font-size: 12px;
          font-weight: 700;
          color: #a8f040;
          background: rgba(168,240,64,0.09);
          border: 1px solid rgba(168,240,64,0.2);
          padding: 6px 13px 6px 10px;
          border-radius: 100px;
          width: fit-content;
          letter-spacing: 0.01em;
        }
        .dot {
          width: 7px; height: 7px;
          background: #a8f040;
          border-radius: 50%;
          flex-shrink: 0;
          animation: pulse 2s ease-in-out infinite;
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.4; transform: scale(0.85); }
        }

        .h1 {
          font-size: 36px;
          font-weight: 800;
          line-height: 1.1;
          letter-spacing: -1.2px;
          color: #f0f0ee;
          margin: 0;
        }
        .sub {
          font-size: 16px;
          line-height: 1.65;
          color: #7a8a72;
          margin: 0;
        }
        .sub strong { color: #c8d8c0; font-weight: 700; }
        .note {
          font-size: 12px;
          color: #3a4a35;
          text-align: center;
          margin: 0;
          letter-spacing: 0.01em;
        }

        /* ── PRIMARY CTA ── */
        .btn-cta {
          display: flex;
          align-items: center;
          justify-content: center;
          background: #a8f040;
          color: #080c08;
          font-family: inherit;
          font-size: 16px;
          font-weight: 800;
          padding: 0 28px;
          height: 56px;
          border-radius: 100px;
          border: none;
          cursor: pointer;
          text-decoration: none;
          box-shadow: 0 0 32px rgba(168,240,64,0.25);
          transition: transform 0.12s, box-shadow 0.12s, opacity 0.12s;
          letter-spacing: -0.2px;
          -webkit-appearance: none;
          width: 100%;
        }
        .btn-cta:active {
          transform: scale(0.97);
          box-shadow: 0 0 14px rgba(168,240,64,0.15);
          opacity: 0.92;
        }

        /* ── PHONE MOCKUP ── */
        .mockup-section {
          position: relative;
          display: flex;
          flex-direction: column;
          align-items: center;
          padding: 0 0 44px;
          gap: 14px;
        }
        .mockup-label {
          font-size: 12px;
          font-weight: 600;
          color: #3a4a35;
          text-align: center;
          letter-spacing: 0.03em;
          margin: 0;
          text-transform: uppercase;
        }
        .glow {
          position: absolute;
          bottom: 10px; left: 50%;
          transform: translateX(-50%);
          width: 200px; height: 80px;
          background: radial-gradient(ellipse, rgba(168,240,64,0.15) 0%, transparent 70%);
          pointer-events: none;
        }
        .phone {
          width: 252px;
          background: #0f150e;
          border-radius: 30px;
          border: 1.5px solid rgba(255,255,255,0.08);
          box-shadow:
            0 32px 72px rgba(0,0,0,0.7),
            0 0 0 1px rgba(255,255,255,0.02),
            inset 0 1px 0 rgba(255,255,255,0.05);
          padding: 18px 15px 20px;
        }
        .notch {
          width: 52px; height: 5px;
          background: rgba(255,255,255,0.09);
          border-radius: 3px;
          margin: 0 auto 14px;
        }
        .screen {
          background: #0b0f0b;
          border-radius: 18px;
          padding: 16px 14px;
          border: 1px solid rgba(255,255,255,0.04);
          display: flex;
          flex-direction: column;
          gap: 9px;
        }
        .tag {
          font-size: 8.5px;
          font-weight: 700;
          letter-spacing: 0.12em;
          color: #2a3a28;
          margin: 0;
          text-transform: uppercase;
        }
        .issue {
          font-size: 20px;
          font-weight: 800;
          color: #a8f040;
          letter-spacing: -0.5px;
          margin: 0;
        }
        .cue {
          font-size: 12px;
          color: #a0b498;
          font-style: italic;
          line-height: 1.45;
          margin: 0;
        }
        .drill {
          font-size: 12px;
          color: #a0b498;
          line-height: 1.45;
          margin: 0;
        }
        .divider {
          border: none;
          border-top: 1px solid rgba(255,255,255,0.04);
          margin: 0;
        }
        .pills {
          display: flex;
          gap: 6px;
          flex-wrap: wrap;
          margin-top: 2px;
        }
        .pill {
          font-size: 10px;
          font-weight: 700;
          padding: 4px 10px;
          border-radius: 100px;
        }
        .gpill {
          background: rgba(168,240,64,0.12);
          color: #a8f040;
          border: 1px solid rgba(168,240,64,0.2);
        }
        .ypill {
          background: rgba(255,200,64,0.09);
          color: #e8b830;
          border: 1px solid rgba(255,200,64,0.18);
        }

        /* ── HOW IT WORKS ── */
        .steps-section {
          padding: 44px 22px;
          border-top: 1px solid rgba(255,255,255,0.05);
        }
        .section-eyebrow {
          font-size: 11px;
          font-weight: 700;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          color: #a8f040;
          margin: 0 0 10px;
        }
        .h2 {
          font-size: 28px;
          font-weight: 800;
          line-height: 1.12;
          letter-spacing: -0.8px;
          margin: 0 0 32px;
          color: #f0f0ee;
        }
        .ol {
          list-style: none;
          margin: 0; padding: 0;
          display: flex;
          flex-direction: column;
          gap: 26px;
        }
        .li {
          display: flex;
          gap: 16px;
          align-items: flex-start;
        }
        .num {
          font-size: 11px;
          font-weight: 800;
          color: #a8f040;
          background: rgba(168,240,64,0.08);
          border: 1px solid rgba(168,240,64,0.16);
          border-radius: 8px;
          padding: 6px 10px;
          flex-shrink: 0;
          letter-spacing: 0.04em;
          line-height: 1;
          min-width: 40px;
          text-align: center;
        }
        .li strong {
          display: block;
          font-size: 15px;
          font-weight: 700;
          color: #e8e8e6;
          margin-bottom: 4px;
        }
        .li p {
          font-size: 14px;
          color: #4a5a44;
          line-height: 1.6;
          margin: 0;
        }

        /* ── 1-1-1 CARDS ── */
        .cards-section {
          padding: 0 22px 44px;
          display: flex;
          flex-direction: column;
          gap: 10px;
        }
        .card {
          background: #0c120b;
          border: 1px solid rgba(255,255,255,0.055);
          border-radius: 16px;
          padding: 18px 18px;
          display: flex;
          align-items: flex-start;
          gap: 14px;
        }
        .icon { font-size: 22px; flex-shrink: 0; margin-top: 1px; }
        .card-title {
          font-size: 15px;
          font-weight: 700;
          color: #e8e8e6;
          margin: 0 0 4px;
        }
        .card-sub {
          font-size: 13px;
          color: #3e4e3a;
          line-height: 1.5;
          margin: 0;
        }

        /* ── BOTTOM CTA ── */
        .cta-section {
          margin: 0 16px 44px;
          background: linear-gradient(150deg, #0f1a0a 0%, #0b1208 100%);
          border: 1px solid rgba(168,240,64,0.12);
          border-radius: 22px;
          padding: 36px 22px;
          text-align: center;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 14px;
        }
        .cta-eyebrow {
          font-size: 11px;
          font-weight: 700;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: #a8f040;
          margin: 0;
        }
        .cta-h2 {
          font-size: 26px;
          font-weight: 800;
          letter-spacing: -0.6px;
          color: #f0f0ee;
          line-height: 1.12;
          margin: 0;
        }
        .cta-sub {
          font-size: 14px;
          color: #4a5a44;
          line-height: 1.6;
          margin: 0;
          max-width: 300px;
        }

        /* ── FOOTER ── */
        .footer {
          padding: 28px 22px 52px;
          border-top: 1px solid rgba(255,255,255,0.05);
          display: flex;
          flex-direction: column;
          gap: 8px;
          align-items: center;
          text-align: center;
        }
        .footer-p {
          font-size: 13px;
          color: #2e3e2a;
          margin: 0;
        }
        .flink { color: #a8f040; text-decoration: none; }
        .copy { font-size: 11px; color: #1e2e1a; margin: 0; }
      `}</style>
    </>
  );
}
