export default function Home() {
  return (
    <>
      {/* Desktop background fill — keeps #080c08 edge-to-edge on wide screens */}
      <div style={{ background: '#080c08', minHeight: '100vh' }}>
        <main className="sc">

          {/* ── NAV ── */}
          <nav className="nav">
            <span className="logo">SwingCue</span>
            <a href="#cta" className="btn-nav">Get Early Access</a>
          </nav>

          {/* ── HERO ── */}
          <section className="hero">
            <div className="badge">AI Golf Coach · In Your Pocket</div>
            <h1 className="h1">
              Fix your swing.<br />
              <span className="green">Not your score.</span>
            </h1>
            <p className="sub">
              Upload your swing video. See exactly what&apos;s wrong.
              Get one clear fix — no jargon, no guesswork.
            </p>
            <a href="#cta" className="btn-cta">Analyse My Swing →</a>
            <p className="note">Free to try · No app download needed</p>
          </section>

          {/* ── PHONE MOCKUP ── */}
          <section className="mockup">
            <div className="phone">
              <div className="notch" />
              <div className="screen">
                <p className="tag">MAIN ISSUE</p>
                <p className="issue">Early Extension</p>
                <hr className="div" />
                <p className="tag">FIX CUE</p>
                <p className="cue">&ldquo;Stay in your posture through impact&rdquo;</p>
                <hr className="div" />
                <p className="tag">DRILL</p>
                <p className="drill">Wall drill — 5 reps before every round</p>
                <div className="pills">
                  <span className="pill green-pill">Score 72 / 100</span>
                  <span className="pill yellow-pill">Medium</span>
                </div>
              </div>
            </div>
            <div className="glow" />
          </section>

          {/* ── HOW IT WORKS ── */}
          <section className="steps">
            <h2 className="h2">Three steps.<br />One clear answer.</h2>
            <ol className="ol">
              <li className="li">
                <span className="num">01</span>
                <div>
                  <strong>Film your swing</strong>
                  <p>Face-on or down-the-line — your phone camera is all you need.</p>
                </div>
              </li>
              <li className="li">
                <span className="num">02</span>
                <div>
                  <strong>Upload &amp; analyse</strong>
                  <p>Our AI reviews your video frame by frame in seconds.</p>
                </div>
              </li>
              <li className="li">
                <span className="num">03</span>
                <div>
                  <strong>See your fix</strong>
                  <p>One issue. One cue. One drill. Nothing else to overwhelm you.</p>
                </div>
              </li>
            </ol>
          </section>

          {/* ── ONE-ONE-ONE CARDS ── */}
          <section className="cards">
            <div className="card">
              <span className="icon">🎯</span>
              <div>
                <p className="card-title">One Issue</p>
                <p className="card-sub">The single biggest thing holding you back</p>
              </div>
            </div>
            <div className="card">
              <span className="icon">💬</span>
              <div>
                <p className="card-title">One Cue</p>
                <p className="card-sub">A plain-English phrase to carry to the range</p>
              </div>
            </div>
            <div className="card">
              <span className="icon">🏌️</span>
              <div>
                <p className="card-title">One Drill</p>
                <p className="card-sub">The fastest path from knowing to improving</p>
              </div>
            </div>
          </section>

          {/* ── CTA / WAITLIST ── */}
          <section className="cta-box" id="cta">
            <h2 className="cta-title">Ready to stop guessing?</h2>
            <p className="cta-sub">
              Join early access — be the first to upload your swing.
            </p>
            <a
              href="mailto:support@swingcue.ai?subject=SwingCue%20Early%20Access"
              className="btn-cta"
            >
              Request Early Access →
            </a>
            <p className="note">No spam · support@swingcue.ai</p>
          </section>

          {/* ── FOOTER ── */}
          <footer className="footer">
            <span className="logo">SwingCue</span>
            <p>Questions? <a href="mailto:support@swingcue.ai" className="flink">support@swingcue.ai</a></p>
            <p className="copy">© {new Date().getFullYear()} SwingCue. All rights reserved.</p>
          </footer>

        </main>
      </div>

      <style>{`
        /* ─── CONTAINER ─── */
        .sc {
          max-width: 430px;
          margin: 0 auto;
          background: #080c08;
          color: #f0f0ee;
          font-family: 'DM Sans', system-ui, sans-serif;
          overflow-x: hidden;
          min-height: 100vh;
        }

        /* ─── NAV ─── */
        .nav {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 16px 20px;
          position: sticky;
          top: 0;
          background: rgba(8,12,8,0.93);
          backdrop-filter: blur(16px);
          -webkit-backdrop-filter: blur(16px);
          z-index: 100;
          border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .logo {
          font-size: 19px;
          font-weight: 800;
          letter-spacing: -0.5px;
          color: #a8f040;
        }
        .btn-nav {
          font-size: 13px;
          font-weight: 600;
          color: #a8f040;
          text-decoration: none;
          border: 1px solid rgba(168,240,64,0.4);
          padding: 8px 16px;
          border-radius: 100px;
          transition: background 0.18s;
          white-space: nowrap;
        }
        .btn-nav:active { background: rgba(168,240,64,0.15); }

        /* ─── HERO ─── */
        .hero {
          padding: 44px 22px 36px;
          display: flex;
          flex-direction: column;
          gap: 16px;
          position: relative;
        }
        .hero::after {
          content: '';
          position: absolute;
          top: 0; right: -40px;
          width: 240px; height: 240px;
          background: radial-gradient(circle, rgba(168,240,64,0.1) 0%, transparent 70%);
          pointer-events: none;
        }
        .badge {
          display: inline-block;
          font-size: 11px;
          font-weight: 600;
          letter-spacing: 0.07em;
          text-transform: uppercase;
          color: #a8f040;
          background: rgba(168,240,64,0.09);
          border: 1px solid rgba(168,240,64,0.22);
          padding: 6px 13px;
          border-radius: 100px;
          width: fit-content;
        }
        .h1 {
          font-size: 38px;
          font-weight: 800;
          line-height: 1.08;
          letter-spacing: -1.5px;
          color: #f0f0ee;
          margin: 0;
        }
        .green { color: #a8f040; }
        .sub {
          font-size: 16px;
          line-height: 1.65;
          color: #7a8f72;
          margin: 0;
        }
        .note {
          font-size: 12px;
          color: #4a5a44;
          text-align: center;
          margin: 0;
        }

        /* ─── PRIMARY CTA BUTTON ─── */
        .btn-cta {
          display: flex;
          align-items: center;
          justify-content: center;
          background: #a8f040;
          color: #080c08;
          font-family: inherit;
          font-size: 16px;
          font-weight: 700;
          padding: 16px 28px;
          border-radius: 100px;
          border: none;
          cursor: pointer;
          text-decoration: none;
          min-height: 52px;
          box-shadow: 0 0 28px rgba(168,240,64,0.28);
          transition: transform 0.12s, box-shadow 0.12s;
          -webkit-appearance: none;
        }
        .btn-cta:active {
          transform: scale(0.97);
          box-shadow: 0 0 16px rgba(168,240,64,0.2);
        }

        /* ─── PHONE MOCKUP ─── */
        .mockup {
          position: relative;
          display: flex;
          justify-content: center;
          padding: 8px 0 44px;
        }
        .glow {
          position: absolute;
          bottom: 12px; left: 50%;
          transform: translateX(-50%);
          width: 180px; height: 80px;
          background: radial-gradient(ellipse, rgba(168,240,64,0.18) 0%, transparent 70%);
          pointer-events: none;
        }
        .phone {
          width: 248px;
          background: #10160f;
          border-radius: 30px;
          border: 1.5px solid rgba(255,255,255,0.09);
          box-shadow:
            0 28px 60px rgba(0,0,0,0.7),
            0 0 0 1px rgba(255,255,255,0.03),
            inset 0 1px 0 rgba(255,255,255,0.06);
          padding: 18px 15px 20px;
        }
        .notch {
          width: 56px; height: 5px;
          background: rgba(255,255,255,0.1);
          border-radius: 3px;
          margin: 0 auto 15px;
        }
        .screen {
          background: #0b100b;
          border-radius: 18px;
          padding: 16px 14px;
          border: 1px solid rgba(255,255,255,0.05);
          display: flex;
          flex-direction: column;
          gap: 9px;
        }
        .tag {
          font-size: 9px;
          font-weight: 700;
          letter-spacing: 0.12em;
          color: #3a4a35;
          margin: 0;
        }
        .issue {
          font-size: 21px;
          font-weight: 800;
          color: #a8f040;
          letter-spacing: -0.5px;
          margin: 0;
        }
        .cue {
          font-size: 12px;
          color: #b8ccb0;
          font-style: italic;
          line-height: 1.45;
          margin: 0;
        }
        .drill {
          font-size: 12px;
          color: #b8ccb0;
          line-height: 1.45;
          margin: 0;
        }
        .div {
          border: none;
          border-top: 1px solid rgba(255,255,255,0.05);
          margin: 0;
        }
        .pills {
          display: flex;
          gap: 6px;
          margin-top: 3px;
        }
        .pill {
          font-size: 10px;
          font-weight: 600;
          padding: 4px 10px;
          border-radius: 100px;
        }
        .green-pill {
          background: rgba(168,240,64,0.13);
          color: #a8f040;
          border: 1px solid rgba(168,240,64,0.22);
        }
        .yellow-pill {
          background: rgba(255,200,64,0.1);
          color: #ffc840;
          border: 1px solid rgba(255,200,64,0.22);
        }

        /* ─── HOW IT WORKS ─── */
        .steps {
          padding: 44px 22px;
          border-top: 1px solid rgba(255,255,255,0.05);
        }
        .h2 {
          font-size: 27px;
          font-weight: 800;
          line-height: 1.15;
          letter-spacing: -0.7px;
          margin: 0 0 30px;
          color: #f0f0ee;
        }
        .ol {
          list-style: none;
          margin: 0; padding: 0;
          display: flex;
          flex-direction: column;
          gap: 24px;
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
          background: rgba(168,240,64,0.09);
          border: 1px solid rgba(168,240,64,0.18);
          border-radius: 8px;
          padding: 6px 10px;
          flex-shrink: 0;
          letter-spacing: 0.04em;
          line-height: 1;
        }
        .li strong {
          display: block;
          font-size: 15px;
          font-weight: 700;
          color: #f0f0ee;
          margin-bottom: 3px;
        }
        .li p {
          font-size: 14px;
          color: #5a6a54;
          line-height: 1.55;
          margin: 0;
        }

        /* ─── 1-1-1 CARDS ─── */
        .cards {
          padding: 0 22px 44px;
          display: flex;
          flex-direction: column;
          gap: 10px;
        }
        .card {
          background: #0d120d;
          border: 1px solid rgba(255,255,255,0.06);
          border-radius: 16px;
          padding: 18px;
          display: flex;
          align-items: center;
          gap: 14px;
        }
        .icon { font-size: 22px; flex-shrink: 0; }
        .card-title {
          font-size: 15px;
          font-weight: 700;
          color: #f0f0ee;
          margin: 0 0 3px;
        }
        .card-sub {
          font-size: 13px;
          color: #4a5a44;
          line-height: 1.45;
          margin: 0;
        }

        /* ─── CTA BOX ─── */
        .cta-box {
          margin: 0 16px 44px;
          background: linear-gradient(145deg, #101a09 0%, #0c1309 100%);
          border: 1px solid rgba(168,240,64,0.13);
          border-radius: 22px;
          padding: 34px 22px;
          text-align: center;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 14px;
        }
        .cta-title {
          font-size: 24px;
          font-weight: 800;
          letter-spacing: -0.6px;
          color: #f0f0ee;
          margin: 0;
        }
        .cta-sub {
          font-size: 14px;
          color: #5a6a54;
          line-height: 1.6;
          margin: 0;
        }

        /* ─── FOOTER ─── */
        .footer {
          padding: 28px 22px 52px;
          border-top: 1px solid rgba(255,255,255,0.05);
          display: flex;
          flex-direction: column;
          gap: 8px;
          align-items: center;
          text-align: center;
        }
        .footer p {
          font-size: 13px;
          color: #3a4a35;
          margin: 0;
        }
        .flink { color: #a8f040; text-decoration: none; }
        .copy { font-size: 11px; color: #232e20; }
      `}</style>
    </>
  );
}
