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

          {/* ══════════════════════════════════════
              HERO — must work in one iPhone screen
          ══════════════════════════════════════ */}
          <section className="hero">
            <div className="badge">
              <span className="dot" />
              Early access open
            </div>

            {/* 2-line headline — no orphan words -->
            <!-- Each line fits cleanly at 38px in 386px text width */}
            <h1 className="h1">
              Film your swing.<br />
              <span className="h1-green">Get your one fix.</span>
            </h1>

            <p className="sub">
              Point your phone at your swing, upload the clip, and our AI
              finds the biggest flaw — with <em>one cue</em> and <em>one drill</em> to fix it.
            </p>

            {/* Output format pills — user knows exactly what they get */}
            <div className="output-row">
              <span className="out-pill">🎯 One Issue</span>
              <span className="out-sep">·</span>
              <span className="out-pill">💬 One Cue</span>
              <span className="out-sep">·</span>
              <span className="out-pill">🏌️ One Drill</span>
            </div>

            <a href={MAILTO} className="btn-cta">
              Get My Swing Analyzed →
            </a>

            <p className="note">Free · Works in your browser · No download needed</p>
          </section>

          {/* ══════════════════════════════════════
              RESULT PREVIEW
              Full-width card — not a phone frame.
              Shows exactly what the app outputs.
          ══════════════════════════════════════ */}
          <section className="result-section">
            <p className="result-eyebrow">Sample analysis result</p>

            <div className="result-card">
              {/* Header row */}
              <div className="result-header">
                <span className="result-score">72 / 100</span>
                <span className="result-sev">⚠ Medium severity</span>
              </div>

              {/* Issue */}
              <div className="result-block issue-block">
                <p className="block-label">🎯 &nbsp;MAIN ISSUE</p>
                <p className="block-title">Early Extension</p>
                <p className="block-body">
                  Your hips fire early and pull you off the ball before impact —
                  the most common power leak for amateur golfers.
                </p>
              </div>

              <div className="result-divider" />

              {/* Cue */}
              <div className="result-block">
                <p className="block-label">💬 &nbsp;YOUR CUE</p>
                <p className="block-quote">
                  &ldquo;Stay in your posture through impact.&rdquo;
                </p>
                <p className="block-hint">Say this to yourself on the backswing.</p>
              </div>

              <div className="result-divider" />

              {/* Drill */}
              <div className="result-block">
                <p className="block-label">🏌️ &nbsp;TODAY&apos;S DRILL</p>
                <p className="block-drill-name">Wall Drill</p>
                <p className="block-body">
                  Stand with your backside lightly touching a wall. Make practice swings
                  without losing contact. 5 slow reps before every round.
                </p>
              </div>
            </div>
          </section>

          {/* ══════════════════════════════════════
              HOW IT WORKS
          ══════════════════════════════════════ */}
          <section className="steps-section">
            <p className="eyebrow">How it works</p>
            <h2 className="h2">Three steps to your fix.</h2>
            <ol className="ol">
              <li className="li">
                <span className="num">01</span>
                <div>
                  <strong>Film your swing</strong>
                  <p>Face-on or down-the-line. Your phone camera works perfectly — no tripod required.</p>
                </div>
              </li>
              <li className="li">
                <span className="num">02</span>
                <div>
                  <strong>Upload the clip</strong>
                  <p>Our AI reviews your swing frame by frame and identifies the biggest fault.</p>
                </div>
              </li>
              <li className="li">
                <span className="num">03</span>
                <div>
                  <strong>Read your fix</strong>
                  <p>You get one clear issue, one cue to carry onto the course, and one drill for the range.</p>
                </div>
              </li>
            </ol>
          </section>

          {/* ══════════════════════════════════════
              WHO IT'S FOR (replaces redundant cards)
          ══════════════════════════════════════ */}
          <section className="for-section">
            <p className="eyebrow">Who it&apos;s built for</p>
            <h2 className="h2-sm">Golfers who want to get better — without getting overwhelmed.</h2>
            <div className="for-grid">
              <div className="for-item">
                <span className="for-check">✓</span>
                <span>You shoot between 85 and 105 and feel stuck</span>
              </div>
              <div className="for-item">
                <span className="for-check">✓</span>
                <span>You want honest feedback, not endless swing theory</span>
              </div>
              <div className="for-item">
                <span className="for-check">✓</span>
                <span>You don&apos;t have time for weekly $100 lessons</span>
              </div>
              <div className="for-item">
                <span className="for-check">✓</span>
                <span>You want one thing to work on before your next round</span>
              </div>
            </div>
          </section>

          {/* ══════════════════════════════════════
              BOTTOM CTA — make it feel urgent
          ══════════════════════════════════════ */}
          <section className="cta-section" id="cta">
            <div className="cta-inner">
              <p className="eyebrow" style={{color:'#a8f040'}}>Early access · Free to join</p>
              <h2 className="cta-h2">
                Know your fix before<br />your next round.
              </h2>
              <p className="cta-sub">
                Get in early. Email us and we&apos;ll set you up as one of our
                first users when the full upload experience goes live.
              </p>
              <a href={MAILTO} className="btn-cta">
                Get My Swing Analyzed →
              </a>
              <p className="note" style={{marginTop:4}}>
                Replies within 24 hours · support@swingcue.ai
              </p>
            </div>
          </section>

          {/* ── FOOTER ── */}
          <footer className="footer">
            <span className="logo">SwingCue</span>
            <p className="footer-p">
              Questions?{" "}
              <a href="mailto:support@swingcue.ai" className="flink">support@swingcue.ai</a>
            </p>
            <p className="copy">© {new Date().getFullYear()} SwingCue · All rights reserved.</p>
          </footer>

        </main>
      </div>

      <style>{`
        /* ── BASE ── */
        *, *::before, *::after { box-sizing: border-box; }
        * { -webkit-tap-highlight-color: transparent; }
        p, h1, h2 { margin: 0; }

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
          background: rgba(8,12,8,0.95);
          backdrop-filter: blur(20px);
          -webkit-backdrop-filter: blur(20px);
          z-index: 100;
          border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .logo { font-size: 18px; font-weight: 800; letter-spacing: -0.4px; color: #a8f040; }
        .btn-nav {
          font-size: 13px; font-weight: 700;
          color: #080c08; background: #a8f040;
          text-decoration: none;
          padding: 8px 16px; border-radius: 100px;
          white-space: nowrap; min-height: 36px;
          display: flex; align-items: center;
          transition: opacity 0.15s;
        }
        .btn-nav:active { opacity: 0.8; }

        /* ── HERO ── */
        .hero {
          padding: 26px 22px 30px;
          display: flex; flex-direction: column; gap: 16px;
          position: relative;
        }
        .hero::after {
          content: ''; position: absolute;
          top: 0; right: -20px; width: 240px; height: 240px;
          background: radial-gradient(circle, rgba(168,240,64,0.07) 0%, transparent 70%);
          pointer-events: none; z-index: 0;
        }
        .hero > * { position: relative; z-index: 1; }

        /* Live badge */
        .badge {
          display: inline-flex; align-items: center; gap: 7px;
          font-size: 12px; font-weight: 700; color: #a8f040;
          background: rgba(168,240,64,0.09);
          border: 1px solid rgba(168,240,64,0.2);
          padding: 6px 13px 6px 10px; border-radius: 100px;
          width: fit-content;
        }
        .dot {
          width: 7px; height: 7px; background: #a8f040;
          border-radius: 50%; flex-shrink: 0;
          animation: pulse 2s ease-in-out infinite;
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.35; transform: scale(0.8); }
        }

        /* Headline — 2 clean lines, no orphans */
        .h1 {
          font-size: 38px; font-weight: 800;
          line-height: 1.08; letter-spacing: -1.4px;
          color: #f0f0ee;
        }
        .h1-green { color: #a8f040; }

        .sub {
          font-size: 16px; line-height: 1.65; color: #6a7a64;
        }
        .sub em { font-style: normal; color: #c0d0b8; font-weight: 600; }

        /* Output pills row */
        .output-row {
          display: flex; align-items: center; flex-wrap: wrap; gap: 6px;
        }
        .out-pill {
          font-size: 12px; font-weight: 700;
          color: #a8f040;
          background: rgba(168,240,64,0.08);
          border: 1px solid rgba(168,240,64,0.18);
          padding: 5px 11px; border-radius: 100px;
          white-space: nowrap;
        }
        .out-sep { color: rgba(168,240,64,0.25); font-size: 14px; }

        /* Primary CTA */
        .btn-cta {
          display: flex; align-items: center; justify-content: center;
          background: #a8f040; color: #080c08;
          font-family: inherit; font-size: 16px; font-weight: 800;
          height: 56px; border-radius: 100px; border: none;
          cursor: pointer; text-decoration: none; width: 100%;
          box-shadow: 0 0 32px rgba(168,240,64,0.22);
          transition: transform 0.12s, box-shadow 0.12s;
          letter-spacing: -0.2px; -webkit-appearance: none;
        }
        .btn-cta:active {
          transform: scale(0.97);
          box-shadow: 0 0 12px rgba(168,240,64,0.15);
        }

        .note {
          font-size: 12px; color: #2e3e2a; text-align: center; letter-spacing: 0.01em;
        }

        /* ── RESULT PREVIEW ── */
        .result-section {
          padding: 0 16px 44px;
        }
        .result-eyebrow {
          font-size: 11px; font-weight: 700;
          letter-spacing: 0.1em; text-transform: uppercase;
          color: #2e3e28; text-align: center;
          margin-bottom: 14px;
        }
        .result-card {
          background: #0c130b;
          border: 1px solid rgba(168,240,64,0.12);
          border-radius: 20px;
          overflow: hidden;
        }
        .result-header {
          display: flex; justify-content: space-between; align-items: center;
          padding: 14px 18px;
          background: rgba(168,240,64,0.05);
          border-bottom: 1px solid rgba(168,240,64,0.08);
        }
        .result-score {
          font-size: 15px; font-weight: 800; color: #a8f040;
          letter-spacing: -0.3px;
        }
        .result-sev {
          font-size: 12px; font-weight: 600;
          color: #c8a030; background: rgba(255,200,64,0.08);
          border: 1px solid rgba(255,200,64,0.15);
          padding: 3px 10px; border-radius: 100px;
        }
        .result-block {
          padding: 18px 18px 16px;
          display: flex; flex-direction: column; gap: 6px;
        }
        .issue-block { padding-bottom: 18px; }
        .block-label {
          font-size: 10px; font-weight: 700;
          letter-spacing: 0.1em; text-transform: uppercase;
          color: #3a4a36;
        }
        .block-title {
          font-size: 24px; font-weight: 800;
          color: #a8f040; letter-spacing: -0.6px;
          line-height: 1.1;
        }
        .block-body {
          font-size: 13px; line-height: 1.6; color: #5a6a54;
        }
        .block-quote {
          font-size: 17px; font-weight: 700; font-style: italic;
          color: #d8e8d0; line-height: 1.35; letter-spacing: -0.2px;
        }
        .block-hint {
          font-size: 12px; color: #3a4a36; font-style: italic;
        }
        .block-drill-name {
          font-size: 18px; font-weight: 800; color: #e8e8e0;
          letter-spacing: -0.3px;
        }
        .result-divider {
          height: 1px; background: rgba(255,255,255,0.04);
          margin: 0 18px;
        }

        /* ── HOW IT WORKS ── */
        .steps-section {
          padding: 44px 22px;
          border-top: 1px solid rgba(255,255,255,0.05);
        }
        .eyebrow {
          font-size: 11px; font-weight: 700;
          letter-spacing: 0.1em; text-transform: uppercase;
          color: #a8f040; margin-bottom: 8px;
        }
        .h2 {
          font-size: 26px; font-weight: 800; line-height: 1.15;
          letter-spacing: -0.7px; margin-bottom: 28px; color: #f0f0ee;
        }
        .ol { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 24px; }
        .li { display: flex; gap: 16px; align-items: flex-start; }
        .num {
          font-size: 11px; font-weight: 800; color: #a8f040;
          background: rgba(168,240,64,0.08); border: 1px solid rgba(168,240,64,0.15);
          border-radius: 8px; padding: 6px 10px; flex-shrink: 0;
          letter-spacing: 0.04em; min-width: 40px; text-align: center;
        }
        .li strong { display: block; font-size: 15px; font-weight: 700; color: #e8e8e6; margin-bottom: 4px; }
        .li p { font-size: 14px; color: #3e4e3a; line-height: 1.6; margin: 0; }

        /* ── WHO IT'S FOR ── */
        .for-section {
          padding: 0 22px 44px;
        }
        .h2-sm {
          font-size: 22px; font-weight: 800; line-height: 1.2;
          letter-spacing: -0.5px; margin-bottom: 22px;
          color: #f0f0ee;
        }
        .for-grid { display: flex; flex-direction: column; gap: 12px; }
        .for-item {
          display: flex; align-items: flex-start; gap: 12px;
          font-size: 14px; color: #5a6a54; line-height: 1.5;
        }
        .for-check {
          color: #a8f040; font-size: 14px; font-weight: 800;
          flex-shrink: 0; margin-top: 1px;
        }

        /* ── BOTTOM CTA ── */
        .cta-section {
          padding: 0 16px 44px;
        }
        .cta-inner {
          background: linear-gradient(155deg, #101c09 0%, #0a1207 50%, #080c08 100%);
          border: 1px solid rgba(168,240,64,0.15);
          border-radius: 22px;
          padding: 36px 22px;
          text-align: center;
          display: flex; flex-direction: column; align-items: center; gap: 14px;
          position: relative; overflow: hidden;
        }
        .cta-inner::before {
          content: ''; position: absolute;
          top: -60px; left: 50%; transform: translateX(-50%);
          width: 280px; height: 200px;
          background: radial-gradient(ellipse, rgba(168,240,64,0.08) 0%, transparent 70%);
          pointer-events: none;
        }
        .cta-h2 {
          font-size: 28px; font-weight: 800;
          letter-spacing: -0.8px; line-height: 1.1;
          color: #f0f0ee; position: relative;
        }
        .cta-sub {
          font-size: 14px; color: #3e4e3a;
          line-height: 1.65; max-width: 300px; position: relative;
        }

        /* ── FOOTER ── */
        .footer {
          padding: 28px 22px 52px;
          border-top: 1px solid rgba(255,255,255,0.05);
          display: flex; flex-direction: column; gap: 8px;
          align-items: center; text-align: center;
        }
        .footer-p { font-size: 13px; color: #2a3828; margin: 0; }
        .flink { color: #a8f040; text-decoration: none; }
        .copy { font-size: 11px; color: #1a2818; margin: 0; }
      `}</style>
    </>
  );
}
