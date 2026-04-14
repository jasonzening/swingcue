'use client';

import { useRef, useEffect, useState, useCallback } from 'react';
import type {
  OverlayTimeline, OverlayEl, PhaseMarkers,
  LineEl, CircleEl, ArrowEl, LabelEl
} from '@/lib/timeline/generateOverlayTimeline';

interface Props {
  videoUrl: string;
  timeline: OverlayTimeline;
  issueLabel: string;
}

const SPEEDS = [0.25, 0.5, 1.0];

const PHASE_BUTTONS: { key: keyof PhaseMarkers; label: string }[] = [
  { key: 'address', label: 'Setup' },
  { key: 'top', label: 'Top' },
  { key: 'transition', label: 'Transition' },
  { key: 'impact', label: 'Impact' },
  { key: 'finish', label: 'Finish' },
];

const RED = '#ff4848';
const GREEN = '#48e848';
const WHITE = 'rgba(255,255,255,0.9)';
const YELLOW = '#ffd040';

function getColor(c: string) {
  if (c === 'red') return RED;
  if (c === 'green') return GREEN;
  if (c === 'yellow') return YELLOW;
  return WHITE;
}

/* Lerp between two keyframe element arrays by fraction */
function lerpNum(a: number, b: number, t: number) { return a + (b - a) * t; }

function findKeyframes(timeline: OverlayTimeline, normalizedTime: number) {
  const kfs = timeline.keyframes;
  if (kfs.length === 0) return null;
  if (normalizedTime <= kfs[0].t) return { kf: kfs[0], next: kfs[0], frac: 0 };
  if (normalizedTime >= kfs[kfs.length - 1].t) {
    const last = kfs[kfs.length - 1];
    return { kf: last, next: last, frac: 0 };
  }
  for (let i = 0; i < kfs.length - 1; i++) {
    if (normalizedTime >= kfs[i].t && normalizedTime <= kfs[i + 1].t) {
      const span = kfs[i + 1].t - kfs[i].t;
      const frac = span === 0 ? 0 : (normalizedTime - kfs[i].t) / span;
      return { kf: kfs[i], next: kfs[i + 1], frac };
    }
  }
  const last = kfs[kfs.length - 1];
  return { kf: last, next: last, frac: 0 };
}

function getCurrentPhase(phases: PhaseMarkers, t: number): string {
  if (t >= phases.finish) return 'finish';
  if (t >= phases.impact) return 'impact';
  if (t >= phases.transition) return 'transition';
  if (t >= phases.top) return 'top';
  if (t >= phases.takeaway) return 'takeaway';
  return 'address';
}

export function SwingPlayer({ videoUrl, timeline, issueLabel }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1.0);
  const [progress, setProgress] = useState(0);      // 0-1
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [currentPhase, setCurrentPhase] = useState('address');
  const [dragging, setDragging] = useState(false);
  const progressBarRef = useRef<HTMLDivElement>(null);

  /* ── Canvas rendering ── */
  const renderFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;

    // Clear
    ctx.clearRect(0, 0, W, H);

    // Get current normalized time (0-1)
    const dur = video.duration || 1;
    const normT = video.currentTime / dur;

    // Find surrounding keyframes
    const result = findKeyframes(timeline, normT);
    if (!result) return;

    const { kf, next, frac } = result;

    // Render elements from current keyframe (interpolated with next)
    const drawElement = (el: OverlayEl) => {
      const col = getColor(el.color);
      const opacity = ('opacity' in el && el.opacity !== undefined) ? el.opacity : 1;
      ctx.globalAlpha = opacity;

      if (el.type === 'line') {
        const le = el as LineEl;
        ctx.beginPath();
        ctx.strokeStyle = col;
        ctx.lineWidth = (le.strokeWidth ?? 3) * (W / 320);
        if (le.dashed) {
          ctx.setLineDash([6 * W / 320, 4 * W / 320]);
        } else {
          ctx.setLineDash([]);
        }
        ctx.moveTo(le.x1 * W, le.y1 * H);
        ctx.lineTo(le.x2 * W, le.y2 * H);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      else if (el.type === 'circle') {
        const ce = el as CircleEl;
        ctx.beginPath();
        ctx.fillStyle = col;
        ctx.arc(ce.cx * W, ce.cy * H, ce.r * W, 0, Math.PI * 2);
        ctx.fill();
      }

      else if (el.type === 'arrow') {
        const ae = el as ArrowEl;
        const fromX = ae.fromX * W;
        const fromY = ae.fromY * H;
        const toX = ae.toX * W;
        const toY = ae.toY * H;
        const headLen = 10 * W / 320;
        const angle = Math.atan2(toY - fromY, toX - fromX);

        ctx.beginPath();
        ctx.strokeStyle = col;
        ctx.fillStyle = col;
        ctx.lineWidth = 2.5 * W / 320;
        ctx.setLineDash([]);
        ctx.moveTo(fromX, fromY);
        ctx.lineTo(toX, toY);
        ctx.stroke();

        // Arrowhead
        ctx.beginPath();
        ctx.moveTo(toX, toY);
        ctx.lineTo(toX - headLen * Math.cos(angle - Math.PI / 6), toY - headLen * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(toX - headLen * Math.cos(angle + Math.PI / 6), toY - headLen * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fill();
      }

      else if (el.type === 'label') {
        const le = el as LabelEl;
        const fontSize = ((le.size ?? 12) * W) / 320;
        ctx.font = `700 ${fontSize}px "DM Sans", system-ui, sans-serif`;
        ctx.fillStyle = col;
        ctx.textAlign = 'center';

        // Shadow for readability
        ctx.shadowColor = 'rgba(0,0,0,0.8)';
        ctx.shadowBlur = 4;
        ctx.fillText(le.text, le.x * W, le.y * H);
        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;
      }

      ctx.globalAlpha = 1;
    };

    // Draw all elements from current keyframe
    // Simple approach: draw current frame's elements (interpolation for positions is complex, do it per element type later)
    kf.elements.forEach(drawElement);

    // If we're between keyframes with >30% progress toward next, fade in next frame's unique elements
    if (frac > 0.5 && next !== kf) {
      ctx.globalAlpha = (frac - 0.5) * 2 * 0.6; // fade in
      next.elements.forEach(drawElement);
      ctx.globalAlpha = 1;
    }

    // Update React state (less frequently, using mod)
    setProgress(normT);
    setCurrentTime(video.currentTime);
    setCurrentPhase(getCurrentPhase(timeline.phases, normT));
  }, [timeline]);

  /* ── Animation loop ── */
  useEffect(() => {
    let raf: number;
    const loop = () => {
      renderFrame();
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    animFrameRef.current = raf;
    return () => cancelAnimationFrame(raf);
  }, [renderFrame]);

  /* ── Resize canvas to match video ── */
  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const resize = () => {
      const rect = video.getBoundingClientRect();
      canvas.width = rect.width || 320;
      canvas.height = rect.height || 240;
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
    };
    video.addEventListener('loadedmetadata', () => { setDuration(video.duration); resize(); });
    window.addEventListener('resize', resize);
    resize();
    return () => window.removeEventListener('resize', resize);
  }, []);

  /* ── Controls ── */
  const togglePlay = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) { v.play(); setPlaying(true); }
    else { v.pause(); setPlaying(false); }
  };

  const handleSpeedChange = (s: number) => {
    setSpeed(s);
    if (videoRef.current) videoRef.current.playbackRate = s;
  };

  const handlePhaseJump = (phase: keyof PhaseMarkers) => {
    const v = videoRef.current;
    if (!v || !v.duration) return;
    const t = timeline.phases[phase] * v.duration;
    v.currentTime = Math.max(0, Math.min(t, v.duration));
    if (v.paused) { setPlaying(false); }
  };

  const stepFrame = (dir: 1 | -1) => {
    const v = videoRef.current;
    if (!v) return;
    v.pause();
    setPlaying(false);
    v.currentTime = Math.max(0, Math.min(v.duration, v.currentTime + dir * (1 / 30)));
  };

  /* ── Scrub ── */
  const handleScrub = useCallback((clientX: number) => {
    const bar = progressBarRef.current;
    const v = videoRef.current;
    if (!bar || !v || !v.duration) return;
    const rect = bar.getBoundingClientRect();
    const fraction = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    v.currentTime = fraction * v.duration;
  }, []);

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
  };

  return (
    <div className="player">
      {/* ── Video + Canvas Stack ── */}
      <div className="video-wrap">
        <video
          ref={videoRef}
          src={videoUrl}
          playsInline
          className="video"
          onEnded={() => setPlaying(false)}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
        />
        <canvas
          ref={canvasRef}
          className="overlay-canvas"
        />

        {/* Phase badge */}
        <div className="phase-badge">{currentPhase.toUpperCase()}</div>

        {/* Legend */}
        <div className="legend">
          <span className="leg-item red-leg">● Current</span>
          <span className="leg-item green-leg">● Target</span>
        </div>

        {/* Center tap to play/pause */}
        <div className="tap-area" onClick={togglePlay} />
      </div>

      {/* ── Speed + Step Controls ── */}
      <div className="controls-row">
        <button className="step-btn" onClick={() => stepFrame(-1)} title="Back one frame">⏮</button>
        <button className="play-btn" onClick={togglePlay}>
          {playing ? '⏸' : '▶'}
        </button>
        <button className="step-btn" onClick={() => stepFrame(1)} title="Forward one frame">⏭</button>

        <div className="speed-btns">
          {SPEEDS.map(s => (
            <button
              key={s}
              className={`speed-btn ${speed === s ? 'active' : ''}`}
              onClick={() => handleSpeedChange(s)}
            >
              {s}x
            </button>
          ))}
        </div>
      </div>

      {/* ── Scrub Bar ── */}
      <div className="scrub-wrap">
        <span className="time-label">{formatTime(currentTime)}</span>
        <div
          ref={progressBarRef}
          className="scrub-track"
          onMouseDown={e => { setDragging(true); handleScrub(e.clientX); }}
          onMouseMove={e => { if (dragging) handleScrub(e.clientX); }}
          onMouseUp={() => setDragging(false)}
          onMouseLeave={() => setDragging(false)}
          onTouchStart={e => { setDragging(true); handleScrub(e.touches[0].clientX); }}
          onTouchMove={e => { if (dragging) handleScrub(e.touches[0].clientX); }}
          onTouchEnd={() => setDragging(false)}
        >
          <div className="scrub-fill" style={{ width: `${progress * 100}%` }} />
          <div className="scrub-thumb" style={{ left: `calc(${progress * 100}% - 7px)` }} />
          {/* Phase markers on scrub bar */}
          {Object.entries(timeline.phases).map(([phase, t]) => (
            <div key={phase} className="phase-tick" style={{ left: `${t * 100}%` }} title={phase} />
          ))}
        </div>
        <span className="time-label">{formatTime(duration)}</span>
      </div>

      {/* ── Phase Jump Buttons ── */}
      <div className="phase-row">
        {PHASE_BUTTONS.map(({ key, label }) => (
          <button
            key={key}
            className={`phase-btn ${currentPhase === key ? 'active' : ''}`}
            onClick={() => handlePhaseJump(key)}
          >
            {label}
          </button>
        ))}
      </div>

      <style>{css}</style>
    </div>
  );
}

const css = `
  .player { display: flex; flex-direction: column; background: #060a06; border-radius: 0; overflow: hidden; width: 100%; }

  .video-wrap { position: relative; width: 100%; background: #000; }
  .video { width: 100%; display: block; max-height: 360px; object-fit: contain; background: #000; }
  .overlay-canvas {
    position: absolute; top: 0; left: 0;
    pointer-events: none;
    width: 100%; height: 100%;
  }
  .tap-area { position: absolute; inset: 0; cursor: pointer; }
  .phase-badge {
    position: absolute; top: 8px; left: 8px;
    background: rgba(0,0,0,0.65); color: #a8f040;
    font-size: 10px; font-weight: 800; letter-spacing: 0.1em;
    padding: 4px 10px; border-radius: 100px;
    font-family: 'DM Sans', system-ui;
    pointer-events: none;
  }
  .legend {
    position: absolute; bottom: 8px; right: 8px;
    background: rgba(0,0,0,0.65);
    display: flex; gap: 10px; padding: 5px 10px; border-radius: 100px;
    pointer-events: none;
  }
  .leg-item { font-size: 10px; font-weight: 700; font-family: 'DM Sans', system-ui; }
  .red-leg { color: #ff4848; }
  .green-leg { color: #48e848; }

  /* Controls */
  .controls-row {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 14px;
    background: #0a100a;
    border-bottom: 1px solid rgba(255,255,255,0.05);
  }
  .play-btn {
    font-size: 20px; background: #a8f040; color: #080c08;
    border: none; border-radius: 50%; width: 38px; height: 38px;
    cursor: pointer; display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; transition: transform 0.1s;
  }
  .play-btn:active { transform: scale(0.92); }
  .step-btn {
    font-size: 16px; background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    color: #8a9a82; border-radius: 8px; width: 34px; height: 34px;
    cursor: pointer; display: flex; align-items: center; justify-content: center;
  }
  .speed-btns { display: flex; gap: 4px; margin-left: auto; }
  .speed-btn {
    font-size: 12px; font-weight: 700; font-family: 'DM Sans', system-ui;
    background: rgba(255,255,255,0.05); color: #4a5a44;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 6px; padding: 5px 9px; cursor: pointer;
    transition: all 0.15s;
  }
  .speed-btn.active { background: rgba(168,240,64,0.12); color: #a8f040; border-color: rgba(168,240,64,0.3); }

  /* Scrub */
  .scrub-wrap {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 14px 8px;
    background: #0a100a;
    border-bottom: 1px solid rgba(255,255,255,0.05);
  }
  .time-label { font-size: 11px; font-weight: 600; color: #3a4a35; font-family: 'DM Sans', system-ui; white-space: nowrap; }
  .scrub-track {
    flex: 1; height: 24px; display: flex; align-items: center;
    position: relative; cursor: pointer; touch-action: none;
  }
  .scrub-fill {
    position: absolute; left: 0; top: 50%;
    height: 4px; background: #a8f040; border-radius: 2px;
    transform: translateY(-50%); pointer-events: none;
    transition: none;
  }
  .scrub-track::before {
    content: ''; position: absolute; left: 0; right: 0; top: 50%;
    height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px;
    transform: translateY(-50%);
  }
  .scrub-thumb {
    position: absolute; top: 50%; width: 14px; height: 14px;
    background: #fff; border-radius: 50%; transform: translateY(-50%);
    box-shadow: 0 1px 4px rgba(0,0,0,0.5); pointer-events: none;
    z-index: 2;
  }
  .phase-tick {
    position: absolute; top: 50%; width: 2px; height: 8px;
    background: rgba(168,240,64,0.5); transform: translate(-50%, -50%);
    border-radius: 1px; pointer-events: none;
  }

  /* Phase jump */
  .phase-row {
    display: flex; gap: 0; background: #0a100a; padding: 8px 14px;
    overflow-x: auto; -webkit-overflow-scrolling: touch;
  }
  .phase-row::-webkit-scrollbar { display: none; }
  .phase-btn {
    flex-shrink: 0; font-size: 12px; font-weight: 700; font-family: 'DM Sans', system-ui;
    color: #3a4a35; background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    padding: 7px 14px; border-radius: 8px; cursor: pointer;
    margin-right: 6px; white-space: nowrap; transition: all 0.15s;
  }
  .phase-btn:active { transform: scale(0.95); }
  .phase-btn.active { color: #a8f040; background: rgba(168,240,64,0.1); border-color: rgba(168,240,64,0.25); }
`;
