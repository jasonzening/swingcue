'use client';

/**
 * SwingPlayer.tsx — Interactive Swing Player
 *
 * 视频是页面主角。播放器尽量满屏，控制条紧凑在下方。
 * Canvas overlay 与视频时间精确同步。
 * 三层切换：Body / Arms / Club / All
 */

import { useRef, useEffect, useState, useCallback } from 'react';
import { renderFrame } from '@/lib/overlay/OverlayRenderer';
import { getOverlayAtTime, getCurrentPhase, formatTime } from '@/lib/overlay/playerSync';
import type { OverlayTimeline, PhaseMarkers } from '@/types/analysis';

interface Props {
  videoUrl: string;
  timeline: OverlayTimeline;
  phases: PhaseMarkers;        // in seconds
  duration: number;            // video duration in seconds
}

type LayerKey = 'body' | 'arms' | 'club' | 'all';

const SPEEDS = [0.25, 0.5, 1.0];

const LAYERS: { key: LayerKey; icon: string; label: string }[] = [
  { key: 'body',  icon: '🧍', label: 'Body' },
  { key: 'arms',  icon: '🤲', label: 'Hands' },
  { key: 'club',  icon: '⛳', label: 'Club' },
  { key: 'all',   icon: '👁',  label: 'All' },
];

const PHASE_BTNS: { key: keyof PhaseMarkers; label: string }[] = [
  { key: 'setupTime',      label: 'Setup' },
  { key: 'topTime',        label: 'Top' },
  { key: 'transitionTime', label: 'Trans.' },
  { key: 'impactTime',     label: 'Impact' },
  { key: 'finishTime',     label: 'Finish' },
];

export function SwingPlayer({ videoUrl, timeline, phases, duration: propDur }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef    = useRef<number>(0);
  const barRef    = useRef<HTMLDivElement>(null);

  const [layer,   setLayer]   = useState<LayerKey>('all');
  const [playing, setPlaying] = useState(false);
  const [speed,   setSpeed]   = useState(1.0);
  const [progress, setProgress] = useState(0);
  const [curTime,  setCurTime]  = useState(0);
  const [dur,      setDur]      = useState(propDur || 1);
  const [phase,    setPhase]    = useState<string>('setup');
  const [dragging, setDragging] = useState(false);

  /* ── Canvas sync ── */
  const syncCanvas = useCallback(() => {
    const v = videoRef.current;
    const c = canvasRef.current;
    if (!v || !c) return;
    const rect = v.getBoundingClientRect();
    if (rect.width > 0) { c.width = rect.width; c.height = rect.height; }
  }, []);

  /* ── Render loop ── */
  const renderTick = useCallback(() => {
    const v = videoRef.current;
    const c = canvasRef.current;
    if (!v || !c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;

    const t = v.currentTime;
    const d = v.duration || dur || 1;

    setProgress(t / d);
    setCurTime(t);
    setPhase(getCurrentPhase(phases, t, d));

    const elements = getOverlayAtTime(timeline, t);
    renderFrame(ctx, elements, c.width || 320, c.height || 240, layer);
  }, [timeline, phases, layer, dur]);

  useEffect(() => {
    let id: number;
    const loop = () => { renderTick(); id = requestAnimationFrame(loop); };
    id = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(id);
  }, [renderTick]);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const onMeta = () => { setDur(v.duration); syncCanvas(); };
    v.addEventListener('loadedmetadata', onMeta);
    window.addEventListener('resize', syncCanvas);
    return () => { v.removeEventListener('loadedmetadata', onMeta); window.removeEventListener('resize', syncCanvas); };
  }, [syncCanvas]);

  /* ── Controls ── */
  const togglePlay = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) { v.play(); setPlaying(true); }
    else { v.pause(); setPlaying(false); }
  };

  const setSpd = (s: number) => { setSpeed(s); if (videoRef.current) videoRef.current.playbackRate = s; };

  const jumpTo = (key: keyof PhaseMarkers) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = phases[key];
  };

  const step = (dir: 1 | -1) => {
    const v = videoRef.current;
    if (!v) return;
    v.pause(); setPlaying(false);
    v.currentTime = Math.max(0, Math.min(v.duration, v.currentTime + dir / 30));
  };

  const scrub = useCallback((clientX: number) => {
    const v = videoRef.current;
    if (!barRef.current || !v?.duration) return;
    const r = barRef.current.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (clientX - r.left) / r.width));
    v.currentTime = frac * v.duration;
  }, []);

  /* ── Layer -> label color ── */
  const layerBadgeText = () => {
    if (layer === 'body')  return 'BODY';
    if (layer === 'arms')  return 'HANDS';
    if (layer === 'club')  return 'CLUB';
    return 'ALL';
  };

  return (
    <div className="sp">
      {/* ══ VIDEO + CANVAS ══ */}
      <div className="sp-vw">
        <video
          ref={videoRef}
          src={videoUrl}
          playsInline
          className="sp-vid"
          crossOrigin="anonymous"
          onEnded={() => setPlaying(false)}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
        />
        <canvas ref={canvasRef} className="sp-cvs" />

        {/* Badges */}
        <div className="sp-badges">
          <span className="sp-phase-badge">{phase.toUpperCase()}</span>
        </div>
        <div className="sp-layer-badge">{layerBadgeText()}</div>

        {/* Legend */}
        <div className="sp-legend">
          <span className="leg-r">● Current</span>
          <span className="leg-g">● Target</span>
          <span className="leg-y">● Path</span>
        </div>

        <div className="sp-tap" onClick={togglePlay} />
      </div>

      {/* ══ LAYER TOGGLE ══ */}
      <div className="sp-layers">
        {LAYERS.map(({ key, icon, label }) => (
          <button
            key={key}
            className={`sp-lb ${layer === key ? 'sp-lb-on' : ''}`}
            onClick={() => setLayer(key)}
          >
            <span className="sp-lb-icon">{icon}</span>
            <span className="sp-lb-text">{label}</span>
          </button>
        ))}
      </div>

      {/* ══ CONTROLS ══ */}
      <div className="sp-ctrl">
        <button className="sp-step" onClick={() => step(-1)}>⏮</button>
        <button className="sp-play" onClick={togglePlay}>{playing ? '⏸' : '▶'}</button>
        <button className="sp-step" onClick={() => step(1)}>⏭</button>
        <div className="sp-spd-row">
          {SPEEDS.map(s => (
            <button key={s} className={`sp-spd ${speed === s ? 'sp-spd-on' : ''}`} onClick={() => setSpd(s)}>
              {s}x
            </button>
          ))}
        </div>
      </div>

      {/* ══ SCRUB ══ */}
      <div className="sp-scrub-wrap">
        <span className="sp-time">{formatTime(curTime)}</span>
        <div
          ref={barRef}
          className="sp-bar"
          onMouseDown={e => { setDragging(true); scrub(e.clientX); }}
          onMouseMove={e => { if (dragging) scrub(e.clientX); }}
          onMouseUp={() => setDragging(false)}
          onMouseLeave={() => setDragging(false)}
          onTouchStart={e => { setDragging(true); scrub(e.touches[0].clientX); }}
          onTouchMove={e => { if (dragging) scrub(e.touches[0].clientX); }}
          onTouchEnd={() => setDragging(false)}
        >
          <div className="sp-fill" style={{ width: `${progress * 100}%` }} />
          <div className="sp-thumb" style={{ left: `calc(${progress * 100}% - 7px)` }} />
          {/* Phase tick marks */}
          {Object.entries(phases).map(([k, t]) => {
            const frac = (t as number) / (dur || 1);
            return <div key={k} className="sp-tick" style={{ left: `${frac * 100}%` }} />;
          })}
        </div>
        <span className="sp-time">{formatTime(dur)}</span>
      </div>

      {/* ══ PHASE JUMP ══ */}
      <div className="sp-phases">
        {PHASE_BTNS.map(({ key, label }) => {
          const phaseKey = key.replace('Time', '') as string;
          const isActive = phase === phaseKey || (key === 'setupTime' && phase === 'setup');
          return (
            <button
              key={key}
              className={`sp-pb ${isActive ? 'sp-pb-on' : ''}`}
              onClick={() => jumpTo(key)}
            >
              {label}
            </button>
          );
        })}
      </div>

      <style>{css}</style>
    </div>
  );
}

const css = `
  .sp { display:flex; flex-direction:column; background:#050805; width:100%; }

  /* ── Video wrap ── */
  .sp-vw { position:relative; width:100%; background:#000; flex-shrink:0; }
  .sp-vid { width:100%; display:block; object-fit:contain; background:#000; }
  .sp-cvs { position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none; }
  .sp-tap { position:absolute; inset:0; cursor:pointer; z-index:2; }

  /* ── Badges ── */
  .sp-badges { position:absolute; top:9px; left:10px; z-index:3; pointer-events:none; }
  .sp-phase-badge { display:inline-block; background:rgba(0,0,0,0.72); color:#a8f040; font-size:9px; font-weight:800; letter-spacing:.12em; padding:3px 9px; border-radius:100px; font-family:'DM Sans',system-ui; }
  .sp-layer-badge { position:absolute; top:9px; right:10px; display:inline-block; background:rgba(0,0,0,0.65); color:rgba(255,255,255,0.60); font-size:9px; font-weight:700; letter-spacing:.08em; padding:3px 9px; border-radius:100px; font-family:'DM Sans',system-ui; pointer-events:none; z-index:3; }

  /* ── Legend ── */
  .sp-legend { position:absolute; bottom:8px; right:10px; background:rgba(0,0,0,0.70); display:flex; gap:8px; padding:5px 10px; border-radius:100px; z-index:3; pointer-events:none; }
  .sp-legend span { font-size:9px; font-weight:700; font-family:'DM Sans',system-ui; }
  .leg-r { color:#ff3c3c; }
  .leg-g { color:#3cee3c; }
  .leg-y { color:#ffd040; }

  /* ── Layer toggle ── */
  .sp-layers { display:flex; gap:0; background:#0a100a; border-bottom:1px solid rgba(255,255,255,0.05); padding:8px 12px; overflow-x:auto; -webkit-overflow-scrolling:touch; }
  .sp-layers::-webkit-scrollbar { display:none; }
  .sp-lb { display:flex; align-items:center; gap:5px; flex-shrink:0; font-family:'DM Sans',system-ui; font-size:12px; font-weight:700; color:#2a3a25; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.07); padding:7px 14px; border-radius:8px; cursor:pointer; margin-right:7px; -webkit-tap-highlight-color:transparent; transition:all 0.12s; }
  .sp-lb:active { transform:scale(0.93); }
  .sp-lb-on { color:#a8f040 !important; background:rgba(168,240,64,0.11) !important; border-color:rgba(168,240,64,0.28) !important; }
  .sp-lb-icon { font-size:14px; }
  .sp-lb-text { font-size:12px; }

  /* ── Controls ── */
  .sp-ctrl { display:flex; align-items:center; gap:8px; padding:9px 14px; background:#0a100a; border-bottom:1px solid rgba(255,255,255,0.05); }
  .sp-play { font-size:17px; background:#a8f040; color:#080c08; border:none; border-radius:50%; width:38px; height:38px; cursor:pointer; display:flex; align-items:center; justify-content:center; flex-shrink:0; -webkit-tap-highlight-color:transparent; transition:transform 0.1s; }
  .sp-play:active { transform:scale(0.88); }
  .sp-step { font-size:13px; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.09); color:#6a7a62; border-radius:7px; width:34px; height:34px; cursor:pointer; display:flex; align-items:center; justify-content:center; flex-shrink:0; -webkit-tap-highlight-color:transparent; }
  .sp-spd-row { display:flex; gap:5px; margin-left:auto; }
  .sp-spd { font-size:12px; font-weight:700; font-family:'DM Sans',system-ui; background:rgba(255,255,255,0.04); color:#3a4a35; border:1px solid rgba(255,255,255,0.07); border-radius:6px; padding:5px 10px; cursor:pointer; -webkit-tap-highlight-color:transparent; transition:all 0.12s; }
  .sp-spd-on { background:rgba(168,240,64,0.11) !important; color:#a8f040 !important; border-color:rgba(168,240,64,0.28) !important; }

  /* ── Scrub ── */
  .sp-scrub-wrap { display:flex; align-items:center; gap:8px; padding:5px 14px 7px; background:#0a100a; border-bottom:1px solid rgba(255,255,255,0.05); }
  .sp-time { font-size:11px; font-weight:600; color:#2a3a25; font-family:'DM Sans',system-ui; white-space:nowrap; width:28px; }
  .sp-bar { flex:1; height:24px; display:flex; align-items:center; position:relative; cursor:pointer; touch-action:none; }
  .sp-bar::before { content:''; position:absolute; left:0; right:0; top:50%; height:4px; background:rgba(255,255,255,0.09); border-radius:2px; transform:translateY(-50%); }
  .sp-fill { position:absolute; left:0; top:50%; height:4px; background:#a8f040; border-radius:2px; transform:translateY(-50%); pointer-events:none; }
  .sp-thumb { position:absolute; top:50%; width:14px; height:14px; background:#fff; border-radius:50%; transform:translateY(-50%); box-shadow:0 1px 5px rgba(0,0,0,0.6); pointer-events:none; z-index:2; }
  .sp-tick { position:absolute; top:50%; width:2px; height:8px; background:rgba(168,240,64,0.45); transform:translate(-50%,-50%); border-radius:1px; pointer-events:none; }

  /* ── Phase buttons ── */
  .sp-phases { display:flex; background:#0a100a; padding:7px 14px; overflow-x:auto; -webkit-overflow-scrolling:touch; }
  .sp-phases::-webkit-scrollbar { display:none; }
  .sp-pb { flex-shrink:0; font-size:12px; font-weight:700; font-family:'DM Sans',system-ui; color:#2a3a25; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.06); padding:6px 14px; border-radius:8px; cursor:pointer; margin-right:6px; white-space:nowrap; -webkit-tap-highlight-color:transparent; transition:all 0.12s; }
  .sp-pb:active { transform:scale(0.93); }
  .sp-pb-on { color:#a8f040 !important; background:rgba(168,240,64,0.10) !important; border-color:rgba(168,240,64,0.24) !important; }
`;
