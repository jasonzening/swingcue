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
import type { OverlayElement, OverlayTimeline, PhaseMarkers, PoseTimeline } from '@/types/analysis';
import { SkeletonOverlay } from '@/components/SkeletonOverlay';
// PR-5: frame-level disc geometry from PR-4 pose_timeline_2d.
import { frameAt } from '@/lib/disc/frameAt';
import { computeShoulderDisc, computeHipDisc } from '@/lib/disc/computeDiscParams';
import { unwrapAngle } from '@/lib/disc/unwrap';
import type { DiscAnchor, DiscParams } from '@/lib/disc/types';

interface Props {
  videoUrl: string;
  timeline: OverlayTimeline;
  phases: PhaseMarkers;        // in seconds
  duration: number;            // video duration in seconds
  dataSource?: string;         // 'mediapipe' | 'stub' — dev indicator
  // PR-4: 17-COCO frame-level timeline. When present, enables the
  // skeleton overlay toggle. NULL when pose_timeline_2d failed
  // validation or the video predates PR-4.
  poseTimeline?: PoseTimeline | null;
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

// PR-5: the OverlayElement union has no 'ellipse' variant in its
// declared `type` field — the PR-3 keypointOverlay shoves ellipse
// elements through via `as unknown as OverlayElement` cast. Read the
// runtime `type` via a wider shape so we can filter them out.
function isEllipseElement(el: OverlayElement): boolean {
  return (el as unknown as { type?: string }).type === 'ellipse';
}

/**
 * PR-5: draw one disc on the canvas in video-native-pixel coordinates,
 * scaled to the canvas's display dims. `scaleX`/`scaleY` are computed
 * from `canvas.width / poseTimeline.video_width` (and y analogously).
 *
 * Coordinate / angle convention: see PR-5_DESIGN.md §3.
 */
function drawDisc(
  ctx: CanvasRenderingContext2D,
  p: DiscParams,
  color: string,
  scaleX: number,
  scaleY: number,
): void {
  ctx.save();
  ctx.translate(p.cx * scaleX, p.cy * scaleY);
  ctx.rotate(p.angleRad);
  ctx.beginPath();
  ctx.ellipse(0, 0, p.rx * scaleX, p.ry * scaleY, 0, 0, Math.PI * 2);
  ctx.strokeStyle = color;
  ctx.lineWidth = 4;
  ctx.globalAlpha = 0.92;
  ctx.stroke();
  ctx.restore();
}

export function SwingPlayer({ videoUrl, timeline, phases, duration: propDur, dataSource, poseTimeline }: Props) {
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
  // PR-4: skeleton overlay toggle. Default OFF per design (extreme
  // simplicity philosophy — debug + demo tool, not core UX). Button
  // disabled when poseTimeline is null.
  const [skeletonOn, setSkeletonOn] = useState(false);

  // PR-5 hotfix: per-disc rolling state so atan2 wrap-around (±π
  // boundary crossings between adjacent frames) doesn't make the
  // disc visually flip 360° between rAF ticks. See unwrap.ts.
  const lastShoulderRef = useRef<{ angleRad: number; ts: number } | null>(null);
  const lastHipRef      = useRef<{ angleRad: number; ts: number } | null>(null);

  // PR-5.1: per-video disc-size anchor. Initialised lazily from the
  // earliest setup-phase frame; held for the entire video. drawDisc
  // overrides each frame's raw `rx` with the anchor value so the disc
  // keeps its setup size during rotation. See PR-5.1_DESIGN.md §3.C.
  const discAnchorRef = useRef<DiscAnchor | null>(null);

  /* ── Canvas sync ── */
  const syncCanvas = useCallback(() => {
    const v = videoRef.current;
    const c = canvasRef.current;
    if (!v || !c) return;
    const rect = v.getBoundingClientRect();
    const w = rect.width || v.offsetWidth || 640;
    const h = rect.height || v.offsetHeight || 360;
    if (w > 0) { c.width = w; c.height = h; }
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

    // PR-5: filter PR-3 timeline ellipses; disc geometry is now computed
    // frame-level from pose_timeline_2d below. Non-ellipse elements
    // (dots, lines, labels, badges, arrows, curves, zones) still render
    // via the existing renderFrame path.
    const elements = getOverlayAtTime(timeline, t).filter(el => !isEllipseElement(el));
    const cw = c.width || 320;
    const ch = c.height || 240;
    renderFrame(ctx, elements, cw, ch, layer);

    // PR-5: frame-level discs from pose_timeline_2d, with PR-5 hotfix
    // atan2 unwrap (so finish-phase rotation past ±π doesn't snap 360°)
    // + PR-5.1 anatomical correction + distance-ratio rotation + size
    // anchor (so disc keeps setup baseline rx during rotation).
    if (poseTimeline) {
      const poseFrame = frameAt(t, poseTimeline);
      if (poseFrame) {
        const scaleX = cw / poseTimeline.video_width;
        const scaleY = ch / poseTimeline.video_height;

        // PR-5.1: lazy-init the per-video disc-size anchor from the
        // earliest setup-phase frame where all 4 shoulder + hip kp
        // have confidence ≥ 0.5. ts < 0.8s ensures we only capture
        // setup geometry, not mid-swing positions.
        if (!discAnchorRef.current && poseFrame.ts < 0.8) {
          const ls = poseFrame.keypoints.left_shoulder;
          const rs = poseFrame.keypoints.right_shoulder;
          const lh = poseFrame.keypoints.left_hip;
          const rh = poseFrame.keypoints.right_hip;
          const allGood =
            ls[0] !== null && ls[1] !== null && ls[2] > 0.5 &&
            rs[0] !== null && rs[1] !== null && rs[2] > 0.5 &&
            lh[0] !== null && lh[1] !== null && lh[2] > 0.5 &&
            rh[0] !== null && rh[1] !== null && rh[2] > 0.5;
          if (allGood) {
            const sDx = (ls[0] as number) - (rs[0] as number);
            const sDy = (ls[1] as number) - (rs[1] as number);
            const hDx = (lh[0] as number) - (rh[0] as number);
            const hDy = (lh[1] as number) - (rh[1] as number);
            discAnchorRef.current = {
              shoulderRx: Math.sqrt(sDx * sDx + sDy * sDy) / 2,
              hipRx:      Math.sqrt(hDx * hDx + hDy * hDy) / 2,
            };
          }
        }

        const baselineShoulderDist = discAnchorRef.current
          ? discAnchorRef.current.shoulderRx * 2 : null;
        const baselineHipDist = discAnchorRef.current
          ? discAnchorRef.current.hipRx * 2 : null;

        const shoulder = computeShoulderDisc(poseFrame, baselineShoulderDist);
        if (shoulder) {
          const unwrapped = unwrapAngle(
            shoulder.angleRad,
            lastShoulderRef.current?.angleRad ?? null,
            lastShoulderRef.current?.ts ?? null,
            t,
          );
          lastShoulderRef.current = { angleRad: unwrapped, ts: t };
          // PR-5.1 §3.C: override raw rx with the per-video anchor so
          // the disc doesn't shrink when shoulders are foreshortened.
          const fixedRx = discAnchorRef.current?.shoulderRx ?? shoulder.rx;
          drawDisc(
            ctx,
            { ...shoulder, rx: fixedRx, ry: fixedRx * 0.2, angleRad: unwrapped },
            '#FFFFFF', scaleX, scaleY,
          );
        }

        // Hip lift in computeHipDisc targets the corrected shoulder midpoint;
        // pass null when the shoulder disc couldn't be computed this frame.
        const shoulderMid = shoulder ? { cx: shoulder.cx, cy: shoulder.cy } : null;
        const hip = computeHipDisc(poseFrame, baselineHipDist, shoulderMid);
        if (hip) {
          const unwrapped = unwrapAngle(
            hip.angleRad,
            lastHipRef.current?.angleRad ?? null,
            lastHipRef.current?.ts ?? null,
            t,
          );
          lastHipRef.current = { angleRad: unwrapped, ts: t };
          const fixedRx = discAnchorRef.current?.hipRx ?? hip.rx;
          drawDisc(
            ctx,
            { ...hip, rx: fixedRx, ry: fixedRx * 0.2, angleRad: unwrapped },
            '#FFFFFF', scaleX, scaleY,
          );
        }
      }
    }
  }, [timeline, phases, layer, dur, poseTimeline]);

  useEffect(() => {
    let id: number;
    const loop = () => { renderTick(); id = requestAnimationFrame(loop); };
    id = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(id);
  }, [renderTick]);

  // PR-5 §5.5: also force a one-shot draw on mount + loadedmetadata + seeked.
  // Defense-in-depth against the case where the rAF loop hasn't ticked yet
  // (or the video is in `ended` state and tab paint has settled), which
  // previously left the canvas blank / overlays stuck at (0, 0).
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const syncOnce = () => { renderTick(); };
    v.addEventListener('loadedmetadata', syncOnce);
    v.addEventListener('seeked', syncOnce);
    syncOnce();
    return () => {
      v.removeEventListener('loadedmetadata', syncOnce);
      v.removeEventListener('seeked', syncOnce);
    };
  }, [renderTick]);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const onMeta = () => { setDur(v.duration); syncCanvas(); };
    const onData = () => { syncCanvas(); };
    v.addEventListener('loadedmetadata', onMeta);
    v.addEventListener('loadeddata', onData);
    v.addEventListener('canplay', syncCanvas);
    window.addEventListener('resize', syncCanvas);
    // Initial sync attempt
    setTimeout(syncCanvas, 100);
    return () => {
      v.removeEventListener('loadedmetadata', onMeta);
      v.removeEventListener('loadeddata', onData);
      v.removeEventListener('canplay', syncCanvas);
      window.removeEventListener('resize', syncCanvas);
    };
  }, [syncCanvas, videoUrl]);

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
          onEnded={() => setPlaying(false)}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onLoadedMetadata={syncCanvas}
          onLoadedData={syncCanvas}
        />
        <canvas ref={canvasRef} className="sp-cvs" />

        {/* PR-4: skeleton overlay (toggle, default off) */}
        {skeletonOn && poseTimeline && (
          <SkeletonOverlay timeline={poseTimeline} videoEl={videoRef.current} />
        )}

        {/* Badges */}
        <div className="sp-badges">
          <span className="sp-phase-badge">{phase.toUpperCase()}</span>
          {dataSource && (() => {
            const isReal =
              dataSource === 'mediapipe' ||
              dataSource === 'sam3d' ||
              dataSource === 'yolo';
            const label =
              dataSource === 'yolo'      ? 'YOLO 11m' :
              dataSource === 'sam3d'     ? 'SAM 3D' :
              dataSource === 'mediapipe' ? 'Real keypoints' :
                                           'Demo overlay';
            return (
              <span className={`sp-src-badge ${isReal ? 'sp-src-real' : 'sp-src-demo'}`}>
                {label}
              </span>
            );
          })()}
        </div>
        <div className="sp-layer-badge">{layerBadgeText()}</div>

        {/* PR-4: skeleton toggle. Disabled when no timeline data — older
            videos predate PR-4 (re-analyze to enable). */}
        <button
          type="button"
          className={`sp-skel-toggle ${skeletonOn ? 'sp-skel-on' : ''} ${!poseTimeline ? 'sp-skel-disabled' : ''}`}
          onClick={() => poseTimeline && setSkeletonOn(o => !o)}
          disabled={!poseTimeline}
          title={poseTimeline ? 'Toggle skeleton overlay' : 'Re-analyze this swing to enable skeleton view'}
          aria-label="Toggle skeleton overlay"
        >
          🦴
        </button>

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
  .sp-src-badge { display:inline-block; margin-left:6px; font-size:8px; font-weight:700; letter-spacing:.06em; padding:3px 8px; border-radius:100px; font-family:'DM Sans',system-ui; }
  .sp-src-real { background:rgba(60,238,60,0.18); color:#3cee3c; }
  .sp-src-demo { background:rgba(255,180,40,0.18); color:#ffb428; }
  .sp-layer-badge { position:absolute; top:9px; right:10px; display:inline-block; background:rgba(0,0,0,0.65); color:rgba(255,255,255,0.60); font-size:9px; font-weight:700; letter-spacing:.08em; padding:3px 9px; border-radius:100px; font-family:'DM Sans',system-ui; pointer-events:none; z-index:3; }

  /* PR-4 skeleton toggle */
  .sp-skel-toggle { position:absolute; top:36px; right:10px; z-index:4; width:30px; height:30px; padding:0; font-size:16px; line-height:1; border-radius:50%; background:rgba(0,0,0,0.65); border:1px solid rgba(255,255,255,0.10); color:#a8f040; cursor:pointer; display:flex; align-items:center; justify-content:center; -webkit-tap-highlight-color:transparent; transition:transform 0.12s, background 0.12s; }
  .sp-skel-toggle:active { transform:scale(0.88); }
  .sp-skel-on { background:rgba(168,240,64,0.18) !important; border-color:rgba(168,240,64,0.45) !important; }
  .sp-skel-disabled { opacity:0.35; cursor:not-allowed; color:rgba(255,255,255,0.40); }

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
