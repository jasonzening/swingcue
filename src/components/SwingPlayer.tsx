'use client';

/**
 * SwingPlayer.tsx
 *
 * Interactive golf swing video player with synchronized overlay.
 *
 * Architecture:
 *   - <video>  : original video playback layer
 *   - <canvas> : overlay layer, drawn via requestAnimationFrame
 *
 * Overlay supports 3 layers (toggled by user):
 *   🧍 Body   – joint dots + spine/shoulder/hip lines
 *   🤲 Hands  – arm chains + grip V-shape + hand path trajectory
 *   ⛳ Club   – estimated club shaft + club head dot
 *   👁 All    – everything above
 *
 * When real MediaPipe data is stored in overlay_timeline_json.poseFrames,
 * the canvas draws real joint positions. Otherwise falls back to the
 * issue-specific reference guide from the stub timeline.
 *
 * Visual language (matching reference images from user):
 *   Green filled dot  = target/correct joint position
 *   Red filled dot    = current/tracked joint position
 *   Yellow           = path trajectory, correction arrows
 *   White            = structural reference lines
 */

import { useRef, useEffect, useState, useCallback } from 'react';
import { KP, getPoseAt, type Keypoint, type PoseFrame } from '@/lib/pose/PoseDetector';
import type { OverlayTimeline, PhaseMarkers } from '@/lib/timeline/generateOverlayTimeline';

/* ══════════════════════════════════════════════════════
   TYPES
══════════════════════════════════════════════════════ */
interface Props {
  videoUrl: string;
  timeline: OverlayTimeline;
  issueType?: string;
}

type LayerKey = 'body' | 'hands' | 'club' | 'all';

const SPEEDS = [0.25, 0.5, 1.0];

const LAYERS: { key: LayerKey; icon: string; label: string }[] = [
  { key: 'body',  icon: '🧍', label: 'Body' },
  { key: 'hands', icon: '🤲', label: 'Hands' },
  { key: 'club',  icon: '⛳', label: 'Club' },
  { key: 'all',   icon: '👁', label: 'All' },
];

const PHASES: { key: keyof PhaseMarkers; label: string }[] = [
  { key: 'address',    label: 'Setup' },
  { key: 'top',        label: 'Top' },
  { key: 'transition', label: 'Trans.' },
  { key: 'impact',     label: 'Impact' },
  { key: 'finish',     label: 'Finish' },
];

type Pt = { x: number; y: number };

/* ══════════════════════════════════════════════════════
   CANVAS DRAWING PRIMITIVES
══════════════════════════════════════════════════════ */
const C = {
  RED:    '#ff3c3c',
  GREEN:  '#3cee3c',
  YELLOW: '#ffd040',
  WHITE:  'rgba(255,255,255,0.88)',
  BLUE:   '#40c8ff',
};

function filledDot(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, color: string, opacity = 1) {
  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.shadowColor = 'rgba(0,0,0,0.7)';
  ctx.shadowBlur = 6;
  ctx.fill();
  ctx.shadowBlur = 0;
  // Black ring
  ctx.strokeStyle = 'rgba(0,0,0,0.55)';
  ctx.lineWidth = Math.max(1.5, r * 0.2);
  ctx.stroke();
  ctx.restore();
}

function structureLine(
  ctx: CanvasRenderingContext2D,
  x1: number, y1: number, x2: number, y2: number,
  color: string, width: number, opacity = 1, dashed = false
) {
  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = 'round';
  ctx.shadowColor = 'rgba(0,0,0,0.6)';
  ctx.shadowBlur = 4;
  if (dashed) ctx.setLineDash([width * 2.5, width * 1.5]);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
}

function arrowLine(
  ctx: CanvasRenderingContext2D,
  fromX: number, fromY: number, toX: number, toY: number,
  color: string, width: number, opacity = 1
) {
  ctx.save();
  ctx.globalAlpha = opacity;
  const angle = Math.atan2(toY - fromY, toX - fromX);
  const dist = Math.hypot(toX - fromX, toY - fromY);
  const headLen = Math.min(20, dist * 0.45);

  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = 'round';
  ctx.shadowColor = 'rgba(0,0,0,0.7)';
  ctx.shadowBlur = 5;

  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.stroke();

  // Arrowhead
  ctx.beginPath();
  ctx.moveTo(toX, toY);
  ctx.lineTo(toX - headLen * Math.cos(angle - 0.42), toY - headLen * Math.sin(angle - 0.42));
  ctx.lineTo(toX - headLen * Math.cos(angle + 0.42), toY - headLen * Math.sin(angle + 0.42));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function smoothCurve(ctx: CanvasRenderingContext2D, pts: Pt[], color: string, width: number, opacity = 1) {
  if (pts.length < 2) return;
  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.shadowColor = 'rgba(0,0,0,0.55)';
  ctx.shadowBlur = 5;
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length - 1; i++) {
    const cpX = (pts[i].x + pts[i + 1].x) / 2;
    const cpY = (pts[i].y + pts[i + 1].y) / 2;
    ctx.quadraticCurveTo(pts[i].x, pts[i].y, cpX, cpY);
  }
  ctx.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
  ctx.stroke();
  // Terminal dot
  const last = pts[pts.length - 1];
  ctx.beginPath();
  ctx.arc(last.x, last.y, width * 2, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.restore();
}

function overlayLabel(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color: string, size: number, opacity = 1) {
  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.font = `800 ${size}px "DM Sans", system-ui, sans-serif`;
  ctx.textAlign = 'center';
  ctx.shadowColor = 'rgba(0,0,0,0.95)';
  ctx.shadowBlur = 8;
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
  ctx.restore();
}

/* ══════════════════════════════════════════════════════
   POSE-BASED OVERLAY RENDERER
   Draws real joint positions from MediaPipe/MoveNet data.
   Visual style exactly matches the reference images.
══════════════════════════════════════════════════════ */
function renderPoseLayer(
  ctx: CanvasRenderingContext2D,
  W: number, H: number,
  keypoints: Keypoint[],
  handPath: Pt[],
  layer: LayerKey,
  issueType?: string
) {
  const S = Math.min(W, H); // scale reference
  const dotR = S * 0.030;   // dot radius (~24px on 800px)
  const lineW = S * 0.008;  // line width

  // Helper: kp in pixel coords
  const px = (i: number): Pt => ({ x: (keypoints[i]?.x ?? 0.5) * W, y: (keypoints[i]?.y ?? 0.5) * H });
  const ok = (i: number, thresh = 0.25) => (keypoints[i]?.score ?? 1) > thresh;

  const lS = px(KP.L_SHOULDER);
  const rS = px(KP.R_SHOULDER);
  const lE = px(KP.L_ELBOW);
  const rE = px(KP.R_ELBOW);
  const lW = px(KP.L_WRIST);
  const rW = px(KP.R_WRIST);
  const lH = px(KP.L_HIP);
  const rH = px(KP.R_HIP);
  const lK = px(KP.L_KNEE);
  const rK = px(KP.R_KNEE);
  const nose = px(KP.NOSE);
  const midS: Pt = { x: (lS.x + rS.x) / 2, y: (lS.y + rS.y) / 2 };
  const midH: Pt = { x: (lH.x + rH.x) / 2, y: (lH.y + rH.y) / 2 };
  const midW: Pt = { x: (lW.x + rW.x) / 2, y: (lW.y + rW.y) / 2 };

  /* ──────────────────────────────
     BODY LAYER
     - Spine (neck→hip)
     - Shoulder line (green, like image 1)
     - Hip line (green)
     - Knee lines (white subtle)
     - Large green dots at shoulders + hips
  ────────────────────────────── */
  if (layer === 'body' || layer === 'all') {
    // Knee connectors (subtle)
    if (ok(KP.L_HIP) && ok(KP.L_KNEE)) {
      structureLine(ctx, lH.x, lH.y, lK.x, lK.y, C.WHITE, lineW * 0.6, 0.45);
      structureLine(ctx, rH.x, rH.y, rK.x, rK.y, C.WHITE, lineW * 0.6, 0.45);
    }
    // Spine
    if (ok(KP.L_SHOULDER) && ok(KP.L_HIP)) {
      structureLine(ctx, midS.x, midS.y, midH.x, midH.y, C.WHITE, lineW * 1.1, 0.75);
    }
    // Shoulder line (green — key coaching reference like image 1)
    if (ok(KP.L_SHOULDER) && ok(KP.R_SHOULDER)) {
      structureLine(ctx, lS.x, lS.y, rS.x, rS.y, C.GREEN, lineW * 1.2, 0.88);
    }
    // Hip line (green)
    if (ok(KP.L_HIP) && ok(KP.R_HIP)) {
      structureLine(ctx, lH.x, lH.y, rH.x, rH.y, C.GREEN, lineW * 1.0, 0.75);
    }

    // DOTS — large filled circles like reference image 1
    if (ok(KP.L_SHOULDER)) filledDot(ctx, lS.x, lS.y, dotR, C.GREEN, 0.92);
    if (ok(KP.R_SHOULDER)) filledDot(ctx, rS.x, rS.y, dotR, C.GREEN, 0.92);
    if (ok(KP.L_HIP))      filledDot(ctx, lH.x, lH.y, dotR * 0.82, C.GREEN, 0.75);
    if (ok(KP.R_HIP))      filledDot(ctx, rH.x, rH.y, dotR * 0.82, C.GREEN, 0.75);
    if (ok(KP.NOSE))       filledDot(ctx, nose.x, nose.y, dotR * 0.65, C.WHITE, 0.65);

    // Issue-specific body diagnostics
    if (issueType === 'early_extension' && ok(KP.L_SHOULDER) && ok(KP.L_HIP)) {
      // Spine angle extension line (shows where spine should reach)
      const dx = midH.x - midS.x, dy = midH.y - midS.y;
      structureLine(ctx, midH.x, midH.y, midH.x + dx * 0.5, midH.y + dy * 0.5, C.YELLOW, lineW * 0.7, 0.5, true);
    }
    if (issueType === 'head_movement' && ok(KP.NOSE)) {
      // Vertical reference line through head
      structureLine(ctx, nose.x, H * 0.02, nose.x, H * 0.22, C.YELLOW, lineW * 0.6, 0.45, true);
    }
  }

  /* ──────────────────────────────
     HANDS / ARMS LAYER
     - Left arm chain: shoulder → elbow → wrist (GREEN)
     - Right arm chain: shoulder → elbow → wrist (RED)
     - Grip V-shape: both shoulders → grip midpoint (like image 1!)
     - Hand path trajectory curve (YELLOW)
  ────────────────────────────── */
  if (layer === 'hands' || layer === 'all') {
    // Left arm chain (green — lead arm)
    if (ok(KP.L_SHOULDER) && ok(KP.L_ELBOW)) structureLine(ctx, lS.x, lS.y, lE.x, lE.y, C.GREEN, lineW * 1.15, 0.85);
    if (ok(KP.L_ELBOW) && ok(KP.L_WRIST))    structureLine(ctx, lE.x, lE.y, lW.x, lW.y, C.GREEN, lineW * 1.15, 0.85);

    // Right arm chain (red — trail arm)
    if (ok(KP.R_SHOULDER) && ok(KP.R_ELBOW)) structureLine(ctx, rS.x, rS.y, rE.x, rE.y, C.RED, lineW * 1.15, 0.85);
    if (ok(KP.R_ELBOW) && ok(KP.R_WRIST))    structureLine(ctx, rE.x, rE.y, rW.x, rW.y, C.RED, lineW * 1.15, 0.85);

    // V-SHAPE GRIP (Image 1 visual: left shoulder → grip, right shoulder → grip)
    if (ok(KP.L_SHOULDER) && ok(KP.R_SHOULDER) && ok(KP.L_WRIST) && ok(KP.R_WRIST)) {
      structureLine(ctx, lS.x, lS.y, midW.x, midW.y, C.GREEN, lineW * 1.0, 0.75);
      structureLine(ctx, rS.x, rS.y, midW.x, midW.y, C.GREEN, lineW * 1.0, 0.75);
    }

    // Elbow dots
    if (ok(KP.L_ELBOW)) filledDot(ctx, lE.x, lE.y, dotR * 0.72, C.GREEN, 0.88);
    if (ok(KP.R_ELBOW)) filledDot(ctx, rE.x, rE.y, dotR * 0.72, C.RED,   0.88);

    // WRIST/GRIP dot — large, prominent (like image 1 bottom dot)
    if (ok(KP.L_WRIST) && ok(KP.R_WRIST)) {
      filledDot(ctx, midW.x, midW.y, dotR * 1.1, C.GREEN, 1.0); // grip point
    } else {
      if (ok(KP.L_WRIST)) filledDot(ctx, lW.x, lW.y, dotR, C.GREEN, 1.0);
      if (ok(KP.R_WRIST)) filledDot(ctx, rW.x, rW.y, dotR, C.RED,   1.0);
    }

    // HAND PATH TRAJECTORY — yellow curve (like Image 2)
    if (handPath.length > 2) {
      smoothCurve(ctx, handPath, C.YELLOW, lineW * 1.5, 0.88);
    }
  }

  /* ──────────────────────────────
     CLUB LAYER
     Estimated from wrist + forearm direction vector.
     Phase 2: real club head tracking from Python.
  ────────────────────────────── */
  if (layer === 'club' || layer === 'all') {
    if (ok(KP.R_ELBOW) && ok(KP.R_WRIST)) {
      // Forearm direction vector → extrapolate to club head
      const fvX = rW.x - rE.x, fvY = rW.y - rE.y;
      const fLen = Math.hypot(fvX, fvY);
      if (fLen > 8) {
        const nX = fvX / fLen, nY = fvY / fLen;
        const clubLen = S * 0.32; // ~320px on 1000px wide
        const headX = rW.x + nX * clubLen;
        const headY = rW.y + nY * clubLen;

        // Club shaft line
        structureLine(ctx, rW.x, rW.y, headX, headY, C.YELLOW, lineW * 1.4, 0.88);
        // Club head dot
        filledDot(ctx, headX, headY, dotR * 1.0, C.YELLOW, 1.0);
        overlayLabel(ctx, 'Club head', headX, headY - dotR * 2.2, C.YELLOW, Math.max(9, S * 0.034), 0.8);
      }
    }
    // Grip dot (club handle end — near wrist midpoint)
    if (ok(KP.L_WRIST) && ok(KP.R_WRIST)) {
      filledDot(ctx, midW.x, midW.y, dotR * 0.65, C.YELLOW, 0.75);
    }
  }
}

/* ══════════════════════════════════════════════════════
   REFERENCE GUIDE FALLBACK
   When no MediaPipe data is available, shows a light
   reference guide from the timeline stub data.
══════════════════════════════════════════════════════ */
function renderGuideOverlay(
  ctx: CanvasRenderingContext2D,
  W: number, H: number,
  normT: number,
  layer: string,
  timeline: OverlayTimeline
) {
  const kfs = timeline.keyframes;
  if (!kfs.length) return;

  // Find surrounding keyframes
  let prev = kfs[0], next = kfs[0], frac = 0;
  for (let i = 0; i < kfs.length - 1; i++) {
    if (normT >= kfs[i].t && normT <= kfs[i + 1].t) {
      prev = kfs[i]; next = kfs[i + 1];
      const span = next.t - prev.t;
      frac = span > 0 ? (normT - prev.t) / span : 0;
      break;
    }
    if (normT >= kfs[kfs.length - 1].t) { prev = next = kfs[kfs.length - 1]; }
  }

  const lerp = (a: number, b: number) => a + (b - a) * frac;

  // Render elements, filtering by layer
  const renderEl = (els: typeof prev.elements) => {
    for (const el of els) {
      if (el.type === 'line') {
        const L = el as { type: 'line'; color: string; x1: number; y1: number; x2: number; y2: number; strokeWidth?: number; dashed?: boolean; opacity?: number };
        const layerOk = (layer === 'all')
          || (layer === 'body' && (L.color === 'white' || L.color === 'green'))
          || (layer === 'hands' && L.color === 'red')
          || (layer === 'club' && L.color === 'yellow');

        if (!layerOk && layer !== 'all') continue;
        const color = L.color === 'red' ? C.RED : L.color === 'green' ? C.GREEN : L.color === 'yellow' ? C.YELLOW : C.WHITE;
        structureLine(ctx, L.x1 * W, L.y1 * H, L.x2 * W, L.y2 * H, color, (L.strokeWidth ?? 2.5) * W / 320, L.opacity ?? 0.8, L.dashed);
      }
      else if (el.type === 'circle') {
        const CE = el as { type: 'circle'; color: string; cx: number; cy: number; r: number; opacity?: number };
        const color = CE.color === 'red' ? C.RED : CE.color === 'green' ? C.GREEN : CE.color === 'yellow' ? C.YELLOW : C.WHITE;
        filledDot(ctx, CE.cx * W, CE.cy * H, CE.r * W, color, CE.opacity ?? 0.85);
      }
      else if (el.type === 'arrow') {
        const AE = el as { type: 'arrow'; color: string; fromX: number; fromY: number; toX: number; toY: number; opacity?: number };
        const color = AE.color === 'red' ? C.RED : AE.color === 'green' ? C.GREEN : AE.color === 'yellow' ? C.YELLOW : C.WHITE;
        arrowLine(ctx, AE.fromX * W, AE.fromY * H, AE.toX * W, AE.toY * H, color, 3 * W / 320, AE.opacity ?? 0.9);
      }
      else if (el.type === 'label') {
        const LE = el as { type: 'label'; color: string; x: number; y: number; text: string; size?: number; opacity?: number };
        const color = LE.color === 'red' ? C.RED : LE.color === 'green' ? C.GREEN : LE.color === 'yellow' ? C.YELLOW : C.WHITE;
        overlayLabel(ctx, LE.text, LE.x * W, LE.y * H, color, (LE.size ?? 12) * W / 320, LE.opacity ?? 1);
      }
    }
  };

  renderEl(prev.elements);
  if (frac > 0.5 && next !== prev) {
    ctx.globalAlpha = (frac - 0.5) * 1.8;
    renderEl(next.elements);
    ctx.globalAlpha = 1;
  }

  // Status hint
  overlayLabel(ctx, 'Reference guide — AI tracking pending', W / 2, H - 12, 'rgba(168,240,64,0.55)', 9 * W / 320);
}

/* ══════════════════════════════════════════════════════
   MAIN COMPONENT
══════════════════════════════════════════════════════ */
export function SwingPlayer({ videoUrl, timeline, issueType }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const handPathRef = useRef<Pt[]>([]);
  const prevTimeRef = useRef(-1);
  const poseFramesRef = useRef<PoseFrame[]>([]);

  const [layer, setLayer] = useState<LayerKey>('all');
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1.0);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [currentPhase, setCurrentPhase] = useState('address');
  const [dragging, setDragging] = useState(false);
  const [hasPose, setHasPose] = useState(false);
  const progressBarRef = useRef<HTMLDivElement>(null);

  // Load pose frames from timeline if available
  useEffect(() => {
    type TL = typeof timeline & { poseFrames?: PoseFrame[] };
    const tl = timeline as TL;
    if (tl.poseFrames && tl.poseFrames.length > 0) {
      poseFramesRef.current = tl.poseFrames;
      setHasPose(true);
    }
  }, [timeline]);

  // Sync canvas size to video
  const syncCanvas = useCallback(() => {
    const v = videoRef.current;
    const c = canvasRef.current;
    if (!v || !c) return;
    const rect = v.getBoundingClientRect();
    if (rect.width > 0) { c.width = rect.width; c.height = rect.height; }
  }, []);

  // rAF render loop
  const renderFrame = useCallback(() => {
    const v = videoRef.current;
    const c = canvasRef.current;
    if (!v || !c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;

    const W = c.width || 320;
    const H = c.height || 240;
    ctx.clearRect(0, 0, W, H);

    const t = v.currentTime;
    const dur = v.duration || 1;
    const normT = t / dur;

    setProgress(normT);
    setCurrentTime(t);

    // Determine phase
    const p = timeline.phases;
    let phase = 'address';
    if (normT >= p.finish) phase = 'finish';
    else if (normT >= p.impact) phase = 'impact';
    else if (normT >= p.transition) phase = 'transition';
    else if (normT >= p.top) phase = 'top';
    // takeaway check
    setCurrentPhase(phase);

    if (hasPose && poseFramesRef.current.length > 0) {
      // ── REAL POSE DATA (from Python MediaPipe analysis) ──
      const keypoints = getPoseAt(poseFramesRef.current, t);
      if (keypoints) {
        // Accumulate hand path
        const lW = keypoints[KP.L_WRIST];
        const rW = keypoints[KP.R_WRIST];
        if (lW && rW) {
          const gx = ((lW.x + rW.x) / 2) * W;
          const gy = ((lW.y + rW.y) / 2) * H;
          const hist = handPathRef.current;
          // Reset on scrub backward
          if (t < prevTimeRef.current - 0.5) hist.length = 0;
          const last = hist[hist.length - 1];
          if (!last || Math.hypot(gx - last.x, gy - last.y) > W * 0.008) {
            hist.push({ x: gx, y: gy });
            if (hist.length > 120) hist.shift();
          }
        }
        prevTimeRef.current = t;
        renderPoseLayer(ctx, W, H, keypoints, handPathRef.current, layer, issueType);
      }
    } else {
      // ── GUIDE OVERLAY FALLBACK ──
      renderGuideOverlay(ctx, W, H, normT, layer, timeline);
    }
  }, [hasPose, layer, issueType, timeline]);

  useEffect(() => {
    let id: number;
    const loop = () => { renderFrame(); id = requestAnimationFrame(loop); };
    id = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(id);
  }, [renderFrame]);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    v.addEventListener('loadedmetadata', () => { setDuration(v.duration); syncCanvas(); });
    window.addEventListener('resize', syncCanvas);
    return () => window.removeEventListener('resize', syncCanvas);
  }, [syncCanvas]);

  // Controls
  const togglePlay = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) { v.play(); setPlaying(true); } else { v.pause(); setPlaying(false); }
  };

  const setSpeedFn = (s: number) => {
    setSpeed(s);
    if (videoRef.current) videoRef.current.playbackRate = s;
  };

  const jumpPhase = (ph: keyof PhaseMarkers) => {
    const v = videoRef.current;
    if (!v || !v.duration) return;
    v.currentTime = timeline.phases[ph] * v.duration;
    handPathRef.current = [];
  };

  const stepFrame = (dir: 1 | -1) => {
    const v = videoRef.current;
    if (!v) return;
    v.pause(); setPlaying(false);
    v.currentTime = Math.max(0, Math.min(v.duration, v.currentTime + dir / 30));
  };

  const scrub = useCallback((clientX: number) => {
    const bar = progressBarRef.current;
    const v = videoRef.current;
    if (!bar || !v || !v.duration) return;
    const r = bar.getBoundingClientRect();
    v.currentTime = Math.max(0, Math.min(1, (clientX - r.left) / r.width)) * v.duration;
    handPathRef.current = [];
  }, []);

  const fmt = (s: number) => `${Math.floor(s / 60)}:${Math.floor(s % 60).toString().padStart(2, '0')}`;

  return (
    <div className="sp-root">
      {/* ══ VIDEO + CANVAS STACK ══ */}
      <div className="sp-video-wrap">
        <video
          ref={videoRef}
          src={videoUrl}
          playsInline
          className="sp-video"
          crossOrigin="anonymous"
          onEnded={() => setPlaying(false)}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
        />
        <canvas ref={canvasRef} className="sp-canvas" />

        {/* Phase badge */}
        <div className="sp-top-bar">
          <span className="sp-phase-badge">{currentPhase.toUpperCase()}</span>
          <span className={`sp-ai-badge ${hasPose ? 'sp-ai-ready' : 'sp-ai-pending'}`}>
            {hasPose ? '🤖 AI Tracking' : '📐 Guide'}
          </span>
        </div>

        {/* Legend */}
        <div className="sp-legend">
          <span className="sp-leg sp-leg-l">● Left</span>
          <span className="sp-leg sp-leg-r">● Right/Target</span>
          <span className="sp-leg sp-leg-y">● Path</span>
        </div>

        <div className="sp-tap" onClick={togglePlay} />
      </div>

      {/* ══ LAYER TOGGLE ══ */}
      <div className="sp-layer-row">
        <span className="sp-layer-label">View:</span>
        {LAYERS.map(({ key, icon, label }) => (
          <button key={key} className={`sp-layer-btn ${layer === key ? 'sp-layer-active' : ''}`} onClick={() => setLayer(key)}>
            <span>{icon}</span><span>{label}</span>
          </button>
        ))}
      </div>

      {/* ══ PLAYBACK CONTROLS ══ */}
      <div className="sp-controls">
        <button className="sp-step-btn" onClick={() => stepFrame(-1)}>⏮</button>
        <button className="sp-play-btn" onClick={togglePlay}>{playing ? '⏸' : '▶'}</button>
        <button className="sp-step-btn" onClick={() => stepFrame(1)}>⏭</button>
        <div className="sp-speed-row">
          {SPEEDS.map(s => (
            <button key={s} className={`sp-speed-btn ${speed === s ? 'sp-speed-active' : ''}`} onClick={() => setSpeedFn(s)}>{s}x</button>
          ))}
        </div>
      </div>

      {/* ══ SCRUB BAR ══ */}
      <div className="sp-scrub-wrap">
        <span className="sp-time">{fmt(currentTime)}</span>
        <div
          ref={progressBarRef}
          className="sp-scrub-track"
          onMouseDown={e => { setDragging(true); scrub(e.clientX); }}
          onMouseMove={e => { if (dragging) scrub(e.clientX); }}
          onMouseUp={() => setDragging(false)}
          onMouseLeave={() => setDragging(false)}
          onTouchStart={e => { setDragging(true); scrub(e.touches[0].clientX); }}
          onTouchMove={e => { if (dragging) scrub(e.touches[0].clientX); }}
          onTouchEnd={() => setDragging(false)}
        >
          <div className="sp-scrub-fill" style={{ width: `${progress * 100}%` }} />
          <div className="sp-scrub-thumb" style={{ left: `calc(${progress * 100}% - 7px)` }} />
          {Object.entries(timeline.phases).map(([ph, t]) => (
            <div key={ph} className="sp-phase-tick" style={{ left: `${(t as number) * 100}%` }} />
          ))}
        </div>
        <span className="sp-time">{fmt(duration)}</span>
      </div>

      {/* ══ PHASE JUMP ══ */}
      <div className="sp-phases">
        {PHASES.map(({ key, label }) => (
          <button key={key} className={`sp-phase-btn ${currentPhase === key ? 'sp-phase-active' : ''}`} onClick={() => jumpPhase(key)}>
            {label}
          </button>
        ))}
      </div>

      <style>{css}</style>
    </div>
  );
}

const css = `
  .sp-root { display:flex; flex-direction:column; background:#060a06; width:100%; }

  /* Video */
  .sp-video-wrap { position:relative; width:100%; background:#000; }
  .sp-video { width:100%; display:block; max-height:340px; object-fit:contain; background:#000; }
  .sp-canvas { position:absolute; top:0; left:0; pointer-events:none; width:100%; height:100%; }
  .sp-tap { position:absolute; inset:0; cursor:pointer; }

  /* Badges */
  .sp-top-bar { position:absolute; top:8px; left:8px; display:flex; flex-direction:column; gap:5px; pointer-events:none; }
  .sp-phase-badge { display:inline-block; background:rgba(0,0,0,0.68); color:#a8f040; font-size:10px; font-weight:800; letter-spacing:.1em; padding:4px 10px; border-radius:100px; font-family:'DM Sans',system-ui; }
  .sp-ai-badge { display:inline-block; font-size:10px; font-weight:700; padding:4px 10px; border-radius:100px; font-family:'DM Sans',system-ui; width:fit-content; }
  .sp-ai-ready { background:rgba(60,238,60,0.18); color:#3cee3c; border:1px solid rgba(60,238,60,0.35); }
  .sp-ai-pending { background:rgba(0,0,0,0.6); color:rgba(168,240,64,0.6); }

  /* Legend */
  .sp-legend { position:absolute; bottom:8px; right:8px; background:rgba(0,0,0,0.68); display:flex; gap:8px; padding:5px 10px; border-radius:100px; pointer-events:none; }
  .sp-leg { font-size:10px; font-weight:700; font-family:'DM Sans',system-ui; }
  .sp-leg-l { color:#ff3c3c; }
  .sp-leg-r { color:#3cee3c; }
  .sp-leg-y { color:#ffd040; }

  /* Layer toggle */
  .sp-layer-row { display:flex; align-items:center; gap:5px; padding:8px 12px; background:#0a100a; border-bottom:1px solid rgba(255,255,255,0.05); overflow-x:auto; -webkit-overflow-scrolling:touch; }
  .sp-layer-row::-webkit-scrollbar { display:none; }
  .sp-layer-label { font-size:11px; font-weight:600; color:#2a3a25; white-space:nowrap; flex-shrink:0; font-family:'DM Sans',system-ui; }
  .sp-layer-btn { display:flex; align-items:center; gap:4px; flex-shrink:0; font-size:12px; font-weight:700; font-family:'DM Sans',system-ui; color:#3a4a35; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.07); padding:6px 12px; border-radius:8px; cursor:pointer; transition:all 0.15s; -webkit-tap-highlight-color:transparent; }
  .sp-layer-btn:active { transform:scale(0.94); }
  .sp-layer-active { color:#a8f040 !important; background:rgba(168,240,64,0.12) !important; border-color:rgba(168,240,64,0.3) !important; }

  /* Controls */
  .sp-controls { display:flex; align-items:center; gap:8px; padding:10px 14px; background:#0a100a; border-bottom:1px solid rgba(255,255,255,0.05); }
  .sp-play-btn { font-size:18px; background:#a8f040; color:#080c08; border:none; border-radius:50%; width:38px; height:38px; cursor:pointer; display:flex; align-items:center; justify-content:center; flex-shrink:0; -webkit-tap-highlight-color:transparent; transition:transform 0.1s; }
  .sp-play-btn:active { transform:scale(0.91); }
  .sp-step-btn { font-size:14px; background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.1); color:#7a8a72; border-radius:8px; width:34px; height:34px; cursor:pointer; display:flex; align-items:center; justify-content:center; -webkit-tap-highlight-color:transparent; }
  .sp-speed-row { display:flex; gap:4px; margin-left:auto; }
  .sp-speed-btn { font-size:12px; font-weight:700; font-family:'DM Sans',system-ui; background:rgba(255,255,255,0.05); color:#4a5a44; border:1px solid rgba(255,255,255,0.08); border-radius:6px; padding:5px 9px; cursor:pointer; transition:all 0.15s; }
  .sp-speed-active { background:rgba(168,240,64,0.12) !important; color:#a8f040 !important; border-color:rgba(168,240,64,0.3) !important; }

  /* Scrub */
  .sp-scrub-wrap { display:flex; align-items:center; gap:8px; padding:6px 14px 8px; background:#0a100a; border-bottom:1px solid rgba(255,255,255,0.05); }
  .sp-time { font-size:11px; font-weight:600; color:#3a4a35; font-family:'DM Sans',system-ui; white-space:nowrap; }
  .sp-scrub-track { flex:1; height:24px; display:flex; align-items:center; position:relative; cursor:pointer; touch-action:none; }
  .sp-scrub-track::before { content:''; position:absolute; left:0; right:0; top:50%; height:4px; background:rgba(255,255,255,0.1); border-radius:2px; transform:translateY(-50%); }
  .sp-scrub-fill { position:absolute; left:0; top:50%; height:4px; background:#a8f040; border-radius:2px; transform:translateY(-50%); pointer-events:none; }
  .sp-scrub-thumb { position:absolute; top:50%; width:14px; height:14px; background:#fff; border-radius:50%; transform:translateY(-50%); box-shadow:0 1px 6px rgba(0,0,0,0.6); pointer-events:none; z-index:2; }
  .sp-phase-tick { position:absolute; top:50%; width:2px; height:9px; background:rgba(168,240,64,0.5); transform:translate(-50%,-50%); border-radius:1px; pointer-events:none; }

  /* Phase buttons */
  .sp-phases { display:flex; gap:0; background:#0a100a; padding:8px 14px; overflow-x:auto; -webkit-overflow-scrolling:touch; }
  .sp-phases::-webkit-scrollbar { display:none; }
  .sp-phase-btn { flex-shrink:0; font-size:12px; font-weight:700; font-family:'DM Sans',system-ui; color:#3a4a35; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.06); padding:7px 13px; border-radius:8px; cursor:pointer; margin-right:6px; white-space:nowrap; transition:all 0.15s; -webkit-tap-highlight-color:transparent; }
  .sp-phase-active { color:#a8f040 !important; background:rgba(168,240,64,0.1) !important; border-color:rgba(168,240,64,0.25) !important; }
`;
