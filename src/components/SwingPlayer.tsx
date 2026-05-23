'use client';

/**
 * SwingPlayer.tsx — Interactive Swing Player
 *
 * 视频是页面主角。播放器尽量满屏，控制条紧凑在下方。
 * Canvas overlay 与视频时间精确同步。
 * 三层切换：Body / Arms / Club / All
 *
 * ── SwingCue disc semantics (NORMATIVE, PR-5.7 lock-in) ───────────────────
 *
 * The disc is a PHASE-AWARE COACHING VISUAL PLANE anchored by body
 * keypoints — not a physically exact 3D reconstruction of body geometry.
 *
 * Three-layer signal model (per PR-5.7 audit + Jason ref-image):
 *
 *   1. PHASE    — primary visual state. The swing phase (setup → top →
 *                 transition → impact → finish) drives a per-layer
 *                 compression curve on the disc's visual width. Shoulder
 *                 and hip compress on independent curves because the
 *                 body opens differently at each.
 *   2. KEYPOINTS — anchor only. Body keypoints determine:
 *                   - Center position (cx, cy = midpoint of L/R kp)
 *                   - Anchor direction (angleRad = atan2 of L/R kp)
 *                   - Body-layer identity (shoulder plane vs hip plane)
 *                 Body keypoints DO NOT directly determine visual width.
 *   3. currentDist — ±10% micro-correction layered on top of the phase
 *                 compression so the disc still "breathes" with live kp
 *                 foreshortening, without letting kp dominate the
 *                 coaching readability of the plane.
 *
 * final_rx = baseline_rx × phase_compression × (1 + micro_correction)
 *
 * Baseline_rx is captured once from a median of setup-phase frames
 * (PR-5.6), held stable across the swing. This is a coaching overlay,
 * not a physically exact 3D reconstruction. Readability and
 * instructional clarity are prioritised over raw 2D foreshortening
 * fidelity.
 *
 * Lesson history:
 *   PR-5.1 — rx locked absolutely        → disc detached during rotation
 *   PR-5.5 — rx tied to currentDist      → disc collapsed at top/finish
 *   PR-5.6 — baseline-locked, kp anchor  → readable AND anchored, but
 *                                          visually static through swing
 *   PR-5.7 — phase-driven compression    → readable, anchored, AND
 *                                          communicates body rotation
 */

import { useRef, useEffect, useState, useCallback } from 'react';
import { renderFrame } from '@/lib/overlay/OverlayRenderer';
import { getOverlayAtTime, getCurrentPhase, formatTime } from '@/lib/overlay/playerSync';
import type { OverlayElement, OverlayTimeline, PhaseMarkers, PoseTimeline } from '@/types/analysis';
import { SkeletonOverlay } from '@/components/SkeletonOverlay';
// PR-7c-frontend-v2: enhanced coaching overlay — sourced from
// production MediaPipe pose_timeline_2d (the same data feeding
// SkeletonOverlay). Auto-enabled whenever poseTimeline is present.
import { CoachingAnchorOverlay } from '@/components/CoachingAnchorOverlay';
import { EnhancedCoachingBadge } from '@/components/EnhancedCoachingBadge';
// PR-5: frame-level disc geometry from PR-4 pose_timeline_2d.
// PR-5.9: `frameAt` kept (deprecated) for any out-of-tree consumer;
// SwingPlayer now uses `interpolatedFrame` for continuous tracking.
import { interpolatedFrame } from '@/lib/disc/frameAt';
import {
  computeShoulderDisc,
  computeHipDisc,
  DISC_RX_RATIO,
  PERSPECTIVE_RY_RATIO,
} from '@/lib/disc/computeDiscParams';
import { unwrapAngle } from '@/lib/disc/unwrap';
import {
  getPhaseCompression,
  computeMicroCorrection,
} from '@/lib/disc/phaseCompression';
// PR-5.8A: defaults for the coaching-anchor expansion props. The URL
// is parsed in result/[id]/page.tsx; this module only consumes.
import {
  SHOULDER_EXPAND_DEFAULT,
  HIP_EXPAND_DEFAULT,
} from '@/lib/skeleton/coachingAnchors';

// ── PR-5.4 visual constants ──────────────────────────────────────────────
// Neon green for both discs and the kp-line glow. Jason's single-color
// decision (PR-5.4): shoulder vs hip will be distinguished later by a
// rotation-angle numeric readout, not by hue.
const NEON_GREEN = '#00ff88';
// 3D perspective tilt — simulates viewing a horizontal body-rotation
// plane from below. Applied via ctx.transform y-axis squish by cos(tilt).
const PERSPECTIVE_TILT_DEG = 25;
const PERSPECTIVE_TILT_RAD = (PERSPECTIVE_TILT_DEG * Math.PI) / 180;

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
  // PR-5.8A: render-time coaching-anchor expansion factors. URL-sourced
  // (?shoulderExpand=, ?hipExpand=) by the result page. Consumed by
  // SkeletonOverlay (PR-5.8A commit 2) and computeShoulderDisc/Hip
  // (commit 3). Optional; defaults live in lib/skeleton/coachingAnchors.
  shoulderExpand?: number;
  hipExpand?: number;
  // PR-5.9 Task 5: debug overlay mode. Only `'pose'` is meaningful right
  // now — when set, SkeletonOverlay renders the raw_keypoints sibling
  // (when present) as small blue dots alongside the final keypoints.
  // URL-sourced (?debug=pose) by the result page.
  debugMode?: 'pose';
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
 * PR-5.4: draw a 3D-tilted neon disc representing a body rotation plane.
 *
 * The ellipse is rendered in a transformed coordinate system that
 * simulates viewing a horizontal plane from below at PERSPECTIVE_TILT_DEG.
 * Stroke is a neon outer (with shadow blur for the glow halo) plus a
 * thinner white inner highlight — the double-stroke is what gives the
 * "physical hoop" look in Jason's reference sample.
 *
 * cx/cy/rx/ry are all canvas-px (caller pre-scales). cx/cy stay exactly
 * on the kp midpoint (PR-5.3 honesty preserved) — only the visual radius
 * and tilt change.
 */
function drawTiltedDisc(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  rx: number,
  ry: number,
  angleRad: number,
  color: string,
): void {
  ctx.save();
  // 1. Move origin to disc center, apply in-plane rotation (the
  //    shoulder/hip line slope from PR-5.1 acos rotation).
  ctx.translate(cx, cy);
  ctx.rotate(angleRad);
  // 2. Camera-below-plane perspective: squish the y axis by cos(tilt)
  //    so the rotated ellipse looks like a tilted horizontal plane.
  const cosT = Math.cos(PERSPECTIVE_TILT_RAD);
  ctx.transform(1, 0, 0, cosT, 0, 0);
  // 3. Outer neon stroke + glow.
  ctx.shadowColor = color;
  ctx.shadowBlur = 12;
  ctx.strokeStyle = color;
  ctx.lineWidth = 5;
  ctx.beginPath();
  ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
  ctx.stroke();
  // 4. Inner white highlight (no shadow) — depth pass.
  ctx.shadowBlur = 0;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.restore();
}

/**
 * PR-5.6: draw the disc's anchor axis — a white-with-neon-glow segment
 * that spans the disc's major axis from -rx*0.85 to +rx*0.85 along
 * angleRad. Replaces PR-5.4's raw L/R kp connector (which collapsed to
 * a short segment at top/finish when the kp pair foreshortened toward
 * each other). The anchor axis is computed from the stable baseline
 * rx, so it reads as "the disc's equator" through the whole swing.
 *
 * Endpoints are in canvas px (caller pre-scales cx/cy and rx).
 */
function drawAnchorAxis(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  angleRad: number,
  rx: number,
): void {
  const halfLen = rx * 0.85;
  const dxAxis = Math.cos(angleRad) * halfLen;
  const dyAxis = Math.sin(angleRad) * halfLen;
  ctx.save();
  ctx.shadowColor = NEON_GREEN;
  ctx.shadowBlur = 8;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 4;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(cx - dxAxis, cy - dyAxis);
  ctx.lineTo(cx + dxAxis, cy + dyAxis);
  ctx.stroke();
  ctx.restore();
}

/**
 * PR-5.6: simple median for visual baseline sample aggregation. Sorts a
 * copy (preserving caller's array order) and returns the lower-middle
 * element. Used only on small arrays (<= ~10 elements) during setup-
 * phase collection, so the O(n log n) sort cost is negligible.
 *
 * Pre-condition: `samples.length >= 1` — caller must check.
 */
function median(samples: number[]): number {
  const sorted = [...samples].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

export function SwingPlayer({
  videoUrl,
  timeline,
  phases,
  duration: propDur,
  dataSource,
  poseTimeline,
  // PR-5.8A: coaching-anchor expansion factors. Defaults applied here
  // so internal call sites (SkeletonOverlay, computeShoulderDisc/Hip)
  // see a guaranteed number.
  shoulderExpand = SHOULDER_EXPAND_DEFAULT,
  hipExpand = HIP_EXPAND_DEFAULT,
  // PR-5.9 Task 5: debug overlay mode forwarded to SkeletonOverlay.
  debugMode,
}: Props) {
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

  // PR-7c-frontend-v2: enhanced overlay auto-enabled whenever production
  // MediaPipe pose_timeline_2d is available (= essentially every PR-4-
  // analyzed video). Replaces the v1 Supabase-Storage corrected-timeline
  // fetch which was per-video opt-in.
  const enhancedMode = !!poseTimeline;

  // PR-5 hotfix: per-disc rolling state so atan2 wrap-around (±π
  // boundary crossings between adjacent frames) doesn't make the
  // disc visually flip 360° between rAF ticks. See unwrap.ts.
  const lastShoulderRef = useRef<{ angleRad: number; ts: number } | null>(null);
  const lastHipRef      = useRef<{ angleRad: number; ts: number } | null>(null);

  // PR-5.6: per-video visual baseline (rx) for the disc. Set ONCE from
  // a median of setup-phase frames; held stable through the entire
  // swing so the coaching plane stays readable while cx/cy/angleRad
  // continue to track the live body keypoints. NOT used for rotation
  // correction (PR-5.1 §3.A acos amplification is permanently retired).
  const discAnchorRef = useRef<{ shoulderRx: number; hipRx: number } | null>(null);
  // Sample buffer for the median computation above. Pushed each rAF
  // frame while ts < 0.8 and all 4 source kp are valid. Cleared after
  // discAnchorRef is locked, to release the GC root.
  const baselineSamplesRef = useRef<{
    shoulderSamples: number[];
    hipSamples: number[];
  } | null>(null);

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
    if (!v) return;

    // State updates run in BOTH modes (enhanced + fallback) so the
    // scrub bar, time display, and phase badge keep working in
    // enhanced mode where the <canvas> is unmounted by the JSX gate.
    const t = v.currentTime;
    const d = v.duration || dur || 1;

    setProgress(t / d);
    setCurTime(t);
    setPhase(getCurrentPhase(phases, t, d));

    // PR-7c-frontend-v6: canvas draws only run in fallback mode. In
    // enhanced mode the JSX gate (`{!enhancedMode && <canvas .../>}`)
    // unmounts the canvas → canvasRef.current is null → bail here.
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;

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
      // PR-5.9 Task 3: interpolated lookup — disc cx/cy/angle now animate
      // continuously between pose samples instead of snapping per sample.
      const poseFrame = interpolatedFrame(poseTimeline, t);
      if (poseFrame) {
        const scaleX = cw / poseTimeline.video_width;
        const scaleY = ch / poseTimeline.video_height;

        // PR-5.6: collect & lock the median-based visual baseline (rx)
        // during setup. Cx/cy/angleRad still track live keypoints; only
        // rx uses this stable baseline. See file-level docstring.
        if (!discAnchorRef.current) {
          const ls = poseFrame.keypoints.left_shoulder;
          const rs = poseFrame.keypoints.right_shoulder;
          const lh = poseFrame.keypoints.left_hip;
          const rh = poseFrame.keypoints.right_hip;
          const allGood =
            ls[0] !== null && ls[1] !== null && ls[2] > 0.5 &&
            rs[0] !== null && rs[1] !== null && rs[2] > 0.5 &&
            lh[0] !== null && lh[1] !== null && lh[2] > 0.5 &&
            rh[0] !== null && rh[1] !== null && rh[2] > 0.5;

          if (poseFrame.ts < 0.8) {
            // Setup window — accumulate samples for median.
            if (allGood) {
              const sDx = (ls[0] as number) - (rs[0] as number);
              const sDy = (ls[1] as number) - (rs[1] as number);
              const hDx = (lh[0] as number) - (rh[0] as number);
              const hDy = (lh[1] as number) - (rh[1] as number);
              const buf = baselineSamplesRef.current
                ?? { shoulderSamples: [], hipSamples: [] };
              buf.shoulderSamples.push(Math.sqrt(sDx * sDx + sDy * sDy));
              buf.hipSamples.push(Math.sqrt(hDx * hDx + hDy * hDy));
              baselineSamplesRef.current = buf;
            }
            // Lock when we have enough samples OR we've crossed 0.6s.
            const buf = baselineSamplesRef.current;
            if (buf
                && buf.shoulderSamples.length >= 1
                && buf.hipSamples.length >= 1
                && (buf.shoulderSamples.length >= 5 || poseFrame.ts > 0.6)) {
              discAnchorRef.current = {
                shoulderRx: (median(buf.shoulderSamples) * DISC_RX_RATIO) / 2,
                hipRx:      (median(buf.hipSamples)      * DISC_RX_RATIO) / 2,
              };
              baselineSamplesRef.current = null;
            }
          } else {
            // Past 0.8s and still not locked — degraded path.
            const buf = baselineSamplesRef.current;
            if (buf
                && buf.shoulderSamples.length >= 1
                && buf.hipSamples.length >= 1) {
              discAnchorRef.current = {
                shoulderRx: (median(buf.shoulderSamples) * DISC_RX_RATIO) / 2,
                hipRx:      (median(buf.hipSamples)      * DISC_RX_RATIO) / 2,
              };
              baselineSamplesRef.current = null;
            } else if (allGood) {
              // No samples collected in setup window — use this first
              // valid post-setup frame directly.
              const sDx = (ls[0] as number) - (rs[0] as number);
              const sDy = (ls[1] as number) - (rs[1] as number);
              const hDx = (lh[0] as number) - (rh[0] as number);
              const hDy = (lh[1] as number) - (rh[1] as number);
              discAnchorRef.current = {
                shoulderRx: (Math.sqrt(sDx * sDx + sDy * sDy) / 2) * DISC_RX_RATIO,
                hipRx:      (Math.sqrt(hDx * hDx + hDy * hDy) / 2) * DISC_RX_RATIO,
              };
            }
          }
        }

        // PR-5.7: phase-driven compression (primary) × baseline rx ×
        // (1 + micro currentDist correction). PR-5.6 baseline anchor
        // remains the visual size source; phase + micro modulate it.
        // angle path: still atan2 (null baselineDist) — PR-5.1 §3.A
        // acos amplification is permanently retired.
        // PR-5.8A: pass the URL-sourced expansion factor so the disc
        // anchor and chord endpoints align with the expanded skeleton
        // dots/lines drawn by SkeletonOverlay (single source of truth).
        const shoulder = computeShoulderDisc(poseFrame, null, shoulderExpand);
        if (shoulder) {
          const unwrapped = unwrapAngle(
            shoulder.angleRad,
            lastShoulderRef.current?.angleRad ?? null,
            lastShoulderRef.current?.ts ?? null,
            t,
          );
          lastShoulderRef.current = { angleRad: unwrapped, ts: t };
          // Baseline rx (PR-5.6 median anchor); falls back to live rx
          // during the first few rAF ticks before samples are collected.
          const baselineRx = discAnchorRef.current?.shoulderRx ?? shoulder.rx;
          // PR-5.7 phase compression (smoothstep across 5 phase anchors).
          const phaseComp = getPhaseCompression(t, phases, 'shoulder');
          // PR-5.7 ±10% micro: dist is rx × 2 / DISC_RX_RATIO since rx
          // already bakes in the ratio (see computeDiscParams).
          const baselineDist = (baselineRx * 2) / DISC_RX_RATIO;
          const currentDist  = (shoulder.rx * 2) / DISC_RX_RATIO;
          const micro = computeMicroCorrection(currentDist, baselineDist);
          const finalRx = baselineRx * phaseComp * (1 + micro);
          const finalRxCv = finalRx * scaleX;
          drawTiltedDisc(
            ctx,
            shoulder.cx * scaleX,
            shoulder.cy * scaleY,
            finalRxCv,
            finalRx * PERSPECTIVE_RY_RATIO * scaleY,
            unwrapped,
            NEON_GREEN,
          );
          drawAnchorAxis(
            ctx,
            shoulder.cx * scaleX,
            shoulder.cy * scaleY,
            unwrapped,
            finalRxCv,
          );
        }

        // Hip lift in computeHipDisc targets the corrected shoulder midpoint;
        // pass null when the shoulder disc couldn't be computed this frame.
        const shoulderMid = shoulder ? { cx: shoulder.cx, cy: shoulder.cy } : null;
        const hip = computeHipDisc(poseFrame, null, shoulderMid, hipExpand);
        if (hip) {
          const unwrapped = unwrapAngle(
            hip.angleRad,
            lastHipRef.current?.angleRad ?? null,
            lastHipRef.current?.ts ?? null,
            t,
          );
          lastHipRef.current = { angleRad: unwrapped, ts: t };
          const baselineRx = discAnchorRef.current?.hipRx ?? hip.rx;
          const phaseComp = getPhaseCompression(t, phases, 'hip');
          const baselineDist = (baselineRx * 2) / DISC_RX_RATIO;
          const currentDist  = (hip.rx * 2) / DISC_RX_RATIO;
          const micro = computeMicroCorrection(currentDist, baselineDist);
          const finalRx = baselineRx * phaseComp * (1 + micro);
          const finalRxCv = finalRx * scaleX;
          drawTiltedDisc(
            ctx,
            hip.cx * scaleX,
            hip.cy * scaleY,
            finalRxCv,
            finalRx * PERSPECTIVE_RY_RATIO * scaleY,
            unwrapped,
            NEON_GREEN,
          );
          drawAnchorAxis(
            ctx,
            hip.cx * scaleX,
            hip.cy * scaleY,
            unwrapped,
            finalRxCv,
          );
        }
      }
    }
  }, [timeline, phases, layer, dur, poseTimeline, shoulderExpand, hipExpand]);

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
        {!enhancedMode && <canvas ref={canvasRef} className="sp-cvs" />}

        {/* PR-4: skeleton overlay (toggle, default off).
            PR-7c-frontend: hidden when enhanced mode is active so the
            magenta anchors don't visually compete with the MediaPipe
            skeleton. SkeletonOverlay early-returns null on hidden=true. */}
        {skeletonOn && poseTimeline && (
          <SkeletonOverlay
            timeline={poseTimeline}
            videoEl={videoRef.current}
            shoulderExpand={shoulderExpand}
            hipExpand={hipExpand}
            debugMode={debugMode}
            hidden={enhancedMode}
          />
        )}

        {/* PR-7c-frontend-v2: enhanced coaching anchors sourced from
            production MediaPipe pose_timeline_2d. Mounted whenever
            poseTimeline is available — replaces the MediaPipe
            SkeletonOverlay visually but reads the SAME underlying data. */}
        {enhancedMode && poseTimeline && (
          <CoachingAnchorOverlay
            poseTimeline={poseTimeline}
            phaseMarkers={phases}
            videoEl={videoRef.current}
          />
        )}

        {/* PR-7c-frontend: badge — only visible in enhanced mode. */}
        {enhancedMode && <EnhancedCoachingBadge />}

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
        {/* PR-7c-frontend: hide PR-5 disc/skeleton chrome in enhanced
            mode. The disc canvas is inert (SkeletonOverlay hidden),
            so layer-badge / skel-toggle / legend all label things
            that no longer render. */}
        {!enhancedMode && (
          <div className="sp-layer-badge">{layerBadgeText()}</div>
        )}

        {/* PR-4: skeleton toggle. Disabled when no timeline data — older
            videos predate PR-4 (re-analyze to enable).
            PR-7c-frontend: hidden in enhanced mode (toggle is a visual
            no-op since SkeletonOverlay returns null via hidden=true). */}
        {!enhancedMode && (
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
        )}

        {/* Legend — disc color key (Current/Target/Path).
            PR-7c-frontend: hidden in enhanced mode where disc doesn't
            render. Visible in fallback mode for existing users. */}
        {!enhancedMode && (
          <div className="sp-legend">
            <span className="leg-r">● Current</span>
            <span className="leg-g">● Target</span>
            <span className="leg-y">● Path</span>
          </div>
        )}

        <div className="sp-tap" onClick={togglePlay} />
      </div>

      {/* ══ LAYER TOGGLE ══
          PR-7c-frontend: hidden in enhanced mode. setLayer() gates
          disc-layer rendering which is inert when CoachingAnchorOverlay
          owns the visual. Visible in fallback mode for existing users. */}
      {!enhancedMode && (
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
      )}

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
