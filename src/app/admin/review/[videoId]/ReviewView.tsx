'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { TASK_PHASES, type TaskPhase, type Handedness } from '@/lib/types/annotation';

/**
 * PR-7A.1 Phase 2 — Validation review overlay.
 *
 * CRITICAL INVARIANT (do not break):
 *   The <video> element + SVG overlay below use the EXACT same setup
 *   as the workbench (src/app/admin/annotate/[videoId]/AnnotateWorkbench.tsx).
 *   Container: position relative, video native aspect ratio.
 *   Video: position absolute, inset 0, object-fit: contain.
 *   SVG:   position absolute, inset 0, viewBox in native pixel dims,
 *          preserveAspectRatio="xMidYMid meet" — same letterboxing rule
 *          as the browser's <video>.
 *   Frame seek: video.currentTime = (frameIdx + 0.5) / fps  (matches
 *               the workbench's seekToFrame so the rendered frame is
 *               the same one the annotator clicked on).
 *   fps_sampled = wham_video_meta.processed_fps (canonical), passed
 *               in as `fpsSampled` prop.
 *
 * Coords land 1:1 because everything renders into native-pixel viewBox
 * scaled by the browser the same way object-fit:contain scales the video.
 */

/* ════════════════════════════════════════════════════════════════════
   Types — shared with the server page via direct import
   ════════════════════════════════════════════════════════════════════ */

export type ReviewRow = {
  phase: string;
  verdict: string;
  notes: string | null;
};

export type PerPhaseData = {
  phase: TaskPhase;
  frameIdx: number;
  timeSec: number;
  armLead: {
    shoulder: { x: number; y: number } | null;
    elbow:    { x: number; y: number } | null;
    wrist:    { x: number; y: number } | null;
    visibility: string;
  } | null;
  armTrail: {
    shoulder: { x: number; y: number } | null;
    elbow:    { x: number; y: number } | null;
    wrist:    { x: number; y: number } | null;
    visibility: string;
  } | null;
  hipPair: {
    leadHip:  { x: number; y: number };
    trailHip: { x: number; y: number };
  } | null;
  wham: Record<string, { x: number; y: number } | null> | null;
  mediaPipe: Record<string, [number | null, number | null, number]> | null;
  existingReview: ReviewRow | null;
};

export interface ReviewViewProps {
  videoId: string;
  videoFilename: string | null;
  viewType: 'face_on' | 'down_the_line';
  handedness: Handedness;
  fpsSampled: number;
  videoWidth: number;
  videoHeight: number;
  signedUrl: string;
  perPhase: PerPhaseData[];
}

/* ════════════════════════════════════════════════════════════════════
   Color palette — locked from spec
   ════════════════════════════════════════════════════════════════════ */

const COLOR_GT_LEAD   = '#FFD86B';
const COLOR_GT_TRAIL  = '#4FB3FF';
const COLOR_GT_HIP    = '#FFFFFF';
const COLOR_WHAM      = '#9C9C9C';
const COLOR_MP        = '#E0E0E0';
const COLOR_REVIEWED  = '#1D9E75';
const COLOR_CORRECT   = '#1D9E75';
const COLOR_INCORRECT = '#A32D2D';
const COLOR_UNSURE    = '#888780';

/* ════════════════════════════════════════════════════════════════════
   Lead/Trail → WHAM joint name mapping
   ─────────────────────────────────────────────────────────────────────
   No existing helper in the codebase for this — defining it here
   based on the spec rule:
     face_on right-handed: lead = WHAM right_, trail = WHAM left_
   For face_on left-handed: flip.
   For DTL: TODO once spec'd; defaults to face_on RH mapping for now
   (the v1 test target 998e1930 is face_on right-handed).

   Why this isn't an "anatomical" mapping: WHAM's left_ and right_
   keys follow image-orientation convention AFTER the PR-7a.2 arm-
   chain swap, so they're already mirrored relative to the golfer's
   anatomy. Lead = arm closer to target = anatomical left for RH
   golfer = image-right = WHAM's right_ side.
   ════════════════════════════════════════════════════════════════════ */

function whamSideFor(
  arm: 'lead' | 'trail',
  _viewType: 'face_on' | 'down_the_line',
  handedness: Handedness,
): 'left' | 'right' {
  // For face_on:
  //   RH golfer: lead = WHAM right_, trail = WHAM left_
  //   LH golfer: lead = WHAM left_,  trail = WHAM right_
  // For DTL: same heuristic as face_on for v1 (architect can refine
  // once DTL videos are in the test set). _viewType is taken as an
  // arg now so the call sites are forwards-compatible.
  if (handedness === 'right') {
    return arm === 'lead' ? 'right' : 'left';
  }
  return arm === 'lead' ? 'left' : 'right';
}

/* ════════════════════════════════════════════════════════════════════
   Component
   ════════════════════════════════════════════════════════════════════ */

type Verdict = 'correct' | 'incorrect' | 'unsure';

export function ReviewView(props: ReviewViewProps) {
  const {
    videoId, videoWidth, videoHeight, fpsSampled, signedUrl,
    handedness, viewType, perPhase,
  } = props;

  // ── State ────────────────────────────────────────────────────────
  const [activeIdx, setActiveIdx] = useState(0);
  const active = perPhase[activeIdx];

  const [showGtArm, setShowGtArm] = useState(true);
  const [showGtHip, setShowGtHip] = useState(true);
  const [showWham,  setShowWham]  = useState(true);
  const [showMp,    setShowMp]    = useState(false);

  // Verdict working state — pre-filled from existingReview on phase change
  const [verdict, setVerdict] = useState<Verdict | null>(
    (active?.existingReview?.verdict as Verdict | undefined) ?? null,
  );
  const [notes, setNotes] = useState<string>(active?.existingReview?.notes ?? '');
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  // Reviews keyed by phase — keeps a local cache that updates on save
  // so the tab "reviewed" check appears immediately without a round-trip.
  const [reviewByPhase, setReviewByPhase] = useState<Map<string, ReviewRow>>(
    () => new Map(perPhase.flatMap(p => p.existingReview ? [[p.phase, p.existingReview] as [string, ReviewRow]] : [])),
  );

  // Reset verdict + notes when active phase changes
  useEffect(() => {
    const existing = reviewByPhase.get(active?.phase ?? '');
    setVerdict((existing?.verdict as Verdict | undefined) ?? null);
    setNotes(existing?.notes ?? '');
    setSaveError(null);
  }, [activeIdx, active?.phase, reviewByPhase]);

  // ── Video seek when active phase changes ─────────────────────────
  const videoRef = useRef<HTMLVideoElement>(null);
  const [, setSeekedFrameIdx] = useState<number | null>(null);

  const seekToFrame = useCallback((frameIdx: number) => {
    const v = videoRef.current;
    if (!v) return;
    const t = (frameIdx + 0.5) / fpsSampled;
    try { v.currentTime = t; } catch { /* defer */ }
  }, [fpsSampled]);

  useEffect(() => {
    if (active) seekToFrame(active.frameIdx);
  }, [active, seekToFrame]);

  const onVideoLoadedMetadata = useCallback(() => {
    if (active) seekToFrame(active.frameIdx);
  }, [active, seekToFrame]);

  // 'seeked' confirms the video actually reached the target frame so
  // the SVG overlay redraws coincident with the right pixel content
  // (avoids 1-frame flicker on rapid tab clicks).
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const handler = () => {
      const f = Math.round(v.currentTime * fpsSampled - 0.5);
      setSeekedFrameIdx(f);
    };
    v.addEventListener('seeked', handler);
    return () => v.removeEventListener('seeked', handler);
  }, [fpsSampled]);

  // ── Save verdict ─────────────────────────────────────────────────
  const handleSave = useCallback(async () => {
    if (!active || !verdict) return;
    setSaving(true);
    setSaveError(null);
    try {
      const res = await fetch('/api/admin/landmark-validation-review', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_id: videoId,
          phase: active.phase,
          verdict,
          notes: notes.length > 0 ? notes : null,
          compared_sources: ['wham', 'mediapipe'],
        }),
      });
      if (!res.ok) {
        const body = await res.text().catch(() => '');
        setSaveError(`HTTP ${res.status} · ${body.slice(0, 200)}`);
        setSaving(false);
        return;
      }
      // Update local cache so the tab marker + future revisits reflect.
      const next = new Map(reviewByPhase);
      next.set(active.phase, { phase: active.phase, verdict, notes: notes.length > 0 ? notes : null });
      setReviewByPhase(next);
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : 'unknown');
    } finally {
      setSaving(false);
    }
  }, [active, verdict, notes, videoId, reviewByPhase]);

  const reviewedCount = useMemo(() => reviewByPhase.size, [reviewByPhase]);

  /* ─── Render ─── */

  return (
    <div className="rv-shell">
      {/* HUD */}
      <div className="rv-hud">
        <span className="rv-badge">VALIDATION REVIEW</span>
        <span className="rv-progress">{reviewedCount} / {TASK_PHASES.length} phases reviewed</span>
      </div>

      {/* Phase tabs */}
      <div className="rv-tabs">
        {perPhase.map((p, i) => {
          const isActive = i === activeIdx;
          const isReviewed = reviewByPhase.has(p.phase);
          return (
            <button
              key={p.phase}
              type="button"
              className={`rv-tab ${isActive ? 'on' : ''} ${isReviewed ? 'reviewed' : ''}`}
              onClick={() => setActiveIdx(i)}
            >
              <span>{p.phase}</span>
              {isReviewed && <span className="rv-check" aria-label="reviewed">✓</span>}
            </button>
          );
        })}
      </div>

      {active && (
        <div className="rv-main">
          <div className="rv-frame-info">
            frame {active.frameIdx} · {active.timeSec.toFixed(2)}s
          </div>

          {/* Video + SVG overlay — exact workbench coord space */}
          <div
            className="rv-stage"
            style={{ aspectRatio: `${videoWidth} / ${videoHeight}` }}
          >
            <video
              ref={videoRef}
              className="rv-video"
              src={signedUrl}
              muted
              playsInline
              preload="auto"
              onLoadedMetadata={onVideoLoadedMetadata}
            />
            <svg
              className="rv-overlay"
              viewBox={`0 0 ${videoWidth} ${videoHeight}`}
              preserveAspectRatio="xMidYMid meet"
            >
              {showGtArm && active.armLead && (
                <ArmLayer arm={active.armLead} color={COLOR_GT_LEAD} />
              )}
              {showGtArm && active.armTrail && (
                <ArmLayer arm={active.armTrail} color={COLOR_GT_TRAIL} />
              )}
              {showGtHip && active.hipPair && (
                <HipDots pair={active.hipPair} color={COLOR_GT_HIP} />
              )}
              {showWham && active.wham && (
                <WhamSkeleton kpts={active.wham} viewType={viewType} handedness={handedness} />
              )}
              {showMp && active.mediaPipe && (
                <MediaPipeDots kpts={active.mediaPipe} />
              )}
            </svg>
          </div>

          {/* Source toggles */}
          <div className="rv-toggles">
            <label className="rv-toggle">
              <input type="checkbox" checked={showGtArm} onChange={e => setShowGtArm(e.target.checked)} />
              <span className="rv-swatch" style={{ background: COLOR_GT_LEAD }} />
              <span className="rv-swatch" style={{ background: COLOR_GT_TRAIL }} />
              <span>Jason GT (arm)</span>
            </label>
            <label className="rv-toggle">
              <input type="checkbox" checked={showGtHip} onChange={e => setShowGtHip(e.target.checked)} />
              <span className="rv-swatch" style={{ background: COLOR_GT_HIP }} />
              <span>Jason GT (hip)</span>
            </label>
            <label className="rv-toggle">
              <input type="checkbox" checked={showWham} onChange={e => setShowWham(e.target.checked)} />
              <span className="rv-swatch" style={{ background: COLOR_WHAM }} />
              <span>WHAM mesh</span>
            </label>
            <label className="rv-toggle">
              <input type="checkbox" checked={showMp} onChange={e => setShowMp(e.target.checked)} />
              <span className="rv-swatch" style={{ background: COLOR_MP }} />
              <span>MediaPipe</span>
            </label>
          </div>

          {/* Verdict bar */}
          <div className="rv-verdict-row">
            <button
              type="button"
              className={`rv-verdict ${verdict === 'correct' ? 'on' : ''}`}
              style={{ borderColor: COLOR_CORRECT, color: verdict === 'correct' ? '#000' : COLOR_CORRECT, background: verdict === 'correct' ? COLOR_CORRECT : 'transparent' }}
              onClick={() => setVerdict('correct')}
            >✓ Correct</button>
            <button
              type="button"
              className={`rv-verdict ${verdict === 'incorrect' ? 'on' : ''}`}
              style={{ borderColor: COLOR_INCORRECT, color: verdict === 'incorrect' ? '#fff' : COLOR_INCORRECT, background: verdict === 'incorrect' ? COLOR_INCORRECT : 'transparent' }}
              onClick={() => setVerdict('incorrect')}
            >✗ Incorrect</button>
            <button
              type="button"
              className={`rv-verdict ${verdict === 'unsure' ? 'on' : ''}`}
              style={{ borderColor: COLOR_UNSURE, color: verdict === 'unsure' ? '#fff' : COLOR_UNSURE, background: verdict === 'unsure' ? COLOR_UNSURE : 'transparent' }}
              onClick={() => setVerdict('unsure')}
            >? Unsure</button>
          </div>

          <textarea
            className="rv-notes"
            placeholder="Optional notes (e.g. 'wrist drifted 2cm left of bone')"
            value={notes}
            onChange={e => setNotes(e.target.value)}
            rows={2}
          />

          {saveError && <div className="rv-error">save failed: {saveError}</div>}

          <button
            type="button"
            className="rv-save"
            disabled={!verdict || saving}
            onClick={handleSave}
          >
            {saving ? 'Saving…' : 'Save verdict'}
          </button>
        </div>
      )}

      <style>{CSS}</style>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════
   Overlay primitives
   ════════════════════════════════════════════════════════════════════ */

function ArmLayer(props: {
  arm: NonNullable<PerPhaseData['armLead']>;
  color: string;
}) {
  const { arm, color } = props;
  const pts: Array<{ x: number; y: number }> = [];
  if (arm.shoulder) pts.push(arm.shoulder);
  if (arm.elbow)    pts.push(arm.elbow);
  if (arm.wrist)    pts.push(arm.wrist);
  const polylinePts = pts.map(p => `${p.x},${p.y}`).join(' ');
  return (
    <g>
      {pts.length >= 2 && (
        <polyline
          points={polylinePts}
          fill="none"
          stroke={color}
          strokeWidth={3}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      )}
      {pts.map((p, i) => (
        <circle
          key={i}
          cx={p.x} cy={p.y} r={8}
          fill={color}
          stroke="#000"
          strokeWidth={1.5}
        />
      ))}
    </g>
  );
}

function HipDots(props: {
  pair: NonNullable<PerPhaseData['hipPair']>;
  color: string;
}) {
  const { pair, color } = props;
  return (
    <g>
      <circle cx={pair.leadHip.x}  cy={pair.leadHip.y}  r={8}
              fill={color} stroke="#000" strokeWidth={1.5} />
      <circle cx={pair.trailHip.x} cy={pair.trailHip.y} r={8}
              fill={color} stroke="#000" strokeWidth={1.5} />
    </g>
  );
}

function WhamSkeleton(props: {
  kpts: Record<string, { x: number; y: number } | null>;
  viewType: 'face_on' | 'down_the_line';
  handedness: Handedness;
}) {
  const { kpts, viewType, handedness } = props;

  function chain(side: 'left' | 'right') {
    const sh = kpts[`${side}_shoulder`];
    const el = kpts[`${side}_elbow`];
    const wr = kpts[`${side}_wrist`];
    return [sh, el, wr].filter((p): p is { x: number; y: number } => p != null);
  }

  // Render LEAD chain (yellow side) + TRAIL chain (blue side) per the
  // workbench's lead/trail naming — both rendered gray so the user can
  // visually compare each WHAM arm to the matching-colored GT arm.
  const leadSide  = whamSideFor('lead',  viewType, handedness);
  const trailSide = whamSideFor('trail', viewType, handedness);
  const leadChain  = chain(leadSide);
  const trailChain = chain(trailSide);

  // Hips
  const lh = kpts['left_hip'];
  const rh = kpts['right_hip'];

  function drawChain(pts: Array<{ x: number; y: number }>, key: string) {
    if (pts.length === 0) return null;
    const polylinePts = pts.map(p => `${p.x},${p.y}`).join(' ');
    return (
      <g key={key}>
        {pts.length >= 2 && (
          <polyline
            points={polylinePts}
            fill="none"
            stroke={COLOR_WHAM}
            strokeWidth={2}
            strokeDasharray="4,3"
          />
        )}
        {pts.map((p, i) => (
          <rect
            key={i}
            x={p.x - 6} y={p.y - 6} width={12} height={12}
            fill={COLOR_WHAM}
            stroke="#000"
            strokeWidth={0.6}
          />
        ))}
      </g>
    );
  }

  return (
    <g>
      {drawChain(leadChain,  'wham-lead')}
      {drawChain(trailChain, 'wham-trail')}
      {lh && (
        <rect x={lh.x - 6} y={lh.y - 6} width={12} height={12}
              fill={COLOR_WHAM} stroke="#000" strokeWidth={0.6} />
      )}
      {rh && (
        <rect x={rh.x - 6} y={rh.y - 6} width={12} height={12}
              fill={COLOR_WHAM} stroke="#000" strokeWidth={0.6} />
      )}
    </g>
  );
}

function MediaPipeDots(props: {
  kpts: Record<string, [number | null, number | null, number]>;
}) {
  const { kpts } = props;
  // Render small X markers for arm + hip keypoints only — those are
  // the ones being validated. Other body parts skipped to reduce noise.
  const NAMES = [
    'left_shoulder', 'left_elbow', 'left_wrist',
    'right_shoulder', 'right_elbow', 'right_wrist',
    'left_hip', 'right_hip',
  ];
  const pts: Array<{ x: number; y: number; name: string }> = [];
  for (const n of NAMES) {
    const k = kpts[n];
    if (!k) continue;
    const [x, y] = k;
    if (x == null || y == null) continue;
    pts.push({ x, y, name: n });
  }
  return (
    <g>
      {pts.map((p, i) => (
        <g key={i}>
          <line x1={p.x - 4} y1={p.y - 4} x2={p.x + 4} y2={p.y + 4}
                stroke={COLOR_MP} strokeWidth={1.5} />
          <line x1={p.x + 4} y1={p.y - 4} x2={p.x - 4} y2={p.y + 4}
                stroke={COLOR_MP} strokeWidth={1.5} />
        </g>
      ))}
    </g>
  );
}

/* ════════════════════════════════════════════════════════════════════
   Styles
   ════════════════════════════════════════════════════════════════════ */

const CSS = `
  .rv-shell {
    max-width: 540px;
    margin: 0 auto;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    color: var(--text-primary);
    font-family: 'DM Sans', system-ui, sans-serif;
  }
  .rv-hud {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }
  .rv-badge {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 3px 8px;
    border: 1px solid var(--text-primary);
    border-radius: 4px;
    color: var(--text-primary);
  }
  .rv-progress { font-size: 11px; color: var(--text-muted); }

  .rv-tabs {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 4px;
  }
  .rv-tab {
    padding: 6px 4px;
    background: transparent;
    color: var(--text-muted);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 4px;
    cursor: pointer;
    font: inherit;
    font-size: 10px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
    transition: border-color 0.12s, color 0.12s;
  }
  .rv-tab:hover { border-color: var(--text-primary); color: var(--text-primary); }
  .rv-tab.on { border-color: var(--text-primary); color: var(--text-primary); }
  .rv-tab.reviewed { color: ${COLOR_REVIEWED}; }
  .rv-tab.reviewed.on { border-color: ${COLOR_REVIEWED}; }
  .rv-check { font-size: 11px; }

  .rv-frame-info {
    font-size: 11px;
    color: var(--text-muted);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }

  .rv-stage {
    background: #000;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    position: relative;
    overflow: hidden;
  }
  .rv-video, .rv-overlay {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
  }
  .rv-video { object-fit: contain; background: #000; }
  .rv-overlay { pointer-events: none; }

  .rv-toggles {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 8px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 4px;
  }
  .rv-toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: var(--text-primary);
    cursor: pointer;
  }
  .rv-toggle input { width: 14px; height: 14px; cursor: pointer; }
  .rv-swatch {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 2px;
    border: 1px solid rgba(0, 0, 0, 0.3);
  }

  .rv-verdict-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 6px;
  }
  .rv-verdict {
    padding: 10px;
    border-width: 1px;
    border-style: solid;
    border-radius: 4px;
    cursor: pointer;
    font: inherit;
    font-size: 13px;
    font-weight: 600;
    transition: background 0.12s, color 0.12s;
  }

  .rv-notes {
    background: var(--surface-card);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 4px;
    padding: 8px;
    color: var(--text-primary);
    font: inherit;
    font-size: 12px;
    resize: vertical;
    min-height: 48px;
  }
  .rv-notes:focus {
    outline: none;
    border-color: var(--text-primary);
  }

  .rv-error {
    color: var(--annot-error);
    font-size: 11px;
    border: 1px solid var(--annot-error);
    padding: 6px 8px;
    border-radius: 4px;
  }

  .rv-save {
    padding: 10px;
    background: var(--text-primary);
    color: #080c08;
    border: 1px solid var(--text-primary);
    border-radius: 4px;
    cursor: pointer;
    font: inherit;
    font-size: 13px;
    font-weight: 700;
  }
  .rv-save:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
`;
