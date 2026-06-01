'use client';

import {
  useCallback, useEffect, useMemo, useRef, useState,
  type PointerEvent as ReactPointerEvent,
} from 'react';
import { TASK_PHASES, type TaskPhase, type Handedness } from '@/lib/types/annotation';

/**
 * PR-7A.1 Phase 3 — Drag-to-correct calibration.
 *
 * The Phase 2 verdict-only flow couldn't capture continuous error.
 * This rewrite lets the reviewer drag each GT dot to the true bone-
 * top position; the new positions get saved as
 * landmark_validation_review.calibrated_keypoints. The PR-7B training
 * signal is the per-joint delta vs the original GT.
 *
 * CRITICAL INVARIANT (carries over from Phase 2):
 *   The <video> element + SVG overlay use the EXACT same setup as the
 *   workbench (commit pr-7a.1-p3-c1's letterbox fix applies to both
 *   places). Container .rv-stage: position relative, aspect-ratio =
 *   videoWidth / videoHeight, overflow hidden. Video + SVG positioned
 *   absolute, inset 0; both with object-fit: contain (video) and
 *   preserveAspectRatio="xMidYMid meet" (SVG). SVG viewBox in native
 *   pixel dims. Drag conversion uses svg.createSVGPoint() +
 *   matrixTransform(getScreenCTM().inverse()) which handles
 *   letterboxing automatically — same result the workbench's
 *   clientToNative helper produces.
 */

/* ════════════════════════════════════════════════════════════════════
   Types — shared with the server page via direct import
   ════════════════════════════════════════════════════════════════════ */

// PR-7A.2: extended from 8 to 14 joints. The new entries (head_crown,
// chin, lead/trail_knee, lead/trail_ankle) all live as nullable values
// in the same calibrated_keypoints JSONB column — additive. Existing
// 8-joint reviews load fine because absent keys simply aren't dragged.
const JOINT_KEYS = [
  'lead_shoulder', 'lead_elbow', 'lead_wrist',
  'trail_shoulder', 'trail_elbow', 'trail_wrist',
  'lead_hip', 'trail_hip',
  'head_crown', 'chin',
  'lead_knee', 'trail_knee',
  'lead_ankle', 'trail_ankle',
] as const;
export type JointKey = typeof JOINT_KEYS[number];

export type CalibratedKpts = Partial<Record<JointKey, { x: number; y: number }>>;

export type ReviewRow = {
  phase: string;
  verdict: string;
  notes: string | null;
  calibrated_keypoints?: CalibratedKpts | null;
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
  // PR-7A.2 — head + leg GT clusters. Null when the annotator skipped
  // that task for this phase; the review page renders gracefully (no
  // dot, no connection line, no crash).
  headSet: {
    headCrown: { x: number; y: number };
    chin:      { x: number; y: number };
  } | null;
  legLead: {
    knee:  { x: number; y: number };
    ankle: { x: number; y: number };
  } | null;
  legTrail: {
    knee:  { x: number; y: number };
    ankle: { x: number; y: number };
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
// PR-7A.2 color protocol: lead/trail HIP follow the same yellow/blue
// convention as the arm + leg clusters (was white in PR-7A.1 — that
// made hip side-confusion easy and didn't compose visually with the
// new leg dots). Head joints stay white (no lead/trail meaning on a
// single-skull-per-frame cluster). Pelvis_center is also white but
// smaller + semi-transparent so it reads as "derived" not "annotated".
const COLOR_GT_HIP_LEAD  = COLOR_GT_LEAD;
const COLOR_GT_HIP_TRAIL = COLOR_GT_TRAIL;
const COLOR_GT_HEAD      = '#FFFFFF';
const COLOR_GT_PELVIS    = '#FFFFFF';
const COLOR_WHAM      = '#9C9C9C';
const COLOR_MP        = '#E0E0E0';
const COLOR_REVIEWED  = '#1D9E75';
const COLOR_UNSAVED   = '#E6A23C';

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
  if (handedness === 'right') {
    return arm === 'lead' ? 'right' : 'left';
  }
  return arm === 'lead' ? 'left' : 'right';
}

/**
 * WHAM joint name for a given GT joint key. Returns null if WHAM has
 * no equivalent landmark (currently only `chin` — WHAM exposes head
 * and head_crown but no jaw/chin point). PR-7A.2 maps knee + ankle to
 * WHAM's anatomical bone-surface landmarks (`lateral_epicondyle_*`
 * and `lateral_malleolus_*`) rather than the SMPL joint centers
 * (`left_knee`, `left_ankle` etc.) — they're the same WHAM column-set
 * the annotation guide describes as the correct click target.
 */
function whamJointName(
  joint: JointKey,
  viewType: 'face_on' | 'down_the_line',
  handedness: Handedness,
): string | null {
  if (joint === 'head_crown') return 'head_crown';
  if (joint === 'chin')        return null;

  const isLead = joint.startsWith('lead_');
  const body = joint.slice(isLead ? 5 : 6);
  const side = whamSideFor(isLead ? 'lead' : 'trail', viewType, handedness);

  // Knee + ankle: prefer WHAM's anatomical bone-landmark columns over
  // the SMPL joint centers — that's what the annotation guide trained
  // the annotator to click against.
  if (body === 'knee')  return `lateral_epicondyle_${side}`;
  if (body === 'ankle') return `lateral_malleolus_${side}`;
  // shoulder / elbow / wrist / hip — WHAM uses the simple `side_body`
  // naming for these (image-orientation, post-PR-7a.2 arm-chain swap).
  return `${side}_${body}`;
}

/** Resolve the original (pre-drag) position for a joint from PerPhaseData. */
function originalPos(p: PerPhaseData, joint: JointKey): { x: number; y: number } | null {
  switch (joint) {
    case 'lead_shoulder':  return p.armLead?.shoulder  ?? null;
    case 'lead_elbow':     return p.armLead?.elbow     ?? null;
    case 'lead_wrist':     return p.armLead?.wrist     ?? null;
    case 'trail_shoulder': return p.armTrail?.shoulder ?? null;
    case 'trail_elbow':    return p.armTrail?.elbow    ?? null;
    case 'trail_wrist':    return p.armTrail?.wrist    ?? null;
    case 'lead_hip':       return p.hipPair?.leadHip   ?? null;
    case 'trail_hip':      return p.hipPair?.trailHip  ?? null;
    case 'head_crown':     return p.headSet?.headCrown ?? null;
    case 'chin':           return p.headSet?.chin      ?? null;
    case 'lead_knee':      return p.legLead?.knee      ?? null;
    case 'trail_knee':     return p.legTrail?.knee     ?? null;
    case 'lead_ankle':     return p.legLead?.ankle     ?? null;
    case 'trail_ankle':    return p.legTrail?.ankle    ?? null;
  }
}

/** Display position: staged value if dragged, else original GT. */
function displayPos(
  p: PerPhaseData,
  joint: JointKey,
  staged: CalibratedKpts | undefined,
): { x: number; y: number } | null {
  return staged?.[joint] ?? originalPos(p, joint);
}

/* ════════════════════════════════════════════════════════════════════
   Component
   ════════════════════════════════════════════════════════════════════ */

export function ReviewView(props: ReviewViewProps) {
  const {
    videoId, videoWidth, videoHeight, fpsSampled, signedUrl,
    handedness, viewType, perPhase,
  } = props;

  /* ── Stage state ─────────────────────────────────────────────── */

  const [activeIdx, setActiveIdx] = useState(0);
  const active = perPhase[activeIdx];

  // PR-7A.2: 2 GT toggles — "arms + hips" (the existing 8-joint set)
  // and "head + legs" (the new 6-joint set). Lets the reviewer focus
  // on one cluster while hiding the other if both at once is busy.
  const [showGtArmHip,  setShowGtArmHip]  = useState(true);
  const [showGtHeadLeg, setShowGtHeadLeg] = useState(true);
  const [showWham,      setShowWham]      = useState(true);
  const [showMp,        setShowMp]        = useState(false);

  const [notes, setNotes] = useState<string>(active?.existingReview?.notes ?? '');
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  /* ── Staged drag positions (per-phase, persists across tab switches) ── */

  const [stagedByPhase, setStagedByPhase] = useState<Partial<Record<TaskPhase, CalibratedKpts>>>(
    () => {
      const out: Partial<Record<TaskPhase, CalibratedKpts>> = {};
      for (const p of perPhase) {
        if (p.existingReview?.calibrated_keypoints) {
          out[p.phase] = { ...p.existingReview.calibrated_keypoints };
        }
      }
      return out;
    },
  );

  /* ── Dirty tracking — staged drags that haven't been saved yet ── */

  const [dirtyPhases, setDirtyPhases] = useState<Set<TaskPhase>>(() => new Set());

  /* ── Reviewed (saved at least once) — for tab ✓ marker ── */

  const [reviewByPhase, setReviewByPhase] = useState<Map<string, ReviewRow>>(
    () => new Map(
      perPhase.flatMap(p => p.existingReview ? [[p.phase, p.existingReview] as [string, ReviewRow]] : []),
    ),
  );

  /* ── Notes pre-fill on phase change ── */

  useEffect(() => {
    const existing = reviewByPhase.get(active?.phase ?? '');
    setNotes(existing?.notes ?? '');
    setSaveError(null);
  }, [activeIdx, active?.phase, reviewByPhase]);

  /* ── Video seek when active phase changes ── */

  const videoRef = useRef<HTMLVideoElement>(null);
  const [, setSeekedFrameIdx] = useState<number | null>(null);

  const seekToFrame = useCallback((frameIdx: number) => {
    const v = videoRef.current;
    if (!v) return;
    const t = (frameIdx + 0.5) / fpsSampled;
    try { v.currentTime = t; } catch { /* defer to onLoadedMetadata */ }
  }, [fpsSampled]);

  useEffect(() => {
    if (active) seekToFrame(active.frameIdx);
  }, [active, seekToFrame]);

  const onVideoLoadedMetadata = useCallback(() => {
    if (active) seekToFrame(active.frameIdx);
  }, [active, seekToFrame]);

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

  /* ── Drag interaction ─────────────────────────────────────────── */

  const svgRef = useRef<SVGSVGElement>(null);
  const [draggingJoint, setDraggingJoint] = useState<JointKey | null>(null);

  // Convert a pointer event's client coords to SVG viewBox (= native
  // pixel) coords. Handles object-fit:contain letterboxing automatically
  // because we're going through the browser's CTM (current transform
  // matrix), which already encodes the SVG's preserveAspectRatio logic.
  const clientToViewBox = useCallback(
    (e: ReactPointerEvent<SVGSVGElement>): { x: number; y: number } | null => {
      const svg = svgRef.current;
      if (!svg) return null;
      const pt = svg.createSVGPoint();
      pt.x = e.clientX;
      pt.y = e.clientY;
      const ctm = svg.getScreenCTM();
      if (!ctm) return null;
      const t = pt.matrixTransform(ctm.inverse());
      return {
        x: Math.max(0, Math.min(videoWidth,  t.x)),
        y: Math.max(0, Math.min(videoHeight, t.y)),
      };
    },
    [videoWidth, videoHeight],
  );

  const markDirty = useCallback((phase: TaskPhase) => {
    setDirtyPhases(prev => {
      if (prev.has(phase)) return prev;
      const next = new Set(prev);
      next.add(phase);
      return next;
    });
  }, []);

  const onDotPointerDown = useCallback(
    (e: ReactPointerEvent<SVGCircleElement>, joint: JointKey) => {
      if (!active) return;
      e.preventDefault();
      e.stopPropagation();
      try { (e.target as Element).setPointerCapture(e.pointerId); } catch { /* noop */ }
      setDraggingJoint(joint);
    },
    [active],
  );

  const onSvgPointerMove = useCallback(
    (e: ReactPointerEvent<SVGSVGElement>) => {
      if (!draggingJoint || !active) return;
      const pos = clientToViewBox(e);
      if (!pos) return;
      setStagedByPhase(prev => {
        const cur = prev[active.phase] ?? {};
        return { ...prev, [active.phase]: { ...cur, [draggingJoint]: pos } };
      });
      markDirty(active.phase);
    },
    [draggingJoint, active, clientToViewBox, markDirty],
  );

  const onSvgPointerUp = useCallback(
    (e: ReactPointerEvent<SVGSVGElement>) => {
      if (!draggingJoint) return;
      try { (e.target as Element).releasePointerCapture(e.pointerId); } catch { /* noop */ }
      setDraggingJoint(null);
    },
    [draggingJoint],
  );

  /* ── Reset current phase ─────────────────────────────────────── */

  const resetPhase = useCallback(() => {
    if (!active) return;
    setStagedByPhase(prev => {
      if (!(active.phase in prev)) return prev;
      const next = { ...prev };
      delete next[active.phase];
      return next;
    });
    setDirtyPhases(prev => {
      if (!prev.has(active.phase)) return prev;
      const next = new Set(prev);
      next.delete(active.phase);
      return next;
    });
  }, [active]);

  /* ── Save calibration ─────────────────────────────────────────── */

  const handleSave = useCallback(async () => {
    if (!active) return;
    setSaving(true);
    setSaveError(null);

    const staged = stagedByPhase[active.phase];
    const hasAnyDrag = staged && Object.keys(staged).length > 0;

    // Build full 8-joint payload when any joint was dragged on this
    // phase; staged values for dragged joints, original positions for
    // untouched ones. Joints with no original GT (e.g. user skipped
    // the hip task) are omitted — the partial object is still legal
    // per the parseCalibratedKeypoints validator.
    let calibratedKeypoints: CalibratedKpts | null = null;
    if (hasAnyDrag) {
      calibratedKeypoints = {};
      for (const joint of JOINT_KEYS) {
        const pos = displayPos(active, joint, staged);
        if (pos) calibratedKeypoints[joint] = pos;
      }
    }

    const verdict: 'correct' | 'incorrect' = hasAnyDrag ? 'incorrect' : 'correct';

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
          calibrated_keypoints: calibratedKeypoints,
        }),
      });
      if (!res.ok) {
        const body = await res.text().catch(() => '');
        setSaveError(`HTTP ${res.status} · ${body.slice(0, 200)}`);
        setSaving(false);
        return;
      }
      // Update local cache so tab marker reflects.
      const next = new Map(reviewByPhase);
      next.set(active.phase, {
        phase: active.phase,
        verdict,
        notes: notes.length > 0 ? notes : null,
        calibrated_keypoints: calibratedKeypoints,
      });
      setReviewByPhase(next);
      // Clear dirty flag for this phase (staged positions persist —
      // they're now the saved positions, no longer "unsaved").
      setDirtyPhases(prev => {
        if (!prev.has(active.phase)) return prev;
        const n = new Set(prev);
        n.delete(active.phase);
        return n;
      });
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : 'unknown');
    } finally {
      setSaving(false);
    }
  }, [active, stagedByPhase, videoId, notes, reviewByPhase]);

  const reviewedCount = useMemo(() => reviewByPhase.size, [reviewByPhase]);
  const stagedCurrent = active ? stagedByPhase[active.phase] : undefined;
  const hasStagedCurrent = !!stagedCurrent && Object.keys(stagedCurrent).length > 0;

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
          const isDirty = dirtyPhases.has(p.phase);
          return (
            <button
              key={p.phase}
              type="button"
              className={`rv-tab ${isActive ? 'on' : ''} ${isReviewed ? 'reviewed' : ''}`}
              onClick={() => setActiveIdx(i)}
            >
              <span>{p.phase}</span>
              {isDirty && <span className="rv-dirty" title="unsaved drags">●</span>}
              {isReviewed && <span className="rv-check" aria-label="reviewed">✓</span>}
            </button>
          );
        })}
      </div>

      {active && (
        <div className="rv-main">
          <p className="rv-hint">
            Drag dots to where bone-top truly is. Lead = yellow, trail = blue,
            head = white. Pelvis center is derived from hips. Gray lines show
            offsets from WHAM (dashed) and MediaPipe (lighter).
          </p>

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
              ref={svgRef}
              className={`rv-overlay ${draggingJoint ? 'dragging' : ''}`}
              viewBox={`0 0 ${videoWidth} ${videoHeight}`}
              preserveAspectRatio="xMidYMid meet"
              onPointerMove={onSvgPointerMove}
              onPointerUp={onSvgPointerUp}
              onPointerCancel={onSvgPointerUp}
            >
              {/* CONNECTION LINES — z-order below dots */}
              <ConnectionLines
                active={active}
                staged={stagedCurrent}
                viewType={viewType}
                handedness={handedness}
                showWham={showWham}
                showMp={showMp}
                showGtArmHip={showGtArmHip}
                showGtHeadLeg={showGtHeadLeg}
              />

              {/* WHAM mesh markers (non-interactive) */}
              {showWham && active.wham && (
                <WhamSkeleton kpts={active.wham} viewType={viewType} handedness={handedness} />
              )}

              {/* MediaPipe markers (non-interactive) */}
              {showMp && active.mediaPipe && (
                <MediaPipeDots kpts={active.mediaPipe} />
              )}

              {/* DERIVED pelvis_center — small white circle, NOT
                  draggable. Mid-point of (lead_hip, trail_hip) using
                  whichever positions are current (staged-drag or
                  original GT). Rendered AFTER WHAM/MP so it sits on
                  top but BEFORE GT dots so the draggable dots win. */}
              {showGtArmHip && (
                <PelvisCenter
                  active={active}
                  staged={stagedCurrent}
                />
              )}

              {/* GT DOTS — draggable, top z-order */}
              {showGtArmHip && (
                <>
                  <ArmLayer
                    arm="lead" color={COLOR_GT_LEAD}
                    active={active} staged={stagedCurrent}
                    draggingJoint={draggingJoint}
                    onDotPointerDown={onDotPointerDown}
                  />
                  <ArmLayer
                    arm="trail" color={COLOR_GT_TRAIL}
                    active={active} staged={stagedCurrent}
                    draggingJoint={draggingJoint}
                    onDotPointerDown={onDotPointerDown}
                  />
                  <HipDots
                    leadColor={COLOR_GT_HIP_LEAD}
                    trailColor={COLOR_GT_HIP_TRAIL}
                    active={active} staged={stagedCurrent}
                    draggingJoint={draggingJoint}
                    onDotPointerDown={onDotPointerDown}
                  />
                </>
              )}

              {showGtHeadLeg && (
                <>
                  <HeadDots
                    color={COLOR_GT_HEAD}
                    active={active} staged={stagedCurrent}
                    draggingJoint={draggingJoint}
                    onDotPointerDown={onDotPointerDown}
                  />
                  <LegLayer
                    arm="lead" color={COLOR_GT_LEAD}
                    active={active} staged={stagedCurrent}
                    draggingJoint={draggingJoint}
                    onDotPointerDown={onDotPointerDown}
                  />
                  <LegLayer
                    arm="trail" color={COLOR_GT_TRAIL}
                    active={active} staged={stagedCurrent}
                    draggingJoint={draggingJoint}
                    onDotPointerDown={onDotPointerDown}
                  />
                </>
              )}
            </svg>
          </div>

          {/* Source toggles — PR-7A.2: GT split into two cluster
              groups so the reviewer can focus on one without the
              other crowding the frame. */}
          <div className="rv-toggles">
            <label className="rv-toggle">
              <input type="checkbox" checked={showGtArmHip} onChange={e => setShowGtArmHip(e.target.checked)} />
              <span className="rv-swatch" style={{ background: COLOR_GT_LEAD }} />
              <span className="rv-swatch" style={{ background: COLOR_GT_TRAIL }} />
              <span>Jason GT (arms + hips)</span>
            </label>
            <label className="rv-toggle">
              <input type="checkbox" checked={showGtHeadLeg} onChange={e => setShowGtHeadLeg(e.target.checked)} />
              <span className="rv-swatch" style={{ background: COLOR_GT_HEAD }} />
              <span className="rv-swatch" style={{ background: COLOR_GT_LEAD }} />
              <span className="rv-swatch" style={{ background: COLOR_GT_TRAIL }} />
              <span>Jason GT (head + legs)</span>
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

          <textarea
            className="rv-notes"
            placeholder="Optional notes (e.g. 'wrist drifted 2cm left of bone')"
            value={notes}
            onChange={e => setNotes(e.target.value)}
            rows={2}
          />

          {saveError && <div className="rv-error">save failed: {saveError}</div>}

          <div className="rv-save-row">
            <button
              type="button"
              className="rv-reset-link"
              onClick={resetPhase}
              disabled={!hasStagedCurrent}
            >
              Reset this phase
            </button>
            <button
              type="button"
              className="rv-save"
              disabled={saving}
              onClick={handleSave}
            >
              {saving ? 'Saving…' : 'Save calibration'}
            </button>
          </div>
        </div>
      )}

      <style>{CSS}</style>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════
   Connection lines (GT ↔ WHAM, GT ↔ MediaPipe)
   ════════════════════════════════════════════════════════════════════ */

// Cluster lookups for the connection-line filter — must match the two
// GT toggles so the lines hide together with their endpoints.
const ARM_HIP_JOINTS: readonly JointKey[] = [
  'lead_shoulder', 'lead_elbow', 'lead_wrist',
  'trail_shoulder', 'trail_elbow', 'trail_wrist',
  'lead_hip', 'trail_hip',
];
const HEAD_LEG_JOINTS: readonly JointKey[] = [
  'head_crown', 'chin',
  'lead_knee', 'trail_knee',
  'lead_ankle', 'trail_ankle',
];

/**
 * MediaPipe COCO landmark name for a given GT joint. Returns null
 * when there's no useful equivalent (chin, head_crown — COCO has only
 * `nose`, no jaw or skull landmark). Knee + ankle COCO names match
 * the WHAM image-side convention via whamSideFor.
 */
function mpJointName(
  joint: JointKey,
  viewType: 'face_on' | 'down_the_line',
  handedness: Handedness,
): string | null {
  if (joint === 'head_crown' || joint === 'chin') return null;
  const isLead = joint.startsWith('lead_');
  const body = joint.slice(isLead ? 5 : 6);
  const side = whamSideFor(isLead ? 'lead' : 'trail', viewType, handedness);
  // shoulder/elbow/wrist/hip/knee/ankle all use the COCO `side_body`
  // naming; nothing special for the PR-7A.2 additions.
  return `${side}_${body}`;
}

function ConnectionLines(props: {
  active: PerPhaseData;
  staged: CalibratedKpts | undefined;
  viewType: 'face_on' | 'down_the_line';
  handedness: Handedness;
  showWham: boolean;
  showMp: boolean;
  showGtArmHip: boolean;
  showGtHeadLeg: boolean;
}) {
  const { active, staged, viewType, handedness, showWham, showMp,
          showGtArmHip, showGtHeadLeg } = props;

  const lines: React.ReactElement[] = [];

  for (const joint of JOINT_KEYS) {
    const inArmHip  = (ARM_HIP_JOINTS  as readonly string[]).includes(joint);
    const inHeadLeg = (HEAD_LEG_JOINTS as readonly string[]).includes(joint);
    if (inArmHip  && !showGtArmHip)  continue;
    if (inHeadLeg && !showGtHeadLeg) continue;

    const gtPos = displayPos(active, joint, staged);
    if (!gtPos) continue;

    const whamName = whamJointName(joint, viewType, handedness);
    if (showWham && active.wham && whamName) {
      const w = active.wham[whamName];
      if (w) {
        lines.push(
          <line
            key={`w-${joint}`}
            x1={gtPos.x} y1={gtPos.y} x2={w.x} y2={w.y}
            stroke={COLOR_WHAM} strokeWidth={1}
            strokeDasharray="3,2" opacity={0.4}
          />,
        );
      }
    }
    const mpName = mpJointName(joint, viewType, handedness);
    if (showMp && active.mediaPipe && mpName) {
      const mp = active.mediaPipe[mpName];
      if (mp) {
        const [mx, my] = mp;
        if (mx != null && my != null) {
          lines.push(
            <line
              key={`m-${joint}`}
              x1={gtPos.x} y1={gtPos.y} x2={mx} y2={my}
              stroke={COLOR_MP} strokeWidth={1}
              strokeDasharray="2,3" opacity={0.3}
            />,
          );
        }
      }
    }
  }

  return <g>{lines}</g>;
}

/* ════════════════════════════════════════════════════════════════════
   GT overlay primitives — draggable
   ════════════════════════════════════════════════════════════════════ */

function ArmLayer(props: {
  arm: 'lead' | 'trail';
  color: string;
  active: PerPhaseData;
  staged: CalibratedKpts | undefined;
  draggingJoint: JointKey | null;
  onDotPointerDown: (e: ReactPointerEvent<SVGCircleElement>, j: JointKey) => void;
}) {
  const { arm, color, active, staged, draggingJoint, onDotPointerDown } = props;
  const joints: JointKey[] = arm === 'lead'
    ? ['lead_shoulder', 'lead_elbow', 'lead_wrist']
    : ['trail_shoulder', 'trail_elbow', 'trail_wrist'];

  const points = joints
    .map(j => ({ joint: j, pos: displayPos(active, j, staged) }))
    .filter((e): e is { joint: JointKey; pos: { x: number; y: number } } => e.pos !== null);

  const polylinePts = points.map(p => `${p.pos.x},${p.pos.y}`).join(' ');

  return (
    <g>
      {points.length >= 2 && (
        <polyline
          points={polylinePts}
          fill="none"
          stroke={color}
          strokeWidth={3}
          strokeLinejoin="round"
          strokeLinecap="round"
          style={{ pointerEvents: 'none' }}
        />
      )}
      {points.map(({ joint, pos }) => (
        <DraggableDot
          key={joint}
          joint={joint}
          x={pos.x} y={pos.y}
          color={color}
          isDragging={draggingJoint === joint}
          onPointerDown={onDotPointerDown}
        />
      ))}
    </g>
  );
}

function HipDots(props: {
  // PR-7A.2: per-side colors (yellow lead / blue trail) so the hip
  // dots share the visual language of the arm + leg clusters.
  leadColor: string;
  trailColor: string;
  active: PerPhaseData;
  staged: CalibratedKpts | undefined;
  draggingJoint: JointKey | null;
  onDotPointerDown: (e: ReactPointerEvent<SVGCircleElement>, j: JointKey) => void;
}) {
  const { leadColor, trailColor, active, staged, draggingJoint, onDotPointerDown } = props;
  const lead  = displayPos(active, 'lead_hip',  staged);
  const trail = displayPos(active, 'trail_hip', staged);
  return (
    <g>
      {lead && (
        <DraggableDot
          joint="lead_hip" x={lead.x} y={lead.y} color={leadColor}
          isDragging={draggingJoint === 'lead_hip'}
          onPointerDown={onDotPointerDown}
        />
      )}
      {trail && (
        <DraggableDot
          joint="trail_hip" x={trail.x} y={trail.y} color={trailColor}
          isDragging={draggingJoint === 'trail_hip'}
          onPointerDown={onDotPointerDown}
        />
      )}
    </g>
  );
}

/**
 * PR-7A.2: derived pelvis-center (white, NOT draggable, smaller radius
 * + lower opacity than annotated joints). Mid-point of (lead_hip,
 * trail_hip). Re-computes on every render so it tracks the staged
 * drag positions in real time.
 */
function PelvisCenter(props: {
  active: PerPhaseData;
  staged: CalibratedKpts | undefined;
}) {
  const { active, staged } = props;
  const lead  = displayPos(active, 'lead_hip',  staged);
  const trail = displayPos(active, 'trail_hip', staged);
  if (!lead || !trail) return null;
  const cx = (lead.x + trail.x) / 2;
  const cy = (lead.y + trail.y) / 2;
  return (
    <g style={{ pointerEvents: 'none' }}>
      <circle
        cx={cx} cy={cy} r={5}
        fill={COLOR_GT_PELVIS}
        stroke="#000"
        strokeWidth={1}
        opacity={0.7}
      />
    </g>
  );
}

function HeadDots(props: {
  color: string;
  active: PerPhaseData;
  staged: CalibratedKpts | undefined;
  draggingJoint: JointKey | null;
  onDotPointerDown: (e: ReactPointerEvent<SVGCircleElement>, j: JointKey) => void;
}) {
  const { color, active, staged, draggingJoint, onDotPointerDown } = props;
  const crown = displayPos(active, 'head_crown', staged);
  const chin  = displayPos(active, 'chin',       staged);
  return (
    <g>
      {crown && (
        <DraggableDot
          joint="head_crown" x={crown.x} y={crown.y} color={color}
          isDragging={draggingJoint === 'head_crown'}
          onPointerDown={onDotPointerDown}
        />
      )}
      {chin && (
        <DraggableDot
          joint="chin" x={chin.x} y={chin.y} color={color}
          isDragging={draggingJoint === 'chin'}
          onPointerDown={onDotPointerDown}
        />
      )}
    </g>
  );
}

function LegLayer(props: {
  arm: 'lead' | 'trail';
  color: string;
  active: PerPhaseData;
  staged: CalibratedKpts | undefined;
  draggingJoint: JointKey | null;
  onDotPointerDown: (e: ReactPointerEvent<SVGCircleElement>, j: JointKey) => void;
}) {
  const { arm, color, active, staged, draggingJoint, onDotPointerDown } = props;
  const kneeKey:  JointKey = arm === 'lead' ? 'lead_knee'  : 'trail_knee';
  const ankleKey: JointKey = arm === 'lead' ? 'lead_ankle' : 'trail_ankle';
  const knee  = displayPos(active, kneeKey,  staged);
  const ankle = displayPos(active, ankleKey, staged);
  return (
    <g>
      {knee && ankle && (
        <polyline
          points={`${knee.x},${knee.y} ${ankle.x},${ankle.y}`}
          fill="none"
          stroke={color}
          strokeWidth={3}
          strokeLinejoin="round"
          strokeLinecap="round"
          style={{ pointerEvents: 'none' }}
        />
      )}
      {knee && (
        <DraggableDot
          joint={kneeKey} x={knee.x} y={knee.y} color={color}
          isDragging={draggingJoint === kneeKey}
          onPointerDown={onDotPointerDown}
        />
      )}
      {ankle && (
        <DraggableDot
          joint={ankleKey} x={ankle.x} y={ankle.y} color={color}
          isDragging={draggingJoint === ankleKey}
          onPointerDown={onDotPointerDown}
        />
      )}
    </g>
  );
}

function DraggableDot(props: {
  joint: JointKey;
  x: number; y: number;
  color: string;
  isDragging: boolean;
  onPointerDown: (e: ReactPointerEvent<SVGCircleElement>, j: JointKey) => void;
}) {
  const { joint, x, y, color, isDragging, onPointerDown } = props;
  return (
    <g>
      {/* Interactive hint ring — slightly larger, faint white, helps
          the user see the dot is draggable. */}
      <circle
        cx={x} cy={y} r={10}
        fill="none"
        stroke="#FFFFFF"
        strokeWidth={1}
        opacity={0.3}
        style={{ pointerEvents: 'none' }}
      />
      {/* The actual draggable hit area + visible dot. */}
      <circle
        cx={x} cy={y} r={8}
        fill={color}
        stroke="#000"
        strokeWidth={1.5}
        style={{
          pointerEvents: 'auto',
          cursor: isDragging ? 'grabbing' : 'grab',
          touchAction: 'none',
        }}
        onPointerDown={e => onPointerDown(e, joint)}
      />
    </g>
  );
}

/* ════════════════════════════════════════════════════════════════════
   WHAM + MediaPipe overlays (non-interactive)
   ════════════════════════════════════════════════════════════════════ */

function WhamSkeleton(props: {
  kpts: Record<string, { x: number; y: number } | null>;
  viewType: 'face_on' | 'down_the_line';
  handedness: Handedness;
}) {
  const { kpts, viewType, handedness } = props;

  function chain(...names: string[]) {
    return names
      .map(n => kpts[n])
      .filter((p): p is { x: number; y: number } => p != null);
  }

  const leadSide  = whamSideFor('lead',  viewType, handedness);
  const trailSide = whamSideFor('trail', viewType, handedness);
  const leadArm   = chain(`${leadSide}_shoulder`,  `${leadSide}_elbow`,  `${leadSide}_wrist`);
  const trailArm  = chain(`${trailSide}_shoulder`, `${trailSide}_elbow`, `${trailSide}_wrist`);
  // PR-7A.2: WHAM exposes anatomical bone-surface landmarks
  // (lateral_epicondyle / lateral_malleolus) — prefer those over the
  // SMPL joint centers so the WHAM marker lands where the annotation
  // guide instructed the annotator to click.
  const leadLeg   = chain(`lateral_epicondyle_${leadSide}`,  `lateral_malleolus_${leadSide}`);
  const trailLeg  = chain(`lateral_epicondyle_${trailSide}`, `lateral_malleolus_${trailSide}`);

  const lh = kpts['left_hip'];
  const rh = kpts['right_hip'];
  const headCrown = kpts['head_crown'];

  function drawChain(pts: Array<{ x: number; y: number }>, key: string) {
    if (pts.length === 0) return null;
    const polylinePts = pts.map(p => `${p.x},${p.y}`).join(' ');
    return (
      <g key={key} style={{ pointerEvents: 'none' }}>
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
    <g style={{ pointerEvents: 'none' }}>
      {drawChain(leadArm,   'wham-lead-arm')}
      {drawChain(trailArm,  'wham-trail-arm')}
      {drawChain(leadLeg,   'wham-lead-leg')}
      {drawChain(trailLeg,  'wham-trail-leg')}
      {lh && (
        <rect x={lh.x - 6} y={lh.y - 6} width={12} height={12}
              fill={COLOR_WHAM} stroke="#000" strokeWidth={0.6} />
      )}
      {rh && (
        <rect x={rh.x - 6} y={rh.y - 6} width={12} height={12}
              fill={COLOR_WHAM} stroke="#000" strokeWidth={0.6} />
      )}
      {headCrown && (
        <rect x={headCrown.x - 6} y={headCrown.y - 6} width={12} height={12}
              fill={COLOR_WHAM} stroke="#000" strokeWidth={0.6} />
      )}
    </g>
  );
}

function MediaPipeDots(props: {
  kpts: Record<string, [number | null, number | null, number]>;
}) {
  const { kpts } = props;
  // PR-7A.2: include knee + ankle so the new leg cluster has MP
  // markers to compare against. MP COCO has no head_crown or chin so
  // the head cluster only compares against WHAM.
  const NAMES = [
    'left_shoulder', 'left_elbow', 'left_wrist',
    'right_shoulder', 'right_elbow', 'right_wrist',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
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
    <g style={{ pointerEvents: 'none' }}>
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
  .rv-dirty { color: ${COLOR_UNSAVED}; font-size: 13px; line-height: 1; }

  .rv-hint {
    font-size: 11px;
    color: var(--text-muted);
    margin: 0;
    line-height: 1.4;
    padding: 6px 8px;
    border-left: 2px solid rgba(255, 255, 255, 0.15);
  }

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
  /* SVG pointer-events:none lets clicks pass through to non-interactive
     elements. Draggable GT dots override this with pointer-events:auto
     on the inner circle (see DraggableDot). */
  .rv-overlay { pointer-events: none; touch-action: none; }
  .rv-overlay.dragging { cursor: grabbing; }

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

  .rv-save-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }
  .rv-reset-link {
    background: transparent;
    border: none;
    color: var(--text-muted);
    font: inherit;
    font-size: 12px;
    cursor: pointer;
    padding: 6px 4px;
    text-decoration: underline;
  }
  .rv-reset-link:hover:not(:disabled) { color: var(--text-primary); }
  .rv-reset-link:disabled {
    opacity: 0.4;
    cursor: not-allowed;
    text-decoration: none;
  }
  .rv-save {
    padding: 10px 18px;
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
