'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  APP_VERSION,
  type AnnotationArm,
  type AnnotationRecord,
  type AnnotationTask,
  type AnnotationVisibility,
  type Handedness,
  type VideoMetaForAnnotation,
} from '@/lib/types/annotation';

/* ════════════════════════════════════════════════════════════════════
   Internal landmark annotation workbench (PR-7A).

   Guided, one-click-at-a-time capture of golf-specific keypoints
   (shoulder, elbow, wrist) for 4 phases × 2 arms = 8 tasks per video.

   Stages:
     loading        → fetch meta + existing annotations
     error          → terminal display
     handedness     → ask right / left (defaults right)
     phase_confirm  → derived phase frames look right? confirm / recalibrate
     phase_calibrate → manual scrub-and-mark for 4 phases in turn
     annotating     → 8-task linear flow with click → save → advance
     done           → all tasks complete, offer export + back link
   ════════════════════════════════════════════════════════════════════ */

type Stage =
  | 'loading' | 'error'
  | 'handedness'
  | 'phase_confirm' | 'phase_calibrate'
  | 'annotating'
  | 'done';

// Tier 2 phase set (PR-7A tier-2 expansion 2026-05-29).
// PHASE_ORDER drives both task generation order (phase-major, then arm)
// and the manual phase-calibration sequence. 7 phases × 2 arms = 14 tasks.
type PhaseKey =
  | 'setup' | 'takeaway' | 'top' | 'transition'
  | 'impact' | 'post_impact' | 'finish';
const PHASE_ORDER: PhaseKey[] = [
  'setup', 'takeaway', 'top', 'transition',
  'impact', 'post_impact', 'finish',
];

type PhaseFrames = {
  setup:       number;
  takeaway:    number;
  top:         number;
  transition:  number;
  impact:      number;
  post_impact: number;
  finish:      number;
};
type Point = { x: number; y: number };
type PointStepIdx = 0 | 1 | 2 | 3;
const POINT_LABELS = ['Shoulder', 'Elbow', 'Wrist'] as const;

// Arm-side accent colors — locked-in hex to skip the CSS-var resolution
// dance inside canvas draws (avoids initial-paint flash + getComputedStyle
// per redraw). Single source of truth still lives in globals.css; these
// constants mirror the --annot-* tokens. Update both together.
const COLOR_LEAD   = '#FFD86B';
const COLOR_TRAIL  = '#4FB3FF';

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function armColor(arm: AnnotationArm): string {
  return arm === 'lead' ? COLOR_LEAD : COLOR_TRAIL;
}

interface Props { videoId: string; }

/* ════════════════════════════════════════════════════════════════════
   Task generation + completion checks
   ════════════════════════════════════════════════════════════════════ */

function generateTasks(pf: PhaseFrames): AnnotationTask[] {
  const arms: AnnotationArm[] = ['lead', 'trail'];
  const out: AnnotationTask[] = [];
  let idx = 0;
  for (const phase of PHASE_ORDER) {
    for (const arm of arms) {
      out.push({ index: idx++, phase, arm, frameIdx: pf[phase] });
    }
  }
  return out;
}

function isTaskDone(t: AnnotationTask, existing: AnnotationRecord[]): boolean {
  return existing.some(a =>
    a.frame_idx === t.frameIdx &&
    a.arm === t.arm &&
    a.task_type === 'manual_gt',
  );
}

function findFirstUnannotatedIndex(
  tasks: AnnotationTask[],
  existing: AnnotationRecord[],
): number {
  for (const t of tasks) if (!isTaskDone(t, existing)) return t.index;
  return tasks.length;
}

/**
 * Linear scan from `fromIdx` forward, returning the first task that isn't
 * yet in `existing`. Used by save / skip advance so the workbench jumps
 * over previously-completed tasks instead of forcing a manual Skip click
 * on each one (matters when re-entering a partially-annotated video).
 * Returns `tasks.length` when every remaining task is done — caller treats
 * that as "all done, show completion screen".
 */
function findNextUnannotatedIndex(
  tasks: AnnotationTask[],
  existing: AnnotationRecord[],
  fromIdx: number,
): number {
  for (let i = fromIdx; i < tasks.length; i++) {
    if (!isTaskDone(tasks[i], existing)) return i;
  }
  return tasks.length;
}

function timeToFrame(timeSec: number, fps: number): number {
  return Math.max(0, Math.round(timeSec * fps));
}

function derivePhaseFrames(
  meta: VideoMetaForAnnotation,
): PhaseFrames | null {
  const pm = meta.phaseMarkers;
  // Hard requirement: the 4 "primary" markers must exist. Without any of
  // them we can't derive the Tier-2 phases either, so route to manual
  // calibration.
  if (
    pm.setupTime == null || pm.topTime == null ||
    pm.impactTime == null || pm.finishTime == null
  ) return null;

  // transitionTime is the only marker the auto-deriver can fall back on
  // (top + 0.4*(impact-top) is a defensible default when it's missing).
  const transitionTime = pm.transitionTime ??
    (pm.topTime + 0.4 * (pm.impactTime - pm.topTime));

  return {
    setup:       timeToFrame(pm.setupTime, meta.fps),
    takeaway:    timeToFrame(pm.setupTime  + 0.25 * (pm.topTime    - pm.setupTime),  meta.fps),
    top:         timeToFrame(pm.topTime,    meta.fps),
    transition:  timeToFrame(transitionTime, meta.fps),
    impact:      timeToFrame(pm.impactTime, meta.fps),
    post_impact: timeToFrame(pm.impactTime + 0.4  * (pm.finishTime - pm.impactTime), meta.fps),
    finish:      timeToFrame(pm.finishTime, meta.fps),
  };
}

/* ════════════════════════════════════════════════════════════════════
   Component
   ════════════════════════════════════════════════════════════════════ */

export function AnnotateWorkbench({ videoId }: Props) {
  const router = useRouter();

  const [stage, setStage] = useState<Stage>('loading');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [videoMeta, setVideoMeta] = useState<VideoMetaForAnnotation | null>(null);
  const [existingAnns, setExistingAnns] = useState<AnnotationRecord[]>([]);

  const [handedness, setHandedness] = useState<Handedness>('right');

  const [phaseFrames, setPhaseFrames] = useState<PhaseFrames | null>(null);
  const [calibPhaseTarget, setCalibPhaseTarget] = useState<PhaseKey | null>(null);
  const [calibCaptured, setCalibCaptured] = useState<Partial<PhaseFrames>>({});

  const [tasks, setTasks] = useState<AnnotationTask[]>([]);
  const [currentTaskIdx, setCurrentTaskIdx] = useState(0);

  // Per-task interaction state. Visibility is determined at save-time
  // (clear = auto-saved when the third point lands; occluded/uncertain
  // = explicit button or keyboard), so it doesn't need its own state.
  const [activePoints, setActivePoints] = useState<(Point | null)[]>([null, null, null]);
  const [pointStepIdx, setPointStepIdx] = useState<PointStepIdx>(0);

  const [currentFrameIdx, setCurrentFrameIdx] = useState(0);
  const [savingError, setSavingError] = useState<string | null>(null);

  const videoRef  = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const currentTask: AnnotationTask | null =
    stage === 'annotating' && tasks[currentTaskIdx] ? tasks[currentTaskIdx] : null;

  /* ──────────────────────────────────────────────────────────────
     LOAD: video meta + existing annotations
     ────────────────────────────────────────────────────────────── */

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [metaRes, annRes] = await Promise.all([
          fetch(`/api/admin/videos/${videoId}`, { cache: 'no-store' }),
          fetch(
            `/api/admin/annotations/${videoId}?taskType=manual_gt`,
            { cache: 'no-store' },
          ),
        ]);
        if (cancelled) return;
        if (!metaRes.ok) {
          setErrorMsg(`video meta: HTTP ${metaRes.status}`);
          setStage('error');
          return;
        }
        if (!annRes.ok) {
          setErrorMsg(`annotations: HTTP ${annRes.status}`);
          setStage('error');
          return;
        }
        const meta = (await metaRes.json()) as VideoMetaForAnnotation;
        const annBody = (await annRes.json()) as { annotations: AnnotationRecord[] };
        if (cancelled) return;
        if (!meta.width || !meta.height || !meta.fps) {
          setErrorMsg(
            `video dimensions missing (w=${meta.width} h=${meta.height} fps=${meta.fps})`,
          );
          setStage('error');
          return;
        }
        setVideoMeta(meta);
        setExistingAnns(annBody.annotations ?? []);
        setStage('handedness');
      } catch (e) {
        if (cancelled) return;
        setErrorMsg(e instanceof Error ? e.message : 'unknown load error');
        setStage('error');
      }
    })();
    return () => { cancelled = true; };
  }, [videoId]);

  /* ──────────────────────────────────────────────────────────────
     Stage transition helpers
     ────────────────────────────────────────────────────────────── */

  const confirmHandedness = useCallback(() => {
    if (!videoMeta) return;
    const derived = derivePhaseFrames(videoMeta);
    if (derived) {
      setPhaseFrames(derived);
      setStage('phase_confirm');
    } else {
      setCalibPhaseTarget('setup');
      setCalibCaptured({});
      setStage('phase_calibrate');
    }
  }, [videoMeta]);

  const enterCalibrate = useCallback(() => {
    setPhaseFrames(null);
    setCalibPhaseTarget('setup');
    setCalibCaptured({});
    setStage('phase_calibrate');
  }, []);

  const startAnnotating = useCallback((pf: PhaseFrames) => {
    const generated = generateTasks(pf);
    const firstIdx = findFirstUnannotatedIndex(generated, existingAnns);
    setPhaseFrames(pf);
    setTasks(generated);
    if (firstIdx >= generated.length) {
      setStage('done');
      return;
    }
    setCurrentTaskIdx(firstIdx);
    setActivePoints([null, null, null]);
    setPointStepIdx(0);
    setStage('annotating');
  }, [existingAnns]);

  /* ──────────────────────────────────────────────────────────────
     Frame seeking
     ────────────────────────────────────────────────────────────── */

  // Pure DOM: just sets video.currentTime. The 'seeked' event handler
  // below derives currentFrameIdx from the video's actual currentTime —
  // making the video element the single source of truth and keeping
  // this function side-effect-free wrt React state (so calling it from
  // an effect doesn't trip react-hooks/set-state-in-effect).
  const seekToFrame = useCallback((frame: number) => {
    if (!videoMeta || !videoRef.current) return;
    const fc = videoMeta.frameCount > 0
      ? videoMeta.frameCount
      : Math.max(1, Math.round(videoMeta.durationSec * videoMeta.fps));
    const f = Math.max(0, Math.min(fc - 1, Math.round(frame)));
    const t = (f + 0.5) / videoMeta.fps;
    try {
      videoRef.current.currentTime = t;
    } catch {
      // Seek can throw before metadata loads — defer to onLoadedMetadata.
    }
  }, [videoMeta]);

  // Whenever the active task changes, seek the video to that task's frame.
  // For phase_calibrate, the user drives the seek via buttons / arrow keys
  // so there's nothing to do reactively here.
  useEffect(() => {
    if (stage === 'annotating' && currentTask) {
      seekToFrame(currentTask.frameIdx);
    }
  }, [stage, currentTask, seekToFrame]);

  // When the video element first mounts and meta loads, seek to whatever
  // frame the current stage expects.
  const onVideoLoadedMetadata = useCallback(() => {
    if (stage === 'annotating' && currentTask) {
      seekToFrame(currentTask.frameIdx);
    } else if (stage === 'phase_calibrate') {
      seekToFrame(currentFrameIdx);
    }
  }, [stage, currentTask, currentFrameIdx, seekToFrame]);

  /* ──────────────────────────────────────────────────────────────
     Canvas draw
     ────────────────────────────────────────────────────────────── */

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !videoMeta) return;
    if (canvas.width !== videoMeta.width)   canvas.width  = videoMeta.width;
    if (canvas.height !== videoMeta.height) canvas.height = videoMeta.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Saved annotations on this frame — drawn arm-coloured but faint so
    // they read as "already done, just here for context". 3px filled
    // dot, no outer ring, thin 1px connector.
    const savedForFrame = existingAnns.filter(a =>
      a.frame_idx === currentFrameIdx && a.task_type === 'manual_gt',
    );
    for (const a of savedForFrame) {
      // PR-7A.1: arm is now nullable on AnnotationRecord (hip rows
      // carry arm=NULL). The arm-coord branch only handles rows whose
      // arm is set; hip rows are filtered above by task_type, but
      // narrow defensively here so future task types can't trip the
      // armColor() call.
      if (a.arm == null) continue;
      const pts: Point[] = [];
      if (a.shoulder_x != null && a.shoulder_y != null) pts.push({ x: a.shoulder_x, y: a.shoulder_y });
      if (a.elbow_x    != null && a.elbow_y    != null) pts.push({ x: a.elbow_x,    y: a.elbow_y    });
      if (a.wrist_x    != null && a.wrist_y    != null) pts.push({ x: a.wrist_x,    y: a.wrist_y    });
      const base = armColor(a.arm);
      drawPolyline(ctx, pts, hexToRgba(base, 0.25), 1);
      for (const p of pts) drawDot(ctx, p, 3, hexToRgba(base, 0.35));
    }

    // Active points — the arm currently being annotated. Full-saturation
    // arm colour, 4px filled dot + 8px outer ring, 1.5px connector. Dot
    // is small enough that the underlying joint stays visible through it.
    const activeArm: AnnotationArm = currentTask?.arm ?? 'lead';
    const activeBase = armColor(activeArm);
    const activeFiltered: Point[] = activePoints.filter((p): p is Point => p !== null);
    if (activeFiltered.length > 1) {
      drawPolyline(ctx, activeFiltered, hexToRgba(activeBase, 0.6), 1.5);
    }
    for (const p of activeFiltered) {
      drawDot(ctx, p, 4, activeBase, 8, hexToRgba(activeBase, 0.4));
    }
  }, [videoMeta, existingAnns, currentFrameIdx, activePoints, currentTask]);

  // Redraw whenever inputs change.
  useEffect(() => { drawCanvas(); }, [drawCanvas]);

  // 'seeked' fires after the video reaches the requested frame. This is
  // where currentFrameIdx becomes truth — derived from the video, not
  // assumed from the request — so the React UI never drifts out of
  // sync with the displayed frame. drawCanvas re-runs automatically
  // via its own dep on currentFrameIdx; no need to call it here.
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !videoMeta) return;
    const handler = () => {
      const f = Math.max(0, Math.round(v.currentTime * videoMeta.fps - 0.5));
      setCurrentFrameIdx(f);
    };
    v.addEventListener('seeked', handler);
    return () => v.removeEventListener('seeked', handler);
  }, [videoMeta]);

  /* ──────────────────────────────────────────────────────────────
     Click handling
     ────────────────────────────────────────────────────────────── */

  const saveAndAdvance = useCallback(async (
    points: (Point | null)[],
    vis: AnnotationVisibility,
  ) => {
    if (!videoMeta || !currentTask) return;
    setSavingError(null);
    const body: AnnotationRecord = {
      video_id: videoMeta.videoId,
      frame_idx: currentTask.frameIdx,
      phase: currentTask.phase,
      task_type: 'manual_gt',
      arm: currentTask.arm,
      visibility: vis,
      shoulder_x: points[0]?.x ?? null,
      shoulder_y: points[0]?.y ?? null,
      elbow_x:    points[1]?.x ?? null,
      elbow_y:    points[1]?.y ?? null,
      wrist_x:    points[2]?.x ?? null,
      wrist_y:    points[2]?.y ?? null,
      handedness,
      source_app_version: APP_VERSION,
    };
    try {
      const res = await fetch('/api/admin/annotations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const detail = await res.text().catch(() => '');
        setSavingError(`save failed: HTTP ${res.status} ${detail.slice(0, 200)}`);
        return;
      }
      // Replace any prior record for this (frame, arm, task_type) tuple
      // in local state so the canvas reflects the just-saved keypoints
      // when the next task lands on the same frame. We also need the
      // post-save annotation list locally for findNextUnannotatedIndex
      // below — React state update is async, so the closure's
      // existingAnns is still pre-save.
      const updatedAnns = [
        ...existingAnns.filter(a => !(
          a.frame_idx === body.frame_idx &&
          a.arm === body.arm &&
          a.task_type === body.task_type
        )),
        body,
      ];
      setExistingAnns(updatedAnns);

      const next = findNextUnannotatedIndex(tasks, updatedAnns, currentTaskIdx + 1);
      if (next >= tasks.length) {
        setStage('done');
        return;
      }
      setCurrentTaskIdx(next);
      setActivePoints([null, null, null]);
      setPointStepIdx(0);
    } catch (e) {
      setSavingError(e instanceof Error ? e.message : 'unknown save error');
    }
  }, [videoMeta, currentTask, currentTaskIdx, tasks, existingAnns, handedness]);

  const advanceWithoutSaving = useCallback(() => {
    const next = findNextUnannotatedIndex(tasks, existingAnns, currentTaskIdx + 1);
    if (next >= tasks.length) {
      setStage('done');
      return;
    }
    setCurrentTaskIdx(next);
    setActivePoints([null, null, null]);
    setPointStepIdx(0);
  }, [currentTaskIdx, tasks, existingAnns]);

  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (stage !== 'annotating' || !videoMeta) return;
    if (pointStepIdx >= 3) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const sx = videoMeta.width  / rect.width;
    const sy = videoMeta.height / rect.height;
    const x = Math.round((e.clientX - rect.left) * sx);
    const y = Math.round((e.clientY - rect.top)  * sy);

    const next = [...activePoints];
    next[pointStepIdx] = { x, y };
    setActivePoints(next);
    const newStep = (pointStepIdx + 1) as PointStepIdx;
    setPointStepIdx(newStep);

    if (newStep === 3) {
      void saveAndAdvance(next, 'clear');
    }
  }, [stage, videoMeta, pointStepIdx, activePoints, saveAndAdvance]);

  /* ──────────────────────────────────────────────────────────────
     Action buttons (also wired to keyboard)
     ────────────────────────────────────────────────────────────── */

  const undoLastPoint = useCallback(() => {
    if (stage !== 'annotating' || pointStepIdx === 0) return;
    const newStep = (pointStepIdx - 1) as PointStepIdx;
    const next = [...activePoints];
    next[newStep] = null;
    setActivePoints(next);
    setPointStepIdx(newStep);
  }, [stage, pointStepIdx, activePoints]);

  const saveAsOccluded = useCallback(() => {
    if (stage !== 'annotating') return;
    void saveAndAdvance(activePoints, 'occluded');
  }, [stage, activePoints, saveAndAdvance]);

  const saveAsUncertain = useCallback(() => {
    if (stage !== 'annotating' || pointStepIdx !== 3) return;
    void saveAndAdvance(activePoints, 'uncertain');
  }, [stage, pointStepIdx, activePoints, saveAndAdvance]);

  /* ──────────────────────────────────────────────────────────────
     Keyboard
     ────────────────────────────────────────────────────────────── */

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (stage === 'annotating') {
        switch (e.key.toLowerCase()) {
          case 'z': e.preventDefault(); undoLastPoint(); break;
          case 'o': e.preventDefault(); saveAsOccluded(); break;
          case 'u': e.preventDefault(); saveAsUncertain(); break;
          case 's':
          case ' ':
            e.preventDefault();
            advanceWithoutSaving();
            break;
        }
      } else if (stage === 'phase_calibrate') {
        if (e.key === 'ArrowLeft')  { e.preventDefault(); seekToFrame(currentFrameIdx - 1); }
        if (e.key === 'ArrowRight') { e.preventDefault(); seekToFrame(currentFrameIdx + 1); }
        if (e.key === 'ArrowUp')    { e.preventDefault(); seekToFrame(currentFrameIdx + 10); }
        if (e.key === 'ArrowDown')  { e.preventDefault(); seekToFrame(currentFrameIdx - 10); }
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [
    stage,
    undoLastPoint, saveAsOccluded, saveAsUncertain, advanceWithoutSaving,
    currentFrameIdx, seekToFrame,
  ]);

  /* ──────────────────────────────────────────────────────────────
     Phase calibrate: "this is X" capture
     ────────────────────────────────────────────────────────────── */

  const captureCurrentAsPhase = useCallback((phase: PhaseKey) => {
    const next: Partial<PhaseFrames> = {
      ...calibCaptured,
      [phase]: currentFrameIdx,
    };
    setCalibCaptured(next);
    const remaining = PHASE_ORDER.find(p => next[p] === undefined);
    if (remaining) {
      setCalibPhaseTarget(remaining);
    } else {
      // All 4 captured.
      startAnnotating(next as PhaseFrames);
    }
  }, [calibCaptured, currentFrameIdx, startAnnotating]);

  /* ──────────────────────────────────────────────────────────────
     Render helpers
     ────────────────────────────────────────────────────────────── */

  const completedCount = useMemo(() => {
    if (tasks.length === 0) return 0;
    return tasks.filter(t => isTaskDone(t, existingAnns)).length;
  }, [tasks, existingAnns]);

  if (stage === 'loading') {
    return <FullScreen><div className="wb-muted">Loading…</div><style>{css}</style></FullScreen>;
  }
  if (stage === 'error') {
    return (
      <FullScreen>
        <div className="wb-error">{errorMsg ?? 'unknown error'}</div>
        <button className="wb-btn" onClick={() => router.push('/admin/annotate')}>
          ← Back to videos
        </button>
        <style>{css}</style>
      </FullScreen>
    );
  }
  if (stage === 'handedness') {
    return (
      <FullScreen>
        <div className="wb-card">
          <div className="wb-task-title">Step 1 of 2</div>
          <div className="wb-task-name">Golfer handedness</div>
          <div className="wb-card-body">
            <button
              className={`wb-handed-btn ${handedness === 'right' ? 'wb-handed-on' : ''}`}
              onClick={() => setHandedness('right')}
            >Right-handed</button>
            <button
              className={`wb-handed-btn ${handedness === 'left' ? 'wb-handed-on' : ''}`}
              onClick={() => setHandedness('left')}
            >Left-handed</button>
          </div>
          <button className="wb-btn wb-btn-primary" onClick={confirmHandedness}>
            Continue →
          </button>
        </div>
        <style>{css}</style>
      </FullScreen>
    );
  }
  if (stage === 'phase_confirm' && phaseFrames) {
    return (
      <FullScreen>
        <div className="wb-card">
          <div className="wb-task-title">Step 2 of 2</div>
          <div className="wb-task-name">Confirm phase frames</div>
          <div className="wb-card-body wb-stack">
            {PHASE_ORDER.map(p => (
              <div key={p} className="wb-kv">
                <span className="wb-kv-k">{p}</span>
                <span className="wb-kv-v">frame {phaseFrames[p]}</span>
              </div>
            ))}
          </div>
          <div className="wb-btn-row">
            <button className="wb-btn" onClick={enterCalibrate}>Phase markers wrong</button>
            <button
              className="wb-btn wb-btn-primary"
              onClick={() => startAnnotating(phaseFrames)}
            >
              Looks right → annotate
            </button>
          </div>
        </div>
        <style>{css}</style>
      </FullScreen>
    );
  }

  /* ─── phase_calibrate / annotating / done — main split layout ─── */

  return (
    <div className="wb-root">
      <header className="wb-header">
        <div className="wb-title">Landmark Annotation Workbench</div>
        <div className="wb-subtitle">
          {videoMeta?.filename ?? videoId.slice(0, 8)} · {handedness}-handed
        </div>
        <div className="wb-spacer" />
        <button
          className="wb-btn wb-btn-ghost"
          onClick={() => router.push('/admin/annotate')}
        >
          ← Videos
        </button>
      </header>

      <main className="wb-main">
        <div className="wb-stage">
          {videoMeta && (
            <>
              <video
                ref={videoRef}
                className="wb-video"
                src={videoMeta.signedVideoUrl}
                muted
                playsInline
                preload="auto"
                onLoadedMetadata={onVideoLoadedMetadata}
              />
              <canvas
                ref={canvasRef}
                className="wb-canvas"
                onClick={handleCanvasClick}
              />
            </>
          )}
        </div>

        <aside className="wb-panel">
          {stage === 'phase_calibrate' && (
            <PhaseCalibratePanel
              target={calibPhaseTarget}
              captured={calibCaptured}
              frameIdx={currentFrameIdx}
              fps={videoMeta?.fps ?? 30}
              seekToFrame={seekToFrame}
              captureCurrent={captureCurrentAsPhase}
            />
          )}
          {stage === 'annotating' && currentTask && (
            <AnnotatingPanel
              tasks={tasks}
              existingAnns={existingAnns}
              completedCount={completedCount}
              currentTaskIdx={currentTaskIdx}
              activePoints={activePoints}
              pointStepIdx={pointStepIdx}
              frameIdx={currentFrameIdx}
              fps={videoMeta?.fps ?? 30}
              savingError={savingError}
              onUndo={undoLastPoint}
              onOccluded={saveAsOccluded}
              onUncertain={saveAsUncertain}
              onSkip={advanceWithoutSaving}
            />
          )}
          {stage === 'done' && (
            <DonePanel
              videoId={videoId}
              total={tasks.length}
              onBack={() => router.push('/admin/annotate')}
            />
          )}
        </aside>
      </main>
      <style>{css}</style>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════
   Sub-components — kept inline to match codebase single-file style
   ════════════════════════════════════════════════════════════════════ */

function FullScreen({ children }: { children: React.ReactNode }) {
  return <div className="wb-fullscreen">{children}</div>;
}

function PhaseCalibratePanel(props: {
  target: PhaseKey | null;
  captured: Partial<PhaseFrames>;
  frameIdx: number;
  fps: number;
  seekToFrame: (f: number) => void;
  captureCurrent: (phase: PhaseKey) => void;
}) {
  const { target, captured, frameIdx, fps, seekToFrame, captureCurrent } = props;
  if (!target) return <div className="wb-muted">All phases captured…</div>;
  const tSec = (frameIdx / Math.max(1, fps)).toFixed(2);
  return (
    <>
      <div className="wb-task-title">Phase calibration</div>
      <div className="wb-task-name">Mark the {target.toUpperCase()} frame</div>
      <div className="wb-muted">
        Scrub with ← → (±1 frame) or ↑ ↓ (±10), then click below.
      </div>

      <div className="wb-frame-row">
        <button className="wb-btn wb-btn-small" onClick={() => seekToFrame(frameIdx - 10)}>−10</button>
        <button className="wb-btn wb-btn-small" onClick={() => seekToFrame(frameIdx - 1)}>−1</button>
        <span className="wb-frame-info">frame {frameIdx} · {tSec}s</span>
        <button className="wb-btn wb-btn-small" onClick={() => seekToFrame(frameIdx + 1)}>+1</button>
        <button className="wb-btn wb-btn-small" onClick={() => seekToFrame(frameIdx + 10)}>+10</button>
      </div>

      <button
        className="wb-btn wb-btn-primary"
        onClick={() => captureCurrent(target)}
      >
        This is {target.toUpperCase()} → frame {frameIdx}
      </button>

      <div className="wb-stack wb-card-body">
        {PHASE_ORDER.map(p => (
          <div key={p} className={`wb-kv ${captured[p] !== undefined ? 'wb-kv-done' : ''}`}>
            <span className="wb-kv-k">{p}</span>
            <span className="wb-kv-v">
              {captured[p] !== undefined ? `frame ${captured[p]}` : '—'}
            </span>
          </div>
        ))}
      </div>
    </>
  );
}

function AnnotatingPanel(props: {
  tasks: AnnotationTask[];
  existingAnns: AnnotationRecord[];
  completedCount: number;
  currentTaskIdx: number;
  activePoints: (Point | null)[];
  pointStepIdx: number;
  frameIdx: number;
  fps: number;
  savingError: string | null;
  onUndo: () => void;
  onOccluded: () => void;
  onUncertain: () => void;
  onSkip: () => void;
}) {
  const {
    tasks, existingAnns, completedCount, currentTaskIdx,
    activePoints, pointStepIdx,
    frameIdx, fps, savingError,
    onUndo, onOccluded, onUncertain, onSkip,
  } = props;
  const task = tasks[currentTaskIdx];
  const tSec = (frameIdx / Math.max(1, fps)).toFixed(2);
  // Arm-side suffix used by both the progress-bar current cell and the
  // active click-step row, so the workbench shows-not-tells which arm
  // is being annotated.
  const armSuffix = task.arm; // 'lead' | 'trail'
  return (
    <>
      <div className="wb-progress">
        {tasks.map((t, i) => {
          const isCurrent = i === currentTaskIdx;
          // Done = actually saved to DB (so skipped tasks don't lie as "done").
          const isDone = isTaskDone(t, existingAnns);
          const currentCls = isCurrent ? `current current-${armSuffix}` : '';
          return (
            <div
              key={t.index}
              className={`wb-progress-cell ${isDone ? 'done' : ''} ${currentCls}`}
            />
          );
        })}
      </div>

      <div className="wb-task-title">
        Task {currentTaskIdx + 1} of {tasks.length} · {completedCount} done
      </div>
      <div className="wb-task-name">
        {task.phase.toUpperCase()} · {task.arm.toUpperCase()} arm
      </div>

      <div className="wb-steps">
        {POINT_LABELS.map((label, i) => {
          const filled = activePoints[i] !== null;
          const active = pointStepIdx === i;
          const activeCls = active ? `active active-${armSuffix}` : '';
          return (
            <div
              key={label}
              className={`wb-step ${filled ? 'done' : ''} ${activeCls}`}
            >
              <span>{i + 1}.</span>
              <span>{label}</span>
              <span className="wb-spacer" />
              <span>{filled ? '●' : '○'}</span>
            </div>
          );
        })}
      </div>

      <div className="wb-muted">frame {frameIdx} · {tSec}s</div>

      {savingError && <div className="wb-error">{savingError}</div>}

      <div className="wb-btn-row">
        <button className="wb-btn" onClick={onUndo}>Undo (z)</button>
        <button className="wb-btn" onClick={onOccluded}>Occluded (o)</button>
        <button className="wb-btn" onClick={onUncertain}>Uncertain (u)</button>
        <button className="wb-btn" onClick={onSkip}>Skip (s)</button>
      </div>
    </>
  );
}

function DonePanel(props: {
  videoId: string;
  total: number;
  onBack: () => void;
}) {
  const { videoId, total, onBack } = props;
  return (
    <>
      <div className="wb-task-title">Complete</div>
      <div className="wb-task-name">All {total} tasks done</div>
      <div className="wb-muted">
        Annotations are saved server-side. Export as JSON for fixture parity
        with the standalone HTML tool.
      </div>
      <div className="wb-btn-row">
        <a
          className="wb-btn wb-btn-primary"
          href={`/api/admin/annotations/${videoId}/export`}
        >
          Export JSON
        </a>
        <button className="wb-btn" onClick={onBack}>← Back to videos</button>
      </div>
    </>
  );
}

/* ════════════════════════════════════════════════════════════════════
   Canvas drawing primitives
   ════════════════════════════════════════════════════════════════════ */

function drawDot(
  ctx: CanvasRenderingContext2D,
  p: Point,
  r: number,
  fill: string,
  ringR?: number,
  ringColor?: string,
) {
  ctx.beginPath();
  ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  if (ringR != null && ringColor) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, ringR, 0, Math.PI * 2);
    ctx.strokeStyle = ringColor;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

function drawPolyline(
  ctx: CanvasRenderingContext2D,
  pts: Point[],
  stroke: string,
  width: number,
) {
  if (pts.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
  ctx.strokeStyle = stroke;
  ctx.lineWidth = width;
  ctx.stroke();
}

/* ════════════════════════════════════════════════════════════════════
   Styles — single inline template per existing codebase convention.
   Pure black/gray/white. --annot-error reserved for the error line only.
   ════════════════════════════════════════════════════════════════════ */

const css = `
  .wb-root {
    min-height: 100vh;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: 'DM Sans', system-ui, sans-serif;
    display: grid;
    grid-template-rows: auto 1fr;
  }
  .wb-header {
    height: 40px;
    padding: 6px 16px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .wb-title { font-weight: 500; font-size: 13px; letter-spacing: -0.01em; }
  .wb-subtitle { color: var(--text-muted); font-size: 11px; }
  .wb-spacer { flex: 1; }

  .wb-main {
    display: grid;
    grid-template-columns: 1fr 280px;
    gap: 12px;
    padding: 12px;
    min-height: 0;
  }
  @media (max-width: 1024px) {
    .wb-main { grid-template-columns: 1fr; }
  }

  .wb-stage {
    background: #000;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    position: relative;
    overflow: hidden;
    aspect-ratio: 16 / 9;
  }
  .wb-video, .wb-canvas {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
  }
  .wb-video { object-fit: contain; background: #000; }
  .wb-canvas { cursor: crosshair; }

  .wb-panel {
    background: var(--surface-card);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  .wb-progress {
    display: grid;
    grid-template-columns: repeat(14, 1fr);
    gap: 3px;
  }
  .wb-progress-cell {
    height: 8px;
    border-radius: 2px;
    border: 1px solid rgba(255, 255, 255, 0.08);
  }
  .wb-progress-cell.done {
    background: var(--text-primary);
    opacity: 0.4;
    border-color: transparent;
  }
  /* .current is the base state; arm-suffixed variants below pick the
     accent so the annotator can see at a glance which arm is active. */
  .wb-progress-cell.current        { border-color: var(--text-primary); }
  .wb-progress-cell.current-lead   { border-color: var(--annot-lead); }
  .wb-progress-cell.current-trail  { border-color: var(--annot-trail); }

  .wb-task-title {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .wb-task-name {
    font-size: 20px;
    font-weight: 700;
  }

  .wb-steps {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 13px;
  }
  .wb-step {
    display: flex;
    gap: 8px;
    padding: 6px 8px;
    border-radius: 4px;
    border: 1px solid rgba(255, 255, 255, 0.08);
  }
  .wb-step.active {
    border-color: var(--text-primary);
    color: var(--text-primary);
  }
  .wb-step.active-lead  { border-color: var(--annot-lead);  color: var(--annot-lead);  }
  .wb-step.active-trail { border-color: var(--annot-trail); color: var(--annot-trail); }
  .wb-step.done {
    opacity: 0.4;
  }

  .wb-btn-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
  }
  .wb-btn {
    padding: 8px 10px;
    background: transparent;
    color: var(--text-primary);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 4px;
    cursor: pointer;
    font: inherit;
    font-size: 12px;
    font-weight: 600;
    text-decoration: none;
    text-align: center;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    transition: border-color 0.12s, color 0.12s, background 0.12s;
  }
  .wb-btn:hover {
    border-color: var(--text-primary);
  }
  .wb-btn-primary {
    background: var(--text-primary);
    color: #080c08;
    border-color: var(--text-primary);
  }
  .wb-btn-primary:hover {
    background: var(--text-primary);
  }
  .wb-btn-small {
    padding: 6px 10px;
    font-size: 11px;
  }
  .wb-btn-ghost {
    border-color: rgba(255, 255, 255, 0.12);
    color: var(--text-muted);
  }

  .wb-error {
    color: var(--annot-error);
    font-size: 12px;
    border: 1px solid var(--annot-error);
    padding: 8px 10px;
    border-radius: 4px;
  }
  .wb-muted {
    color: var(--text-muted);
    font-size: 12px;
  }

  .wb-fullscreen {
    min-height: 100vh;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: 'DM Sans', system-ui, sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 16px;
    padding: 24px;
  }
  .wb-card {
    background: var(--surface-card);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 24px;
    width: 100%;
    max-width: 420px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .wb-card-body {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .wb-stack { display: flex; flex-direction: column; gap: 4px; }

  .wb-handed-btn {
    padding: 14px 16px;
    background: transparent;
    color: var(--text-primary);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 6px;
    cursor: pointer;
    font: inherit;
    font-size: 14px;
    font-weight: 600;
  }
  .wb-handed-on {
    border-color: var(--text-primary);
  }

  .wb-kv {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 8px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 4px;
    font-size: 12px;
  }
  .wb-kv-done { opacity: 0.5; }
  .wb-kv-k { text-transform: uppercase; letter-spacing: 0.04em; color: var(--text-muted); }
  .wb-kv-v { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }

  .wb-frame-row {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }
  .wb-frame-info {
    flex: 1;
    text-align: center;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px;
    color: var(--text-muted);
  }
`;
