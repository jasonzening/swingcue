'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  SOURCE_APP_VERSION,
  TASK_PHASES,
  type AnnotationArm,
  type AnnotationRecord,
  type AnnotationVisibility,
  type ArmTask,
  type Handedness,
  type HeadSetAnnotation,
  type HeadSetTask,
  type HipPairAnnotation,
  type HipPairTask,
  type LegAnnotation,
  type LegTask,
  type TaskPhase,
  type VideoMetaForAnnotation,
  type WorkbenchTask,
} from '@/lib/types/annotation';
import {
  derivePhaseFrames,
  type PhaseFrames,
} from '@/lib/admin/phaseFrames';
import {
  ANNOTATION_GUIDE_CSS,
  ANNOTATION_GUIDE_STORAGE_KEY,
  AnnotationGuideBody,
  AnkleDiagram,
  ChinDiagram,
  HeadCrownDiagram,
  HipDiagram,
  KneeDiagram,
} from '@/app/admin/annotation-guide/page';

/* ════════════════════════════════════════════════════════════════════
   PR-7A.1 Annotation workbench v2 — bone-top-centerline.

   15 tasks per video: 10 arm (5 phases × lead/trail, interleaved
   per phase) + 5 hip-pair (one per phase, after the arm batch).

   Stages:
     loading        → fetch meta + existing annotations
     error          → terminal display
     handedness     → ask right / left (defaults right)
     phase_confirm  → auto-derived 5 phase frames look right?
     phase_calibrate → manual scrub-and-mark for 5 phases in turn
     annotating     → 15-task linear flow with auto-skip of done tasks
     done           → all tasks complete / declined
   ════════════════════════════════════════════════════════════════════ */

type Stage =
  | 'loading' | 'error'
  | 'handedness'
  | 'phase_confirm' | 'phase_calibrate'
  | 'annotating'
  | 'done';

const PHASE_ORDER: readonly TaskPhase[] = TASK_PHASES;

// PhaseFrames type + derivation helpers extracted to
// src/lib/admin/phaseFrames.ts so the review page (Phase 2) imports
// the exact same function. Single source of truth — the workbench and
// the validation overlay MUST agree on which frame_idx each phase maps
// to, or arm-dot coords land on the wrong frame.

type Point = { x: number; y: number };

// Arm tasks: 0|1|2|3 (steps 0–2 are shoulder/elbow/wrist; 3 = "all done").
// Hip tasks: 0|1|2 (steps 0–1 are lead/trail hip; 2 = "all done").
// Single state variable across both kinds — the per-task render code
// only reads up to its valid range.
type PointStepIdx = 0 | 1 | 2 | 3;

const POINT_LABELS_ARM      = ['Shoulder', 'Elbow', 'Wrist'] as const;
const POINT_LABELS_HIP      = ['Lead hip', 'Trail hip'] as const;
const POINT_LABELS_HEAD_SET = ['Head crown', 'Chin'] as const;
const POINT_LABELS_LEG      = ['Knee', 'Ankle'] as const;

// Side accent colors — locked-in hex (mirrors --annot-* tokens in
// globals.css). For hip tasks, step 0 (lead hip) uses LEAD; step 1
// (trail hip) uses TRAIL. For leg tasks, both points (knee + ankle)
// inherit the task's arm color. Head joints use COLOR_HEAD (white)
// per PR-7A.2 color protocol.
const COLOR_LEAD   = '#FFD86B';
const COLOR_TRAIL  = '#4FB3FF';
const COLOR_HEAD   = '#FFFFFF';

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

const ARM_TASK_COUNT  = TASK_PHASES.length * 2;   // 5 phases × 2 arms = 10
const HIP_TASK_COUNT  = TASK_PHASES.length;       // 5 phases × 1 hip  = 5
const HEAD_TASK_COUNT = TASK_PHASES.length;       // 5 phases × 1 head = 5
const LEG_TASK_COUNT  = TASK_PHASES.length * 2;   // 5 phases × 2 legs = 10
// Total 30. Kept implicit via the per-cluster counts above.

/**
 * PR-7A.2: per-phase interleaving — 6 tasks per phase × 5 phases = 30
 * total. Order within phase: arm_lead, arm_trail, hip_pair, head_set,
 * leg_lead, leg_trail. Previously v1 batched arms (10) then hips (5);
 * the new flow keeps the annotator on a single phase's frame_idx
 * across all six clusters, then advances to the next phase.
 *
 * Existing 998e1930 + 51ca9428 v2 rows continue to register as done
 * — `findFirstUnannotatedIndex` is order-agnostic, it just scans for
 * the first task whose existence isn't in the DB.
 */
function generateTasks(pf: PhaseFrames): WorkbenchTask[] {
  const out: WorkbenchTask[] = [];
  let idx = 0;
  for (const phase of PHASE_ORDER) {
    const f = pf[phase];
    out.push({ kind: 'arm',      index: idx++, phase, arm: 'lead',  frameIdx: f });
    out.push({ kind: 'arm',      index: idx++, phase, arm: 'trail', frameIdx: f });
    out.push({ kind: 'hip_pair', index: idx++, phase,                frameIdx: f });
    out.push({ kind: 'head_set', index: idx++, phase,                frameIdx: f });
    out.push({ kind: 'leg',      index: idx++, phase, arm: 'lead',  frameIdx: f });
    out.push({ kind: 'leg',      index: idx++, phase, arm: 'trail', frameIdx: f });
  }
  return out;
}

function isArmTaskDone(t: ArmTask, existing: AnnotationRecord[]): boolean {
  return existing.some(a =>
    a.frame_idx === t.frameIdx && a.arm === t.arm && a.task_type === 'manual_gt',
  );
}

function isHipTaskDone(t: HipPairTask, existing: AnnotationRecord[]): boolean {
  return existing.some(a =>
    a.frame_idx === t.frameIdx && a.task_type === 'manual_gt_hip_pair',
  );
}

function isHeadSetTaskDone(t: HeadSetTask, existing: AnnotationRecord[]): boolean {
  return existing.some(a =>
    a.frame_idx === t.frameIdx && a.task_type === 'manual_gt_head_set',
  );
}

function isLegTaskDone(t: LegTask, existing: AnnotationRecord[]): boolean {
  return existing.some(a =>
    a.frame_idx === t.frameIdx && a.arm === t.arm && a.task_type === 'manual_gt_leg',
  );
}

function isTaskDone(t: WorkbenchTask, existing: AnnotationRecord[]): boolean {
  switch (t.kind) {
    case 'arm':      return isArmTaskDone(t, existing);
    case 'hip_pair': return isHipTaskDone(t, existing);
    case 'head_set': return isHeadSetTaskDone(t, existing);
    case 'leg':      return isLegTaskDone(t, existing);
  }
}

function findFirstUnannotatedIndex(
  tasks: WorkbenchTask[],
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
 */
function findNextUnannotatedIndex(
  tasks: WorkbenchTask[],
  existing: AnnotationRecord[],
  fromIdx: number,
): number {
  for (let i = fromIdx; i < tasks.length; i++) {
    if (!isTaskDone(tasks[i], existing)) return i;
  }
  return tasks.length;
}

// timeToFrame + derivePhaseFrames extracted (see import above).

/**
 * PR-7A.1 Phase 3 — letterbox-aware CSS-px → native-px conversion.
 *
 * The .wb-stage container is forced 16:9 (landscape) via aspect-ratio
 * CSS, but the typical swing video is 720×1280 portrait. With
 * object-fit:contain on .wb-video the visible video is letterboxed
 * (black bars left + right). The canvas covers the FULL container
 * (incl. the black bars), so a click's CSS coordinates can't be
 * mapped to native pixels by simple proportional scaling — we have
 * to subtract the letterbox offset and rescale by the displayed
 * region only.
 *
 * Pre-fix horizontal coords stored at roughly mirror-and-compress
 * around the centerline, which is why the GT on 998e1930 looks
 * off-bone in the Phase 2 review page.
 */
function clientToNative(
  e: React.MouseEvent<HTMLCanvasElement> | React.PointerEvent<HTMLCanvasElement>,
  canvas: HTMLCanvasElement,
  nativeW: number,
  nativeH: number,
): { x: number; y: number } | null {
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return null;

  const videoAspect = nativeW / nativeH;
  const containerAspect = rect.width / rect.height;

  let displayedW: number, displayedH: number, offsetX: number, offsetY: number;
  if (videoAspect < containerAspect) {
    // Letterboxed horizontally (portrait video in 16:9 container).
    displayedH = rect.height;
    displayedW = displayedH * videoAspect;
    offsetX = (rect.width - displayedW) / 2;
    offsetY = 0;
  } else {
    // Letterboxed vertically (landscape video in narrow container).
    displayedW = rect.width;
    displayedH = displayedW / videoAspect;
    offsetX = 0;
    offsetY = (rect.height - displayedH) / 2;
  }

  const cx = e.clientX - rect.left - offsetX;
  const cy = e.clientY - rect.top  - offsetY;

  // Clamp to displayed area so clicks in the letterbox bars don't
  // extrapolate to negative or out-of-range native pixels.
  const clampedX = Math.max(0, Math.min(displayedW, cx));
  const clampedY = Math.max(0, Math.min(displayedH, cy));

  return {
    x: Math.round(clampedX / displayedW * nativeW),
    y: Math.round(clampedY / displayedH * nativeH),
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
  const [calibPhaseTarget, setCalibPhaseTarget] = useState<TaskPhase | null>(null);
  const [calibCaptured, setCalibCaptured] = useState<Partial<PhaseFrames>>({});

  const [tasks, setTasks] = useState<WorkbenchTask[]>([]);
  const [currentTaskIdx, setCurrentTaskIdx] = useState(0);

  // Per-task interaction state.
  const [activePoints, setActivePoints] = useState<(Point | null)[]>([null, null, null]);
  const [pointStepIdx, setPointStepIdx] = useState<PointStepIdx>(0);

  const [currentFrameIdx, setCurrentFrameIdx] = useState(0);
  const [savingError, setSavingError] = useState<string | null>(null);

  // PR-7A.1: annotation-guide modal. First-visit shows automatically
  // (when localStorage key is absent); the sidebar "View guide" link
  // re-opens it with the close button always working (no re-ack).
  // Lazy initializer reads localStorage on the first render — avoids
  // calling setState from a useEffect body (React 19 set-state-in-effect
  // rule). SSR-safe via the typeof window guard.
  const [guideOpen, setGuideOpen] = useState<'first' | 'reopen' | null>(() => {
    if (typeof window === 'undefined') return null;
    return localStorage.getItem(ANNOTATION_GUIDE_STORAGE_KEY) ? null : 'first';
  });

  const videoRef  = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const currentTask: WorkbenchTask | null =
    stage === 'annotating' && tasks[currentTaskIdx] ? tasks[currentTaskIdx] : null;

  // Required points for the current task: 3 for arm, 2 for hip / head /
  // leg. The PointStepIdx union (0|1|2|3) covers the arm worst-case.
  const requiredPointCount =
    currentTask?.kind === 'arm' ? 3 :
    currentTask ? 2 : 3;

  /* ──────────────────────────────────────────────────────────────
     LOAD: video meta + existing annotations
     ────────────────────────────────────────────────────────────── */

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [metaRes, annRes] = await Promise.all([
          fetch(`/api/admin/videos/${videoId}`, { cache: 'no-store' }),
          // PR-7A.2: fetch all v2 task types (?taskType=all) so the
          // completion checks see arm + hip + head + leg rows in one
          // request — fixes the hip-resume gap from PR-7A.1 where only
          // arm rows were fetched and hip tasks always re-prompted.
          fetch(
            `/api/admin/annotations/${videoId}?taskType=all`,
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
  // below derives currentFrameIdx from the video's actual currentTime.
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

    // Saved annotations on this frame — drawn faint for context.
    const savedForFrame = existingAnns.filter(a => a.frame_idx === currentFrameIdx);
    for (const a of savedForFrame) {
      if (a.task_type === 'manual_gt' && a.arm != null) {
        // Arm record: lead/trail color, shoulder→elbow→wrist polyline.
        const pts: Point[] = [];
        if (a.shoulder_x != null && a.shoulder_y != null) pts.push({ x: a.shoulder_x, y: a.shoulder_y });
        if (a.elbow_x    != null && a.elbow_y    != null) pts.push({ x: a.elbow_x,    y: a.elbow_y    });
        if (a.wrist_x    != null && a.wrist_y    != null) pts.push({ x: a.wrist_x,    y: a.wrist_y    });
        const base = armColor(a.arm);
        drawPolyline(ctx, pts, hexToRgba(base, 0.25), 1);
        for (const p of pts) drawDot(ctx, p, 3, hexToRgba(base, 0.35));
      } else if (a.task_type === 'manual_gt_hip_pair') {
        // Hip-pair record: lead hip yellow, trail hip blue. No polyline
        // — the two hips don't form an anatomical chain.
        if (a.lead_hip_x != null && a.lead_hip_y != null) {
          drawDot(ctx, { x: a.lead_hip_x, y: a.lead_hip_y }, 3, hexToRgba(COLOR_LEAD, 0.35));
        }
        if (a.trail_hip_x != null && a.trail_hip_y != null) {
          drawDot(ctx, { x: a.trail_hip_x, y: a.trail_hip_y }, 3, hexToRgba(COLOR_TRAIL, 0.35));
        }
      } else if (a.task_type === 'manual_gt_head_set') {
        // Head-set record: head_crown + chin both white per PR-7A.2
        // color protocol. No polyline — the skull and jaw don't form
        // a meaningful drawn chain at workbench scale.
        if (a.head_crown_x != null && a.head_crown_y != null) {
          drawDot(ctx, { x: a.head_crown_x, y: a.head_crown_y }, 3, hexToRgba(COLOR_HEAD, 0.35));
        }
        if (a.chin_x != null && a.chin_y != null) {
          drawDot(ctx, { x: a.chin_x, y: a.chin_y }, 3, hexToRgba(COLOR_HEAD, 0.35));
        }
      } else if (a.task_type === 'manual_gt_leg' && a.arm != null) {
        // Leg record: knee + ankle in the row's arm color (lead = yellow,
        // trail = blue). Polyline knee → ankle since they form a
        // straight bone segment (tibia).
        const base = armColor(a.arm);
        const pts: Point[] = [];
        if (a.knee_x  != null && a.knee_y  != null) pts.push({ x: a.knee_x,  y: a.knee_y  });
        if (a.ankle_x != null && a.ankle_y != null) pts.push({ x: a.ankle_x, y: a.ankle_y });
        drawPolyline(ctx, pts, hexToRgba(base, 0.25), 1);
        for (const p of pts) drawDot(ctx, p, 3, hexToRgba(base, 0.35));
      }
    }

    // Active points — the task currently being annotated.
    if (currentTask?.kind === 'arm') {
      const base = armColor(currentTask.arm);
      const activeFiltered: Point[] = activePoints.filter((p): p is Point => p !== null);
      if (activeFiltered.length > 1) {
        drawPolyline(ctx, activeFiltered, hexToRgba(base, 0.6), 1.5);
      }
      for (const p of activeFiltered) {
        drawDot(ctx, p, 4, base, 8, hexToRgba(base, 0.4));
      }
    } else if (currentTask?.kind === 'hip_pair') {
      // Hip: step 0 = lead (yellow), step 1 = trail (blue). Render each
      // active point in its side's color.
      const leadPt  = activePoints[0];
      const trailPt = activePoints[1];
      if (leadPt) {
        drawDot(ctx, leadPt, 4, COLOR_LEAD, 8, hexToRgba(COLOR_LEAD, 0.4));
      }
      if (trailPt) {
        drawDot(ctx, trailPt, 4, COLOR_TRAIL, 8, hexToRgba(COLOR_TRAIL, 0.4));
      }
    } else if (currentTask?.kind === 'head_set') {
      // Head: step 0 = head_crown, step 1 = chin. Both white per
      // PR-7A.2 color protocol.
      const crownPt = activePoints[0];
      const chinPt  = activePoints[1];
      if (crownPt) drawDot(ctx, crownPt, 4, COLOR_HEAD, 8, hexToRgba(COLOR_HEAD, 0.4));
      if (chinPt)  drawDot(ctx, chinPt,  4, COLOR_HEAD, 8, hexToRgba(COLOR_HEAD, 0.4));
    } else if (currentTask?.kind === 'leg') {
      // Leg: step 0 = knee, step 1 = ankle. Both inherit the task's
      // arm color (lead = yellow, trail = blue). Connected with a
      // tibia polyline once both are placed.
      const base = armColor(currentTask.arm);
      const filtered: Point[] = activePoints
        .slice(0, 2)
        .filter((p): p is Point => p !== null);
      if (filtered.length > 1) {
        drawPolyline(ctx, filtered, hexToRgba(base, 0.6), 1.5);
      }
      for (const p of filtered) {
        drawDot(ctx, p, 4, base, 8, hexToRgba(base, 0.4));
      }
    }
  }, [videoMeta, existingAnns, currentFrameIdx, activePoints, currentTask]);

  // Redraw whenever inputs change.
  useEffect(() => { drawCanvas(); }, [drawCanvas]);

  // 'seeked' fires after the video reaches the requested frame. This is
  // where currentFrameIdx becomes truth — derived from the video, not
  // assumed from the request — so the React UI never drifts out of
  // sync with the displayed frame.
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
     Save flows — arm + hip-pair are distinct shapes
     ────────────────────────────────────────────────────────────── */

  const advanceToNext = useCallback((updatedAnns: AnnotationRecord[]) => {
    const next = findNextUnannotatedIndex(tasks, updatedAnns, currentTaskIdx + 1);
    if (next >= tasks.length) {
      setStage('done');
      return;
    }
    setCurrentTaskIdx(next);
    setActivePoints([null, null, null]);
    setPointStepIdx(0);
  }, [tasks, currentTaskIdx]);

  const saveArmAndAdvance = useCallback(async (
    task: ArmTask,
    points: (Point | null)[],
    vis: AnnotationVisibility,
  ) => {
    if (!videoMeta) return;
    setSavingError(null);
    const body: AnnotationRecord = {
      video_id: videoMeta.videoId,
      frame_idx: task.frameIdx,
      phase: task.phase,
      task_type: 'manual_gt',
      arm: task.arm,
      visibility: vis,
      shoulder_x: points[0]?.x ?? null,
      shoulder_y: points[0]?.y ?? null,
      elbow_x:    points[1]?.x ?? null,
      elbow_y:    points[1]?.y ?? null,
      wrist_x:    points[2]?.x ?? null,
      wrist_y:    points[2]?.y ?? null,
      lead_hip_x: null, lead_hip_y: null,
      trail_hip_x: null, trail_hip_y: null,
      handedness,
      source_app_version: SOURCE_APP_VERSION,
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
      const updatedAnns = [
        ...existingAnns.filter(a => !(
          a.frame_idx === body.frame_idx &&
          a.arm === body.arm &&
          a.task_type === body.task_type
        )),
        body,
      ];
      setExistingAnns(updatedAnns);
      advanceToNext(updatedAnns);
    } catch (e) {
      setSavingError(e instanceof Error ? e.message : 'unknown save error');
    }
  }, [videoMeta, existingAnns, handedness, advanceToNext]);

  const saveHipPairAndAdvance = useCallback(async (
    task: HipPairTask,
    leadHip: Point,
    trailHip: Point,
  ) => {
    if (!videoMeta) return;
    setSavingError(null);
    // Strict HipPairAnnotation shape; the API discriminates on task_type.
    const body: HipPairAnnotation = {
      video_id: videoMeta.videoId,
      frame_idx: task.frameIdx,
      phase: task.phase,
      task_type: 'manual_gt_hip_pair',
      lead_hip_x: leadHip.x,  lead_hip_y: leadHip.y,
      trail_hip_x: trailHip.x, trail_hip_y: trailHip.y,
      arm: null,
      handedness,
      visibility: 'clear',
      source_app_version: SOURCE_APP_VERSION,
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
      // Store the just-saved row locally as an AnnotationRecord so the
      // canvas + done-checks reflect it. Mirror the umbrella shape.
      const updatedRecord: AnnotationRecord = {
        video_id: body.video_id,
        frame_idx: body.frame_idx,
        phase: body.phase,
        task_type: body.task_type,
        arm: null,
        visibility: body.visibility,
        shoulder_x: null, shoulder_y: null,
        elbow_x: null,    elbow_y: null,
        wrist_x: null,    wrist_y: null,
        lead_hip_x: body.lead_hip_x,   lead_hip_y: body.lead_hip_y,
        trail_hip_x: body.trail_hip_x, trail_hip_y: body.trail_hip_y,
        handedness: body.handedness,
        source_app_version: body.source_app_version,
      };
      const updatedAnns = [
        ...existingAnns.filter(a => !(
          a.frame_idx === body.frame_idx &&
          a.task_type === body.task_type
        )),
        updatedRecord,
      ];
      setExistingAnns(updatedAnns);
      advanceToNext(updatedAnns);
    } catch (e) {
      setSavingError(e instanceof Error ? e.message : 'unknown save error');
    }
  }, [videoMeta, existingAnns, handedness, advanceToNext]);

  const saveHeadSetAndAdvance = useCallback(async (
    task: HeadSetTask,
    headCrown: Point,
    chin: Point,
  ) => {
    if (!videoMeta) return;
    setSavingError(null);
    const body: HeadSetAnnotation = {
      video_id: videoMeta.videoId,
      frame_idx: task.frameIdx,
      phase: task.phase,
      task_type: 'manual_gt_head_set',
      head_crown_x: headCrown.x, head_crown_y: headCrown.y,
      chin_x: chin.x,            chin_y: chin.y,
      arm: null,
      handedness,
      visibility: 'clear',
      source_app_version: SOURCE_APP_VERSION,
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
      const updatedRecord: AnnotationRecord = {
        video_id: body.video_id,
        frame_idx: body.frame_idx,
        phase: body.phase,
        task_type: body.task_type,
        arm: null,
        visibility: body.visibility,
        shoulder_x: null, shoulder_y: null,
        elbow_x: null,    elbow_y: null,
        wrist_x: null,    wrist_y: null,
        lead_hip_x: null,   lead_hip_y: null,
        trail_hip_x: null,  trail_hip_y: null,
        head_crown_x: body.head_crown_x, head_crown_y: body.head_crown_y,
        chin_x: body.chin_x,             chin_y: body.chin_y,
        knee_x: null,  knee_y: null,
        ankle_x: null, ankle_y: null,
        handedness: body.handedness,
        source_app_version: body.source_app_version,
      };
      const updatedAnns = [
        ...existingAnns.filter(a => !(
          a.frame_idx === body.frame_idx &&
          a.task_type === body.task_type
        )),
        updatedRecord,
      ];
      setExistingAnns(updatedAnns);
      advanceToNext(updatedAnns);
    } catch (e) {
      setSavingError(e instanceof Error ? e.message : 'unknown save error');
    }
  }, [videoMeta, existingAnns, handedness, advanceToNext]);

  const saveLegAndAdvance = useCallback(async (
    task: LegTask,
    knee: Point,
    ankle: Point,
  ) => {
    if (!videoMeta) return;
    setSavingError(null);
    const body: LegAnnotation = {
      video_id: videoMeta.videoId,
      frame_idx: task.frameIdx,
      phase: task.phase,
      task_type: 'manual_gt_leg',
      arm: task.arm,
      knee_x:  knee.x,  knee_y:  knee.y,
      ankle_x: ankle.x, ankle_y: ankle.y,
      handedness,
      visibility: 'clear',
      source_app_version: SOURCE_APP_VERSION,
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
      const updatedRecord: AnnotationRecord = {
        video_id: body.video_id,
        frame_idx: body.frame_idx,
        phase: body.phase,
        task_type: body.task_type,
        arm: body.arm,
        visibility: body.visibility,
        shoulder_x: null, shoulder_y: null,
        elbow_x: null,    elbow_y: null,
        wrist_x: null,    wrist_y: null,
        lead_hip_x: null,  lead_hip_y: null,
        trail_hip_x: null, trail_hip_y: null,
        head_crown_x: null, head_crown_y: null,
        chin_x: null,       chin_y: null,
        knee_x:  body.knee_x,  knee_y:  body.knee_y,
        ankle_x: body.ankle_x, ankle_y: body.ankle_y,
        handedness: body.handedness,
        source_app_version: body.source_app_version,
      };
      const updatedAnns = [
        ...existingAnns.filter(a => !(
          a.frame_idx === body.frame_idx &&
          a.arm === body.arm &&
          a.task_type === body.task_type
        )),
        updatedRecord,
      ];
      setExistingAnns(updatedAnns);
      advanceToNext(updatedAnns);
    } catch (e) {
      setSavingError(e instanceof Error ? e.message : 'unknown save error');
    }
  }, [videoMeta, existingAnns, handedness, advanceToNext]);

  const advanceWithoutSaving = useCallback(() => {
    advanceToNext(existingAnns);
  }, [existingAnns, advanceToNext]);

  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (stage !== 'annotating' || !videoMeta || !currentTask) return;
    if (pointStepIdx >= requiredPointCount) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    // PR-7A.1 Phase 3: letterbox-aware conversion. Old math was a
    // simple proportional scale that ignored object-fit:contain
    // letterboxing → portrait clicks compressed toward centerline.
    const pos = clientToNative(e, canvas, videoMeta.width, videoMeta.height);
    if (!pos) return;
    const { x, y } = pos;

    const next = [...activePoints];
    next[pointStepIdx] = { x, y };
    setActivePoints(next);
    const newStep = (pointStepIdx + 1) as PointStepIdx;
    setPointStepIdx(newStep);

    // Auto-save when all required points are clicked.
    if (currentTask.kind === 'arm' && newStep === 3) {
      void saveArmAndAdvance(currentTask, next, 'clear');
    } else if (currentTask.kind === 'hip_pair' && newStep === 2) {
      const leadHip  = next[0];
      const trailHip = next[1];
      // Both should be non-null by construction (newStep===2 means we
      // just clicked the second point), but narrow defensively.
      if (leadHip && trailHip) {
        void saveHipPairAndAdvance(currentTask, leadHip, trailHip);
      }
    } else if (currentTask.kind === 'head_set' && newStep === 2) {
      const headCrown = next[0];
      const chin      = next[1];
      if (headCrown && chin) {
        void saveHeadSetAndAdvance(currentTask, headCrown, chin);
      }
    } else if (currentTask.kind === 'leg' && newStep === 2) {
      const knee  = next[0];
      const ankle = next[1];
      if (knee && ankle) {
        void saveLegAndAdvance(currentTask, knee, ankle);
      }
    }
  }, [stage, videoMeta, currentTask, pointStepIdx, requiredPointCount, activePoints,
      saveArmAndAdvance, saveHipPairAndAdvance, saveHeadSetAndAdvance, saveLegAndAdvance]);

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
    // Only valid for arm tasks (hip visibility is always 'clear' since
    // it's an estimated internal joint center, not directly visible).
    if (stage !== 'annotating' || !currentTask || currentTask.kind !== 'arm') return;
    void saveArmAndAdvance(currentTask, activePoints, 'occluded');
  }, [stage, currentTask, activePoints, saveArmAndAdvance]);

  const saveAsUncertain = useCallback(() => {
    if (stage !== 'annotating' || !currentTask || currentTask.kind !== 'arm') return;
    if (pointStepIdx !== 3) return;
    void saveArmAndAdvance(currentTask, activePoints, 'uncertain');
  }, [stage, currentTask, pointStepIdx, activePoints, saveArmAndAdvance]);

  /* ──────────────────────────────────────────────────────────────
     Keyboard
     ────────────────────────────────────────────────────────────── */

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      // Modal swallows shortcuts (guide handles its own ESC internally).
      if (guideOpen) return;
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
    stage, guideOpen,
    undoLastPoint, saveAsOccluded, saveAsUncertain, advanceWithoutSaving,
    currentFrameIdx, seekToFrame,
  ]);

  /* ──────────────────────────────────────────────────────────────
     Phase calibrate: "this is X" capture
     ────────────────────────────────────────────────────────────── */

  const captureCurrentAsPhase = useCallback((phase: TaskPhase) => {
    const next: Partial<PhaseFrames> = {
      ...calibCaptured,
      [phase]: currentFrameIdx,
    };
    setCalibCaptured(next);
    const remaining = PHASE_ORDER.find(p => next[p] === undefined);
    if (remaining) {
      setCalibPhaseTarget(remaining);
    } else {
      startAnnotating(next as PhaseFrames);
    }
  }, [calibCaptured, currentFrameIdx, startAnnotating]);

  /* ──────────────────────────────────────────────────────────────
     Render helpers
     ────────────────────────────────────────────────────────────── */

  const { armDone, hipDone, headDone, legDone } = useMemo(() => {
    let a = 0, h = 0, head = 0, leg = 0;
    for (const t of tasks) {
      if (!isTaskDone(t, existingAnns)) continue;
      switch (t.kind) {
        case 'arm':      a++; break;
        case 'hip_pair': h++; break;
        case 'head_set': head++; break;
        case 'leg':      leg++; break;
      }
    }
    return { armDone: a, hipDone: h, headDone: head, legDone: leg };
  }, [tasks, existingAnns]);

  const closeGuide = useCallback(() => setGuideOpen(null), []);
  const reopenGuide = useCallback(() => setGuideOpen('reopen'), []);

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
        {renderGuideModal()}
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
        {renderGuideModal()}
        <style>{css}</style>
      </FullScreen>
    );
  }

  function renderGuideModal() {
    if (!guideOpen) return null;
    return (
      <div
        className="wb-modal-scrim"
        onClick={guideOpen === 'reopen' ? closeGuide : undefined}
      >
        <div className="wb-modal-card" onClick={e => e.stopPropagation()}>
          {guideOpen === 'reopen' && (
            <button
              className="wb-modal-close"
              onClick={closeGuide}
              aria-label="Close annotation guide"
            >×</button>
          )}
          <AnnotationGuideBody
            mode={guideOpen === 'first' ? 'modal' : 'standalone'}
            onAcknowledge={closeGuide}
          />
        </div>
      </div>
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
        <button className="wb-btn wb-btn-ghost wb-btn-link" onClick={reopenGuide}>
          View annotation guide
        </button>
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
              videoId={videoId}
              currentTask={currentTask}
              tasks={tasks}
              existingAnns={existingAnns}
              currentTaskIdx={currentTaskIdx}
              activePoints={activePoints}
              pointStepIdx={pointStepIdx}
              frameIdx={currentFrameIdx}
              fps={videoMeta?.fps ?? 30}
              armDone={armDone}
              hipDone={hipDone}
              headDone={headDone}
              legDone={legDone}
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
              armDone={armDone}
              hipDone={hipDone}
              headDone={headDone}
              legDone={legDone}
              onBack={() => router.push('/admin/annotate')}
            />
          )}
        </aside>
      </main>
      {renderGuideModal()}
      <style>{css + ANNOTATION_GUIDE_CSS + MODAL_CSS}</style>
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
  target: TaskPhase | null;
  captured: Partial<PhaseFrames>;
  frameIdx: number;
  fps: number;
  seekToFrame: (f: number) => void;
  captureCurrent: (phase: TaskPhase) => void;
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
  videoId: string;
  currentTask: WorkbenchTask;
  tasks: WorkbenchTask[];
  existingAnns: AnnotationRecord[];
  currentTaskIdx: number;
  activePoints: (Point | null)[];
  pointStepIdx: number;
  frameIdx: number;
  fps: number;
  armDone: number;
  hipDone: number;
  headDone: number;
  legDone: number;
  savingError: string | null;
  onUndo: () => void;
  onOccluded: () => void;
  onUncertain: () => void;
  onSkip: () => void;
}) {
  const { videoId, currentTask, tasks, existingAnns, currentTaskIdx,
          frameIdx, fps, armDone, hipDone, headDone, legDone, savingError } = props;

  const tSec = (frameIdx / Math.max(1, fps)).toFixed(2);
  // PR-7A.1 Phase 2 / PR-7A.2 update: CTA to validation review
  // surfaces once arm tasks are done. The remaining clusters (hip,
  // head, leg) are nice-to-have but not gated; once the arm chain is
  // verifiable, the review page is useful.
  const showReviewCta = armDone >= ARM_TASK_COUNT;
  return (
    <>
      <div className="wb-progress wb-progress-30">
        {tasks.map((t, i) => {
          const isCurrent = i === currentTaskIdx;
          const isDone = isTaskDone(t, existingAnns);
          let sideClass = '';
          if (t.kind === 'arm' || t.kind === 'leg') {
            sideClass = `current-${t.arm}`;
          } else if (t.kind === 'hip_pair') {
            sideClass = isCurrent && props.pointStepIdx === 0 ? 'current-lead'
                      : isCurrent && props.pointStepIdx >= 1 ? 'current-trail'
                      : 'current-lead';
          } else {
            // head_set — white head color carried by .current-head class
            sideClass = 'current-head';
          }
          const currentCls = isCurrent ? `current ${sideClass}` : '';
          const kindCls =
            t.kind === 'hip_pair' ? 'wb-progress-cell-hip'  :
            t.kind === 'head_set' ? 'wb-progress-cell-head' :
            t.kind === 'leg'      ? 'wb-progress-cell-leg'  : '';
          return (
            <div
              key={t.index}
              className={`wb-progress-cell ${isDone ? 'done' : ''} ${currentCls} ${kindCls}`}
            />
          );
        })}
      </div>

      <div className="wb-task-title">
        {armDone} / {ARM_TASK_COUNT} ARM ·{' '}
        {hipDone} / {HIP_TASK_COUNT} HIP ·{' '}
        {headDone} / {HEAD_TASK_COUNT} HEAD ·{' '}
        {legDone} / {LEG_TASK_COUNT} LEG
      </div>

      {currentTask.kind === 'arm' ? (
        <ArmTaskBody {...props} task={currentTask} />
      ) : currentTask.kind === 'hip_pair' ? (
        <HipTaskBody {...props} task={currentTask} />
      ) : currentTask.kind === 'head_set' ? (
        <HeadSetTaskBody {...props} task={currentTask} />
      ) : (
        <LegTaskBody {...props} task={currentTask} />
      )}

      <div className="wb-muted">frame {frameIdx} · {tSec}s</div>

      {savingError && <div className="wb-error">{savingError}</div>}

      {showReviewCta && (
        <a className="wb-review-cta" href={`/admin/review/${videoId}`}>
          🎯 Validate your annotations →
        </a>
      )}
    </>
  );
}

function ArmTaskBody(props: {
  task: ArmTask;
  activePoints: (Point | null)[];
  pointStepIdx: number;
  onUndo: () => void;
  onOccluded: () => void;
  onUncertain: () => void;
  onSkip: () => void;
}) {
  const { task, activePoints, pointStepIdx, onUndo, onOccluded, onUncertain, onSkip } = props;
  const armSuffix = task.arm;
  return (
    <>
      <div className="wb-task-name">
        {task.phase.toUpperCase()} · {task.arm.toUpperCase()} arm
      </div>

      <div className="wb-steps">
        {POINT_LABELS_ARM.map((label, i) => {
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

      <div className="wb-btn-row">
        <button className="wb-btn" onClick={onUndo}>Undo (z)</button>
        <button className="wb-btn" onClick={onOccluded}>Occluded (o)</button>
        <button className="wb-btn" onClick={onUncertain}>Uncertain (u)</button>
        <button className="wb-btn" onClick={onSkip}>Skip (s)</button>
      </div>
    </>
  );
}

function HipTaskBody(props: {
  task: HipPairTask;
  activePoints: (Point | null)[];
  pointStepIdx: number;
  onUndo: () => void;
  onSkip: () => void;
}) {
  const { task, activePoints, pointStepIdx, onUndo, onSkip } = props;
  // Step 0 highlights lead-yellow, step 1 highlights trail-blue.
  return (
    <>
      <div className="wb-task-name">
        {task.phase.toUpperCase()} · HIP (pair)
      </div>

      <div className="wb-hip-badge">
        5 hip tasks · optional · 估计内部旋转中心
      </div>

      <div className="wb-steps">
        {POINT_LABELS_HIP.map((label, i) => {
          const filled = activePoints[i] !== null;
          const active = pointStepIdx === i;
          const sideSuffix = i === 0 ? 'lead' : 'trail';
          const activeCls = active ? `active active-${sideSuffix}` : '';
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

      {/* Inline HipDiagram reminder so the annotator can see "click the
          internal joint center, NOT the trochanter bump" without
          leaving the workbench. */}
      <div className="wb-hip-diagram">
        <HipDiagram />
      </div>
      <p className="wb-hip-warning">
        Estimated internal joint center. NEVER click the visible trochanter bump.
      </p>

      <div className="wb-btn-row">
        <button className="wb-btn" onClick={onUndo}>Undo (z)</button>
        <button className="wb-btn" onClick={onSkip}>Skip (s)</button>
      </div>
    </>
  );
}

function HeadSetTaskBody(props: {
  task: HeadSetTask;
  activePoints: (Point | null)[];
  pointStepIdx: number;
  onUndo: () => void;
  onSkip: () => void;
}) {
  const { task, activePoints, pointStepIdx, onUndo, onSkip } = props;
  // Show the diagram matching the current step — crown for step 0,
  // chin for step 1. Once both clicks are placed the panel auto-saves
  // and advances, so we never linger on step 2.
  const DiagramForStep = pointStepIdx <= 0 ? HeadCrownDiagram : ChinDiagram;
  return (
    <>
      <div className="wb-task-name">
        {task.phase.toUpperCase()} · HEAD (set)
      </div>

      <div className="wb-hip-badge">
        5 head tasks · 颅顶 + 下颌 bone landmarks
      </div>

      <div className="wb-steps">
        {POINT_LABELS_HEAD_SET.map((label, i) => {
          const filled = activePoints[i] !== null;
          const active = pointStepIdx === i;
          // Both steps use neutral primary color — head joints render
          // as WHITE per the PR-7A.2 color protocol, and the panel
          // accent mirrors that.
          const activeCls = active ? 'active active-head' : '';
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

      <div className="wb-hip-diagram">
        <DiagramForStep />
      </div>
      <p className="wb-hip-warning">
        Visible bone landmarks. Crown = top-of-skull apex, NOT hairline.
        Chin = jaw-bone bottom, NOT soft tissue.
      </p>

      <div className="wb-btn-row">
        <button className="wb-btn" onClick={onUndo}>Undo (z)</button>
        <button className="wb-btn" onClick={onSkip}>Skip (s)</button>
      </div>
    </>
  );
}

function LegTaskBody(props: {
  task: LegTask;
  activePoints: (Point | null)[];
  pointStepIdx: number;
  onUndo: () => void;
  onSkip: () => void;
}) {
  const { task, activePoints, pointStepIdx, onUndo, onSkip } = props;
  const armSuffix = task.arm;
  const DiagramForStep = pointStepIdx <= 0 ? KneeDiagram : AnkleDiagram;
  return (
    <>
      <div className="wb-task-name">
        {task.phase.toUpperCase()} · LEG ({task.arm.toUpperCase()})
      </div>

      <div className="wb-hip-badge">
        10 leg tasks · 外侧骨突 — visible bone bumps
      </div>

      <div className="wb-steps">
        {POINT_LABELS_LEG.map((label, i) => {
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

      <div className="wb-hip-diagram">
        <DiagramForStep />
      </div>
      <p className="wb-hip-warning">
        Visible bone landmarks (unlike hip). Knee = lateral epicondyle
        (OUTSIDE bump), NOT patella. Ankle = lateral malleolus (OUTSIDE
        bump), NOT shoe edge.
      </p>

      <div className="wb-btn-row">
        <button className="wb-btn" onClick={onUndo}>Undo (z)</button>
        <button className="wb-btn" onClick={onSkip}>Skip (s)</button>
      </div>
    </>
  );
}

function DonePanel(props: {
  videoId: string;
  armDone: number;
  hipDone: number;
  headDone: number;
  legDone: number;
  onBack: () => void;
}) {
  const { videoId, armDone, hipDone, headDone, legDone, onBack } = props;
  return (
    <>
      <div className="wb-task-title">Complete</div>
      <div className="wb-task-name">
        {armDone} / {ARM_TASK_COUNT} arm · {hipDone} / {HIP_TASK_COUNT} hip ·{' '}
        {headDone} / {HEAD_TASK_COUNT} head · {legDone} / {LEG_TASK_COUNT} leg
      </div>
      <div className="wb-muted">
        Annotations are saved server-side. Validate them against WHAM and
        MediaPipe to confirm coord alignment.
      </div>
      <a className="wb-review-cta" href={`/admin/review/${videoId}`}>
        🎯 Validate your annotations →
      </a>
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

const MODAL_CSS = `
  .wb-modal-scrim {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.55);
    display: flex;
    align-items: flex-start;
    justify-content: center;
    overflow-y: auto;
    z-index: 100;
    padding: 24px;
  }
  .wb-modal-card {
    position: relative;
    background: #ffffff;
    border-radius: 0.75rem;
    max-width: 56rem;
    width: 100%;
    padding: 1.5rem;
    margin: auto;
    color: #111827;
  }
  .wb-modal-close {
    position: absolute;
    top: 8px;
    right: 12px;
    background: transparent;
    border: none;
    color: #6b7280;
    font-size: 24px;
    line-height: 1;
    cursor: pointer;
    padding: 4px 8px;
    border-radius: 4px;
  }
  .wb-modal-close:hover { background: #f3f4f6; color: #111827; }
`;

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
  /* PR-7A.1 Phase 3: canvas needs object-fit: contain too. Without
     it, the canvas stretches edge-to-edge inside the 16:9 container,
     so drawn dots at native (x, y) render at wrong CSS positions
     relative to the letterboxed video. With object-fit:contain on
     both, the canvas + video share the SAME scale + offset → drawn
     dots land on top of the body pixel they describe. */
  .wb-canvas { cursor: crosshair; object-fit: contain; }

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
    grid-template-columns: repeat(15, 1fr);
    gap: 3px;
  }
  /* PR-7A.2: 30-task grid (10 arm + 5 hip + 5 head + 10 leg). Two-row
     layout keeps each cell tappable; the kind-specific bg tints make
     the cluster boundaries visible at a glance. */
  .wb-progress-30 {
    grid-template-columns: repeat(15, 1fr);
    grid-auto-rows: 8px;
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
  .wb-progress-cell.current        { border-color: var(--text-primary); }
  .wb-progress-cell.current-lead   { border-color: var(--annot-lead); }
  .wb-progress-cell.current-trail  { border-color: var(--annot-trail); }
  .wb-progress-cell.current-head   { border-color: #ffffff; }
  /* Subtle bg tints differentiate clusters in the 30-cell strip. */
  .wb-progress-cell-hip  { background: rgba(255, 255, 255, 0.04); }
  .wb-progress-cell-head { background: rgba(255, 255, 255, 0.07); }
  .wb-progress-cell-leg  { background: rgba(255, 255, 255, 0.10); }

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

  .wb-hip-badge {
    font-size: 11px;
    color: var(--text-muted);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 4px;
    padding: 6px 8px;
    letter-spacing: 0.04em;
  }
  .wb-hip-diagram {
    background: #f9fafb;
    border-radius: 6px;
    padding: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .wb-hip-warning {
    font-size: 11px;
    color: var(--text-muted);
    margin: 0;
    line-height: 1.5;
    border-left: 2px solid rgba(255, 255, 255, 0.18);
    padding-left: 8px;
    font-style: italic;
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
  .wb-step.active-head  { border-color: #ffffff; color: #ffffff; }
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
  .wb-btn-link {
    border-color: transparent;
    text-decoration: underline;
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

  /* PR-7A.1 Phase 2: validation review CTA. Neutral admin palette —
     subtle gray surface, white text, no brand color. Appears in the
     annotating panel once arm tasks complete + in the done panel. */
  .wb-review-cta {
    display: block;
    padding: 10px 12px;
    background: var(--surface-card-alt);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 4px;
    color: var(--text-primary);
    font: inherit;
    font-size: 12px;
    font-weight: 600;
    text-align: center;
    text-decoration: none;
    cursor: pointer;
    transition: border-color 0.12s, background 0.12s;
  }
  .wb-review-cta:hover {
    border-color: var(--text-primary);
    background: var(--surface-card);
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
