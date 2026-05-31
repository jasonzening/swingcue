'use client';

// PR-7A.1 Phase 2 commit 2 stub. The full overlay + verdict UI lands
// in commit 3 — this stub exists so the server component's import
// resolves and the build is green at the commit 2 boundary.

import type { TaskPhase, Handedness } from '@/lib/types/annotation';

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

export function ReviewView(props: ReviewViewProps) {
  return (
    <div style={{ padding: 16, fontFamily: 'system-ui', fontSize: 12 }}>
      <p style={{ color: '#888' }}>
        ReviewView stub · commit 2 boundary. Full overlay + verdict UI lands in commit 3.
      </p>
      <pre style={{ background: '#0B0F12', padding: 12, borderRadius: 4, overflow: 'auto' }}>
        {JSON.stringify({
          videoId: props.videoId,
          viewType: props.viewType,
          handedness: props.handedness,
          fps: props.fpsSampled,
          dims: [props.videoWidth, props.videoHeight],
          perPhase: props.perPhase.map(p => ({
            phase: p.phase,
            frameIdx: p.frameIdx,
            timeSec: Number(p.timeSec.toFixed(3)),
            hasArmLead: !!p.armLead,
            hasArmTrail: !!p.armTrail,
            hasHip: !!p.hipPair,
            hasWham: !!p.wham,
            hasMp: !!p.mediaPipe,
            existingReview: p.existingReview?.verdict ?? null,
          })),
        }, null, 2)}
      </pre>
    </div>
  );
}
