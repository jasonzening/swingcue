/**
 * sparsePhaseOverlay.ts — pose_3d_phases → 5-frame OverlayTimeline.
 *
 * Each completed PoseRow becomes one OverlayFrame containing a shoulder disc
 * and a hip disc, both built directly from the row's denormalized anchor
 * columns (shoulder_left_x/y, shoulder_right_x/y, hip_left_x/y, hip_right_x/y).
 *
 * Coordinate convention:
 *   - Source columns are in raw image pixels (frame at fal-call time).
 *   - Normalization (px ÷ image_width / image_height) happens INSIDE the
 *     disc builders, BEFORE the values reach the ellipse element. The
 *     timeline emitted here is in the same normalized 0-1 space that the
 *     existing OverlayRenderer consumes (`prx = rx * W`, `pcy = cy * H`).
 *
 * Ellipse elements deliberately omit `visRatio`, `zAsym`, `bodyHalfRatio` —
 * OverlayRenderer falls back to neutral defaults (1.0 / 0 / 0.27), which
 * gives a symmetric disc. The 3D perspective tilt those fields encode is
 * deferred to PR-2D using keypoints_3d.
 */

import type {
  OverlayElement,
  OverlayFrame,
  OverlayTimeline,
  PhaseMarkers,
  LineElement,
  DotElement,
} from '@/types/analysis';
import {
  COCO_KP,
  MIN_CONFIDENCE,
  PHASE_ORDER,
  type PhaseName,
  type PoseRow,
} from '@/lib/sam3d/keypoints';

type AC = 'red' | 'green' | 'yellow' | 'white';
type Layer = OverlayElement['layer'];

let _uid = 0;
const uid = (p: string) => `${p}-${++_uid}`;
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

/** Normalize angle to [-90, 90]. Disc rendering is 180° symmetric. */
function normalizeAngle(deg: number): number {
  let a = deg;
  if (a > 90) a -= 180;
  if (a < -90) a += 180;
  return a;
}

const mkLine = (
  x1: number, y1: number, x2: number, y2: number,
  c: AC, w = 2.2, op = 0.70, layer: Layer = 'body',
): LineElement => ({
  type: 'line', id: uid('l'),
  x1, y1, x2, y2,
  color: c, strokeWidth: w, opacity: op, layer,
});

const mkDot = (
  x: number, y: number,
  c: AC, r = 0.006, op = 0.50, layer: Layer = 'body',
): DotElement => ({
  type: 'dot', id: uid('d'),
  x, y,
  color: c, radius: r, opacity: op, layer,
});

/**
 * Build a basic ellipse element (cast pattern matches keypointOverlay.ts).
 * No visRatio / zAsym / bodyHalfRatio — renderer applies defaults.
 */
function mkEllipse(
  cx: number, cy: number, rx: number, ry: number,
  angleDeg: number, color: AC, layer: Layer,
  strokeWidth = 5.0, opacity = 0.92,
): OverlayElement {
  return {
    type: 'ellipse' as OverlayElement['type'],
    id: uid('e'),
    cx, cy, rx, ry, angleDeg,
    color, strokeWidth, opacity, layer,
  } as unknown as OverlayElement;
}

type DiscOpts = {
  rxMult: number;
  rxMin: number;
  rxMax: number;
  ryRatio: number;
  maxAngle: number;
  layer: Layer;
  /** Outward radial expansion of endpoint dots, in normalized units. */
  dotExpand: number;
};

const SHOULDER_OPTS: DiscOpts = {
  rxMult: 1.85, rxMin: 0.20, rxMax: 0.50, ryRatio: 0.20,
  maxAngle: 30, layer: 'body', dotExpand: 0,
};
const HIP_OPTS: DiscOpts = {
  rxMult: 2.10, rxMin: 0.16, rxMax: 0.50, ryRatio: 0.18,
  maxAngle: 30, layer: 'club', dotExpand: 0.52,
};

/**
 * Build disc elements (ellipse + 2 endpoint dots + center guide line) from
 * a single anchor pair, in source pixels. Returns [] if any input is null
 * or the points coincide.
 *
 * IMPORTANT: pixel-to-normalized division happens here, before any element
 * is constructed, so the OverlayElement always carries 0-1 coords.
 */
function buildDisc(
  leftXpx: number | null, leftYpx: number | null,
  rightXpx: number | null, rightYpx: number | null,
  imageW: number, imageH: number,
  opts: DiscOpts,
): OverlayElement[] {
  if (
    leftXpx === null || leftYpx === null ||
    rightXpx === null || rightYpx === null
  ) return [];

  // Normalize to 0-1 (same convention MediaPipe path used).
  const lx = leftXpx  / imageW;
  const ly = leftYpx  / imageH;
  const rx = rightXpx / imageW;
  const ry = rightYpx / imageH;

  const dx = rx - lx;
  const dy = ry - ly;
  const dist = Math.hypot(dx, dy);
  if (dist < 0.01) return [];

  const cx = (lx + rx) / 2;
  const cy = (ly + ry) / 2;

  const ellipseRx = clamp(dist * opts.rxMult, opts.rxMin, opts.rxMax);
  const ellipseRy = ellipseRx * opts.ryRatio;
  const angleDeg = clamp(
    normalizeAngle(Math.atan2(dy, dx) * 180 / Math.PI),
    -opts.maxAngle, opts.maxAngle,
  );

  // Endpoint dots — for hips, push outward along the disc axis so the dots
  // land at the visible edge of the (larger-than-anchor-span) ellipse.
  const expand = dist * opts.dotExpand;
  const ux = dist > 0 ? dx / dist : 1;
  const uy = dist > 0 ? dy / dist : 0;
  const lDotX = lx - ux * expand;
  const lDotY = ly - uy * expand;
  const rDotX = rx + ux * expand;
  const rDotY = ry + uy * expand;

  const els: OverlayElement[] = [];
  els.push(mkDot(lDotX, lDotY, 'yellow', 0.006, 0.30, opts.layer));
  els.push(mkDot(rDotX, rDotY, 'white',  0.006, 0.30, opts.layer));
  els.push(mkEllipse(cx, cy, ellipseRx, ellipseRy, angleDeg, 'white', opts.layer));

  // Center guide line through disc along long axis.
  const ar = angleDeg * Math.PI / 180;
  const cosA = Math.cos(ar);
  const sinA = Math.sin(ar);
  const guideHalfLen = ellipseRx * 0.65;
  els.push(mkLine(
    cx - guideHalfLen * cosA, cy - guideHalfLen * sinA,
    cx + guideHalfLen * cosA, cy + guideHalfLen * sinA,
    'white', 2.2, 0.70, opts.layer,
  ));

  return els;
}

type AnchorPair = {
  leftX: number; leftY: number;
  rightX: number; rightY: number;
};

/**
 * Read a confident keypoint from a YOLO row. Returns null when YOLO did
 * not run, the array shape is unexpected, or the confidence is below
 * MIN_CONFIDENCE. Caller then falls back to SAM materialised columns.
 */
function readYoloPair(
  row: PoseRow,
  leftIdx: number,
  rightIdx: number,
): AnchorPair | null {
  const kps = row.yolo_keypoints_2d;
  if (kps === null || kps.length < 17) return null;
  const l = kps[leftIdx];
  const r = kps[rightIdx];
  if (!l || l.length < 3 || !r || r.length < 3) return null;
  if (l[2] < MIN_CONFIDENCE || r[2] < MIN_CONFIDENCE) return null;
  return { leftX: l[0], leftY: l[1], rightX: r[0], rightY: r[1] };
}

/**
 * Read the SAM materialised anchors (PR-2 fallback path). Returns null
 * if either side is missing.
 */
function readSamPair(
  leftX: number | null, leftY: number | null,
  rightX: number | null, rightY: number | null,
): AnchorPair | null {
  if (leftX === null || leftY === null || rightX === null || rightY === null) {
    return null;
  }
  return { leftX, leftY, rightX, rightY };
}

/** YOLO preferred, SAM fallback. Returns null if neither source has data. */
function getShoulderPair(row: PoseRow): AnchorPair | null {
  return (
    readYoloPair(row, COCO_KP.LEFT_SHOULDER, COCO_KP.RIGHT_SHOULDER) ??
    readSamPair(
      row.shoulder_left_x, row.shoulder_left_y,
      row.shoulder_right_x, row.shoulder_right_y,
    )
  );
}

/** YOLO preferred, SAM fallback. Returns null if neither source has data. */
function getHipPair(row: PoseRow): AnchorPair | null {
  return (
    readYoloPair(row, COCO_KP.LEFT_HIP, COCO_KP.RIGHT_HIP) ??
    readSamPair(
      row.hip_left_x, row.hip_left_y,
      row.hip_right_x, row.hip_right_y,
    )
  );
}

/** Public: shoulder disc from a PoseRow. Returns [] if no source has anchors. */
export function buildShoulderDisc(row: PoseRow): OverlayElement[] {
  const pair = getShoulderPair(row);
  if (pair === null) return [];
  return buildDisc(
    pair.leftX, pair.leftY, pair.rightX, pair.rightY,
    row.image_width, row.image_height,
    SHOULDER_OPTS,
  );
}

/** Public: hip disc from a PoseRow. Returns [] if no source has anchors. */
export function buildHipDisc(row: PoseRow): OverlayElement[] {
  const pair = getHipPair(row);
  if (pair === null) return [];
  return buildDisc(
    pair.leftX, pair.leftY, pair.rightX, pair.rightY,
    row.image_width, row.image_height,
    HIP_OPTS,
  );
}

/**
 * Build a 1..5 frame OverlayTimeline from pose_3d_phases rows.
 *
 * Skips:
 *   - rows with fal_status !== 'completed'
 *   - rows where image_width or image_height is 0/missing (warn)
 *
 * Frame time priority:
 *   1. row.frame_timestamp_ms (preferred — exact frame timestamp)
 *   2. phaseMarkers[`${phase}Time`] (fallback — analyzer phase detector)
 */
export function generateSparsePhaseOverlayTimeline(input: {
  poseRows: PoseRow[];
  phaseMarkers: PhaseMarkers;
}): OverlayTimeline {
  _uid = 0;
  const { poseRows, phaseMarkers } = input;

  const frames: OverlayFrame[] = [];

  // Defensive sort — caller is expected to provide canonical order but cost
  // is negligible and protects against direct callers in the future.
  const rank = (p: PhaseName) => PHASE_ORDER.indexOf(p);
  const sorted = [...poseRows].sort((a, b) => rank(a.phase_name) - rank(b.phase_name));

  for (const row of sorted) {
    if (row.fal_status !== 'completed') continue;

    if (!row.image_width || !row.image_height) {
      console.warn(
        `[sparse] phase ${row.phase_name}: image_width/height invalid ` +
        `(${row.image_width}×${row.image_height}); skipping`,
      );
      continue;
    }

    const phase = row.phase_name;
    const phaseKey = `${phase}Time` as keyof PhaseMarkers;
    const time = row.frame_timestamp_ms !== null
      ? row.frame_timestamp_ms / 1000
      : phaseMarkers[phaseKey];

    const elements: OverlayElement[] = [
      ...buildShoulderDisc(row),
      ...buildHipDisc(row),
    ];

    if (elements.length === 0) {
      console.warn(`[sparse] phase ${phase}: no discs built (all anchors null)`);
      continue;
    }

    frames.push({ time, phase, elements });
  }

  return { frames };
}
