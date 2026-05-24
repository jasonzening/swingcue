'use client';

/**
 * AnchorTuningPanel — PR-7c-frontend-v8.1 self-service ratio tuning.
 *
 * Renders ONLY when SwingPlayer sees `?tune=anchors` in the URL.
 *
 * v8 → v8.1 changes (per-anchor decoupling + tuning flow improvements):
 *   - 4 coupled ratios → 10 per-anchor independent ratios
 *     (L_SHOULDER UP/OUT, R_SHOULDER UP/OUT, L_HIP UP/OUT, R_HIP UP/OUT,
 *      HEAD UP/OUT)
 *   - Head sliders are BIPOLAR (-0.10 to +0.10) — direct signed shift,
 *     no out_sign computation. Avoids the proj-near-0 wobble bug.
 *   - 5-button phase stepper jumps videoEl.currentTime to setup / top /
 *     transition / impact / finish timestamps. Validates ratios across
 *     phases without manual scrubbing.
 *   - Collapse chevron in header shrinks panel to title bar only.
 *   - Width 280 → 320 to fit grouped layout.
 *
 * Goals:
 *   - Validate ratios at multiple swing phases (DTL perspective shifts
 *     with body rotation; one-phase tuning misses other-phase drift).
 *   - Per-anchor independence: near vs far shoulder in DTL view need
 *     different shifts. Coupled tuning was a footgun.
 *
 * Production impact: zero. Lazy-loaded by SwingPlayer via `next/dynamic`,
 * so the chunk is only fetched when the URL param is present.
 */

import { useEffect, useRef, useState, type ChangeEvent, type CSSProperties } from 'react';
import type { PhaseMarkers, PoseTimeline } from '@/types/analysis';
import { getCurrentPhase } from '@/lib/overlay/playerSync';
import {
  poseRawAnchorsAtTime,
  computeVisualAnchors,
} from '@/lib/coaching/poseTimelineAnchors';

type Ratios = {
  LEFT_SHOULDER_UP:    number;  LEFT_SHOULDER_OUT:   number;
  RIGHT_SHOULDER_UP:   number;  RIGHT_SHOULDER_OUT:  number;
  LEFT_HIP_UP:         number;  LEFT_HIP_OUT:        number;
  RIGHT_HIP_UP:        number;  RIGHT_HIP_OUT:       number;
  HEAD_UP:             number;  HEAD_OUT:            number;
};

type RatioKey = keyof Ratios;

type Props = {
  videoEl: HTMLVideoElement | null;
  poseTimeline: PoseTimeline | null | undefined;
  phaseMarkers: PhaseMarkers;
  durationSec: number;
  ratios: Ratios;
  onRatiosChange: (next: Ratios) => void;
};

// Paired (positive-only) sliders: 0 to 0.25, step 0.005.
const PAIRED_MIN = 0;
const PAIRED_MAX = 0.25;
const PAIRED_STEP = 0.005;

// Head sliders are BIPOLAR: -0.25 to +0.25 (extended from v8.1's ±0.10
// after Jason hit max during setup tuning). Negative = down/one-side,
// positive = up/other-side. Center 0.0 = nose direct.
const HEAD_MIN = -0.25;
const HEAD_MAX = 0.25;
const HEAD_STEP = 0.005;

/** Frame display info — null fields when video/pose isn't ready yet. */
type DebugFrame = {
  frameIdx: number | null;
  totalFrames: number | null;
  phase: string;
  spineLen: number | null;
  anchors: Record<
    'left_shoulder' | 'right_shoulder' | 'left_hip' | 'right_hip' | 'head',
    { raw: readonly [number, number, number] | null; shifted: readonly [number, number] | null }
  > | null;
};

const EMPTY_DEBUG: DebugFrame = {
  frameIdx: null,
  totalFrames: null,
  phase: '—',
  spineLen: null,
  anchors: null,
};

type PhaseButton = {
  key: string;
  label: string;
  ts: number;
};

export function AnchorTuningPanel({
  videoEl,
  poseTimeline,
  phaseMarkers,
  durationSec,
  ratios,
  onRatiosChange,
}: Props) {
  const [debug, setDebug] = useState<DebugFrame>(EMPTY_DEBUG);
  const [copied, setCopied] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const ratiosRef = useRef(ratios);
  ratiosRef.current = ratios;

  useEffect(() => {
    if (!videoEl || !poseTimeline) {
      setDebug(EMPTY_DEBUG);
      return;
    }

    const totalFrames = poseTimeline.frames.length;
    let raf = 0;

    const tick = () => {
      const t = videoEl.currentTime;
      const safeDur =
        Number.isFinite(durationSec) && durationSec > 0 ? durationSec : 1;
      const phase = getCurrentPhase(phaseMarkers, t, safeDur);
      const raw = poseRawAnchorsAtTime(poseTimeline, t);

      if (!raw) {
        setDebug({
          frameIdx: null,
          totalFrames,
          phase,
          spineLen: null,
          anchors: null,
        });
        raf = requestAnimationFrame(tick);
        return;
      }

      const visual = computeVisualAnchors(raw, ratiosRef.current);

      const sh = raw.left_shoulder.xy && raw.right_shoulder.xy
        ? Math.hypot(
            ((raw.left_hip.xy?.[0] ?? 0) + (raw.right_hip.xy?.[0] ?? 0)) / 2
              - (raw.left_shoulder.xy[0] + raw.right_shoulder.xy[0]) / 2,
            ((raw.left_hip.xy?.[1] ?? 0) + (raw.right_hip.xy?.[1] ?? 0)) / 2
              - (raw.left_shoulder.xy[1] + raw.right_shoulder.xy[1]) / 2,
          )
        : null;

      const frameIdx = findClosestFrameIdx(poseTimeline, t);

      const pack = (
        rawKp: { xy: readonly [number, number] | null; confidence: number },
        shiftedXY: readonly [number, number] | null,
      ) => ({
        raw: rawKp.xy
          ? ([rawKp.xy[0], rawKp.xy[1], rawKp.confidence] as const)
          : null,
        shifted: shiftedXY,
      });

      setDebug({
        frameIdx,
        totalFrames,
        phase,
        spineLen: sh,
        anchors: {
          left_shoulder:  pack(raw.left_shoulder,  visual.left_shoulder.xy),
          right_shoulder: pack(raw.right_shoulder, visual.right_shoulder.xy),
          left_hip:       pack(raw.left_hip,       visual.left_hip.xy),
          right_hip:      pack(raw.right_hip,      visual.right_hip.xy),
          head:           pack(raw.nose,           visual.head.xy),
        },
      });
      raf = requestAnimationFrame(tick);
    };

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [videoEl, poseTimeline, phaseMarkers, durationSec]);

  // Header — always visible, even when collapsed.
  const renderHeader = () => (
    <div style={headerRowStyle}>
      <span style={titleStyle}>ANCHOR TUNING</span>
      <button
        onClick={() => setCollapsed(!collapsed)}
        style={chevronBtnStyle}
        title={collapsed ? 'Expand' : 'Collapse'}
      >
        {collapsed ? '+' : '−'}
      </button>
    </div>
  );

  if (!poseTimeline) {
    return (
      <div className="atp-panel" style={panelStyle}>
        {renderHeader()}
        {!collapsed && (
          <div style={noDataStyle}>
            No pose data on this video — tuning unavailable.
          </div>
        )}
      </div>
    );
  }

  if (collapsed) {
    return (
      <div className="atp-panel" style={panelStyleCollapsed}>
        {renderHeader()}
      </div>
    );
  }

  const handleSlider = (key: RatioKey) =>
    (e: ChangeEvent<HTMLInputElement>) => {
      const next = parseFloat(e.target.value);
      onRatiosChange({ ...ratios, [key]: next });
    };

  const handleCopy = async () => {
    const fmt = (n: number) => n.toFixed(3);

    // v8.1.1: extract video_id from /result/[id] URL pathname.
    // Panel is lazy-loaded via next/dynamic with ssr:false so window
    // is always defined when handleCopy fires.
    const videoId =
      typeof window !== 'undefined'
        ? window.location.pathname.split('/').filter(Boolean).pop() ?? 'unknown'
        : 'unknown';
    const t = videoEl ? videoEl.currentTime : 0;

    // Per-anchor lines: raw "x.x, y.y" with 1-decimal MediaPipe precision +
    // conf, then "→ x, y" with rounded tuned coords. Head shows "nose" prefix
    // so its raw source is unambiguous.
    const anchorLine = (
      name: string,
      pair: DebugFrame['anchors'] extends Record<string, infer V> | null ? V : never,
      headNoseLabel = false,
    ): string => {
      const raw = pair.raw
        ? `(${pair.raw[0].toFixed(1)}, ${pair.raw[1].toFixed(1)}) conf=${pair.raw[2].toFixed(2)}`
        : '—';
      const shifted = pair.shifted
        ? `(${Math.round(pair.shifted[0])}, ${Math.round(pair.shifted[1])})`
        : '—';
      const prefix = headNoseLabel ? 'nose ' : '';
      return `//   ${name.padEnd(6)} ${prefix}${raw}  →  ${shifted}`;
    };

    const a = debug.anchors;
    const anchorBlock = a
      ? [
          anchorLine('L_sh',  a.left_shoulder),
          anchorLine('R_sh',  a.right_shoulder),
          anchorLine('L_hip', a.left_hip),
          anchorLine('R_hip', a.right_hip),
          anchorLine('Head',  a.head, true),
        ].join('\n')
      : '//   (no pose data at this frame)';

    const text =
      '// ────────────────────────────────────────────────\n'
      + '// SwingCue Anchor Tuning Snapshot\n'
      + '// ────────────────────────────────────────────────\n'
      + `// video_id:  ${videoId}\n`
      + `// frame:     ${debug.frameIdx ?? '—'} / ${debug.totalFrames ?? '—'}  (t=${t.toFixed(3)}s)\n`
      + `// phase:     ${debug.phase}\n`
      + `// spine_len: ${debug.spineLen != null ? Math.round(debug.spineLen) + ' px' : '—'}\n`
      + '//\n'
      + '// Anchor positions at this frame (raw → v8 tuned):\n'
      + anchorBlock + '\n'
      + '//\n'
      + '// Tuned ratios (paste into VISUAL_ANCHOR_CONFIG):\n'
      + `  LEFT_SHOULDER_UP:    ${fmt(ratios.LEFT_SHOULDER_UP)},\n`
      + `  LEFT_SHOULDER_OUT:   ${fmt(ratios.LEFT_SHOULDER_OUT)},\n`
      + `  RIGHT_SHOULDER_UP:   ${fmt(ratios.RIGHT_SHOULDER_UP)},\n`
      + `  RIGHT_SHOULDER_OUT:  ${fmt(ratios.RIGHT_SHOULDER_OUT)},\n`
      + `  LEFT_HIP_UP:         ${fmt(ratios.LEFT_HIP_UP)},\n`
      + `  LEFT_HIP_OUT:        ${fmt(ratios.LEFT_HIP_OUT)},\n`
      + `  RIGHT_HIP_UP:        ${fmt(ratios.RIGHT_HIP_UP)},\n`
      + `  RIGHT_HIP_OUT:       ${fmt(ratios.RIGHT_HIP_OUT)},\n`
      + `  HEAD_UP:             ${fmt(ratios.HEAD_UP)},\n`
      + `  HEAD_OUT:            ${fmt(ratios.HEAD_OUT)},\n`;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      window.prompt('Copy these values:', text);
    }
  };

  // Phase stepper buttons read timestamps from phaseMarkers and seek
  // videoEl.currentTime on click. Active phase (matches debug.phase)
  // gets a magenta highlight.
  const phaseButtons: PhaseButton[] = [
    { key: 'setup',      label: 'SETUP',  ts: phaseMarkers.setupTime },
    { key: 'top',        label: 'TOP',    ts: phaseMarkers.topTime },
    { key: 'transition', label: 'TRANS',  ts: phaseMarkers.transitionTime },
    { key: 'impact',     label: 'IMPACT', ts: phaseMarkers.impactTime },
    { key: 'finish',     label: 'FINISH', ts: phaseMarkers.finishTime },
  ];

  const handlePhaseClick = (ts: number) => {
    if (videoEl) videoEl.currentTime = ts;
  };

  return (
    <div className="atp-panel" style={panelStyle}>
      {renderHeader()}

      <SectionLabel label="L SHOULDER" />
      <SliderRow label="UP"  value={ratios.LEFT_SHOULDER_UP}  onChange={handleSlider('LEFT_SHOULDER_UP')}  min={PAIRED_MIN} max={PAIRED_MAX} step={PAIRED_STEP} />
      <SliderRow label="OUT" value={ratios.LEFT_SHOULDER_OUT} onChange={handleSlider('LEFT_SHOULDER_OUT')} min={PAIRED_MIN} max={PAIRED_MAX} step={PAIRED_STEP} />

      <SectionLabel label="R SHOULDER" />
      <SliderRow label="UP"  value={ratios.RIGHT_SHOULDER_UP}  onChange={handleSlider('RIGHT_SHOULDER_UP')}  min={PAIRED_MIN} max={PAIRED_MAX} step={PAIRED_STEP} />
      <SliderRow label="OUT" value={ratios.RIGHT_SHOULDER_OUT} onChange={handleSlider('RIGHT_SHOULDER_OUT')} min={PAIRED_MIN} max={PAIRED_MAX} step={PAIRED_STEP} />

      <SectionLabel label="L HIP" />
      <SliderRow label="UP"  value={ratios.LEFT_HIP_UP}  onChange={handleSlider('LEFT_HIP_UP')}  min={PAIRED_MIN} max={PAIRED_MAX} step={PAIRED_STEP} />
      <SliderRow label="OUT" value={ratios.LEFT_HIP_OUT} onChange={handleSlider('LEFT_HIP_OUT')} min={PAIRED_MIN} max={PAIRED_MAX} step={PAIRED_STEP} />

      <SectionLabel label="R HIP" />
      <SliderRow label="UP"  value={ratios.RIGHT_HIP_UP}  onChange={handleSlider('RIGHT_HIP_UP')}  min={PAIRED_MIN} max={PAIRED_MAX} step={PAIRED_STEP} />
      <SliderRow label="OUT" value={ratios.RIGHT_HIP_OUT} onChange={handleSlider('RIGHT_HIP_OUT')} min={PAIRED_MIN} max={PAIRED_MAX} step={PAIRED_STEP} />

      <SectionLabel label="HEAD  (bipolar ±0.25)" />
      <SliderRow label="UP"  value={ratios.HEAD_UP}  onChange={handleSlider('HEAD_UP')}  min={HEAD_MIN} max={HEAD_MAX} step={HEAD_STEP} bipolar />
      <SliderRow label="OUT" value={ratios.HEAD_OUT} onChange={handleSlider('HEAD_OUT')} min={HEAD_MIN} max={HEAD_MAX} step={HEAD_STEP} bipolar />

      <div style={separatorStyle} />

      <div style={debugRowStyle}>
        <span>Frame</span>
        <span style={debugValStyle}>
          {debug.frameIdx ?? '—'}
          {debug.totalFrames != null ? ` / ${debug.totalFrames}` : ''}
        </span>
      </div>
      <div style={debugRowStyle}>
        <span>Spine</span>
        <span style={debugValStyle}>
          {debug.spineLen != null ? `${Math.round(debug.spineLen)} px` : '—'}
        </span>
      </div>
      <div style={debugRowStyle}>
        <span>Phase</span>
        <span style={debugValStyle}>{debug.phase}</span>
      </div>

      <div style={phaseStepperStyle}>
        {phaseButtons.map((pb) => {
          const isActive = pb.key === debug.phase;
          return (
            <button
              key={pb.key}
              onClick={() => handlePhaseClick(pb.ts)}
              style={isActive ? phaseBtnActiveStyle : phaseBtnStyle}
              title={`Jump to t=${pb.ts.toFixed(2)}s`}
            >
              {pb.label}
            </button>
          );
        })}
      </div>

      <div style={separatorThinStyle} />

      {debug.anchors ? (
        <div style={anchorsBlockStyle}>
          <AnchorRow name="L_sh"  pair={debug.anchors.left_shoulder} />
          <AnchorRow name="R_sh"  pair={debug.anchors.right_shoulder} />
          <AnchorRow name="L_hip" pair={debug.anchors.left_hip} />
          <AnchorRow name="R_hip" pair={debug.anchors.right_hip} />
          <AnchorRow name="Head"  pair={debug.anchors.head} />
        </div>
      ) : (
        <div style={noDataStyle}>No anchors at current time</div>
      )}

      <div style={separatorStyle} />

      <button onClick={handleCopy} style={copyBtnStyle}>
        {copied ? 'COPIED!' : 'COPY VISUAL_ANCHOR_CONFIG'}
      </button>
    </div>
  );
}

// ── Subcomponents ──────────────────────────────────────────────────

function SectionLabel({ label }: { label: string }) {
  return <div style={sectionLabelStyle}>{label}</div>;
}

function SliderRow({
  label, value, onChange, min, max, step, bipolar,
}: {
  label: string;
  value: number;
  onChange: (e: ChangeEvent<HTMLInputElement>) => void;
  min: number;
  max: number;
  step: number;
  bipolar?: boolean;
}) {
  return (
    <div style={sliderRowStyle}>
      <div style={sliderLabelRowStyle}>
        <span style={sliderLabelStyle}>{label}</span>
        <span style={sliderValueStyle}>
          {bipolar && value > 0 ? '+' : ''}{value.toFixed(3)}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={onChange}
        style={sliderInputStyle}
      />
    </div>
  );
}

function AnchorRow({
  name, pair,
}: {
  name: string;
  pair: { raw: readonly [number, number, number] | null; shifted: readonly [number, number] | null };
}) {
  return (
    <div style={anchorRowStyle}>
      <span style={anchorNameStyle}>{name}</span>
      <span style={debugValStyle}>
        {pair.raw
          ? `${Math.round(pair.raw[0])},${Math.round(pair.raw[1])} (${pair.raw[2].toFixed(2)})`
          : '—'}
        {' → '}
        {pair.shifted
          ? `${Math.round(pair.shifted[0])},${Math.round(pair.shifted[1])}`
          : '—'}
      </span>
    </div>
  );
}

// ── Helpers ────────────────────────────────────────────────────────

function findClosestFrameIdx(timeline: PoseTimeline, t: number): number {
  const frames = timeline.frames;
  if (frames.length === 0) return 0;
  let bestIdx = 0;
  let bestDelta = Math.abs(frames[0].ts - t);
  for (let i = 1; i < frames.length; i++) {
    const d = Math.abs(frames[i].ts - t);
    if (d < bestDelta) {
      bestDelta = d;
      bestIdx = i;
    }
  }
  return bestIdx;
}

// ── Inline styles ──────────────────────────────────────────────────

const panelStyle: CSSProperties = {
  position: 'absolute',
  top: 16,
  right: 16,
  width: 320,
  maxHeight: 'calc(100% - 32px)',
  overflowY: 'auto',
  zIndex: 10,
  background: 'rgba(0, 0, 0, 0.82)',
  border: '1px solid rgba(255, 0, 255, 0.4)',
  borderRadius: 6,
  padding: 12,
  color: '#fff',
  fontFamily: 'ui-monospace, SFMono-Regular, "Menlo", monospace',
  fontSize: 11,
  lineHeight: 1.4,
  pointerEvents: 'auto',
  boxShadow: '0 4px 16px rgba(0, 0, 0, 0.6)',
};

const panelStyleCollapsed: CSSProperties = {
  ...panelStyle,
  maxHeight: 'none',
  overflowY: 'visible',
};

const headerRowStyle: CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  marginBottom: 8,
};

const titleStyle: CSSProperties = {
  fontSize: 12,
  fontWeight: 700,
  letterSpacing: 1,
  color: '#FF00FF',
};

const chevronBtnStyle: CSSProperties = {
  background: 'transparent',
  border: '1px solid rgba(255,0,255,0.4)',
  color: '#FF00FF',
  width: 22,
  height: 22,
  borderRadius: 4,
  cursor: 'pointer',
  fontSize: 14,
  fontWeight: 700,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  fontFamily: 'inherit',
  padding: 0,
  lineHeight: 1,
};

const sectionLabelStyle: CSSProperties = {
  fontSize: 10,
  fontWeight: 700,
  letterSpacing: 0.8,
  color: 'rgba(255,255,255,0.65)',
  marginTop: 8,
  marginBottom: 4,
  textTransform: 'uppercase',
};

const noDataStyle: CSSProperties = {
  color: 'rgba(255,255,255,0.5)',
  fontStyle: 'italic',
  padding: '6px 0',
};

const sliderRowStyle: CSSProperties = {
  marginBottom: 5,
  paddingLeft: 8,
};

const sliderLabelRowStyle: CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'baseline',
  marginBottom: 1,
};

const sliderLabelStyle: CSSProperties = {
  fontSize: 10,
  color: 'rgba(255,255,255,0.7)',
  fontWeight: 600,
};

const sliderValueStyle: CSSProperties = {
  fontSize: 11,
  color: '#FF00FF',
  fontWeight: 600,
  fontVariantNumeric: 'tabular-nums',
};

const sliderInputStyle: CSSProperties = {
  width: '100%',
  accentColor: '#FF00FF',
  cursor: 'pointer',
  height: 14,
};

const separatorStyle: CSSProperties = {
  height: 1,
  background: 'rgba(255,255,255,0.15)',
  margin: '10px 0',
};

const separatorThinStyle: CSSProperties = {
  height: 1,
  background: 'rgba(255,255,255,0.08)',
  margin: '6px 0',
};

const debugRowStyle: CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'baseline',
  marginBottom: 2,
};

const debugValStyle: CSSProperties = {
  color: 'rgba(255,255,255,0.95)',
  fontVariantNumeric: 'tabular-nums',
};

const phaseStepperStyle: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(5, 1fr)',
  gap: 3,
  marginTop: 6,
};

const phaseBtnStyle: CSSProperties = {
  padding: '4px 0',
  background: 'rgba(255,255,255,0.08)',
  color: 'rgba(255,255,255,0.7)',
  border: '1px solid rgba(255,255,255,0.15)',
  borderRadius: 3,
  fontSize: 9,
  fontWeight: 600,
  fontFamily: 'inherit',
  cursor: 'pointer',
  letterSpacing: 0.3,
};

const phaseBtnActiveStyle: CSSProperties = {
  ...phaseBtnStyle,
  background: 'rgba(255, 0, 255, 0.25)',
  color: '#FF00FF',
  borderColor: 'rgba(255, 0, 255, 0.6)',
};

const anchorsBlockStyle: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 2,
};

const anchorRowStyle: CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  fontSize: 10,
};

const anchorNameStyle: CSSProperties = {
  color: '#FF00FF',
  fontWeight: 600,
  marginRight: 6,
};

const copyBtnStyle: CSSProperties = {
  width: '100%',
  padding: '8px 12px',
  background: 'rgba(255, 0, 255, 0.18)',
  color: '#FF00FF',
  border: '1px solid rgba(255, 0, 255, 0.5)',
  borderRadius: 4,
  fontSize: 12,
  fontWeight: 600,
  fontFamily: 'inherit',
  letterSpacing: 0.5,
  cursor: 'pointer',
};
