'use client';

/**
 * AnchorTuningPanel — PR-7c-frontend-v8 self-service ratio tuning.
 *
 * Renders ONLY when SwingPlayer sees `?tune=anchors` in the URL.
 * Provides 4 live sliders for VISUAL_ANCHOR_CONFIG ratios plus a
 * debug readout of the current frame's raw vs shifted anchors.
 * Click "Copy" to get a paste-ready VISUAL_ANCHOR_CONFIG block on
 * the clipboard.
 *
 * Goals:
 *   - Replace the v3→v7 cycle (guess→push→screenshot→describe→retune)
 *     with direct in-browser visual feedback.
 *   - Stateless sandbox: no localStorage, no DB. Once tuned, Jason
 *     copies values to chat, Claude commits final to VISUAL_ANCHOR_CONFIG,
 *     ships v8.1 in one push.
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
  SHOULDER_UP_RATIO:  number;
  SHOULDER_OUT_RATIO: number;
  HIP_UP_RATIO:       number;
  HIP_OUT_RATIO:      number;
};

type Props = {
  videoEl: HTMLVideoElement | null;
  poseTimeline: PoseTimeline | null;
  phaseMarkers: PhaseMarkers;
  durationSec: number;
  ratios: Ratios;
  onRatiosChange: (next: Ratios) => void;
};

const SLIDER_MIN = 0;
const SLIDER_MAX = 0.25;
const SLIDER_STEP = 0.005;

/** Frame display info — null fields when video/pose isn't ready yet. */
type DebugFrame = {
  frameIdx: number | null;
  totalFrames: number | null;
  phase: string;
  spineLen: number | null;
  // Per-anchor: raw [x,y,conf] and shifted [x,y] (head shifted = raw nose)
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
  const ratiosRef = useRef(ratios);
  ratiosRef.current = ratios;

  // Live frame readout. Runs its own rAF (independent of the overlay's)
  // so the panel updates even when the video is paused at a specific
  // phase — useful for tuning frame-by-frame.
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

      // Frame idx — best-effort. Use timestamp lookup if available; else
      // estimate by linear scan of the frames array.
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

  if (!poseTimeline) {
    return (
      <div className="atp-panel" style={panelStyle}>
        <div style={titleStyle}>ANCHOR TUNING</div>
        <div style={noDataStyle}>
          No pose data on this video — tuning unavailable.
        </div>
      </div>
    );
  }

  const handleSlider = (key: keyof Ratios) =>
    (e: ChangeEvent<HTMLInputElement>) => {
      const next = parseFloat(e.target.value);
      onRatiosChange({ ...ratios, [key]: next });
    };

  const handleCopy = async () => {
    const text =
      '// Tuned values (paste into VISUAL_ANCHOR_CONFIG):\n'
      + `  SHOULDER_UP_RATIO:  ${ratios.SHOULDER_UP_RATIO.toFixed(3)},\n`
      + `  SHOULDER_OUT_RATIO: ${ratios.SHOULDER_OUT_RATIO.toFixed(3)},\n`
      + `  HIP_UP_RATIO:       ${ratios.HIP_UP_RATIO.toFixed(3)},\n`
      + `  HIP_OUT_RATIO:      ${ratios.HIP_OUT_RATIO.toFixed(3)},\n`;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard API may fail in iframe / non-HTTPS. Fallback: select-by-prompt.
      window.prompt('Copy these values:', text);
    }
  };

  return (
    <div className="atp-panel" style={panelStyle}>
      <div style={titleStyle}>ANCHOR TUNING</div>

      <SliderRow label="SHOULDER_UP"   value={ratios.SHOULDER_UP_RATIO}   onChange={handleSlider('SHOULDER_UP_RATIO')}   />
      <SliderRow label="SHOULDER_OUT"  value={ratios.SHOULDER_OUT_RATIO}  onChange={handleSlider('SHOULDER_OUT_RATIO')}  />
      <SliderRow label="HIP_UP"        value={ratios.HIP_UP_RATIO}        onChange={handleSlider('HIP_UP_RATIO')}        />
      <SliderRow label="HIP_OUT"       value={ratios.HIP_OUT_RATIO}       onChange={handleSlider('HIP_OUT_RATIO')}       />

      <div style={separatorStyle} />

      <div style={debugBlockStyle}>
        <div style={debugRowStyle}>
          <span>Frame</span>
          <span style={debugValStyle}>
            {debug.frameIdx ?? '—'}
            {debug.totalFrames != null ? ` / ${debug.totalFrames}` : ''}
          </span>
        </div>
        <div style={debugRowStyle}>
          <span>Phase</span>
          <span style={debugValStyle}>{debug.phase}</span>
        </div>
        <div style={debugRowStyle}>
          <span>Spine</span>
          <span style={debugValStyle}>
            {debug.spineLen != null ? `${Math.round(debug.spineLen)} px` : '—'}
          </span>
        </div>

        <div style={separatorThinStyle} />

        {debug.anchors ? (
          <>
            <AnchorRow name="L_shoulder"  pair={debug.anchors.left_shoulder} />
            <AnchorRow name="R_shoulder"  pair={debug.anchors.right_shoulder} />
            <AnchorRow name="L_hip"       pair={debug.anchors.left_hip} />
            <AnchorRow name="R_hip"       pair={debug.anchors.right_hip} />
            <AnchorRow name="head (nose)" pair={debug.anchors.head} shiftedHidden />
          </>
        ) : (
          <div style={noDataStyle}>No anchors at current time</div>
        )}
      </div>

      <div style={separatorStyle} />

      <button onClick={handleCopy} style={copyBtnStyle}>
        {copied ? 'COPIED!' : 'COPY VISUAL_ANCHOR_CONFIG'}
      </button>
    </div>
  );
}

// ── Subcomponents ──────────────────────────────────────────────────

function SliderRow({
  label, value, onChange,
}: {
  label: string;
  value: number;
  onChange: (e: ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <div style={sliderRowStyle}>
      <div style={sliderLabelRowStyle}>
        <span style={sliderLabelStyle}>{label}</span>
        <span style={sliderValueStyle}>{value.toFixed(3)}</span>
      </div>
      <input
        type="range"
        min={SLIDER_MIN}
        max={SLIDER_MAX}
        step={SLIDER_STEP}
        value={value}
        onChange={onChange}
        style={sliderInputStyle}
      />
    </div>
  );
}

function AnchorRow({
  name, pair, shiftedHidden,
}: {
  name: string;
  pair: { raw: readonly [number, number, number] | null; shifted: readonly [number, number] | null };
  shiftedHidden?: boolean;
}) {
  return (
    <div style={anchorRowStyle}>
      <div style={anchorNameStyle}>{name}</div>
      <div style={anchorLineStyle}>
        <span style={anchorLabelStyle}>raw</span>
        <span style={debugValStyle}>
          {pair.raw
            ? `${Math.round(pair.raw[0])},${Math.round(pair.raw[1])} (${pair.raw[2].toFixed(2)})`
            : '—'}
        </span>
      </div>
      {!shiftedHidden && (
        <div style={anchorLineStyle}>
          <span style={anchorLabelStyle}>v7</span>
          <span style={debugValStyle}>
            {pair.shifted
              ? `${Math.round(pair.shifted[0])},${Math.round(pair.shifted[1])}`
              : '—'}
          </span>
        </div>
      )}
    </div>
  );
}

// ── Helpers ────────────────────────────────────────────────────────

/** Find the frame whose timestamp is closest to `t`. Linear scan —
 * this is debug-only code and pose timelines are <1000 frames. */
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

// ── Inline styles (single source, no separate CSS file) ────────────

const panelStyle: CSSProperties = {
  position: 'absolute',
  top: 16,
  right: 16,
  width: 280,
  maxHeight: 'calc(100% - 32px)',
  overflowY: 'auto',
  zIndex: 10,
  background: 'rgba(0, 0, 0, 0.78)',
  border: '1px solid rgba(255, 0, 255, 0.35)',
  borderRadius: 6,
  padding: 12,
  color: '#fff',
  fontFamily: 'ui-monospace, SFMono-Regular, "Menlo", monospace',
  fontSize: 11,
  lineHeight: 1.4,
  pointerEvents: 'auto',
  boxShadow: '0 4px 12px rgba(0, 0, 0, 0.5)',
};

const titleStyle: CSSProperties = {
  fontSize: 12,
  fontWeight: 700,
  letterSpacing: 1,
  color: '#FF00FF',
  marginBottom: 10,
  fontFamily: 'inherit',
};

const noDataStyle: CSSProperties = {
  color: 'rgba(255,255,255,0.5)',
  fontStyle: 'italic',
  padding: '6px 0',
};

const sliderRowStyle: CSSProperties = {
  marginBottom: 8,
};

const sliderLabelRowStyle: CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'baseline',
  marginBottom: 2,
};

const sliderLabelStyle: CSSProperties = {
  fontSize: 11,
  color: 'rgba(255,255,255,0.85)',
};

const sliderValueStyle: CSSProperties = {
  fontSize: 12,
  color: '#FF00FF',
  fontWeight: 600,
};

const sliderInputStyle: CSSProperties = {
  width: '100%',
  accentColor: '#FF00FF',
  cursor: 'pointer',
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

const debugBlockStyle: CSSProperties = {
  fontSize: 11,
};

const debugRowStyle: CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'baseline',
  marginBottom: 3,
};

const debugValStyle: CSSProperties = {
  color: 'rgba(255,255,255,0.95)',
  fontVariantNumeric: 'tabular-nums',
};

const anchorRowStyle: CSSProperties = {
  marginTop: 6,
};

const anchorNameStyle: CSSProperties = {
  fontSize: 11,
  color: '#FF00FF',
  fontWeight: 600,
  marginBottom: 1,
};

const anchorLineStyle: CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  paddingLeft: 8,
};

const anchorLabelStyle: CSSProperties = {
  color: 'rgba(255,255,255,0.55)',
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
