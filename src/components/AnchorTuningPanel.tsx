'use client';

/**
 * AnchorTuningPanel — PR-7c-frontend-v9 keyframe interpolation tuning.
 *
 * Renders ONLY when SwingPlayer sees `?tune=anchors` in the URL.
 *
 * v8.1 → v9 changes (keyframe architecture):
 *   - PHASE-INVARIANT ASSUMPTION FALSIFIED: Jason's 5-snapshot v8.1.1
 *     data showed ratios varying 3-8x across phases. Single-set tuning
 *     can't cover the swing.
 *   - New: per-video keyframe array. User saves N keyframes at chosen
 *     frame indices; production overlay lerps between them per frame.
 *   - REMOVED: 5-button phase stepper (DB phase data wrong for many
 *     videos — PR-3.1 backend bug deferred). Replaced with
 *     "Jump to frame [N] [GO]" input.
 *   - ADDED: keyframe list UI with [jump] [edit] [×] per row, plus
 *     "Save current frame as keyframe" + explicit "Load interp at
 *     current frame" button.
 *   - Slider ranges bumped: paired 0-0.25 → 0-0.40, head ±0.25 → ±0.40
 *     (Jason hit max on multiple values).
 *   - Copy format now outputs entire VIDEO_KEYFRAMES[video_id] array.
 *
 * Interaction model (Decision C from v9 surface):
 *   - Sliders do NOT auto-update on video scrub. Explicit user control.
 *   - "Load interp at current frame" button pulls keyframe-interpolated
 *     values into sliders (clears any unsaved drag state).
 *   - Slider drag updates overlay LIVE (matches v8.1 tune-mode UX).
 *   - Save button captures (currentFrame, currentRatios) as keyframe.
 *   - Save at existing frame_idx: REPLACES with confirm.
 *   - Edit button: loads ratios + seeks to that frame (both actions).
 *   - Delete button: instant, no confirm (Copy = backup).
 *
 * State source-of-truth:
 *   - keyframes[] lives in this component's React state
 *   - Initialized from VIDEO_KEYFRAMES[videoId] on mount (deep clone)
 *   - Production code reads from the in-source VIDEO_KEYFRAMES const;
 *     this panel's edits never round-trip to backend. Jason copies →
 *     pastes to chat → Claude commits to source.
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
  findClosestFrameIdx,
  getRatiosAtFrame,
  VIDEO_KEYFRAMES,
  DEFAULT_RATIOS,
  type AnchorKeyframe,
  type VisualAnchorConfig,
} from '@/lib/coaching/poseTimelineAnchors';

// The 10 ratios that sliders control. Mirror of VisualAnchorConfig minus
// the 2 meta fields (HEAD_USE_NOSE, MIN_BODY_AXIS_LEN_PX) which aren't
// user-tunable in the panel.
type Ratios = {
  LEFT_SHOULDER_UP:    number;  LEFT_SHOULDER_OUT:   number;
  RIGHT_SHOULDER_UP:   number;  RIGHT_SHOULDER_OUT:  number;
  LEFT_HIP_UP:         number;  LEFT_HIP_OUT:        number;
  RIGHT_HIP_UP:        number;  RIGHT_HIP_OUT:       number;
  HEAD_UP:             number;  HEAD_OUT:            number;
};

type RatioKey = keyof Ratios;

type Props = {
  videoId: string;
  videoEl: HTMLVideoElement | null;
  poseTimeline: PoseTimeline | null | undefined;
  phaseMarkers: PhaseMarkers;
  durationSec: number;
  ratios: Ratios;
  onRatiosChange: (next: Ratios) => void;
};

// v9.1: bumped from v9's 0.40 paired range — Jason still hit max
// during cross-phase tuning.
const PAIRED_MIN = 0;
const PAIRED_MAX = 0.60;
const PAIRED_STEP = 0.005;

// v9.1: bumped from v9's ±0.40 head range — same reason.
const HEAD_MIN = -0.60;
const HEAD_MAX = 0.60;
const HEAD_STEP = 0.005;

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

/** Strip the 2 meta fields from a VisualAnchorConfig to get just the
 * 10 user-tunable ratios. */
function configToRatios(c: VisualAnchorConfig): Ratios {
  return {
    LEFT_SHOULDER_UP: c.LEFT_SHOULDER_UP,   LEFT_SHOULDER_OUT: c.LEFT_SHOULDER_OUT,
    RIGHT_SHOULDER_UP: c.RIGHT_SHOULDER_UP, RIGHT_SHOULDER_OUT: c.RIGHT_SHOULDER_OUT,
    LEFT_HIP_UP: c.LEFT_HIP_UP,             LEFT_HIP_OUT: c.LEFT_HIP_OUT,
    RIGHT_HIP_UP: c.RIGHT_HIP_UP,           RIGHT_HIP_OUT: c.RIGHT_HIP_OUT,
    HEAD_UP: c.HEAD_UP,                     HEAD_OUT: c.HEAD_OUT,
  };
}

/** Build a full VisualAnchorConfig from panel ratios + the 2 meta
 * fields (carried over from DEFAULT_RATIOS — panel doesn't tune them). */
function ratiosToConfig(r: Ratios): VisualAnchorConfig {
  return {
    ...r,
    HEAD_USE_NOSE: DEFAULT_RATIOS.HEAD_USE_NOSE,
    MIN_BODY_AXIS_LEN_PX: DEFAULT_RATIOS.MIN_BODY_AXIS_LEN_PX,
  };
}

export function AnchorTuningPanel({
  videoId,
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
  const [jumpInput, setJumpInput] = useState('');

  // v9 keyframe state: deep-cloned from VIDEO_KEYFRAMES[videoId] on
  // mount so panel edits never mutate the production const.
  const [keyframes, setKeyframes] = useState<AnchorKeyframe[]>(() => {
    const source = VIDEO_KEYFRAMES[videoId] ?? [];
    return source.map((kf) => ({
      frame_idx: kf.frame_idx,
      ratios: { ...kf.ratios },
    }));
  });

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

      const frameIdx = findClosestFrameIdx(poseTimeline, t);
      // Debug overlay in panel uses live slider override (matches what
      // the production overlay would render in tune mode at this frame).
      const visual = computeVisualAnchors(
        raw,
        frameIdx,
        videoId,
        ratiosToConfig(ratiosRef.current),
      );

      const sh = raw.left_shoulder.xy && raw.right_shoulder.xy
        ? Math.hypot(
            ((raw.left_hip.xy?.[0] ?? 0) + (raw.right_hip.xy?.[0] ?? 0)) / 2
              - (raw.left_shoulder.xy[0] + raw.right_shoulder.xy[0]) / 2,
            ((raw.left_hip.xy?.[1] ?? 0) + (raw.right_hip.xy?.[1] ?? 0)) / 2
              - (raw.left_shoulder.xy[1] + raw.right_shoulder.xy[1]) / 2,
          )
        : null;

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
  }, [videoEl, poseTimeline, phaseMarkers, durationSec, videoId]);

  // Header — always visible.
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

  // ── Frame navigation ───────────────────────────────────────────

  /** v9.1: frame_idx → time fallback chain. Priority order:
   *   1. frame.ts directly from poseTimeline.frames (always present
   *      and correct in production data)
   *   2. fps_sampled (= 30 for current production videos)
   *   3. fps_native (often null in DB for older videos)
   *   4. Hardcoded 30 fps default
   *
   * Defensive — current production data always has frame.ts so path
   * 1 wins. Fallbacks exist in case of future schema changes or older
   * timelines missing ts. */
  const frameIdxToTime = (frameIdx: number): number | null => {
    if (!poseTimeline?.frames) return null;
    if (frameIdx < 0 || frameIdx >= poseTimeline.frames.length) return null;
    const frame = poseTimeline.frames[frameIdx];
    if (typeof frame?.ts === 'number') return frame.ts;
    const fps = poseTimeline.fps_sampled;
    if (typeof fps === 'number' && fps > 0) return frameIdx / fps;
    return frameIdx / 30;
  };

  /** v9.1: hardened seek — pause first (avoid racing playback), set
   * currentTime, attach a no-op 'seeked' listener to nudge browsers
   * that defer repaint on paused-video seek. */
  const seekToFrame = (idx: number) => {
    if (!videoEl || !poseTimeline) return;
    const clamped = Math.max(0, Math.min(idx, poseTimeline.frames.length - 1));
    const t = frameIdxToTime(clamped);
    if (t == null) return;
    videoEl.pause();
    videoEl.addEventListener('seeked', () => { /* force frame commit */ }, { once: true });
    videoEl.currentTime = t;
  };

  const handleJumpGo = () => {
    const n = parseInt(jumpInput, 10);
    if (Number.isFinite(n)) seekToFrame(n);
  };

  const handleLoadInterp = () => {
    if (debug.frameIdx == null) return;
    const interpConfig = getRatiosAtFrame(debug.frameIdx, keyframes);
    onRatiosChange(configToRatios(interpConfig));
  };

  // ── Keyframe operations ────────────────────────────────────────

  const handleSaveKeyframe = () => {
    if (debug.frameIdx == null) return;
    const newConfig = ratiosToConfig(ratios);
    const existingIdx = keyframes.findIndex((k) => k.frame_idx === debug.frameIdx);
    if (existingIdx >= 0) {
      const confirmed = window.confirm(
        `Replace keyframe at frame ${debug.frameIdx} with current slider values?`,
      );
      if (!confirmed) return;
      const next = [...keyframes];
      next[existingIdx] = { frame_idx: debug.frameIdx, ratios: newConfig };
      setKeyframes(next);
    } else {
      setKeyframes(
        [...keyframes, { frame_idx: debug.frameIdx, ratios: newConfig }]
          .sort((a, b) => a.frame_idx - b.frame_idx),
      );
    }
  };

  const handleEditKeyframe = (frameIdx: number) => {
    const kf = keyframes.find((k) => k.frame_idx === frameIdx);
    if (!kf) return;
    onRatiosChange(configToRatios(kf.ratios));
    seekToFrame(frameIdx);
  };

  const handleDeleteKeyframe = (frameIdx: number) => {
    setKeyframes(keyframes.filter((k) => k.frame_idx !== frameIdx));
  };

  // ── Copy: full VIDEO_KEYFRAMES[videoId] array ─────────────────

  const handleCopy = async () => {
    const fmt = (n: number) => n.toFixed(3);
    const sorted = [...keyframes].sort((a, b) => a.frame_idx - b.frame_idx);
    const ratiosBlock = (r: VisualAnchorConfig) =>
      [
        `      LEFT_SHOULDER_UP:    ${fmt(r.LEFT_SHOULDER_UP)},`,
        `      LEFT_SHOULDER_OUT:   ${fmt(r.LEFT_SHOULDER_OUT)},`,
        `      RIGHT_SHOULDER_UP:   ${fmt(r.RIGHT_SHOULDER_UP)},`,
        `      RIGHT_SHOULDER_OUT:  ${fmt(r.RIGHT_SHOULDER_OUT)},`,
        `      LEFT_HIP_UP:         ${fmt(r.LEFT_HIP_UP)},`,
        `      LEFT_HIP_OUT:        ${fmt(r.LEFT_HIP_OUT)},`,
        `      RIGHT_HIP_UP:        ${fmt(r.RIGHT_HIP_UP)},`,
        `      RIGHT_HIP_OUT:       ${fmt(r.RIGHT_HIP_OUT)},`,
        `      HEAD_UP:             ${fmt(r.HEAD_UP)},`,
        `      HEAD_OUT:            ${fmt(r.HEAD_OUT)},`,
      ].join('\n');
    const body = sorted
      .map(
        (kf) =>
          `    { frame_idx: ${kf.frame_idx}, ratios: {\n${ratiosBlock(kf.ratios)}\n    }},`,
      )
      .join('\n');
    const text =
      '// ────────────────────────────────────────────────\n'
      + '// SwingCue Anchor Keyframes\n'
      + '// ────────────────────────────────────────────────\n'
      + `// video_id:        ${videoId}\n`
      + `// total_keyframes: ${sorted.length}\n`
      + '//\n'
      + '// Paste into VIDEO_KEYFRAMES[video_id]:\n'
      + '[\n'
      + body + '\n'
      + ']\n';
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      window.prompt('Copy these values:', text);
    }
  };

  const sortedKfs = [...keyframes].sort((a, b) => a.frame_idx - b.frame_idx);

  return (
    <div className="atp-panel" style={panelStyle}>
      {renderHeader()}

      <SectionLabel label="L SHOULDER  (0 → 0.60)" />
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

      <SectionLabel label="HEAD  (bipolar ±0.60)" />
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

      <div style={frameJumpRowStyle}>
        <span style={frameJumpLabelStyle}>Jump frame</span>
        <input
          type="text"
          inputMode="numeric"
          value={jumpInput}
          onChange={(e) => setJumpInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') handleJumpGo(); }}
          placeholder="N"
          style={frameJumpInputStyle}
        />
        <button onClick={handleJumpGo} style={frameJumpBtnStyle}>GO</button>
      </div>

      <button onClick={handleLoadInterp} style={loadInterpBtnStyle}>
        LOAD INTERP AT CURRENT FRAME
      </button>

      <div style={separatorThinStyle} />

      {/* Keyframe list */}
      <div style={kfSectionHeaderStyle}>
        KEYFRAMES ({sortedKfs.length})
      </div>
      {sortedKfs.length === 0 ? (
        <div style={noDataStyle}>(no keyframes yet — save current frame to start)</div>
      ) : (
        <div style={kfListStyle}>
          {sortedKfs.map((kf) => {
            const isCurrent = debug.frameIdx === kf.frame_idx;
            return (
              <div key={kf.frame_idx} style={isCurrent ? kfRowActiveStyle : kfRowStyle}>
                <span style={kfFrameStyle}>f={kf.frame_idx}</span>
                <button onClick={() => seekToFrame(kf.frame_idx)} style={kfBtnStyle} title="Jump">↪</button>
                <button onClick={() => handleEditKeyframe(kf.frame_idx)} style={kfBtnStyle} title="Edit (load + jump)">✎</button>
                <button onClick={() => handleDeleteKeyframe(kf.frame_idx)} style={kfBtnDangerStyle} title="Delete">×</button>
              </div>
            );
          })}
        </div>
      )}
      <button onClick={handleSaveKeyframe} style={saveKfBtnStyle}>
        + SAVE CURRENT FRAME AS KEYFRAME
      </button>

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
        {copied ? 'COPIED!' : 'COPY VIDEO_KEYFRAMES ARRAY'}
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
  fontSize: 10,
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

const frameJumpRowStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 6,
  marginTop: 6,
  marginBottom: 4,
};

const frameJumpLabelStyle: CSSProperties = {
  fontSize: 10,
  color: 'rgba(255,255,255,0.7)',
  fontWeight: 600,
  minWidth: 70,
};

const frameJumpInputStyle: CSSProperties = {
  flex: 1,
  background: 'rgba(255,255,255,0.06)',
  border: '1px solid rgba(255,255,255,0.18)',
  borderRadius: 3,
  color: '#fff',
  fontSize: 11,
  padding: '3px 6px',
  fontFamily: 'inherit',
  fontVariantNumeric: 'tabular-nums',
};

const frameJumpBtnStyle: CSSProperties = {
  padding: '3px 10px',
  background: 'rgba(255,0,255,0.18)',
  color: '#FF00FF',
  border: '1px solid rgba(255,0,255,0.5)',
  borderRadius: 3,
  fontSize: 10,
  fontWeight: 700,
  fontFamily: 'inherit',
  cursor: 'pointer',
  letterSpacing: 0.4,
};

const loadInterpBtnStyle: CSSProperties = {
  width: '100%',
  padding: '5px 8px',
  background: 'rgba(255,255,255,0.06)',
  color: 'rgba(255,255,255,0.85)',
  border: '1px solid rgba(255,255,255,0.18)',
  borderRadius: 3,
  fontSize: 10,
  fontWeight: 600,
  fontFamily: 'inherit',
  cursor: 'pointer',
  letterSpacing: 0.4,
  marginTop: 4,
};

const kfSectionHeaderStyle: CSSProperties = {
  fontSize: 10,
  fontWeight: 700,
  letterSpacing: 0.8,
  color: 'rgba(255,255,255,0.85)',
  marginTop: 4,
  marginBottom: 4,
  textTransform: 'uppercase',
};

const kfListStyle: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 2,
  marginBottom: 6,
};

const kfRowStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 4,
  padding: '3px 6px',
  background: 'rgba(255,255,255,0.04)',
  borderRadius: 3,
  fontSize: 10,
};

const kfRowActiveStyle: CSSProperties = {
  ...kfRowStyle,
  background: 'rgba(255,0,255,0.18)',
  outline: '1px solid rgba(255,0,255,0.5)',
};

const kfFrameStyle: CSSProperties = {
  flex: 1,
  color: '#fff',
  fontVariantNumeric: 'tabular-nums',
  fontWeight: 600,
};

const kfBtnStyle: CSSProperties = {
  width: 22,
  height: 22,
  padding: 0,
  background: 'rgba(255,255,255,0.08)',
  color: 'rgba(255,255,255,0.85)',
  border: '1px solid rgba(255,255,255,0.18)',
  borderRadius: 3,
  fontSize: 11,
  fontFamily: 'inherit',
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  lineHeight: 1,
};

const kfBtnDangerStyle: CSSProperties = {
  ...kfBtnStyle,
  background: 'rgba(255, 80, 80, 0.12)',
  color: 'rgba(255, 120, 120, 0.95)',
  borderColor: 'rgba(255, 80, 80, 0.4)',
  fontWeight: 700,
};

const saveKfBtnStyle: CSSProperties = {
  width: '100%',
  padding: '6px 8px',
  background: 'rgba(168, 240, 64, 0.12)',
  color: '#A8F040',
  border: '1px solid rgba(168, 240, 64, 0.45)',
  borderRadius: 4,
  fontSize: 11,
  fontWeight: 600,
  fontFamily: 'inherit',
  cursor: 'pointer',
  letterSpacing: 0.4,
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
