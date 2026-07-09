#!/usr/bin/env python3
"""
CUE-CHICKENWING-001 展示环 v1
慢动作回放 + 鸡翅膀指示器到点自然亮起 (HTML5 interactive player)
"""
import json, cv2, numpy as np
from pathlib import Path

ROOT  = Path("/home/jason/projects/swingcue-postest")
CACHE = ROOT / "engine/kp_cache/batch2/fo-wrong-4.json"
VID   = Path("/mnt/c/Users/jason/Zening/Swingcue/Video/fo-wrong-4.mp4")
OUT   = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/cw001_v1_demo.html")
OUT.parent.mkdir(parents=True, exist_ok=True)

cap   = cv2.VideoCapture(str(VID))
FPS   = cap.get(cv2.CAP_PROP_FPS)   # 30.001
TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
VW    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 720
VH    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 1280
cap.release()
print(f"Video: {VW}x{VH} @ {FPS:.3f}fps  {TOTAL} frames")

with open(CACHE) as f:
    raw = json.load(f)['frames']

KEYS = ['left_shoulder','left_elbow','left_wrist',
        'right_shoulder','right_elbow','right_wrist',
        'left_hip','right_hip']

kp_all = {}
for fi, fr in enumerate(raw):
    if not fr.get('persons'): continue
    kps = fr['persons'][0]['keypoints']
    kp_all[fi] = {k: [round(kps[k]['x'],1), round(kps[k]['y'],1)]
                   for k in KEYS if k in kps}
kp_json = json.dumps(kp_all)
print(f"KP frames exported: {len(kp_all)}")

# CW window (from chickenwing_fw4_scan.py results)
# trigger: fr148-149 (direction flips), clear: fr155-163, display: fr149
CW_FADE_IN  = 145
CW_ON       = 148
CW_OFF      = 163
CW_FADE_OUT = 167

# Progress bar CW zone percentage
cw_pct_start = round(CW_FADE_IN / TOTAL * 100, 2)
cw_pct_end   = round(CW_FADE_OUT / TOTAL * 100, 2)

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SwingCue · Chicken Wing · fo-wrong-4</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0d0d14;
    color: #ddd;
    font-family: 'SF Mono', 'Consolas', monospace;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 24px 16px;
    min-height: 100vh;
  }}
  .title-row {{
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 14px; color: #666; font-size: 11px; letter-spacing: 0.08em;
  }}
  .clip-label {{ color: #aaa; font-size: 12px; }}
  .player-wrap {{
    position: relative;
    width: 360px;
    background: #000;
    border-radius: 6px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6);
  }}
  video {{
    width: 100%;
    display: block;
  }}
  #overlay {{
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
  }}
  /* CW badge — floats top-right of video */
  #cwBadge {{
    position: absolute;
    top: 10px; right: 10px;
    background: rgba(228,15,17,0.85);
    color: #fff;
    font-size: 10px;
    font-weight: bold;
    letter-spacing: 0.1em;
    padding: 3px 8px;
    border-radius: 3px;
    opacity: 0;
    transition: opacity 0.15s;
    pointer-events: none;
  }}
  /* Controls */
  .controls {{
    width: 360px;
    margin-top: 10px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}
  .ctrl-row {{
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  #playBtn {{
    background: #222;
    border: 1px solid #444;
    color: #eee;
    font-size: 16px;
    width: 34px; height: 34px;
    border-radius: 4px;
    cursor: pointer;
    flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
  }}
  #playBtn:hover {{ background: #333; }}
  .progress-wrap {{
    flex: 1;
    position: relative;
    height: 34px;
    display: flex; align-items: center;
  }}
  /* CW zone tint behind progress bar */
  .cw-zone {{
    position: absolute;
    height: 6px;
    background: rgba(228,15,17,0.28);
    border-radius: 3px;
    pointer-events: none;
    left: {cw_pct_start}%;
    width: {cw_pct_end - cw_pct_start:.2f}%;
    top: 50%; transform: translateY(-50%);
    z-index: 1;
  }}
  #progress {{
    -webkit-appearance: none;
    width: 100%;
    height: 6px;
    border-radius: 3px;
    background: #333;
    outline: none;
    cursor: pointer;
    position: relative; z-index: 2;
    background-color: transparent;
  }}
  #progress::-webkit-slider-thumb {{
    -webkit-appearance: none;
    width: 14px; height: 14px;
    border-radius: 50%;
    background: #e0e0e0;
    cursor: pointer;
    box-shadow: 0 0 4px rgba(0,0,0,0.5);
  }}
  .progress-track {{
    position: absolute;
    width: 100%; height: 6px;
    background: #2a2a3a;
    border-radius: 3px;
    top: 50%; transform: translateY(-50%);
    z-index: 0;
  }}
  .progress-fill {{
    position: absolute;
    height: 6px;
    background: #4a4a6a;
    border-radius: 3px;
    top: 50%; transform: translateY(-50%);
    z-index: 0;
    width: 0%;
    transition: width 0.05s linear;
    pointer-events: none;
  }}
  .speed-badge {{
    font-size: 10px; color: #888;
    background: #1a1a28; border: 1px solid #333;
    padding: 2px 6px; border-radius: 3px;
    white-space: nowrap; flex-shrink: 0;
  }}
  .info-row {{
    display: flex; justify-content: space-between;
    font-size: 10px; color: #555;
    padding: 0 2px;
  }}
  #frameDisp {{ color: #888; }}
  #cueState {{ color: #555; }}
  /* Frame step buttons */
  .step-btn {{
    background: #1a1a28; border: 1px solid #333; color: #888;
    font-size: 11px; padding: 2px 8px; border-radius: 3px;
    cursor: pointer; flex-shrink: 0;
  }}
  .step-btn:hover {{ color: #ccc; background: #252535; }}
  .hint {{
    font-size: 9px; color: #444; margin-top: 6px; text-align: center;
    letter-spacing: 0.05em;
  }}
</style>
</head>
<body>

<div class="title-row">
  <span>SWINGCUE</span>
  <span style="color:#333">·</span>
  <span class="clip-label">fo-wrong-4</span>
  <span style="color:#333">·</span>
  <span>CUE-CHICKENWING-001 · v1</span>
</div>

<div class="player-wrap" id="playerWrap">
  <video id="vid"
    src="file:///C:/Users/jason/Zening/Swingcue/Video/fo-wrong-4.mp4"
    preload="auto">
  </video>
  <canvas id="overlay"></canvas>
  <div id="cwBadge">⚠ CHICKEN WING</div>
</div>

<div class="controls">
  <div class="ctrl-row">
    <button id="playBtn" title="Space">▶</button>
    <div class="progress-wrap">
      <div class="progress-track"></div>
      <div class="progress-fill" id="progressFill"></div>
      <div class="cw-zone"></div>
      <input type="range" id="progress" min="0" max="10000" value="0" step="1">
    </div>
    <span class="speed-badge">0.25×</span>
  </div>
  <div class="ctrl-row" style="justify-content:space-between">
    <div class="ctrl-row" style="gap:4px">
      <button class="step-btn" id="stepBack">◀ 1fr</button>
      <button class="step-btn" id="stepFwd">1fr ▶</button>
    </div>
    <div id="frameDisp">fr0 · 0.00s</div>
  </div>
  <div class="info-row">
    <span id="cueState" style="color:#333">──</span>
    <span style="color:#333">{TOTAL} frames · {FPS:.0f}fps</span>
  </div>
</div>
<div class="hint">SPACE play/pause  ·  ← → frame step  ·  drag progress to scrub</div>

<script>
// ─── Constants ────────────────────────────────────────────────────────────────
const FPS    = {FPS:.6f};
const VID_W  = {VW};
const VID_H  = {VH};
const TOTAL  = {TOTAL};

const CW = {{ FADE_IN: {CW_FADE_IN}, ON: {CW_ON}, OFF: {CW_OFF}, FADE_OUT: {CW_FADE_OUT} }};
const PLAYBACK_RATE = 0.25;

// Colors (from reference image sampling — BGR→RGB)
const RED   = 'rgb(228,15,17)';
const GREEN = 'rgb(48,220,12)';

const NODE_R     = 6;    // display px
const LINE_W     = 4;    // display px
const STANDOFF   = 35;   // display px — gap from elbow to arrow tip
const ARROW_LEN  = 22;   // display px
const ARROW_HW   = 14;   // display px half-width
const ARROW_NOTCH= 0.38;
const ARROW_GAP  = 28;   // display px between tandem arrows
const N_ARROWS   = 2;

// Arrow animation cycle (from v11 GIF timing)
const CYCLE_MS   = 1730;
const PUSH_END   = 1070; // ms
const PAUSE_END  = 1430; // ms
const INVIS_END  = 1490; // ms

// All keypoints (from kp_cache)
const KP_ALL = {kp_json};

// ─── Elements ─────────────────────────────────────────────────────────────────
const vid         = document.getElementById('vid');
const canvas      = document.getElementById('overlay');
const ctx         = canvas.getContext('2d');
const playBtn     = document.getElementById('playBtn');
const progressEl  = document.getElementById('progress');
const progressFill= document.getElementById('progressFill');
const frameDisp   = document.getElementById('frameDisp');
const cueState    = document.getElementById('cueState');
const cwBadge     = document.getElementById('cwBadge');
const stepBack    = document.getElementById('stepBack');
const stepFwd     = document.getElementById('stepFwd');

// ─── Canvas sizing ────────────────────────────────────────────────────────────
function syncCanvas() {{
  const r = vid.getBoundingClientRect();
  canvas.width  = r.width;
  canvas.height = r.height;
}}
vid.addEventListener('loadedmetadata', syncCanvas);
new ResizeObserver(syncCanvas).observe(vid);
syncCanvas();

// ─── Playback control ─────────────────────────────────────────────────────────
vid.playbackRate = PLAYBACK_RATE;
vid.addEventListener('play',  () => {{ playBtn.textContent = '⏸'; }});
vid.addEventListener('pause', () => {{ playBtn.textContent = '▶'; }});
playBtn.addEventListener('click', togglePlay);

function togglePlay() {{
  if (vid.paused) vid.play(); else vid.pause();
}}

// ─── Progress bar ─────────────────────────────────────────────────────────────
let dragging = false;

vid.addEventListener('timeupdate', () => {{
  if (!dragging) syncProgress();
}});

function syncProgress() {{
  const pct = vid.duration ? vid.currentTime / vid.duration : 0;
  progressEl.value = pct * 10000;
  progressFill.style.width = (pct * 100) + '%';
  updateDisp();
}}

progressEl.addEventListener('mousedown', () => {{ dragging = true; }});
progressEl.addEventListener('touchstart', () => {{ dragging = true; }});
progressEl.addEventListener('input', () => {{
  vid.currentTime = progressEl.value / 10000 * vid.duration;
  progressFill.style.width = (progressEl.value / 100) + '%';
  updateDisp();
}});
progressEl.addEventListener('mouseup',  () => {{ dragging = false; }});
progressEl.addEventListener('touchend', () => {{ dragging = false; }});

function updateDisp() {{
  const fr = Math.round(vid.currentTime * FPS);
  frameDisp.textContent = `fr${{fr}} · ${{vid.currentTime.toFixed(2)}}s`;
}}

stepBack.addEventListener('click', () => {{ vid.pause(); vid.currentTime = Math.max(0, vid.currentTime - 1/FPS); }});
stepFwd.addEventListener('click',  () => {{ vid.pause(); vid.currentTime = Math.min(vid.duration||999, vid.currentTime + 1/FPS); }});

document.addEventListener('keydown', e => {{
  if (e.target.tagName === 'INPUT') return;
  if (e.key === ' ')          {{ e.preventDefault(); togglePlay(); }}
  if (e.key === 'ArrowRight') {{ vid.pause(); vid.currentTime += 1/FPS; }}
  if (e.key === 'ArrowLeft')  {{ vid.pause(); vid.currentTime -= 1/FPS; }}
}});

// ─── Indicator alpha ──────────────────────────────────────────────────────────
function getAlpha(frame) {{
  const {{FADE_IN, ON, OFF, FADE_OUT}} = CW;
  if (frame <= FADE_IN || frame >= FADE_OUT) return 0;
  if (frame >= ON && frame <= OFF) return 1;
  if (frame < ON)  return (frame - FADE_IN)  / (ON  - FADE_IN);
  return               1 - (frame - OFF)     / (FADE_OUT - OFF);
}}

// ─── Nearest keypoint lookup ──────────────────────────────────────────────────
const KP_KEYS = Object.keys(KP_ALL).map(Number).sort((a,b)=>a-b);

function nearestKP(frame) {{
  if (!KP_KEYS.length) return null;
  let best = KP_KEYS[0], bestD = Math.abs(KP_KEYS[0]-frame);
  for (const k of KP_KEYS) {{
    const d = Math.abs(k-frame);
    if (d < bestD) {{ bestD=d; best=k; }}
    if (k > frame+30) break;
  }}
  return KP_ALL[best];
}}

// ─── Per-frame geometry ───────────────────────────────────────────────────────
function computeGeo(kp) {{
  const scaleX = canvas.width  / VID_W;
  const scaleY = canvas.height / VID_H;
  // Use same scale for x and y (video fills canvas maintaining aspect)
  // Since video is displayed with object-fit default (contain or fill), 
  // and we set CSS width=100% on video, letterbox may apply on height.
  // Actual rendered aspect: canvas.width / (canvas.width * VH/VW) = VW/VH
  // canvas.height should equal canvas.width * VH/VW
  const scale = canvas.width / VID_W;  // uniform scale

  function sc(name) {{
    const p = kp[name]; if (!p) return null;
    return [p[0]*scale, p[1]*scale];
  }}
  const ls = sc('left_shoulder'), le = sc('left_elbow'), lw = sc('left_wrist');
  const rs = sc('right_shoulder'), re = sc('right_elbow'), rw = sc('right_wrist');
  const lh = sc('left_hip'), rh = sc('right_hip');
  if (!ls || !le || !lw) return null;

  // green_elbow = projection of le onto ls-lw line
  const sw = [lw[0]-ls[0], lw[1]-ls[1]];
  const swSq = sw[0]*sw[0]+sw[1]*sw[1]+1e-9;
  const t = Math.max(0.1, Math.min(0.9,
    ((le[0]-ls[0])*sw[0] + (le[1]-ls[1])*sw[1]) / swSq));
  const ge = [ls[0]+t*sw[0], ls[1]+t*sw[1]];

  // outward = le - ge, normalized
  const out = [le[0]-ge[0], le[1]-ge[1]];
  const outMag = Math.sqrt(out[0]*out[0]+out[1]*out[1])+1e-9;
  const outN = [out[0]/outMag, out[1]/outMag];

  // arrow travel (in display px, not scaled — intentional, keeps indicator size consistent)
  const ts_ = [le[0]+outN[0]*STANDOFF, le[1]+outN[1]*STANDOFF];
  const te_ = ge;
  const tv = [te_[0]-ts_[0], te_[1]-ts_[1]];
  const tvMag = Math.sqrt(tv[0]*tv[0]+tv[1]*tv[1])+1e-9;
  const tvN = [tv[0]/tvMag, tv[1]/tvMag];

  return {{ ls, le, lw, rs, re, rw, lh, rh, ge, ts: ts_, te: te_, tvN, tvMag }};
}}

// ─── Arrow animation state ────────────────────────────────────────────────────
function arrowState(nowMs) {{
  const t = nowMs % CYCLE_MS;
  if (t < 120)         return {{ e: 0.0, show: true  }};  // pause at start
  if (t < PUSH_END)    {{ const s=(t-120)/(PUSH_END-120); return {{ e: s*s*(3-2*s), show: true  }}; }}
  if (t < PAUSE_END)   return {{ e: 1.0, show: true  }};  // pause at end
  if (t < INVIS_END)   return {{ e: 0.0, show: false }};  // invisible
  return                      {{ e: 0.0, show: true  }};  // pause at start (tail)
}}

// ─── Draw helpers ─────────────────────────────────────────────────────────────
function drawBarb(tipX, tipY, dirX, dirY) {{
  const px = -dirY, py = dirX;
  const tailX = tipX - dirX*ARROW_LEN,  tailY = tipY - dirY*ARROW_LEN;
  const nX = tipX - dirX*ARROW_LEN*(1-ARROW_NOTCH);
  const nY = tipY - dirY*ARROW_LEN*(1-ARROW_NOTCH);
  ctx.fillStyle = GREEN;
  ctx.beginPath();
  ctx.moveTo(tipX, tipY);
  ctx.lineTo(tailX+px*ARROW_HW, tailY+py*ARROW_HW);
  ctx.lineTo(nX, nY);
  ctx.lineTo(tailX-px*ARROW_HW, tailY-py*ARROW_HW);
  ctx.closePath();
  ctx.fill();
}}

function drawIndicator(geo, arw, alpha) {{
  if (alpha <= 0) return;
  const {{ls,le,lw,rs,re,rw,lh,rh,ge,ts,te,tvN}} = geo;
  ctx.save();
  ctx.globalAlpha = alpha;

  // ── muted skeleton (right arm + hips) ──
  ctx.strokeStyle = 'rgba(90,90,90,0.75)';
  ctx.lineWidth = 2;
  if (rs&&re) {{ ctx.beginPath(); ctx.moveTo(...rs); ctx.lineTo(...re); ctx.stroke(); }}
  if (re&&rw) {{ ctx.beginPath(); ctx.moveTo(...re); ctx.lineTo(...rw); ctx.stroke(); }}
  if (lh&&rh) {{ ctx.beginPath(); ctx.moveTo(...lh); ctx.lineTo(...rh); ctx.stroke(); }}

  // ── green glow line ls→lw ──
  ctx.save();
  ctx.globalAlpha = alpha * 0.22;
  ctx.strokeStyle = GREEN;
  ctx.lineWidth = 12;
  ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(...ls); ctx.lineTo(...lw); ctx.stroke();
  ctx.restore();

  // ── green core line ls→lw ──
  ctx.strokeStyle = GREEN;
  ctx.lineWidth = LINE_W;
  ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(...ls); ctx.lineTo(...lw); ctx.stroke();

  // ── red arm ls→le→lw ──
  ctx.strokeStyle = RED;
  ctx.lineWidth = LINE_W;
  ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(...ls); ctx.lineTo(...le); ctx.lineTo(...lw); ctx.stroke();

  // ── nodes (rebuild after lines so they're on top) ──
  // Green nodes: ls, lw, ge
  for (const p of [ls, lw]) {{
    ctx.fillStyle = GREEN;
    ctx.beginPath(); ctx.arc(...p, NODE_R, 0, Math.PI*2); ctx.fill();
  }}
  ctx.fillStyle = GREEN;
  ctx.beginPath(); ctx.arc(...ge, NODE_R*0.83, 0, Math.PI*2); ctx.fill();

  // Red nodes: ls, le, lw (on top of green where they overlap at shoulder)
  for (const p of [ls, le, lw]) {{
    ctx.fillStyle = RED;
    ctx.beginPath(); ctx.arc(...p, NODE_R, 0, Math.PI*2); ctx.fill();
  }}

  // ── barb arrows ──
  if (arw.show) {{
    const e = arw.e;
    const leadTip = [ts[0]+e*(te[0]-ts[0]), ts[1]+e*(te[1]-ts[1])];
    for (let i=0; i<N_ARROWS; i++) {{
      const tx = leadTip[0] - tvN[0]*i*ARROW_GAP;
      const ty = leadTip[1] - tvN[1]*i*ARROW_GAP;
      // don't draw before start or after end
      const sdist = (tx-ts[0])*tvN[0]+(ty-ts[1])*tvN[1];
      if (sdist < -ARROW_GAP*0.5) break;
      const edist = (tx-te[0])*tvN[0]+(ty-te[1])*tvN[1];
      if (edist > 0) continue;
      drawBarb(tx, ty, tvN[0], tvN[1]);
    }}
  }}

  ctx.restore();
}}

// ─── Main render loop ─────────────────────────────────────────────────────────
function renderLoop(nowMs) {{
  requestAnimationFrame(renderLoop);

  const frame = vid.currentTime * FPS;
  const alpha = getAlpha(frame);

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (alpha > 0) {{
    const kp = nearestKP(Math.round(frame));
    if (kp) {{
      const geo = computeGeo(kp);
      if (geo) {{
        const arw = arrowState(nowMs);
        drawIndicator(geo, arw, alpha);
      }}
    }}
  }}

  // CW badge visibility
  cwBadge.style.opacity = alpha > 0.1 ? String(alpha) : '0';

  // Cue state label
  if (alpha >= 1)      cueState.textContent = '⚠ CHICKEN WING  fr' + Math.round(frame);
  else if (alpha > 0)  cueState.textContent = '↑ fading in...';
  else                 cueState.textContent = '──';

  if (alpha >= 1) cueState.style.color = 'rgb(228,50,50)';
  else            cueState.style.color = '#555';
}}
requestAnimationFrame(renderLoop);

// ─── Auto-start slow-mo ───────────────────────────────────────────────────────
vid.addEventListener('loadeddata', () => {{
  vid.playbackRate = PLAYBACK_RATE;
  // don't auto-play, let user press play
}});
</script>
</body>
</html>
"""

OUT.write_text(html, encoding='utf-8')
print(f"=> {OUT}")
print(f"\nCW window: fr{CW_FADE_IN}(fade-in) → fr{CW_ON}(on) → fr{CW_OFF}(off) → fr{CW_FADE_OUT}(fade-out)")
print(f"Progress bar CW zone: {cw_pct_start}% ~ {cw_pct_end}%")
