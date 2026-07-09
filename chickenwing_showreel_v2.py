#!/usr/bin/env python3
"""
CUE-CHICKENWING-001 展示环 v2
离散定格点: 慢放 → 到fr149自动停住 → 静止画面上箭头循环演示 → Continue继续
"""
import json, cv2, numpy as np, base64
from pathlib import Path

ROOT  = Path("/home/jason/projects/swingcue-postest")
CACHE = ROOT / "engine/kp_cache/batch2/fo-wrong-4.json"
VID   = Path("/mnt/c/Users/jason/Zening/Swingcue/Video/fo-wrong-4.mp4")
OUT   = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/cw001_v2_demo.html")
OUT.parent.mkdir(parents=True, exist_ok=True)

FPS    = 30.001
TOTAL  = 209
VW, VH = 720, 1280

# Freeze config — one entry per display frame
FREEZE_CFG = [
    { "frame": 149, "cue": "CHICKEN WING", "desc": "方向刚翻转 · 身体正对 · fr149" }
]
# fr163+ body turned away → not selected (occluded, per principle)

# ── Extract display frame as static JPEG background ───────────────────────────
DISPLAY_FR = 149
print(f"Extracting display frame fr{DISPLAY_FR}...")
cap = cv2.VideoCapture(str(VID))
cap.set(cv2.CAP_PROP_POS_FRAMES, DISPLAY_FR)
ret, bgr = cap.read(); cap.release()
assert ret, f"Failed to read frame {DISPLAY_FR}"
_, jpg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
frozen_b64 = base64.b64encode(jpg.tobytes()).decode()
print(f"Frozen frame JPEG: {len(jpg.tobytes())//1024}KB → base64 {len(frozen_b64)//1024}KB")

# ── Keypoints for display frame ───────────────────────────────────────────────
with open(CACHE) as f:
    kp_raw = json.load(f)['frames']

def get_kp(fi):
    fr = kp_raw[fi]
    if not fr.get('persons'): return {}
    return {k: [round(v['x'],1), round(v['y'],1)]
            for k,v in fr['persons'][0]['keypoints'].items()}

kp149   = get_kp(DISPLAY_FR)
kp149_j = json.dumps(kp149)

# ── Pre-compute geometry so we can echo values in the report ──────────────────
def arr(kp,n):
    p = kp.get(n); return np.array(p, float) if p else None

ls = arr(kp149,'left_shoulder'); le = arr(kp149,'left_elbow')
lw = arr(kp149,'left_wrist')
sw = lw - ls
t  = float(np.clip(np.dot(le-ls,sw)/(np.dot(sw,sw)+1e-9), 0.1, 0.9))
ge = ls + t*sw
out = le - ge; outN = out/(np.linalg.norm(out)+1e-9)
ts = le + outN*35   # STANDOFF=35
tv = ge - ts; tvMag = np.linalg.norm(tv)

print(f"fr149 geometry: ls={ls.astype(int)}, le={le.astype(int)}, lw={lw.astype(int)}")
print(f"  green_elbow={ge.astype(int)}, travel={tvMag:.1f}px")

# ──────────────────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SwingCue · Chicken Wing Demo · fo-wrong-4</title>
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{
  background:#0a0a10; color:#ccc;
  font-family:'SF Mono',Consolas,monospace;
  display:flex; flex-direction:column; align-items:center;
  padding:20px 16px; min-height:100vh;
}}
.title {{
  font-size:11px; color:#555; letter-spacing:.1em;
  margin-bottom:12px; display:flex; gap:10px; align-items:center;
}}
.title span.hi {{ color:#999; }}

/* ── player stack ── */
.stage {{
  position:relative;
  width:360px;
  background:#000;
  border-radius:6px;
  overflow:hidden;
  box-shadow:0 12px 40px rgba(0,0,0,.7);
}}
#vid {{
  width:100%;
  display:block;
}}
/* frozen overlay: static JPEG shown when paused at a freeze frame */
#frozenImg {{
  position:absolute; inset:0;
  width:100%; height:100%;
  display:none;
  object-fit:fill;
}}
#cvs {{
  position:absolute; inset:0;
  width:100%; height:100%;
  pointer-events:none;
}}

/* Continue button — centred over the frozen frame */
#continueBtn {{
  position:absolute;
  bottom:52px; left:50%;
  transform:translateX(-50%);
  background:rgba(10,10,20,.82);
  border:1px solid rgba(255,255,255,.18);
  color:#ddd;
  font-family:inherit;
  font-size:12px;
  letter-spacing:.08em;
  padding:7px 22px;
  border-radius:20px;
  cursor:pointer;
  display:none;
  z-index:10;
  white-space:nowrap;
  backdrop-filter:blur(4px);
}}
#continueBtn:hover {{ background:rgba(30,30,50,.9); color:#fff; }}

/* CW badge */
#badge {{
  position:absolute; top:10px; right:10px;
  background:rgba(228,15,17,.9);
  color:#fff; font-size:10px; font-weight:bold;
  letter-spacing:.12em; padding:3px 8px;
  border-radius:3px; display:none; z-index:5;
}}

/* ── controls ── */
.controls {{
  width:360px; margin-top:10px;
  display:flex; flex-direction:column; gap:7px;
}}
.row {{ display:flex; align-items:center; gap:8px; }}
#playBtn {{
  background:#1c1c2c; border:1px solid #333; color:#ddd;
  font-size:15px; width:34px; height:34px;
  border-radius:4px; cursor:pointer; flex-shrink:0;
  display:flex; align-items:center; justify-content:center;
}}
#playBtn:hover {{ background:#252540; }}

/* progress bar */
.prog-wrap {{
  flex:1; position:relative; height:34px;
  display:flex; align-items:center;
}}
.prog-track {{
  position:absolute; width:100%; height:6px;
  background:#1e1e2e; border-radius:3px;
}}
.prog-fill {{
  position:absolute; height:6px;
  background:#3a3a5a; border-radius:3px;
  width:0%; pointer-events:none; transition:width .04s linear;
}}
/* freeze-point dot on track */
.freeze-dot {{
  position:absolute; width:8px; height:8px;
  background:rgb(228,15,17); border-radius:50%;
  top:50%; transform:translate(-50%,-50%);
  left:{round(DISPLAY_FR/TOTAL*100,2)}%;
  z-index:3; pointer-events:none;
}}
#prog {{
  -webkit-appearance:none; width:100%; height:6px;
  border-radius:3px; background:transparent;
  outline:none; cursor:pointer; position:relative; z-index:4;
}}
#prog::-webkit-slider-thumb {{
  -webkit-appearance:none; width:14px; height:14px;
  border-radius:50%; background:#ddd; cursor:pointer;
  box-shadow:0 0 4px rgba(0,0,0,.5);
}}
.speed-badge {{
  font-size:10px; color:#666;
  background:#141420; border:1px solid #2a2a3a;
  padding:2px 7px; border-radius:3px;
  white-space:nowrap; flex-shrink:0;
}}
.row2 {{ display:flex; justify-content:space-between; font-size:10px; }}
.step-btn {{
  background:#141420; border:1px solid #2a2a3a; color:#666;
  font-size:11px; padding:2px 9px; border-radius:3px; cursor:pointer;
}}
.step-btn:hover {{ color:#bbb; background:#1c1c30; }}
#frameDisp {{ color:#777; }}
#cueDisp   {{ color:#444; }}
.hint {{
  font-size:9px; color:#333; margin-top:6px; text-align:center;
  letter-spacing:.05em;
}}
</style>
</head>
<body>

<div class="title">
  <span>SWINGCUE</span>
  <span>·</span>
  <span class="hi">fo-wrong-4</span>
  <span>·</span>
  <span>CUE-CHICKENWING-001  v2</span>
</div>

<div class="stage" id="stage">
  <video id="vid"
    src="file:///C:/Users/jason/Zening/Swingcue/Video/fo-wrong-4.mp4"
    preload="auto">
  </video>
  <img id="frozenImg" src="data:image/jpeg;base64,{frozen_b64}" alt="fr149">
  <canvas id="cvs"></canvas>
  <button id="continueBtn">▶ Continue</button>
  <div id="badge">⚠ CHICKEN WING</div>
</div>

<div class="controls">
  <div class="row">
    <button id="playBtn">▶</button>
    <div class="prog-wrap">
      <div class="prog-track"></div>
      <div class="prog-fill" id="progFill"></div>
      <div class="freeze-dot" title="fr{DISPLAY_FR} · 定格点"></div>
      <input type="range" id="prog" min="0" max="10000" value="0">
    </div>
    <span class="speed-badge">0.25×</span>
  </div>
  <div class="row" style="justify-content:space-between">
    <div class="row" style="gap:4px">
      <button class="step-btn" id="stepB">◀ 1fr</button>
      <button class="step-btn" id="stepF">1fr ▶</button>
    </div>
    <span id="frameDisp">fr0 · 0.00s</span>
  </div>
  <div class="row2">
    <span id="cueDisp">──</span>
    <span style="color:#333">{TOTAL}fr · {FPS:.0f}fps</span>
  </div>
</div>
<div class="hint">SPACE play/pause  ·  ← → frame step  ·  drag to scrub  ·  red dot = cue point</div>

<script>
'use strict';
// ── constants ───────────────────────────────────────────────────────────────
const FPS    = {FPS:.6f};
const TOTAL  = {TOTAL};
const VW     = {VW}, VH = {VH};
const FR     = 149;    // the one freeze / display frame
const SNAP_R = 5;      // frames radius to trigger freeze on scrub

// colors (from reference image sampling)
const RED   = 'rgb(228,15,17)';
const GREEN = 'rgb(48,220,12)';
const GLOW  = 'rgba(48,220,12,0.20)';
const MUTED = 'rgba(80,80,80,0.8)';

// indicator geometry params
const STANDOFF   = 35;
const ARROW_LEN  = 22;
const ARROW_HW   = 14;
const ARROW_NOTCH= 0.38;
const ARROW_GAP  = 28;
const N_ARROWS   = 2;
const NODE_R     = 6;
const LINE_W     = 4;

// animation cycle (same as v11 GIF)
const CYCLE_MS   = 1730;
const PUSH_MS    = 950;   // 0 → 120ms pause_start → 120+PUSH_MS push
const PAUSE1_MS  = 1310;  // pause at tip
const INVIS_MS   = 1430;  // invisible
// 1430→1730 = pause at start again

// keypoints for fr149 (fixed)
const KP = {kp149_j};

// ── elements ─────────────────────────────────────────────────────────────────
const vid        = document.getElementById('vid');
const frozenImg  = document.getElementById('frozenImg');
const cvs        = document.getElementById('cvs');
const ctx        = cvs.getContext('2d');
const playBtn    = document.getElementById('playBtn');
const prog       = document.getElementById('prog');
const progFill   = document.getElementById('progFill');
const frameDisp  = document.getElementById('frameDisp');
const cueDisp    = document.getElementById('cueDisp');
const badge      = document.getElementById('badge');
const contBtn    = document.getElementById('continueBtn');
const stepB      = document.getElementById('stepB');
const stepF      = document.getElementById('stepF');

// ── canvas sizing ─────────────────────────────────────────────────────────────
function syncCanvas() {{
  const r = vid.getBoundingClientRect();
  cvs.width = r.width; cvs.height = r.height;
}}
vid.addEventListener('loadedmetadata', syncCanvas);
new ResizeObserver(syncCanvas).observe(vid);
syncCanvas();

// ── geometry (recomputed on canvas resize, fixed KP) ─────────────────────────
function computeGeo() {{
  const scale = cvs.width / VW;
  function sc(n) {{ const p=KP[n]; return p ? [p[0]*scale, p[1]*scale] : null; }}
  const ls=sc('left_shoulder'), le=sc('left_elbow'), lw=sc('left_wrist');
  const rs=sc('right_shoulder'), re=sc('right_elbow'), rw=sc('right_wrist');
  const lh=sc('left_hip'), rh=sc('right_hip');
  if (!ls||!le||!lw) return null;

  const sw=[lw[0]-ls[0], lw[1]-ls[1]];
  const ssq=sw[0]*sw[0]+sw[1]*sw[1]+1e-9;
  const t=Math.max(0.1,Math.min(0.9,
    ((le[0]-ls[0])*sw[0]+(le[1]-ls[1])*sw[1])/ssq));
  const ge=[ls[0]+t*sw[0], ls[1]+t*sw[1]];

  const out=[le[0]-ge[0], le[1]-ge[1]];
  const omag=Math.sqrt(out[0]*out[0]+out[1]*out[1])+1e-9;
  const on=[out[0]/omag, out[1]/omag];

  const ts=[le[0]+on[0]*STANDOFF, le[1]+on[1]*STANDOFF];
  const te=ge;
  const tv=[te[0]-ts[0], te[1]-ts[1]];
  const tvmag=Math.sqrt(tv[0]*tv[0]+tv[1]*tv[1])+1e-9;
  const tvn=[tv[0]/tvmag, tv[1]/tvmag];

  return {{ls,le,lw,rs,re,rw,lh,rh,ge,ts,te,tvn,tvmag}};
}}

// ── state machine ─────────────────────────────────────────────────────────────
// states: 'playing' | 'frozen' | 'user_paused'
let state = 'playing';
let fired = false;   // have we auto-fired the freeze this pass?
let dragging = false;

vid.playbackRate = 0.25;

function setFrozen() {{
  state = 'frozen';
  vid.pause();
  // snap to exact frame
  vid.currentTime = FR / FPS;
  frozenImg.style.display = 'block';   // static JPEG in front
  badge.style.display = 'block';
  contBtn.style.display = 'block';
  playBtn.textContent = '⏸';
  cueDisp.textContent = '⚠ CHICKEN WING  fr149 · 定格演示';
  cueDisp.style.color = 'rgb(228,80,80)';
}}

function resumePlay() {{
  state = 'playing';
  frozenImg.style.display = 'none';
  badge.style.display = 'none';
  contBtn.style.display = 'none';
  // advance 1 frame so we don't immediately re-trigger
  vid.currentTime = Math.min(vid.duration||999, (FR+2)/FPS);
  vid.play();
  cueDisp.textContent = '──';
  cueDisp.style.color = '#444';
}}

contBtn.addEventListener('click', resumePlay);

// ── playback listeners ────────────────────────────────────────────────────────
vid.addEventListener('play',  () => {{ if (state !== 'frozen') playBtn.textContent='⏸'; }});
vid.addEventListener('pause', () => {{
  if (state === 'playing') {{ state='user_paused'; playBtn.textContent='▶'; }}
}});

playBtn.addEventListener('click', () => {{
  if (state === 'frozen') {{ resumePlay(); return; }}
  if (vid.paused) {{ state='playing'; vid.play(); }}
  else            {{ vid.pause(); }}
}});

vid.addEventListener('timeupdate', () => {{
  if (dragging) return;
  syncProg();

  if (state === 'playing' && !fired) {{
    const frame = vid.currentTime * FPS;
    if (frame >= FR) {{
      fired = true;
      setFrozen();
    }}
  }}
  // reset fired when user scrubs back far enough
  const frame = vid.currentTime * FPS;
  if (frame < FR - 20) fired = false;
}});

vid.addEventListener('ended', () => {{
  state = 'user_paused';
  playBtn.textContent = '▶';
}});

// ── progress bar ─────────────────────────────────────────────────────────────
prog.addEventListener('mousedown',  () => {{ dragging=true; }});
prog.addEventListener('touchstart', () => {{ dragging=true; }});
prog.addEventListener('input', () => {{
  const t = prog.value/10000 * (vid.duration||1);
  vid.currentTime = t;
  syncProg();
}});
prog.addEventListener('mouseup',  onScrubEnd);
prog.addEventListener('touchend', onScrubEnd);

function onScrubEnd() {{
  dragging = false;
  const frame = vid.currentTime * FPS;
  // if user scrubs near the display frame and is paused: show freeze
  if (vid.paused && Math.abs(frame - FR) <= SNAP_R && state !== 'frozen') {{
    setFrozen();
    fired = true;
  }}
}}

function syncProg() {{
  const pct = vid.duration ? vid.currentTime/vid.duration : 0;
  prog.value = pct * 10000;
  progFill.style.width = (pct*100)+'%';
  const fr = Math.round(vid.currentTime * FPS);
  frameDisp.textContent = `fr${{fr}} · ${{vid.currentTime.toFixed(2)}}s`;
}}

// ── keyboard ──────────────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {{
  if (e.target.tagName==='INPUT') return;
  if (e.key===' ') {{
    e.preventDefault();
    if (state==='frozen') {{ resumePlay(); return; }}
    if (vid.paused) {{ state='playing'; vid.play(); }} else vid.pause();
  }}
  if (e.key==='ArrowRight') {{ vid.pause(); state='user_paused'; vid.currentTime+=1/FPS; }}
  if (e.key==='ArrowLeft')  {{ vid.pause(); state='user_paused'; vid.currentTime-=1/FPS; }}
}});

stepB.addEventListener('click', () => {{ vid.pause(); state='user_paused'; vid.currentTime=Math.max(0,vid.currentTime-1/FPS); }});
stepF.addEventListener('click', () => {{ vid.pause(); state='user_paused'; vid.currentTime+=1/FPS; }});

// ── indicator drawing ─────────────────────────────────────────────────────────
function drawBarb(tipX, tipY, dx, dy) {{
  const px=-dy, py=dx;
  const tailX=tipX-dx*ARROW_LEN, tailY=tipY-dy*ARROW_LEN;
  const nX=tipX-dx*ARROW_LEN*(1-ARROW_NOTCH);
  const nY=tipY-dy*ARROW_LEN*(1-ARROW_NOTCH);
  ctx.fillStyle = GREEN;
  ctx.beginPath();
  ctx.moveTo(tipX,tipY);
  ctx.lineTo(tailX+px*ARROW_HW, tailY+py*ARROW_HW);
  ctx.lineTo(nX, nY);
  ctx.lineTo(tailX-px*ARROW_HW, tailY-py*ARROW_HW);
  ctx.closePath(); ctx.fill();
}}

function drawIndicator(geo, arwE, arwShow) {{
  const {{ls,le,lw,rs,re,rw,lh,rh,ge,ts,te,tvn}} = geo;

  // muted skeleton
  ctx.strokeStyle=MUTED; ctx.lineWidth=2; ctx.lineCap='round';
  for (const [a,b] of [[rs,re],[re,rw],[lh,rh]]) {{
    if (!a||!b) continue;
    ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
  }}

  // green glow
  ctx.save();
  ctx.strokeStyle=GLOW; ctx.lineWidth=13; ctx.lineCap='round';
  ctx.beginPath(); ctx.moveTo(...ls); ctx.lineTo(...lw); ctx.stroke();
  ctx.restore();

  // green core line
  ctx.strokeStyle=GREEN; ctx.lineWidth=LINE_W; ctx.lineCap='round';
  ctx.beginPath(); ctx.moveTo(...ls); ctx.lineTo(...lw); ctx.stroke();

  // red arm
  ctx.strokeStyle=RED; ctx.lineWidth=LINE_W; ctx.lineCap='round';
  ctx.beginPath(); ctx.moveTo(...ls); ctx.lineTo(...le); ctx.lineTo(...lw); ctx.stroke();

  // green nodes (ls, lw, ge)
  ctx.fillStyle=GREEN;
  for (const p of [ls,lw]) {{
    ctx.beginPath(); ctx.arc(...p, NODE_R, 0, Math.PI*2); ctx.fill();
  }}
  ctx.beginPath(); ctx.arc(...ge, NODE_R*0.83, 0, Math.PI*2); ctx.fill();

  // red nodes on top (ls, le, lw)
  ctx.fillStyle=RED;
  for (const p of [ls,le,lw]) {{
    ctx.beginPath(); ctx.arc(...p, NODE_R, 0, Math.PI*2); ctx.fill();
  }}

  // barb arrows
  if (arwShow) {{
    const leadTip=[ts[0]+arwE*(te[0]-ts[0]), ts[1]+arwE*(te[1]-ts[1])];
    for (let i=0; i<N_ARROWS; i++) {{
      const tx=leadTip[0]-tvn[0]*i*ARROW_GAP;
      const ty=leadTip[1]-tvn[1]*i*ARROW_GAP;
      const sdist=(tx-ts[0])*tvn[0]+(ty-ts[1])*tvn[1];
      if (sdist < -ARROW_GAP*0.5) break;
      const edist=(tx-te[0])*tvn[0]+(ty-te[1])*tvn[1];
      if (edist>0) continue;
      drawBarb(tx,ty,tvn[0],tvn[1]);
    }}
  }}
}}

// arrow animation state from wallclock
function arrowState(nowMs) {{
  const t = nowMs % CYCLE_MS;
  // 0~120: pause at start
  if (t < 120)       return {{e:0, show:true}};
  // 120~PUSH_END: push
  if (t < PAUSE1_MS) {{ const s=(t-120)/(PAUSE1_MS-120); const e=s*s*(3-2*s); return {{e,show:true}}; }}
  // PAUSE1_MS~INVIS_MS: pause at tip
  if (t < INVIS_MS)  return {{e:1, show:true}};
  // invisible
  return {{e:0, show:false}};
}}

// ── render loop ───────────────────────────────────────────────────────────────
let geo = null;
new ResizeObserver(() => {{ geo = null; }}).observe(cvs);  // invalidate on resize

function renderLoop(nowMs) {{
  requestAnimationFrame(renderLoop);
  ctx.clearRect(0,0,cvs.width,cvs.height);

  // only draw when frozen (static frame visible)
  if (state !== 'frozen') return;

  if (!geo) geo = computeGeo();
  if (!geo) return;

  const {{e, show}} = arrowState(nowMs);
  drawIndicator(geo, e, show);
}}
requestAnimationFrame(renderLoop);
</script>
</body>
</html>
"""

OUT.write_text(html, encoding='utf-8')
print(f"=> {OUT}  ({OUT.stat().st_size//1024}KB)")
print(f"\nFreeze point: fr{DISPLAY_FR} = {DISPLAY_FR/FPS:.3f}s")
print(f"Freeze dot on progress bar: {DISPLAY_FR/TOTAL*100:.1f}%")
print(f"Static bg: {len(frozen_b64)//1024}KB base64 embedded")
