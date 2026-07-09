#!/usr/bin/env python3
"""
CUE-CHICKENWING-001 展示环 v3
- 去掉 barb 箭头
- 红线+绿线静止停留在定格上 (无动画)
- 线宽/节点缩小一半
"""
import json, cv2, numpy as np, base64
from pathlib import Path

ROOT  = Path("/home/jason/projects/swingcue-postest")
CACHE = ROOT / "engine/kp_cache/batch2/fo-wrong-4.json"
VID   = Path("/mnt/c/Users/jason/Zening/Swingcue/Video/fo-wrong-4.mp4")
OUT   = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/cw001_v3_demo.html")
OUT.parent.mkdir(parents=True, exist_ok=True)

FPS    = 30.001
TOTAL  = 209
VW, VH = 720, 1280
DISPLAY_FR = 149

# ── extract display frame as JPEG ─────────────────────────────────────────────
cap = cv2.VideoCapture(str(VID))
cap.set(cv2.CAP_PROP_POS_FRAMES, DISPLAY_FR)
ret, bgr = cap.read(); cap.release()
assert ret
_, jpg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
frozen_b64 = base64.b64encode(jpg.tobytes()).decode()
print(f"Frozen JPEG: {len(jpg.tobytes())//1024}KB → base64 {len(frozen_b64)//1024}KB")

# ── keypoints fr149 ───────────────────────────────────────────────────────────
with open(CACHE) as f:
    kp_raw = json.load(f)['frames']

def get_kp(fi):
    fr = kp_raw[fi]
    if not fr.get('persons'): return {}
    return {k: [round(v['x'],1), round(v['y'],1)]
            for k,v in fr['persons'][0]['keypoints'].items()}

kp149_j = json.dumps(get_kp(DISPLAY_FR))

OUT.write_text(f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SwingCue · Chicken Wing Demo · fo-wrong-4 · v3</title>
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
.stage {{
  position:relative; width:360px;
  background:#000; border-radius:6px; overflow:hidden;
  box-shadow:0 12px 40px rgba(0,0,0,.7);
}}
#vid {{ width:100%; display:block; }}
#frozenImg {{
  position:absolute; inset:0;
  width:100%; height:100%;
  object-fit:fill; display:none;
}}
#cvs {{
  position:absolute; inset:0;
  width:100%; height:100%;
  pointer-events:none;
}}
#continueBtn {{
  position:absolute;
  bottom:52px; left:50%;
  transform:translateX(-50%);
  background:rgba(10,10,20,.82);
  border:1px solid rgba(255,255,255,.18);
  color:#ddd; font-family:inherit;
  font-size:12px; letter-spacing:.08em;
  padding:7px 22px; border-radius:20px;
  cursor:pointer; display:none; z-index:10;
  white-space:nowrap; backdrop-filter:blur(4px);
}}
#continueBtn:hover {{ background:rgba(30,30,50,.9); color:#fff; }}
#badge {{
  position:absolute; top:10px; right:10px;
  background:rgba(228,15,17,.9);
  color:#fff; font-size:10px; font-weight:bold;
  letter-spacing:.12em; padding:3px 8px;
  border-radius:3px; display:none; z-index:5;
}}
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
  <span>SWINGCUE</span><span>·</span>
  <span class="hi">fo-wrong-4</span><span>·</span>
  <span>CUE-CHICKENWING-001  v3</span>
</div>

<div class="stage">
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
  <div class="row" style="justify-content:space-between">
    <span id="cueDisp">──</span>
    <span style="color:#333">{TOTAL}fr · {FPS:.0f}fps</span>
  </div>
</div>
<div class="hint">SPACE play/pause  ·  ← → frame step  ·  drag to scrub  ·  red dot = cue point</div>

<script>
'use strict';
const FPS   = {FPS:.6f};
const TOTAL = {TOTAL};
const VW    = {VW}, VH = {VH};
const FR    = 149;
const SNAP_R = 5;

// colors
const RED   = 'rgb(228,15,17)';
const GREEN = 'rgb(48,220,12)';
const GLOW  = 'rgba(48,220,12,0.18)';
const MUTED = 'rgba(80,80,80,0.7)';

// ── HALF the v2 values ───────────────────────────────────────────────────────
const LINE_W = 2;    // was 4
const NODE_R = 3;    // was 6
const GE_R   = 2.5;  // green elbow dot (was 5)

const KP = {kp149_j};

const vid       = document.getElementById('vid');
const frozenImg = document.getElementById('frozenImg');
const cvs       = document.getElementById('cvs');
const ctx       = cvs.getContext('2d');
const playBtn   = document.getElementById('playBtn');
const progEl    = document.getElementById('prog');
const progFill  = document.getElementById('progFill');
const frameDisp = document.getElementById('frameDisp');
const cueDisp   = document.getElementById('cueDisp');
const badge     = document.getElementById('badge');
const contBtn   = document.getElementById('continueBtn');
const stepB     = document.getElementById('stepB');
const stepF     = document.getElementById('stepF');

// ── canvas sizing ─────────────────────────────────────────────────────────────
function syncCanvas() {{
  const r = vid.getBoundingClientRect();
  cvs.width = r.width; cvs.height = r.height;
}}
vid.addEventListener('loadedmetadata', syncCanvas);
new ResizeObserver(syncCanvas).observe(vid);
syncCanvas();

// ── geometry ──────────────────────────────────────────────────────────────────
function computeGeo() {{
  const scale = cvs.width / VW;
  function sc(n) {{ const p=KP[n]; return p ? [p[0]*scale, p[1]*scale] : null; }}
  const ls=sc('left_shoulder'), le=sc('left_elbow'), lw=sc('left_wrist');
  const rs=sc('right_shoulder'), re=sc('right_elbow'), rw=sc('right_wrist');
  const lh=sc('left_hip'), rh=sc('right_hip');
  if (!ls||!le||!lw) return null;

  // green_elbow = projection of le onto ls-lw line
  const sw=[lw[0]-ls[0], lw[1]-ls[1]];
  const ssq=sw[0]*sw[0]+sw[1]*sw[1]+1e-9;
  const t=Math.max(0.1,Math.min(0.9,
    ((le[0]-ls[0])*sw[0]+(le[1]-ls[1])*sw[1])/ssq));
  const ge=[ls[0]+t*sw[0], ls[1]+t*sw[1]];

  return {{ls,le,lw,rs,re,rw,lh,rh,ge}};
}}

// ── state machine ─────────────────────────────────────────────────────────────
let state = 'playing';
let fired = false;
let dragging = false;

vid.playbackRate = 0.25;

function setFrozen() {{
  state = 'frozen';
  vid.pause();
  vid.currentTime = FR / FPS;
  frozenImg.style.display = 'block';
  badge.style.display = 'block';
  contBtn.style.display = 'block';
  playBtn.textContent = '⏸';
  cueDisp.textContent = '⚠ CHICKEN WING  fr149 · 定格';
  cueDisp.style.color = 'rgb(228,80,80)';
}}

function resumePlay() {{
  state = 'playing';
  frozenImg.style.display = 'none';
  badge.style.display = 'none';
  contBtn.style.display = 'none';
  cueDisp.textContent = '──';
  cueDisp.style.color = '#444';
  vid.currentTime = Math.min(vid.duration||999, (FR+2)/FPS);
  vid.play();
}}

contBtn.addEventListener('click', resumePlay);
vid.addEventListener('play',  () => {{ if (state !== 'frozen') playBtn.textContent='⏸'; }});
vid.addEventListener('pause', () => {{ if (state === 'playing') {{ state='user_paused'; playBtn.textContent='▶'; }} }});

playBtn.addEventListener('click', () => {{
  if (state === 'frozen') {{ resumePlay(); return; }}
  if (vid.paused) {{ state='playing'; vid.play(); }} else vid.pause();
}});

vid.addEventListener('timeupdate', () => {{
  if (dragging) return;
  syncProg();
  if (state === 'playing' && !fired && vid.currentTime * FPS >= FR) {{
    fired = true; setFrozen();
  }}
  if (vid.currentTime * FPS < FR - 20) fired = false;
}});

vid.addEventListener('ended', () => {{ state='user_paused'; playBtn.textContent='▶'; }});

progEl.addEventListener('mousedown',  () => {{ dragging=true; }});
progEl.addEventListener('touchstart', () => {{ dragging=true; }});
progEl.addEventListener('input', () => {{
  vid.currentTime = progEl.value/10000 * (vid.duration||1);
  syncProg();
}});
function onScrubEnd() {{
  dragging = false;
  const frame = vid.currentTime * FPS;
  if (vid.paused && Math.abs(frame-FR) <= SNAP_R && state !== 'frozen') {{
    setFrozen(); fired = true;
  }}
}}
progEl.addEventListener('mouseup',  onScrubEnd);
progEl.addEventListener('touchend', onScrubEnd);

function syncProg() {{
  const pct = vid.duration ? vid.currentTime/vid.duration : 0;
  progEl.value = pct * 10000;
  progFill.style.width = (pct*100)+'%';
  frameDisp.textContent = `fr${{Math.round(vid.currentTime*FPS)}} · ${{vid.currentTime.toFixed(2)}}s`;
}}

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

// ── draw (static — no animation) ──────────────────────────────────────────────
function drawIndicator(geo) {{
  const {{ls,le,lw,rs,re,rw,lh,rh,ge}} = geo;
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';

  // muted skeleton (right arm + hips)
  ctx.strokeStyle = MUTED; ctx.lineWidth = 1.5;
  for (const [a,b] of [[rs,re],[re,rw],[lh,rh]]) {{
    if (!a||!b) continue;
    ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
  }}

  // green glow under green line
  ctx.save();
  ctx.strokeStyle = GLOW; ctx.lineWidth = 7; ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(...ls); ctx.lineTo(...lw); ctx.stroke();
  ctx.restore();

  // green line ls→lw (correct arm path)
  ctx.strokeStyle = GREEN; ctx.lineWidth = LINE_W;
  ctx.beginPath(); ctx.moveTo(...ls); ctx.lineTo(...lw); ctx.stroke();

  // red line ls→le→lw (actual arm — chicken wing)
  ctx.strokeStyle = RED; ctx.lineWidth = LINE_W;
  ctx.beginPath(); ctx.moveTo(...ls); ctx.lineTo(...le); ctx.lineTo(...lw); ctx.stroke();

  // nodes — green: ls, lw, ge (shoulder, wrist, correct-elbow)
  ctx.fillStyle = GREEN;
  for (const p of [ls, lw]) {{
    ctx.beginPath(); ctx.arc(...p, NODE_R, 0, Math.PI*2); ctx.fill();
  }}
  ctx.beginPath(); ctx.arc(...ge, GE_R, 0, Math.PI*2); ctx.fill();

  // nodes — red on top: ls, le, lw
  ctx.fillStyle = RED;
  for (const p of [ls, le, lw]) {{
    ctx.beginPath(); ctx.arc(...p, NODE_R, 0, Math.PI*2); ctx.fill();
  }}
}}

// ── render loop ───────────────────────────────────────────────────────────────
let geo = null;
new ResizeObserver(() => {{ geo = null; }}).observe(cvs);

function renderLoop() {{
  requestAnimationFrame(renderLoop);
  ctx.clearRect(0, 0, cvs.width, cvs.height);
  if (state !== 'frozen') return;
  if (!geo) geo = computeGeo();
  if (geo) drawIndicator(geo);
}}
requestAnimationFrame(renderLoop);
</script>
</body>
</html>
""", encoding='utf-8')

print(f"=> {OUT}  ({OUT.stat().st_size//1024}KB)")
print(f"Freeze point: fr{DISPLAY_FR} = {DISPLAY_FR/FPS:.3f}s  ({round(DISPLAY_FR/TOTAL*100,1)}% progress)")
print(f"LINE_W: 4→2  NODE_R: 6→3  (half size)")
print(f"Arrows: removed")
