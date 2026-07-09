#!/usr/bin/env python3
"""
CUE-CHICKENWING-001 展示环 v4
Time remapping: 一条连续视频流
  慢放(0.25x) → 渐慢(ease-in) → 定格2s(指示器) → 渐快(ease-out) → 继续
全程无切换跳帧; 指示器烧录进定格段
"""
import json, cv2, numpy as np, subprocess
from pathlib import Path

ROOT     = Path("/home/jason/projects/swingcue-postest")
CACHE    = ROOT / "engine/kp_cache/batch2/fo-wrong-4.json"
VID      = Path("/mnt/c/Users/jason/Zening/Swingcue/Video/fo-wrong-4.mp4")
PREVIEW  = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview")
TMP_AVI  = PREVIEW / "cw001_v4_tmp.avi"
OUT_VID  = PREVIEW / "cw001_v4_demo.mp4"
OUT_HTML = PREVIEW / "cw001_v4_demo.html"
PREVIEW.mkdir(parents=True, exist_ok=True)

FPS_SRC      = 30.001
TOTAL        = 209
FPS_OUT      = 30
FR_FREEZE    = 149          # display frame (fr149)
FREEZE_SEC   = 2.0
FREEZE_NFRAM = int(FREEZE_SEC * FPS_OUT)   # 60 output frames
EASE_NFRAM   = 25           # ease in / out each (output frames)
NORMAL_RATE  = 0.25         # source frames advanced per output frame

RED   = (17,  15, 228)   # BGR — RGB(228,15,17)
GREEN = (12, 220,  48)   # BGR — RGB(48,220,12)
LINE_W = 2; NODE_R = 3

# ── 1. Load all source frames ─────────────────────────────────────────────────
print("Loading source frames...", flush=True)
cap = cv2.VideoCapture(str(VID))
VW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
VH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
src_frames = []
for _ in range(TOTAL):
    ret, f = cap.read()
    src_frames.append(f if ret else (src_frames[-1] if src_frames
                                      else np.zeros((VH,VW,3), np.uint8)))
cap.release()
print(f"  {len(src_frames)} frames  {VW}x{VH}", flush=True)

# ── 2. Load keypoints for freeze frame ───────────────────────────────────────
with open(CACHE) as f:
    kp_raw = json.load(f)['frames']

def get_kp(fi):
    fi = max(0, min(TOTAL-1, fi))
    fr = kp_raw[fi]
    if not fr.get('persons'): return {}
    return {k: np.array([v['x'], v['y']], float)
            for k,v in fr['persons'][0]['keypoints'].items()}

kp_freeze = get_kp(FR_FREEZE)

# ── 3. Draw indicator (burned in, no arrows) ──────────────────────────────────
def draw_indicator(img, kp):
    ls = kp.get('left_shoulder'); le = kp.get('left_elbow')
    lw = kp.get('left_wrist')
    if ls is None or le is None or lw is None: return

    sw  = lw - ls
    t   = float(np.clip(np.dot(le-ls, sw)/(np.dot(sw,sw)+1e-9), 0.1, 0.9))
    ge  = ls + t * sw

    def ip(a): return (int(round(float(a[0]))), int(round(float(a[1]))))

    # glow under green line
    ov = img.copy()
    cv2.line(ov, ip(ls), ip(lw), GREEN, 9, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.18, img, 0.82, 0, img)

    # green line (correct arm path)
    cv2.line(img, ip(ls), ip(lw), GREEN, LINE_W, cv2.LINE_AA)
    # red arm (chicken wing)
    cv2.line(img, ip(ls), ip(le), RED, LINE_W, cv2.LINE_AA)
    cv2.line(img, ip(le), ip(lw), RED, LINE_W, cv2.LINE_AA)

    # nodes — green: shoulder, wrist, correct-elbow
    for p in [ls, lw]:
        cv2.circle(img, ip(p), NODE_R, GREEN, -1, cv2.LINE_AA)
    cv2.circle(img, ip(ge), max(1, int(NODE_R*0.83)), GREEN, -1, cv2.LINE_AA)
    # nodes — red on top: shoulder, elbow, wrist
    for p in [ls, le, lw]:
        cv2.circle(img, ip(p), NODE_R, RED, -1, cv2.LINE_AA)

# ── 4. Build time-remapped timeline ──────────────────────────────────────────
# Compute source advance during ease-in with smoothstep rate (1→0)
ease_advance = sum(
    NORMAL_RATE * (1 - (i/EASE_NFRAM)**2 * (3 - 2*(i/EASE_NFRAM)))
    for i in range(EASE_NFRAM)
)
ease_src_start = FR_FREEZE - ease_advance
print(f"ease_advance={ease_advance:.2f}  ease_src_start={ease_src_start:.2f}", flush=True)

# Each entry: (source_frame_int, draw_indicator: bool)
timeline = []
src = 0.0

def ss(t): return t*t*(3-2*t)   # smoothstep

# Phase 1 — normal 0.25x play
while src < ease_src_start:
    timeline.append((int(round(src)), False))
    src += NORMAL_RATE

# Phase 2 — ease in: rate 0.25 → 0 (smoothstep)
for i in range(EASE_NFRAM):
    timeline.append((int(min(FR_FREEZE, round(src))), False))
    src += NORMAL_RATE * (1 - ss(i / EASE_NFRAM))

src = float(FR_FREEZE)   # snap

# Phase 3 — freeze: fr149 repeated, indicator on
for _ in range(FREEZE_NFRAM):
    timeline.append((FR_FREEZE, True))

# Phase 4 — ease out: rate 0 → 0.25 (smoothstep)
for i in range(EASE_NFRAM):
    src += NORMAL_RATE * ss((i+1) / EASE_NFRAM)
    timeline.append((int(min(TOTAL-1, round(src))), False))

# Phase 5 — normal 0.25x to end
while src < TOTAL:
    timeline.append((int(min(TOTAL-1, round(src))), False))
    src += NORMAL_RATE

n_total  = len(timeline)
dur_s    = n_total / FPS_OUT
# locate freeze section in output timeline
freeze_out_start = next(i for i,(fr,ind) in enumerate(timeline) if ind)
freeze_out_end   = len(timeline) - next(i for i,(fr,ind) in enumerate(reversed(timeline)) if ind) - 1
print(f"Timeline: {n_total} frames = {dur_s:.1f}s", flush=True)
print(f"Freeze section: out_fr {freeze_out_start}..{freeze_out_end}  ({freeze_out_start/FPS_OUT:.2f}s ~ {freeze_out_end/FPS_OUT:.2f}s)")

# ── 5. Render to MJPG AVI ────────────────────────────────────────────────────
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = cv2.VideoWriter(str(TMP_AVI), fourcc, FPS_OUT, (VW, VH))
assert writer.isOpened(), "VideoWriter failed"

print(f"Rendering {n_total} frames...", flush=True)
for idx, (sf, do_ind) in enumerate(timeline):
    if idx % 150 == 0:
        print(f"  {idx}/{n_total}  src_fr={sf}  ind={do_ind}", flush=True)
    img = src_frames[max(0, min(TOTAL-1, sf))].copy()
    if do_ind:
        draw_indicator(img, kp_freeze)
    writer.write(img)
writer.release()
print("Render complete", flush=True)

# ── 6. Re-encode to H.264 MP4 ────────────────────────────────────────────────
cmd = ['ffmpeg', '-y',
       '-i', str(TMP_AVI),
       '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
       '-pix_fmt', 'yuv420p',
       str(OUT_VID)]
print(f"Encoding...", flush=True)
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0:
    print("ffmpeg stderr:", r.stderr[-500:])
    raise RuntimeError("ffmpeg failed")
TMP_AVI.unlink(missing_ok=True)
sz = OUT_VID.stat().st_size
print(f"=> {OUT_VID}  ({sz//1024//1024}MB {sz//1024}KB)", flush=True)

# ── 7. HTML player ────────────────────────────────────────────────────────────
fs_s  = freeze_out_start / FPS_OUT
fe_s  = freeze_out_end   / FPS_OUT
fz_pct_l = fs_s / dur_s * 100
fz_pct_w = (fe_s - fs_s) / dur_s * 100

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SwingCue · Chicken Wing · fo-wrong-4 · v4</title>
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{
  background:#0a0a10; color:#ccc;
  font-family:'SF Mono',Consolas,monospace;
  display:flex; flex-direction:column; align-items:center;
  padding:20px 16px; min-height:100vh;
}}
.title {{ font-size:11px; color:#555; letter-spacing:.1em; margin-bottom:12px;
          display:flex; gap:10px; align-items:center; }}
.title .hi {{ color:#999; }}
.stage {{ position:relative; width:360px; background:#000;
           border-radius:6px; overflow:hidden; box-shadow:0 12px 40px rgba(0,0,0,.7); }}
#vid {{ width:100%; display:block; }}
#badge {{
  position:absolute; top:10px; right:10px;
  color:#fff; font-size:10px; font-weight:bold; letter-spacing:.12em;
  padding:3px 8px; border-radius:3px; z-index:5;
  background:rgba(228,15,17,0); transition:background .25s;
}}
.controls {{ width:360px; margin-top:10px; display:flex; flex-direction:column; gap:7px; }}
.row {{ display:flex; align-items:center; gap:8px; }}
#playBtn {{
  background:#1c1c2c; border:1px solid #333; color:#ddd;
  font-size:15px; width:34px; height:34px;
  border-radius:4px; cursor:pointer; flex-shrink:0;
  display:flex; align-items:center; justify-content:center;
}}
#playBtn:hover {{ background:#252540; }}
.prog-wrap {{ flex:1; position:relative; height:34px; display:flex; align-items:center; }}
.prog-track {{ position:absolute; width:100%; height:6px; background:#1e1e2e; border-radius:3px; }}
.prog-fill  {{ position:absolute; height:6px; background:#3a3a5a; border-radius:3px;
               width:0%; pointer-events:none; z-index:1; }}
.cw-zone {{
  position:absolute; height:6px; border-radius:3px;
  background:rgba(228,15,17,0.32);
  left:{fz_pct_l:.2f}%; width:{fz_pct_w:.2f}%;
  top:50%; transform:translateY(-50%);
  pointer-events:none; z-index:2;
}}
#prog {{
  -webkit-appearance:none; width:100%; height:6px;
  border-radius:3px; background:transparent;
  outline:none; cursor:pointer; position:relative; z-index:3;
}}
#prog::-webkit-slider-thumb {{
  -webkit-appearance:none; width:14px; height:14px;
  border-radius:50%; background:#ccc; cursor:pointer;
}}
.speed-badge {{
  font-size:10px; color:#666; background:#141420;
  border:1px solid #2a2a3a; padding:2px 7px; border-radius:3px;
  white-space:nowrap; flex-shrink:0;
}}
.step-btn {{ background:#141420; border:1px solid #2a2a3a; color:#666;
             font-size:11px; padding:2px 9px; border-radius:3px; cursor:pointer; }}
.step-btn:hover {{ color:#bbb; background:#1c1c30; }}
#timeD {{ color:#777; font-size:10px; }}
#cueD  {{ color:#444; font-size:10px; }}
.hint {{ font-size:9px; color:#333; margin-top:6px; text-align:center; letter-spacing:.05em; }}
</style>
</head>
<body>
<div class="title">
  <span>SWINGCUE</span><span>·</span>
  <span class="hi">fo-wrong-4</span><span>·</span>
  <span>CHICKEN WING v4</span>
</div>
<div class="stage">
  <video id="vid" src="cw001_v4_demo.mp4" preload="auto"></video>
  <div id="badge">⚠ CHICKEN WING</div>
</div>
<div class="controls">
  <div class="row">
    <button id="playBtn">▶</button>
    <div class="prog-wrap">
      <div class="prog-track"></div>
      <div class="prog-fill" id="progFill"></div>
      <div class="cw-zone"></div>
      <input type="range" id="prog" min="0" max="10000" value="0">
    </div>
    <span class="speed-badge">time remapped</span>
  </div>
  <div class="row" style="justify-content:space-between">
    <div class="row" style="gap:4px">
      <button class="step-btn" id="stepB">◀ 0.1s</button>
      <button class="step-btn" id="stepF">0.1s ▶</button>
    </div>
    <span id="timeD">0.00 / {dur_s:.1f}s</span>
  </div>
  <div class="row" style="justify-content:space-between">
    <span id="cueD">──</span>
    <span style="color:#333">red zone = cue · {dur_s:.0f}s total</span>
  </div>
</div>
<div class="hint">SPACE play/pause  ·  drag to scrub  ·  ← → 0.1s step</div>

<script>
const FS = {fs_s:.3f}, FE = {fe_s:.3f};
const vid=document.getElementById('vid');
const playBtn=document.getElementById('playBtn');
const prog=document.getElementById('prog');
const fill=document.getElementById('progFill');
const timeD=document.getElementById('timeD');
const cueD=document.getElementById('cueD');
const badge=document.getElementById('badge');

vid.addEventListener('play',()=>{{playBtn.textContent='⏸';}});
vid.addEventListener('pause',()=>{{playBtn.textContent='▶';}});
playBtn.addEventListener('click',()=>{{vid.paused?vid.play():vid.pause();}});

vid.addEventListener('timeupdate',()=>{{
  const t=vid.currentTime, d=vid.duration||1;
  prog.value=t/d*10000;
  fill.style.width=(t/d*100)+'%';
  timeD.textContent=t.toFixed(2)+' / {dur_s:.1f}s';
  const inCue=t>=FS&&t<=FE;
  badge.style.background=inCue?'rgba(228,15,17,0.9)':'rgba(228,15,17,0)';
  cueD.textContent=inCue?'⚠ CHICKEN WING  定格演示':'──';
  cueD.style.color=inCue?'rgb(228,80,80)':'#444';
}});

prog.addEventListener('input',()=>{{vid.currentTime=prog.value/10000*(vid.duration||1);}});
document.getElementById('stepB').addEventListener('click',()=>{{vid.currentTime=Math.max(0,vid.currentTime-.1);}});
document.getElementById('stepF').addEventListener('click',()=>{{vid.currentTime=Math.min(vid.duration||999,vid.currentTime+.1);}});
document.addEventListener('keydown',e=>{{
  if(e.target.tagName==='INPUT') return;
  if(e.key===' '){{e.preventDefault();vid.paused?vid.play():vid.pause();}}
  if(e.key==='ArrowRight') vid.currentTime=Math.min(vid.duration||999,vid.currentTime+.1);
  if(e.key==='ArrowLeft')  vid.currentTime=Math.max(0,vid.currentTime-.1);
}});
</script>
</body>
</html>"""

OUT_HTML.write_text(html, encoding='utf-8')
print(f"=> {OUT_HTML}")
print(f"\nSUMMARY")
print(f"  Output: {n_total} frames @ {FPS_OUT}fps = {dur_s:.1f}s")
print(f"  Ease in:  out fr {freeze_out_start-EASE_NFRAM}..{freeze_out_start-1}  ({(freeze_out_start-EASE_NFRAM)/FPS_OUT:.2f}s ~ {(freeze_out_start)/FPS_OUT:.2f}s)")
print(f"  Freeze:   out fr {freeze_out_start}..{freeze_out_end}  ({fs_s:.2f}s ~ {fe_s:.2f}s)  indicator ON")
print(f"  Ease out: out fr {freeze_out_end+1}..{freeze_out_end+EASE_NFRAM}  ({fe_s:.2f}s ~ {(freeze_out_end+EASE_NFRAM)/FPS_OUT:.2f}s)")
print(f"  HTML cw-zone: {fz_pct_l:.1f}% ~ {fz_pct_l+fz_pct_w:.1f}%")
