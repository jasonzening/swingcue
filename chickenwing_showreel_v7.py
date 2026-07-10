#!/usr/bin/env python3
"""
CUE-CHICKENWING-001 展示环 v7
修正 Phase 3:
  - 红线肘节点被推着从 le_ → ge_ (保持红色, 形变是连续位移)
  - 到位(ge_)之后 才渐变为绿色
  = "一条线被推着走→到位→变绿"，不是两条线切换

标准时长 (6.0s total):
  Ph1 红线画出   1000ms
  Ph2 箭头飞入    667ms
  Ph3 推直过程   3333ms  (← Ph3a 推到位 ~2300ms + Ph3b 到位变绿 ~1000ms)
  Ph4 绿线停留   1000ms
"""
import json, cv2, numpy as np, subprocess
from pathlib import Path

ROOT    = Path("/home/jason/projects/swingcue-postest")
CACHE   = ROOT / "engine/kp_cache/batch2/fo-wrong-4.json"
VID     = Path("/mnt/c/Users/jason/Zening/Swingcue/Video/fo-wrong-4.mp4")
PREVIEW = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview")
TMP_AVI = PREVIEW / "cw001_v7_tmp.avi"
OUT_VID = PREVIEW / "cw001_v7_demo.mp4"
OUT_HTML= PREVIEW / "cw001_v7_demo.html"
PREVIEW.mkdir(parents=True, exist_ok=True)

FPS_SRC     = 30.001
TOTAL       = 209
FPS_OUT     = 30
FR_FREEZE   = 149
EASE_NFRAM  = 25
NORMAL_RATE = 0.25

RED   = (17,  15, 228)
GREEN = (12, 220,  48)
LINE_W = 2; NODE_R = 3
ARROW_LEN = 22; ARROW_HW = 14; ARROW_NOTCH = 0.38; STANDOFF = 35

# ── Standard phase durations (v6 verified, now codified) ─────────────────────
FREEZE_NFRAM = 180   # 6000ms total

PH1_END = 30    # 0..29    1000ms  draw-on red
PH2_END = 50    # 30..49    667ms  arrow fly-in
# Ph3 split: 3a = push (red stays red), 3b = recolor (red→green at position)
PH3A_END = 120  # 50..119  2333ms  push (red line moves to green position, stays red)
PH3_END  = 150  # 120..149 1000ms  recolor at position (red→green)
PH4_END  = 180  # 150..179 1000ms  hold green

STANDARD_TIMING = {
    "Ph1_draw_red_ms": int(PH1_END/FPS_OUT*1000),
    "Ph2_arrow_flyin_ms": int((PH2_END-PH1_END)/FPS_OUT*1000),
    "Ph3_push_ms": int((PH3A_END-PH2_END)/FPS_OUT*1000),
    "Ph3b_recolor_ms": int((PH3_END-PH3A_END)/FPS_OUT*1000),
    "Ph3_total_ms": int((PH3_END-PH2_END)/FPS_OUT*1000),
    "Ph4_hold_ms": int((PH4_END-PH3_END)/FPS_OUT*1000),
    "total_ms": int(FREEZE_NFRAM/FPS_OUT*1000),
}

# ── helpers ────────────────────────────────────────────────────────────────────
def ss(t): return t*t*(3-2*t)
def lerp_color(c1,c2,t): return tuple(int(c1[i]+t*(c2[i]-c1[i])) for i in range(3))
def ip(a): return (int(round(float(a[0]))), int(round(float(a[1]))))

def draw_barb(img, tip_np, fwd_np, color):
    perp     = np.array([-fwd_np[1], fwd_np[0]])
    tail_c   = tip_np - fwd_np*ARROW_LEN
    notch_pt = tip_np - fwd_np*(ARROW_LEN*(1-ARROW_NOTCH))
    pts = np.array([ip(tip_np),
                    ip(tail_c+perp*ARROW_HW),
                    ip(notch_pt),
                    ip(tail_c-perp*ARROW_HW)], dtype=np.int32)
    cv2.fillPoly(img, [pts], color)
    cv2.polylines(img, [pts], True, color, 1, cv2.LINE_AA)

# ── geometry fr149 ────────────────────────────────────────────────────────────
with open(CACHE) as f:
    kp_raw = json.load(f)['frames']

kp = {k: np.array([v['x'],v['y']], float)
      for k,v in kp_raw[FR_FREEZE]['persons'][0]['keypoints'].items()}
ls_ = kp['left_shoulder']; le_ = kp['left_elbow']; lw_ = kp['left_wrist']
sw_ = lw_-ls_
t_  = float(np.clip(np.dot(le_-ls_,sw_)/(np.dot(sw_,sw_)+1e-9),0.1,0.9))
ge_ = ls_+t_*sw_
out_= le_-ge_; outN_ = out_/(np.linalg.norm(out_)+1e-9)
seg1 = np.linalg.norm(le_-ls_)
seg2 = np.linalg.norm(lw_-le_)
push_dir = -outN_

print(f"Geometry: ls={ls_.astype(int)} le={le_.astype(int)} lw={lw_.astype(int)}")
print(f"  ge={ge_.astype(int)} le→ge={np.linalg.norm(ge_-le_):.1f}px")
print(f"\nStandard timing:")
for k,v in STANDARD_TIMING.items(): print(f"  {k}: {v}ms")

# ── animated draw ─────────────────────────────────────────────────────────────
def draw_indicator_animated(img, fi):

    # ─ Phase 1: draw-on red arm ────────────────────────────────────────────
    if fi < PH1_END:
        t_ph = ss(fi/(PH1_END-1)) if fi > 0 else 0.0
        draw_len = t_ph*(seg1+seg2)
        if draw_len <= seg1:
            end_pt = ls_+(le_-ls_)*(draw_len/seg1)
            cv2.line(img, ip(ls_), ip(end_pt), RED, LINE_W, cv2.LINE_AA)
        else:
            cv2.line(img, ip(ls_), ip(le_), RED, LINE_W, cv2.LINE_AA)
            rem = draw_len-seg1
            end_pt = le_+(lw_-le_)*(rem/seg2)
            cv2.line(img, ip(le_), ip(end_pt), RED, LINE_W, cv2.LINE_AA)
        cv2.circle(img, ip(ls_), NODE_R, RED, -1, cv2.LINE_AA)
        if draw_len >= seg1:
            cv2.circle(img, ip(le_), NODE_R, RED, -1, cv2.LINE_AA)

    # ─ Phase 2: full red arm + arrow flies in ──────────────────────────────
    elif fi < PH2_END:
        cv2.line(img, ip(ls_), ip(le_), RED, LINE_W, cv2.LINE_AA)
        cv2.line(img, ip(le_), ip(lw_), RED, LINE_W, cv2.LINE_AA)
        for p in [ls_, le_, lw_]:
            cv2.circle(img, ip(p), NODE_R, RED, -1, cv2.LINE_AA)
        t_ph = ss((fi-PH1_END)/(PH2_END-PH1_END-1))
        tip = (le_+outN_*STANDOFF)+t_ph*(le_-(le_+outN_*STANDOFF))
        draw_barb(img, tip, push_dir, RED)

    # ─ Phase 3a: red arm being pushed (stays red, elbow le→ge) ────────────
    elif fi < PH3A_END:
        t_ph = ss((fi-PH2_END)/(PH3A_END-PH2_END-1))
        curr_le = le_+t_ph*(ge_-le_)   # elbow moves along push direction

        # red arm stays red — this is the "being pushed" phase
        cv2.line(img, ip(ls_), ip(curr_le), RED, LINE_W, cv2.LINE_AA)
        cv2.line(img, ip(curr_le), ip(lw_), RED, LINE_W, cv2.LINE_AA)
        for p in [ls_, lw_]:
            cv2.circle(img, ip(p), NODE_R, RED, -1, cv2.LINE_AA)
        cv2.circle(img, ip(curr_le), NODE_R, RED, -1, cv2.LINE_AA)

        # arrow tracks curr_le, standoff shrinks as elbow nears ge
        cur_so = STANDOFF*(1-t_ph)+2
        tip = curr_le+outN_*cur_so
        draw_barb(img, tip, push_dir, RED)

    # ─ Phase 3b: at green position — recolor red→green ────────────────────
    elif fi < PH3_END:
        t_ph = ss((fi-PH3A_END)/(PH3_END-PH3A_END-1))
        # elbow is now AT ge_ (fully pushed); color transitions
        col = lerp_color(RED, GREEN, t_ph)

        # glow fades in as green arrives
        if t_ph > 0.2:
            ga = min(1.0, (t_ph-0.2)/0.5)
            ov = img.copy()
            cv2.line(ov, ip(ls_), ip(lw_), GREEN, 9, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.18*ga, img, 1-0.18*ga, 0, img)

        # arm at final position (ge_) changing color
        cv2.line(img, ip(ls_), ip(ge_), col, LINE_W, cv2.LINE_AA)
        cv2.line(img, ip(ge_), ip(lw_), col, LINE_W, cv2.LINE_AA)
        for p in [ls_, lw_]:
            cv2.circle(img, ip(p), NODE_R, col, -1, cv2.LINE_AA)
        cv2.circle(img, ip(ge_), NODE_R, col, -1, cv2.LINE_AA)

        # arrow fades out after push is done
        if t_ph < 0.4:
            arrow_alpha = 1.0 - t_ph/0.4
            ov3 = img.copy()
            draw_barb(ov3, ge_+outN_*2, push_dir, RED)
            cv2.addWeighted(ov3, arrow_alpha, img, 1-arrow_alpha, 0, img)

    # ─ Phase 4: hold final green ───────────────────────────────────────────
    else:
        ov = img.copy()
        cv2.line(ov, ip(ls_), ip(lw_), GREEN, 9, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.18, img, 0.82, 0, img)
        cv2.line(img, ip(ls_), ip(lw_), GREEN, LINE_W, cv2.LINE_AA)
        for p in [ls_, lw_]:
            cv2.circle(img, ip(p), NODE_R, GREEN, -1, cv2.LINE_AA)
        cv2.circle(img, ip(ge_), max(1,int(NODE_R*0.83)), GREEN, -1, cv2.LINE_AA)

# ── load source frames ────────────────────────────────────────────────────────
print("\nLoading source frames...", flush=True)
cap = cv2.VideoCapture(str(VID))
VW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
VH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
src = []
for _ in range(TOTAL):
    ret, f = cap.read()
    src.append(f if ret else (src[-1] if src else np.zeros((VH,VW,3),np.uint8)))
cap.release()
print(f"  {len(src)} frames {VW}x{VH}", flush=True)

# ── build timeline ─────────────────────────────────────────────────────────────
ease_adv = sum(NORMAL_RATE*(1-ss(i/EASE_NFRAM)) for i in range(EASE_NFRAM))
ease_src_start = FR_FREEZE - ease_adv
timeline = []
pos = 0.0

while pos < ease_src_start:
    timeline.append((int(round(pos)), -1)); pos += NORMAL_RATE

for i in range(EASE_NFRAM):
    timeline.append((int(min(FR_FREEZE,round(pos))), -1))
    pos += NORMAL_RATE*(1-ss(i/EASE_NFRAM))

pos = float(FR_FREEZE)
for fi_local in range(FREEZE_NFRAM):
    timeline.append((FR_FREEZE, fi_local))

for i in range(EASE_NFRAM):
    pos += NORMAL_RATE*ss((i+1)/EASE_NFRAM)
    timeline.append((int(min(TOTAL-1,round(pos))), -1))

while pos < TOTAL:
    timeline.append((int(min(TOTAL-1,round(pos))), -1)); pos += NORMAL_RATE

n_total = len(timeline)
dur_s   = n_total/FPS_OUT
freeze_out_start = next(i for i,(sf,fi) in enumerate(timeline) if fi==0)
freeze_out_end   = next(i for i,(sf,fi) in enumerate(timeline) if fi==FREEZE_NFRAM-1)
print(f"Timeline: {n_total} frames = {dur_s:.1f}s", flush=True)
print(f"Freeze: out_fr {freeze_out_start}..{freeze_out_end}  ({freeze_out_start/FPS_OUT:.2f}~{freeze_out_end/FPS_OUT:.2f}s)", flush=True)

# ── render ─────────────────────────────────────────────────────────────────────
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = cv2.VideoWriter(str(TMP_AVI), fourcc, FPS_OUT, (VW, VH))
assert writer.isOpened()
print(f"Rendering {n_total} frames...", flush=True)
for idx, (sf, fi_local) in enumerate(timeline):
    if idx % 200 == 0:
        print(f"  {idx}/{n_total}  src={sf}  fi={fi_local}", flush=True)
    img = src[max(0,min(TOTAL-1,sf))].copy()
    if fi_local >= 0:
        draw_indicator_animated(img, fi_local)
    writer.write(img)
writer.release()
print("Render done", flush=True)

cmd = ['ffmpeg','-y','-i',str(TMP_AVI),
       '-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p',
       str(OUT_VID)]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0:
    print(r.stderr[-500:]); raise RuntimeError("ffmpeg failed")
TMP_AVI.unlink(missing_ok=True)
sz = OUT_VID.stat().st_size
print(f"=> {OUT_VID}  ({sz//1024}KB)", flush=True)

# ── HTML player ────────────────────────────────────────────────────────────────
fs_s = freeze_out_start/FPS_OUT; fe_s = freeze_out_end/FPS_OUT
fz_l = fs_s/dur_s*100; fz_w = (fe_s-fs_s)/dur_s*100

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SwingCue · Chicken Wing · fo-wrong-4 · v7</title>
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
.hi {{ color:#999; }}
.stage {{ position:relative; width:360px; background:#000;
          border-radius:6px; overflow:hidden; box-shadow:0 12px 40px rgba(0,0,0,.7); }}
#vid {{ width:100%; display:block; }}
#badge {{
  position:absolute; top:10px; right:10px;
  color:#fff; font-size:10px; font-weight:bold; letter-spacing:.12em;
  padding:3px 8px; border-radius:3px; z-index:5;
  opacity:0; background:rgba(228,15,17,0.9); transition:opacity .3s;
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
.prog-fill  {{ position:absolute; height:6px; background:#3a3a5a; border-radius:3px; width:0%; pointer-events:none; z-index:1; }}
.cw-zone {{
  position:absolute; height:6px; border-radius:3px;
  background:rgba(228,15,17,0.32);
  left:{fz_l:.2f}%; width:{fz_w:.2f}%;
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
.legend {{ width:360px; margin-top:10px; display:flex; gap:5px; font-size:9px; }}
.litem {{ flex:1; text-align:center; padding:3px 2px; border-radius:3px; border:1px solid #222; color:#666; }}
</style>
</head>
<body>
<div class="title">
  <span>SWINGCUE</span><span>·</span>
  <span class="hi">fo-wrong-4</span><span>·</span>
  <span>CHICKEN WING v7</span>
</div>
<div class="stage">
  <video id="vid" src="cw001_v7_demo.mp4" preload="auto"></video>
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
    <span style="color:#333">red zone = cue · {dur_s:.0f}s</span>
  </div>
</div>
<div class="legend">
  <div class="litem" style="border-color:#3a1a1a;color:#884">① 红线<br>1000ms</div>
  <div class="litem" style="border-color:#2a2a1a;color:#884">② 箭头<br>667ms</div>
  <div class="litem" style="border-color:#1a2a1a;color:#664;flex:1.5">③a 推移<br>2333ms</div>
  <div class="litem" style="border-color:#1a3a1a;color:#484">③b 变绿<br>1000ms</div>
  <div class="litem" style="border-color:#1a3a1a;color:#484">④ 停留<br>1000ms</div>
</div>
<div class="hint">SPACE play/pause  ·  drag red zone to replay  ·  ← → 0.1s step</div>
<script>
const FS={fs_s:.3f}, FE={fe_s:.3f};
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
  const t=vid.currentTime,d=vid.duration||1;
  prog.value=t/d*10000; fill.style.width=(t/d*100)+'%';
  timeD.textContent=t.toFixed(2)+' / {dur_s:.1f}s';
  const inCue=t>=FS&&t<=FE;
  badge.style.opacity=inCue?'1':'0';
  if(inCue){{
    const pct=(t-FS)/(FE-FS);
    const ph=pct<0.167?'① 画红线':pct<0.278?'② 箭头出现':pct<0.667?'③a 推移(红)':pct<0.833?'③b 到位变绿':'④ 绿线停';
    cueD.textContent='⚠ '+ph; cueD.style.color='rgb(228,80,80)';
  }}else{{cueD.textContent='──'; cueD.style.color='#444';}}
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

print(f"""
╔══════════════════════════════════════════════════════╗
║  CUE_DESIGN_LANGUAGE — 标准指示器动画时长 (定案)      ║
╠══════════════════════════════════════════════════════╣
║  Ph1  红线 draw-on          1000 ms                 ║
║  Ph2  箭头飞入               667 ms                 ║
║  Ph3a 红线推移(到位, 保持红)  2333 ms  ← 核心教学    ║
║  Ph3b 到位变绿               1000 ms                 ║
║  Ph3  推直总时长             3333 ms                 ║
║  Ph4  绿线停留               1000 ms                 ║
║  ─────────────────────────────────────────────        ║
║  总时长(定格段)              6000 ms                 ║
╚══════════════════════════════════════════════════════╝
FPS_OUT=30  FREEZE_NFRAM=180  EASE_NFRAM=25 (0.83s)
""")
