#!/usr/bin/env python3
"""
CUE-CHICKENWING-001 展示环 v11
闪光领先/包裹变色 (绝不落后):
  flash 起点 fi=134 (线仍纯红) → 颜色在 fi=141 才开始渐变 (flash已亮47%)
  fi=149 峰值: flash最亮 + 颜色同帧变100%绿
  fi=150..158 flash衰减 → 稳定绿线
  时间线: 红线推进(纯红) → [先闪] → [闪光中红变绿] → 光退 → 稳定绿
"""
import json, cv2, numpy as np, subprocess
from pathlib import Path

ROOT    = Path("/home/jason/projects/swingcue-postest")
CACHE   = ROOT / "engine/kp_cache/batch2/fo-wrong-4.json"
VID     = Path("/mnt/c/Users/jason/Zening/Swingcue/Video/fo-wrong-4.mp4")
PREVIEW = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview")
TMP_AVI = PREVIEW / "cw001_v12_tmp.avi"
OUT_VID = PREVIEW / "cw001_v12_demo.mp4"
OUT_HTML= PREVIEW / "cw001_v12_demo.html"
PREVIEW.mkdir(parents=True, exist_ok=True)

TOTAL       = 209
FPS_OUT     = 30
FR_FREEZE   = 149
EASE_NFRAM  = 25
NORMAL_RATE = 0.25

RED   = (17,  15, 228)
GREEN = (12, 220,  48)
WHITE = (255, 255, 255)
LINE_W = 2; NODE_R = 3
ARROW_LEN = 22; ARROW_HW = 14; ARROW_NOTCH = 0.38; STANDOFF = 35

FREEZE_NFRAM = 195   # 6000ms + 500ms Ph0 = 6500ms total  (PH0=15fr)
PH0_END  = 15    #  500ms 问题名称+指向箭头
PH1_END  = 45    # +1000ms draw-on red
PH2_END  = 65    #  +667ms arrow fly-in
PH3_END  = 165   # +3333ms push+recolor
PH4_END  = 195   # +1000ms hold green

# ── v11: flash 领先颜色, 颜色锁在 flash 窗口内部 ─────────────────────────────
# fi=134: flash开始上升 (线仍100%红)
# fi=141: 颜色才开始渐变 (此时flash已爬升47%)  ← flash领先颜色7帧
# fi=149: flash峰值 + 颜色100%绿 (同帧)
# fi=150..158: flash衰减, 稳定绿线
FLASH_PEAK  = PH3_END - 1   # fi=164 (shifted by PH0=15)
FLASH_RISE  = 15             # flash 从 fi=149 开始上升 (领先颜色7帧)
FLASH_FALL  = 9
COLOR_START_FI = FLASH_PEAK - 8   # fi=156
COLOR_END_FI   = FLASH_PEAK       # fi=164

def flash_intensity(fi):
    """Triangle curve peaking at FLASH_PEAK, 0 outside window."""
    if fi < FLASH_PEAK - FLASH_RISE or fi > FLASH_PEAK + FLASH_FALL:
        return 0.0
    if fi <= FLASH_PEAK:
        raw = (fi - (FLASH_PEAK - FLASH_RISE)) / FLASH_RISE
    else:
        raw = 1.0 - (fi - FLASH_PEAK) / FLASH_FALL
    return raw * raw * (3 - 2*raw)   # smoothstep for soft ramp

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

def apply_flash(img, fint, curr_elbow):
    """Overlay flash burst onto img. fint in [0,1]."""
    if fint <= 0: return

    # ① wide glow explosion (3 widths)
    ov = img.copy()
    for gw in [38, 24, 14]:
        cv2.line(ov, ip(ls_), ip(lw_), GREEN, gw, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.60*fint, img, 1-0.60*fint, 0, img)

    # ② bright white-tinted core at peak
    core_col = lerp_color(GREEN, WHITE, 0.50*fint)
    cv2.line(img, ip(ls_), ip(lw_), core_col, LINE_W+1, cv2.LINE_AA)

    # ③ elbow node pulse — expands proportional to intensity
    pulse_r = int(NODE_R + 8*fint)
    pulse_col = lerp_color(GREEN, WHITE, 0.55*fint)
    cv2.circle(img, ip(curr_elbow), pulse_r, pulse_col, -1, cv2.LINE_AA)

    # ④ shoulder + wrist nodes pulse (smaller)
    for p in [ls_, lw_]:
        pr = int(NODE_R + 4*fint)
        cv2.circle(img, ip(p), pr, pulse_col, -1, cv2.LINE_AA)

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
print(f"Flash window: fi {FLASH_PEAK-FLASH_RISE}..{FLASH_PEAK+FLASH_FALL}  peak=fi{FLASH_PEAK}")
print(f"  Rise: {FLASH_RISE} fr (fi {FLASH_PEAK-FLASH_RISE}..{FLASH_PEAK}); Fall: {FLASH_FALL} fr")
print(f"  Color starts fi={COLOR_START_FI} (flash already {(FLASH_RISE-8)/FLASH_RISE*100:.0f}% risen)")
print(f"  Total flash duration: {(FLASH_RISE+FLASH_FALL+1)/FPS_OUT*1000:.0f}ms")

# ── animated draw ─────────────────────────────────────────────────────────────
def draw_indicator_animated(img, fi):
    fint = flash_intensity(fi)

    # ── Ph0: 问题名称 + 指向箭头 (CUE_DESIGN_LANGUAGE §Ph0) ─────────────────
    if fi < PH0_END:
        t_ph = ss(fi / max(PH0_END-1, 1))
        alpha = min(1.0, t_ph * 3.0)   # 快速淡入

        # 文字: "鸡翅膀 / Chicken Wing"
        # 位置: 画面左上角, 不挡身体
        VH_img, VW_img = img.shape[:2]
        label1 = "\u9e21\u7fc5\u8180"          # 鸡翅膀
        label2 = "Chicken Wing"
        if alpha > 0.05:
            ov = img.copy()
            cv2.putText(ov, label1, (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, RED, 3, cv2.LINE_AA)
            cv2.putText(ov, label2, (30, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, RED, 2, cv2.LINE_AA)
            # 指向箭头: 从文字右侧 → 左肘 (错误部位)
            txt_end = np.array([310.0, 112.0])   # 文字末端右侧
            elbow_pt = le_.copy()                  # 左肘关键点
            arrow_vec = elbow_pt - txt_end
            arrow_norm = arrow_vec / (np.linalg.norm(arrow_vec) + 1e-9)
            tip = elbow_pt - arrow_norm * 30       # tip 距肘30px
            draw_barb(ov, tip, arrow_norm, RED)
            # 指示线
            cv2.line(ov, ip(txt_end), ip(tip - arrow_norm*22),
                     RED, LINE_W, cv2.LINE_AA)
            cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)
        return

    # ── Ph1: draw-on red arm ──────────────────────────────────────────────────
    if fi < PH1_END:
        t_ph = ss((fi-PH0_END)/max(PH1_END-PH0_END-1,1)) if fi > PH0_END else 0.0
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

    # ── Ph2: full red arm + arrow flies in ────────────────────────────────────
    elif fi < PH2_END:
        cv2.line(img, ip(ls_), ip(le_), RED, LINE_W, cv2.LINE_AA)
        cv2.line(img, ip(le_), ip(lw_), RED, LINE_W, cv2.LINE_AA)
        for p in [ls_, le_, lw_]:
            cv2.circle(img, ip(p), NODE_R, RED, -1, cv2.LINE_AA)
        t_ph = ss((fi-PH1_END)/(PH2_END-PH1_END-1))
        tip  = (le_+outN_*STANDOFF) + t_ph*(le_-(le_+outN_*STANDOFF))
        draw_barb(img, tip, push_dir, RED)

    # ── Ph3: push (纯红推进) + flash领先上升 + 闪光中变绿 ───────────────────────
    elif fi < PH3_END:
        t_push = ss((fi-PH2_END)/(PH3_END-PH2_END-1))
        curr_le = le_ + t_push*(ge_-le_)

        # v11: 颜色锁在 fi=141..149 (8帧), fi<141线保持纯红
        if fi < COLOR_START_FI:
            t_col = 0.0
        elif fi >= COLOR_END_FI:
            t_col = 1.0
        else:
            t_col = ss((fi - COLOR_START_FI) / (COLOR_END_FI - COLOR_START_FI))
        col = lerp_color(RED, GREEN, t_col)

        # glow behind line (proportional to t_col)
        if t_col > 0.05:
            ga = min(1.0, t_col/0.6)
            ov = img.copy()
            cv2.line(ov, ip(ls_), ip(lw_), GREEN, 9, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.18*ga, img, 1-0.18*ga, 0, img)

        # arm lines
        cv2.line(img, ip(ls_), ip(curr_le), col, LINE_W, cv2.LINE_AA)
        cv2.line(img, ip(curr_le), ip(lw_),  col, LINE_W, cv2.LINE_AA)
        for p in [ls_, lw_]:
            cv2.circle(img, ip(p), NODE_R, col, -1, cv2.LINE_AA)
        cv2.circle(img, ip(curr_le), NODE_R, col, -1, cv2.LINE_AA)

        if t_col > 0.3:
            ga2 = min(1.0,(t_col-0.3)/0.3)
            ov2 = img.copy()
            cv2.circle(ov2, ip(ge_), max(1,int(NODE_R*0.83)), GREEN, -1, cv2.LINE_AA)
            cv2.addWeighted(ov2, ga2, img, 1-ga2, 0, img)

        # arrow tracks curr_le, color synced with t_col
        cur_so = STANDOFF*(1-t_push)+2
        tip = curr_le + outN_*cur_so
        draw_barb(img, tip, push_dir, col)

        # flash rises in Ph3's last frames (fi 139..149)
        if fint > 0:
            apply_flash(img, fint, curr_le)

    # ── Ph4: hold green (flash decays from peak in first ~9 frames) ──────────
    else:
        # base stable green
        ov = img.copy()
        cv2.line(ov, ip(ls_), ip(lw_), GREEN, 9, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.18, img, 0.82, 0, img)
        cv2.line(img, ip(ls_), ip(lw_), GREEN, LINE_W, cv2.LINE_AA)
        for p in [ls_, lw_]:
            cv2.circle(img, ip(p), NODE_R, GREEN, -1, cv2.LINE_AA)
        cv2.circle(img, ip(ge_), max(1,int(NODE_R*0.83)), GREEN, -1, cv2.LINE_AA)

        # flash decays (fi 150..158)
        if fint > 0:
            apply_flash(img, fint, ge_)

# ── load source frames ─────────────────────────────────────────────────────────
print("Loading source frames...", flush=True)
cap = cv2.VideoCapture(str(VID))
VW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
VH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
src = []
for _ in range(TOTAL):
    ret, f = cap.read()
    src.append(f if ret else (src[-1] if src else np.zeros((VH,VW,3),np.uint8)))
cap.release()
print(f"  {len(src)} fr {VW}x{VH}", flush=True)

# ── timeline ──────────────────────────────────────────────────────────────────
ease_adv = sum(NORMAL_RATE*(1-ss(i/EASE_NFRAM)) for i in range(EASE_NFRAM))
timeline = []
pos = 0.0

while pos < FR_FREEZE - ease_adv:
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
print(f"Timeline: {n_total} fr = {dur_s:.1f}s", flush=True)
print(f"Freeze: {freeze_out_start/FPS_OUT:.2f}~{freeze_out_end/FPS_OUT:.2f}s", flush=True)

# ── render ─────────────────────────────────────────────────────────────────────
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = cv2.VideoWriter(str(TMP_AVI), fourcc, FPS_OUT, (VW,VH))
assert writer.isOpened()
print(f"Rendering {n_total} frames...", flush=True)
for idx, (sf, fi_local) in enumerate(timeline):
    if idx % 200 == 0: print(f"  {idx}/{n_total}", flush=True)
    img = src[max(0,min(TOTAL-1,sf))].copy()
    if fi_local >= 0:
        draw_indicator_animated(img, fi_local)
    writer.write(img)
writer.release()
print("Render done", flush=True)

r = subprocess.run(['ffmpeg','-y','-i',str(TMP_AVI),
    '-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p',
    str(OUT_VID)], capture_output=True, text=True)
if r.returncode != 0: print(r.stderr[-400:]); raise RuntimeError("ffmpeg failed")
TMP_AVI.unlink(missing_ok=True)
print(f"=> {OUT_VID}  ({OUT_VID.stat().st_size//1024}KB)", flush=True)

# ── HTML player ────────────────────────────────────────────────────────────────
fs_s = freeze_out_start/FPS_OUT; fe_s = freeze_out_end/FPS_OUT
fz_l = fs_s/dur_s*100; fz_w = (fe_s-fs_s)/dur_s*100

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SwingCue · Chicken Wing · fo-wrong-4 · v11</title>
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
.prog-fill  {{ position:absolute; height:6px; background:#3a3a5a; border-radius:3px;
               width:0%; pointer-events:none; z-index:1; }}
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
.speed-badge {{ font-size:10px; color:#666; background:#141420;
  border:1px solid #2a2a3a; padding:2px 7px; border-radius:3px;
  white-space:nowrap; flex-shrink:0; }}
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
  <span>CHICKEN WING v11</span>
</div>
<div class="stage">
  <video id="vid" src="cw001_v11_demo.mp4" preload="auto"></video>
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
  <div class="litem" style="border-color:#3a1a1a;color:#884">① 红线 1000</div>
  <div class="litem" style="border-color:#2a2a1a;color:#884">② 箭头 667</div>
  <div class="litem" style="border-color:#2a3a1a;color:#574;flex:2">③ 推移+变色 3333ms ★</div>
  <div class="litem" style="border-color:#1a4a2a;color:#5c5">④⚡ 1000</div>
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
    const ph=pct<0.167?'① 画红线':pct<0.278?'② 箭头出现':pct<0.883?'③ 推移→变色⚡':'④ 绿线稳定';
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
Flash alignment:
  FLASH_PEAK  = fi {FLASH_PEAK}  (last frame of Ph3, t_push=1.0, color=100% green)
  Rise window = fi {FLASH_PEAK-FLASH_RISE}..{FLASH_PEAK}  ({FLASH_RISE} fr within Ph3)
  Fall window = fi {FLASH_PEAK}..{FLASH_PEAK+FLASH_FALL}  ({FLASH_FALL} fr within Ph4)
  Total flash = {(FLASH_RISE+FLASH_FALL+1)/FPS_OUT*1000:.0f}ms
  Peak coincides exactly with "color change complete" moment
""")
