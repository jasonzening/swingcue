#!/usr/bin/env python3
"""
CUE-SLIDE-001 showreel v1
time remapping 定格在 fr140 (首次出界帧)
黄→红闪 + 箭头推回绿线 + 闪变绿
frame track: 髋外缘线 = lead_hip_kp_x + 0.4*hip_w
"""
import json, cv2, numpy as np, subprocess, math
from pathlib import Path

ROOT    = Path("/home/jason/projects/swingcue-postest")
CACHE   = ROOT / "engine/kp_cache/batch2/fo-wrong-4.json"
VID     = Path("/mnt/c/Users/jason/Zening/Swingcue/Video/fo-wrong-4.mp4")
PREVIEW = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview")
TMP_AVI = PREVIEW / "slide001_v4_tmp.avi"
OUT_VID = PREVIEW / "slide001_v4_demo.mp4"
PREVIEW.mkdir(parents=True, exist_ok=True)

with open(CACHE) as f:
    data = json.load(f)
frames_data = data['frames']

TOTAL       = 209
FPS_OUT     = 30
A           = 85
IMP         = 147
FR_FREEZE   = 147       # impact帧 = 越界最明显(outer=410, outside by 13px)

# 颜色
YELLOW = (0,  210, 255)    # BGR: 黄
RED    = (17,  15, 228)    # BGR: 红
GREEN  = (12, 220,  48)    # BGR: 绿
WHITE  = (255, 255, 255)
LINE_W = 2
NODE_R = 3

# ── geometry helpers ─────────────────────────────────────────────────────────
def kpt(fi, name):
    fr = frames_data[fi]
    if not fr['persons']: return None
    v = fr['persons'][0]['keypoints'].get(name)
    return np.array([v['x'], v['y']]) if v else None

def outer_x(fi):
    """lead 侧髋外缘 x = lead_hip_kp_x + 0.4 * hip_w"""
    lh = kpt(fi,'left_hip'); rh = kpt(fi,'right_hip')
    if lh is None or rh is None: return None
    hw = float(lh[0] - rh[0])
    return float(lh[0] + 0.4 * hw)

la_A  = kpt(A,  'left_ankle')
ra_A  = kpt(A,  'right_ankle')
ls_A  = kpt(A,  'left_shoulder')
rs_A  = kpt(A,  'right_shoulder')
SW0   = float(np.linalg.norm(rs_A - ls_A))
lh_a0 = kpt(A, 'left_hip'); rh_a0 = kpt(A, 'right_hip')
HW0   = float(np.linalg.norm(lh_a0 - rh_a0))   # address hip width

X_GREEN_LEAD  = float(la_A[0])   # 397: lead  脚绿线
X_GREEN_TRAIL = float(ra_A[0])   # 262: trail 脚绿线

# freeze 帧的几何
lh_F  = kpt(FR_FREEZE, 'left_hip')
rh_F  = kpt(FR_FREEZE, 'right_hip')
hw_F  = float(lh_F[0] - rh_F[0])
OX_F  = float(lh_F[0] + 0.4 * hw_F)   # 红线 x at freeze = 399px
HIP_Y = float((lh_F[1] + rh_F[1]) / 2)

# 绿目标 x = lead 绿线
X_GREEN = X_GREEN_LEAD

# 箭头方向: 从红线(右/lead侧)往左(trail侧)推回
PUSH_DIR = np.array([-1.0, 0.0])
STANDOFF = 35

# 线段高度 (fr FR_FREEZE)
nose_F = kpt(FR_FREEZE,'nose')
la_F   = kpt(FR_FREEZE,'left_ankle')
LT = max(int((nose_F[1] if nose_F is not None else 250) - 50), 15)
LB = min(int((la_F[1]  if la_F  is not None else 900) + 50), 960)

def ip(p): return (int(round(float(p[0]))), int(round(float(p[1]))))

def draw_barb(img, tip, fwd, col, length=22, hw=14, notch=0.38):
    fwd = np.array(fwd, float); fwd /= np.linalg.norm(fwd) + 1e-9
    perp = np.array([-fwd[1], fwd[0]])
    tail = tip - fwd * length
    notch_pt = tip - fwd * (length * (1 - notch))
    pts = np.array([ip(tip), ip(tail+perp*hw), ip(notch_pt), ip(tail-perp*hw)], dtype=np.int32)
    cv2.fillPoly(img, [pts], col)
    cv2.polylines(img, [pts], True, col, 1, cv2.LINE_AA)

def draw_guide_arrow(img, src, tgt, color=YELLOW, pulse=1.0):
    """
    CUE_DESIGN_LANGUAGE §引导箭头:
    细长带箭头引线 (从标注框 → 问题部位)。
    和 barb 改正箭头 严格区分, 不可混用。
    """
    av = np.array(tgt, dtype=float) - np.array(src, dtype=float)
    dist = np.linalg.norm(av)
    if dist < 1: return
    an = av / dist
    tip = np.array(tgt, dtype=float) - an * 12
    c = tuple(int(x * (0.5 + 0.5 * pulse)) for x in color)
    cv2.line(img, ip(src), ip(tip), c, 1, cv2.LINE_AA)
    perp = np.array([-an[1], an[0]])
    notch = tip - an * 11
    pts = np.array([ip(tip), ip(notch + perp*5), ip(notch - perp*5)], dtype=np.int32)
    cv2.fillPoly(img, [pts], c)

def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + t*(c2[i]-c1[i])) for i in range(3))

def ss(t): return t*t*(3-2*t)

HIP_CONF_THR = 0.50    # RTMPose 置信度低于此 = 遮挡
HIP_W_THR    = 0.25    # hip_w / HW0 低于此 = 几乎侧身/遮挡

def hip_visible(fi):
    """
    True = 越界侧髋关节可见, 可以画线.
    遮挡判定 (任一触发即停):
      1. left_hip 或 right_hip RTMPose score < HIP_CONF_THR
      2. hip_w / HW0 < HIP_W_THR (身体转到近侧面, 髋宽缩小)
    """
    fr = frames_data[fi]
    if not fr['persons']: return False
    kp = fr['persons'][0]['keypoints']
    lh_sc = kp.get('left_hip',  {}).get('score', 0.0)
    rh_sc = kp.get('right_hip', {}).get('score', 0.0)
    if lh_sc < HIP_CONF_THR or rh_sc < HIP_CONF_THR:
        return False
    lh = kpt(fi,'left_hip'); rh = kpt(fi,'right_hip')
    if lh is None or rh is None: return False
    hw = float(np.linalg.norm(lh - rh))
    if hw / HW0 < HIP_W_THR:
        return False
    return True


def draw_vline(img, x, col, lw=LINE_W):
    cv2.line(img, (int(x), LT), (int(x), LB), col, lw, cv2.LINE_AA)

def apply_flash(img, fint, x_center):
    """green爆闪 + 白核心"""
    if fint <= 0: return
    ov = img.copy()
    for gw in [28, 18, 10]:
        cv2.line(ov, (int(x_center), LT), (int(x_center), LB), GREEN, gw, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.55*fint, img, 1-0.55*fint, 0, img)
    core = lerp_color(GREEN, WHITE, 0.5*fint)
    cv2.line(img, (int(x_center), LT), (int(x_center), LB), core, LINE_W, cv2.LINE_AA)

# ── freeze segment animation params ──────────────────────────────────────────
FREEZE_NFRAM = 180   # 6 s @ 30fps
# phase boundaries (frames within freeze)
#  Ph1: 黄线hold + 红闪触发    0..29   (1s)
#  Ph2: 箭头飞入              30..49   (0.67s)
#  Ph3: 红线推移→变绿         50..149  (3.33s)
FREEZE_NFRAM = 240   # 8000ms: Ph0=60(2s) + Ph1~Ph4=180
PH0_END = 60     # 2000ms 就近标注+闪动箭头
PH1_END = 90     # +1000ms 红爆闪
PH2_END = 110    #  +667ms 箭头飞入
PH3_END = 210    # +3333ms 推移变绿
PH4_END = 240    # +1000ms hold

RFLASH_PEAK = 65; RFLASH_RISE = 5; RFLASH_FALL = 8    # shifted to Ph1
GFLASH_PEAK = PH3_END-1; GFLASH_RISE = 10; GFLASH_FALL = 9
COL_START = GFLASH_PEAK - 8
COL_END   = GFLASH_PEAK

COLOR_LEAD = 0.60   # 推到60%时开始变色 (红→绿)

def red_flash(fi):
    if fi < RFLASH_PEAK - RFLASH_RISE or fi > RFLASH_PEAK + RFLASH_FALL: return 0.0
    if fi <= RFLASH_PEAK:
        return (fi - (RFLASH_PEAK - RFLASH_RISE)) / RFLASH_RISE
    return 1.0 - (fi - RFLASH_PEAK) / RFLASH_FALL

def green_flash(fi):
    peak = GFLASH_PEAK; rise = GFLASH_RISE; fall = GFLASH_FALL
    if fi < peak-rise or fi > peak+fall: return 0.0
    if fi <= peak: return (fi-(peak-rise))/rise
    return 1.0-(fi-peak)/fall

def col_t(fi):
    """0→1 as fi goes COL_START→COL_END, used in Ph3"""
    if fi < COL_START: return 0.0
    if fi >= COL_END:  return 1.0
    return ss((fi-COL_START)/(COL_END-COL_START))

def draw_freeze(img, fi):
    """fi = frame index within freeze (0..194)"""

    # always draw green reference lines
    draw_vline(img, X_GREEN_LEAD,  GREEN)
    draw_vline(img, X_GREEN_TRAIL, GREEN)

    # ── Ph0: 标注层 (CUE_DESIGN_LANGUAGE §Ph0 v3) ───────────────────────────
    # 三色语义: 黄=标注/引导, 红=越界错误, 绿=基准/正确
    # 标注位置: 右上方空白区 (问题在右侧髋, 放同侧上方)
    # 引线: 黄色细长带箭头 (draw_guide_arrow), 非 barb
    if fi < PH0_END:
        alpha = min(1.0, fi / 10.0)
        pulse = 0.5 + 0.5 * math.sin(fi * math.pi / 8.0)

        # 问题点: 髋外缘 (fr147 实测)
        HIP_TARGET = (int(OX_F), int(HIP_Y))

        # 标注框: 右上方空白区, 同侧
        TX, TY = 430, 280

        # 引线起点: 标注底部
        guide_src = (TX + 80, TY + 58)

        if alpha > 0.02:
            ov = img.copy()
            # 黄色文字 (标注层)
            cv2.putText(ov, "\u9ac0\u90e8\u6a2a\u6ed1", (TX, TY),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, YELLOW, 3, cv2.LINE_AA)
            cv2.putText(ov, "Hip Slide", (TX, TY + 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, YELLOW, 2, cv2.LINE_AA)
            # 黄色细长引导箭头 → 髋外缘
            draw_guide_arrow(ov, guide_src, HIP_TARGET, YELLOW, pulse)
            cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
        return

    if fi < PH1_END:
        # Ph1: 黄线 → 红闪 (fi=0 trigger)
        rfint = red_flash(fi)
        if rfint > 0.05:
            col = lerp_color(YELLOW, RED, rfint)
            # red glow
            ov = img.copy()
            for gw in [24, 14, 8]:
                cv2.line(ov, (int(OX_F), LT), (int(OX_F), LB), RED, gw, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.50*rfint, img, 1-0.50*rfint, 0, img)
        else:
            col = YELLOW if fi < 2 else RED
        draw_vline(img, OX_F, col)

    elif fi < PH2_END:
        # Ph2: 红线 hold + 箭头飞入
        draw_vline(img, OX_F, RED)
        t_ph = ss((fi-PH1_END)/(PH2_END-PH1_END-1))
        # 箭头从远处飞入，tip从 OX_F+STANDOFF+50 到 OX_F+STANDOFF
        tip_x = (OX_F + STANDOFF + 50) + t_ph * ((OX_F + STANDOFF) - (OX_F + STANDOFF + 50))
        tip = np.array([tip_x, HIP_Y])
        draw_barb(img, tip, PUSH_DIR, RED)

    elif fi < PH3_END:
        # Ph3: 红线被推向绿线 + 颜色渐变 + 绿闪
        t_push = ss((fi-PH2_END)/(PH3_END-PH2_END-1))
        curr_x = OX_F + t_push * (X_GREEN - OX_F)   # 从OX_F推到X_GREEN

        # color
        tc = col_t(fi)
        col = lerp_color(RED, GREEN, tc)

        # glow (proportional to tc)
        if tc > 0.05:
            ga = min(1.0, tc/0.6)
            ov = img.copy()
            cv2.line(ov, (int(curr_x), LT), (int(curr_x), LB), GREEN, 9, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.18*ga, img, 1-0.18*ga, 0, img)

        draw_vline(img, curr_x, col)

        # arrow: tip at curr_x, tail on lead side
        cur_so = STANDOFF*(1-t_push)+2
        tip = np.array([curr_x + cur_so, HIP_Y])
        draw_barb(img, tip, PUSH_DIR, col)

        # green flash in last ~10 frames of Ph3
        gfint = green_flash(fi)
        if gfint > 0:
            apply_flash(img, gfint, curr_x)

    else:
        # Ph4: 绿线 hold
        ov = img.copy()
        cv2.line(ov, (int(X_GREEN), LT), (int(X_GREEN), LB), GREEN, 9, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.18, img, 0.82, 0, img)
        draw_vline(img, X_GREEN, GREEN)

        # flash decay
        gfint = green_flash(fi)
        if gfint > 0:
            apply_flash(img, gfint, X_GREEN)

def draw_normal(img, fi):
    """
    Normal play frames: draw hip outer edge line colored by inside/outside.
    Stops if hip is occluded (low confidence or body rotated away).
    """
    # 绿基准线始终显示
    draw_vline(img, X_GREEN_LEAD,  GREEN)
    draw_vline(img, X_GREEN_TRAIL, GREEN)

    # 髋遮挡 → 不画髋线
    if not hip_visible(fi):
        return

    ox = outer_x(fi)
    if ox is None: return
    inside = ox < X_GREEN_LEAD
    col = YELLOW if inside else RED
    draw_vline(img, ox, col)

# ── build timeline ────────────────────────────────────────────────────────────
# 复用鸡翅膀结构: 慢放0.25x → ease-in → 定格6s → ease-out → 慢放继续
EASE_NFRAM  = 25
NORMAL_RATE = 0.25  # 0.25x = 每4帧输出1帧视频 → 每1源帧 → 4输出帧

timeline = []   # list of (src_fr, meta)  meta=True→draw_normal  meta=int→draw_freeze(fi)

# 1. 慢放 A → FR_FREEZE (不含freeze本身), 0.25x
for src_f in range(A, FR_FREEZE):
    for _ in range(4):   # 4 output frames per source frame
        timeline.append((src_f, True))

# 2. ease-in: 25帧内从0.25x渐慢到静止
# 直接重复 FR_FREEZE 帧越来越多次
for i in range(EASE_NFRAM):
    timeline.append((FR_FREEZE, True))

# 3. freeze: 180帧 animation
for fi in range(FREEZE_NFRAM):
    timeline.append((FR_FREEZE, fi))   # fi=int = freeze animation frame

# 4. ease-out: 25帧
for i in range(EASE_NFRAM):
    timeline.append((FR_FREEZE, True))

# 5. 慢放 FR_FREEZE+1 → TOTAL
for src_f in range(FR_FREEZE+1, TOTAL):
    for _ in range(4):
        timeline.append((src_f, True))

print(f"Timeline: {len(timeline)} fr = {len(timeline)/FPS_OUT:.1f}s")
freeze_start = next(i for i,(a,b) in enumerate(timeline) if isinstance(b,int) and b==0)
print(f"Freeze starts at output fr{freeze_start} = {freeze_start/FPS_OUT:.2f}s")

# ── load source frames ────────────────────────────────────────────────────────
print("Loading source frames...", flush=True)
cap = cv2.VideoCapture(str(VID))
VW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
VH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
src = []
for _ in range(TOTAL):
    ret, f = cap.read()
    src.append(f if ret else (src[-1] if src else np.zeros((VH,VW,3),np.uint8)))
cap.release()
print(f"  {TOTAL} fr {VW}x{VH}")

# ── render ────────────────────────────────────────────────────────────────────
print(f"Rendering {len(timeline)} frames...")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter(str(TMP_AVI), fourcc, FPS_OUT, (VW, VH))

for i, entry in enumerate(timeline):
    if i % 200 == 0: print(f"  {i}/{len(timeline)}", flush=True)
    sfr, meta = entry
    img = src[min(sfr, TOTAL-1)].copy()

    if isinstance(meta, bool):
        if meta:
            draw_normal(img, sfr)
    else:
        fi = meta   # freeze animation frame
        draw_freeze(img, fi)

    writer.write(img)

writer.release()
print("Render done")

# ── ffmpeg encode ─────────────────────────────────────────────────────────────
cmd = ['ffmpeg', '-y', '-i', str(TMP_AVI),
       '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
       '-pix_fmt', 'yuv420p', str(OUT_VID)]
subprocess.run(cmd, check=True, capture_output=True)
TMP_AVI.unlink(missing_ok=True)
print(f"=> {OUT_VID}  ({OUT_VID.stat().st_size//1024}KB)")

print(f"\nGeometry:")
print(f"  trail green x={int(X_GREEN_TRAIL)}  lead green x={int(X_GREEN_LEAD)}")
print(f"  freeze fr={FR_FREEZE}: outer_x={OX_F:.0f}  outside lead green by {OX_F-X_GREEN_LEAD:.0f}px")
print(f"  push: {OX_F:.0f} → {X_GREEN:.0f}  ({OX_F-X_GREEN:.0f}px)")
print(f"  yellow: fr{A}~fr139 | red trigger: fr140 (首次出界)")
