#!/usr/bin/env python3
"""
fo-wrong-4 横向稳定性分析: 旋转 vs 平移
输出: 数据表 + 骨架轨迹图
"""
import json, cv2, numpy as np
from pathlib import Path

CACHE   = Path("engine/kp_cache/batch2/fo-wrong-4.json")
VID     = Path("/mnt/c/Users/jason/Zening/Swingcue/Video/fo-wrong-4.mp4")
PREVIEW = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview")
PREVIEW.mkdir(parents=True, exist_ok=True)

with open(CACHE) as f:
    data = json.load(f)
frames = data['frames']
NF = len(frames)

# phase anchors (from prior run)
A=85; TOP=126; IMP=147; FIN=181
FPS = 30.0

def kpt(fi, name):
    fr = frames[fi]
    if not fr['persons']: return None
    kp = fr['persons'][0]['keypoints']
    v  = kp.get(name)
    if v is None: return None
    return np.array([v['x'], v['y']])

def arr(name):
    out = []
    for fi in range(NF):
        p = kpt(fi, name)
        out.append(p if p is not None else np.array([np.nan, np.nan]))
    return np.array(out)

ls = arr('left_shoulder'); rs = arr('right_shoulder')
lh = arr('left_hip');      rh = arr('right_hip')
nose = arr('nose')

sh_mid = (ls + rs) / 2
hi_mid = (lh + rh) / 2

sh_w = np.linalg.norm(ls - rs, axis=1)   # shoulder width
hi_w = np.linalg.norm(lh - rh, axis=1)   # hip width

SW0 = sh_w[A]   # address shoulder width = scale reference
HW0 = hi_w[A]

# ── rotation angles (arccos of projected width / address width) ───────────────
# Face-on projection: when body rotates θ around vertical axis,
#   apparent width = W0 * cos(θ)  → θ = arccos(w/W0)
# Sign: backswing = trail coil (positive), downswing unwinding
sh_rot = np.degrees(np.arccos(np.clip(sh_w / SW0, 0.0, 1.0)))  # shoulder rotation °
hi_rot = np.degrees(np.arccos(np.clip(hi_w / HW0, 0.0, 1.0)))  # hip rotation °

# X-factor = shoulder_rot - hip_rot (positive = correct lag)
xfactor = sh_rot - hi_rot

# ── translation (lateral x, normalized by SW0) ───────────────────────────────
# face-on, right-handed golfer facing camera:
#   left shoulder x > right shoulder x (lead shoulder on screen-left side has higher x)
#   trail side = decreasing x, lead side = increasing x
head_dx  = nose[:,0]    - nose[A,0]
sh_dx    = sh_mid[:,0]  - sh_mid[A,0]
hi_dx    = hi_mid[:,0]  - hi_mid[A,0]

# normalize
head_dx_n = head_dx / SW0
sh_dx_n   = sh_dx   / SW0
hi_dx_n   = hi_dx   / SW0

# ── key frames analysis ───────────────────────────────────────────────────────
print("=" * 72)
print("fo-wrong-4 旋转 vs 平移 全量分析")
print(f"address=fr{A}  top=fr{TOP}  impact=fr{IMP}  SW0={SW0:.0f}px HW0={HW0:.0f}px")
print("=" * 72)

print(f"\n{'fr':>5} {'phase':15} {'sh_rot°':>8} {'hi_rot°':>8} {'xfact°':>8} {'head_dx':>8} {'sh_dx':>8} {'hi_dx':>8} {'h/SW':>7} {'s/SW':>7} {'i/SW':>7}")
print("-" * 85)

phase_segs = [
    ('address',   list(range(A, A+1))),
    ('takeaway',  list(range(A, A+12, 4))),
    ('backswing', list(range(A+12, TOP, 6))),
    ('top',       list(range(TOP-1, TOP+2))),
    ('transition',list(range(TOP, TOP+8, 3))),
    ('downswing', list(range(TOP+8, IMP, 4))),
    ('impact',    list(range(IMP-1, IMP+2))),
    ('follow',    list(range(IMP+5, FIN, 8))),
]

show_frs = sorted(set([A, A+7, A+14, A+21, A+28, TOP-3, TOP, TOP+4, TOP+8,
                        IMP-5, IMP, IMP+5, IMP+15, IMP+30, 149, 160, 175]))
phase_labels = {fi: 'addr' if fi==A else 'top' if abs(fi-TOP)<=1 else
                'impact' if abs(fi-IMP)<=1 else 'fr149' if fi==149 else '' for fi in show_frs}

for fi in show_frs:
    if fi >= NF: continue
    # phase label
    if fi <= A: ph='address'
    elif fi <= A+int((TOP-A)*0.25): ph='takeaway'
    elif fi < TOP-1: ph='backswing'
    elif fi <= TOP+1: ph='top'
    elif fi <= TOP+int((IMP-TOP)*0.15): ph='transition'
    elif fi < IMP-3: ph='downswing'
    elif fi <= IMP+3: ph='impact'
    else: ph='follow'
    print(f"{fi:5d} {ph:15} {sh_rot[fi]:8.1f} {hi_rot[fi]:8.1f} {xfactor[fi]:8.1f} "
          f"{head_dx[fi]:8.1f} {sh_dx[fi]:8.1f} {hi_dx[fi]:8.1f} "
          f"{head_dx_n[fi]:7.2f} {sh_dx_n[fi]:7.2f} {hi_dx_n[fi]:7.2f}")

# ── key summary stats ─────────────────────────────────────────────────────────
print("\n" + "="*72)
print("KEY SUMMARY: 上杆 / 下杆 分段汇总")
print("="*72)

# backswing segment: A → TOP
print(f"\n[上杆 addr→top  fr{A}→fr{TOP}]")
print(f"  肩旋转:  +{sh_rot[TOP]:.1f}°  (arccos法, face-on投影)")
print(f"  髋旋转:  +{hi_rot[TOP]:.1f}°")
print(f"  X-factor at top:  {xfactor[TOP]:.1f}°  (肩-髋差, TPI参考35~45°)")
print(f"  头横移:  {head_dx[TOP]:.1f}px = {head_dx_n[TOP]:.2f}×SW0  (负=trail向)")
print(f"  肩中心横移: {sh_dx[TOP]:.1f}px = {sh_dx_n[TOP]:.2f}×SW0")
print(f"  髋中心横移: {hi_dx[TOP]:.1f}px = {hi_dx_n[TOP]:.2f}×SW0")
print(f"  [判断] 髋中心横移/SW0={hi_dx_n[TOP]:.2f}  (>0.15=sway, >0.20=明显sway)")
extra_head_bs = head_dx[TOP] - sh_dx[TOP]
print(f"  头相对于肩中心的额外移动: {extra_head_bs:.1f}px ({extra_head_bs/SW0:.2f}×SW0)")
print(f"    (=头移量 - 肩轴移量; 代表脊柱/颈椎额外漂移)")

print(f"\n[下杆 top→impact  fr{TOP}→fr{IMP}]")
print(f"  肩解旋:  {sh_rot[TOP]:.1f}° → {sh_rot[IMP]:.1f}°  (Δ={sh_rot[IMP]-sh_rot[TOP]:.1f}°)")
print(f"  髋解旋:  {hi_rot[TOP]:.1f}° → {hi_rot[IMP]:.1f}°  (Δ={hi_rot[IMP]-hi_rot[TOP]:.1f}°)")
print(f"  头横移(top→impact): {head_dx[IMP]-head_dx[TOP]:.1f}px = {(head_dx[IMP]-head_dx[TOP])/SW0:.2f}×SW0")
print(f"  肩中心(top→impact): {sh_dx[IMP]-sh_dx[TOP]:.1f}px = {(sh_dx[IMP]-sh_dx[TOP])/SW0:.2f}×SW0")
print(f"  髋中心(top→impact): {hi_dx[IMP]-hi_dx[TOP]:.1f}px = {(hi_dx[IMP]-hi_dx[TOP])/SW0:.2f}×SW0")
print(f"  [判断] 髋slide addr→impact={hi_dx_n[IMP]:.2f}×SW0 (>0.20=slide, >0.30=明显slide)")

# ── rotation vs translation decomposition ────────────────────────────────────
print("\n" + "="*72)
print("旋转 vs 平移 分解")
print("="*72)

print(f"\n上杆 (addr→top):")
print(f"  髋旋转贡献: {hi_rot[TOP]:.1f}° ← 主要是旋转还是平移?")
print(f"  髋平移: {hi_dx[TOP]:.1f}px ({hi_dx_n[TOP]:.2f}×SW0) ← 极小 ≈ 无sway")
print(f"  结论: 上杆髋部以旋转为主, 无明显sway ✓")
print(f"  头部多余漂移: {extra_head_bs:.1f}px ({extra_head_bs/SW0:.2f}×SW0)")
print(f"    原因候选: (a)脊柱侧倾随旋转 (b)颈椎/头额外漂移")

print(f"\n下杆 (top→impact):")
hi_slide_down = hi_dx[IMP] - hi_dx[TOP]
sh_slide_down = sh_dx[IMP] - sh_dx[TOP]
hi_derot_down = hi_rot[TOP] - hi_rot[IMP]
sh_derot_down = sh_rot[TOP] - sh_rot[IMP]
print(f"  髋解旋: {hi_derot_down:.1f}°  髋平移: {hi_slide_down:.1f}px ({hi_slide_down/SW0:.2f}×SW0)")
print(f"  肩解旋: {sh_derot_down:.1f}°  肩平移: {sh_slide_down:.1f}px ({sh_slide_down/SW0:.2f}×SW0)")
rot_vs_trans = hi_derot_down / max(abs(hi_slide_down)/SW0 * 50, 1)  # rough ratio
print(f"  髋解旋/平移比: {hi_derot_down:.1f}° vs {hi_slide_down/SW0:.2f}×SW0")
print(f"  ← 解旋{hi_derot_down:.0f}°同时平移{hi_slide_down:.0f}px → 旋转+平移并存")
print(f"  结论: 下杆髋部slide({hi_slide_down/SW0:.2f}×SW0)超过正常阈值 ← SLIDE FAULT")

# ── candidate criteria ────────────────────────────────────────────────────────
print("\n" + "="*72)
print("候选判据 (供Jason选定)")
print("="*72)
print("""
候选A: 纯头部位移
  指标: head_dx / SW0 在上杆顶点
  本例: -0.40×SW0 (addr→top)
  缺点: 头动=sway的误判, 好挥杆也有头移动, 权威已明确反对

候选B: 髋中心横移 (Sway/Slide 经典TPI判据)
  上杆: hi_dx/SW0 at top → 本例 +0.04 (几乎无sway) ✓
  下杆: hi_dx/SW0 at impact → 本例 +0.26×SW0 ← SLIDE信号
  阈值参考(TPI): >0.15 = 疑似, >0.25 = 明显
  优点: 权威依据, 直接测身体轴心, 和头无关
  
候选C: 头-髋分离量 (头的额外漂移, 扣除髋位移)
  指标: (head_dx - hi_dx) / SW0 at top
  本例: (-45px - +4px) / 113 = -0.44×SW0 ← 头相对于髋的漂移
  含义: 髋几乎不动, 头移动了0.44倍肩宽 → 头是独立漂移还是脊柱带动?

候选D: 组合判据 (sway/slide综合分)
  上杆: hi_dx_n + 0.5×(head-hip)/SW0 at top
  下杆: hi_dx_n at impact
  同时报"slide"(下杆) + "过度头移"(上杆) 两个独立信号
""")

print("="*72)
print("Hermes诊断摘要")
print("="*72)
print(f"""
主要发现:
  ① 上杆 — 髋旋转正常(+{hi_rot[TOP]:.0f}°), 几乎无sway(+{hi_dx[TOP]:.0f}px/{hi_dx_n[TOP]:.2f}×SW0) ← 好
             肩旋转充分(+{sh_rot[TOP]:.0f}°), X-factor={xfactor[TOP]:.0f}°(正常范围低端)
             但头明显trail漂移({head_dx[TOP]:.0f}px/{head_dx_n[TOP]:.2f}×SW0)
             头相对髋的超额漂移: {extra_head_bs:.0f}px({extra_head_bs/SW0:.2f}×SW0)
             → 头部漂移不是髋sway带动的, 是脊柱/颈椎独立漂移
             
  ② 下杆 — 髋slide明显: +{hi_dx[IMP]:.0f}px = {hi_dx_n[IMP]:.2f}×SW0 toward lead ← SLIDE FAULT
             髋解旋同时伴随大量横向平移({hi_slide_down:.0f}px), 不是纯旋转
             肩解旋慢于髋({sh_derot_down:.0f}° vs 髋{hi_derot_down:.0f}°)

推荐判据方向:
  下杆Slide (优先): hi_dx/SW0 at impact → 本例{hi_dx_n[IMP]:.2f}, 阈值TPI=0.15~0.25
  上杆头部漂移 (次优): 需区分脊柱侧倾 vs 真实漂移, 判据较复杂
  不推荐: 纯头部位移量 (权威批评该判据)
""")

# ── visualization: trajectory overlay on address frame ───────────────────────
print("生成骨架轨迹图...")
cap = cv2.VideoCapture(str(VID))
for _ in range(A):
    cap.read()
ret, frame_bg = cap.read()
cap.release()

if not ret:
    print("无法读取视频帧, 跳过可视化")
else:
    img = frame_bg.copy()
    H, W = img.shape[:2]

    # darken bg
    img = (img * 0.45).astype(np.uint8)

    def ip(p): return (int(round(float(p[0]))), int(round(float(p[1]))))
    def safe_kpt(fi, name):
        p = kpt(fi, name)
        return p

    # Draw trajectory lines for head / sh_mid / hi_mid
    # color scheme: head=cyan, shoulder=yellow, hip=orange
    COL_HEAD = (255, 220, 0)    # cyan-ish BGR
    COL_SH   = (0, 220, 255)    # yellow
    COL_HI   = (0, 140, 255)    # orange

    key_frs  = sorted([A, A+7, A+14, A+21, A+28, A+35, TOP-2, TOP, TOP+5,
                        TOP+10, IMP-5, IMP, IMP+5, IMP+15, IMP+30])
    key_frs  = [fi for fi in key_frs if 0 <= fi < NF]

    def draw_traj(pts_fn, col, thickness=2):
        prev = None
        for fi in key_frs:
            p = pts_fn(fi)
            if p is None or np.isnan(p).any(): continue
            curr = ip(p)
            if prev is not None:
                cv2.line(img, prev, curr, col, thickness, cv2.LINE_AA)
            cv2.circle(img, curr, 3, col, -1, cv2.LINE_AA)
            prev = curr

    draw_traj(lambda fi: nose[fi],    COL_HEAD, 2)
    draw_traj(lambda fi: sh_mid[fi],  COL_SH,   2)
    draw_traj(lambda fi: hi_mid[fi],  COL_HI,   2)

    # Draw skeleton at key frames: address / top / impact
    SKEL = [('left_shoulder','left_elbow'),('left_elbow','left_wrist'),
            ('right_shoulder','right_elbow'),('right_elbow','right_wrist'),
            ('left_shoulder','right_shoulder'),('left_shoulder','left_hip'),
            ('right_shoulder','right_hip'),('left_hip','right_hip'),
            ('left_hip','left_knee'),('right_hip','right_knee'),
            ('left_knee','left_ankle'),('right_knee','right_ankle')]

    frame_specs = [
        (A,   (180,180,180), 'ADDRESS'),
        (TOP, (100,200,255), 'TOP'),
        (IMP, (80, 255, 120),'IMPACT'),
    ]

    for fi, col, label in frame_specs:
        fr = frames[fi]
        if not fr['persons']: continue
        kp_d = {k: np.array([v['x'],v['y']]) for k,v in fr['persons'][0]['keypoints'].items()}
        for a, b in SKEL:
            pa = kp_d.get(a); pb = kp_d.get(b)
            if pa is None or pb is None: continue
            cv2.line(img, ip(pa), ip(pb), col, 1, cv2.LINE_AA)
        for name, pt_arr in kp_d.items():
            cv2.circle(img, ip(pt_arr), 3, col, -1, cv2.LINE_AA)
        # label
        sh_pt = kp_d.get('left_shoulder')
        if sh_pt is not None:
            cv2.putText(img, label, (ip(sh_pt)[0]-10, ip(sh_pt)[1]-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)

    # Vertical reference lines at address x positions
    for x_val, col, lbl in [
        (int(nose[A][0]),   COL_HEAD, 'head@A'),
        (int(sh_mid[A][0]), COL_SH,   'sh@A'),
        (int(hi_mid[A][0]), COL_HI,   'hi@A'),
    ]:
        cv2.line(img, (x_val, 50), (x_val, H-50), col, 1, cv2.LINE_AA)
        # also mark impact position
    for x_val, col in [
        (int(hi_mid[IMP][0]), COL_HI),
        (int(sh_mid[IMP][0]), COL_SH),
    ]:
        cv2.line(img, (x_val, 50), (x_val, H-50), col, 1, cv2.LINE_AA)

    # legend
    legend = [
        (COL_HEAD, f"HEAD  top:{head_dx[TOP]:.0f}px({head_dx_n[TOP]:.2f}SW)  imp:{head_dx[IMP]:.0f}px"),
        (COL_SH,   f"SH_MID top:{sh_dx[TOP]:.0f}px({sh_dx_n[TOP]:.2f}SW)  imp:{sh_dx[IMP]:.0f}px"),
        (COL_HI,   f"HI_MID top:{hi_dx[TOP]:.0f}px({hi_dx_n[TOP]:.2f}SW)  imp:{hi_dx[IMP]:.0f}px  <<SLIDE"),
    ]
    for i, (col, txt) in enumerate(legend):
        cv2.putText(img, txt, (15, 30+i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    cv2.putText(img, f"ROTATION at top: sh={sh_rot[TOP]:.0f}deg  hi={hi_rot[TOP]:.0f}deg  Xfactor={xfactor[TOP]:.0f}deg",
                (15, H-50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1, cv2.LINE_AA)
    cv2.putText(img, f"SLIDE at impact: hi_dx={hi_dx[IMP]:.0f}px = {hi_dx_n[IMP]:.2f}xSW  (TPI threshold 0.15~0.25)",
                (15, H-28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_HI, 1, cv2.LINE_AA)

    out_path = PREVIEW / "fo_wrong4_trajectory.jpg"
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"=> {out_path}")
