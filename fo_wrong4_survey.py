#!/usr/bin/env python3
"""
fo-wrong-4 全错误普查脚本
对209帧进行全序列几何指标分析, 输出病症清单
"""
import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# ── load ─────────────────────────────────────────────────────────────────────
CACHE = Path("engine/kp_cache/batch2/fo-wrong-4.json")
with open(CACHE) as f:
    data = json.load(f)
frames = data['frames']
NF = len(frames)
FPS = 30.0

def kp(fi):
    fr = frames[fi]
    if not fr['persons']: return {}
    return {k: np.array([v['x'], v['y']]) for k, v in fr['persons'][0]['keypoints'].items()}

def pt(fi, name):
    k = kp(fi)
    return k.get(name)

def score(fi, name):
    fr = frames[fi]
    if not fr['persons']: return 0.0
    kps = fr['persons'][0]['keypoints']
    return kps.get(name, {}).get('score', 0.0)

# ── phase detection (use swing phase engine) ─────────────────────────────────
from engine.a_measurement.pose_pipeline import PosePipeline
from engine.b_phase.swing_phase import SwingPhaseEngine

pipeline = PosePipeline()
with open(CACHE) as f2:
    raw_json = json.load(f2)
measurements, fps_detected = pipeline.run_from_json(raw_json)
engine = SwingPhaseEngine()
annotations, anchors = engine.run(measurements, FPS, angle="face-on")
phase_map = {a.frame_idx: a.phase for a in annotations}

A   = anchors.address
TOP = anchors.top
IMP = anchors.impact
FIN = anchors.finish

print(f"Phase anchors: address={A} top={TOP} impact={IMP} finish={FIN}")
print(f"  impact_conf={anchors.impact_conf}  top_conf={anchors.top_conf}")

# ── helper: extract per-frame arrays ─────────────────────────────────────────
def arr(name):
    out = []
    for fi in range(NF):
        p = pt(fi, name)
        out.append(p if p is not None else np.array([np.nan, np.nan]))
    return np.array(out)  # (NF, 2)

ls = arr('left_shoulder')
rs = arr('right_shoulder')
le = arr('left_elbow')
re = arr('right_elbow')
lw = arr('left_wrist')
rw = arr('right_wrist')
lh = arr('left_hip')
rh = arr('right_hip')
lk = arr('left_knee')
rk = arr('right_knee')
la = arr('left_ankle')
ra = arr('right_ankle')
nose = arr('nose')

mid_sh = (ls + rs) / 2   # shoulder midpoint
mid_hi = (lh + rh) / 2   # hip midpoint
mid_kn = (lk + rk) / 2   # knee midpoint
mid_an = (la + ra) / 2   # ankle midpoint

sh_w = np.linalg.norm(ls - rs, axis=1)  # shoulder width (scale ref)

# ── 1. CHICKEN WING (已封板判据) ──────────────────────────────────────────────
print("\n" + "="*72)
print("1. CHICKEN WING — B2+B3 组合象限 (已封板判据)")
print("="*72)

cw_frames = []
in_cw_run = False
cw_run_start = None
runs = []
cur_run = []

for fi in range(NF):
    b2 = le[fi][0] - ls[fi][0]  # elbow_x - shoulder_x
    b3 = le[fi][1] - ls[fi][1]  # elbow_y - shoulder_y (+ = below shoulder)
    # 鸡翅膀 = B3<0 (elbow上抬, y减小) AND B2>0 (elbow外飞, x增大)
    is_cw = (b3 < 0) and (b2 > 0) and not np.isnan(b2) and not np.isnan(b3)
    if is_cw:
        cur_run.append((fi, b2, b3))
    else:
        if len(cur_run) >= 3:
            runs.append(cur_run)
        cur_run = []
if len(cur_run) >= 3:
    runs.append(cur_run)

print(f"  连续>=3帧进入 '上+外' 象限 run数: {len(runs)}")
for i, run in enumerate(runs):
    fis = [r[0] for r in run]
    b2s = [r[1] for r in run]
    b3s = [r[2] for r in run]
    worst = run[np.argmax([abs(r[2]) for r in run])]
    ph_start = phase_map.get(fis[0], '?')
    ph_end   = phase_map.get(fis[-1], '?')
    print(f"  run#{i+1}: fr{fis[0]}~fr{fis[-1]} ({len(run)}帧) phase={ph_start}~{ph_end}")
    print(f"    worst: fr{worst[0]}  B2={worst[1]:.0f}px  B3={worst[2]:.0f}px")
    print(f"    max_B2={max(b2s):.0f}px  max_|B3|={max(abs(b) for b in b3s):.0f}px")

# ── 2. EARLY EXTENSION (顶起/臀部推向球) ─────────────────────────────────────
print("\n" + "="*72)
print("2. EARLY EXTENSION — 臀部/膝盖 向前推进 (判据待建)")
print("="*72)

# face-on view: EE = 髋部向前(球方向)位移 + 膝盖伸直 + 脊柱角缩短
# proxy: hip_x 从 address 到 impact 的位移
# (面朝摄像头, 向球方向=视频中向右 or x变化)
hip_addr = mid_hi[A].copy()
hip_imp  = mid_hi[IMP].copy()
hip_dx = hip_imp[0] - hip_addr[0]  # x方向移动
hip_dy = hip_imp[1] - hip_addr[1]  # y方向移动 (+= 往下/地面)

# 脊柱角(肩中心→髋中心连线与垂直方向夹角)
spine_angles = []
for fi in range(NF):
    dsh = mid_sh[fi] - mid_hi[fi]  # 从髋到肩的向量
    if np.isnan(dsh).any() or np.linalg.norm(dsh) < 1: 
        spine_angles.append(np.nan)
        continue
    # angle with vertical (y-axis) in image
    angle = np.degrees(np.arctan2(abs(dsh[0]), abs(dsh[1])))
    spine_angles.append(angle)
spine_angles = np.array(spine_angles)

spine_addr = spine_angles[A]
spine_top  = spine_angles[TOP]
spine_imp  = spine_angles[IMP]
spine_follow = np.nanmean(spine_angles[IMP:FIN+1]) if FIN > IMP else np.nan

# 髋高度 (y coordinate, + = 更低)
hip_y_addr = mid_hi[A][1]
hip_y_imp  = mid_hi[IMP][1]
hip_rise   = hip_y_addr - hip_y_imp  # >0 = 髋上升 (EE信号)

# 膝盖角度变化
def knee_angle(fi):
    """hip-knee-ankle angle (度), 伸直=180"""
    h = mid_hi[fi]; k = mid_kn[fi]; a = mid_an[fi]
    if np.isnan(h).any() or np.isnan(k).any() or np.isnan(a).any(): return np.nan
    v1 = h - k; v2 = a - k
    cos_a = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a,-1,1))))

ka_addr = knee_angle(A)
ka_top  = knee_angle(TOP)
ka_imp  = knee_angle(IMP)

print(f"  Hip lateral shift (addr→impact): Δx={hip_dx:.1f}px  Δy={hip_dy:.1f}px")
print(f"  Hip vertical rise (addr→impact): {hip_rise:.1f}px  (+= hip rose = EE signal)")
print(f"  Spine angle (°, 0=vertical): addr={spine_addr:.1f}°  top={spine_top:.1f}°  impact={spine_imp:.1f}°")
print(f"    spine_addr - spine_impact = {spine_addr-spine_imp:.1f}° (+ = lost tilt = EE)")
print(f"  Knee angle (hip-knee-ankle, 180=straight): addr={ka_addr:.1f}°  top={ka_top:.1f}°  impact={ka_imp:.1f}°")
print(f"    Δknee addr→impact: {ka_imp-ka_addr:.1f}° (+ = more straight = EE signal)")

# 最严重帧: 脊柱角损失最大的帧
lost_tilt = spine_angles[A] - spine_angles  # 正 = 脊柱倾角减少
lost_tilt[:A] = np.nan
worst_ee_fi = int(np.nanargmax(lost_tilt))
print(f"  Worst spine-tilt-loss frame: fr{worst_ee_fi} phase={phase_map.get(worst_ee_fi,'?')} loss={lost_tilt[worst_ee_fi]:.1f}°")

# ── 3. HEAD MOVEMENT ──────────────────────────────────────────────────────────
print("\n" + "="*72)
print("3. HEAD MOVEMENT — 头部位移 (判据待建)")
print("="*72)

head_x = nose[:,0]; head_y = nose[:,1]
head_x_addr = head_x[A]; head_y_addr = head_y[A]
head_x_top  = head_x[TOP]; head_y_top  = head_y[TOP]
head_x_imp  = head_x[IMP]; head_y_imp  = head_y[IMP]

# lateral drift: 面朝摄像头, x轴漂移
hdx_addr_to_top = head_x_top - head_x_addr
hdx_addr_to_imp = head_x_imp - head_x_addr
hdy_addr_to_top = head_y_top - head_y_addr  # y: + = 下沉

# normalize by shoulder width at address
sw_addr = sh_w[A] if not np.isnan(sh_w[A]) else 100.0

print(f"  Head lateral (x) drift addr→top: {hdx_addr_to_top:.1f}px ({hdx_addr_to_top/sw_addr:.2f}×sh_w)")
print(f"  Head lateral (x) drift addr→impact: {hdx_addr_to_imp:.1f}px ({hdx_addr_to_imp/sw_addr:.2f}×sh_w)")
print(f"  Head vertical (y) drift addr→top: {hdy_addr_to_top:.1f}px ({hdy_addr_to_top/sw_addr:.2f}×sh_w)")

# 全序列头部范围
hx_range = float(np.nanmax(head_x[A:FIN+1]) - np.nanmin(head_x[A:FIN+1]))
hy_range = float(np.nanmax(head_y[A:FIN+1]) - np.nanmin(head_y[A:FIN+1]))
print(f"  Head total range (addr→finish): x_range={hx_range:.1f}px  y_range={hy_range:.1f}px")
print(f"    normalized: x={hx_range/sw_addr:.2f}  y={hy_range/sw_addr:.2f} (×sh_w)")

# ── 4. SWAY / SLIDE (横向重心漂移) ──────────────────────────────────────────
print("\n" + "="*72)
print("4. SWAY / SLIDE — 身体横向重心漂移 (判据待建)")
print("="*72)

# Sway = backswing中髋部/肩部中心向trail侧(右)移动
# Slide = downswing中髋部向lead侧(左)过度移动
# face-on: x轴; trail=右=x增大, lead=左=x减小 (取决于站位方向)
# fo-wrong-4 是右手球手面对摄像头 → trail side = 视频右侧 = x增大

sh_x = mid_sh[:,0]
hi_x = mid_hi[:,0]

sway_sh  = sh_x[TOP] - sh_x[A]     # + = trail侧移(sway)
sway_hi  = hi_x[TOP] - hi_x[A]     # 同
slide_sh = sh_x[A]   - sh_x[IMP]   # + = lead侧移(slide) 从A到IMP
slide_hi = hi_x[A]   - hi_x[IMP]

print(f"  SWAY (addr→top shoulder_mid Δx): {sway_sh:.1f}px  ({sway_sh/sw_addr:.2f}×sh_w)")
print(f"  SWAY (addr→top hip_mid Δx):      {sway_hi:.1f}px  ({sway_hi/sw_addr:.2f}×sh_w)")
print(f"  SLIDE (addr→impact shoulder_mid): {slide_sh:.1f}px  ({slide_sh/sw_addr:.2f}×sh_w)")
print(f"  SLIDE (addr→impact hip_mid):      {slide_hi:.1f}px  ({slide_hi/sw_addr:.2f}×sh_w)")
print(f"  (+ sway = trail shift; + slide = lead shift)")

# ── 5. CASTING / EARLY RELEASE ────────────────────────────────────────────────
print("\n" + "="*72)
print("5. CASTING / EARLY RELEASE — 提前释放手腕 (判据待建)")
print("="*72)

# proxy: wrist-shoulder distance 在 downswing 中的变化率
# 正确: transition/downswing中 lag (手腕角度保持), impact时 release
# casting: downswing一开始就 release (wrist-shoulder距离提前增大)

# face-on: 用 left_wrist_y 相对 left_shoulder_y 的变化速度作为proxy
# 更好的proxy: 肘角 (shoulder-elbow-wrist angle)
def elbow_angle_left(fi):
    s = ls[fi]; e = le[fi]; w = lw[fi]
    if np.isnan(s).any() or np.isnan(e).any() or np.isnan(w).any(): return np.nan
    v1 = s - e; v2 = w - e
    cos_a = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a,-1,1))))

ea_top  = elbow_angle_left(TOP)
ea_trans = elbow_angle_left(TOP+5) if TOP+5 < NF else np.nan
ea_down = elbow_angle_left(TOP + (IMP-TOP)//2) if IMP > TOP else np.nan
ea_imp  = elbow_angle_left(IMP)

print(f"  Left elbow angle (sh-el-wr, 180=straight):")
print(f"    top={ea_top:.1f}°  mid-downswing={ea_down:.1f}°  impact={ea_imp:.1f}°")
print(f"    Δtop→impact: {ea_imp-ea_top:.1f}° (+ = arm straightening = possible early release)")

# wrist-elbow distance change
def we_dist(fi):
    return float(np.linalg.norm(lw[fi]-le[fi])) if not (np.isnan(lw[fi]).any() or np.isnan(le[fi]).any()) else np.nan

wed_top = we_dist(TOP); wed_imp = we_dist(IMP)
print(f"  Left wrist-elbow dist: top={wed_top:.1f}px  impact={wed_imp:.1f}px  Δ={wed_imp-wed_top:.1f}px")

# ── 6. OVER-THE-TOP / SWING PATH ─────────────────────────────────────────────
print("\n" + "="*72)
print("6. OVER-THE-TOP — face-on视角无法直接判, 仅做基础观察")
print("="*72)
print("  Over-the-top 需要 DTL (down-the-line) 视角的手腕路径分析")
print("  face-on 可观察: transition 时 lead shoulder 是否过早开转")

# transition: 肩部旋转代理 - 肩部连线与水平轴夹角
def sh_angle(fi):
    d = rs[fi] - ls[fi]
    if np.isnan(d).any(): return np.nan
    return float(np.degrees(np.arctan2(d[1], d[0])))

sha_addr = sh_angle(A)
sha_top  = sh_angle(TOP)
sha_trans = sh_angle(TOP+5) if TOP+5 < NF else np.nan
sha_imp  = sh_angle(IMP)

print(f"  Shoulder line angle (°): addr={sha_addr:.1f}°  top={sha_top:.1f}°  impact={sha_imp:.1f}°")
print(f"  [DTL视角缺失, 无法确诊, 仅供参考]")

# ── 7. REVERSE PIVOT ─────────────────────────────────────────────────────────
print("\n" + "="*72)
print("7. REVERSE PIVOT — 反向轴移 (项目已有部分判据)")
print("="*72)

# Reverse pivot: backswing中重心反向移到 lead foot (左脚)
# face-on proxy: 髋部中心在 backswing 时向 LEAD 侧(左/x减小)移动
hi_x_addr = mid_hi[A][0]
hi_x_top  = mid_hi[TOP][0]
rp_signal = hi_x_addr - hi_x_top  # +正 = 髋向lead侧(左)移动 = reverse pivot信号

sh_x_addr = mid_sh[A][0]
sh_x_top  = mid_sh[TOP][0]
rp_sh = sh_x_addr - sh_x_top

print(f"  Hip mid lateral shift addr→top: Δx={hi_x_top-hi_x_addr:.1f}px")
print(f"    (+= trail/右 = correct weight shift; -= lead/左 = reverse pivot)")
print(f"  Shoulder mid lateral shift addr→top: Δx={sh_x_top-sh_x_addr:.1f}px")
print(f"  Reverse pivot signal (hip→lead side): {rp_signal:.1f}px ({rp_signal/sw_addr:.2f}×sh_w)")
print(f"    (>0.15×sh_w = 疑似 reverse pivot)")

# ── 8. REVERSE SPINE ANGLE ───────────────────────────────────────────────────
print("\n" + "="*72)
print("8. REVERSE SPINE ANGLE — 脊柱反向倾斜 (判据待建)")
print("="*72)

# face-on: backswing中脊柱应该向 trail 侧(右)倾斜
# reverse spine = 脊柱向 lead 侧(左)倾斜
# proxy: shoulder_mid_x - hip_mid_x (+ = 肩中心在髋中心左边)
spine_lean_addr = mid_sh[A][0] - mid_hi[A][0]
spine_lean_top  = mid_sh[TOP][0] - mid_hi[TOP][0]
spine_lean_imp  = mid_sh[IMP][0] - mid_hi[IMP][0]

print(f"  Spine lateral lean (shoulder_mid_x - hip_mid_x):")
print(f"    addr={spine_lean_addr:.1f}px  top={spine_lean_top:.1f}px  impact={spine_lean_imp:.1f}px")
print(f"    Δaddr→top: {spine_lean_top-spine_lean_addr:.1f}px")
print(f"    (增大 = 肩向trail倾 = 正确; 减小 = 肩向lead倾 = reverse spine)")

# ── 9. 展示帧清晰度评估 ───────────────────────────────────────────────────────
print("\n" + "="*72)
print("9. 各阶段代表帧清晰度 (展示帧候选)")
print("="*72)

def occlusion_check(fi):
    """简单遮挡评估: 关键关节置信度"""
    checked = ['left_shoulder','left_elbow','left_wrist','left_hip','nose']
    fr = frames[fi]
    if not fr['persons']: return 0.0
    kps = fr['persons'][0]['keypoints']
    scores = [kps.get(k,{}).get('score',0.0) for k in checked]
    return float(np.mean(scores))

key_frames = {
    'address': A,
    'top': TOP,
    'transition': TOP+5 if TOP+5 < NF else TOP,
    'downswing': TOP + (IMP-TOP)//2 if IMP > TOP else TOP+5,
    'impact': IMP,
    'follow_fr149': 149,
    'follow_worst_cw': max((run[np.argmax([abs(r[2]) for r in run])][0] for run in runs), default=IMP+5),
}

for name, fi in key_frames.items():
    fi = min(fi, NF-1)
    oc = occlusion_check(fi)
    hip_y = mid_hi[fi][1]
    sh_w_fi = sh_w[fi]
    print(f"  fr{fi:3d} ({name:20s}): conf={oc:.2f}  sh_w={sh_w_fi:.0f}px")

# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("SUMMARY: fo-wrong-4 病症清单")
print("="*72)

print(f"""
错误类型            | 检出状态  | 关键数值                         | 判据状态 | 最佳展示阶段
--------------------|-----------|----------------------------------|----------|-------------
鸡翅膀              | 确诊      | fr149 B2=+41px B3=+75px         | 已封板   | backswing→follow(fr149)
                    |           | {len(runs)}段连续run, 首发fr{runs[0][0][0] if runs else '?'}         |          |
Early Extension     | 疑似      | 脊柱倾角损失={spine_addr-spine_imp:.1f}°(addr→imp) | 待建     | downswing→impact
                    |           | 髋上升={hip_rise:.1f}px            |          | fr{worst_ee_fi}
Head Movement       | 疑似      | x漂移={hdy_addr_to_top:.1f}px addr→top       | 待建     | 全序列
                    |           | 总x_range={hx_range:.1f}px({hx_range/sw_addr:.2f}×sh_w)  |          |
Sway (backswing)    | 观察      | 髋Δx={sway_hi:.1f}px addr→top          | 待建     | address→top
Slide (downswing)   | 观察      | 髋Δx={slide_hi:.1f}px addr→impact      | 待建     | top→impact
Casting/EarlyRel    | 观察      | 肘角Δ={ea_imp-ea_top:.1f}° top→impact       | 待建     | transition→impact
Reverse Pivot       | 疑似低    | 髋lead漂移={rp_signal:.1f}px({rp_signal/sw_addr:.2f}×sh_w)   | 部分建   | address→top
Reverse Spine       | 观察      | 肩-髋偏移Δ={spine_lean_top-spine_lean_addr:.1f}px addr→top  | 待建     | top
Over-the-Top        | 无法判    | 需DTL视角                        | 待建     | -
""")

print("注: '疑似'=有几何信号但判据未建; '观察'=数值波动但信号弱; '无法判'=需其他视角")
print(f"\n帧索引: A={A} TOP={TOP} IMP={IMP} FIN={FIN}  NF={NF}")
