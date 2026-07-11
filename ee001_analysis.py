#!/usr/bin/env python3
"""
fo-wrong-4 Early Extension 代理信号分析 (face-on)
S1: 脊柱倾角变化  address→impact  (正面用 nose→hip 中心连线)
S2: 头部 y 上升量 / SW0           (downswing top→impact)
S3: 髋部中心 y 上升量 / SW0
S4: trail 脚跟(右脚踝) y 变化
"""
import json, math, numpy as np
from pathlib import Path

CACHE = Path("/home/jason/projects/swingcue-postest/engine/kp_cache/batch2/fo-wrong-4.json")
with open(CACHE) as f:
    raw = json.load(f)

KP_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]
IDX = {n: i for i, n in enumerate(KP_NAMES)}

def kpt(fi, name):
    persons = raw["frames"][fi]["persons"]
    p = persons[0] if persons else None
    if p is None:
        return (0.0, 0.0, 0.0)
    kp = p["keypoints"]
    k  = kp[name]
    return k['x'], k['y'], k['score']

# ── 关键帧 (from PROJECT_STATE / KP cache) ───────────────────────────────────
FR_ADDR   = 85    # address
FR_TOP    = 50    # top of backswing (approx, per rot_vs_trans analysis)
FR_IMPACT = 147   # impact

# address 帧基准
la_addr = kpt(FR_ADDR, 'left_ankle')
ra_addr = kpt(FR_ADDR, 'right_ankle')
SW0 = abs(la_addr[0] - ra_addr[0])   # 113px

nose_addr   = kpt(FR_ADDR, 'nose')
lh_addr     = kpt(FR_ADDR, 'left_hip')
rh_addr     = kpt(FR_ADDR, 'right_hip')
hip_c_addr  = ((lh_addr[0]+rh_addr[0])/2, (lh_addr[1]+rh_addr[1])/2)

nose_top    = kpt(FR_TOP, 'nose')
lh_top      = kpt(FR_TOP, 'left_hip')
rh_top      = kpt(FR_TOP, 'right_hip')
hip_c_top   = ((lh_top[0]+rh_top[0])/2, (lh_top[1]+rh_top[1])/2)

nose_imp    = kpt(FR_IMPACT, 'nose')
lh_imp      = kpt(FR_IMPACT, 'left_hip')
rh_imp      = kpt(FR_IMPACT, 'right_hip')
hip_c_imp   = ((lh_imp[0]+rh_imp[0])/2, (lh_imp[1]+rh_imp[1])/2)

ra_imp      = kpt(FR_IMPACT, 'right_ankle')   # trail foot heel proxy

# ── S1: 脊柱倾角 (face-on: nose → hip_center 连线与竖直方向夹角) ─────────────
# 注意: 面对摄像机时, 前倾 = nose x 略偏向 trail 侧 (或 lead 侧取决于朝向)
# 用 dy/dx: angle = atan2(|dx|, dy)  -- 偏离竖直的角度
def spine_angle(nose_kp, hip_c):
    dx = nose_kp[0] - hip_c[0]
    dy = hip_c[1]   - nose_kp[1]   # 正 = nose 在上, hip 在下 (正常)
    return math.degrees(math.atan2(abs(dx), max(dy, 1)))

ang_addr = spine_angle(nose_addr, hip_c_addr)
ang_top  = spine_angle(nose_top,  hip_c_top)
ang_imp  = spine_angle(nose_imp,  hip_c_imp)

print("=" * 60)
print("S1 脊柱倾角 (nose→hip 连线偏竖直角)")
print(f"  address  fr{FR_ADDR}:  {ang_addr:.1f}°")
print(f"  top      fr{FR_TOP}:   {ang_top:.1f}°")
print(f"  impact   fr{FR_IMPACT}:  {ang_imp:.1f}°")
print(f"  addr→impact: {ang_imp - ang_addr:+.1f}°  (正=偏更垂直/站起来, 负=更前倾)")
print()
print("  [口径核对] 权威参考: address 30°→impact 19°, 丢失≈11°")
print("  [注意] face-on nose→hip 不是真实脊柱矢状前倾, 是 2D 代理")
print("  原之前报告的 3.6°→13.5°: 用的是左肩-左髋连线(更靠近侧面分量)")
print("  本次重算结论方向一致: 变小=站起来 ✓" if ang_imp < ang_addr else
      "  ⚠️  角度增大: 需核查坐标方向")
print()

# ── S2: 头部 y 上升 (y 轴: 向下为正, 所以 y 减小 = 头部上升) ────────────────
head_y_top    = nose_top[1]
head_y_imp    = nose_imp[1]
head_rise_pix = head_y_top - head_y_imp   # 正 = 上升 (y减小)
head_rise_sw0 = head_rise_pix / SW0
print("S2 头部 y 上升 (top→impact)")
print(f"  top fr{FR_TOP}: nose_y={head_y_top:.0f}  impact fr{FR_IMPACT}: nose_y={head_y_imp:.0f}")
print(f"  上升: {head_rise_pix:.0f}px = {head_rise_sw0:+.3f}×SW0")
print(f"  (正 = 头部抬升, 负 = 下沉)")
print()

# ── S3: 髋部中心 y 上升 (top→impact) ────────────────────────────────────────
hip_y_top = hip_c_top[1]
hip_y_imp = hip_c_imp[1]
hip_rise_pix = hip_y_top - hip_y_imp   # 正 = 上升
hip_rise_sw0 = hip_rise_pix / SW0
print("S3 髋部中心 y 上升 (top→impact)")
print(f"  top: hip_c_y={hip_y_top:.0f}  impact: hip_c_y={hip_y_imp:.0f}")
print(f"  上升: {hip_rise_pix:.0f}px = {hip_rise_sw0:+.3f}×SW0")
print()

# ── S4: trail 脚跟(右踝) y 变化 (addr→impact) ───────────────────────────────
ra_y_addr = ra_addr[1]
ra_y_imp  = ra_imp[1]
trail_heel_rise = ra_y_addr - ra_y_imp   # 正 = 右踝抬升 (脚跟离地)
trail_heel_sw0  = trail_heel_rise / SW0
print("S4 trail 脚跟 y 变化 (right_ankle, addr→impact)")
print(f"  addr: ra_y={ra_y_addr:.0f}  impact: ra_y={ra_y_imp:.0f}")
print(f"  变化: {trail_heel_rise:.0f}px = {trail_heel_sw0:+.3f}×SW0")
print(f"  ({'脚跟抬起' if trail_heel_rise > 10 else '脚跟贴地/无明显离地'})")
print()

# ── 逐帧 downswing 扫描 ───────────────────────────────────────────────────────
print("=" * 60)
print("Downswing 逐帧 (top→impact): S1/S2/S3")
print(f"{'fr':>4} | {'脊柱角°':>7} | {'头y变':>8} | {'髋c_y变':>8} | {'节点'}")
print("-" * 52)
DS_START, DS_END = 50, 148

# anchor: top frame
ref_nose_y  = nose_top[1]
ref_hip_y   = hip_c_top[1]
ref_spine   = ang_top

NF = len(raw["frames"])
for fi in range(DS_START, min(DS_END+1, NF), 3):
    nose_fi = kpt(fi, 'nose')
    lh_fi_t = kpt(fi, 'left_hip')
    rh_fi_t = kpt(fi, 'right_hip')
    lh_fi = (lh_fi_t[0], lh_fi_t[1])
    rh_fi = (rh_fi_t[0], rh_fi_t[1])
    hc = ((lh_fi[0]+rh_fi[0])/2, (lh_fi[1]+rh_fi[1])/2)
    sp = spine_angle(nose_fi, hc)
    dNy = (ref_nose_y - nose_fi[1]) / SW0   # +正=头抬升
    dHy = (ref_hip_y  - hc[1])      / SW0   # +正=髋抬升
    note = ""
    if fi == FR_TOP:    note = "← top"
    if fi == FR_IMPACT: note = "← impact"
    if fi == 100:       note = "← mid-ds"
    print(f"{fi:>4} | {sp:>7.1f} | {dNy:>+8.3f} | {dHy:>+8.3f} | {note}")

print()
print("=" * 60)
print("综合判断")
print(f"  SW0 = {SW0:.0f}px")
print(f"  S1: 脊柱角 {ang_addr:.1f}°→{ang_imp:.1f}° = {ang_imp-ang_addr:+.1f}°")
print(f"  S2: 头部抬升 {head_rise_sw0:+.3f}×SW0 ({head_rise_pix:.0f}px)")
print(f"  S3: 髋中心抬升 {hip_rise_sw0:+.3f}×SW0 ({hip_rise_pix:.0f}px)")
print(f"  S4: trail脚跟 {trail_heel_sw0:+.3f}×SW0")
print()

# 判断逻辑
s1_pos = (ang_imp - ang_addr)   # 正 = 站起来 (脊柱变垂直)
s2_pos = head_rise_sw0          # 正 = 头抬升
s3_pos = hip_rise_sw0           # 正 = 髋抬升
hip_rot = 17                    # 度 (已知, from rot_vs_trans analysis)

flags = []
if s1_pos < -5.0:               # 脊柱角减小 = 站起来 (face-on proxy 变小)
    flags.append(f"S1✓ 脊柱倾角丢失 {s1_pos:.1f}°")
if s2_pos > 0.10:
    flags.append(f"S2✓ 头部抬升 {s2_pos:.3f}×SW0")
if s3_pos > 0.05:
    flags.append(f"S3✓ 髋部上升 {s3_pos:.3f}×SW0")
if trail_heel_sw0 > 0.05:
    flags.append(f"S4✓ trail脚跟离地 {trail_heel_sw0:.3f}×SW0")
if hip_rot < 30:
    flags.append(f"髋旋转不足 {hip_rot}° (应≥40°) ← 关键: 站起来代偿旋转")

print("触发信号:", ", ".join(flags) if flags else "无")
n_sig = sum([
    s1_pos < -5.0,
    s2_pos > 0.10,
    s3_pos > 0.05,
    trail_heel_sw0 > 0.05,
])
print(f"代理信号触发: {n_sig}/4")
print()
print("候选判据 (供Jason定案):")
print("  EE判定 = S1丢失>5°+ S2头抬>0.08xSW0 + S3髋升>0.05xSW0 (至少2个)")
print("  且结合: 髋旋转<30° (否则可能是高手正常伸展)")
print()
print()
print("=" * 60)
print("补充: address 帧头部位置 vs impact 对比 (真正 EE 指标)")
nose_addr_y = nose_addr[1]
nose_imp_y  = nose_imp[1]
head_addr_vs_imp = nose_addr_y - nose_imp_y   # 正 = impact时头更高(EE)
print(f"  address fr{FR_ADDR} nose_y={nose_addr_y:.0f}")
print(f"  impact  fr{FR_IMPACT} nose_y={nose_imp_y:.0f}")
print(f"  impact头比address {'更高' if head_addr_vs_imp > 0 else '更低'}: {abs(head_addr_vs_imp):.0f}px = {abs(head_addr_vs_imp/SW0):.3f}×SW0")
print()

# 下杆头部最高点 (EE 的关键信号窗口: fr90-fr140)
peak_rise = 0; peak_fi = 0
addr_nose_y = nose_addr[1]
for fi in range(90, 141):
    n = kpt(fi, 'nose')
    rise_from_addr = addr_nose_y - n[1]   # addr 基准, 正=上升
    if rise_from_addr > peak_rise:
        peak_rise = rise_from_addr
        peak_fi = fi

print(f"下杆中段头部最高点 (fr90-140): fr{peak_fi}  最大上升={peak_rise:.0f}px = {peak_rise/SW0:.3f}×SW0 (相对address)")
print(f"  (标准EE: 下杆时头部明显高于address位 → 脊柱已伸展)")
print()

# 同步看那一帧的 S3 髋位
lh_pk  = kpt(peak_fi, 'left_hip')
rh_pk  = kpt(peak_fi, 'right_hip')
hc_pk  = ((lh_pk[0]+rh_pk[0])/2, (lh_pk[1]+rh_pk[1])/2)
hip_rise_pk = addr_nose_y - nose_addr[1]  # placeholder
hip_rise_from_addr = hip_c_addr[1] - hc_pk[1]   # 正=髋上升
print(f"同帧 fr{peak_fi}: 髋中心上升 vs address = {hip_rise_from_addr:.0f}px = {hip_rise_from_addr/SW0:+.3f}×SW0")

