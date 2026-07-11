#!/usr/bin/env python3
"""
swing_preprocess.py — 自动裁剪 + 多次挥杆检测切分
CUE_DESIGN_LANGUAGE: 复用 8-phase 手腕轨迹检测

用法:
  python3 swing_preprocess.py <video_path> [--out_dir DIR] [--run_pose]

输出:
  - 控制台: 检测到 N 次挥杆, 各次帧范围
  - 裁剪视频: <out_dir>/swing_N.mp4 (N=1,2,3...)
  - 若有 KP cache: 自动用 cache, 否则跑 RTMPose (需 --run_pose)

检测逻辑:
  手腕轨迹 (wrist_y) 出现完整 "上升→顶点→下降" 周期 = 一次挥杆
  顶点定义: 左/右手腕 y 最小值 (y 轴向下, 所以最小值=最高位置)
  自动裁剪范围: address-30f ~ finish+30f (各约1秒 buffer)
"""
import json, cv2, numpy as np, sys, argparse, subprocess
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter

ROOT = Path(__file__).parent

# ── 参数解析 ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Swing auto-trim & multi-swing detection")
parser.add_argument("video",  help="输入视频路径")
parser.add_argument("--out_dir", default=None, help="输出目录 (默认: 输入视频所在目录)")
parser.add_argument("--kp_cache", default=None, help="已有 KP cache JSON (跳过 pose 推理)")
parser.add_argument("--run_pose", action="store_true", help="自动跑 RTMPose 生成 KP cache")
parser.add_argument("--fps_out",  type=float, default=None, help="输出帧率 (默认: 同输入)")
parser.add_argument("--buffer_sec", type=float, default=1.0, help="挥杆前后 buffer 秒数 (默认1.0)")
args = parser.parse_args()

vid_path = Path(args.video)
assert vid_path.exists(), f"视频不存在: {vid_path}"

out_dir = Path(args.out_dir) if args.out_dir else vid_path.parent
out_dir.mkdir(parents=True, exist_ok=True)

# ── 读取 KP cache ─────────────────────────────────────────────────────────────
cache_path = None
if args.kp_cache:
    cache_path = Path(args.kp_cache)
else:
    # 自动查找同名 cache
    stem = vid_path.stem
    candidates = list(ROOT.glob(f"engine/kp_cache/**/{stem}.json"))
    if candidates:
        cache_path = candidates[0]
        print(f"[auto] 找到 KP cache: {cache_path}")

if cache_path is None:
    if args.run_pose:
        print("[INFO] 未找到 KP cache, 准备跑 RTMPose 推理 (需要 mmpose 环境)...")
        # 调用 infer 脚本 (如有)
        infer_script = ROOT / "engine" / "run_rtmpose.py"
        if infer_script.exists():
            subprocess.run([sys.executable, str(infer_script), str(vid_path)], check=True)
            candidates = list(ROOT.glob(f"engine/kp_cache/**/{vid_path.stem}.json"))
            if candidates:
                cache_path = candidates[0]
        if cache_path is None:
            print("[ERROR] RTMPose 推理后仍未找到 cache, 退出")
            sys.exit(1)
    else:
        print("[ERROR] 未找到 KP cache, 请提供 --kp_cache 或加 --run_pose")
        sys.exit(1)

# ── 读取视频信息 ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(str(vid_path))
FPS   = cap.get(cv2.CAP_PROP_FPS)
TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

fps_out = args.fps_out or FPS
BUFFER  = int(args.buffer_sec * FPS)

print(f"[INFO] 视频: {vid_path.name}  {W}x{H} {FPS:.1f}fps {TOTAL}fr ({TOTAL/FPS:.1f}s)")

# ── 读 KP cache → 提取手腕轨迹 ───────────────────────────────────────────────
with open(cache_path) as f:
    raw = json.load(f)

frames = raw["frames"]
NF = len(frames)
print(f"[INFO] KP cache: {NF} 帧  (视频: {TOTAL}帧)")

def kpt(fi, name):
    if fi >= NF: return (0, 9999, 0)
    persons = frames[fi].get("persons", [])
    if not persons: return (0, 9999, 0)
    p = persons[0]
    kp = p["keypoints"]
    k  = kp[name]
    return k['x'], k['y'], k['score']

# 提取左/右腕 y 轨迹 (y轴: 向下为正, 最小值=最高位置=挥杆顶点)
wrist_y = np.full(NF, np.nan)
conf_thr = 0.35

for fi in range(NF):
    lw = kpt(fi, 'left_wrist')
    rw = kpt(fi, 'right_wrist')
    lc, rc = lw[2], rw[2]
    vals = []
    if lc > conf_thr: vals.append(lw[1])
    if rc > conf_thr: vals.append(rw[1])
    if vals:
        wrist_y[fi] = min(vals)   # 取双手最高点 (y最小)

# 插值填补短暂遮挡
valid = ~np.isnan(wrist_y)
if valid.sum() < NF * 0.3:
    print("[WARN] 手腕轨迹缺失过多 (<30%), 结果可能不准")

xi = np.arange(NF)
wrist_y_interp = np.interp(xi, xi[valid], wrist_y[valid])

# 平滑
win = max(5, int(FPS * 0.15) | 1)   # ~0.15s 平滑窗, 必须奇数
if win % 2 == 0: win += 1
wrist_smooth = savgol_filter(wrist_y_interp, window_length=win, polyorder=2)

# ── 检测挥杆顶点 (wrist_y 局部最小值) ────────────────────────────────────────
# 顶点 = 手腕 y 最低 (最高位置), 即 wrist_smooth 的谷值
# find_peaks 找极大值, 所以对 -wrist_smooth 找极大值 = 对 wrist_smooth 找谷值
min_dist = int(FPS * 1.0)   # 两次挥杆顶点间距 >= 1.0s (更宽松)
prominence = max(60, H * 0.05)  # 手腕上升 >= max(60px, 5%画面高) (更宽松)

peaks, props = find_peaks(
    -wrist_smooth,
    distance=min_dist,
    prominence=prominence
)

print(f"\n[检测] 手腕顶点: {len(peaks)} 个 @ fr{list(peaks)}")

if len(peaks) == 0:
    print("[WARN] 未检测到完整挥杆顶点, 尝试放宽阈值...")
    peaks, _ = find_peaks(-wrist_smooth, distance=min_dist, prominence=prominence*0.5)
    print(f"  放宽后: {len(peaks)} 个 @ fr{list(peaks)}")

# ── 为每个顶点定位 address / finish ──────────────────────────────────────────
swings = []

for i, top_fr in enumerate(peaks):
    # Address: 顶点前找 "手腕开始显著上升" 的起点
    # 方法: 顶点前向前扫, 找 wrist_y 开始从平台上升的拐点
    search_back = max(0, top_fr - int(FPS * 4))
    seg = wrist_smooth[search_back:top_fr]
    if len(seg) < 5:
        addr_fr = search_back
    else:
        # 找段内最大值 (最低位置) 之后第一帧
        local_max_idx = np.argmax(seg)
        addr_fr = search_back + local_max_idx

    # Finish: 顶点后找 "手腕重新下落到接近起始高度" 的帧
    search_fwd = min(NF - 1, top_fr + int(FPS * 4))
    seg2 = wrist_smooth[top_fr:search_fwd]
    if len(seg2) < 5:
        fin_fr = search_fwd
    else:
        # 下落后找手腕回到相对稳定的位置: 找段内最大值
        local_max_idx2 = np.argmax(seg2)
        fin_fr = top_fr + local_max_idx2

    # 加 buffer
    clip_start = max(0,       addr_fr - BUFFER)
    clip_end   = min(NF - 1,  fin_fr  + BUFFER)

    swings.append({
        "idx":        i + 1,
        "top_fr":     int(top_fr),
        "addr_fr":    int(addr_fr),
        "fin_fr":     int(fin_fr),
        "clip_start": int(clip_start),
        "clip_end":   int(clip_end),
    })

# ── 打印结果 ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"检测到 {len(swings)} 次挥杆")
print(f"{'='*60}")
for s in swings:
    dur = (s['clip_end'] - s['clip_start']) / FPS
    print(f"  挥杆 #{s['idx']}:")
    print(f"    address ~ fr{s['addr_fr']}  top ~ fr{s['top_fr']}  finish ~ fr{s['fin_fr']}")
    print(f"    裁剪范围: fr{s['clip_start']} ~ fr{s['clip_end']}  ({dur:.1f}s)")

# ── 输出裁剪视频 ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"输出裁剪视频 → {out_dir}")

cap = cv2.VideoCapture(str(vid_path))
all_frames = []
print("  读取源帧...")
ret, frame = cap.read()
while ret:
    all_frames.append(frame)
    ret, frame = cap.read()
cap.release()
print(f"  读取完毕: {len(all_frames)} 帧")

for s in swings:
    out_name = f"{vid_path.stem}_swing{s['idx']}.mp4"
    out_path  = out_dir / out_name
    tmp_path  = out_dir / f"_tmp_{out_name}.avi"

    clip_frames = all_frames[s['clip_start']:s['clip_end']+1]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vw = cv2.VideoWriter(str(tmp_path), fourcc, fps_out, (W, H))
    for fr in clip_frames:
        vw.write(fr)
    vw.release()

    # 转 mp4
    subprocess.run([
        "ffmpeg", "-y", "-i", str(tmp_path),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p", str(out_path)
    ], check=True, capture_output=True)
    tmp_path.unlink(missing_ok=True)

    sz_kb = out_path.stat().st_size // 1024
    dur   = len(clip_frames) / FPS
    print(f"  => {out_path.name}  {len(clip_frames)}fr ({dur:.1f}s)  {sz_kb}KB")

print(f"\n[完成] {len(swings)} 个片段已输出")

# ── 保存检测结果 JSON ─────────────────────────────────────────────────────────
result_json = out_dir / f"{vid_path.stem}_swings.json"
import json as _json
with open(result_json, "w") as f:
    _json.dump({
        "source": str(vid_path),
        "fps":    FPS,
        "total_frames": TOTAL,
        "swing_count":  len(swings),
        "swings":       swings,
    }, f, indent=2, ensure_ascii=False)
print(f"[结果] 详情已保存: {result_json}")
