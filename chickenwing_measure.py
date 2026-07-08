"""
CUE-CHICKENWING-001 — 几何量测量
目标: 在 impact 帧计算鸡翅膀候选几何量, 给 Jason 看数值 → Jason 定阈值
输出: chickenwing_measure.jpg → Windows
"""

import json, math, cv2, numpy as np
from pathlib import Path

# ─── paths ───────────────────────────────────────────────────────────────────
ROOT        = Path("/home/jason/projects/swingcue-postest")
COACH_CACHE = ROOT / "engine/kp_cache/ghost004/coach-fo.json"
USER_CACHE  = ROOT / "engine/kp_cache/batch2/fo-ok-1.json"
COACH_VID   = Path("/mnt/c/Users/jason/Zening/Swingcue/教学视频/coach-video/coach-fo.mp4")
USER_VID    = ROOT / "input/fo-ok-1.mp4"
OUT_WIN     = Path("/mnt/c/Users/jason/Desktop/rtmpose_results/preview/chickenwing_measure.jpg")
OUT_WIN.parent.mkdir(parents=True, exist_ok=True)

# ─── frames of interest ──────────────────────────────────────────────────────
COACH_IMPACT_FR  = 52   # phase_report anchor
COACH_IMPACT_WIN = (49, 55)
USER_IMPACT_FR   = 88   # GHOST-004 GT
USER_IMPACT_WIN  = (84, 92)

# ─── COCO-17 keypoint accessor ───────────────────────────────────────────────
def get_kp(cache_frames, frame_idx):
    """Return dict {name: (x,y,score)} for given frame index."""
    fr = cache_frames[frame_idx]
    if not fr['persons']:
        return {}
    kp_raw = fr['persons'][0]['keypoints']
    return {k: (v['x'], v['y'], v['score']) for k,v in kp_raw.items()}

def pt(kp, name):
    if name not in kp:
        return None
    return np.array(kp[name][:2])

# ─── geometry helpers ─────────────────────────────────────────────────────────
def angle_deg(a, b, c):
    """Angle at vertex b (a-b-c), degrees."""
    ba = a - b;  bc = c - b
    cos_ = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return math.degrees(math.acos(np.clip(cos_, -1, 1)))

def point_to_line_dist(p, a, b):
    """Signed perpendicular distance from p to line a→b (+ = left of direction)."""
    ab = b - a
    ap = p - a
    cross = ab[0]*ap[1] - ab[1]*ap[0]
    return cross / (np.linalg.norm(ab) + 1e-9)

def triangle_area(p1, p2, p3):
    return 0.5 * abs((p2-p1)[0]*(p3-p1)[1] - (p3-p1)[0]*(p2-p1)[1])

# ─── compute metrics ─────────────────────────────────────────────────────────
def compute_metrics(kp, label=""):
    """
    Chicken wing candidate geometric quantities (face-on view, right-handed golfer).
    Lead arm = LEFT arm.  Trail arm = RIGHT arm.
    """
    ls  = pt(kp, 'left_shoulder')
    le  = pt(kp, 'left_elbow')
    lw  = pt(kp, 'left_wrist')
    rs  = pt(kp, 'right_shoulder')
    re  = pt(kp, 'right_elbow')
    rw  = pt(kp, 'right_wrist')
    lh  = pt(kp, 'left_hip')
    rh  = pt(kp, 'right_hip')

    results = {}

    # ── M1: Lead elbow angle (left shoulder–elbow–wrist)
    # Straight arm = 180°, chicken wing = large bend (< ~155°)
    if ls is not None and le is not None and lw is not None:
        results['M1_lead_elbow_angle_deg'] = angle_deg(ls, le, lw)

    # ── M2: Trail elbow angle (right shoulder–elbow–wrist)
    if rs is not None and re is not None and rw is not None:
        results['M2_trail_elbow_angle_deg'] = angle_deg(rs, re, rw)

    # ── M3: Lead elbow lateral flare
    # Perpendicular distance of lead elbow from shoulder-wrist line.
    # Positive = elbow flares OUT (away from body center), pixels
    if ls is not None and le is not None and lw is not None:
        results['M3_lead_elbow_flare_px'] = point_to_line_dist(le, ls, lw)

    # ── M4: Trail elbow lateral flare (same logic, right side)
    if rs is not None and re is not None and rw is not None:
        results['M4_trail_elbow_flare_px'] = -point_to_line_dist(re, rs, rw)  # negate: right side

    # ── M5: Arm triangle compactness
    # Area of triangle [l_shoulder, r_shoulder, mid_wrist].
    # Compact arms = small area. Chicken wing = large area.
    if ls is not None and rs is not None and lw is not None and rw is not None:
        mid_wrist = (lw + rw) / 2
        results['M5_arm_triangle_area_px2'] = triangle_area(ls, rs, mid_wrist)

    # ── M6: Lead elbow-to-torso gap
    # Horizontal dist from lead elbow to torso midline (avg of shoulder & hip mid-x).
    # Chicken wing = elbow far from torso.
    if le is not None and ls is not None and rs is not None:
        mid_shoulder_x = (ls[0] + rs[0]) / 2
        results['M6_lead_elbow_to_midline_px'] = le[0] - mid_shoulder_x  # + = elbow left of center

    # ── M7: Inter-wrist distance normalized by shoulder width
    # Compact = wrists close together relative to shoulder width
    if lw is not None and rw is not None and ls is not None and rs is not None:
        sw = np.linalg.norm(ls - rs)
        ww = np.linalg.norm(lw - rw)
        results['M7_wrist_dist_norm'] = ww / (sw + 1e-9)

    print(f"\n=== {label} ===")
    for k,v in results.items():
        print(f"  {k}: {v:.2f}")
    return results

# ─── load caches ─────────────────────────────────────────────────────────────
with open(COACH_CACHE) as f: coach_data = json.load(f)
with open(USER_CACHE)  as f: user_data  = json.load(f)
coach_frames = coach_data['frames']
user_frames  = user_data['frames']

# compute across impact window to find worst/best
print("\n── COACH IMPACT WINDOW ──")
coach_metrics_all = []
for fi in range(*COACH_IMPACT_WIN):
    kp = get_kp(coach_frames, fi)
    m  = compute_metrics(kp, f"coach fr{fi:03d}")
    coach_metrics_all.append((fi, m))

print("\n── USER IMPACT WINDOW ──")
user_metrics_all = []
for fi in range(*USER_IMPACT_WIN):
    kp = get_kp(user_frames, fi)
    m  = compute_metrics(kp, f"user fr{fi:03d}")
    user_metrics_all.append((fi, m))

# anchor frames for visualization
coach_kp = get_kp(coach_frames, COACH_IMPACT_FR)
user_kp  = get_kp(user_frames,  USER_IMPACT_FR)
coach_m  = compute_metrics(coach_kp, "COACH fr052 (anchor)")
user_m   = compute_metrics(user_kp,  "USER fr088 (anchor)")

# ─── render comparison image ─────────────────────────────────────────────────
def extract_frame(vid_path, frame_idx):
    cap = cv2.VideoCapture(str(vid_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

coach_img = extract_frame(COACH_VID, COACH_IMPACT_FR)
user_img  = extract_frame(USER_VID,  USER_IMPACT_FR)

def draw_arm_skeleton(img, kp, color_lead, color_trail, color_text):
    """Draw arm joints + elbow flare indicator."""
    def ipx(name):
        p = pt(kp, name)
        return None if p is None else (int(p[0]), int(p[1]))

    ls = ipx('left_shoulder');  le = ipx('left_elbow');  lw = ipx('left_wrist')
    rs = ipx('right_shoulder'); re = ipx('right_elbow'); rw = ipx('right_wrist')
    lh = ipx('left_hip');       rh = ipx('right_hip')

    # lead arm (left) — solid thick
    if ls and le: cv2.line(img, ls, le, color_lead, 3)
    if le and lw: cv2.line(img, le, lw, color_lead, 3)
    # trail arm (right) — dashed thin
    if rs and re: cv2.line(img, rs, re, color_trail, 2)
    if re and rw: cv2.line(img, re, rw, color_trail, 2)
    # shoulder line
    if ls and rs: cv2.line(img, ls, rs, (180,180,180), 1)
    # joint circles
    for p, r in [(ls,6),(le,8),(lw,6),(rs,5),(re,7),(rw,5)]:
        if p: cv2.circle(img, p, r, color_lead if p in [ls,le,lw] else color_trail, -1)

    # elbow flare arrow (lead elbow only)
    if ls and le and lw:
        ls_np = np.array(ls); lw_np = np.array(lw); le_np = np.array(le)
        # project elbow onto shoulder-wrist line → ideal position
        ab = lw_np - ls_np
        t  = np.dot(le_np - ls_np, ab) / (np.dot(ab, ab) + 1e-9)
        proj = ls_np + t * ab
        proj_pt = (int(proj[0]), int(proj[1]))
        cv2.arrowedLine(img, proj_pt, le, (0,0,255), 2, tipLength=0.3)
        cv2.putText(img, "elbow flare", (le[0]+10, le[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text, 1)

    # hip line for reference
    if lh and rh: cv2.line(img, lh, rh, (100,100,100), 1)

    return img

if coach_img is not None:
    draw_arm_skeleton(coach_img, coach_kp,
                      color_lead=(0,255,100), color_trail=(0,200,80), color_text=(255,255,255))
    cv2.putText(coach_img, f"COACH impact fr052", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,100), 2)
    cv2.putText(coach_img, f"M1 lead_elbow: {coach_m.get('M1_lead_elbow_angle_deg',0):.1f}deg", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(coach_img, f"M3 lead_flare: {coach_m.get('M3_lead_elbow_flare_px',0):.1f}px", (20,110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(coach_img, f"M5 arm_tri:    {coach_m.get('M5_arm_triangle_area_px2',0):.0f}px2", (20,140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

if user_img is not None:
    draw_arm_skeleton(user_img, user_kp,
                      color_lead=(0,100,255), color_trail=(0,80,200), color_text=(255,255,255))
    cv2.putText(user_img, f"USER impact fr088", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,100,255), 2)
    cv2.putText(user_img, f"M1 lead_elbow: {user_m.get('M1_lead_elbow_angle_deg',0):.1f}deg", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(user_img, f"M3 lead_flare: {user_m.get('M3_lead_elbow_flare_px',0):.1f}px", (20,110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(user_img, f"M5 arm_tri:    {user_m.get('M5_arm_triangle_area_px2',0):.0f}px2", (20,140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

# stack side-by-side (resize to same height)
if coach_img is not None and user_img is not None:
    h = max(coach_img.shape[0], user_img.shape[0])
    def pad_h(img, H):
        dh = H - img.shape[0]
        return np.pad(img, ((0,dh),(0,0),(0,0)), constant_values=30)
    ci = pad_h(coach_img, h)
    ui = pad_h(user_img, h)

    # add separator
    sep = np.full((h, 6, 3), 60, dtype=np.uint8)
    combined = np.hstack([ci, sep, ui])

    # summary table at bottom
    summary_h = 240
    summary = np.full((summary_h, combined.shape[1], 3), 20, dtype=np.uint8)
    lines = [
        "=== GEOMETRIC QUANTITY SUMMARY (impact anchor frames) ===",
        "",
        f"  M1  Lead elbow angle (shoulder-elbow-wrist):  coach={coach_m.get('M1_lead_elbow_angle_deg',0):.1f}deg   user={user_m.get('M1_lead_elbow_angle_deg',0):.1f}deg",
        f"  M2  Trail elbow angle:                         coach={coach_m.get('M2_trail_elbow_angle_deg',0):.1f}deg   user={user_m.get('M2_trail_elbow_angle_deg',0):.1f}deg",
        f"  M3  Lead elbow lateral flare (px):            coach={coach_m.get('M3_lead_elbow_flare_px',0):.1f}px     user={user_m.get('M3_lead_elbow_flare_px',0):.1f}px",
        f"  M4  Trail elbow lateral flare (px):           coach={coach_m.get('M4_trail_elbow_flare_px',0):.1f}px     user={user_m.get('M4_trail_elbow_flare_px',0):.1f}px",
        f"  M5  Arm triangle area (px2):                  coach={coach_m.get('M5_arm_triangle_area_px2',0):.0f}px2   user={user_m.get('M5_arm_triangle_area_px2',0):.0f}px2",
        f"  M6  Lead elbow to midline (px):               coach={coach_m.get('M6_lead_elbow_to_midline_px',0):.1f}px     user={user_m.get('M6_lead_elbow_to_midline_px',0):.1f}px",
        f"  M7  Wrist dist / shoulder width:              coach={coach_m.get('M7_wrist_dist_norm',0):.3f}        user={user_m.get('M7_wrist_dist_norm',0):.3f}",
        "",
        "  Jason: 请看数值+图, 决定: 哪个M最直观? 阈值定多少?",
    ]
    for i, line in enumerate(lines):
        cv2.putText(summary, line, (20, 30+i*20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (200,200,80) if i==0 else (180,180,180), 1)

    final = np.vstack([combined, summary])
    cv2.imwrite(str(OUT_WIN), final)
    print(f"\n=> saved: {OUT_WIN}")
else:
    print("ERROR: could not extract frames from video")
