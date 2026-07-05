"""
engine/features/triline_geometry.py
正面三线几何测量层 v0.1

Three reference lines measured from RTMPose keypoints in face-on camera view:
  - Shoulder line (肩线): left_shoulder ↔ right_shoulder
  - Pelvis line  (髋线): left_hip       ↔ right_hip
  - Ankle line   (踝线): left_ankle     ↔ right_ankle

All features per-frame. NaN where kp_guard fails.
Normalization: all displacement/width features divided by shoulder_width (肩宽归一化).

# future: 通用关节点几何 → 下沉 core; 高尔夫判据 → 保留 golf-pack

Author: swingcue-postest  Date: 2026-07-04
"""

import math
from typing import Optional, Dict, Any

KP_THR   = 0.30   # minimum keypoint confidence score
MIN_SHW  = 15.0   # minimum shoulder width (px); below → DTL-ish, NaN

NaN = float("nan")


# ─── low-level helpers ────────────────────────────────────────────────────────

def _safe_pt(kps: dict, name: str) -> Optional[tuple]:
    """Return (x, y) if joint passes kp_guard, else None."""
    k = kps.get(name, {})
    if k.get("score", 0) >= KP_THR and (k.get("x", 0) != 0 or k.get("y", 0) != 0):
        return (k["x"], k["y"])
    return None


def _line_angle_deg(p1: tuple, p2: tuple) -> float:
    """
    Angle of line (p1→p2) w.r.t. image horizontal, in degrees.
    Positive = p2 is below p1 in image coords (y increases downward).
    Range: [-180, 180).
    """
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))


def _lateral_tilt_deg(low_pt: tuple, high_pt: tuple) -> float:
    """
    Lateral tilt of the axis (low_pt→high_pt) vs image vertical.
    Image vertical = (0, -1) direction (pointing upward in image coords).

    Sign convention: positive = axis leans RIGHT in image.
    For face-on right-handed golfer: positive = leans toward trail side.

    Formula: atan2(dx, -dy)  where dx = high.x - low.x, dy = high.y - low.y
    """
    dx = high_pt[0] - low_pt[0]
    dy = high_pt[1] - low_pt[1]
    return math.degrees(math.atan2(dx, -dy))


# ─── per-frame computation ────────────────────────────────────────────────────

def compute_triline_frame(
    kps: dict,
    addr_pelvis_cx: Optional[float] = None,
    addr_sh_cx: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute all triline geometry features for a single frame.

    Args:
        kps: keypoint dict {name: {"x": float, "y": float, "score": float}}
             as returned by RTMPose (via engine/a_measurement/pose_pipeline.py)
        addr_pelvis_cx: pelvis center-x at address frame (for displacement norm)
        addr_sh_cx:     shoulder center-x at address (for shoulder displacement)

    Returns:
        dict with keys below. Values are float or NaN where invalid.

    Output fields:
        # Shoulder line (§2.2)
        shoulder_width_px       - raw width in pixels (not normalized)
        shoulder_width_norm     - always 1.0 when valid (self-reference)
        shoulder_center_x_norm  - (sh_cx - addr_sh_cx) / sh_w, NaN if no addr
        shoulder_line_angle     - angle of shoulder line vs horizontal (deg)
        shoulder_lateral_tilt   - torso axis (hip→shoulder) vs vertical (deg)
                                  POSITIVE = leans right (trail side)
                                  (= spine_lateral_tilt from v0.1)

        # Pelvis line (§2.1)
        pelvis_width_norm       - pelvis_width / sh_w
        pelvis_center_x_norm    - (pel_cx - addr_pel_cx) / sh_w
        pelvis_line_angle       - angle of pelvis line vs horizontal (deg)

        # Ankle line (§2.3)
        ankle_width_norm        - ankle_width / sh_w

        # Cross-line (§2.4)
        shoulder_pelvis_separation - shoulder_line_angle - pelvis_line_angle (deg)
                                     (2D X-factor proxy)

        # Meta
        kp_guard    - "ok" | reason string if guard failed
        sh_w        - shoulder width in px (for audit)
    """

    result: Dict[str, Any] = {
        "shoulder_width_px":          NaN,
        "shoulder_width_norm":        NaN,
        "shoulder_center_x_norm":     NaN,
        "shoulder_line_angle":        NaN,
        "shoulder_lateral_tilt":      NaN,
        "pelvis_width_norm":          NaN,
        "pelvis_center_x_norm":       NaN,
        "pelvis_line_angle":          NaN,
        "ankle_width_norm":           NaN,
        "shoulder_pelvis_separation": NaN,
        "kp_guard":                   "ok",
        "sh_w":                       NaN,
    }

    # ── Shoulder line ─────────────────────────────────────────────────────────
    ls = _safe_pt(kps, "left_shoulder")
    rs = _safe_pt(kps, "right_shoulder")
    if not (ls and rs):
        result["kp_guard"] = "missing_shoulder"
        return result

    sh_w = math.hypot(rs[0] - ls[0], rs[1] - ls[1])
    result["sh_w"] = sh_w

    if sh_w < MIN_SHW:
        result["kp_guard"] = f"sh_w_small:{sh_w:.1f}px"
        return result

    result["shoulder_width_px"]   = sh_w
    result["shoulder_width_norm"] = 1.0  # normalized by itself

    sh_cx = (ls[0] + rs[0]) / 2
    sh_cy = (ls[1] + rs[1]) / 2

    if addr_sh_cx is not None:
        result["shoulder_center_x_norm"] = (sh_cx - addr_sh_cx) / sh_w

    result["shoulder_line_angle"] = _line_angle_deg(ls, rs)

    # ── Pelvis line ───────────────────────────────────────────────────────────
    lh = _safe_pt(kps, "left_hip")
    rh = _safe_pt(kps, "right_hip")
    if lh and rh:
        ph_w = math.hypot(rh[0] - lh[0], rh[1] - lh[1])
        result["pelvis_width_norm"] = ph_w / sh_w

        pel_cx = (lh[0] + rh[0]) / 2
        if addr_pelvis_cx is not None:
            result["pelvis_center_x_norm"] = (pel_cx - addr_pelvis_cx) / sh_w

        result["pelvis_line_angle"] = _line_angle_deg(lh, rh)

        # shoulder_lateral_tilt: torso axis (pelvis_mid → shoulder_mid) vs vertical
        # Same as spine_lateral_tilt v0.1 formula
        pel_cy = (lh[1] + rh[1]) / 2
        result["shoulder_lateral_tilt"] = _lateral_tilt_deg(
            (pel_cx, pel_cy), (sh_cx, sh_cy)
        )

        # X-factor proxy
        if not math.isnan(result["shoulder_line_angle"]) and not math.isnan(result["pelvis_line_angle"]):
            result["shoulder_pelvis_separation"] = (
                result["shoulder_line_angle"] - result["pelvis_line_angle"]
            )

    # ── Ankle line ────────────────────────────────────────────────────────────
    la = _safe_pt(kps, "left_ankle")
    ra = _safe_pt(kps, "right_ankle")
    if la and ra:
        aw = math.hypot(ra[0] - la[0], ra[1] - la[1])
        result["ankle_width_norm"] = aw / sh_w

    return result


# ─── sequence helpers ─────────────────────────────────────────────────────────

def detect_address_frame(frames: list, addr_pct: float = 0.05) -> Optional[int]:
    """
    Return index of the address frame in a list of frame dicts.
    Uses first addr_pct of frames (by position), picks frame with max shoulder width
    (shoulders widest = most face-on, least rotated).
    """
    n = len(frames)
    if not n:
        return None
    end = max(1, int(n * addr_pct))
    best_i, best_w = 0, -1.0
    for i in range(min(end, n)):
        fd = frames[i]
        p  = fd.get("persons", [])
        if not p:
            continue
        kps = p[0].get("keypoints", {})
        ls  = _safe_pt(kps, "left_shoulder")
        rs  = _safe_pt(kps, "right_shoulder")
        if ls and rs:
            w = math.hypot(rs[0] - ls[0], rs[1] - ls[1])
            if w > best_w:
                best_w, best_i = w, i
    return best_i


def detect_top_frame(frames: list, front_pct: float = 0.65) -> Optional[int]:
    """
    Return index of the top-of-backswing frame.
    Proxy: min wrist Y (highest wrist position) in front front_pct of frames.
    """
    n = len(frames)
    if not n:
        return None
    end   = max(1, int(n * front_pct))
    best_i, best_wy = None, float("inf")
    for i in range(min(end, n)):
        fd  = frames[i]
        p   = fd.get("persons", [])
        if not p:
            continue
        kps = p[0].get("keypoints", {})
        lw  = _safe_pt(kps, "left_wrist")
        rw  = _safe_pt(kps, "right_wrist")
        wy  = min(p[1] for p in [lw, rw] if p)  if (lw or rw) else None
        if wy is not None and wy < best_wy:
            best_wy, best_i = wy, i
    return best_i


def compute_triline_sequence(
    kp_json: dict,
    addr_pct: float = 0.05,
    front_pct: float = 0.65,
) -> dict:
    """
    Run compute_triline_frame across all frames in a kp_json dict.
    Auto-detects address and top frames.

    Returns:
        {
          "frames": [feature_dict, ...],  # one per kp frame
          "addr_idx": int,
          "top_idx":  int,
          "addr_features": dict,
          "top_features":  dict,
          "n_valid": int,                 # frames where kp_guard=="ok"
        }
    """
    frames = kp_json.get("frames", [])

    addr_idx = detect_address_frame(frames, addr_pct)
    top_idx  = detect_top_frame(frames, front_pct)

    # Get address anchor values for displacement normalization
    addr_pelvis_cx = None
    addr_sh_cx     = None
    if addr_idx is not None:
        fd = frames[addr_idx]
        p  = fd.get("persons", [])
        if p:
            kps = p[0].get("keypoints", {})
            lh  = _safe_pt(kps, "left_hip");  rh = _safe_pt(kps, "right_hip")
            ls  = _safe_pt(kps, "left_shoulder"); rs = _safe_pt(kps, "right_shoulder")
            if lh and rh:
                addr_pelvis_cx = (lh[0] + rh[0]) / 2
            if ls and rs:
                addr_sh_cx = (ls[0] + rs[0]) / 2

    feature_frames = []
    n_valid = 0
    for fd in frames:
        p = fd.get("persons", [])
        if p:
            kps = p[0].get("keypoints", {})
            feat = compute_triline_frame(kps, addr_pelvis_cx, addr_sh_cx)
        else:
            feat = {"kp_guard": "no_person", "shoulder_lateral_tilt": NaN}
        feature_frames.append(feat)
        if feat.get("kp_guard") == "ok":
            n_valid += 1

    addr_feat = feature_frames[addr_idx] if (addr_idx is not None and addr_idx < len(feature_frames)) else {}
    top_feat  = feature_frames[top_idx]  if (top_idx  is not None and top_idx  < len(feature_frames)) else {}

    return {
        "frames":        feature_frames,
        "addr_idx":      addr_idx,
        "top_idx":       top_idx,
        "addr_features": addr_feat,
        "top_features":  top_feat,
        "n_valid":       n_valid,
    }


# ─── rendering ───────────────────────────────────────────────────────────────

def render_triline_frame(
    bgr: "np.ndarray",
    kps: dict,
    features: dict,
    label: str = "",
) -> "np.ndarray":
    """
    Draw hip/shoulder/ankle lines on a BGR frame.

    Lines:
      Shoulder line — RED
      Pelvis line   — BLUE
      Ankle line    — GREEN
      Torso axis    — ORANGE dashed (hip_mid→shoulder_mid)
      Vertical ref  — WHITE dashed (through shoulder_mid)

    Returns annotated BGR image (copy).
    """
    import cv2
    import numpy as np

    img = bgr.copy()
    h, w = img.shape[:2]

    def pt(name):
        k = kps.get(name, {})
        if k.get("score", 0) >= KP_THR:
            return (int(k["x"]), int(k["y"]))
        return None

    def draw_line(p1, p2, color, thickness=3):
        if p1 and p2:
            cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

    def draw_dot(p, color, r=8):
        if p:
            cv2.circle(img, p, r, color, -1, cv2.LINE_AA)
            cv2.circle(img, p, r + 2, (0, 0, 0), 2, cv2.LINE_AA)

    # Joint points
    ls = pt("left_shoulder");  rs = pt("right_shoulder")
    lh = pt("left_hip");       rh = pt("right_hip")
    la = pt("left_ankle");     ra = pt("right_ankle")

    # Shoulder line (RED)
    draw_line(ls, rs, (0, 0, 220), 4)
    draw_dot(ls, (0, 50, 200))
    draw_dot(rs, (0, 50, 200))

    # Pelvis line (BLUE)
    draw_line(lh, rh, (220, 80, 0), 4)
    draw_dot(lh, (180, 60, 0))
    draw_dot(rh, (180, 60, 0))

    # Ankle line (GREEN)
    draw_line(la, ra, (0, 180, 0), 3)
    draw_dot(la, (0, 140, 0), r=6)
    draw_dot(ra, (0, 140, 0), r=6)

    # Torso axis (ORANGE): hip_mid → shoulder_mid, extended
    if ls and rs and lh and rh:
        sh_cx = (ls[0] + rs[0]) // 2;  sh_cy = (ls[1] + rs[1]) // 2
        hp_cx = (lh[0] + rh[0]) // 2;  hp_cy = (lh[1] + rh[1]) // 2
        # Extend line both ways
        dx = sh_cx - hp_cx; dy = sh_cy - hp_cy
        scale = 2.5
        ext_top = (int(sh_cx + dx * scale), int(sh_cy + dy * scale))
        ext_bot = (int(hp_cx - dx * scale), int(hp_cy - dy * scale))
        cv2.line(img, ext_bot, ext_top, (0, 165, 255), 2, cv2.LINE_AA)
        # Vertical reference through shoulder_mid (WHITE dashed)
        for y0 in range(0, h, 20):
            y1 = min(y0 + 10, h - 1)
            cv2.line(img, (sh_cx, y0), (sh_cx, y1), (255, 255, 255), 1, cv2.LINE_AA)

    # Annotation text
    tilt = features.get("shoulder_lateral_tilt", NaN)
    sway = features.get("pelvis_center_x_norm", NaN)
    tilt_s = f"{tilt:+.1f}" if not math.isnan(tilt) else "N/A"
    sway_s = f"{sway:+.3f}" if not math.isnan(sway) else "N/A"

    # Background rect
    txt_lines = [
        label,
        f"tilt: {tilt_s}  deg",
        f"sway: {sway_s}  sw",
    ]
    y0_txt = 18
    for tl in txt_lines:
        (tw, th2), _ = cv2.getTextSize(tl, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(img, (6, y0_txt - th2 - 4), (10 + tw, y0_txt + 4), (0, 0, 0), -1)
        cv2.putText(img, tl, (8, y0_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
        y0_txt += th2 + 12

    # Legend (bottom)
    legends = [
        ("shoulder",    (0, 0, 220)),
        ("pelvis",      (220, 80, 0)),
        ("ankle",       (0, 180, 0)),
        ("torso axis",  (0, 165, 255)),
    ]
    xleg = 8
    for lname, col in legends:
        cv2.rectangle(img, (xleg, h - 22), (xleg + 20, h - 8), col, -1)
        cv2.putText(img, lname, (xleg + 24, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        xleg += 90

    return img
