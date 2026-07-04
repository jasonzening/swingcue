"""
engine/profiler/video_profiler.py
Layer 1 Video Profiler — unified video identity card.

Industry-agnostic. Answers "what is this video?" only.
No golf action diagnosis or fault labels.

Output: VideoProfile (dataclass) serializable to identity-card JSON.

Components integrated:
  - camera_view.py      : geometric camera angle (face_on / dtl / other / uncertain)
  - swing_type_detector : wrist/shoulder motion proxy (full_swing / static_demo / mixed)
  - layout_detector     : single / split_screen / pip
  - camera_profile.py   : subject_center, subject_height_ratio, camera_height
  - orientation/resolver: handedness (requires B-layer anchors; runs simplified if no anchors)
  - layer0 perception_gate data: is_golf_swing, persons (loaded from existing record if present)

Usage (minimal):
  from engine.profiler.video_profiler import VideoProfiler
  profiler = VideoProfiler()
  profile = profiler.profile_from_kp_json(
      video_id="fo-eet-1",
      kp_json=<loaded dict>,
      video_width=1920,
      video_height=1080,
  )
  profile.to_dict()

Usage (with existing layer0 record):
  profile = profiler.profile_from_kp_json(..., layer0_record=<dict>)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

from engine.profiler.camera_view import detect_camera_view, CameraViewResult
from engine.profiler.swing_type_detector import detect_swing_type
from engine.profiler.layout_detector import detect_layout
from engine.profiler.camera_profile import compute_camera_profile

# Optional: orientation resolver (requires B-layer measurements)
# We run a lightweight handedness probe directly from kp_json without full B-layer
KP_THR: float = 0.30


# ── Lightweight handedness probe (no B-layer needed) ──────────────────────────
def _probe_handedness(kp_json: dict, camera_view: str = "unknown") -> tuple[str, float, str]:
    """
    Handedness probe from kp_json.  Uses wrist-x extremum at top of backswing.

    Algorithm (v3):
      1. Detect golfer facing direction from shoulder-x asymmetry at address.
         Standard face-on: left_shoulder_x > right_shoulder_x (golfer faces LEFT in image).
         DTL: shoulder delta ≈ 0 (stacked in depth) → handedness unreliable → unknown.
      2. Find wrist-x extremum in first 65% of clip:
         Standard facing (+1): min-x = top of backswing (wrist goes image-left = player-right).
         Mirrored facing (-1): max-x = top of backswing (wrist goes image-right = player-right).
      3. body_delta = (extremum_x - addr_x) × facing_sign.
         body_delta < 0 → wrist moved to player's right (trail side) → right-handed.

    DTL clips return unknown + needs_human (correct per spec).
    """
    frames = kp_json.get("frames", [])
    n = len(frames)
    if n < 10:
        return ("unknown", 0.2, "too_few_frames")

    def get_kps(fd):
        p = fd.get("persons", [])
        return p[0].get("keypoints", {}) if p else {}

    # ── Step 1: Shoulder facing direction at address ──────────────────────────
    addr_lo = max(1, int(n * 0.04))
    addr_hi = int(n * 0.20)
    sh_deltas = []
    for i in range(addr_lo, addr_hi, 3):
        kps = get_kps(frames[i])
        ls = kps.get("left_shoulder",  {})
        rs = kps.get("right_shoulder", {})
        if ls.get("score", 0) >= 0.30 and rs.get("score", 0) >= 0.30:
            sh_deltas.append(ls["x"] - rs["x"])

    if not sh_deltas:
        return ("unknown", 0.2, "no_shoulder_kp")

    mean_sh_delta = sum(sh_deltas) / len(sh_deltas)

    # DTL: shoulder delta ≈ 0 → cannot reliably determine handedness from image-x
    if camera_view == "dtl" or abs(mean_sh_delta) < 20:
        return ("unknown", 0.40, f"dtl_or_ambiguous_facing sh_delta={mean_sh_delta:.1f}")

    facing_sign = 1 if mean_sh_delta > 0 else -1  # +1=standard, -1=mirrored

    # ── Step 2: Wrist-x series in first 65% ─────────────────────────────────
    valid_wx = []
    for i in range(0, min(n, int(n * 0.65) + 1)):
        kps = get_kps(frames[i])
        lw = kps.get("left_wrist",  {})
        rw = kps.get("right_wrist", {})
        lx = lw["x"] if lw.get("score", 0) >= KP_THR else None
        rx = rw["x"] if rw.get("score", 0) >= KP_THR else None
        if lx is not None and rx is not None:
            valid_wx.append((i, (lx + rx) / 2))
        elif lx is not None:
            valid_wx.append((i, lx))
        elif rx is not None:
            valid_wx.append((i, rx))

    if len(valid_wx) < 5:
        return ("unknown", 0.25, "insufficient_wrist_kp")

    # Address wrist x mean
    addr_wx = [t for t in valid_wx if addr_lo <= t[0] <= addr_hi]
    if not addr_wx:
        addr_wx = valid_wx[:5]
    addr_x_mean = sum(t[1] for t in addr_wx) / len(addr_wx)

    # Extremum: standard facing → min-x; mirrored → max-x
    top_x = min(v[1] for v in valid_wx) if facing_sign > 0 else max(v[1] for v in valid_wx)
    delta_x = top_x - addr_x_mean

    if abs(delta_x) < 15:
        return ("unknown", 0.30, f"delta_too_small={delta_x:.1f}px")

    body_delta = delta_x * facing_sign
    handedness = "right" if body_delta < 0 else "left"
    conf = min(0.82, 0.50 + abs(delta_x) / 250.0)

    return (handedness, round(conf, 3),
            f"facing={facing_sign} delta_img={delta_x:.0f}px body={body_delta:.0f}")


# ── Main VideoProfile dataclass ────────────────────────────────────────────────

@dataclass
class VideoProfile:
    video_id:      str
    is_golf_swing: Optional[bool]
    persons:       Optional[int]
    layout:        str           # single / split_screen / pip
    camera_view:   str           # face_on / dtl / other / uncertain
    swing_type:    str           # full_swing / static_demo / mixed / unknown
    handedness:    str           # right / left / unknown
    camera_profile: dict         # subject_center_x/y, subject_height_ratio, camera_height
    confidence:    dict          # per-field 0-1
    needs_human:   List[str]
    notes:         str

    def to_dict(self) -> dict:
        return {
            "video_id":      self.video_id,
            "is_golf_swing": self.is_golf_swing,
            "persons":       self.persons,
            "layout":        self.layout,
            "camera_view":   self.camera_view,
            "swing_type":    self.swing_type,
            "handedness":    self.handedness,
            "camera_profile": self.camera_profile,
            "confidence":    self.confidence,
            "needs_human":   self.needs_human,
            "notes":         self.notes,
        }


# ── Profiler class ─────────────────────────────────────────────────────────────

NEEDS_HUMAN_CONF_THR: float = 0.55   # below this → field flagged as needs_human


class VideoProfiler:
    """
    Produce a VideoProfile (identity card) for a video given its kp_json.

    Inputs:
      video_id        : unique identifier string
      kp_json         : RTMPose keypoint cache (dict, loaded from JSON)
      video_width/height: frame dimensions (optional but improves camera_profile)
      layer0_record   : existing Layer0 perception gate record (optional)
      split_hint      : "split_screen" | "single" | None — from splitter if available
      address_frame   : known address anchor frame index (optional, from B-layer)
    """

    def profile_from_kp_json(
        self,
        video_id:       str,
        kp_json:        dict,
        video_width:    Optional[int]  = None,
        video_height:   Optional[int]  = None,
        layer0_record:  Optional[dict] = None,
        split_hint:     Optional[str]  = None,
        address_frame:  Optional[int]  = None,
    ) -> VideoProfile:

        needs_human: List[str] = []
        note_parts:  List[str] = []
        conf:        dict      = {}

        # ── 1. is_golf_swing + persons (from layer0 or fallback) ──────────────
        is_golf_swing: Optional[bool] = None
        persons:       Optional[int]  = None

        if layer0_record:
            verdict = layer0_record.get("verdict", "")
            is_golf_swing = verdict == "PASS" or layer0_record.get("is_golf_swing", None)
            persons = layer0_record.get("persons", None)
            # Try to get from frame data
            if persons is None:
                frames_data = layer0_record.get("frames", [])
                if frames_data:
                    persons_vals = [f.get("q2_persons", 0) for f in frames_data if f.get("hard_pass")]
                    if persons_vals:
                        persons = max(set(persons_vals), key=persons_vals.count)
        else:
            # Probe from kp_json: count non-empty frames
            non_empty = sum(1 for fd in kp_json.get("frames", []) if fd.get("persons"))
            if non_empty > len(kp_json.get("frames", [])) * 0.3:
                is_golf_swing = True   # assume true if RTMPose detected person most frames
                persons = 1
            note_parts.append("no_layer0_record")

        # ── 2. Layout ──────────────────────────────────────────────────────────
        layout_res = detect_layout(
            kp_json,
            video_width=video_width,
            video_height=video_height,
            split_hint=split_hint,
        )
        layout        = layout_res["layout"]
        conf["layout"] = layout_res["confidence"]
        if conf["layout"] < NEEDS_HUMAN_CONF_THR:
            needs_human.append("layout")

        # ── 3. Camera view (geometric, Gate 1 validated) ──────────────────────
        cam_res: CameraViewResult = detect_camera_view(kp_json)
        camera_view        = cam_res.camera_view
        conf["camera_view"] = cam_res.confidence
        if camera_view == "uncertain" or cam_res.needs_human:
            needs_human.append("camera_view")
        if cam_res.note != "ok":
            note_parts.append(f"cam_view:{cam_res.note}")

        # ── 4. Swing type ─────────────────────────────────────────────────────
        swing_res = detect_swing_type(kp_json)
        swing_type        = swing_res["swing_type"]
        conf["swing_type"] = swing_res["confidence"]
        if conf["swing_type"] < NEEDS_HUMAN_CONF_THR:
            needs_human.append("swing_type")

        # ── 5. Handedness ─────────────────────────────────────────────────────
        handedness, hand_conf, hand_method = _probe_handedness(kp_json, camera_view=camera_view)
        conf["handedness"] = hand_conf
        if handedness == "unknown" or hand_conf < NEEDS_HUMAN_CONF_THR:
            needs_human.append("handedness")
        note_parts.append(f"handedness_method:{hand_method}")

        # ── 6. Camera profile ─────────────────────────────────────────────────
        cp = compute_camera_profile(
            kp_json,
            video_width=video_width,
            video_height=video_height,
            address_frame=address_frame,
        )
        camera_profile = {
            "subject_center_x":    cp["subject_center_x"],
            "subject_center_y":    cp["subject_center_y"],
            "subject_height_ratio": cp["subject_height_ratio"],
            "camera_height":       cp["camera_height"],
        }
        cp_conf = cp["confidence"]
        if cp_conf < NEEDS_HUMAN_CONF_THR:
            needs_human.append("camera_profile")
        if cp["note"] != "ok":
            note_parts.append(f"cam_profile:{cp['note']}")

        # ── 7. Round confidences ───────────────────────────────────────────────
        conf = {k: round(v, 3) for k, v in conf.items()}

        # ── 8. Assemble notes ──────────────────────────────────────────────────
        # Compact scene note
        view_str  = camera_view.replace("_", "-")
        swing_str = swing_type.replace("_", " ")
        hand_str  = handedness
        lay_str   = layout.replace("_", "-")
        notes = f"{view_str} {lay_str} {swing_str} ({hand_str}-handed)"

        return VideoProfile(
            video_id      = video_id,
            is_golf_swing = is_golf_swing,
            persons       = persons,
            layout        = layout,
            camera_view   = camera_view,
            swing_type    = swing_type,
            handedness    = handedness,
            camera_profile= camera_profile,
            confidence    = conf,
            needs_human   = sorted(set(needs_human)),
            notes         = notes,
        )
