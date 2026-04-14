"""
phase_detector.py — 从关键点轨迹中检测挥杆阶段

算法原理（Face-On 视角）：
  - 追踪腕部中点（左右腕平均）的 y 坐标（上下轨迹）
  - y 最小值（= 画面最高点）= 挥杆顶点（Top）
  - Top 之后，腕部 y 下行速度最大处 ≈ Impact
  - 腕部静止区段开头 = Setup
  - Top 和 Impact 之间 = Transition（约 Top 之后 25%）
  - Impact 之后 = Finish（约 Impact 后 60%）
"""

import numpy as np
from typing import List, Optional
from analyzer import KeypointFrame


def _get_wrist_midpoint_y(frame: KeypointFrame) -> Optional[float]:
    """
    Returns average y of left and right wrist.
    Falls back to whichever is available.
    In MediaPipe: y=0 = top of frame, y=1 = bottom.
    So LOWER y value = HIGHER position in frame.
    """
    lw = frame.landmarks.leftWrist
    rw = frame.landmarks.rightWrist
    if lw and rw:
        return (lw.y + rw.y) / 2.0
    elif lw:
        return lw.y
    elif rw:
        return rw.y
    return None


def _smooth(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average."""
    if len(arr) < window:
        return arr
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode='edge')
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode='valid')[:len(arr)]


def detect_phases(frames: List[KeypointFrame], duration: float) -> dict:
    """
    Detect swing phases from keypoint frames.
    Returns a dict with: setupTime, topTime, transitionTime, impactTime, finishTime
    All in seconds.
    """
    MIN_CONF_FRAMES = 3

    # Extract wrist trajectory
    valid_frames = []
    wrist_y_list = []
    times_list = []

    for f in frames:
        y = _get_wrist_midpoint_y(f)
        if y is not None:
            valid_frames.append(f)
            wrist_y_list.append(y)
            times_list.append(f.time)

    if len(valid_frames) < MIN_CONF_FRAMES:
        # Not enough data — return proportional estimates
        return _proportional_phases(duration)

    wrist_y = np.array(wrist_y_list)
    times   = np.array(times_list)

    # Smooth
    if len(wrist_y) >= 5:
        wrist_y_smooth = _smooth(wrist_y, window=min(5, len(wrist_y)))
    else:
        wrist_y_smooth = wrist_y

    # ── STEP 1: Find TOP of swing ──
    # Top = minimum y value (= highest position = wrists at peak of backswing)
    # We search the first 70% of the video (top doesn't happen at the very end)
    search_end = max(1, int(len(wrist_y_smooth) * 0.75))
    top_idx_raw = int(np.argmin(wrist_y_smooth[:search_end]))

    # Don't let top be in the very first 10% of the video
    first_10pct = max(1, int(len(wrist_y_smooth) * 0.10))
    top_idx = max(first_10pct, top_idx_raw)
    top_time = float(times[top_idx])

    # ── STEP 2: Find IMPACT ──
    # After top, wrists move downward (y increases).
    # Impact = maximum positive velocity (fastest downswing) OR
    #          when wrist y returns to near its starting value.
    after_top_y = wrist_y_smooth[top_idx:]
    after_top_t = times[top_idx:]

    impact_time = None

    if len(after_top_y) >= 3:
        # Calculate velocity (dy/dt)
        dy = np.diff(after_top_y)
        dt = np.diff(after_top_t)
        dt = np.where(dt < 1e-6, 1e-6, dt)
        velocity = dy / dt

        # Find peak positive velocity (fastest downswing)
        if len(velocity) > 0:
            max_vel_idx = int(np.argmax(velocity))
            # The impact frame is roughly at max velocity + 1 frame
            impact_raw_idx = top_idx + max_vel_idx + 1
            impact_time = float(times[min(impact_raw_idx, len(times) - 1)])

    # Fallback: impact at 75% of video
    if impact_time is None or impact_time <= top_time:
        impact_time = top_time + (duration - top_time) * 0.55

    # Clamp impact to reasonable range
    impact_time = max(top_time + 0.05, min(duration - 0.15, impact_time))

    # ── STEP 3: Other phases ──
    # Setup: 10% into the video OR first stable wrist position
    setup_time = max(0.0, min(top_time * 0.15, 0.3))

    # Transition: between top and impact (about 25% of that interval)
    transition_time = top_time + (impact_time - top_time) * 0.25

    # Finish: after impact
    finish_time = impact_time + (duration - impact_time) * 0.55
    finish_time = min(finish_time, duration - 0.1)

    # ── Sanity checks ──
    phases = {
        'setupTime':      round(max(0.0, setup_time), 3),
        'topTime':        round(max(setup_time + 0.05, top_time), 3),
        'transitionTime': round(max(top_time + 0.03, transition_time), 3),
        'impactTime':     round(max(transition_time + 0.03, impact_time), 3),
        'finishTime':     round(max(impact_time + 0.05, finish_time), 3),
    }

    return phases


def _proportional_phases(duration: float) -> dict:
    """Proportional fallback when not enough keypoint data."""
    return {
        'setupTime':      round(duration * 0.02, 3),
        'topTime':        round(duration * 0.50, 3),
        'transitionTime': round(duration * 0.62, 3),
        'impactTime':     round(duration * 0.75, 3),
        'finishTime':     round(duration * 0.90, 3),
    }


def score_issue_from_keypoints(frames: List[KeypointFrame]) -> dict:
    """
    Simple rule-based issue scoring from keypoints.
    Returns dict: { issue: str, confidence: float, metrics: dict }

    Rules (all for face-on view, right-handed golfer):
    - head_movement:      nose x-displacement > threshold across swing
    - early_extension:    hip y rises (decreases in MediaPipe) before impact  
    - weight_shift_issue: hip midpoint x doesn't shift toward lead side
    - steep_downswing:    wrist path comes from too far outside (complex — skip for MVP)
    """
    if len(frames) < 4:
        return {'issue': 'early_extension', 'confidence': 0.4, 'metrics': {}}

    metrics: dict = {}
    scores: dict = {}

    # Extract time-indexed data
    nose_x, nose_y = [], []
    hip_y_mid = []
    wrist_x_mid = []
    lhip_x, rhip_x = [], []
    times = []

    for f in frames:
        lm = f.landmarks
        times.append(f.time)

        if lm.head:
            nose_x.append(lm.head.x)
            nose_y.append(lm.head.y)
        else:
            nose_x.append(None)
            nose_y.append(None)

        lhx = lm.leftHip.x if lm.leftHip else None
        rhx = lm.rightHip.x if lm.rightHip else None
        lhy = lm.leftHip.y if lm.leftHip else None
        rhy = lm.rightHip.y if lm.rightHip else None

        lhip_x.append(lhx)
        rhip_x.append(rhx)

        if lhy is not None and rhy is not None:
            hip_y_mid.append((lhy + rhy) / 2)
        else:
            hip_y_mid.append(None)

        lw = lm.leftWrist
        rw = lm.rightWrist
        if lw and rw:
            wrist_x_mid.append((lw.x + rw.x) / 2)
        elif lw:
            wrist_x_mid.append(lw.x)
        elif rw:
            wrist_x_mid.append(rw.x)
        else:
            wrist_x_mid.append(None)

    # ── Head movement ──
    valid_nose_x = [x for x in nose_x if x is not None]
    if len(valid_nose_x) >= 4:
        nose_range = max(valid_nose_x) - min(valid_nose_x)
        metrics['head_x_range'] = round(nose_range, 4)
        # > 0.06 = significant lateral movement (6% of video width)
        scores['head_movement'] = min(1.0, nose_range / 0.08)
    else:
        scores['head_movement'] = 0.3

    # ── Early extension: hip rising before impact ──
    # In MediaPipe, y = 0 at top, y = 1 at bottom.
    # "Rising" = y DECREASING (hips moving UP toward camera level)
    # We check if hip y decreases significantly between mid-swing and impact
    valid_hip_y = [(t, y) for t, y in zip(times, hip_y_mid) if y is not None]
    if len(valid_hip_y) >= 4:
        n = len(valid_hip_y)
        first_half_y = [y for _, y in valid_hip_y[:n // 2]]
        second_half_y = [y for _, y in valid_hip_y[n // 2:]]
        if first_half_y and second_half_y:
            avg_first = sum(first_half_y) / len(first_half_y)
            avg_second = sum(second_half_y) / len(second_half_y)
            # If hips are HIGHER in second half (y decreased) = early extension
            hip_rise = avg_first - avg_second  # positive = hips rose
            metrics['hip_rise'] = round(hip_rise, 4)
            scores['early_extension'] = min(1.0, max(0.0, hip_rise / 0.04))
    else:
        scores['early_extension'] = 0.35

    # ── Weight shift: hip x doesn't move toward target ──
    valid_lhip = [(t, x) for t, x in zip(times, lhip_x) if x is not None]
    valid_rhip = [(t, x) for t, x in zip(times, rhip_x) if x is not None]
    if len(valid_lhip) >= 4 and len(valid_rhip) >= 4:
        # Compute hip center x shift from first half to second half
        n = min(len(valid_lhip), len(valid_rhip))
        hip_center_x = [(lhip_x[i] + rhip_x[i]) / 2
                        for i in range(len(times))
                        if lhip_x[i] is not None and rhip_x[i] is not None]
        if len(hip_center_x) >= 4:
            first_q = hip_center_x[:len(hip_center_x) // 4]
            last_q = hip_center_x[3 * len(hip_center_x) // 4:]
            shift = abs(sum(last_q) / len(last_q) - sum(first_q) / len(first_q))
            metrics['hip_shift'] = round(shift, 4)
            # If shift < 0.03 = very little weight shift
            scores['weight_shift_issue'] = min(1.0, max(0.0, (0.05 - shift) / 0.05))
    else:
        scores['weight_shift_issue'] = 0.25

    # Default scores for issues we can't yet detect from pose alone
    scores.setdefault('steep_downswing',  0.30)
    scores.setdefault('hand_path_issue',  0.25)
    scores.setdefault('head_movement',    scores.get('head_movement', 0.25))

    # ── Priority weighting (per spec: weight certain issues higher) ──
    # These issues are more important for beginners to fix first
    priority = {
        'steep_downswing':  0.90,
        'early_extension':  0.85,
        'weight_shift_issue': 0.80,
        'shoulder_lifts_too_early': 0.75,
        'hand_path_issue':  0.70,
        'head_movement':    0.65,
    }

    # Combined score = confidence * 0.7 + priority * 0.3
    final_scores = {}
    for issue, conf in scores.items():
        p = priority.get(issue, 0.5)
        final_scores[issue] = conf * 0.7 + p * 0.3

    # Pick best issue
    best_issue = max(final_scores, key=lambda k: final_scores[k])
    best_conf  = round(scores.get(best_issue, 0.5), 3)

    return {
        'issue':      best_issue,
        'confidence': best_conf,
        'metrics':    metrics,
        'allScores':  {k: round(v, 3) for k, v in final_scores.items()},
    }
