"""
anatomical_offset.py — apply per-joint anatomical offset.

Generic engine logic; the per-joint configurations are sport- and
view-specific and come from the plugin (via view_aware.select_offset_config).

Two offset modes supported (selected per-entry by value type in the
plugin's offset_config dict):

  A. **Per-joint body-local 3D vector** (primary; PR-7a Path B). Config
     value is a list/tuple of length 3: [d_h, d_v, d_f] in body-local
     metres, where:
         d_h = component along horizontal axis (body left-right)
         d_v = component along spine_up axis  (body up-down)
         d_f = component along body_forward axis (body anterior-posterior)
     Engine builds the body-local basis from current frame's pelvis +
     neck, transforms the stored vector to camera frame, then applies:
         corrected = raw + camera_offset
     This makes offsets ROTATION-INVARIANT: as the golfer rotates
     through the swing (setup → top → impact → finish), the offset
     rotates with the body, staying anatomically correct. Pre-Path-B
     storage was camera-frame; that worked at setup but mis-aligned at
     finish/impact (Issue 2 from Jason's overlay review).

     For hip-class joints (per `HORIZONTAL_ONLY_OFFSET_KEYS`), d_v is
     zeroed before transform — SMPL hip bone-center error is purely
     lateral in body-local frame too (Finding G constraint).

  B. **Scalar inward-pull** (legacy fallback, joints without GT labels).
     Config value is a single float in the group-keyed dict using
     `KEYPOINT_TO_OFFSET_KEY` dispatch. Engine applies:
         corrected = raw + coef * (body_center - raw)         # full-3D
         corrected = raw + coef * along_h * horizontal_basis  # horiz-only
     Used for knee/ankle/wrist — the GT corpus has no labels for these,
     so the body-local-vector fit is unavailable. Keeps conservative
     starting estimates from spec v3 §10.

Per-entry dispatch: if `offset_config[joint_name]` exists AND is a
list of 3 floats → mode A. Otherwise fall back to
`offset_config[KEYPOINT_TO_OFFSET_KEY[joint_name]]` as a scalar → mode B.
Joints with no entry in either path pass through unchanged.
"""
from __future__ import annotations

import math
from typing import Optional


def offset_toward_center(
    keypoint_3d: Optional[list[float]],
    body_center_3d: list[float],
    coefficient: float,
) -> Optional[list[float]]:
    """
    Push one 3D keypoint toward body center by `coefficient` fraction.

    Args:
        keypoint_3d:    [x, y, z] in meters (camera or world frame, the
                         caller picks; offset is applied in whatever
                         frame both points share).
        body_center_3d: same frame as keypoint_3d.
        coefficient:    0.0 = no change, 1.0 = pulled all the way to
                         center. Typically 0.0-0.25.

    Returns: corrected [x, y, z] or None if input was None.
    """
    if keypoint_3d is None:
        return None
    if len(keypoint_3d) != 3 or len(body_center_3d) != 3:
        return None
    return [
        keypoint_3d[0] + coefficient * (body_center_3d[0] - keypoint_3d[0]),
        keypoint_3d[1] + coefficient * (body_center_3d[1] - keypoint_3d[1]),
        keypoint_3d[2] + coefficient * (body_center_3d[2] - keypoint_3d[2]),
    ]


def compute_torso_center(
    left_shoulder: Optional[list[float]],
    right_shoulder: Optional[list[float]],
    left_hip: Optional[list[float]],
    right_hip: Optional[list[float]],
) -> Optional[list[float]]:
    """
    Body center for offset reference = mean of the 4 torso corners
    (shoulders + hips). Robust against single missing joint — falls
    back to whatever subset is present, returns None only if all four
    are missing.
    """
    points = [p for p in (left_shoulder, right_shoulder, left_hip, right_hip)
              if p is not None and len(p) == 3 and None not in p]
    if not points:
        return None
    n = len(points)
    return [sum(p[i] for p in points) / n for i in range(3)]


# Map from raw-keypoint-name → which inward-offset coefficient key to
# pull from the plugin's offset config. Sport-agnostic naming because
# WHAM emits H36M-ordered joints with standardised names; the plugin's
# config dict keys are also generic (`shoulder_inward`, `hip_inward`,
# etc.) so this mapping holds for any sport whose plugin follows the
# same naming convention.
KEYPOINT_TO_OFFSET_KEY: dict[str, str] = {
    "left_shoulder":  "shoulder_inward",
    "right_shoulder": "shoulder_inward",
    "left_hip":       "hip_inward",
    "right_hip":      "hip_inward",
    "head":           "head_inward",
    "neck":           "head_inward",
    "left_knee":      "knee_inward",
    "right_knee":     "knee_inward",
    "left_ankle":     "ankle_inward",
    "right_ankle":    "ankle_inward",
    "left_wrist":     "wrist_inward",
    "right_wrist":    "wrist_inward",
    # spine + pelvis don't get offset (they ARE the centerline).
}

# Offset keys whose pull is constrained to the body-local horizontal
# axis only (no vertical movement). Per Finding G — SMPL hip bone-
# center error is laterally outboard; vertical pull toward torso
# centroid is spurious.
HORIZONTAL_ONLY_OFFSET_KEYS: frozenset[str] = frozenset({"hip_inward"})


def horizontal_basis_unit(
    pelvis: Optional[list[float]],
    neck: Optional[list[float]],
) -> Optional[list[float]]:
    """
    Compute the body-local horizontal (left-right) unit vector in
    camera frame.

        spine_unit  = unit(neck - pelvis)
        cam_z       = (0, 0, 1)
        horizontal  = unit(cross(spine_unit, cam_z))

    Returns None when spine direction is degenerate (zero length) or
    when spine happens to be parallel to camera z-axis (cross = 0).
    """
    if pelvis is None or neck is None or len(pelvis) != 3 or len(neck) != 3:
        return None
    sx, sy, sz = neck[0] - pelvis[0], neck[1] - pelvis[1], neck[2] - pelvis[2]
    mag = math.sqrt(sx * sx + sy * sy + sz * sz)
    if mag < 1e-6:
        return None
    sx, sy, sz = sx / mag, sy / mag, sz / mag
    # cross( (sx, sy, sz), (0, 0, 1) ) = ( sy, -sx, 0 )
    hx, hy, hz = sy, -sx, 0.0
    hmag = math.sqrt(hx * hx + hy * hy + hz * hz)
    if hmag < 1e-6:
        return None
    return [hx / hmag, hy / hmag, hz / hmag]


def body_local_basis(
    pelvis: Optional[list[float]],
    neck: Optional[list[float]],
) -> Optional[tuple[list[float], list[float], list[float]]]:
    """
    Build a body-local orthonormal basis from pelvis + neck (camera-frame).

    Returns: (horizontal, spine_up, body_forward) where each is a 3D
    unit vector in camera frame. Right-handed: horizontal × spine_up =
    body_forward.

        spine_up     = unit(neck - pelvis)                     # body up
        horizontal   = unit(cross(spine_up, camera_z))         # body left
        body_forward = unit(cross(horizontal, spine_up))       # body fwd

    The fitted ANATOMICAL_OFFSETS vectors are stored in this basis as
    [d_h, d_v, d_f]. Per-frame, the engine transforms back to camera:

        camera_offset = d_h * horizontal + d_v * spine_up + d_f * body_forward

    This makes offsets ROTATION-INVARIANT: as the golfer's body rotates
    during the swing (setup → top → impact → finish), the offset rotates
    with the body. A camera-frame-stored offset would mis-align after
    rotation; body-local-stored offset stays anatomically correct.

    Returns None if spine direction is degenerate (pelvis ≈ neck) or
    spine is parallel to camera z (cross product collapses).
    """
    if pelvis is None or neck is None or len(pelvis) != 3 or len(neck) != 3:
        return None
    sx, sy, sz = neck[0] - pelvis[0], neck[1] - pelvis[1], neck[2] - pelvis[2]
    mag = math.sqrt(sx * sx + sy * sy + sz * sz)
    if mag < 1e-6:
        return None
    spine_up = [sx / mag, sy / mag, sz / mag]
    # horizontal = unit(cross(spine_up, cam_z=(0,0,1))) = unit((sy, -sx, 0))
    hx, hy = spine_up[1], -spine_up[0]
    hmag = math.sqrt(hx * hx + hy * hy)
    if hmag < 1e-6:
        return None
    horizontal = [hx / hmag, hy / hmag, 0.0]
    # body_forward = cross(horizontal, spine_up). Already unit since
    # horizontal ⊥ spine_up and both are unit-length.
    fx_ = horizontal[1] * spine_up[2] - horizontal[2] * spine_up[1]
    fy_ = horizontal[2] * spine_up[0] - horizontal[0] * spine_up[2]
    fz_ = horizontal[0] * spine_up[1] - horizontal[1] * spine_up[0]
    fmag = math.sqrt(fx_ * fx_ + fy_ * fy_ + fz_ * fz_)
    if fmag < 1e-6:
        return None
    body_forward = [fx_ / fmag, fy_ / fmag, fz_ / fmag]
    return horizontal, spine_up, body_forward


def body_local_to_camera(
    body_local_vec: list[float],
    basis: tuple[list[float], list[float], list[float]],
) -> list[float]:
    """
    Express a body-local 3-vector [d_h, d_v, d_f] in camera frame.

        camera = d_h * horizontal + d_v * spine_up + d_f * body_forward
    """
    h, s, f = basis
    d_h, d_v, d_f = body_local_vec[0], body_local_vec[1], body_local_vec[2]
    return [
        d_h * h[0] + d_v * s[0] + d_f * f[0],
        d_h * h[1] + d_v * s[1] + d_f * f[1],
        d_h * h[2] + d_v * s[2] + d_f * f[2],
    ]


def camera_to_body_local(
    camera_vec: list[float],
    basis: tuple[list[float], list[float], list[float]],
) -> list[float]:
    """
    Project a camera-frame 3-vector onto the body-local basis.
    Returns [d_h, d_v, d_f]. Inverse of body_local_to_camera since
    the basis is orthonormal.
    """
    h, s, f = basis
    return [
        camera_vec[0] * h[0] + camera_vec[1] * h[1] + camera_vec[2] * h[2],
        camera_vec[0] * s[0] + camera_vec[1] * s[1] + camera_vec[2] * s[2],
        camera_vec[0] * f[0] + camera_vec[1] * f[1] + camera_vec[2] * f[2],
    ]


def offset_along_axis(
    keypoint_3d: Optional[list[float]],
    body_center_3d: list[float],
    axis_unit: list[float],
    coefficient: float,
) -> Optional[list[float]]:
    """
    Project the (center - keypoint) vector onto `axis_unit`, then
    apply `coefficient * projection` along the same axis only.

    Used for hip-class joints whose bone-center error is purely along
    the body-local horizontal axis (Finding G).
    """
    if keypoint_3d is None or len(keypoint_3d) != 3:
        return None
    if len(body_center_3d) != 3 or len(axis_unit) != 3:
        return None
    to_center = [body_center_3d[i] - keypoint_3d[i] for i in range(3)]
    along = sum(to_center[i] * axis_unit[i] for i in range(3))
    return [
        keypoint_3d[i] + coefficient * along * axis_unit[i]
        for i in range(3)
    ]


def _is_vector_3d(v) -> bool:
    """True iff v is a list/tuple of exactly 3 numbers."""
    return (
        isinstance(v, (list, tuple))
        and len(v) == 3
        and all(isinstance(x, (int, float)) for x in v)
    )


def _resolve_vector_for_phase(
    config_value, phase: Optional[str],
) -> Optional[list[float]]:
    """
    PR-7a.1 Fix 3: handle three config-value shapes for mode A:
      - list[3]         → constant vector across all phases (legacy / backwards compat)
      - dict[phase→list[3]] → per-phase vector; look up by phase.
      - anything else   → not a vector (caller falls back to mode B / passthrough).
    Returns the 3-vector to apply, or None if no per-phase entry exists.
    """
    if _is_vector_3d(config_value):
        return [float(c) for c in config_value]
    if isinstance(config_value, dict) and phase is not None:
        v = config_value.get(phase)
        if _is_vector_3d(v):
            return [float(c) for c in v]
    return None


def apply_offset_to_frame(
    keypoints_3d: dict[str, Optional[list[float]]],
    offset_config: dict,
    *,
    basis_override: Optional[tuple[list[float], list[float], list[float]]] = None,
    phase: Optional[str] = None,
) -> dict[str, Optional[list[float]]]:
    """
    Apply per-joint anatomical offset to every keypoint in a frame.

    Per-entry mode selection (see module docstring):
      - If offset_config[joint_name] is a 3-vector → MODE A (per-joint
        3D vector, added directly to raw; y-zeroed for hip-class).
      - Else if KEYPOINT_TO_OFFSET_KEY[joint_name] is in offset_config
        as a scalar → MODE B (legacy scalar inward-pull).
      - Else pass through unchanged.

    Args:
        keypoints_3d:   raw keypoint dict (joint_name → [x, y, z] or None).
        offset_config:  per-view dict from view_aware.select_offset_config.
                          May mix per-joint 3-vectors and group-keyed scalars.
        basis_override: if supplied (3-tuple of horizontal, spine_up,
                          body_forward unit vectors in camera frame),
                          use this basis for the mode-A body-local→camera
                          transform instead of computing from current
                          frame's pelvis+neck. PR-7a.1 Fix 1 setup-lock:
                          orchestrator passes the median-window basis
                          during setup-phase frames to eliminate
                          per-frame basis jitter.

    Returns: new dict with corrected positions.
    """
    center = compute_torso_center(
        keypoints_3d.get("left_shoulder"),
        keypoints_3d.get("right_shoulder"),
        keypoints_3d.get("left_hip"),
        keypoints_3d.get("right_hip"),
    )
    h_axis = horizontal_basis_unit(
        keypoints_3d.get("pelvis"), keypoints_3d.get("neck"),
    )
    # Body-local basis for mode A vector transform (PR-7a Path B).
    # PR-7a.1 Fix 1: caller may override with a locked basis to freeze
    # the transform during setup phase.
    if basis_override is not None:
        basis = basis_override
    else:
        basis = body_local_basis(
            keypoints_3d.get("pelvis"), keypoints_3d.get("neck"),
        )

    out: dict[str, Optional[list[float]]] = {}
    for name, raw in keypoints_3d.items():
        if raw is None:
            out[name] = None
            continue

        # MODE A: per-joint body-local 3D vector → transform to camera.
        # PR-7a.1 Fix 3: config value may be either a flat list[3]
        # (constant across phases) or dict[phase → list[3]] (per-phase).
        direct_raw = offset_config.get(name)
        resolved = _resolve_vector_for_phase(direct_raw, phase)
        if resolved is not None:
            d_h, d_v, d_f = resolved[0], resolved[1], resolved[2]
            # Hip-class: zero d_v in body-local (Finding G preserved
            # in body-local frame — SMPL hip error is purely lateral
            # whether expressed in camera or body coords).
            offset_key = KEYPOINT_TO_OFFSET_KEY.get(name)
            if offset_key in HORIZONTAL_ONLY_OFFSET_KEYS:
                d_v = 0.0
            if basis is None:
                # Spine basis indeterminate this frame → can't transform.
                # Conservative: pass raw through unchanged. Caller can
                # see basis-missing rate via per-frame diagnostics.
                out[name] = list(raw)
                continue
            cam_offset = body_local_to_camera([d_h, d_v, d_f], basis)
            out[name] = [
                raw[0] + cam_offset[0],
                raw[1] + cam_offset[1],
                raw[2] + cam_offset[2],
            ]
            continue

        # MODE B: legacy scalar via group key.
        offset_key = KEYPOINT_TO_OFFSET_KEY.get(name)
        if offset_key is None or offset_key not in offset_config:
            out[name] = list(raw)
            continue
        coef = offset_config[offset_key]
        if not isinstance(coef, (int, float)):
            # Group-keyed slot but value isn't a scalar — defensive
            # passthrough (caller misconfigured).
            out[name] = list(raw)
            continue
        if center is None:
            # Can't compute scalar inward direction without a center.
            out[name] = list(raw)
            continue
        if offset_key in HORIZONTAL_ONLY_OFFSET_KEYS and h_axis is not None:
            out[name] = offset_along_axis(raw, center, h_axis, coef)
        else:
            out[name] = offset_toward_center(raw, center, coef)
    return out
