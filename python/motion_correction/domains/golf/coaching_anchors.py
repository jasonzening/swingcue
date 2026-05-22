"""
coaching_anchors.py — derive visual-overlay anchors from corrected
keypoints. Per spec v3 §5 GolfPlugin.coaching_anchor_namespace.

PR-7a Path B (post-Issue-1 from Jason's overlay review): renderer
was showing only midline-collapsed anchors (shoulder_disc_center +
hip_ring_center) and magenta dots clustered at the chest. Updated
to emit **per-side** visuals (left_shoulder_visual, right_shoulder_visual,
left_hip_visual, right_hip_visual, neck_visual) alongside the
midpoint disc centers. The renderer uses different markers for
the two types so each per-side anchor is verifiable independently.

Anchor namespace (7 entries):
  - per-side visuals (5): left_shoulder_visual, right_shoulder_visual,
                          left_hip_visual, right_hip_visual, neck_visual
  - midpoint disc centers (2): shoulder_disc_center, hip_ring_center

PR-7_REVIEW_RESPONSE.md Constraint 3 preserved:
  coaching_anchors_2d remains structurally separated from
  keypoints_2d_projected. Initial impl reads through to the corrected
  projections (so per-side visuals = corrected per-side joints), but
  the SCHEMA SEPARATION lets PR-7.x diverge visuals from analysis
  joints without a schema migration.
"""
from __future__ import annotations

from typing import Optional


# Names this plugin claims to emit. Engine + downstream renderers can
# enumerate this list to validate the dict it receives.
COACHING_ANCHOR_NAMES: tuple[str, ...] = (
    # Per-side visual anchors — each marker sits on the corresponding
    # corrected joint (acromion / hip socket / throat midpoint).
    "left_shoulder_visual",
    "right_shoulder_visual",
    "left_hip_visual",
    "right_hip_visual",
    "neck_visual",
    # Midpoint disc centers — derived from the per-side pairs.
    "shoulder_disc_center",
    "hip_ring_center",
)


def _midpoint(
    a: Optional[list[float]],
    b: Optional[list[float]],
) -> Optional[list[float]]:
    """Midpoint of two 2D pixel points, None-safe."""
    if a is None or b is None or len(a) != 2 or len(b) != 2:
        return None
    return [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]


def derive(
    keypoints_2d_projected: dict[str, Optional[list[float]]],
    keypoints_3d_corrected: dict[str, Optional[list[float]]],
) -> dict[str, Optional[list[float]]]:
    """
    Derive the 7 golf coaching anchors from corrected keypoints.

    Args:
        keypoints_2d_projected: corrected 2D pixel coords for every
                                  joint the plugin cares about.
        keypoints_3d_corrected: same set in 3D (held for future use;
                                  PR-7.x may compute 3D-aware anchors
                                  without changing the plugin contract).

    Returns: {anchor_name: [u, v] | None} for all 7 anchor names.
    """
    ls = keypoints_2d_projected.get("left_shoulder")
    rs = keypoints_2d_projected.get("right_shoulder")
    lh = keypoints_2d_projected.get("left_hip")
    rh = keypoints_2d_projected.get("right_hip")
    neck = keypoints_2d_projected.get("neck")

    return {
        "left_shoulder_visual":   list(ls) if ls else None,
        "right_shoulder_visual":  list(rs) if rs else None,
        "left_hip_visual":        list(lh) if lh else None,
        "right_hip_visual":       list(rh) if rh else None,
        "neck_visual":            list(neck) if neck else None,
        "shoulder_disc_center":   _midpoint(ls, rs),
        "hip_ring_center":        _midpoint(lh, rh),
    }
