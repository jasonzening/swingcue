"""
Golf domain plugin — first (and only, in PR-7a) plugin per spec v3 §5.

Public surface:
    from motion_correction.domains.golf.plugin import GolfCorrectionPlugin

Sport-specific assets:
  phases.py            — golf swing phase taxonomy
  phase_detector.py    — per-frame phase classification
  config.py            — anatomical offsets + phase smoothing configs
                          (initial estimates; tuned in PR-7b)
  coaching_anchors.py  — derives coaching-overlay anchors from corrected kp
  analysis_metrics.py  — golf-specific derived metrics (hip-shoulder sep, etc.)
"""
