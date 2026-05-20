"""
benchmark — PR-6.0 pose-runner comparison harness.

Self-contained ad-hoc tooling for comparing alternative pose estimators
against the current production pipeline (MediaPipe Pose 0.10.x,
model_complexity=1, used in python/analyzer.py + python/pose_timeline.py).

NOT imported by any production code. Lives outside the deployed Docker
image. Run locally on Python 3.11. See README.md for setup.
"""
