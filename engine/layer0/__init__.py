"""
engine/layer0/__init__.py
engine/layer0/perception_gate.py
=================================
Layer 0 Perception Gate.

Every video entering the pipeline must pass this gate first.
run_pipeline.py will refuse to process a video without a PASS record.

Gate criteria (all must hold across >=4 of 5 sampled frames):
  Q1: Real human performing / preparing golf swing (not poster, TV, animation)
  Q2: Exactly 1 person visible
  Q4: Golfer's full body in frame
  Angle consistency: all frames agree on DTL or face-on

Result stored in engine/layer0/records/<video_stem>.json
"""
