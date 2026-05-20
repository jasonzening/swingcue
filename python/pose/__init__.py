"""
python/pose/ — shared pose-extractor utilities.

PR-6.1a introduces this package as the home for runner-specific
extractors that produce the COCO 17 per-frame dict shape consumed by
pose_timeline.py. The legacy MediaPipe extractor still lives in
pose_timeline.extract_coco_subset_from_mediapipe; new runners (rtmpose,
future) ship in this directory.

Selection between runners is controlled by the POSE_RUNNER_OVERRIDE env
variable, read by analyzer.py. See PR-6.1_SPEC_v2.md §1, §13.
"""
