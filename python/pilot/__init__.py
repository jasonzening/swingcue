"""
python/pilot/ — Phase 2 bone-center 3D body-fitting pilot.

Pure Modal-GPU exploration directory. NEVER imported by production
analyzer or main.py. NEVER touched by PR-6.1 Track 1.

Scope (per docs/files/PHASE_2_BONE_CENTER_PILOT_SPEC_v2.md):
  - phase2a: Modal app skeleton + Volume + per-library Image definitions
  - phase2b: WHAM-first smoke (real inference on b3fea3f0)
  - phase2c: Expand to Human3R / SMPLest-X / EasyMocap / SMPLify-X

Anti-pattern guard (Verdict v2 §9): production code (python/analyzer.py,
python/pose_timeline.py, python/main.py, src/*) MUST remain untouched
through all of phase2a/b/c. The pilot's winning library moves into
production via a separate PR-7 spec after the pilot identifies one.

Strategic principle (Verdict v2 §9): SwingCue final target = anatomical
bone-center keypoints (femoral head, humeral head, talus center, etc.),
not RGB-image surface landmarks. This directory pilots the libraries
that promise to deliver that target.
"""
