"""
python/pilot/runners/ — per-library 3D body fitter runners.

Each module here defines a Modal-decorated entry function that takes a
swing video + (optional) 2D keypoint timeline and returns a
joint_centers_3d timeline. phase2a only creates this package; the
actual runners land in:

  - phase2b: wham_runner.py
  - phase2c: human3r_runner.py, smplest_x_runner.py,
             easymocap_runner.py, smplify_x_runner.py

Runner contract (see _base.py for the shared schema):
  Input:
    video_path:    str            # local or s3://swing-videos/... URL
    video_id:      str            # for output naming
    keypoints_2d:  dict | None    # optional, libraries that consume 2D
  Output:
    PilotRunResult shape: video_id, runner_name, frames[i] joint dicts,
    smpl params, camera extrinsics, notes[]

All runners write outputs to python/pilot/output/<runner>/<video_id>/
locally after Modal returns, mirroring the python/benchmark/output/
layout from PR-6.0 Phase 1B so comparison rendering can be unified.
"""
