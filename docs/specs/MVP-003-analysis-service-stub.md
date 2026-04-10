# MVP-003 Analysis Service Stub

## Goal
Use a deterministic stub to simulate analysis before real CV integration.

## Inputs
- video_id
- video_storage_path
- view_type

## Outputs
See `docs/contracts/analysis_stub_result_contract.json`.

## Rules
- completed payload must include exactly one issue, one cue, one drill
- failed payload must include explicit error details
- supported issue types:
  - weight_shift_issue
  - head_movement
  - early_extension
  - steep_backswing_plane
  - steep_downswing
  - hand_path_issue
- supported phase names:
  - address
  - top
  - downswing
  - impact
  - finish

## Deterministic behavior
The stub may choose result patterns by stable deterministic logic such as view_type or stable hashing from video_id, but the output must be reproducible and contract-compliant.
