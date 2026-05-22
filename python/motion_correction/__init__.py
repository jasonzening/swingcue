"""
motion_correction — generic motion correction engine + per-sport
domain plugins. Layer 2 + Layer 3 of the PR-7 architecture (spec v3
§2).

Strategic principle (spec v3 §0):
  Golf first. Platform-aware. NOT platform-first.

The package structure (engine/ + domains/ + schemas/) exists ONLY
because it imposes cleaner code organization at zero extra cost.
The architecture is in service of golf shipping, not the reverse.

PR-7a scope (per docs/files/PR-7_REVIEW_RESPONSE.md):
  - engine/  : sport-agnostic correction primitives
  - domains/golf/ : first plugin (concrete class, no ABC infrastructure)
  - schemas/ : dataclass shapes for corrected timeline + ground truth

Excluded from PR-7a:
  - plugin discovery / registry / dynamic loading (C1)
  - DomainPlugin ABC base class (C1 — strip-if-friction)
  - tuning sweep (PR-7b)
  - production cutover (PR-7c)
  - frontend (PR-7c)
  - tennis / ski / other plugins (PR-8+)
"""
