"""
motion_correction.engine — sport-agnostic correction primitives.

Plugin pattern conformance (spec v3 §8 acceptance gate): this package
MUST NOT reference any sport-specific identifier, neither as string
literals nor as imports from motion_correction.domains.*. The audit
recipe lives in docs/files/PR-7a_PLUGIN_CONFORMANCE.md.

The engine consumes plugin-provided config + per-frame phase labels
and applies generic correction logic. Sport identity stays in the
caller (orchestrator binds engine + plugin together).
"""
