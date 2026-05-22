# PR-7a Plugin Pattern Conformance Audit

Per spec v3 §8 conformance gate: `engine/` modules MUST NOT reference
any sport-specific identifier. This file documents the audit recipe
and current status.

## Audit recipe

Run from repo root, excluding compiled bytecode (`__pycache__/`):

```bash
grep -rni \
    --exclude-dir=__pycache__ \
    "golf\|GolfCorrection\|GolfPlugin\|tennis\|ski " \
    python/motion_correction/engine/
```

**Pass condition**: 0 lines returned.

If anything matches, treat it as a conformance violation:
- Sport-specific *logic* in engine code → STRIP IMMEDIATELY (hard block per spec v3 §8).
- Sport name in docstring example → revise to generic ("first plugin", "subject", "plugin-flagged", etc.).

## Why it matters

Per spec v3 §0 ("Golf first / Platform-aware / NOT platform-first") +
§4 (DomainPlugin contract): the engine's value is in being a clean
substrate the second plugin (tennis, ski, PT) can land on without
forking it. Every sport-specific reference in engine creates a future
fork-point.

## When the audit is NOT enforced

- `domains/golf/` — by design contains "golf" everywhere.
- `schemas/` — generic but may reference "golf" in example JSON; not
  sport-specific by code logic.
- `tests/engine/` — test fixtures often need a concrete plugin to call
  the engine with; "golf" appearances in test setup are OK.

## Last audit (PR-7a Task 2B)

Date: 2026-05-21
Result: PASS (after docstring sweep)

The engine modules' docstrings originally included "golf" as the
concrete example for several pattern-explainer sections. Per the
strict reading of the conformance gate, those were rewritten to
generic phrasing ("first plugin", "subject", "plugin-flagged"). Code
logic was unaffected — no sport-specific behavior had leaked into
engine modules.
