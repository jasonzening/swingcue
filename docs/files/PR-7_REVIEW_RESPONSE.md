# PR-7 Spec v3 — ChatGPT Review Response

**Date**: 2026-05-21
**Reviewer**: ChatGPT (forwarded by Jason)
**Verdict**: **Y, with pre-flight constraints. Proceed to PR-7a after applying these 7 constraints.**

The spec v3 architectural direction is APPROVED. §0 "Golf first / Platform-aware / NOT platform-first" remains the controlling principle. The 7 constraints below MODIFY the implementation scope of spec v3 §4, §5, §9, §10. Where this document and spec v3 conflict, this document wins.

## Constraint 1 — DomainPlugin ABC must not delay PR-7a

Spec v3 §4 describes DomainPlugin ABC + registry pattern. **For PR-7a, scope down to:**
- KEEP: ABC interface definition + concrete `GolfCorrectionPlugin` class
- DO NOT IMPLEMENT in PR-7a:
  - Plugin discovery (no automatic scanning of domains/)
  - Plugin registry (no `register_plugin()` call or `get_plugin()` lookup)
  - Dynamic loading (no runtime sport selection)
  - Multi-plugin runtime resolution

If during PR-7a CC finds ABC adds friction (subclass boilerplate, type errors, abstract method mismatch), **strip ABC entirely** — keep `GolfCorrectionPlugin` as standalone concrete class. Re-introduce ABC in PR-7.x when tennis or another second plugin forces it.

## Constraint 2 — Multi-view config: face_on primary, down_the_line conservative

Spec v3 §10 `ANATOMICAL_OFFSETS["face_on"]` + `["down_the_line"]` structure stays. **But:**
- face_on = primary calibrated path (10 ground truth samples, sweep-tunable)
- down_the_line = supported with conservative defaults (5 samples, sweep-confirmable but not full optimization)
- **Fallback rule**: if 5 DTL samples cannot converge on a confident offset within sweep range, DTL falls back to face_on offset values + a 1.05-1.10x conservative multiplier on shoulder/hip. Document the fallback decision in PR-7b tuning report.
- If DTL handling slows down PR-7a substantially, defer DTL tuning to PR-7.0.5 follow-up. face_on shipping alone is acceptable MVP.

## Constraint 3 — Keep analysis_anchors and coaching_anchors separated (DO NOT collapse)

Spec v3 §6 schema has BOTH:
- `keypoints_3d_corrected` — for ANALYSIS (angle calcs, biomechanics, phase detection)
- `coaching_anchors_2d` — for VISUAL overlay (disc rendering, skeleton draw)

ChatGPT explicitly INSISTS this separation. Reason: future evolution will hit tension between "anchors visually look right for overlay" vs "anchors mathematically precise for analysis." Without schema separation, one will pollute the other.

PR-7a: output BOTH. Initial implementation: coaching_anchors_2d = keypoints_2d_projected (identical). Divergence permitted in PR-7.x without schema migration.

## Constraint 4 — PR-7a scope (small, shippable in 3-5 days)

PR-7a DELIVERS:
- python/motion_correction/ skeleton (engine/ + domains/golf/)
- GolfCorrectionPlugin concrete implementation (no ABC infrastructure)
- Setup baseline detection (engine + golf override)
- Basic correction pipeline runs offline (raw WHAM JSON in → corrected JSON out)
- Unit tests + smoke tests pass
- Writes corrected timeline JSON to docs/PR-7a_OFFLINE_OUTPUT/<video>_corrected.json for Jason inspection

PR-7a DOES NOT TOUCH:
- ❌ Frontend (src/components/*.tsx UNTOUCHED, src/lib/*.ts UNTOUCHED)
- ❌ Production cutover (no DB schema writes, no production analyze pipeline changes)
- ❌ Modal / Railway deployment changes (unless absolutely unavoidable for offline pipeline)
- ❌ Plugin registry / discovery / dynamic loading
- ❌ Tuning sweep — that's PR-7b
- ❌ Multi-plugin runtime selection
- ❌ Tennis / ski / other domain plugins (those are PR-8+)

## Constraint 5 — Acceptance gates (relaxed + phase-level visual)

Spec v3 §5 numerical gates stay:
- shoulder/hip mean px error < 10 px ✓
- head/spine mean px error < 12 px ✓
- wrist/hand diagnostic only ✓

ADD phase-level visual acceptance (NEW from ChatGPT review §5):
- **Setup**: corrected anchors visibly closer to Jason red-dot labels than raw WHAM output (visual diff check, no numeric threshold)
- **Transition**: no obvious left/right swap across the cross-body motion
- **Impact**: no smoothing-induced motion lag (corrected anchors don't trail the visible body by >2 frames)
- **Finish**: occluded far side anchor remains stable (not collapsing to visible body side)

Phase-level acceptance is JASON VISUAL APPROVAL, not automated. Run after PR-7a offline output is generated; before PR-7b sweep tuning.

## Constraint 6 — Smoothing MUST be phase-aware (uniform alpha REJECTED)

Spec v3 §10 PHASE_CONFIG locks this:
```python
PHASE_CONFIG = {
    "setup":      {"alpha": 0.20, "outlier_ratio": 0.15},  # heavy smoothing
    "backswing":  {"alpha": 0.30, "outlier_ratio": 0.25},  # medium
    "top":        {"alpha": 0.30, "outlier_ratio": 0.35},  # medium
    "transition": {"alpha": 0.30, "outlier_ratio": 0.35},  # medium
    "downswing":  {"alpha": 0.40, "outlier_ratio": 0.40},  # light (fast motion)
    "impact":     {"alpha": 0.40, "outlier_ratio": 0.40},  # light (fast motion)
    "finish":     {"alpha": 0.30, "outlier_ratio": 0.30},  # medium
}
```

Code review HARD BLOCKS any implementation using a single global alpha across phases. Specifically: temporal_smoother.py must accept a `phase` argument per frame and look up the config — no `smooth_keypoint(raw, prev, alpha=0.3)` calls with a fixed alpha.

## Constraint 7 — 15 ground truth samples are sufficient for PR-7b

face_on 10 + down_the_line 5 = 15 samples is the LOCKED ground truth count for PR-7b sweep.

**DO NOT request more labels from Jason before PR-7a starts or during PR-7a.** If sweep convergence is poor, that's PR-7.1 follow-up scope (Jason adds more videos when convenient).

Rationale: 15 samples meets MVP minimum + first iteration value comes from validated end-to-end pipeline, not from infinite ground truth refinement.

---

## Authority order (when documents conflict)

1. spec v3 §0 (Golf first / Platform-aware / NOT platform-first) — ABSOLUTE
2. This review response (7 constraints) — MODIFIES spec v3 implementation scope
3. spec v3 §1-§17 — DEFAULTS otherwise

## Sign-off

ChatGPT verdict: **Proceed to PR-7a after committing this document.**
Jason verdict: **Approved per session record.**
Awaiting CC commit of this file → start PR-7a immediately after.
