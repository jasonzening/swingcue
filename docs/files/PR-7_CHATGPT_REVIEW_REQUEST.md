# PR-7 Motion Correction Platform — ChatGPT Review Request

**Date**: 2026-05-21
**Reviewer**: ChatGPT (architectural / strategic gut-check)
**Reviewer's principal**: Jason Zeng, SwingCue founder
**Spec under review**: `docs/files/PR-7_MOTION_CORRECTION_PLATFORM_SPEC_v3.md` (this file's sibling)
**Reviewed-by-author?**: Yes (Jason already approved at higher level; this review focuses on engineering / sequencing gotchas before implementation begins)

---

## 1. Why this review is happening

- **Spec v3 is a platform refactor**, not the original golf-only v2. Architecture decision in v3 = highest-leverage technical call in the SwingCue roadmap. v2 → v3 added a `DomainPlugin` ABC + generic `engine/` + per-sport `domains/` layering, with golf as the first plugin.
- **Ground truth labeling is 15/15 complete** as of commit `05b5038` on `track2/phase2-bone-center-pilot`. 10 face_on + 5 down_the_line samples ready for PR-7b sweep harness.
- **PR-7a implementation is gated** on this architecture sign-off. The bigger v3 abstracts, the more friction PR-7a hits; the goal is to ship golf, not to perfect a platform pattern.
- Jason added **§0 "Golf first / Platform-aware / NOT platform-first"** to spec v3 (commit `fc8b7a5`) as the most senior strategic principle, ranked above all other sections. Any review recommendation must respect this constraint.

---

## 2. What ChatGPT should review

**Primary input**: the full content of `docs/files/PR-7_MOTION_CORRECTION_PLATFORM_SPEC_v3.md`. Jason will paste it alongside this review-request file. The spec is ~850 lines; the key sections for review are §0, §2-§5, §8-§9, §11-§12, §14.

**Reading order** (suggested):
1. §0 — strategic priority (the lock)
2. §1 — what this PR is / is NOT
3. §2 — 3-layer architecture diagram
4. §4 — DomainPlugin abstract interface (the heaviest abstraction; biggest scope-creep risk)
5. §5 — GolfPlugin reference impl (does it actually fit the ABC?)
6. §6 — output schemas (sport/view envelope)
7. §8 — acceptance gates (incl. NEW v3 plugin-conformance gate)
8. §9 — sub-PR split + per-PR acceptance
9. §11 — future plugins roadmap (is it too ambitious to anchor v3 on?)
10. §12 — multi-view support (is 2 views right-sized for 15-sample ground truth?)
11. §14 — effort estimate (+1 day vs v2 — credible?)

---

## 3. Jason's explicit strategic warning (verbatim, do NOT paraphrase)

> Do not let platform abstraction overtake the golf correction main line. PR-7a's goal is still: get golf corrected keypoints running on real swing videos, not building a perfect universal motion platform.

Path: **Golf first → Platform-aware → NOT platform-first.**

This is §0 of spec v3. Every review recommendation should be evaluated against whether it accelerates or delays golf shipping. A recommendation that adds platform purity at the cost of PR-7a wall-clock is automatically suspect.

---

## 4. 10 specific review questions

For each, ChatGPT should produce: **PASS / MODIFY / DROP** + 1-2 sentence rationale + concrete suggested edit if MODIFY.

### Q1 — DomainPlugin ABC: PR-7a or PR-7.x?

Should the `DomainPlugin` abstract class (§4) exist in PR-7a, or defer to PR-7.x when a 2nd plugin (tennis, ski) creates real demand?

Context: §0 explicitly says "if ABC adds friction → strip it, keep concrete classes." But spec §4 mandates it as PR-7a deliverable. Conflict.

### Q2 — Plugin interface heaviness

Is the 8-method ABC (`detect_phases`, `get_offset_config`, `get_smoothing_config`, `compute_coaching_anchors`, `compute_analysis_metrics`, `equipment_detector`, `validate_ground_truth_label`, plus the 3 dataclass classes `PhaseSpec` / `CoachingAnchorSpec` / `AnalysisMetricSpec`) too heavy for what golf alone needs? Would a thinner protocol (e.g., 3-method protocol or even just module-level functions) be more honest in PR-7a?

### Q3 — Multi-view config (face_on / down_the_line) — premature?

`ANATOMICAL_OFFSETS["face_on"]` + `["down_the_line"]` per spec §10 — is this right-sized for the 15-sample ground truth (10 face_on + 5 down_the_line)? Or premature for sweep optimization? Should down_the_line slot ship as a stub with TODO until more DTL videos exist?

### Q4 — Correction layer model-agnostic?

Does the correction engine remain truly model-agnostic — i.e., can WHAM be swapped for Human3R or SMPLest-X in PR-7.x without engine code changes? Are there hidden coupling points (e.g., assuming 24-joint SMPL vs 31-joint custom regressor)?

### Q5 — PR-7a strict no-frontend rule

Spec §9 says PR-7a is Python-module-only, no frontend touched. Is this clear enough, or are there gaps (e.g., does emitting a new schema implicitly require a frontend type update to compile)?

### Q6 — Schema separation: analysis_anchors vs coaching_anchors

The output schema (§6) has separate `keypoints_3d_corrected` (analysis) and `coaching_anchors_2d` (visual overlay). Is this distinction overdesigned for PR-7a (where they'd start identical), or is the separation worth carrying from day 1 to avoid future migration pain? Per §0 operational rule #3: "if no clear benefit in 7c integration → collapse."

### Q7 — Ground truth v3 envelope sufficient?

Per spec §7: `schema_version`, `sport`, `video_id`, `phase`, `frame_idx`, `view`, `video_width`, `video_height`, `labels`, `labeler_version`, `labeled_at`. Is anything missing that PR-7b sweep harness will regret not having (e.g., video FPS, camera intrinsics estimate, labeler confidence, ambiguity notes)?

### Q8 — Acceptance gates calibration

§8 gates: shoulder/hip < 10 px, head/spine < 12 px, wrist/hand diagnostic-only. Too loose, too strict, or right? Specifically: is 10 px ambitious for a 720×1280 portrait at golf body scale, or trivial? Should wrist/hand have ANY numerical floor (not just "diagnostic")?

### Q9 — Migration / schema risk

Anything in the v3 schema or module structure that a future PR (PR-8 tennis, or even PR-7c production integration) will block on or need to re-migrate? E.g., the `coaching_anchors_2d` keys being plugin-namespaced — does the JSONB shape in production DB handle that without per-sport column proliferation?

### Q10 — Sub-PR split: should PR-7a be split further?

Spec §9 makes PR-7a one 4-5 day sub-PR. Should it split into PR-7a.0 (engine skeleton + tests) and PR-7a.5 (golf plugin + offline JSON) to keep each shippable in 1-2 days? Or is the current 4-5 day chunk the right grain?

---

## 5. Top-3 risks ChatGPT would surface that aren't in Q1-Q10

ChatGPT should add up to 3 additional risks that the 10 questions didn't cover — e.g., implementation-language pitfalls, Modal cost model concerns, ground-truth-label statistical sufficiency, frontend type-system implications for PR-7c, etc.

---

## 6. Explicit go/no-go

**Final ChatGPT answer**:
- **Y / N — should PR-7a start as-is on the current spec v3 + §0?**
- If N: which sections of spec v3 need revision before PR-7a starts (cite section numbers)?
- If Y but with caveats: which Q1-Q10 MODIFY recommendations should be applied as inline patches before PR-7a starts vs deferred to mid-implementation?

---

## 7. Output format ChatGPT should produce

```markdown
## Review summary

### Q1 (DomainPlugin ABC)
- Verdict: MODIFY / PASS / DROP
- Rationale: 1-2 sentences
- Suggested edit: <concrete change to spec v3 if MODIFY>

### Q2 ...

### ... through Q10

## Additional risks
1. <Risk 1 not covered by Q1-Q10>
2. ...
3. ...

## Final recommendation
- PR-7a start as-is? Y / N
- Required pre-flight edits (if any): ...
- Deferred edits (apply mid-implementation): ...

## Estimated time impact
- If MODIFY recommendations adopted: PR-7a wall-clock from 4-5 days to N days
- §0 enforcement: spec is now self-policing, no further safeguards needed
```

---

## 8. Files to paste into ChatGPT

When inviting ChatGPT to review, paste in this order:

1. **This file** (`PR-7_CHATGPT_REVIEW_REQUEST.md`) — the review brief
2. **The spec v3 itself** (`docs/files/PR-7_MOTION_CORRECTION_PLATFORM_SPEC_v3.md`) — the artifact under review

Both files are committed locally on `track2/phase2-bone-center-pilot` (not pushed to GitHub). Jason downloads or copies the file content from his local disk, pastes into a fresh ChatGPT session.

---

## 9. Expected review turnaround

Jason expects ChatGPT to return:
- All 10 Q answers (MODIFY most likely on Q1, Q2, Q6 based on prior v2 review pattern)
- 3 additional risks
- Final Y/N for PR-7a start
- Wall-clock impact estimate

Once review returns, Jason forwards to CC for spec revision (call it v3.1 or v3-final), then PR-7a implementation starts.

If ChatGPT recommends 1-3 simplifications consistent with §0 (likely outcome), spec shrinks 20-30%, PR-7a estimate compresses from 4-5 days to 3-4 days.
