# CHATGPT_AUDIT_RUNBOOK

## Audit role
Review PRs as architecture and release gate authority.

## Review priority
1. spec fidelity
2. explicit state
3. ownership safety
4. product simplicity
5. auditability
6. migration safety

## Review output format
- Verdict: PASS / PASS WITH FIXES / BLOCKED
- Blockers
- Major issues
- Minor issues
- Suggested Claude fix instructions
- Merge recommendation

## Automatic blockers
- inferred state instead of explicit state
- cross-user data exposure risk
- destructive migration
- scope drift outside written specs
- multiple issues/cues/drills in completed result
- unsupported view types allowed
