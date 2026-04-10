# PR_REVIEW_RUBRIC

## PASS conditions
- matches declared scope
- maps to written spec
- no hidden schema or auth surprises
- manual test steps are reproducible
- no obvious product drift

## BLOCKER conditions
- schema or RLS unsafe
- explicit status model violated
- ownership checks missing
- contract mismatch without justification
- unsupported features introduced

## Review questions
1. Does the code follow the written spec?
2. Is state explicit and durable?
3. Is user ownership enforced server-side?
4. Is the PR narrowly scoped?
5. Did the PR avoid unnecessary abstraction?
