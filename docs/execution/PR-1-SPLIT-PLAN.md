# PR-1-SPLIT-PLAN

## Recommended order

### PR-1A
Schema foundation
- tables
- constraints
- timestamps
- owner-safe baseline

### PR-1B
Upload + processing + deterministic stub
- upload route
- analyze route
- status route
- persistence layer
- minimum protected pages required for flow

### PR-1C
Result page core rendering
- fetch completed result
- render one-one-one output with basic hierarchy

### PR-1D
History page
- list current user's prior results

### PR-1E
Auth hardening if needed beyond first-wave minimum

## Rule
Do not merge later PRs on top of unclear or unsafe schema/auth foundations.
