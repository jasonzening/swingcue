# ARCHITECTURE_PRINCIPLES

## Source of truth
Repo documents are the source of truth for scope, behavior, and review standards.

## Execution roles
- ChatGPT defines architecture boundaries, specs, audit standards, and release gates.
- Claude implements code, tests, PRs, and preview deploy preparation.
- Claude must not redefine product philosophy or architecture scope.

## Architecture principles
1. Prefer explicit state over inferred state.
2. Prefer additive change over destructive change.
3. Prefer boring, auditable implementation over premature sophistication.
4. Prefer clear contracts over implicit assumptions.
5. Prefer ownership-safe data access everywhere.
6. Prefer small PRs with clear review boundaries.
7. Prefer deterministic first-wave behavior over speculative AI complexity.

## First-wave technical direction
- Next.js App Router
- TypeScript
- Tailwind
- Supabase Auth
- Supabase Postgres
- Supabase Storage
- deterministic analysis stub first
- real CV / pose analysis later

## Forbidden first-wave architecture drift
Do not add in the first wave:
- MediaPipe or real CV integration
- queues or worker infrastructure unless explicitly approved later
- production-dangerous automation
- analytics dashboards
- comparison workspaces
- social systems

## PR discipline
Every PR must:
- state scope clearly
- state what was intentionally not done
- include manual test steps
- avoid hidden side effects
- map to a written spec or runbook

## State model
Allowed swing video status values:
- uploaded
- processing
- completed
- failed

Status must never be inferred from nullability of related records or fields.
