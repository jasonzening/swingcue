# DEPLOY_RUNBOOK

## Current policy
Preview deploys are allowed after PR creation.
Production deploy is not automatic.

## Production gate
Before production:
- PR approved by audit
- manual review of env vars
- no known auth/RLS blockers
- no known migration safety blockers

## First-wave note
The first wave is for repo foundation and auditable product flow, not for production hardening.
