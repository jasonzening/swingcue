# CLAUDE_EXECUTION_RUNBOOK

## Role
You are the execution engineer for this repo.

## Required behavior
- Read repo docs before coding.
- Treat repo docs as binding.
- Do not expand scope on your own.
- Do not redefine architecture.
- Do not ask the user to restate repo specs already present in the repo.

## Mandatory reading order
1. docs/constitution/PRODUCT_CONSTITUTION.md
2. docs/constitution/ARCHITECTURE_PRINCIPLES.md
3. docs/constitution/DATA_GUARDRAILS.md
4. docs/execution/PR-1-SPLIT-PLAN.md
5. docs/execution/CLAUDE_FIRST_WAVE_TASK_BREAKDOWN.md
6. docs/specs/MVP-006-supabase-schema.md
7. docs/specs/MVP-001-upload-and-processing-flow.md
8. docs/specs/MVP-003-analysis-service-stub.md
9. docs/specs/MVP-005-auth-and-protected-routes.md
10. docs/contracts/analysis_stub_result_contract.json
11. docs/frontend/NEXTJS_APP_ROUTER_STRUCTURE.md

## First wave
Implement:
- schema foundation
- upload flow
- explicit processing state
- deterministic analysis stub
- persistence of completed/failed outcomes
- minimum protected-route hardening needed for safe flow

Do not build:
- polished result-page visuals beyond testability
- full history feature
- real CV integration
- queue systems

## PR output requirements
At the end of the task, provide:
- branch name
- changed files
- intentionally deferred items
- manual test steps
- known risks
- any spec ambiguities
