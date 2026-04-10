# DATA_GUARDRAILS

## Ownership safety
All video, analysis, snapshot, and history reads must be scoped to the authenticated owner.

## Minimum data rules
Store only what is required for MVP flow, auditability, and future extensibility.

## Required explicit fields
- user ownership
- view type
- status
- timestamps for processing start and completion
- failure code/message when needed

## Migration rules
- additive only
- no destructive migration in first wave
- no silent column repurposing
- no enum drift without spec update

## Access rules
- protected pages must require auth
- API routes touching user data must require auth
- queries must not trust client-supplied ownership
- admin paths must not be used from user-facing routes unless strictly necessary and isolated

## Result integrity
Completed result payload must contain exactly one primary issue, one cue, and one drill.
Failure payload must contain explicit failure information.

## Storage conventions
User upload and stub assets must be placed in predictable paths and must support future replacement by real analysis outputs.
