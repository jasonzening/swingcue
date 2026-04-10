# FIRST_PR_AUDIT_EXAMPLE

## Verdict
BLOCKED

## Example blockers
- analyze route infers completed from related row presence instead of explicit status
- result API does not clearly enforce owner-scoped access

## Example fix instruction
Update the flow so swing_videos.status is the primary durable source of status truth. Add explicit owner check in result query path and include test steps proving unauthorized access fails.
