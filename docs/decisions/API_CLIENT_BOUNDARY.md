# API Client Boundary Decision

**Status:** Approved
**Approved by:** ChatGPT (SwingCue Architecture Review)
**Date:** 2026-04-13
**Applies to:** All PRs from PR-1B onward

## Decision

For SwingCue MVP, writes to `swing_analysis` and `swing_phase_snapshots` may be performed through `lib/supabase/admin.ts` using the service-role client, because these writes are backend-controlled execution steps rather than user-direct client writes.

## Implications

- RLS remains critical for user-scoped reads
- RLS is not the primary enforcement mechanism for backend-controlled writes using service-role
- Write safety must be enforced in the API layer through:
  - Authenticated user verification
  - Ownership verification of the target `swing_videos` record
  - Strict parameter validation
  - Explicit route-level control over when analysis rows are created

## Rules

- User-facing reads should prefer user-scoped server client + RLS
- Backend-controlled writes may use service-role **only in server-safe contexts**
- Service-role must never leak into client-accessible code
- No PR may broaden this boundary without explicit review

## Implementation

```
lib/supabase/server.ts   → user-scoped server client (respects RLS)
lib/supabase/admin.ts    → service-role client (bypasses RLS, server-only)
lib/supabase/client.ts   → browser client (user-scoped, public routes only)
```

## Verification Checklist (for each PR touching analysis writes)

- [ ] API route calls requireUser() before any DB operation
- [ ] Ownership of video_id verified against authenticated user
- [ ] admin client never imported from files in /app/(client) or components/
- [ ] No analysis row created without prior video ownership check
