# MVP-005 Auth and Protected Routes

## Goal
Protect user data and keep the first-wave flow ownership-safe.

## Protected pages
- upload page
- processing page
- result page
- history page when added

## Protected API routes
- upload route
- analyze route
- status route
- result route
- history route when added

## Rules
- authenticated user required
- user ownership enforced server-side
- do not trust client-supplied ownership identifiers
- routes must fail safely when unauthenticated or unauthorized
