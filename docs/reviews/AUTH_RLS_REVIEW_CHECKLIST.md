# AUTH_RLS_REVIEW_CHECKLIST

- [ ] every user-facing data query is owner-scoped
- [ ] no client-controlled user_id trust
- [ ] protected routes enforce authenticated access
- [ ] API routes enforce authenticated access
- [ ] storage access pattern is compatible with user ownership
- [ ] no broad service-role usage in request path without strict containment
