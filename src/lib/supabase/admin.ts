import { createServerClient } from '@supabase/ssr';

/**
 * Service-role Supabase client.
 *
 * Bypasses RLS. SERVER ONLY — never import from anything that ends up
 * in a client bundle. Allowed only in:
 *   - src/app/api/** route handlers
 *   - Server actions
 *   - Server components that explicitly need admin access
 *
 * Per docs/decisions/API_CLIENT_BOUNDARY.md.
 */
export function createServiceClient() {
  if (!process.env.SUPABASE_SERVICE_ROLE_KEY) {
    throw new Error('SUPABASE_SERVICE_ROLE_KEY is not set');
  }
  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY,
    {
      cookies: { getAll: () => [], setAll: () => {} },
    },
  );
}
