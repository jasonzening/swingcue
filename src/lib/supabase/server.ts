/**
 * server.ts
 * Server-side Supabase client for use in Server Components and API route handlers.
 * Uses @supabase/ssr createServerClient with Next.js cookies() for session hydration.
 *
 * Scope: PR-1A — server client helper only, no data logic
 */

import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';

export async function createClient() {
  const cookieStore = await cookies();

  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll();
        },
        setAll(cookiesToSet) {
          try {
            cookiesToSet.forEach(({ name, value, options }) =>
              cookieStore.set(name, value, options),
            );
          } catch {
            // setAll called from a Server Component where cookies are read-only.
            // Session refresh is handled by middleware; this is safe to ignore.
          }
        },
      },
    },
  );
}

/**
 * Service-role client for write operations that must bypass RLS
 * (e.g. status transitions during analysis processing).
 *
 * IMPORTANT: Never expose this client to the browser.
 * Only import from server-only paths (API routes, Server Actions).
 */
export function createServiceClient() {
  if (!process.env.SUPABASE_SERVICE_ROLE_KEY) {
    throw new Error('SUPABASE_SERVICE_ROLE_KEY is not set');
  }

  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY,
    {
      cookies: {
        getAll: () => [],
        setAll: () => {},
      },
    },
  );
}
