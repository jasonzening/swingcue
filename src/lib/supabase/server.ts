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

// Re-export for backwards compatibility with existing call sites.
// New code should import directly from '@/lib/supabase/admin'.
export { createServiceClient } from './admin';
