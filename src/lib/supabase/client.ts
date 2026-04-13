/**
 * client.ts
 * Browser-side Supabase client for use in Client Components.
 * Uses @supabase/ssr createBrowserClient for cookie-based session handling.
 *
 * Scope: PR-1A — client helper only, no data logic
 */

import { createBrowserClient } from '@supabase/ssr';

export function createClient() {
  return createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
  );
}
