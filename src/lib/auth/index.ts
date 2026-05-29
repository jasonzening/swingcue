import { NextResponse } from 'next/server';
import { redirect } from 'next/navigation';
import type { User } from '@supabase/supabase-js';
import { createClient } from '@/lib/supabase/server';

/* ============================================================
   ADMIN ALLOWLIST
   ============================================================ */

function getAdminUserIds(): Set<string> {
  const raw = process.env.ADMIN_USER_IDS ?? '';
  return new Set(
    raw.split(',').map(s => s.trim()).filter(Boolean),
  );
}

export function isAdminUserId(id: string): boolean {
  return getAdminUserIds().has(id);
}

/* ============================================================
   ROUTE HANDLER GATES (return NextResponse on failure)
   ============================================================ */

type AuthResult = { user: User } | { response: NextResponse };

/** Server route handler: require any authenticated user. */
export async function requireUser(): Promise<AuthResult> {
  const supabase = await createClient();
  const { data: { user }, error } = await supabase.auth.getUser();
  if (error || !user) {
    return { response: NextResponse.json({ error: 'Unauthorized' }, { status: 401 }) };
  }
  return { user };
}

/** Server route handler: require an admin user (in ADMIN_USER_IDS). */
export async function requireAdmin(): Promise<AuthResult> {
  const result = await requireUser();
  if ('response' in result) return result;
  if (!isAdminUserId(result.user.id)) {
    return { response: NextResponse.json({ error: 'Forbidden' }, { status: 403 }) };
  }
  return result;
}

/* ============================================================
   SERVER COMPONENT GATES (redirect on failure)
   ============================================================ */

/**
 * Server component: require an admin user.
 * Redirects to /sign-in if no session, to / if signed in but not admin.
 */
export async function requireAdminPage(): Promise<User> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) redirect('/sign-in');
  if (!isAdminUserId(user.id)) redirect('/');
  return user;
}
