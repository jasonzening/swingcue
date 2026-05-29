import { requireAdminPage } from '@/lib/auth';
import { AnnotateHome } from './AnnotateHome';

// Admin tools must never be statically generated — the auth gate depends
// on per-request cookies (ADMIN_USER_IDS allowlist check).
export const dynamic = 'force-dynamic';

export default async function Page() {
  await requireAdminPage();
  return <AnnotateHome />;
}
