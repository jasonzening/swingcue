/**
 * Browser-side fetch for pose_3d_phases.
 *
 * Bypasses @supabase/ssr createBrowserClient because its session
 * hydration was observed (via Chrome DevTools) to fail attaching the user
 * access_token as Bearer on this code path — RLS sees auth.uid()=null and
 * returns 200 + [] silently. We read the session token from the cookie
 * directly and craft the PostgREST request with both apikey (anon) and
 * Authorization (Bearer access_token) headers.
 *
 * Returns [] on any error path; result page falls back to dense MediaPipe.
 */
import { PHASE_ORDER, type PhaseName, type PoseRow } from '@/lib/sam3d/keypoints';

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const ANON_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

function readSessionFromCookie(): { access_token?: string } | null {
  if (typeof document === 'undefined') return null;
  const all = document.cookie.split(';').map(c => c.trim());
  const auth = all.filter(c => c.startsWith('sb-') && c.includes('auth-token'));
  if (auth.length === 0) return null;

  const named: Record<string, string> = {};
  for (const c of auth) {
    const [k, ...v] = c.split('=');
    named[k] = decodeURIComponent(v.join('='));
  }
  // Reassemble chunked cookies in sorted key order (sb-...-auth-token.0, .1, ...)
  const raw = Object.keys(named).sort().map(k => named[k]).join('');

  let payload = raw;
  if (payload.startsWith('base64-')) {
    try { payload = atob(payload.slice(7)); } catch { return null; }
  }
  try { return JSON.parse(payload); } catch { return null; }
}

export async function fetchPoseRows(videoId: string): Promise<PoseRow[]> {
  try {
    const session = readSessionFromCookie();
    if (!session?.access_token) {
      console.warn('[pose-fetch] no access_token in cookie; user not authenticated');
      return [];
    }

    const url = `${SUPABASE_URL}/rest/v1/pose_3d_phases?video_id=eq.${encodeURIComponent(videoId)}&select=*`;
    const res = await fetch(url, {
      headers: {
        apikey: ANON_KEY,
        Authorization: `Bearer ${session.access_token}`,
      },
    });

    if (!res.ok) {
      console.warn(`[pose-fetch] HTTP ${res.status}:`, await res.text().catch(() => ''));
      return [];
    }

    const rows = (await res.json()) as PoseRow[];
    const rank = (p: PhaseName) => PHASE_ORDER.indexOf(p);
    return rows.sort((a, b) => rank(a.phase_name) - rank(b.phase_name));
  } catch (e) {
    console.warn('[pose-fetch] unexpected error:', e);
    return [];
  }
}
