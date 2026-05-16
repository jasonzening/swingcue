/**
 * Browser-side fetch for pose_3d_phases.
 *
 * Uses the anon SSR client (cookie-bound). The pose_3d_phases_user_select
 * RLS policy enforces `auth.uid() = user_id`, so a user only sees rows for
 * their own videos. Returns rows sorted by canonical phase order.
 *
 * Returns `[]` on any error path (not authenticated, RLS reject, network
 * fail) — the result page treats empty as "no sparse data, fall back to
 * dense MediaPipe path".
 */

import { createClient } from '@/lib/supabase/client';
import { PHASE_ORDER, type PhaseName, type PoseRow } from '@/lib/sam3d/keypoints';

export async function fetchPoseRows(videoId: string): Promise<PoseRow[]> {
  try {
    const supabase = createClient();
    const { data, error } = await supabase
      .from('pose_3d_phases')
      .select('*')
      .eq('video_id', videoId);

    if (error) {
      console.warn('[pose-fetch] query error:', error.message);
      return [];
    }

    const rows = (data ?? []) as PoseRow[];

    // Canonical phase ordering (PostgREST returns insertion order by default,
    // which is fine but not guaranteed); explicit sort makes downstream code
    // simpler.
    const rank = (p: PhaseName) => PHASE_ORDER.indexOf(p);
    return rows.sort((a, b) => rank(a.phase_name) - rank(b.phase_name));
  } catch (e) {
    console.warn('[pose-fetch] unexpected error:', e);
    return [];
  }
}
