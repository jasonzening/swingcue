'use client';

/**
 * EnhancedCoachingBadge — small pill in the player corner indicating
 * the active video has an enhanced coaching overlay (CorrectedTimeline
 * JSON found in Supabase Storage).
 *
 * PR-7c-frontend (Option I demo mode): only visible when
 * SwingPlayer.tsx detects a corrected-timeline JSON for the current
 * videoId. Absent for the majority of videos that fall back to the
 * existing MediaPipe overlay.
 *
 * Placement: top-left of the video area (positioned by parent).
 * Style: minimal invented — no existing pattern in codebase to match.
 */

type Props = {
  /** Display text. Defaults to "Enhanced Coaching Overlay". */
  label?: string;
};

export function EnhancedCoachingBadge({
  label = 'Enhanced Coaching Overlay',
}: Props) {
  return (
    <div
      className="enhanced-coaching-badge"
      style={{
        position: 'absolute',
        top: 8,
        left: 8,
        zIndex: 5,
        background: 'rgba(0, 0, 0, 0.55)',
        color: '#FFFFFF',
        fontSize: 12,
        fontWeight: 500,
        padding: '4px 10px',
        borderRadius: 6,
        pointerEvents: 'none',
        userSelect: 'none',
        // Subtle outline so the pill is readable on bright backgrounds.
        boxShadow: '0 1px 2px rgba(0, 0, 0, 0.25)',
      }}
    >
      {label}
    </div>
  );
}
