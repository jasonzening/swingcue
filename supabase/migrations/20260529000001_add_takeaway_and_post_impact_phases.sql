-- Extend phase enum on golf_landmark_annotations to support
-- Tier 2 annotation tasks (takeaway + post_impact).
-- These are high-value frames for WHAM correction (movement
-- startup + release-phase fast motion).
-- Already applied to prod via MCP on 2026-05-29; file added for
-- local migration history parity.

alter table public.golf_landmark_annotations
  drop constraint golf_landmark_annotations_phase_check;

alter table public.golf_landmark_annotations
  add constraint golf_landmark_annotations_phase_check
  check (phase in (
    'setup',
    'takeaway',
    'top',
    'transition',
    'impact',
    'post_impact',
    'finish',
    'intermediate'
  ));
