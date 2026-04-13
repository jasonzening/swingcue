-- =============================================================================
-- Migration: 20250409000001_schema_foundation
-- Spec: MVP-006-supabase-schema
-- Scope: PR-1A — schema foundation only
-- Additive: YES (no tables existed before this)
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Enums
-- ---------------------------------------------------------------------------

create type swing_view_type as enum (
  'face_on',
  'down_the_line'
);

create type swing_status as enum (
  'uploaded',
  'processing',
  'completed',
  'failed'
);

create type swing_severity as enum (
  'low',
  'medium',
  'high'
);

create type swing_issue_type as enum (
  'weight_shift_issue',
  'head_movement',
  'early_extension',
  'steep_backswing_plane',
  'steep_downswing',
  'hand_path_issue'
);

create type swing_phase_name as enum (
  'address',
  'top',
  'downswing',
  'impact',
  'finish'
);

-- ---------------------------------------------------------------------------
-- swing_videos
-- ---------------------------------------------------------------------------

create table swing_videos (
  id                      uuid primary key default gen_random_uuid(),
  user_id                 uuid not null references auth.users(id) on delete cascade,

  storage_path            text not null,
  original_filename       text not null,
  file_size_bytes         bigint,
  duration_ms             integer,

  view_type               swing_view_type not null,
  status                  swing_status not null default 'uploaded',

  -- failure fields (null unless status = 'failed')
  error_code              text,
  error_message           text,

  -- processing timestamps (null until transitions occur)
  processing_started_at   timestamptz,
  processing_completed_at timestamptz,

  created_at              timestamptz not null default now(),
  updated_at              timestamptz not null default now()
);

-- Ensure status is never inferred: updated_at stays current
create or replace function update_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create trigger swing_videos_updated_at
  before update on swing_videos
  for each row execute function update_updated_at();

-- ---------------------------------------------------------------------------
-- swing_analysis
-- One row per completed video. Enforces one-one-one rule via FK + unique.
-- ---------------------------------------------------------------------------

create table swing_analysis (
  id                  uuid primary key default gen_random_uuid(),
  video_id            uuid not null references swing_videos(id) on delete cascade,

  issue_type          swing_issue_type not null,
  summary_text        text not null,   -- maps to main_issue_text in contract
  cue_text            text not null,   -- maps to fix_cue_text in contract
  drill_text          text not null,

  overlay_asset_url   text,
  score               numeric(4,1),    -- e.g. 72.5, nullable
  severity            swing_severity,

  created_at          timestamptz not null default now(),
  updated_at          timestamptz not null default now()
);

-- One-one-one rule: exactly one analysis row per video
create unique index swing_analysis_one_per_video on swing_analysis(video_id);

create trigger swing_analysis_updated_at
  before update on swing_analysis
  for each row execute function update_updated_at();

-- ---------------------------------------------------------------------------
-- swing_phase_snapshots
-- Zero or more per video. Phase + index pair is unique per video.
-- ---------------------------------------------------------------------------

create table swing_phase_snapshots (
  id                  uuid primary key default gen_random_uuid(),
  video_id            uuid not null references swing_videos(id) on delete cascade,

  phase_name          swing_phase_name not null,
  snapshot_asset_url  text not null,
  frame_index         integer not null,

  created_at          timestamptz not null default now()
);

-- Prevent duplicate phase+frame combinations per video
create unique index swing_phase_snapshots_unique_phase_frame
  on swing_phase_snapshots(video_id, phase_name, frame_index);

-- ---------------------------------------------------------------------------
-- Row Level Security
-- ---------------------------------------------------------------------------

alter table swing_videos          enable row level security;
alter table swing_analysis        enable row level security;
alter table swing_phase_snapshots enable row level security;

-- swing_videos: owner reads own rows only
create policy "swing_videos: owner select"
  on swing_videos for select
  using (auth.uid() = user_id);

-- swing_videos: owner inserts own rows (user_id enforced by policy, not client)
create policy "swing_videos: owner insert"
  on swing_videos for insert
  with check (auth.uid() = user_id);

-- swing_videos: owner updates own rows only (status transitions, timestamps)
create policy "swing_videos: owner update"
  on swing_videos for update
  using (auth.uid() = user_id);

-- swing_analysis: readable only if the viewer owns the parent video
create policy "swing_analysis: owner select via video"
  on swing_analysis for select
  using (
    exists (
      select 1 from swing_videos v
      where v.id = swing_analysis.video_id
        and v.user_id = auth.uid()
    )
  );

-- swing_analysis: insertable only when parent video is owned by caller
-- (API routes use service role for inserts; this policy guards direct access)
create policy "swing_analysis: owner insert via video"
  on swing_analysis for insert
  with check (
    exists (
      select 1 from swing_videos v
      where v.id = swing_analysis.video_id
        and v.user_id = auth.uid()
    )
  );

-- swing_phase_snapshots: readable only if viewer owns the parent video
create policy "swing_phase_snapshots: owner select via video"
  on swing_phase_snapshots for select
  using (
    exists (
      select 1 from swing_videos v
      where v.id = swing_phase_snapshots.video_id
        and v.user_id = auth.uid()
    )
  );

-- swing_phase_snapshots: insertable only when parent video is owned by caller
create policy "swing_phase_snapshots: owner insert via video"
  on swing_phase_snapshots for insert
  with check (
    exists (
      select 1 from swing_videos v
      where v.id = swing_phase_snapshots.video_id
        and v.user_id = auth.uid()
    )
  );

-- ---------------------------------------------------------------------------
-- No destructive operations. Migration is purely additive.
-- ---------------------------------------------------------------------------
