-- golf_landmark_annotations
-- Long-term human-corrected ground truth for golf-specific keypoints.
-- Drives PR-7A validation and future golf-specific pose model training.
-- Service-role only access (admin annotation workbench uses
-- SUPABASE_SERVICE_ROLE_KEY).

create table if not exists public.golf_landmark_annotations (
  id uuid primary key default gen_random_uuid(),

  video_id      uuid not null references public.swing_videos(id) on delete cascade,
  annotator_id  uuid not null references auth.users(id),

  frame_idx     int  not null,
  phase         text not null check (phase in ('setup','top','impact','finish','transition','intermediate')),

  task_type     text not null default 'manual_gt'
                check (task_type in ('manual_gt','correction_review','active_learning')),

  arm           text not null check (arm in ('lead','trail')),

  visibility    text not null default 'clear'
                check (visibility in ('clear','occluded','uncertain')),

  shoulder_x    real,
  shoulder_y    real,
  elbow_x       real,
  elbow_y       real,
  wrist_x       real,
  wrist_y       real,

  handedness    text not null check (handedness in ('right','left')),

  source_app_version text not null,
  annotated_at  timestamptz not null default now()
);

create unique index if not exists golf_landmark_annotations_uq
  on public.golf_landmark_annotations (video_id, frame_idx, arm, annotator_id, task_type);

create index if not exists golf_landmark_annotations_video_idx
  on public.golf_landmark_annotations (video_id);

create index if not exists golf_landmark_annotations_annotator_idx
  on public.golf_landmark_annotations (annotator_id);

comment on table public.golf_landmark_annotations is
  'Human-corrected golf-specific keypoint ground truth. Powers PR-7A Teaching Landmark validation and future model training. Admin-only access via service role.';

comment on column public.golf_landmark_annotations.task_type is
  'manual_gt = guided one-click annotation by admin (highest trust). correction_review = post-hoc spot-check. active_learning = system-selected uncertain frames (future).';

comment on column public.golf_landmark_annotations.visibility is
  'clear = annotator confident, treat as strong GT. occluded = some keypoints not visible in frame (any *_x/*_y may be null). uncertain = even the annotator was not sure — do not use as strong GT.';

alter table public.golf_landmark_annotations enable row level security;
