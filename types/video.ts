import type { ViewType, VideoStatus } from '@/lib/constants/enums';

export interface SwingVideo {
  id: string;
  user_id: string;
  storage_path: string;
  original_filename: string | null;
  file_size_bytes: number | null;
  duration_ms: number | null;
  view_type: ViewType;
  status: VideoStatus;
  error_code: string | null;
  error_message: string | null;
  processing_started_at: string | null;
  processing_completed_at: string | null;
  created_at: string;
  updated_at: string;
}

export type SwingVideoInsert = Pick<
  SwingVideo,
  'user_id' | 'storage_path' | 'view_type'
> & Partial<Pick<SwingVideo, 'original_filename' | 'file_size_bytes' | 'duration_ms'>>;

export type SwingVideoUpdate = Partial<Pick<
  SwingVideo,
  | 'status'
  | 'error_code'
  | 'error_message'
  | 'processing_started_at'
  | 'processing_completed_at'
  | 'updated_at'
>>;
