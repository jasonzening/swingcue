# MVP-001 Upload and Processing Flow

## Goal
Implement the first auditable end-to-end flow:
upload -> processing -> completed/failed -> result retrieval

## Supported view types
- face_on
- down_the_line

## Required flow
1. Authenticated user uploads a video.
2. User chooses one supported view type.
3. Server stores the video in storage.
4. Server creates a swing_videos record with status = uploaded.
5. User or client flow triggers analysis.
6. System updates status = processing and processing_started_at.
7. Deterministic stub returns either completed payload or failed payload.
8. On completed:
   - status becomes completed
   - processing_completed_at is set
   - one swing_analysis row is stored
   - zero or more swing_phase_snapshots rows are stored
9. On failed:
   - status becomes failed
   - processing_completed_at is set
   - error_code and error_message are stored

## API expectations
- upload route creates storage object + DB record
- analyze route performs explicit transition and persistence
- result route reads owner-safe completed data
- status route returns explicit status

## Non-goals for this spec
- background jobs
- retry orchestration
- progress percentages
- full history UX
