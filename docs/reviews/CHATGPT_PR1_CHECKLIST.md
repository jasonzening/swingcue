# CHATGPT_PR1_CHECKLIST

## Schema
- [ ] swing_videos exists
- [ ] swing_analysis exists
- [ ] swing_phase_snapshots exists
- [ ] view_type explicit
- [ ] status explicit
- [ ] additive-only migration

## Flow
- [ ] upload creates persistent record
- [ ] uploaded -> processing transition explicit
- [ ] completed / failed transition explicit
- [ ] deterministic stub used
- [ ] completed result persists one issue/cue/drill

## Safety
- [ ] protected pages require auth
- [ ] API routes require auth
- [ ] no cross-user reads
- [ ] no hidden admin misuse

## Product
- [ ] only face_on and down_the_line supported
- [ ] no product scope drift
