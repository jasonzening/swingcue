# PR-4.1 ship order

1. `git commit` + `git push` — code live on main.
2. Vercel auto-deploys (~2 min).
3. Jason runs backfill targeted at the calibration video first
   (`python python/scripts/backfill_disc_anchors.py --video-id b3fea3f0-e248-44d7-a923-0bb43172b5bf --commit`).
4. Jason hard-reloads `swingcue.ai/result/b3fea3f0…` and confirms disc lands on the belt.
5. Jason runs full backfill (`python python/scripts/backfill_disc_anchors.py --commit`) — every existing video gets `disc_anchors` + bumped `keypoint_source`.
6. Done. All videos render with body-aligned discs; new uploads go through the new pipeline automatically.

Note: between steps 1 and 3, `b3fea3f0`'s disc stays in its old position — the frontend reads the missing `disc_anchors` field and falls back to the raw midpoint. This is expected behaviour, not a bug. Once step 3 writes the new field, the next page load picks it up.
