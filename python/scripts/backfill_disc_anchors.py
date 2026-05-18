"""
backfill_disc_anchors.py — PR-4.1 one-shot backfill.

Re-derives `disc_anchors` and bumps `keypoint_source` on every existing
`swing_videos.pose_timeline_2d` row. Does NOT re-run pose estimation —
the formula reads existing stored keypoints and projects two new points
per frame (shoulder_center, hip_belt). Idempotent; safe to re-run.

Environment:
  NEXT_PUBLIC_SUPABASE_URL    — e.g. https://xxxx.supabase.co
  SUPABASE_SERVICE_ROLE_KEY   — service role (server-side only, NEVER browser)

Usage:
  # dry-run all completed timelines, print first-frame diff per video:
  python python/scripts/backfill_disc_anchors.py --dry-run

  # target one video:
  python python/scripts/backfill_disc_anchors.py \\
      --video-id b3fea3f0-e248-44d7-a923-0bb43172b5bf --dry-run

  # actually write:
  python python/scripts/backfill_disc_anchors.py \\
      --video-id b3fea3f0-e248-44d7-a923-0bb43172b5bf

  # backfill everything:
  python python/scripts/backfill_disc_anchors.py --commit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Make pose_timeline importable when run from repo root or from python/.
THIS_DIR = Path(__file__).resolve().parent          # …/python/scripts
PYTHON_DIR = THIS_DIR.parent                        # …/python
sys.path.insert(0, str(PYTHON_DIR))

from pose_timeline import compute_disc_anchors  # noqa: E402

NEW_KP_SOURCE = "mediapipe_yolo_hybrid_v1"

# Calibration target — the video used to derive DISC_HIP_BELT_EXTENSION=0.85
# in PR-3.1 audit. Printed with an "expect ~770" hint so Jason can confirm
# the formula stuck without round-tripping through the browser.
CALIBRATION_VIDEO_ID = "b3fea3f0-e248-44d7-a923-0bb43172b5bf"
CALIBRATION_EXPECTED_BELT_Y = 770


def _http(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None = None,
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def fetch_videos(
    supa_url: str, key: str, video_id: str | None,
) -> list[dict]:
    base = f"{supa_url}/rest/v1/swing_videos"
    params = {
        "select": "id,pose_timeline_2d",
        "pose_timeline_2d": "not.is.null",
    }
    if video_id:
        params["id"] = f"eq.{video_id}"
    qs = urllib.parse.urlencode(params, safe=",.:")
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }
    status, payload = _http("GET", f"{base}?{qs}", headers)
    if status >= 400:
        raise SystemExit(f"GET swing_videos failed: {status} {payload!r}")
    return json.loads(payload.decode())


def patch_video(
    supa_url: str, key: str, video_id: str, timeline: dict,
) -> None:
    url = (f"{supa_url}/rest/v1/swing_videos"
           f"?id=eq.{urllib.parse.quote(video_id, safe='')}")
    body = json.dumps({"pose_timeline_2d": timeline}).encode()
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    status, payload = _http("PATCH", url, headers, body)
    if status >= 400:
        raise SystemExit(f"PATCH {video_id} failed: {status} {payload!r}")


def transform(timeline: dict) -> tuple[dict, dict]:
    """
    Apply PR-4.1 transformation to a stored timeline. Returns
    (new_timeline, stats).
    """
    frames = timeline.get("frames", [])
    attached = 0
    skipped = 0
    first_change: dict | None = None
    for f in frames:
        anchors = compute_disc_anchors(f["keypoints"])
        if anchors is None:
            skipped += 1
            continue
        before = f.get("disc_anchors")
        f["disc_anchors"] = anchors
        attached += 1
        if first_change is None:
            first_change = {
                "ts": f.get("ts"),
                "frame_idx": f.get("frame_idx"),
                "before": before,
                "after": anchors,
                "raw_kp": {
                    "left_shoulder": f["keypoints"].get("left_shoulder"),
                    "right_shoulder": f["keypoints"].get("right_shoulder"),
                    "left_hip": f["keypoints"].get("left_hip"),
                    "right_hip": f["keypoints"].get("right_hip"),
                },
            }
    old_source = timeline.get("keypoint_source")
    timeline["keypoint_source"] = NEW_KP_SOURCE
    return timeline, {
        "total_frames": len(frames),
        "attached": attached,
        "skipped_null_kp": skipped,
        "old_keypoint_source": old_source,
        "new_keypoint_source": NEW_KP_SOURCE,
        "first_change": first_change,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video-id", default=None,
        help="Restrict backfill to a single swing_videos.id (uuid).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute and print stats; do not write to DB. Implied "
             "unless --commit is also given.",
    )
    parser.add_argument(
        "--commit", action="store_true",
        help="Actually PATCH the swing_videos rows. Required to write.",
    )
    args = parser.parse_args()

    if not args.commit:
        args.dry_run = True

    supa_url = os.environ.get("NEXT_PUBLIC_SUPABASE_URL", "").rstrip("/")
    service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not supa_url or not service_key:
        sys.exit(
            "[backfill] missing NEXT_PUBLIC_SUPABASE_URL or "
            "SUPABASE_SERVICE_ROLE_KEY in env. Refusing to run."
        )

    print(f"[backfill] target = {supa_url}")
    print(f"[backfill] mode   = {'COMMIT' if args.commit else 'DRY-RUN'}")
    if args.video_id:
        print(f"[backfill] filter = id={args.video_id}")

    rows = fetch_videos(supa_url, service_key, args.video_id)
    print(f"[backfill] fetched {len(rows)} row(s) with non-null pose_timeline_2d")

    n_ok = 0
    n_skipped = 0
    for row in rows:
        vid = row["id"]
        tl = row["pose_timeline_2d"]
        if not isinstance(tl, dict) or not tl.get("frames"):
            print(f"[backfill] {vid}  → no frames, skipped")
            n_skipped += 1
            continue
        new_tl, stats = transform(tl)
        print(
            f"[backfill] {vid}  "
            f"frames={stats['total_frames']} "
            f"attached={stats['attached']} "
            f"skipped_null_kp={stats['skipped_null_kp']} "
            f"kp_source: {stats['old_keypoint_source']} → {stats['new_keypoint_source']}"
        )
        if stats["first_change"]:
            fc = stats["first_change"]
            print(
                f"           first_change ts={fc['ts']:.3f} "
                f"frame_idx={fc['frame_idx']}  "
                f"hip_belt={fc['after']['hip_belt']}  "
                f"shoulder_center={fc['after']['shoulder_center']}"
            )
        if args.commit:
            patch_video(supa_url, service_key, vid, new_tl)
            print(f"           PATCH OK")
        # Sanity-print for the calibration video so Jason can confirm
        # the formula landed near the visual belt without reloading the
        # browser. Fires in BOTH dry-run and commit modes.
        if vid == CALIBRATION_VIDEO_ID and stats["first_change"]:
            written_y = stats["first_change"]["after"]["hip_belt"]["y"]
            print(
                f"[backfill] b3fea3f0 frame 0 hip_belt.y = {written_y:.1f} "
                f"(expect ~{CALIBRATION_EXPECTED_BELT_Y})"
            )
        n_ok += 1
    print(
        f"[backfill] done: processed={n_ok} skipped={n_skipped} "
        f"mode={'COMMIT' if args.commit else 'DRY-RUN'}"
    )


if __name__ == "__main__":
    main()
