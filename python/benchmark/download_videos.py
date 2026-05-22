"""
download_videos.py — fetch the 3 benchmark test videos via service-role.

Looks up `swing_videos.storage_path` for each hard-coded video ID, mints
a Supabase signed URL via the `swing-videos` bucket, downloads the mp4
to `test_videos/<id>.mp4`. Service-role-only — never run in the
browser; the key is read from env.

Env required:
    NEXT_PUBLIC_SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY

CLI:
    cd python/benchmark
    python download_videos.py            # downloads all 3
    python download_videos.py b3fea3f0   # just one (substring match on id)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx

# Three videos picked for the benchmark suite. b3fea3f0 is the
# "calibration video" used throughout PR-3.1 / PR-5.6 / PR-5.7.
# a735cc7d + 5bbcfbc8 are additional samples — pick any two if
# these IDs aren't available in your DB, and update this list.
DEFAULT_VIDEO_IDS: tuple[str, ...] = (
    "b3fea3f0-e248-44d7-a923-0bb43172b5bf",   # face_on calibration video
    "a735cc7d-1d4d-4b73-870f-30dca5c4aac0",   # face_on
    "5bbcfbc8-49b9-4fc4-8b0e-a34c5427aa62",   # face_on
    "b32e0f21-2656-473c-aa87-e1eaf6e1221f",   # down_the_line (filename: downtheline-4miao.mp4)
)

BUCKET = "swing-videos"
SIGNED_URL_EXPIRY_SEC = 600


def _env(key: str) -> str:
    v = os.environ.get(key)
    if not v:
        sys.exit(f"[download] missing env var: {key}")
    return v


def fetch_storage_path(supa_url: str, key: str, video_id: str) -> str | None:
    """GET /rest/v1/swing_videos?id=eq.<vid> → storage_path."""
    url = f"{supa_url}/rest/v1/swing_videos"
    headers = {
        "apikey":        key,
        "Authorization": f"Bearer {key}",
        "Accept":        "application/json",
    }
    params = {"id": f"eq.{video_id}", "select": "id,storage_path"}
    r = httpx.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        return None
    return rows[0].get("storage_path")


def sign_url(supa_url: str, key: str, path: str) -> str | None:
    """POST /storage/v1/object/sign/<bucket>/<path> → signedURL."""
    url = (f"{supa_url}/storage/v1/object/sign/"
           f"{BUCKET}/{path}")
    headers = {
        "apikey":        key,
        "Authorization": f"Bearer {key}",
        "Content-Type":  "application/json",
    }
    body = {"expiresIn": SIGNED_URL_EXPIRY_SEC}
    r = httpx.post(url, headers=headers, json=body, timeout=30)
    r.raise_for_status()
    payload = r.json()
    signed = payload.get("signedURL") or payload.get("signedUrl")
    if not signed:
        return None
    # Returned as a relative path; Supabase expects it appended to /storage/v1.
    if signed.startswith("/"):
        signed = f"{supa_url}/storage/v1{signed}"
    return signed


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        with dest.open("wb") as f:
            for chunk in r.iter_bytes(chunk_size=64 * 1024):
                f.write(chunk)
                written += len(chunk)
        print(f"[download] {dest.name}: {written:,} bytes "
              f"({(written/max(1, total))*100:.0f}% of declared {total})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id_prefix", nargs="?", default=None,
                    help="substring of a UUID; if omitted, all 3 are fetched")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).parent / "test_videos")
    args = ap.parse_args()

    supa_url = _env("NEXT_PUBLIC_SUPABASE_URL").rstrip("/")
    key      = _env("SUPABASE_SERVICE_ROLE_KEY")

    targets = [
        v for v in DEFAULT_VIDEO_IDS
        if (args.video_id_prefix is None or args.video_id_prefix in v)
    ]
    if not targets:
        sys.exit(f"[download] no video id matched prefix {args.video_id_prefix!r}")

    for vid in targets:
        print(f"[download] resolving {vid} …")
        if len(vid) < 36:
            print(f"[download]   skipping — {vid} doesn't look like a full "
                  f"UUID. Edit DEFAULT_VIDEO_IDS in download_videos.py.")
            continue
        path = fetch_storage_path(supa_url, key, vid)
        if not path:
            print(f"[download]   no DB row → skip")
            continue
        signed = sign_url(supa_url, key, path)
        if not signed:
            print(f"[download]   sign failed → skip")
            continue
        dest = args.out_dir / f"{vid}.mp4"
        download(signed, dest)
        print(f"[download]   → {dest}")


if __name__ == "__main__":
    sys.exit(main())
