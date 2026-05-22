"""
run_wham_one.py — sign a Supabase URL for one video_id and invoke
`modal run wham_runner.py::run_wham_local` against it.

Loads NEXT_PUBLIC_SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY from
.env.local without echoing either to stdout. The signed URL (10-min
expiry) is passed to modal via subprocess argv — sensitive but short-
lived; the service-role key never leaves this process.

Usage (from repo root, with .env.local present):
    .venv-pilot/Scripts/python.exe python/pilot/scripts/run_wham_one.py \\
        b32e0f21-2656-473c-aa87-e1eaf6e1221f
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[3]
BUCKET = "swing-videos"
SIGNED_URL_EXPIRY_SEC = 600


def _load_env_local(path: Path) -> None:
    """Parse .env.local into os.environ; tolerate quoted values, comments."""
    if not path.exists():
        sys.exit(f"[run_wham_one] missing {path}")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k:
            os.environ.setdefault(k, v)


def _fetch_storage_path(supa_url: str, key: str, video_id: str) -> str:
    r = httpx.get(
        f"{supa_url}/rest/v1/swing_videos",
        headers={"apikey": key, "Authorization": f"Bearer {key}",
                 "Accept": "application/json"},
        params={"id": f"eq.{video_id}", "select": "id,storage_path"},
        timeout=30,
    )
    r.raise_for_status()
    rows = r.json()
    if not rows:
        sys.exit(f"[run_wham_one] no DB row for {video_id}")
    return rows[0]["storage_path"]


def _sign_url(supa_url: str, key: str, path: str) -> str:
    r = httpx.post(
        f"{supa_url}/storage/v1/object/sign/{BUCKET}/{path}",
        headers={"apikey": key, "Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        json={"expiresIn": SIGNED_URL_EXPIRY_SEC},
        timeout=30,
    )
    r.raise_for_status()
    signed = r.json().get("signedURL") or r.json().get("signedUrl")
    if not signed:
        sys.exit("[run_wham_one] sign failed (no signedURL in response)")
    if signed.startswith("/"):
        signed = f"{supa_url}/storage/v1{signed}"
    return signed


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: run_wham_one.py <video_id>")
    video_id = sys.argv[1]

    _load_env_local(REPO_ROOT / ".env.local")
    supa_url = os.environ.get("NEXT_PUBLIC_SUPABASE_URL", "").rstrip("/")
    key      = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not supa_url or not key:
        sys.exit("[run_wham_one] env not set (NEXT_PUBLIC_SUPABASE_URL / "
                 "SUPABASE_SERVICE_ROLE_KEY)")

    print(f"[run_wham_one] resolving storage_path for {video_id} ...")
    storage_path = _fetch_storage_path(supa_url, key, video_id)
    print(f"[run_wham_one] storage_path={storage_path}")

    signed_url = _sign_url(supa_url, key, storage_path)
    print(f"[run_wham_one] signed (expires in {SIGNED_URL_EXPIRY_SEC}s); invoking modal ...")

    cmd = [
        str(REPO_ROOT / ".venv-pilot" / "Scripts" / "modal.exe"),
        "run",
        str(REPO_ROOT / "python" / "pilot" / "runners" / "wham_runner.py")
        + "::run_wham_local",
        "--video-id", video_id,
        "--video-url", signed_url,
    ]
    # Do NOT print the full cmd (would echo signed URL). Print scrubbed.
    print(f"[run_wham_one] exec: {cmd[0]} run wham_runner.py::run_wham_local "
          f"--video-id {video_id} --video-url <signed:redacted>")
    # Force UTF-8 for the subprocess: modal CLI emits checkmarks that
    # blow up Windows' default GBK/CP936 console codec.
    sub_env = os.environ.copy()
    sub_env["PYTHONIOENCODING"] = "utf-8"
    sub_env["PYTHONUTF8"] = "1"
    rc = subprocess.call(cmd, cwd=str(REPO_ROOT), env=sub_env)
    sys.exit(rc)


if __name__ == "__main__":
    main()
