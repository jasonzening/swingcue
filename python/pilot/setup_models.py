"""
setup_models.py — one-time Volume populator for the phase2b WHAM smoke.

Downloads the 6 WHAM-specific weight files from the Google Drive URLs
listed in WHAM's official fetch_demo_data.sh (commit
2b54f7797391c94876848b905ed875b154c4a295). Stores them at
/models/wham/ inside the persistent `swingcue-pilot-models` Volume.

What this DOES download (license-compliant — public Drive releases):
  - wham_vit_w_3dpw.pth.tar       primary WHAM checkpoint
  - wham_vit_bedlam_w_3dpw.pth.tar alt checkpoint (bedlam-trained)
  - hmr2a.ckpt                     HMR2A subnet
  - dpvo.pth                       DPVO SLAM subnet
  - yolov8x.pt                     YOLOv8 person detector
  - vitpose-h-multi-coco.pth       ViTPose-H 2D keypoint subnet

What this DOES NOT download:
  - SMPL / SMPL-X / SMPL-H body models — research-license, NOT scraped.
    Jason downloads via the official registration sites + uploads via
    `modal volume put swingcue-pilot-models ./local-body-models
    /models/body_models`. See modal_app.py docstring or
    python/pilot/README.md for the layout.

Run pattern (CC drives, once Jason's prep is done):

    ./.venv-pilot/Scripts/python.exe -m modal run \\
        python/pilot/setup_models.py::setup_all_models

Cost: Modal egress + Volume storage. Free tier covers (~1.5 GB total).
Wall clock ~5-10 min for the gdown downloads to complete.
"""

from __future__ import annotations

import os
import subprocess

# Tolerate ad-hoc invocation (`python setup_models.py`) too.
# wham_image is also imported here so the decorator below sees it
# regardless of relative-vs-script invocation.
try:
    from .modal_app import app, model_volume, wham_image, _MODAL_AVAILABLE
except ImportError:  # pragma: no cover
    from modal_app import app, model_volume, wham_image, _MODAL_AVAILABLE


# ---------------------------------------------------------------------------
# WHAM weight catalog — drives setup_all_models's gdown loop.
#
# Source: WHAM's fetch_demo_data.sh at commit
# 2b54f7797391c94876848b905ed875b154c4a295 (2026-05-20). If WHAM
# upstream rotates these IDs, re-fetch the script and update here.
#
# Each entry: (gdrive_id, target_filename, expected_min_size_mb).
# expected_min_size_mb is the fail-fast threshold — if gdown returns a
# file smaller than this, we crash rather than silently leave a stub.
# ---------------------------------------------------------------------------

WHAM_WEIGHTS: tuple[tuple[str, str, int], ...] = (
    # (gdrive_id,                          filename,                            min_size_mb)
    ("1i7kt9RlCCCNEW2aYaDWVr-G778JkLNcB", "wham_vit_w_3dpw.pth.tar",       400),
    ("19qkI-a6xuwob9_RFNSPWf1yWErwVVlks", "wham_vit_bedlam_w_3dpw.pth.tar", 400),
    ("1J6l8teyZrL0zFzHhzkC7efRhU0ZJ5G9Y", "hmr2a.ckpt",                    2000),
    ("1kXTV4EYb-BI3H7J-bkR3Bc4gT9zfnHGT", "dpvo.pth",                       200),
    ("1zJ0KP23tXD42D47cw1Gs7zE2BA_V_ERo", "yolov8x.pt",                     130),
    ("1xyF7F3I7lWtdq82xmEPVQ5zl4HaasBso", "vitpose-h-multi-coco.pth",       800),
)

WHAM_TARGET_DIR = "/models/wham"
BODY_MODELS_DIR = "/models/body_models"  # populated by `modal volume put`


# ---------------------------------------------------------------------------
# Download helper. Uses gdown's Python API (preferred over the CLI for
# error capture); falls back to `gdown` subprocess if the API path
# fails (some Drive files require alternate cookie handling).
# ---------------------------------------------------------------------------

def _gdown_one(gdrive_id: str, dst: str, min_size_mb: int) -> None:
    import gdown
    url = f"https://drive.google.com/uc?id={gdrive_id}&export=download&confirm=t"
    print(f"[setup_models] gdown {gdrive_id} → {dst}")
    try:
        gdown.download(url, dst, quiet=False, fuzzy=True)
    except Exception as e:
        # Fallback: try the CLI form, which has its own confirm-token
        # handling for large files.
        print(f"[setup_models] gdown API failed ({e}); retrying via CLI")
        subprocess.run(
            ["gdown", "--id", gdrive_id, "--output", dst, "--fuzzy"],
            check=True,
        )

    if not os.path.exists(dst):
        raise RuntimeError(f"gdown produced no file at {dst}")
    sz_mb = os.path.getsize(dst) / 1024 / 1024
    print(f"[setup_models]   downloaded: {sz_mb:.1f} MB")
    if sz_mb < min_size_mb:
        raise RuntimeError(
            f"gdown wrote {sz_mb:.1f} MB but expected ≥ {min_size_mb} MB — "
            f"likely a Drive virus-scan HTML page, not the model file. "
            f"Manual re-fetch needed for {gdrive_id}."
        )


# ---------------------------------------------------------------------------
# Verify Jason's `modal volume put` SMPL upload landed in the expected
# layout. Phase2b inference fails fast if not — we don't proceed to
# WHAM run when body models are missing.
# ---------------------------------------------------------------------------

EXPECTED_BODY_MODELS = (
    # (relative_path, min_size_mb) — sized for the canonical
    # research-license downloads.
    ("smpl/SMPL_NEUTRAL.pkl",    30),
    ("smpl/SMPL_MALE.pkl",       30),
    ("smpl/SMPL_FEMALE.pkl",     30),
    ("smplh/SMPLH_NEUTRAL.npz", 150),
    # SMPL-X is nice-to-have for phase2c expansion; warn but don't fail.
)


def _verify_body_models() -> list[str]:
    missing: list[str] = []
    for rel, min_mb in EXPECTED_BODY_MODELS:
        full = os.path.join(BODY_MODELS_DIR, rel)
        if not os.path.exists(full):
            missing.append(f"MISSING {full}")
            continue
        sz_mb = os.path.getsize(full) / 1024 / 1024
        if sz_mb < min_mb:
            missing.append(f"TOO_SMALL {full} ({sz_mb:.1f} MB < {min_mb} MB)")
        else:
            print(f"[setup_models]   body_models OK: {rel} ({sz_mb:.1f} MB)")
    return missing


# ---------------------------------------------------------------------------
# Modal function entrypoint.
# Decorated only if Modal is importable so this file is safe to
# py_compile in CI without the Modal client installed.
# ---------------------------------------------------------------------------

if _MODAL_AVAILABLE:
    # The WHAM image (imported at the top) carries gdown + ffmpeg + the
    # rest of the inference stack; using it for setup is fine and avoids
    # defining a second Image just for the download step.

    @app.function(
        image=wham_image,
        volumes={"/models": model_volume},
        timeout=3600,
    )
    def setup_all_models() -> None:
        """Populate /models/wham/* on the swingcue-pilot-models Volume."""
        os.makedirs(WHAM_TARGET_DIR, exist_ok=True)

        print(
            f"[setup_models] downloading {len(WHAM_WEIGHTS)} WHAM weights "
            f"to {WHAM_TARGET_DIR}"
        )
        for gdrive_id, filename, min_size_mb in WHAM_WEIGHTS:
            dst = os.path.join(WHAM_TARGET_DIR, filename)
            if (
                os.path.exists(dst)
                and os.path.getsize(dst) / 1024 / 1024 >= min_size_mb
            ):
                print(f"[setup_models] skip (already present, ≥ {min_size_mb} MB): {filename}")
                continue
            _gdown_one(gdrive_id, dst, min_size_mb)

        # Commit Volume — without this the next inference function sees
        # an empty Volume.
        model_volume.commit()
        print("[setup_models] Volume committed")

        # Cross-check SMPL upload (Jason's `modal volume put` step).
        # Warn but don't fail — WHAM weight download alone is useful
        # progress even if body models aren't up yet.
        print(f"[setup_models] verifying body_models at {BODY_MODELS_DIR}")
        if not os.path.isdir(BODY_MODELS_DIR):
            print(
                f"[setup_models] WARNING: {BODY_MODELS_DIR} not present. "
                f"Run `modal volume put swingcue-pilot-models "
                f"./local-body-models /models/body_models` before phase2b "
                f"WHAM inference."
            )
        else:
            missing = _verify_body_models()
            if missing:
                print("[setup_models] body_models gaps:")
                for line in missing:
                    print(f"  - {line}")
                print(
                    "[setup_models] phase2b WHAM run will fail until these "
                    "are uploaded via `modal volume put`."
                )

        # Final report
        total_files = 0
        total_mb = 0.0
        for root, _, files in os.walk("/models"):
            for f in files:
                full = os.path.join(root, f)
                total_files += 1
                total_mb += os.path.getsize(full) / 1024 / 1024
        print(
            f"[setup_models] /models inventory: {total_files} files, "
            f"{total_mb:.1f} MB"
        )


# ---------------------------------------------------------------------------
# Local self-test — print the function spec without invoking Modal.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _MODAL_AVAILABLE:
        print(
            "[setup_models] modal not installed — "
            "pip install -r python/pilot/requirements_pilot.txt"
        )
    else:
        print(f"[setup_models] WHAM weights queued ({len(WHAM_WEIGHTS)} files):")
        for gdrive_id, fname, min_mb in WHAM_WEIGHTS:
            print(f"  - {fname:40s} (gdrive={gdrive_id}, ≥{min_mb} MB)")
        print(f"[setup_models] body_models verify checks: {len(EXPECTED_BODY_MODELS)}")
        print(
            "[setup_models] invoke remotely with: "
            "modal run python/pilot/setup_models.py::setup_all_models"
        )
