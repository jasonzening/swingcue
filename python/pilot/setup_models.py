"""
setup_models.py — one-time Volume-populator for the Phase 2 pilot.

Defines the Modal function that downloads model weights into the
`swingcue-pilot-models` Volume. phase2a defines this; phase2b is the
first phase that actually invokes it via Modal.

Run pattern (Jason runs locally AFTER bootstrap is complete):

    modal run python/pilot/setup_models.py::setup_all_models

Subsequent inference @app.function decorators in runner modules mount
the Volume read-only and find weights already there. Volume persists
across deploys.

Cost: ~15-30 GB Modal egress + Volume storage. One-time per library.
Free tier covers this; budget impact lives in phase2b/c per spec §3.
"""

from __future__ import annotations

import os

try:
    from .modal_app import app, model_volume, smpl_credentials, _MODAL_AVAILABLE
except ImportError:  # pragma: no cover — module path differences during dev
    from modal_app import app, model_volume, smpl_credentials, _MODAL_AVAILABLE


# ---------------------------------------------------------------------------
# WHAM weights — phase2b prerequisite.
# ---------------------------------------------------------------------------

WHAM_CHECKPOINT_URL = (
    # Official WHAM release on the maintainer's checkpoint host.
    # TODO(phase2b): WHAM repo's README points at multiple URLs across
    # releases; pin the exact URL we tested against at phase2b time.
    "https://github.com/yohanshin/WHAM/releases/download/v1.0/wham_vit.tar"
)


def _download_wham(target_dir: str) -> None:
    """Stub. phase2b fills in the real download + extract + checksum."""
    import urllib.request
    import os
    os.makedirs(target_dir, exist_ok=True)
    out = os.path.join(target_dir, "wham_vit.tar")
    print(f"[setup_models] WHAM: downloading {WHAM_CHECKPOINT_URL} → {out}")
    urllib.request.urlretrieve(WHAM_CHECKPOINT_URL, out)
    print(f"[setup_models] WHAM: extracting {out}")
    # TODO(phase2b): tar -xf into target_dir, then delete the .tar.
    # TODO(phase2b): SHA-256 checksum verification.


def _download_smpl(target_dir: str, username: str, password: str) -> None:
    """
    Stub. phase2b fills in the real SMPL/SMPL-X/SMPL-H download.

    SMPL models live behind a registration wall at smpl.is.tue.mpg.de
    and smpl-x.is.tue.mpg.de. Credentials passed via the Modal Secret
    `smpl-research-creds`. Production deployment will require Meshcapade
    commercial sublicense (per spec §7).
    """
    print(
        f"[setup_models] SMPL: download stub — phase2b implements the real "
        f"requests.Session() with form-login at {target_dir}"
    )
    # TODO(phase2b): requests.Session() + form-login + Cookie carry
    # TODO(phase2b): download SMPL_NEUTRAL.pkl, SMPL_FEMALE.pkl,
    #                SMPL_MALE.pkl, SMPLX_NEUTRAL.npz, SMPLH_NEUTRAL.npz
    # TODO(phase2b): assert each file size matches the published checksum


# ---------------------------------------------------------------------------
# Modal function entrypoint — decorated only if modal is importable so
# this file is safe to py_compile in CI without the Modal client.
# ---------------------------------------------------------------------------

if _MODAL_AVAILABLE:
    @app.function(
        volumes={"/models": model_volume},
        secrets=[smpl_credentials],
        timeout=3600,
    )
    def setup_all_models() -> None:
        """
        Run once to populate the swingcue-pilot-models Volume.

        After this completes, all per-library Modal functions can mount
        the Volume read-only and find weights already in place.
        """
        # WHAM weights — no auth needed.
        _download_wham("/models/wham")

        # SMPL models — auth via Modal Secret.
        username = os.environ.get("USERNAME", "")
        password = os.environ.get("PASSWORD", "")
        if not username or not password:
            raise RuntimeError(
                "smpl-research-creds Modal Secret missing USERNAME/PASSWORD. "
                "Create with: modal secret create smpl-research-creds "
                "USERNAME=... PASSWORD=..."
            )
        _download_smpl("/models/smpl", username, password)

        # Verify integrity. phase2b adds real checksum verification.
        listing: list[str] = []
        for root, _, files in os.walk("/models"):
            for f in files:
                full = os.path.join(root, f)
                size_mb = os.path.getsize(full) / 1024 / 1024
                listing.append(f"  {full}  ({size_mb:.1f} MB)")
        print("[setup_models] Volume populated:")
        for line in listing:
            print(line)
        print(f"[setup_models] total entries: {len(listing)}")

        # Volume changes need to commit before the function returns or
        # the next inference function won't see them.
        model_volume.commit()


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
        print(f"[setup_models] WHAM URL : {WHAM_CHECKPOINT_URL}")
        print(f"[setup_models] entrypoint : setup_all_models")
        print(
            "[setup_models] invoke with: "
            "modal run python/pilot/setup_models.py::setup_all_models"
        )
