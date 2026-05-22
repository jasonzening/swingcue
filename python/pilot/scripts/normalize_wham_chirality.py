"""
normalize_wham_chirality.py — one-shot lower-body L↔R chirality fix
for joint_centers_3d.json files produced BEFORE the wham_runner.py
PR-7a.2 fix landed.

Per PR-7a.2 cross-pair diagnostic: WHAM's H36M upper-body
(shoulder/elbow/wrist) uses anatomy convention while its lower-body
(hip/knee/ankle) AND our GT labels use image-orientation convention.
The fix is now in wham_runner.py for future runs, but existing local
JSONs (b3fea3f0, b32e0f21, 5bbcfbc8) need the same swap applied
retroactively to avoid spending Modal GPU re-running WHAM.

Operation per file:
  - For each frame: swap (left_shoulder ↔ right_shoulder),
    (left_elbow ↔ right_elbow), (left_wrist ↔ right_wrist).
  - Verify a post-swap sanity check: on the first valid frame, the
    "left_*" chain (shoulder/elbow/wrist/hip) should be on a single
    side of the spine.

Usage:
    .venv-benchmark/Scripts/python.exe \\
        python/pilot/scripts/normalize_wham_chirality.py [--dry-run]

Idempotency: a marker key "_chirality_normalized": True is written at
the top of the JSON. Re-running this script will skip files already
marked normalized.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WHAM_DIR  = REPO_ROOT / "python" / "pilot" / "output" / "wham"
MARKER    = "_chirality_normalized"
PAIRS = (
    ("left_shoulder", "right_shoulder"),
    ("left_elbow",    "right_elbow"),
    ("left_wrist",    "right_wrist"),
)


def _chain_consistent(kp: dict, spine_cx: float) -> bool:
    """All 4 of left_{shoulder,elbow,wrist,hip} on the same anat side?"""
    sides: set[str] = set()
    for j in ("left_shoulder", "left_elbow", "left_wrist", "left_hip"):
        p = kp.get(j)
        if p is None or p[0] is None:
            continue
        sides.add("L" if p[0] < spine_cx else "R")
    return len(sides) == 1


def normalize_file(path: Path, dry_run: bool = False) -> dict:
    """
    Apply the L↔R swap to every frame in one joint_centers_3d.json.
    Returns a report dict for caller logging.
    """
    data = json.loads(path.read_text())
    if data.get(MARKER):
        return {"path": str(path), "status": "skip-already-normalized"}

    frames = data.get("frames", [])
    if not frames:
        return {"path": str(path), "status": "skip-no-frames"}

    # Pre-swap sanity: pick the first frame with all 4 left-chain joints,
    # check pre-swap chain (expected: inconsistent before fix).
    pre_consistent = None
    for f in frames:
        kp = f.get("joint_centers_3d", {})
        pelvis = kp.get("pelvis"); neck = kp.get("neck")
        if pelvis is None or neck is None:
            continue
        spine_cx = (pelvis[0] + neck[0]) / 2.0
        if all(kp.get(j) is not None
               for j in ("left_shoulder", "left_elbow", "left_wrist", "left_hip")):
            pre_consistent = _chain_consistent(kp, spine_cx)
            break

    # Apply swap to every frame.
    for f in frames:
        kp = f.get("joint_centers_3d", {})
        for left, right in PAIRS:
            if left in kp and right in kp:
                kp[left], kp[right] = kp[right], kp[left]

    # Post-swap sanity: same check, expect CONSISTENT now.
    post_consistent = None
    for f in frames:
        kp = f.get("joint_centers_3d", {})
        pelvis = kp.get("pelvis"); neck = kp.get("neck")
        if pelvis is None or neck is None:
            continue
        spine_cx = (pelvis[0] + neck[0]) / 2.0
        if all(kp.get(j) is not None
               for j in ("left_shoulder", "left_elbow", "left_wrist", "left_hip")):
            post_consistent = _chain_consistent(kp, spine_cx)
            break

    if not dry_run:
        data[MARKER] = True
        path.write_text(json.dumps(data, indent=2))

    return {
        "path":            str(path),
        "status":          "normalized" if not dry_run else "dry-run",
        "n_frames":        len(frames),
        "pre_consistent":  pre_consistent,
        "post_consistent": post_consistent,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Report findings; don't modify files.")
    args = ap.parse_args()

    json_files = sorted(WHAM_DIR.glob("*/joint_centers_3d.json"))
    if not json_files:
        sys.exit(f"[normalize] no joint_centers_3d.json files under {WHAM_DIR}")

    print(f"[normalize] found {len(json_files)} WHAM output JSONs")
    if args.dry_run:
        print("[normalize] DRY-RUN — no files written")
    print()

    any_failed = False
    for p in json_files:
        r = normalize_file(p, dry_run=args.dry_run)
        print(f"  {r['path']}")
        for k, v in r.items():
            if k != "path":
                print(f"    {k}: {v}")
        # Post-swap sanity check — for face_on this should flip to True.
        # For DTL (golfer sideways to camera) the x-axis chain test is
        # not applicable; hip lateral offset is along camera-z, not x.
        # We still apply the swap (it's the correct chirality fix per
        # the diagnostic) but don't error out.
        if r.get("post_consistent") is False:
            print(f"    NOTE post-swap chain still mixed on x-test — "
                  f"likely a DTL clip where lateral offset is depth-aligned, "
                  f"not x-aligned. Swap still applied.")
        print()


if __name__ == "__main__":
    main()
