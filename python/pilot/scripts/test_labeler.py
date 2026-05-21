"""
test_labeler.py — automated headless self-diagnosis of the labeler.

Walks the labeler through every stage that a real user click would
trigger, without needing a GUI or human round-trip. Stages mirror
Jason's diagnostic table for the matplotlib version but adapted to
the Tkinter implementation that replaced it.

CC runs this when changing the labeler. PASS-pass-pass means the
interactive version is safe to hand to Jason.

Run:
    ./.venv-benchmark/Scripts/python.exe \\
        python/pilot/scripts/test_labeler.py

Note: this DOES instantiate a Tk root + canvas, because the labeler's
markers are stored on the canvas. Tk on Windows can do this without
displaying the window if mainloop() isn't called. Net effect: ~1 sec
runtime, no visible window, no clicks needed.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Import labeler module — must work without GUI.
from ground_truth_labeler import (
    KEYPOINT_ORDER,
    KP_COLOR,
    TkLabeler,
    save_labels,
    short_id,
    default_output_path,
)


# Test fixtures — real frame from b3fea3f0 setup phase.
TEST_IMAGE = Path("docs/PR-7_GROUND_TRUTH/frames/b3fea3f0_setup_f007.png")
TEST_VIDEO_ID = "b3fea3f0-e248-44d7-a923-0bb43172b5bf"
TEST_PHASE = "setup"
TEST_FRAME = 7

# Synthetic click positions for the 5 keypoints (real WHAM frame-63
# pixel coords from the prior diagnostic).
TEST_CLICKS: list[tuple[str, tuple[int, int]]] = [
    ("left_shoulder",  (486, 492)),
    ("right_shoulder", (399, 494)),
    ("left_hip",       (399, 599)),
    ("right_hip",      (485, 606)),
    ("neck_center",    (437, 457)),
]


_passes = 0
_fails: list[str] = []


def _assert(stage: str, cond: bool, detail: str) -> None:
    """Print pass/fail line; record failures so we can report a tally."""
    global _passes
    marker = "PASS" if cond else "FAIL"
    print(f"  [{stage}] {marker}: {detail}")
    if cond:
        _passes += 1
    else:
        _fails.append(f"{stage}: {detail}")


def run_tests() -> int:
    print("=" * 64)
    print(" ground_truth_labeler.py headless self-test")
    print("=" * 64)

    if not TEST_IMAGE.exists():
        print(f"[ABORT] fixture image missing: {TEST_IMAGE}")
        print(
            f"        run python/pilot/scripts/extract_phase_frames.py first"
        )
        return 2

    # ── Stage A: construct + setup ──────────────────────────────────
    print()
    print("[Stage A] construct TkLabeler + setup() (no mainloop)")
    labeler = TkLabeler(
        image_path=TEST_IMAGE,
        video_id=TEST_VIDEO_ID,
        phase=TEST_PHASE,
        frame_idx=TEST_FRAME,
    )
    labeler.setup()
    _assert("A1", labeler.root is not None, "root window created")
    _assert("A2", labeler.canvas is not None, "canvas created")
    _assert("A3", labeler.W == 720 and labeler.H == 1280,
            f"image dims = {labeler.W}x{labeler.H} (expected 720x1280)")
    _assert("A4", labeler._photo is not None, "PhotoImage holding image data")
    _assert("A5", labeler.canvas.find_all() != (),
            f"canvas has items (image must be drawn) — found {len(labeler.canvas.find_all())}")

    # ── Stage B: click handler dispatches via public test entry ─────
    print()
    print("[Stage B] synthesize 5 clicks via click_at_image_coords()")
    for expected_name, (x, y) in TEST_CLICKS:
        got_name = labeler.click_at_image_coords(x, y)
        _assert(
            f"B-{expected_name}",
            got_name == expected_name,
            f"got_name={got_name!r}, expected={expected_name!r}",
        )

    # ── Stage C: labels dict updated ────────────────────────────────
    print()
    print("[Stage C] verify labels dict")
    _assert("C1", len(labeler.labels) == 5,
            f"label count = {len(labeler.labels)}")
    for name, (x, y) in TEST_CLICKS:
        actual = labeler.labels.get(name)
        _assert(f"C-{name}", actual == (x, y), f"labels[{name}]={actual}")

    # ── Stage D: canvas markers drawn ───────────────────────────────
    print()
    print("[Stage D] verify markers on canvas")
    _assert("D1", len(labeler.markers) == 5,
            f"markers dict has {len(labeler.markers)} entries")
    # Each marker is [oval_id, text_id]. Sum should be 10 IDs.
    total_marker_items = sum(len(v) for v in labeler.markers.values())
    _assert("D2", total_marker_items == 10,
            f"total canvas marker items = {total_marker_items}")
    # Image (id=1) + 5 ovals + 5 texts = 11 canvas items.
    canvas_items = labeler.canvas.find_all()
    _assert("D3", len(canvas_items) >= 11,
            f"canvas total items = {len(canvas_items)} (expected ≥ 11)")

    # ── Stage E: bindings registered ────────────────────────────────
    print()
    print("[Stage E] verify event bindings")
    canvas_bindings = labeler.canvas.bind()
    _assert("E1", "<Button-1>" in canvas_bindings,
            f"canvas <Button-1> bound (have: {canvas_bindings})")
    # `root.bind_all()` (no args) in Tkinter doesn't enumerate the
    # all-binding sequences cleanly. Use Tk's native `bind all` query
    # which returns the actual list as a space-separated string.
    bound_all = labeler.root.tk.eval("bind all").split()
    # Tk normalises `<KeyPress-q>` to just `q` for single-char key
    # bindings. Accept either form.
    for key in ("u", "s", "q"):
        accepted = key in bound_all or f"<KeyPress-{key}>" in bound_all
        _assert(f"E-{key}", accepted,
                f"root <KeyPress-{key}> bound (bound_all={bound_all})")

    # ── Stage F: undo round-trips ───────────────────────────────────
    print()
    print("[Stage F] undo → re-add cycle")
    initial_labels = dict(labeler.labels)
    labeler._undo()
    _assert("F1", len(labeler.labels) == 4,
            f"after undo: {len(labeler.labels)} labels (expected 4)")
    _assert("F2", "neck_center" not in labeler.labels,
            "neck_center (last) removed by undo")
    _assert("F3", len(labeler.markers) == 4,
            f"markers count after undo: {len(labeler.markers)}")
    # Re-add via click_at to restore state for next stage.
    labeler.click_at_image_coords(437, 457)
    _assert("F4", labeler.labels.get("neck_center") == (437, 457),
            "neck_center re-added")

    # ── Stage G: save() round-trip ──────────────────────────────────
    print()
    print("[Stage G] save_labels() JSON shape matches spec §9")
    with tempfile.NamedTemporaryFile(
        suffix=".json", mode="w", delete=False, encoding="utf-8",
    ) as tf:
        out_path = Path(tf.name)
    try:
        save_labels(
            out_path,
            video_id=labeler.video_id,
            phase=labeler.phase,
            frame_idx=labeler.frame_idx,
            video_width=labeler.W,
            video_height=labeler.H,
            labels=labeler.labels,
        )
        _assert("G1", out_path.exists(), f"JSON written to {out_path}")
        payload = json.loads(out_path.read_text())
        _assert("G2", payload.get("video_id") == TEST_VIDEO_ID,
                f"video_id = {payload.get('video_id')!r}")
        _assert("G3", payload.get("phase") == TEST_PHASE,
                f"phase = {payload.get('phase')!r}")
        _assert("G4", payload.get("frame_idx") == TEST_FRAME,
                f"frame_idx = {payload.get('frame_idx')}")
        _assert("G5", payload.get("video_width") == 720
                and payload.get("video_height") == 1280,
                f"video_width/height = "
                f"{payload.get('video_width')}/{payload.get('video_height')}")
        labels_block = payload.get("labels", {})
        _assert("G6", set(labels_block.keys()) == set(KEYPOINT_ORDER),
                f"labels keys = {sorted(labels_block.keys())}")
        for name, (x, y) in TEST_CLICKS:
            entry = labels_block.get(name, {})
            _assert(f"G-{name}",
                    entry.get("x") == x and entry.get("y") == y,
                    f"labels[{name}] = {entry}")
    finally:
        out_path.unlink(missing_ok=True)

    # Cleanup — destroy root before we exit so Python can GC the
    # Tk interpreter; otherwise the process hangs on exit (Tk + Python
    # finalisation order on Windows).
    try:
        labeler.root.destroy()
    except Exception:
        pass

    # ── Tally ───────────────────────────────────────────────────────
    print()
    print("=" * 64)
    if _fails:
        print(f" FAIL — {len(_fails)} failure(s) of {_passes + len(_fails)} checks")
        for f in _fails:
            print(f"   - {f}")
        return 1
    print(f" PASS — {_passes} checks all green")
    print("=" * 64)
    print(" Interactive labeler is safe to hand to Jason.")
    return 0


if __name__ == "__main__":
    sys.exit(run_tests())
