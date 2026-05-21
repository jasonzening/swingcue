"""
ground_truth_labeler.py — interactive matplotlib tool to click 5 anatomical
red-dot ground-truth keypoints per swing phase, save as JSON per
PR-7_GOLF_CORRECTION_LAYER_SPEC_v2.md §9.

Click order (body-frame — golfer's anatomical L/R, NOT screen L/R):
    1. left_shoulder
    2. right_shoulder
    3. left_hip
    4. right_hip
    5. neck_center

Keyboard:
    u   undo last point
    q   quit without saving
    s   save now (only if all 5 collected)

Output JSON (per spec §9):
    {
      "video_id": "...",
      "phase": "setup",
      "frame_idx": 12,
      "video_width": 720,
      "video_height": 1280,
      "labels": {
        "left_shoulder":  {"x": 410, "y": 525},
        "right_shoulder": {"x": 510, "y": 525},
        ...
      }
    }

CLI:
    python -m pilot.scripts.ground_truth_labeler \\
        --video-id b3fea3f0-e248-44d7-a923-0bb43172b5bf \\
        --phase setup \\
        --frame 12 \\
        --image docs/PR-7_GROUND_TRUTH/frames/b3fea3f0_setup_f12.png

Smoke-test mode (no GUI — passes synthetic clicks to verify the save
path works without needing a mouse):
    python -m pilot.scripts.ground_truth_labeler \\
        --video-id ... --phase setup --frame 12 --image ...png \\
        --test-clicks "410,525;510,525;425,720;495,720;460,450"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Click order matters for the user-facing prompt sequence; this is the
# 5-keypoint first-pass set from spec §9.
KEYPOINT_ORDER: tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "neck_center",
)

# Visual styling — colors per SwingCue convention (right=orange,
# left=cyan, midline=grey). Used for the click-marker circles.
KP_COLOR: dict[str, str] = {
    "left_shoulder":  "#3CC8FF",   # cyan
    "left_hip":       "#3CC8FF",
    "right_shoulder": "#FFC83C",   # orange
    "right_hip":      "#FFC83C",
    "neck_center":    "#BCBCBC",   # grey
}

DEFAULT_OUTPUT_DIR = Path("docs/PR-7_GROUND_TRUTH")

# Per spec v3 §7. Bumped from v2 (which didn't have schema_version) to
# explicit v3. Future ground-truth schema changes bump this.
GROUND_TRUTH_SCHEMA_VERSION = "v3"
# Identifies this specific labeler implementation in the saved JSON.
# v1 was the matplotlib labeler (never used in production); v2 is the
# current Tkinter-based labeler. Per spec v3 §7 the schema asks for
# "labeler_v2".
LABELER_VERSION = "labeler_v2"

# Allowed --sport values. Future plugins (tennis, ski, etc per spec
# v3 §11) extend this. Keep golf as the default — first plugin per
# spec v3 §5.
SUPPORTED_SPORTS: tuple[str, ...] = ("golf",)
# Allowed --view values per spec v3 §12. side / back / top are
# explicitly out of MVP scope.
SUPPORTED_VIEWS: tuple[str, ...] = ("face_on", "down_the_line")


def short_id(video_id: str) -> str:
    return video_id.split("-")[0]


def default_output_path(video_id: str, phase: str, sport: str, view: str) -> Path:
    """
    Per spec v3 §7: docs/PR-7_GROUND_TRUTH/<sport>/<short_id>_<phase>_<view>.json
    """
    return (
        DEFAULT_OUTPUT_DIR / sport
        / f"{short_id(video_id)}_{phase}_{view}.json"
    )


def save_labels(
    output_path: Path,
    video_id: str,
    phase: str,
    frame_idx: int,
    video_width: int,
    video_height: int,
    labels: dict[str, tuple[int, int]],
    sport: str = "golf",
    view: str = "face_on",
    schema_version: str = GROUND_TRUTH_SCHEMA_VERSION,
    labeler_version: str = LABELER_VERSION,
) -> None:
    """Write the spec v3 §7 JSON shape to disk."""
    from datetime import datetime, timezone

    payload = {
        "schema_version": schema_version,
        "sport": sport,
        "video_id": video_id,
        "phase": phase,
        "frame_idx": frame_idx,
        "view": view,
        "video_width": video_width,
        "video_height": video_height,
        "labels": {
            name: {"x": int(x), "y": int(y)}
            for name, (x, y) in labels.items()
        },
        "labeler_version": labeler_version,
        # ISO-8601 UTC, second precision. Stamped by the labeler at
        # save time so PR-7b sweep harness can sort labels by recency.
        "labeled_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


class TkLabeler:
    """
    Tkinter Canvas labeler. Replaces a matplotlib version that worked
    elsewhere but on this Windows 11 + matplotlib 3.10 + TkAgg combo
    had `button_press_event` silently dropped (motion + key events
    fired fine, button events never delivered).

    Public surface:
        setup()                           — build root + canvas + bindings;
                                            does NOT call mainloop()
        run() -> dict | None              — full CLI entry; calls setup() +
                                            mainloop(); returns labels on save
        click_at_image_coords(x, y)       — test entry; simulate a click in
                                            native image pixel space without
                                            requiring an event loop
    """

    def __init__(
        self,
        image_path: Path,
        video_id: str,
        phase: str,
        frame_idx: int,
        sport: str = "golf",
        view: str = "face_on",
    ):
        self.image_path = image_path
        self.video_id = video_id
        self.phase = phase
        self.frame_idx = frame_idx
        self.sport = sport
        self.view = view
        self.labels: dict[str, tuple[int, int]] = {}
        self.markers: dict[str, list[int]] = {}  # name -> [oval_id, text_id]
        self.saved: bool = False

        # Lazy state — populated by setup().
        self.root = None
        self.canvas = None
        self.title_label = None
        self.legend_label = None
        self._photo = None  # keep ref so PhotoImage isn't GC'd
        self.W = 0
        self.H = 0
        self.display_w = 0
        self.display_h = 0
        self.scale = 1.0

    def setup(self) -> None:
        """Build root + canvas + image + bindings. No mainloop."""
        import tkinter as tk
        from PIL import Image, ImageTk

        print(
            f"[DEBUG] using Tkinter direct (tk version {tk.TkVersion})",
            flush=True,
        )

        pil_img = Image.open(self.image_path)
        self.W, self.H = pil_img.size

        # Constrain display so the swing portrait fits on a 1080p
        # screen alongside the taskbar.
        MAX_DISPLAY_H = 900
        if self.H > MAX_DISPLAY_H:
            self.scale = MAX_DISPLAY_H / self.H
            self.display_w = int(self.W * self.scale)
            self.display_h = int(self.H * self.scale)
            pil_img = pil_img.resize(
                (self.display_w, self.display_h), Image.LANCZOS,
            )
        else:
            self.scale = 1.0
            self.display_w, self.display_h = self.W, self.H
        print(
            f"[DEBUG] image native {self.W}x{self.H}, "
            f"display {self.display_w}x{self.display_h} "
            f"(scale {self.scale:.3f})", flush=True,
        )

        self.root = tk.Tk()
        self.root.title(
            f"PR-7 labeler — {self.sport}/{self.view}  "
            f"{self.video_id[:8]} {self.phase} f{self.frame_idx}"
        )

        # Layout: title across top, image canvas + legend side-by-side
        # below. Legend on the right (saves horizontal space on portrait).
        self.title_label = tk.Label(
            self.root, text="", font=("Segoe UI", 12, "bold"),
            fg="black", bg="white", anchor="w", padx=10, pady=6,
        )
        self.title_label.pack(fill="x", side="top")

        main = tk.Frame(self.root)
        main.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            main, width=self.display_w, height=self.display_h,
            highlightthickness=0, bd=0, bg="black",
        )
        self.canvas.pack(side="left")

        self._photo = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

        self.legend_label = tk.Label(
            main, text="", font=("Consolas", 10), fg="black", bg="#F0F0F0",
            anchor="nw", justify="left", padx=10, pady=10,
        )
        self.legend_label.pack(side="left", fill="both", expand=True)

        # Bind events. <Button-1> fires reliably on Tk Canvas — the
        # whole reason we abandoned matplotlib for the interactive path.
        self.canvas.bind("<Button-1>", self._on_click)
        # Key bindings have to attach to root (canvas needs focus to
        # receive keys otherwise). bind_all catches everything.
        self.root.bind_all("<KeyPress-u>", lambda e: self._undo())
        self.root.bind_all("<KeyPress-s>", lambda e: self._save())
        self.root.bind_all("<KeyPress-q>", lambda e: self._quit())
        self.canvas.focus_set()
        print(
            "[DEBUG] Tk bindings: canvas<Button-1>, root[KeyPress-u/s/q]",
            flush=True,
        )

        self._refresh_text()

    def _next_target(self) -> str | None:
        for name in KEYPOINT_ORDER:
            if name not in self.labels:
                return name
        return None

    def _refresh_text(self) -> None:
        target = self._next_target()
        if target is None:
            self.title_label.config(
                text="ALL 5 COLLECTED — press [s] save, [u] undo, [q] quit",
                fg="#0A8A00",
            )
        else:
            color = KP_COLOR.get(target, "#000000")
            self.title_label.config(
                text=(
                    f"{self.sport}/{self.view}  "
                    f"{self.video_id[:8]}  {self.phase}  frame={self.frame_idx}"
                    f"   →   click  {target}"
                ),
                fg=color,
            )

        lines = ["Body-frame convention", "(golfer's anatomical L/R):", ""]
        for i, name in enumerate(KEYPOINT_ORDER, 1):
            if name in self.labels:
                x, y = self.labels[name]
                lines.append(f"  [{i}] ✓ {name:<16} ({x:>4d}, {y:>4d})")
            else:
                lines.append(f"  [{i}]   {name}")
        lines.extend([
            "",
            "Keys:",
            "  [u]  undo last point",
            "  [s]  save (needs 5)",
            "  [q]  quit, no save",
        ])
        if self.legend_label is not None:
            self.legend_label.config(text="\n".join(lines))

    def _draw_marker(self, name: str, img_x: int, img_y: int) -> None:
        """Draw oval + label at display-pixel position derived from image
        coords. Marker IDs recorded in self.markers for undo."""
        color = KP_COLOR.get(name, "#FFFFFF")
        dx = img_x * self.scale
        dy = img_y * self.scale
        r = 7
        oval_id = self.canvas.create_oval(
            dx - r, dy - r, dx + r, dy + r,
            fill=color, outline="black", width=1.5,
        )
        text_id = self.canvas.create_text(
            dx + 10, dy - 10, text=name, anchor="w",
            fill=color, font=("Segoe UI", 9, "bold"),
        )
        self.markers[name] = [oval_id, text_id]

    def _erase_marker(self, name: str) -> None:
        ids = self.markers.pop(name, None)
        if ids:
            for cid in ids:
                self.canvas.delete(cid)

    def _record_click(self, img_x: int, img_y: int) -> str | None:
        """
        Append a click at native image coords. Returns the joint name
        used, or None if all 5 already collected.
        """
        target = self._next_target()
        if target is None:
            print("[labeler] all 5 collected; press [s] to save", flush=True)
            return None
        x, y = int(img_x), int(img_y)
        self.labels[target] = (x, y)
        self._draw_marker(target, x, y)
        print(f"[labeler] [{len(self.labels)}/5] {target} = ({x}, {y})", flush=True)
        self._refresh_text()
        return target

    def _on_click(self, event) -> None:
        # event.x / event.y are canvas (display) pixels. Convert back
        # to native image coords (the JSON output convention).
        print(
            f"[DEBUG] click: canvas=({event.x}, {event.y}), "
            f"image=({event.x / self.scale:.1f}, {event.y / self.scale:.1f})",
            flush=True,
        )
        img_x = int(round(event.x / self.scale))
        img_y = int(round(event.y / self.scale))
        # Clamp inside image bounds.
        img_x = max(0, min(self.W - 1, img_x))
        img_y = max(0, min(self.H - 1, img_y))
        self._record_click(img_x, img_y)

    def _undo(self) -> None:
        if not self.labels:
            print("[labeler] nothing to undo", flush=True)
            return
        last_name = list(self.labels.keys())[-1]
        self.labels.pop(last_name)
        self._erase_marker(last_name)
        print(f"[labeler] undid {last_name}", flush=True)
        self._refresh_text()

    def _save(self) -> None:
        if self._next_target() is not None:
            missing = [n for n in KEYPOINT_ORDER if n not in self.labels]
            print(f"[labeler] cannot save — missing {missing}", flush=True)
            return
        print(f"[labeler] save requested ({len(self.labels)} labels)", flush=True)
        self.saved = True
        if self.root is not None:
            self.root.quit()

    def _quit(self) -> None:
        print("[labeler] quit (no save)", flush=True)
        self.saved = False
        if self.root is not None:
            self.root.quit()

    # ── Public test entry point ─────────────────────────────────────
    def click_at_image_coords(self, img_x: int, img_y: int) -> str | None:
        """
        Synthesize a click at native image coords. Used by tests to
        verify the labeler without needing a GUI or event loop.
        """
        return self._record_click(int(img_x), int(img_y))

    # ── Public CLI entry point ──────────────────────────────────────
    def run(self) -> dict[str, tuple[int, int]] | None:
        self.setup()
        print(
            "[DEBUG] entering Tk mainloop — window should appear now",
            flush=True,
        )
        try:
            self.root.mainloop()
        finally:
            try:
                self.root.destroy()
            except Exception:
                pass
        print(
            f"[DEBUG] mainloop returned — saved={self.saved}, "
            f"labels collected: {len(self.labels)}/5",
            flush=True,
        )
        if not self.saved or self._next_target() is not None:
            return None
        return self.labels


def _interactive_label(
    image_path: Path,
    video_id: str,
    phase: str,
    frame_idx: int,
    output_path: Path,
    sport: str = "golf",
    view: str = "face_on",
) -> dict[str, tuple[int, int]] | None:
    """Wrapper around TkLabeler.run() for symmetry with main()."""
    return TkLabeler(
        image_path, video_id, phase, frame_idx, sport=sport, view=view,
    ).run()


def _test_label(test_clicks: str) -> dict[str, tuple[int, int]]:
    """
    Parse synthetic clicks for headless smoke testing.
    Format: "x1,y1;x2,y2;x3,y3;x4,y4;x5,y5" in the keypoint order
    (left_shoulder; right_shoulder; left_hip; right_hip; neck_center).
    """
    pairs = [p.strip() for p in test_clicks.split(";") if p.strip()]
    if len(pairs) != len(KEYPOINT_ORDER):
        raise SystemExit(
            f"--test-clicks needs exactly {len(KEYPOINT_ORDER)} pairs, "
            f"got {len(pairs)}"
        )
    out: dict[str, tuple[int, int]] = {}
    for name, pair in zip(KEYPOINT_ORDER, pairs):
        xs, ys = pair.split(",")
        out[name] = (int(xs), int(ys))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--phase", required=True,
                    choices=["setup", "top", "transition", "impact", "finish"])
    ap.add_argument("--frame", required=True, type=int, help="frame_idx in source video")
    ap.add_argument("--image", required=True, type=Path,
                    help="extracted frame PNG (from extract_phase_frames.py)")
    # Per spec v3 §3 + §9 + §12: sport + view are required envelope
    # fields. Sport defaults to golf (first plugin, spec v3 §5); view
    # is required so labeler never silently mislabels a face_on as
    # down_the_line or vice versa.
    ap.add_argument("--sport", default="golf", choices=list(SUPPORTED_SPORTS),
                    help="sport plugin name (default golf)")
    ap.add_argument("--view", required=True, choices=list(SUPPORTED_VIEWS),
                    help="camera view: face_on or down_the_line")
    ap.add_argument("--output", type=Path, default=None,
                    help="defaults to docs/PR-7_GROUND_TRUTH/<sport>/<short_id>_<phase>_<view>.json")
    ap.add_argument("--test-clicks", default=None,
                    help='headless smoke-test: "x1,y1;x2,y2;x3,y3;x4,y4;x5,y5"')
    args = ap.parse_args()

    if not args.image.exists():
        raise SystemExit(f"image file not found: {args.image}")

    output_path = args.output or default_output_path(
        args.video_id, args.phase, args.sport, args.view,
    )

    # Pull video dims from the image (the extracted frame matches the
    # source video resolution).
    try:
        from PIL import Image
        with Image.open(args.image) as im:
            img_w, img_h = im.size
    except ImportError:
        # Fallback: use cv2 if Pillow isn't around (it is, via opencv-python).
        import cv2 as _cv2
        img = _cv2.imread(str(args.image))
        if img is None:
            raise SystemExit(f"could not read image: {args.image}")
        img_h, img_w = img.shape[:2]

    if args.test_clicks is not None:
        labels = _test_label(args.test_clicks)
        print(f"[labeler] TEST MODE — synthetic clicks: {labels}")
    else:
        result = _interactive_label(
            args.image, args.video_id, args.phase, args.frame, output_path,
            sport=args.sport, view=args.view,
        )
        if result is None:
            print("[labeler] no labels collected — exit without save")
            return
        labels = result

    save_labels(
        output_path,
        video_id=args.video_id,
        phase=args.phase,
        frame_idx=args.frame,
        video_width=img_w,
        video_height=img_h,
        labels=labels,
        sport=args.sport,
        view=args.view,
    )
    sz_b = output_path.stat().st_size
    print(f"[labeler] wrote {output_path} ({sz_b} bytes, {len(labels)} keypoints)")


if __name__ == "__main__":
    sys.exit(main())
