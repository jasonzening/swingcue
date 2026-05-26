"""
render_from_db.py — PR-8c.1 MP4 verification by reading WHAM data
straight from Supabase wham_pose_timeline rows. Validates the full
write-path end-to-end: Modal → Supabase → DB → render.

Usage:
    .venv-pilot/Scripts/python.exe docs/PR-8c_VISUAL_VERIFY/render_from_db.py <video_id>

Outputs:
    docs/PR-8c_VISUAL_VERIFY/<video_id>/
        overlay_full.mp4
        f000_side_by_side.png
        f<mid>_side_by_side.png
        f<end>_side_by_side.png
        render_log.txt
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import httpx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_env_local(path: Path) -> None:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _supa_get(supa_url, key, table, params):
    r = httpx.get(
        f"{supa_url}/rest/v1/{table}",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
        params=params, timeout=60,
    )
    r.raise_for_status()
    return r.json()


def _fetch_video(supa_url, key, video_id) -> Path:
    """Sign storage URL + download to temp."""
    rows = _supa_get(supa_url, key, "swing_videos",
                     {"id": f"eq.{video_id}", "select": "storage_path"})
    storage_path = rows[0]["storage_path"]
    sign = httpx.post(
        f"{supa_url}/storage/v1/object/sign/swing-videos/{storage_path}",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
        json={"expiresIn": 1200}, timeout=30,
    )
    sign.raise_for_status()
    signed = f"{supa_url}/storage/v1{sign.json()['signedURL']}"
    out_path = Path(tempfile.gettempdir()) / f"pr8c_{video_id}.mp4"
    print(f"  downloading video → {out_path.name}")
    with httpx.stream("GET", signed, timeout=120) as resp:
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=1024 * 256):
                f.write(chunk)
    return out_path


def _fetch_wham_data(supa_url, key, video_id):
    """Return (meta, frames_sorted_by_idx)."""
    meta_rows = _supa_get(supa_url, key, "wham_video_meta",
                           {"video_id": f"eq.{video_id}", "select": "*"})
    if not meta_rows:
        sys.exit(f"no wham_video_meta row for {video_id}")
    meta = meta_rows[0]
    # paginate pose_timeline: PostgREST defaults to 1000 rows per req.
    frames: list[dict] = []
    offset = 0
    PAGE = 1000
    while True:
        page = _supa_get(supa_url, key, "wham_pose_timeline", {
            "video_id": f"eq.{video_id}",
            "select":   "frame_idx,frame_timestamp_ms,fit_ok,keypoints_2d_projected",
            "order":    "frame_idx.asc",
            "limit":    str(PAGE),
            "offset":   str(offset),
        })
        if not page:
            break
        frames.extend(page)
        if len(page) < PAGE:
            break
        offset += PAGE
    return meta, frames


# Skeleton edges (same convention as PR-8b v4_z_stabilized render).
PRIMARY_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_hip",      "right_hip"),
    ("neck",          "pelvis"),
]
SECONDARY_EDGES = [
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
]
GREEN_SEC  = (80, 220, 80)
GREEN_PRI  = (0, 255, 100)
MAGENTA    = (255, 0, 255)
WHITE      = (255, 255, 255)


def _to_pt(p): return (int(round(p["x"])), int(round(p["y"])))


def _draw_skeleton(canvas, kp2d: dict):
    for a, b in SECONDARY_EDGES:
        if kp2d.get(a) and kp2d.get(b):
            cv2.line(canvas, _to_pt(kp2d[a]), _to_pt(kp2d[b]), GREEN_SEC, 3)
    for a, b in PRIMARY_EDGES:
        if kp2d.get(a) and kp2d.get(b):
            cv2.line(canvas, _to_pt(kp2d[a]), _to_pt(kp2d[b]), GREEN_PRI, 5)
    for name, p in kp2d.items():
        if name == "head_crown" or p is None:
            continue
        cv2.circle(canvas, _to_pt(p), 7, GREEN_SEC, -1)
        cv2.circle(canvas, _to_pt(p), 8, (0, 0, 0), 1)
    crown = kp2d.get("head_crown")
    head = kp2d.get("head")
    if crown and head:
        cv2.line(canvas, _to_pt(head), _to_pt(crown), MAGENTA, 1)
        cv2.circle(canvas, _to_pt(crown), 10, MAGENTA, -1)
        cv2.circle(canvas, _to_pt(crown), 11, (0, 0, 0), 1)


def _render(video_id: str) -> int:
    _load_env_local(REPO_ROOT / ".env.local")
    supa_url = os.environ["NEXT_PUBLIC_SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

    out_dir = REPO_ROOT / "docs/PR-8c_VISUAL_VERIFY" / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[render_from_db] video_id={video_id}")
    print(f"  fetching WHAM data ...")
    meta, frames = _fetch_wham_data(supa_url, key, video_id)
    print(f"  meta: {meta.get('image_width')}x{meta.get('image_height')} "
          f"{meta.get('frame_count')} frames @{meta.get('processed_fps')}fps "
          f"status={meta.get('status')}")
    print(f"  fetched {len(frames)} pose_timeline rows from DB")

    video_path = _fetch_video(supa_url, key, video_id)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  video: {w}x{h} {n_total} frames @{fps}fps")

    frames_by_idx = {f["frame_idx"]: f.get("keypoints_2d_projected") or {} for f in frames}

    # Pick 3 key sample frames for side-by-side PNGs.
    sample_indices = [
        max(0, n_total // 8),
        n_total // 2,
        min(n_total - 1, n_total * 7 // 8),
    ]
    sample_indices = sorted(set(i for i in sample_indices if i in frames_by_idx))

    cap2 = cv2.VideoCapture(str(video_path))
    for fi in sample_indices:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap2.read()
        if not ok:
            continue
        bare = frame.copy()
        annotated = frame.copy()
        _draw_skeleton(annotated, frames_by_idx.get(fi, {}))
        cv2.rectangle(bare, (0, 0), (bare.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(bare, f"raw video  f={fi}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 1)
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(annotated, f"WHAM-from-DB  f={fi}  crown=magenta",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN_PRI, 1)
        sbs = np.hstack([bare, annotated])
        out_path = out_dir / f"f{fi:03d}_side_by_side.png"
        cv2.imwrite(str(out_path), sbs)
        print(f"  wrote {out_path.name} {sbs.shape[1]}x{sbs.shape[0]}")
    cap2.release()

    # Full overlay MP4 from DB rows.
    cap3 = cv2.VideoCapture(str(video_path))
    mp4_path = out_dir / "overlay_full.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, (w, h))
    fi = 0
    while True:
        ok, frame = cap3.read()
        if not ok:
            break
        canvas = frame.copy()
        kp = frames_by_idx.get(fi, {})
        _draw_skeleton(canvas, kp)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(canvas, f"f={fi:03d}  WHAM-from-DB (post-PR-8c.1)",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
        writer.write(canvas)
        fi += 1
    cap3.release()
    writer.release()
    print(f"  wrote {mp4_path.name} ({fi} frames, {mp4_path.stat().st_size//1024} KB)")

    log = []
    log.append(f"# render_from_db — {video_id}")
    log.append(f"# meta: {json.dumps({k: meta.get(k) for k in ['status','image_width','image_height','processed_fps','frame_count','wham_model']}, indent=2)}")
    jim = meta.get("joint_index_mapping") or {}
    log.append(f"# joint_index_mapping audit:")
    for k in ["head_crown_vertex_index", "_trans_z_stabilization", "_trans_z_median_value_m", "_wham_commit"]:
        log.append(f"#   {k}: {jim.get(k)}")
    log.append(f"# sample frames rendered: {sample_indices}")
    log.append("")
    for fi in sample_indices:
        kp = frames_by_idx.get(fi, {})
        log.append(f"=== frame {fi} ===")
        for name in ["left_shoulder", "right_shoulder", "left_hip", "right_hip",
                      "head", "head_crown", "neck", "pelvis"]:
            p = kp.get(name)
            log.append(f"  {name:18}: {p}")
    (out_dir / "render_log.txt").write_text("\n".join(log), encoding="utf-8")
    print(f"  DONE → {out_dir}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: render_from_db.py <video_id>")
    sys.exit(_render(sys.argv[1]))
