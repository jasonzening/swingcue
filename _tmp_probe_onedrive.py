import subprocess, os, json
from pathlib import Path

onedrive = Path("/mnt/c/Users/jason/OneDrive/Documents")
files = sorted(onedrive.glob("stodownload*.mp4"))
print(f"Found {len(files)} files")
for f in files:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", str(f)],
            capture_output=True, text=True, timeout=10
        )
        if result.stdout:
            d = json.loads(result.stdout)
            vs = [s for s in d.get("streams", []) if s.get("codec_type") == "video"]
            if vs:
                s = vs[0]
                w, h = s.get("width", 0), s.get("height", 0)
                nb = s.get("nb_frames", "?")
                fps_str = s.get("r_frame_rate", "0/1")
                a, b = fps_str.split("/")
                fps = float(a)/float(b) if float(b) > 0 else 0
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"  {f.name}: {w}x{h} {nb}fr @{fps:.0f}fps ({size_mb:.1f}MB)")
    except Exception as e:
        print(f"  {f.name}: ERROR {e}")
