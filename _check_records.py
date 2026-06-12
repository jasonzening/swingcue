import json, glob, sys
sys.path.insert(0, "/home/jason/projects/swingcue-postest")
files = sorted(glob.glob("engine/layer0/records/*.json"))
for f in files:
    d = json.load(open(f))
    stem = d.get("video_stem", "?")
    verdict = d.get("verdict", "?")
    angle = d.get("angle", "?")
    if "stodownload" in stem or "Videos2026" in stem:
        continue
    print(f"{stem:20s} verdict={verdict:12s} angle={angle}")
