import json
from PIL import Image, ImageDraw, ImageFont

with open("fal_result.json") as f:
    result = json.load(f)
person = result["metadata"]["people"][0]
kp2d = person["keypoints_2d"]
print(f"Found {len(kp2d)} keypoints, sample: {kp2d[0]}")

img = Image.open("test_finish.png").convert("RGB")
W, H = img.size
draw = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("arial.ttf", 14)
except (OSError, IOError):
    font = ImageFont.load_default()

for idx, kp in enumerate(kp2d):
    if not (isinstance(kp, (list, tuple)) and len(kp) >= 2):
        continue
    x, y = kp[0], kp[1]
    if 0 <= x <= 1 and 0 <= y <= 1:
        x, y = x * W, y * H
    x, y = int(x), int(y)
    if not (0 <= x < W and 0 <= y < H):
        continue
    r = 5
    draw.ellipse([x - r, y - r, x + r, y + r], fill="red", outline="white", width=2)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            draw.text((x + 8 + dx, y - 7 + dy), str(idx), font=font, fill="black")
    draw.text((x + 8, y - 7), str(idx), font=font, fill="yellow")

img.save("inspected.png")
print("Saved inspected.png")
