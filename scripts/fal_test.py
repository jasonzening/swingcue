"""
Quick test: send an image to fal-ai/sam-3/3d-body, print full response.

Setup (one time):
  pip install fal-client

Run:
  $env:FAL_KEY = "your-fal-key"
  python scripts/fal_test.py test_swing_finish.png
"""
import json
import os
import sys

import fal_client

if not os.environ.get("FAL_KEY"):
    sys.exit("ERROR: FAL_KEY env var not set. Run:  $env:FAL_KEY = \"your-key\"")

img_path = sys.argv[1] if len(sys.argv) > 1 else None
if not img_path or not os.path.exists(img_path):
    sys.exit(f"ERROR: Image not found. Usage: python {sys.argv[0]} path/to/image.png")

print(f"Step 1/3: Uploading {img_path} to fal storage...")
image_url = fal_client.upload_file(img_path)
print(f"  ✓ {image_url}")

print("\nStep 2/3: Submitting to fal-ai/sam-3/3d-body (typically 5-10s)...")


def on_update(update):
    if hasattr(update, "status"):
        print(f"  [{update.status}]")
    elif hasattr(update, "logs") and update.logs:
        for log in update.logs:
            print(f"  log: {log.get('message', log)}")


result = fal_client.subscribe(
    "fal-ai/sam-3/3d-body",
    arguments={
        "image_url": image_url,
        "export_meshes": True,
        "include_3d_keypoints": True,
    },
    with_logs=True,
    on_queue_update=on_update,
)

print("\nStep 3/3: Done. Saving full response to fal_result.json...")
with open("fal_result.json", "w") as f:
    json.dump(result, f, indent=2)

# Print summary
print("\n=== RESPONSE STRUCTURE ===")
print(f"Top-level keys: {list(result.keys())}")

if "metadata" in result and "people" in result["metadata"]:
    person = result["metadata"]["people"][0]
    print(f"\nperson[0] keys: {list(person.keys())}")
    if "keypoints_3d" in person:
        kp = person["keypoints_3d"]
        print(f"  keypoints_3d type: {type(kp).__name__}")
        if isinstance(kp, list):
            print(f"  keypoints_3d length: {len(kp)}")
            if len(kp) > 0:
                print(f"  keypoints_3d[0] sample: {kp[0]}")
        elif isinstance(kp, dict):
            print(f"  keypoints_3d names: {list(kp.keys())[:20]}")
    if "focal_length" in person:
        print(f"  focal_length: {person['focal_length']}")

if "model_glb" in result:
    print(f"\nmodel_glb URL: {result['model_glb']}")

print("\nFull JSON saved to: fal_result.json")
print("Open it and paste the whole file to Claude.")
