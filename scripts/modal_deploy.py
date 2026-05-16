# NOTE: DEPRECATED 2026-05-16. SwingCue switched from Modal-hosted
# SAM 3D Body to fal.ai hosted (fal-ai/sam-3/3d-body). This file is
# kept for historical reference of the abandoned Modal deployment path.
"""
SwingCue PR-2A: SAM-Body4D on Modal (verification harness)

Three commands, run in order:

  1. Build image (15-25 min first time, cached after):
     modal run scripts/modal_deploy.py::build

  2. Download checkpoints to Modal Volume (~5-10 min, one-time):
     modal run scripts/modal_deploy.py::setup

  3. Test inference on a real swing video (~2 min):
     modal run scripts/modal_deploy.py::test --video test_swing.mp4

Notes:
- Diffusion-VAS occlusion refinement is DISABLED (cost control).
  SAM 3 video tracking + SAM 3D Body per-frame still provides temporal consistency.
- A10G GPU (24 GB VRAM) is used. SAM-Body4D resources.md shows 1-person, 100-frame,
  completion-off runs peak at ~15 GB, fitting comfortably.
- HF_TOKEN must already exist as Modal Secret (see prep step 5).
"""
import modal
import os

app = modal.App("swingcue-pose-3d")

# CUDA devel base needed for detectron2 source compilation
image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install(
        "git", "wget", "ffmpeg",
        "libgl1", "libglib2.0-0",
        "build-essential", "ninja-build",
    )
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        "torchaudio==2.7.1",
        extra_index_url="https://download.pytorch.org/whl/cu118",
    )
    .run_commands(
        # Clone SAM-Body4D
        "git clone https://github.com/gaomingqi/sam-body4d.git /opt/sam-body4d",
        # detectron2 source compile (no prebuilt wheel for torch 2.7 + cu118)
        # TORCH_CUDA_ARCH_LIST covers A10G (8.6), L40S (8.9), H100 (9.0)
        "TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9;9.0' "
        "pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' "
        "--no-build-isolation --no-deps",
        # SAM3 vendored copy
        "cd /opt/sam-body4d && pip install -e models/sam3",
        # SAM-Body4D itself + its declared deps
        "cd /opt/sam-body4d && pip install -e .",
    )
    .pip_install("huggingface_hub")
)

# Persistent volume: ~15-20 GB total (5 checkpoints + writable SAM-Body4D copy for configs)
vol = modal.Volume.from_name("swingcue-checkpoints", create_if_missing=True)


# ------------------------------------------------------------------
# Step 1: smoke test (verifies image build + imports + CLI accessible)
# ------------------------------------------------------------------
@app.function(image=image, gpu="A10G", timeout=120)
def smoke():
    import torch
    import subprocess

    print(f"Python: {os.popen('python --version').read().strip()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\n--- offline_app.py --help ---")
    r = subprocess.run(
        ["python", "scripts/offline_app.py", "--help"],
        cwd="/opt/sam-body4d", capture_output=True, text=True,
    )
    print(r.stdout or r.stderr)

    print("\n--- setup.py --help ---")
    r2 = subprocess.run(
        ["python", "scripts/setup.py", "--help"],
        cwd="/opt/sam-body4d", capture_output=True, text=True,
    )
    print(r2.stdout or r2.stderr)

    return {
        "ok": r.returncode == 0 and r2.returncode == 0,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
    }


# ------------------------------------------------------------------
# Step 2: download checkpoints + patch config to disable completion
# ------------------------------------------------------------------
@app.function(
    image=image, gpu="A10G", timeout=1800,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def setup_checkpoints():
    import subprocess
    import shutil
    from huggingface_hub import login

    login(token=os.environ["HF_TOKEN"])

    sentinel = "/vol/.setup_done_v1"
    if os.path.exists(sentinel):
        print("✓ Already set up. Delete /vol/.setup_done_v1 to re-run.")
        return {"status": "skip"}

    # Copy SAM-Body4D to volume so configs/ is writable across containers
    workdir = "/vol/sam-body4d"
    if not os.path.exists(workdir):
        shutil.copytree("/opt/sam-body4d", workdir)
        print(f"✓ Copied SAM-Body4D → {workdir}")

    ckpt_root = "/vol/checkpoints"
    os.makedirs(ckpt_root, exist_ok=True)

    print(f"Running setup.py --ckpt-root {ckpt_root} ...")
    r = subprocess.run(
        ["python", "scripts/setup.py", "--ckpt-root", ckpt_root],
        cwd=workdir, capture_output=True, text=True,
    )
    print("--- stdout (tail) ---")
    print(r.stdout[-2500:])
    if r.stderr.strip():
        print("--- stderr (tail) ---")
        print(r.stderr[-2500:])

    if r.returncode != 0:
        raise RuntimeError(f"setup.py failed with exit {r.returncode}")

    # Patch config: disable Diffusion-VAS completion module
    config_path = f"{workdir}/configs/body4d.yaml"
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = f.read()
        original = cfg
        # Try several likely YAML patterns; harmless if no match
        for old, new in [
            ("completion:\n  enable: true", "completion:\n  enable: false"),
            ("completion: true", "completion: false"),
            ("enable_completion: true", "enable_completion: false"),
        ]:
            if old in cfg:
                cfg = cfg.replace(old, new)
                print(f"✓ Patched: {old.replace(chr(10), ' / ')} → false")
        if cfg != original:
            with open(config_path, "w") as f:
                f.write(cfg)
        else:
            print(f"⚠️ No completion flag matched in config. First 80 lines:")
            print("\n".join(cfg.splitlines()[:80]))
    else:
        print(f"⚠️ Config not found at {config_path}")

    open(sentinel, "w").write("done")
    vol.commit()
    print("✓ Setup complete")
    return {"status": "ok", "ckpt_root": ckpt_root, "workdir": workdir}


# ------------------------------------------------------------------
# Step 3: run pipeline on a real video
# ------------------------------------------------------------------
@app.function(
    image=image, gpu="A10G", timeout=900,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def process_video(video_bytes: bytes) -> dict:
    import subprocess
    import glob
    import time
    from huggingface_hub import login

    login(token=os.environ["HF_TOKEN"])

    video_path = "/tmp/input.mp4"
    with open(video_path, "wb") as f:
        f.write(video_bytes)
    print(f"Input: {len(video_bytes) / 1e6:.1f} MB")

    workdir = "/vol/sam-body4d"
    if not os.path.exists(workdir):
        return {"error": "Run setup_checkpoints first (modal run ::setup)"}

    print(f"Running offline_app.py on {video_path} ...")
    t0 = time.time()
    r = subprocess.run(
        ["python", "scripts/offline_app.py", "--input_video", video_path],
        cwd=workdir, capture_output=True, text=True, timeout=840,
    )
    elapsed = time.time() - t0

    out = {
        "elapsed_sec": round(elapsed, 1),
        "returncode": r.returncode,
        "stdout_tail": r.stdout[-3000:],
        "stderr_tail": r.stderr[-3000:],
    }

    # Locate produced files (exact path unknown until we see first real output)
    found = []
    for root in [
        f"{workdir}/outputs", f"{workdir}/output", f"{workdir}/results",
        "/tmp", workdir,
    ]:
        if os.path.exists(root):
            for ext in ("*.mp4", "*.json", "*.pkl", "*.npz", "*.npy", "*.obj"):
                found.extend(glob.glob(f"{root}/**/{ext}", recursive=True))
    out["output_files"] = sorted(set(found))[:40]

    return out


# ------------------------------------------------------------------
# Local entrypoints (run from your shell with `modal run`)
# ------------------------------------------------------------------
@app.local_entrypoint()
def build():
    """modal run scripts/modal_deploy.py::build"""
    import json
    print(json.dumps(smoke.remote(), indent=2))


@app.local_entrypoint()
def setup():
    """modal run scripts/modal_deploy.py::setup"""
    import json
    print(json.dumps(setup_checkpoints.remote(), indent=2))


@app.local_entrypoint()
def test(video: str):
    """modal run scripts/modal_deploy.py::test --video test_swing.mp4"""
    import json
    with open(video, "rb") as f:
        data = f.read()
    print(f"Sending {len(data) / 1e6:.1f} MB to Modal...")
    print(json.dumps(process_video.remote(data), indent=2))
