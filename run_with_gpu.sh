#!/usr/bin/env bash
# Run RTMPose with CUDA libs properly in LD_LIBRARY_PATH
# This fixes ORT GPU fallback issue on WSL with PyPI-installed CUDA packages

VENV=~/projects/swingcue-postest/.venv
CUDA_LIBS="$VENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$VENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$VENV/lib/python3.12/site-packages/nvidia/cublas/lib:$VENV/lib/python3.12/site-packages/nvidia/cufft/lib:$VENV/lib/python3.12/site-packages/nvidia/cusparse/lib:$VENV/lib/python3.12/site-packages/nvidia/curand/lib:$VENV/lib/python3.12/site-packages/nvidia/cusolver/lib:$VENV/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/usr/lib/wsl/lib"

export LD_LIBRARY_PATH="$CUDA_LIBS:${LD_LIBRARY_PATH:-}"

echo "LD_LIBRARY_PATH set, running RTMPose with GPU..."
exec "$VENV/bin/python3.12" "$@"
